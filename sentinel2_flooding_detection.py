#@title Running workflow 2022
import ee
import geemap
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm  # Import tqdm for the progress bar
import matplotlib.pyplot as plt
from branca.colormap import LinearColormap
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
import fiona
import re
import glob
import folium
import rasterio
import rasterio as rio
import matplotlib.dates as md
from datetime import datetime
from dateutil import parser
from shapely import wkt
from shapely.geometry import Point, LineString, Polygon, box
from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.colors import ListedColormap, BoundaryNorm
from rasterio.plot import show
from IPython.display import display, Markdown, HTML
from folium.plugins import MarkerCluster
from shapely.wkt import loads
from matplotlib.patches import Patch
import json
import ee

# Authenticate and initialize the Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-janet')
# Define the grid and region of interest
grid = ee.FeatureCollection("projects/ee-janet/assets/senegal/52_grid_dagana")
init_dagana = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana")

# Additional layers
dagana_reservoir = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana_reservoir")
dagana_water = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana_water")
dagana_riverbanks = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana_riverbanks")
dagana_wetland = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana_wetland")
exclusion_area = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana_exclusion_region")

exclusion_areas = dagana_riverbanks.geometry() \
    .union(dagana_wetland.geometry()) \
    .union(dagana_reservoir.geometry()) \
    .union(dagana_water.geometry())

# Subtract exclusion areas from the initial Dagana region
dagana = init_dagana.geometry().difference(exclusion_areas)#.difference(exclusion_area.geometry())

# Get the bounding box and center of the ROI for the Folium map
roi_bounds = dagana.bounds().getInfo()['coordinates'][0]
center_lat = (roi_bounds[0][1] + roi_bounds[2][1]) / 2
center_lon = (roi_bounds[0][0] + roi_bounds[2][0]) / 2

# Create a map
m = geemap.Map(center=[center_lat, center_lon], zoom=10)
# Add basemap
m.add_basemap('Esri.WorldImagery')

def run_detection_flooding(aoi, grid, start_date, end_date, year):
    # Function to calculate MNDWI for Sentinel-2
    def calculate_mndwi_s2(image):
        mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        return image.addBands(mndwi)

    # Function to calculate NDWI for Sentinel-2
    def calculate_ndwi_s2(image):
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        return image.addBands(ndwi)

    # Function to calculate AWEI for Sentinel-2
    def calculate_awei_s2(image):
        awei = image.expression(
            '4 * (GREEN - SWIR1) - (0.25 * NIR + 2.75 * SWIR2)',
            {
                'GREEN': image.select('B3'),
                'SWIR1': image.select('B11'),
                'NIR': image.select('B8'),
                'SWIR2': image.select('B12')
            }
        ).rename('AWEI')
        return image.addBands(awei)

    # Function to add s2cloudless cloud probability to Sentinel-2 imagery
    def add_cloud_probability(image):
        # Load the s2cloudless cloud probability image for the same time as the Sentinel-2 image
        cloud_probability = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
            .filterBounds(image.geometry()) \
            .filterDate(image.date(), image.date().advance(1, 'day')) \
            .first()  # Get the first image in the filtered collection

        # Add the cloud probability as a band to the original image
        return image.addBands(cloud_probability.rename('cloud_prob'))

    # Function to mask clouds using s2cloudless cloud probability
    def mask_clouds_s2cloudless(image, cloud_prob_threshold=30):
        # Add cloud probability to the image
        image = add_cloud_probability(image)

        # Create a cloud mask where cloud probability is below the threshold
        cloud_mask = image.select('cloud_prob').lt(cloud_prob_threshold)

        # Apply the cloud mask to the image
        return image.updateMask(cloud_mask)

    # Function to mask clouds in Sentinel-2 imagery
    def mask_clouds_s2(image):
        qa = image.select('QA60')
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return image.updateMask(mask)

    # Function to process dates every 5 days
    def enhanced_date_processing(start_date, end_date, interval_days=5):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        date_list = []
        while start_date <= end_date:
            date_list.append(start_date.strftime("%Y-%m-%d"))
            start_date += timedelta(days=interval_days)
        return date_list
   
    # Function to calculate flood area for each grid cell
    def calculate_grid_flood_area(flood_mask, grid,date):
        def calculate_area(feature):
            area = flood_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=feature.geometry(),
                scale=10,
                maxPixels=1e13
            )
            area_ha = area.getNumber('constant').divide(10000).format('%.2f')
            #return feature.set('flood_area_ha', area_ha)
            return feature.set({'flood_area_ha': area_ha, 'date': date})
        return grid.map(calculate_area)

    # Function to get day of year
    def get_doy(date_string):
        date = datetime.strptime(date_string, '%Y-%m-%d')
        return date.timetuple().tm_yday

    def extract_flood_data(features, date):
        flood_data = []
        for feature in features:
            grid_id = feature['properties']['ID']
            flood_area_ha = feature['properties'].get('flood_area_ha', 0)
            flood_data.append({
                'date': date,
                'grid_id': grid_id,
                'flood_area_ha': flood_area_ha,
                **{k: v for k, v in feature['properties'].items() if k != 'flood_area_ha'}
            })
        return flood_data

    def create_flood_dataframe(flood_data):
        df = pd.DataFrame(flood_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index(['date', 'grid_id'], inplace=True)
        return df

    def process_flood_results(df, grid_properties):
        df['flood_area_ha'] = pd.to_numeric(df['flood_area_ha'], errors='coerce')
        df['flood_area_ha'].fillna(0, inplace=True)
        if isinstance(grid_properties, pd.DataFrame):
            final_df = pd.merge(df.reset_index(), grid_properties, on='grid_id', how='left')
        else:
            final_df = df
        return final_df


    dataset = ee.Image('JRC/GSW1_4/MonthlyHistory/2021_01').clip(dagana)
    # Select water and create a binary mask
    water = dataset.select('water').eq(2)
    masked = water.updateMask(water)
    date_ranges = enhanced_date_processing(start_date, end_date)
    #   .map(mask_clouds_s2) \ .map(mask_clouds_s2cloudless)\
    def process_each_date(aoi, date):
        start_period = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=5)
        end_period = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=5)
        s2_sr_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                    .filterBounds(aoi) \
                    .filterDate(start_period, end_period) \
                    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 18)) \
                    .map(mask_clouds_s2) \
                    .map(calculate_mndwi_s2)\
                   # .map(calculate_awei_s2)
        # Check if the image collection is empty
        if s2_sr_col.size().getInfo() == 0:

            print(f"No images found for date {date} within the specified cloud coverage.")
            return None
        # Mosaic the images
        mosaic = s2_sr_col.mosaic().clip(dagana)
        mosaic_ = mosaic.updateMask(water.Not())
        #Threshold each index
        mndwi_mask = mosaic_.select('MNDWI').gt(0)#updateMask(water_areas.Not()
        return mndwi_mask


    # Add this new function for exporting images
    def export_image(image, description, region, folder):
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            scale=10,
            region=region,
            maxPixels=1e13
        )
        task.start()
        print(f"Started export task for {description}")
    first_image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(dagana).first()
    projection = first_image.select('B2').projection()
    # Function to process and visualize flooding, mapping only at the end of each month
    def process_and_visualize_flooding(aoi, date_ranges, grid):
        flood_data = []
        cumulative_flood_mask = ee.Image(0).reproject(crs=projection, scale=10).clip(aoi)
        flood_vis_params = {
            'min': min([get_doy(d) for d in date_ranges]),
            'max': max([get_doy(d) for d in date_ranges]),
            'palette': ['blue', 'cyan', 'green', 'yellow', 'red']
            }

        current_month = None

        # Initialize the progress bar
        for i, date in tqdm(enumerate(date_ranges), total=len(date_ranges), desc="Processing Dates"):
            current_mndwi = process_each_date(aoi, date)
            if current_mndwi is not None:
                doy = get_doy(date)
                base_date_mask = current_mndwi
                #mask1 = base_date_mask.And(cumulative_flood_mask.eq(1))
                cumulative_flood_mask = cumulative_flood_mask.where(base_date_mask.And(cumulative_flood_mask.eq(0)), doy)
                # Calculate the flood area for the grid at each 5-day interval
                grid_with_flood_area = calculate_grid_flood_area(cumulative_flood_mask.gt(0), grid, date) #                               .updateMask(cumulative_flood_mask.gt(0))
                flood_data.extend(extract_flood_data(grid_with_flood_area.getInfo()['features'], date))
                # Get the month of the current date
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                if current_month is None:
                    current_month = date_obj.month
                # Check if the next date is in a new month or if it's the last date
                is_end_of_month = (i == len(date_ranges) - 1) or (datetime.strptime(date_ranges[i + 1], "%Y-%m-%d").month                                     != current_month)
                if is_end_of_month:
                    # Add layer to map
                    # m.add_layer(cumulative_flood_mask.updateMask(cumulative_flood_mask).gt(0),
                    #             flood_vis_params, f'Flooding Progression up to {date}')
                    m.add_layer(cumulative_flood_mask.updateMask(cumulative_flood_mask.gt(0)),
                                  flood_vis_params, f'Flooding Progression up to {date}')

                    export_folder = "fis_flooding_maps"
                    # Export the cumulative flood mask
                    export_image(
                        cumulative_flood_mask.updateMask(cumulative_flood_mask.gt(0)),
                        f'Flooding_map_{date}',
                        aoi,
                        export_folder
                   )
                    # Reset current_month for the next month
                    if i != len(date_ranges) - 1:
                        current_month = datetime.strptime(date_ranges[i + 1], "%Y-%m-%d").month
            else:
                print(f"Skipping date {date} due to no valid MNDWI")
        m.add_colorbar(flood_vis_params, label="Day of the year",
                       orientation="horizontal",
                       layer_name="Flooding detection")
        return flood_data
    
    flood_data = process_and_visualize_flooding(dagana, date_ranges, grid)
    df = create_flood_dataframe(flood_data)
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # Pivot the table
    df_pivoted = df.pivot(index='grid_id', columns='date', values='flood_area_ha')
    columns_date = df_pivoted.columns
    maxValueIndex1 = df_pivoted[columns_date].idxmax(axis=1)
    df_pivoted['flooding_date'] = maxValueIndex1

    # Assuming the columns_to_keep exist in your data
    columns_to_keep = ['ID', 'LatNP', 'Latitude', 'LonNP', 'Longitude', 'nasapid']
    df_other = df.drop_duplicates(subset=['grid_id'])[columns_to_keep + ['grid_id']].set_index('grid_id')
    df_pivoted = df_pivoted.rename_axis(index='grid_id')
    df_final = df_other.join(df_pivoted)
    df_final = df_final.reset_index()
    date_columns = [col for col in df_final.columns if col not in columns_to_keep + ['grid_id', 'index']]
    df_final = df_final[columns_to_keep + ['grid_id'] + sorted(date_columns)]

    output_file_name = f'floodingData_{year}.csv'
    df_final.to_csv(output_file_name, index=False)
    
    # Add additional layers to the map
    #m.addLayer(dagana_reservoir, {'color': 'blue'}, 'Dagana Reservoir')
    #m.addLayer(dagana_water, {'color': 'cyan'}, 'Dagana Water')
    m.addLayer(dagana_riverbanks, {'color': 'green'}, 'Dagana Riverbanks')
    #m.addLayer(dagana_wetland, {'color': 'brown'}, 'Dagana Wetland')
    # Display the map
    display(m)
    
    # Melt the dataframe to convert date columns to rows
    date_columns = [col for col in df_final.columns if col.startswith('20')]  # Select only date columns
    melted_df = pd.melt(df_final, id_vars=['ID'], value_vars=date_columns,
                        var_name='Date', value_name='Flooded_Area')
    
    # Convert Date column to datetime
    melted_df['Date'] = pd.to_datetime(melted_df['Date'])
    
    # Ensure Flooded_Area is numeric, replacing any non-numeric values with NaN
    melted_df['Flooded_Area'] = pd.to_numeric(melted_df['Flooded_Area'], errors='coerce')
    #print(melted_df.head())
    # Group by Date and sum the Flooded_Area
    total_flooded_area = melted_df.groupby('Date')['Flooded_Area'].sum().reset_index()
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(total_flooded_area['Date'], total_flooded_area['Flooded_Area'], marker='o')
    plt.title('Total Flooded Area Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Flooded Area (hectares)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Print the first few rows of the processed data to verify
    print(total_flooded_area.head())
    
    # Print summary statistics
    print(total_flooded_area['Flooded_Area'].describe())

        # Load data
    rs_mndwi_data_2025 = df_final#pd.read_csv(f'floodingData_{year}.csv', index_col=0)
    
    saed_2025_dhs = pd.read_csv("saed_2025.csv")
    saed_2025_dhs['date'] = pd.to_datetime(saed_2025_dhs['Date'])
    
    # Process MNDWI data for 2023
    rs_hueristics_dff_2025 = rs_mndwi_data_2025.filter(regex=('\\d{4}-?\\d{2}-?\\d{2}$'))
    area_rs_25 = rs_hueristics_dff_2025.sum(axis=0)
    rs_df_25 = pd.DataFrame()
    rs_df_25['Date'] = list(area_rs_25.index)
    rs_df_25['Area(ha)'] = list(area_rs_25.values)
    rs_df_25['Data_source'] = 'S2 MNDWI 2025'
    rs_df_25['date'] = pd.to_datetime(rs_df_25['Date'])
 
    # Combine dataframes for 2025
    combined_df_2025 = pd.concat([rs_df_25, saed_2025_dhs])
    combined_df_2025['Day'] = combined_df_2025['date'].dt.day
    combined_df_2025['Month'] = combined_df_2025['date'].dt.month
    combined_df_2025['Year'] = combined_df_2025['date'].dt.year
    combined_df_2025['Days'] = combined_df_2025['date'].dt.dayofyear

    # Define a consistent color palette for all data sources
    palette = {
        'S2 MNDWI 2025': 'green',
        'SAED prepared area': 'purple',
        'SAED planted area': 'orange'
    }
    
    # Plotting
    #fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    plt.figure(figsize=(10, 6))
    
    # Plot for 2025
    sns.lineplot(data=combined_df_2025, x='date', y='Area(ha)', hue='Data_source', marker='o', palette=palette)
    plt.title('Flooded area over time with RS and SAED data (2025)')
    plt.xlabel('Day of the year')
    plt.ylabel('Area (ha)')
    plt.legend(title='Data Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    #plt.tight_layout()
    plt.show()
    
    #another way
    # Define the start and end dates for vertical lines
    start_date = datetime.strptime('2025-02-15', '%Y-%m-%d').timetuple().tm_yday
    end_date = datetime.strptime('2025-03-15', '%Y-%m-%d').timetuple().tm_yday
    
    # Create single plot
    plt.figure(figsize=(10, 6))
    
    # Plot the data
    sns.lineplot(data=combined_df_2025, x='date', y='Area(ha)', hue='Data_source', marker='o', palette=palette)
    plt.title('Flooded area over time with RS and SAED data (2025)')
    plt.xlabel('Day of the year')
    plt.ylabel('Area (ha)')
    
    # Add shaded area between start and end dates
    plt.axvspan(start_date, end_date, color='grey', alpha=0.3, label='Planting period')
    
    # Add vertical lines
    plt.axvline(start_date, color='black', linestyle='--')
    plt.axvline(end_date, color='black', linestyle='--')
    
    # Add labels for the vertical lines
    plt.text(start_date, 17000, 'Start planting (15FEB)', ha='right', va='bottom', color='black')
    plt.text(end_date, 17000, 'End planting (15MARCH)', ha='left', va='bottom', color='black')
    
    # Adjust legend
    plt.legend(title='Data Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()