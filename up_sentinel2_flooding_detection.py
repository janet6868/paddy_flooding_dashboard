# sentinel2_flooding_detection.py
import ee
import streamlit as st
#import geemap
from streamlit_folium import folium_static
from streamlit_folium import st_folium
#import geemap.foliumap as geema
import geemap.ee_tile_layers as geemap
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
from branca.colormap import LinearColormap
import geopandas as gpd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
import fiona
import folium
import rasterio
import matplotlib.dates as md
from dateutil import parser
from shapely import wkt
from shapely.geometry import Point, LineString, Polygon, box
from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.colors import ListedColormap, BoundaryNorm
from rasterio.plot import show
from folium.plugins import MarkerCluster
from shapely.wkt import loads
from matplotlib.patches import Patch
import json
import re

# ------------------------------------------------------------------
# 1. EARTH ENGINE AUTHENTICATION (LOCAL)
# ------------------------------------------------------------------
# You must have already authenticated Earth Engine once in your environment:
#   earthengine authenticate
# Alternatively, uncomment if you want an interactive prompt:
# from google.auth import compute_engine
# import ee
# credentials = compute_engine.Credentials(scopes=['https://www.googleapis.com/auth/earthengine'])
# ee.Initialize(credentials)

#ee.Authenticate()
st.write("project_id:", st.secrets["project"])
# st.write("DB password:", st.secrets["db_password"])
# st.write("My cool secrets:", st.secrets["my_cool_secrets"]["things_i_like"])
ee.Authenticate()
ee.Initialize(project=rpject_id)
# ------------------------------------------------------------------
# 2. DEFINE YOUR REGIONS, COLLECTIONS, ETC.
# ------------------------------------------------------------------
grid = ee.FeatureCollection("projects/ee-janet/assets/senegal/52_grid_dagana")
init_dagana = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana")

dagana_reservoir = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana_reservoir")
dagana_water = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana_water")
dagana_riverbanks = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana_riverbanks")
dagana_wetland = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana_wetland")
exclusion_area = ee.FeatureCollection("projects/ee-janet/assets/senegal/dagana_exclusion_region")

exclusion_areas = (
    dagana_riverbanks.geometry()
    .union(dagana_wetland.geometry())
    .union(dagana_reservoir.geometry())
    .union(dagana_water.geometry())
)

dagana = init_dagana.geometry().difference(exclusion_areas)

# (Optional) For bounding info, if needed
roi_bounds = dagana.bounds().getInfo()['coordinates'][0]
water_bounds = grid.bounds().getInfo()#['coordinates'][0]
center_lat = (roi_bounds[0][1] + roi_bounds[2][1]) / 2
center_lon = (roi_bounds[0][0] + roi_bounds[2][0]) / 2
m = folium.Map(location=[16.51277780,-15.80500000], zoom_start=10)
#m.save("folium_map.html")

# Add a title on the map
title_html = '''
             <h3 align="center" style="font-size:20px"><b>Spatial Map for Flooded Areas</b></h3>
             '''
m.get_root().html.add_child(folium.Element(title_html))
def add_ee_layer(map_object, ee_image_object, vis_params, name):
    """Add an Earth Engine image as a tile layer to a Folium map."""
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    tile_url = map_id_dict['tile_fetcher'].url_format
    folium.TileLayer(
        tiles=tile_url,
        attr='Google Earth Engine',
        name=name,
        overlay=True,
        control=True,
        max_zoom=15 
    ).add_to(map_object)

def run_detection_flooding(aoi, grid, start_date, end_date, year, local_saed_csv_path=None):
    """
    Processes Sentinel-2 data for flood detection using MNDWI and 
    returns a Pandas DataFrame with the results. Optionally merges with 
    a local SAED CSV file (if path is provided).

    Args:
        aoi (ee.Geometry): Region of interest (Dagana).
        grid (ee.FeatureCollection): Grid feature collection.
        start_date (str): Start date ('YYYY-MM-DD').
        end_date (str): End date ('YYYY-MM-DD').
        year (str): Year string (e.g. '2025').
        local_saed_csv_path (str, optional): Path to local SAED CSV file.
    Returns:
        pd.DataFrame: DataFrame of flood data and optional SAED data.
    """

    # --------------------------------------------------------------
    # HELPER FUNCTIONS
    # --------------------------------------------------------------
    def calculate_mndwi_s2(image):
        """Compute MNDWI for Sentinel-2."""
        mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        return image.addBands(mndwi)

    def mask_clouds_s2(image):
        """Mask clouds using QA60 band."""
        qa = image.select('QA60')
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return image.updateMask(mask)

    def enhanced_date_processing(s_date, e_date, interval_days=5):
        """Create a list of dates separated by a given interval."""
        start_dt = datetime.strptime(s_date, "%Y-%m-%d")
        end_dt = datetime.strptime(e_date, "%Y-%m-%d")
        date_list = []
        while start_dt <= end_dt:
            date_list.append(start_dt.strftime("%Y-%m-%d"))
            start_dt += timedelta(days=interval_days)
        return date_list

    def get_doy(date_string):
        """Return day of year for a date string."""
        d = datetime.strptime(date_string, '%Y-%m-%d')
        return d.timetuple().tm_yday

    def process_each_date(aoi, date):
        """Filter S2 collection for the given date ±5 days, 
           compute MNDWI, and return mask."""
        start_period = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=5)
        end_period = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=5)

        s2_sr_col = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filterDate(start_period, end_period)
            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 18))
            .map(mask_clouds_s2)
            .map(calculate_mndwi_s2)
        )
        if s2_sr_col.size().getInfo() == 0:
            print(f"No images found for date {date} with <=18% cloud cover.")
            return None

        mosaic = s2_sr_col.mosaic().clip(aoi)

        # We also remove permanent water areas
        dataset = ee.Image('JRC/GSW1_4/MonthlyHistory/2021_01').clip(aoi)
        water = dataset.select('water').eq(2)
        mosaic_ = mosaic.updateMask(water.Not())

        mndwi_mask = mosaic_.select('MNDWI').gt(0)
        return mndwi_mask

    def calculate_grid_flood_area(flood_mask, grid_fc, date):
        """Compute flood area per grid cell (in ha)."""
        def calculate_area(feature):
            area = flood_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=feature.geometry(),
                scale=10,
                maxPixels=1e13
            )
            area_ha = area.getNumber('constant').divide(10000).format('%.2f')
            return feature.set({'flood_area_ha': area_ha, 'date': date})

        return grid_fc.map(calculate_area)

    def extract_flood_data(features, date):
        """Convert a list of ee.Feature (dict) to standard Python list."""
        flood_data = []
        for feature in features:
            props = feature['properties']
            # If your grid has 'ID', you'll store it as 'grid_id'
            grid_id = props['ID'] if 'ID' in props else None
            flood_area_ha = props.get('flood_area_ha', 0)
            flood_data.append({
                'date': date,
                'grid_id': grid_id,
                'flood_area_ha': flood_area_ha,
                **{k: v for k, v in props.items() if k not in ['flood_area_ha', 'ID']}
            })
        return flood_data

    def create_flood_dataframe(flood_data):
        """Make a Pandas DataFrame from the flood_data list."""
        df = pd.DataFrame(flood_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index(['date', 'grid_id'], inplace=True)
        return df

    def process_and_visualize_flooding(aoi, date_ranges, grid_fc):
        """Iterate over date ranges, build a cumulative mask, gather data."""
        flood_data = []
        # Prepare an empty 'base' image in the same projection
        first_image = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .first()
        )
        projection = first_image.select('B2').projection()
        # Start with a zero-value image
        cumulative_flood_mask = ee.Image(0).reproject(crs=projection, scale=10).clip(aoi)

        flood_vis_params = {
            'min': min([get_doy(d) for d in date_ranges]),
            'max': max([get_doy(d) for d in date_ranges]),
            'palette': ['blue', 'cyan', 'green', 'yellow', 'red']
        }

        current_month = None

        for i, date in tqdm(enumerate(date_ranges), total=len(date_ranges), desc="Processing Dates"):
            current_mndwi = process_each_date(aoi, date)
            if current_mndwi is not None:
                doy = get_doy(date)
                # Where we have water for this date but not yet flagged in cumulative,
                # set the pixel value to DOY
                cumulative_flood_mask = cumulative_flood_mask.where(
                    current_mndwi.And(cumulative_flood_mask.eq(0)),
                    doy
                )
                # Summaries for each grid cell
                grid_with_flood_area = calculate_grid_flood_area(
                    cumulative_flood_mask.gt(0),
                    grid_fc,
                    date
                )
                flood_data.extend(
                    extract_flood_data(grid_with_flood_area.getInfo()['features'], date)
                )

                # Month-based check for visual
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                if current_month is None:
                    current_month = date_obj.month

                if i < len(date_ranges) - 1:
                    next_month = datetime.strptime(date_ranges[i + 1], "%Y-%m-%d").month
                else:
                    next_month = None

                # End-of-month check
                is_end_of_month = (i == len(date_ranges) - 1) or (next_month != current_month)
                if is_end_of_month:
                    # Create a mask where flood has been detected
                    flood_mask_visual = cumulative_flood_mask.updateMask(cumulative_flood_mask.gt(0))
                    # Add the layer to the Folium map using our helper function
                    add_ee_layer(m, flood_mask_visual, flood_vis_params, f'Flooding up to {date}')
                if next_month is not None:
                    current_month = next_month
   
            else:
                print(f"Skipping date {date}, no valid MNDWI found.")

        colormap = LinearColormap(
            colors=flood_vis_params['palette'],
            vmin=flood_vis_params['min'],
            vmax=flood_vis_params['max'],
            caption="Day of the year"
        )
        colormap.add_to(m)
        return flood_data

    # --------------------------------------------------------------
    # 3. RUN FLOOD DETECTION LOGIC
    # --------------------------------------------------------------
    date_ranges = enhanced_date_processing(start_date, end_date)
    flood_data = process_and_visualize_flooding(aoi, date_ranges, grid)

    df = create_flood_dataframe(flood_data)
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # Pivot table to have each date as a column
    df_pivoted = df.pivot(index='grid_id', columns='date', values='flood_area_ha')
    df_pivoted = df_pivoted.apply(pd.to_numeric, errors='coerce')

    # Identify the date with maximum flood area per grid
    max_date = df_pivoted.idxmax(axis=1)
    df_pivoted['Est_flooding_date'] = max_date

    # Optional columns to keep if your grid has them
    columns_to_keep = ['ID','LatNP','Latitude','LonNP','Longitude','nasapid']
    df_other = df.drop_duplicates(subset=['grid_id'])

    # Add missing columns if they don't exist
    for col in columns_to_keep:
        if col not in df_other.columns:
            df_other[col] = None

    df_other = df_other[columns_to_keep + ['grid_id']].set_index('grid_id')
    df_final = df_pivoted.join(df_other)
    df_final = df_final.reset_index()

    # Reorder columns if needed
    date_columns = [col for col in df_final.columns if re.match(r'\d{4}-\d{2}-\d{2}', str(col))]
    df_final = df_final[['grid_id'] + columns_to_keep + date_columns + ['Est_flooding_date']]

    output_file_name = f'floodingData_{year}.csv'
    df_final.to_csv(output_file_name, index=False)

    print(f"Saved {output_file_name} locally.")

    return df_final, m
