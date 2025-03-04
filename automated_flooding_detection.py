
# ______________ CODE 2____________
# # Import core libraries
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os
import glob
import time

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import DateFormatter, DayLocator

# Geospatial libraries
import geopandas as gpd
import contextily as ctx
import folium
from folium.plugins import MarkerCluster, HeatMap
import geemap
import ee
import shapely.geometry
import warnings
try:
    from IPython.core.display import display
except ImportError:
    warnings.warn("IPython is not available. Some geemap functionality may be limited.")
import geemap


# Import your custom module - adjust as needed
# try:
from up_sentinel2_flooding_detection import (
    run_detection_flooding,
    dagana,  # geometry
    grid     # FeatureCollection
)
# Set the Streamlit page configuration
st.set_page_config(
    page_title="Paddy Flooding Detection Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)
def get_area_column(df):
    """Find the area column in the dataframe"""
    for col in ['flooded_area_ha', 'flooding_area', 'area', 'Area']:
        if col in df.columns:
            return col
    # Fallback to first numeric column that's not grid_id
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if col != 'grid_id':
            return col
    return None
# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3498db;
        margin-top: 2rem;
    }
    .section-divider {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-top: 1px solid #e0e0e0;
    }
    .stat-box {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ---- Helper Functions ----

def read_csv_data(file_path, default_data=None):
    """Read CSV data with error handling"""
    try:
        return pd.read_csv(file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.warning(f"Could not read {file_path}. Using default/sample data.")
        return default_data if default_data is not None else pd.DataFrame()

def process_rs_data(df):
    """Process remote sensing data"""
    # Filter columns with date format
    rs_df = df.filter(regex=r'\d{4}-?\d{2}-?\d{2}$')
    print(rs_df)
    
    # Sum areas by date
    area_rs = rs_df.sum(axis=0)
    
    # Create a new dataframe
    rs_df_combined = pd.DataFrame({
        'Time': area_rs.index,
        'Area(ha)': area_rs.values
    })
    
    # Extract year and add metadata
    rs_df_combined['Year'] = rs_df_combined['Time'].str[:4]
    rs_df_combined['Class'] = 'RS_' + rs_df_combined['Year']
    rs_df_combined['Time'] = pd.to_datetime(rs_df_combined['Time'])
    rs_df_combined['DOY'] = rs_df_combined['Time'].dt.dayofyear
    
    return rs_df_combined

def create_comparison_chart(combined_df):
    """Create an improved comparison chart with Plotly"""
    # Convert datetime to string for better tooltip display
    combined_df['date_str'] = combined_df['date'].dt.strftime('%Y-%m-%d')
    #print(combined_df.head())
    # Define the start and end dates for planting period
    season_start = datetime.strptime('2025-02-15', '%Y-%m-%d')
    season_end = datetime.strptime('2025-03-15', '%Y-%m-%d')
    
    # Create figure
    fig = px.line(
        combined_df, 
        x='date', 
        y='Area(ha)',
        color='Data_source',
        markers=True,
        line_shape='linear',
        color_discrete_map={
            'Remote sensing 2025': 'blue',
            'SAED prepared area': '#9467bd',
            'SAED planted area': '#ff7f0e'
        },
        hover_data=['date_str', 'Area(ha)'],
        title='2025 Dry Hot Season Flooded Areas (ha)'
    )
    
    # Add planting period shaded area
    fig.add_vrect(
        x0=season_start,
        x1=season_end,
        fillcolor="LightGray",
        opacity=0.3,
        layer="below",
        line_width=0,
    )
    
    # Add annotation for planting period
    fig.add_annotation(
        x=season_start + (season_end - season_start) / 2,
        y=combined_df['Area(ha)'].max() * 0.9,
        text="Planting Period<br>(15 FEB - 15 MAR)",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        opacity=0.8
    )
    
    # Add vertical lines for start and end of planting period
    fig.add_vline(x=season_start, line_width=1, line_dash="dash", line_color="black")
    fig.add_vline(x=season_end, line_width=1, line_dash="dash", line_color="black")
    
    # Update layout for better appearance
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Area (ha)",
        legend_title="Data Source",
        font=dict(family="Arial, sans-serif", size=14),
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            tickformat="%b %d",
            tickangle=-45,
            tickmode="auto",
            nticks=12,
            showgrid=True,
            gridcolor='rgba(220,220,220,0.8)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(220,220,220,0.8)'
        ),
        plot_bgcolor='rgba(250,250,250,0.9)'
    )
    
    return fig

def create_cumulative_area_chart(processed_df, title):
    """Create an improved cumulative area chart with Plotly"""
    # Constants for planting and harvesting periods
    START_PLANTING = 46  # 15 Feb
    END_PLANTING = 74    # 15 Mar
    START_HARVESTING = 186  # 5 Jul
    END_HARVESTING = 259    # 16 Sep
    
    # Create a unique color for each year
    years = processed_df['Time'].dt.year.unique()
    color_map = {str(year): px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                for i, year in enumerate(years)}
    
    # Create the figure
    fig = px.line(
        processed_df,
        x='DOY',
        y='Area(ha)',
        color=processed_df['Time'].dt.year.astype(str),
        line_shape='linear',
        markers=True,
        title=title,
        color_discrete_map=color_map,
        labels={'DOY': 'Day of Year', 'Area(ha)': 'Flooded Area (ha)'}
    )
    
    # Add vertical lines for key dates
    events = [
        (START_PLANTING, 'Start Planting (15 Feb)', 'blue'),
        (END_PLANTING, 'End Planting (15 Mar)', 'green'),
        (START_HARVESTING, 'Start Harvesting (5 Jul)', 'orange'),
        (END_HARVESTING, 'End Harvesting (16 Sep)', 'red')
    ]
    
    for doy, label, color in events:
        fig.add_vline(
            x=doy, 
            line_dash="dash", 
            line_color=color,
            annotation_text=label,
            annotation_position="top right"
        )
    
    # Shade planting period
    fig.add_vrect(
        x0=START_PLANTING,
        x1=END_PLANTING,
        fillcolor="LightBlue",
        opacity=0.15,
        layer="below",
        line_width=0,
        annotation_text="Planting Period",
        annotation_position="bottom right"
    )
    
    # Shade harvesting period
    fig.add_vrect(
        x0=START_HARVESTING,
        x1=END_HARVESTING,
        fillcolor="LightGreen",
        opacity=0.15,
        layer="below",
        line_width=0,
        annotation_text="Harvesting Period",
        annotation_position="bottom right"
    )
    
    # Update layout for better appearance
    fig.update_layout(
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="x unified",
        legend_title="Year",
        font=dict(family="Arial, sans-serif", size=14),
        xaxis=dict(
            title="Day of Year (DOY)",
            tickmode="array",
            tickvals=[1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
            ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            showgrid=True,
            gridcolor='rgba(220,220,220,0.8)'
        ),
        yaxis=dict(
            title="Flooded Area (ha)",
            showgrid=True,
            gridcolor='rgba(220,220,220,0.8)'
        ),
        plot_bgcolor='rgba(250,250,250,0.9)'
    )
    
    return fig



def create_statistics_summary(df_final):
    """Create summary statistics for the dashboard"""
    try:
        # Check if df_final is already in the format we need (with date and Area(ha) columns)
        if 'Area(ha)' in df_final.columns and 'Date' in df_final.columns:
            dff = df_final.copy()
        else:
            # Try to find columns with dates in the format
            try:
                df1 = df_final.filter(regex=('\\d{4}-?\\d{2}-?\\d{2}$'))
                df2 = df1.sum(axis=0)
                dff = pd.DataFrame()
                dff['Date'] = list(df2.index)
                
                # Convert values to numeric safely with error coercion
                values = pd.to_numeric(df2.values, errors='coerce')
                dff['Area(ha)'] = list(values)
                
                # Drop any rows with NaN values due to conversion errors
                dff = dff.dropna()
            except Exception as inner_e:
                st.warning(f"Could not process date columns: {inner_e}")
                
                # Fallback - try to identify any area column
                area_col = get_area_column(df_final)
                if area_col:
                    return df_final[area_col].sum(), df_final[area_col].mean(), df_final[area_col].max(), "N/A"
                else:
                    return 0, 0, 0, "N/A"
        
        # Calculate statistics
        total_area = dff['Area(ha)'].sum()
        avg_area = dff['Area(ha)'].mean()
        max_area = dff['Area(ha)'].max()
        
        # Get the latest date in the dataset
        if 'flooding_date' in df_final.columns:
            latest_date = pd.to_datetime(df_final['flooding_date']).max()
            date_str = latest_date.strftime('%Y-%m-%d')
        elif 'Date' in dff.columns:
            try:
                latest_date = pd.to_datetime(dff['Date']).max()
                date_str = latest_date.strftime('%Y-%m-%d')
            except:
                date_str = "N/A"
        else:
            date_str = "N/A"
        
        return total_area, avg_area, max_area, date_str
    
    except Exception as e:
        st.error(f"Error calculating statistics: {e}")
        return 0, 0, 0, "Error"

def create_sample_data(year="2025"):
    """Create sample data for demonstration purposes"""
    # Create sample remote sensing data
    dates = pd.date_range(start=f'{year}-01-21', end=f'{year}-02-23')
    
    # Create a DataFrame with increasing areas
    rs_data = {}
    for i, date in enumerate(dates):
        date_str = date.strftime('%Y-%m-%d')
        area_value = 1000 + i * 200 + np.random.normal(0, 100)
        rs_data[date_str] = max(area_value, 0)  # Ensure no negative areas
    
    rs_df = pd.DataFrame([rs_data])
    
    # Create sample SAED data
    saed_dates = pd.date_range(start=f'{year}-01-15', end=f'{year}-03-01', freq='5D')
    saed_data = []
    
    # Prepared area (slightly higher than RS)
    for date in saed_dates:
        date_str = date.strftime('%Y-%m-%d')
        # Find closest RS date
        closest_rs_date = min(rs_data.keys(), key=lambda x: abs(pd.to_datetime(x) - date))
        prepared_area = rs_data.get(closest_rs_date, 0) * 1.2 + np.random.normal(0, 200)
        planted_area = prepared_area * 0.8 + np.random.normal(0, 100)
        
        saed_data.append({
            'Date': date_str,
            'Area(ha)': max(prepared_area, 0),
            'Data_source': 'SAED prepared area',
            'date': date
        })
        
        saed_data.append({
            'Date': date_str,
            'Area(ha)': max(planted_area, 0),
            'Data_source': 'SAED planted area',
            'date': date
        })
    
    saed_df = pd.DataFrame(saed_data)
    
    # Create RS data in the same format as SAED for comparison
    rs_comparison_data = []
    for date_str, area in rs_data.items():
        rs_comparison_data.append({
            'Date': date_str,
            'Area(ha)': area,
            'Data_source': f'Remote sensing {year}',
            'date': pd.to_datetime(date_str)
        })
    
    rs_comparison_df = pd.DataFrame(rs_comparison_data)
    
    # Combine datasets
    combined_df = pd.concat([rs_comparison_df, saed_df])
    
    return rs_df, saed_df, combined_df

# ---- Main Dashboard Layout ----

# Dashboard header
st.markdown('<h1 class="main-header">Paddy Flooding Detection Dashboard</h1>', unsafe_allow_html=True)

# Sidebar with parameters
with st.sidebar:
    st.image("https://img.freepik.com/premium-vector/rice-field-logo-template-agriculture-symbol_159025-525.jpg", width=100)
    
    st.markdown("### Analysis Parameters")
    
    # Create tabs in the sidebar for better organization
    tab1, tab2 = st.tabs(["Current Season", "Historical Comparison"])
    
    with tab1:
        default_start = datetime(2025, 1, 21)
        default_end = datetime(2025, 2, 23)
        start_date = st.date_input("Start Date", default_start)
        end_date = st.date_input("End Date", default_end)
        year = st.text_input("Year", "2025")
        
        # Advanced options
        with st.expander("Advanced Options"):
            cloud_cover = st.slider("Max Cloud Cover (%)", 0, 100, 20)
            mndwi_threshold = st.slider("MNDWI Threshold", -0.5, 0.5, 0.0, 0.05)
    
    with tab2:
        selected_years = st.multiselect(
            "Select years for comparison",
            options=list(range(2019, 2026)),
            default=[2023, 2024, 2025]
        )
        
        with st.expander("Visualization Options"):
            normalize = st.checkbox("Normalize Values")
            visualization_type = st.radio(
                "Chart Type", 
                ["Line Chart", "Area Chart", "Bar Chart"]
            )
    
    # Add a "Help" expander at the bottom of the sidebar
    with st.expander("Help & Information"):
        st.markdown("""
        **About this dashboard:**
        - Uses Sentinel-2 imagery to detect flooded paddy fields
        - Compares remote sensing data with SAED ground data
        - Provides both current season and historical analysis
        
        For more information, contact: your@email.com
        """)

# Main dashboard content with tabs for better organization
tab1, tab2 = st.tabs(["Current Season Analysis", "Historical Comparison"])

with tab1:
    st.markdown('<h2 class="sub-header">Current Season Flooding Analysis</h2>', unsafe_allow_html=True)
    
    st.info(
        """
        This analysis uses Earth Engine data to detect flooded areas (MNDWI) from Sentinel-2,
        and compares the results with SAED field data.
        """
    )
    
    # Action buttons in a row
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        run_btn = st.button("Run Flood Detection", use_container_width=True)
    with col2:
        download_btn = st.download_button(
            "Download Results", 
            data="data_placeholder", 
            file_name="flood_detection_results.csv",
            use_container_width=True,
            disabled=True  # Enable after analysis is run
        )
    with col3:
        clear_btn = st.button("Clear Results", use_container_width=True)
    
    # Run analysis when button is clicked
    if run_btn or 'df_final' not in st.session_state:
        with st.spinner("Processing flood detection..."):
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Simulate processing with progress
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            #try:
            # Try to run actual detection function
            df_final = run_detection_flooding(
                aoi= dagana, grid=grid,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                year=year
            )
            st.session_state.df_final = df_final
                #print(df_final.head())
                
                # Generate or load sample data for demo
            output_file_name = f'floodingData_{year}.csv'
             #   try:
            rs_df = pd.read_csv(output_file_name)
              #  except (FileNotFoundError, pd.errors.EmptyDataError):
                #     # Create sample data for demo purposes
                #     rs_df, saed_df, combined_df = create_sample_data(year)
                #     st.session_state.rs_df = rs_df
                #     st.session_state.saed_df = saed_df
                #     st.session_state.combined_df = combined_df
                
                #st.success("Analysis completed successfully!")
                
            # except Exception as e:
            #      st.error(f"Error during flood detection: {e}")
            #     # Create sample data for demo purposes
            #     rs_df, saed_df, combined_df = create_sample_data(year)
            #     st.session_state.rs_df = rs_df
            #     st.session_state.saed_df = saed_df
            #     st.session_state.combined_df = combined_df
            #     st.session_state.df_final = pd.DataFrame({
            #         'grid_id': range(1, 11),
            #         'Area(ha)': np.random.uniform(10, 500, 10),
            #         'flooding_date': [datetime.now().strftime('%Y-%m-%d')] * 10
            #     })
    
    # Clear results if clear button is clicked
    if clear_btn:
        if 'df_final' in st.session_state:
            del st.session_state.df_final
        if 'rs_df' in st.session_state:
            del st.session_state.rs_df
        if 'saed_df' in st.session_state:
            del st.session_state.saed_df
        if 'combined_df' in st.session_state:
            del st.session_state.combined_df
        st.experimental_rerun()
    
    # If we have results, display them
    if 'df_final' in st.session_state:
        # Calculate and display key metrics at the top in cards
        total_area, avg_area, max_area, latest_date = create_statistics_summary(st.session_state.df_final)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Total Flooded Area", f"{total_area:.1f} ha")
            st.markdown('</div>', unsafe_allow_html=True)
        with metric_col2:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Average Area Per Grid", f"{avg_area:.1f} ha")
            st.markdown('</div>', unsafe_allow_html=True)
        with metric_col3:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Maximum Area", f"{max_area:.1f} ha")
            st.markdown('</div>', unsafe_allow_html=True)
        with metric_col4:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Latest Data Date", latest_date)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Results display in two columns
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        #map_col, data_col = st.columns([2, 1])
 
        col1, col2 = st.columns([2, 1])

        with col2:
            st.subheader("Tabular Data / Key Metrics")
            st.write("### Flooded Area DataFrame")
            st.dataframe(st.session_state.df_final)

        with col1:
        # 4. Show comparison below the two-column layout
            st.subheader("SAED and Remote Sensing Comparison")
            
            # Here is your existing code for plotting
            output_file_name = f'floodingData_{year}.csv'
            rs_df = pd.read_csv(output_file_name)
            rs_hueristics_dff_2025 = rs_df.filter(regex=('\\d{4}-?\\d{2}-?\\d{2}$'))

            area_rs_25 = rs_hueristics_dff_2025.sum(axis=0)
            rs_df_25 = pd.DataFrame()
            rs_df_25['Date'] = list(area_rs_25.index)
            rs_df_25['Area(ha)'] = list(area_rs_25.values)
            rs_df_25['Data_source'] = 'Remote sensing 2025'
            rs_df_25['date'] = pd.to_datetime(rs_df_25['Date'])

            local_saed_csv_path = "saed_2025.csv"  # Update if your file path differs
            saed_2025_dhs = pd.read_csv(local_saed_csv_path)
            saed_2025_dhs['date'] = pd.to_datetime(saed_2025_dhs['Date'])

            # Combine
            combined_df_2025 = pd.concat([rs_df_25, saed_2025_dhs])
            combined_df_2025['Day'] = combined_df_2025['date'].dt.day
            combined_df_2025['Month'] = combined_df_2025['date'].dt.month
            combined_df_2025['Year'] = combined_df_2025['date'].dt.year
            combined_df_2025['Days'] = combined_df_2025['date'].dt.dayofyear

            # Plot
            palette = {
                'Remote sensing 2025': 'blue',
                'SAED prepared area': 'purple',
                'SAED planted area': 'orange'
            }

            # Store original figure size
            original_figsize = plt.rcParams['figure.figsize']
            # Adjust
            scale_factor = 20 / 2
            plt.rcParams['figure.figsize'] = (scale_factor, scale_factor * 0.6)

            fig, ax = plt.subplots()

            # Define the start and end dates for your typical “current season”
            season_start = datetime.strptime('2025-02-15', '%Y-%m-%d')
            season_end   = datetime.strptime('2025-03-15', '%Y-%m-%d')

            sns.lineplot(
                data=combined_df_2025,
                x='date',
                y='Area(ha)',
                hue='Data_source',
                marker='o',
                palette=palette,
                ax=ax
            )

            ax.set_title('2025 Dry Hot Season Flooded Areas (ha)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Area (ha)')

            # Shade the typical planting period
            ax.axvspan(season_start, season_end, color='grey', alpha=0.3, label='Planting period')

            # Add dashed lines
            ax.axvline(season_start, color='black', linestyle='--')
            ax.axvline(season_end, color='black', linestyle='--')

            # Middle of shading (to place a label)
            middle_date = season_start + (season_end - season_start) / 2
            ax.text(middle_date, 7000, 'Planting period\n(15 FEB - 15 MAR)',
                    ha='center', va='center', color='black',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            ax.legend(title='Data Source', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            st.pyplot(fig)
            plt.rcParams['figure.figsize'] = original_figsize
            plt.close(fig)
        
        
        # Comparison chart at the bottom
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        #st.markdown("### SAED and Remote Sensing Comparison")
        
        # if 'combined_df' in st.session_state:
        #     # Create and display the comparison chart
        #     fig = create_comparison_chart(st.session_state.combined_df)
        #     st.plotly_chart(fig, use_container_width=True)
        # else:
        #     st.warning("No comparison data available.")
def read_github_csv(url):
    """Read CSV from GitHub URL."""
    raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return pd.read_csv(raw_url)
# Constants
START_PLANTING = 46  # 15 Feb
END_PLANTING = 74    # 15 Mar
START_HARVESTING = 186  # 5 Jul
END_HARVESTING = 259    # 16 Sep
with tab2:
    st.markdown('<h2 class="sub-header">Historical Comparison Analysis</h2>', unsafe_allow_html=True)
    
    # # Create a 3-column layout for filtering options
    filter_col1, filter_col2= st.columns(2)
    with filter_col1:
        view_type = st.radio("View Type", ["Cumulative", "Daily"], horizontal=True)
    with filter_col2:
        area_type = st.selectbox("Area Type", ["Dagana Area", "agCelerant Plots"])
    # with filter_col3:
    #     normalize = st.checkbox("Normalize Values")

    def process_data(urls, title):
        """Process and plot data from a list of URLs."""
        dataframes = []
        for url in urls:
            try:
                df = read_github_csv(url).drop(columns=['flooding_date'], errors='ignore')
                dataframes.append(df)
            except Exception as e:
                st.error(f"Error reading {url}: {e}")

        if dataframes:
            combined_df = pd.concat(dataframes, axis=1)
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
            processed_df = process_rs_data(combined_df)
            
            fig = plot_cumulative_area(processed_df, title)
            
            col1, col2 = st.columns([3, 1])
            col1.pyplot(fig)
            col2.subheader("Data Sample")
            col2.write(processed_df.head(20))
        else:
            st.warning(f"No {title} data available.")
    def plot_cumulative_area(df, title):
        """Plot cumulative flooded area."""
        years = df['Time'].dt.year.unique()
        fig, ax = plt.subplots(figsize=(10, 6))

        for year in years:
            year_df = df[df['Time'].dt.year == year]
            ax.plot(year_df['DOY'], year_df['Area(ha)'], marker='o', linestyle='-', label=f'RS Area {year}')

        ax.axvline(START_PLANTING, color='blue', linestyle='--', label='Start Planting (15 Feb)')
        ax.axvline(END_PLANTING, color='green', linestyle='--', label='End Planting (15 Mar)')
        ax.axvline(START_HARVESTING, color='orange', linestyle='--', label='Start Harvesting (5 Jul)')
        ax.axvline(END_HARVESTING, color='red', linestyle='--', label='End Harvesting (16 Sep)')

        ax.set_title(title)
        ax.set_xlabel('Day of Year (DOY)')
        ax.set_ylabel('Area (ha)')
        ax.grid(True)
        ax.legend()
        
        return fig

    def process_data(urls, title):
        """Process and plot data from a list of URLs."""
        dataframes = []
        for url in urls:
           try:
              df = read_github_csv(url).drop(columns=['flooding_date'], errors='ignore')
              dataframes.append(df)
           except Exception as e:
              st.error(f"Error reading {url}: {e}")

        if dataframes:
            combined_df = pd.concat(dataframes, axis=1)
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
            processed_df = process_rs_data(combined_df)
        
            fig = plot_cumulative_area(processed_df, title)
        
            col1, col2 = st.columns([3, 1])
            col1.pyplot(fig)
            col2.subheader("Data Sample")
            col2.write(processed_df.head(20))
        else:
           st.warning(f"No {title} data available.")

    
    for year in range(2019, 2024):
        if year in selected_years:
            dagana_urls = [f'https://github.com/ICRISAT-Senegal/Remote-sensing/blob/main/flooding_data_{year}.csv' for year in range(2019, 2025)]
            process_data(dagana_urls, "2019-2024 Cumulative Flooded Areas using Dagana Plots")
            agcelerant_urls = [f'https://github.com/ICRISAT-Senegal/Remote-sensing/blob/main/combined_flooding_data_{year}.csv' for year in range(2019, 2025)]
            process_data(agcelerant_urls, "2019-2024 Cumulative Flooded Areas using agCelerant Plots")


# Create an expander for methodology
with st.expander("Methodology & Data Sources"):
    st.markdown("""
    ### Data Collection Methodology
    
    This dashboard uses the following data sources and processing steps:
    
    1. **Satellite Imagery**
       - Source: Sentinel-2 MSI Level-2A imagery
       - Resolution: 10m for RGB bands, 20m for water detection bands
       - Cloud filtering: Images with >20% cloud cover are excluded
    
    2. **Water Detection Algorithm**
       - MNDWI (Modified Normalized Difference Water Index) is calculated using:
         `MNDWI = (Green - SWIR) / (Green + SWIR)`
       - Threshold: Pixels with MNDWI > 0 are classified as water
    
    3. **SAED Field Data**
       - Ground truth data collected by SAED field agents
       - Includes prepared area and planted area measurements
    
    4. **Processing Steps**
       - Cloud masking
       - MNDWI calculation
       - Thresholding and classification
       - Area calculation per grid cell
       - Temporal aggregation
    """)

# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#7f8c8d;font-size:0.8rem">
    © 2025 Paddy Flooding Detection Project | Data Sources: Sentinel-2, SAED | Last updated: March 4, 2025
</div>
""", unsafe_allow_html=True)

