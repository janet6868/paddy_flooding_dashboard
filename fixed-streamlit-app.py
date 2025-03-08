# Import core libraries
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
from streamlit_folium import st_folium, folium_static

# Geospatial libraries
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap
import warnings

# Import the updated flood detection module
from fixed_sentinel_detection import (
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
    .folium-map {
        width: 100%;
        height: 500px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
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
    
    # Define the start and end dates for planting period
    season_start = datetime.strptime('2025-02-15', '%Y-%m-%d')
    season_end = datetime.strptime('2025-03-15', '%Y-%m-%d')  # Fixed date
    
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
                    return dff['Area(ha)'].sum(), dff['Area(ha)'].mean(), dff['Area(ha)'].max(), "N/A"
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

def plot_cumulative_area(df, title):
    """Plot cumulative flooded area for historical comparison."""
    # Constants for planting and harvesting periods
    START_PLANTING = 46  # 15 Feb
    END_PLANTING = 74    # 15 Mar
    START_HARVESTING = 186  # 5 Jul
    END_HARVESTING = 259    # 16 Sep
    
    years = df['Time'].dt.year.unique()
    fig, ax = plt.subplots(figsize=(10, 6))

    for year in years:
        year_df = df[df['Time'].dt.year == year]
        ax.plot(year_df['DOY'], year_df['Area(ha)'], marker='o', linestyle='-', label=f'RS Area {year}')

    ax.axvline(START_PLANTING, color='blue', linestyle='--', label='Start Planting (15 Feb)')
    ax.axvline(END_PLANTING, color='green', linestyle='--', label='End Planting (15 Mar)')
    ax.axvline(START_HARVESTING, color='orange', linestyle='--', label='Start Harvesting (5 Jul)')
    ax.axvline(END_HARVESTING, color='red', linestyle='--', label='End Harvesting (16 Sep)')

    # Add shaded regions for planting and harvesting periods
    ax.axvspan(START_PLANTING, END_PLANTING, alpha=0.2, color='lightblue', label='Planting Period')
    ax.axvspan(START_HARVESTING, END_HARVESTING, alpha=0.2, color='lightgreen', label='Harvesting Period')

    ax.set_title(title)
    ax.set_xlabel('Day of Year (DOY)')
    ax.set_ylabel('Area (ha)')
    ax.grid(True)
    ax.legend()
    
    # Set x-axis to show months
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_positions = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    ax.set_xticks(month_positions)
    ax.set_xticklabels(month_labels)
    
    return fig