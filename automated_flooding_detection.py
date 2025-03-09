#________________________v3__________________________________
# Make sure all imports come first
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os
import time

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.dates import DateFormatter, DayLocator
from streamlit_folium import st_folium, folium_static

# Geospatial libraries
import geopandas as gpd
import contextily as ctx
import folium
from folium.plugins import MarkerCluster, HeatMap
import rasterio as rio
from rasterio.plot import show
from PIL import Image

# CRITICAL: st.set_page_config() must be the ABSOLUTE first Streamlit command
st.set_page_config(
    page_title="Paddy Flooding Monitoring",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now we can import the module with local functions
# Importing this after set_page_config to avoid any potential Streamlit commands inside
from sentinel2_flooding_detection import (
    run_detection_flooding,
    dagana,  # geometry for entire Dagana region
    grid     # FeatureCollection
)

# Hard-coded planting & harvesting DOYs, plus TIF colormap
PLANTING_DOY = (46, 74)   # 15 Feb (46) - 15 Mar (74)
HARVEST_DOY  = (186, 259) # 5 Jul (186) - 16 Sep (259)
TIF_COLORS   = ["white", "blue", "green", "yellow", "orange", "brown", "red"]

#####################################################
# Initialize session state variables
#####################################################
# Initialize session state variables if they don't exist
if 'prev_filters' not in st.session_state:
    st.session_state.prev_filters = {
        'season_choice': None,
        'area_choice': None,
        'start_date': None,
        'end_date': None,
        'year': None
    }

if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

def check_filters_changed():
    """Check if any filters have changed from their previous values"""
    current_filters = {
        'season_choice': st.session_state.season_choice,
        'area_choice': st.session_state.area_choice,
        'start_date': st.session_state.start_date,
        'end_date': st.session_state.end_date,
        'year': st.session_state.year
    }
    
    changed = current_filters != st.session_state.prev_filters
    st.session_state.prev_filters = current_filters.copy()
    return changed

def trigger_run_analysis():
    """Callback to set run_analysis flag to True when button is clicked"""
    st.session_state.run_analysis = True

#####################################################
# 2) HELPER FUNCTIONS
#####################################################
def load_remote_flood_data(year: int, region: str) -> pd.DataFrame:
    """
    Load a remote CSV for the given year and region (Dagana or agCelerant)
    from GitHub and return a cleaned DataFrame. 
    Drops columns like 'flooding_date' if present.
    """
    # Adjust logic for your actual GitHub paths
    if region == "Entire Dagana Region":
        url = f"https://github.com/janet6868/paddy_flooding_dashboard/blob/main/flooding_data_{year}.csv"
    else:
        url = f"https://github.com/janet6868/paddy_flooding_dashboard/blob/main/combined_flooding_data_{year}.csv"

    raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    df_temp = pd.read_csv(raw_url).drop(columns=["flooding_date"], errors="ignore")
    return df_temp

def get_area_column(df: pd.DataFrame) -> str:
    """
    Find the first area-like column in df.
    Returns the column name or None if not found.
    """
    for col in ["flooded_area_ha","flooding_area","area","Area"]:
        if col in df.columns:
            return col
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if col != "grid_id":
            return col
    return None

def process_remote_sensing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    From a DataFrame containing date columns of flooding data,
    sum the columns by date and return a new DataFrame with
    Time, Area(ha), Year, DOY.
    """
    rs_df = df.filter(regex=r"\d{4}-?\d{2}-?\d{2}$")
    area_rs = rs_df.sum(axis=0)
    rs_df_combined = pd.DataFrame({
        "Time": area_rs.index,
        "Area(ha)": area_rs.values
    })
    rs_df_combined["Time"] = pd.to_datetime(rs_df_combined["Time"])
    rs_df_combined["Year"] = rs_df_combined["Time"].dt.year
    rs_df_combined["DOY"]  = rs_df_combined["Time"].dt.dayofyear
    return rs_df_combined

def show_flooded_map_png(image_path: str, title: str):
    """
    Open a PNG image from the given path and display it using st.image().
    """
    raw_url_image = image_path.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    try:
        #image = Image.open(raw_url_image)
        #st.image(image, caption=title, use_column_width=True)
        st.image(raw_url_image, caption=title, use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying image '{raw_url_image}': {e}")

def create_statistics_summary(df_final: pd.DataFrame):
    """
    Return (total_area, avg_area, max_area, date_str).
    If df is empty or no area col found, returns zero stats.
    """
    if df_final.empty:
        return 0, 0, 0, "N/A"

    area_col = get_area_column(df_final)
    if area_col:
        total_area = df_final[area_col].sum()
        avg_area   = df_final[area_col].mean()
        max_area   = df_final[area_col].max()
    else:
        total_area, avg_area, max_area = 0, 0, 0

    # Attempt to find a column that is date-like
    date_str = "N/A"
    for col in df_final.columns:
        if "Date" in col or "date" in col:
            try:
                dmax = pd.to_datetime(df_final[col]).max()
                date_str = str(dmax.date())
                break
            except:
                continue

    return total_area, avg_area, max_area, date_str

def create_flooding_map_plot(map_path: str, boundary_path: str, title: str):
    """
    Returns a matplotlib Figure of the TIF map with boundary,
    so we can display via st.pyplot.
    """
    raw_url_map = map_path.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    raw_url_geojson = boundary_path.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    try:
        with rio.open(raw_url_map) as src:
            mp = src.read(1)
            extent = rio.plot.plotting_extent(src)
            raster_crs = src.crs

            # Build colormap
            colors = TIF_COLORS
            max_val = np.max(mp)
            bounds = [0, 20, 40, 60, 80, 100, 120]
            if max_val > 120:
                bounds.append(max_val + 1)
            else:
                bounds.append(121)

            cmap = ListedColormap(colors)
            norm = BoundaryNorm(bounds, len(colors))

            # Mask out certain values
            masked_mp = np.ma.masked_where((mp == 7) | (mp == 0), mp)

            boundary = gpd.read_file(raw_url_geojson)
            if boundary.crs != raster_crs:
                boundary = boundary.to_crs(raster_crs)

            # Clear any existing figures to prevent legend stacking
            plt.close('all')
            
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            show(masked_mp, ax=ax, cmap=cmap, norm=norm, extent=extent)

            # Plot boundary
            boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.5)

            # Create colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array(masked_mp)
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label("Day of the year", fontsize=12, labelpad=10)
            tick_labels = ["0-20","21-40","41-60","61-80","81-100","101-120", f">{120}"]
            cbar.set_ticks(np.array(bounds[:-1]) + np.diff(bounds)/2)
            cbar.set_ticklabels(tick_labels)

            ax.set_title(title, fontsize=12, pad=20)
            ax.set_xlabel("Longitude", fontsize=12)
            ax.set_ylabel("Latitude", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            return fig
    except Exception as e:
        st.error(f"Error processing map '{map_path}': {e}")
        return None

#####################################################
# MAIN APP LAYOUT
#####################################################
st.markdown("<h1 class='main-header'>Paddy Flooding Monitoring </h1>", 
            unsafe_allow_html=True)

# =============== SIDEBAR ===============
with st.sidebar:
    st.image("logo.png", width=400)

    st.markdown("## Choose Season")
    season_choice = st.selectbox(
        "Select a season:",
        ["Dry Hot Season", "Cold Season", "Rainy Season"],
        key="season_choice"
    )

    st.markdown("## Choose Area of Focus")
    area_choice = st.selectbox(
        "Select area:",
        ["Entire Dagana Region", "agCelerant Plots"],
        key="area_choice"
    )

    st.markdown("## Choose Date Range")
    default_start = datetime(2025, 1, 26)
    default_end   = datetime(2025, 3, 9)
    start_date = st.date_input("Start Date:", default_start, key="start_date")
    end_date   = st.date_input("End Date:", default_end, key="end_date")

    st.markdown("## Year")
    year = st.text_input("Year", "2025", key="year")

    with st.expander("Advanced Options"):
        cloud_cover     = st.slider("Max Cloud Cover (%)", 0, 100, 20)
        mndwi_threshold = st.slider("MNDWI Threshold", -0.5, 0.5, 0.0, 0.05)

    with st.expander("Help & Information"):
        st.markdown("""
        **About this monitoring:**
        - Uses Sentinel-2 imagery to detect flooded paddy fields
        - Compares remote sensing data with SAED ground data
        - Provides both current season and historical analysis
        
        For more information, contact: your@email.com
        """)

# AOI logic
if area_choice == "Entire Dagana Region":
    selected_aoi = dagana
else:
    # If you have geometry for "agCelerant Plots," use that. 
    # For demonstration, fallback to dagana:
    selected_aoi = dagana

# =============== MAIN TABS ===============
tab_current, tab_history = st.tabs(["Current Season Analysis", "Historical Comparison"])

#####################################################
# 4) CURRENT SEASON ANALYSIS
#####################################################
with tab_current:
    st.markdown(
        f"<h2 class='sub-header'>{season_choice} – {area_choice}: Current Analysis</h2>", 
        unsafe_allow_html=True
    )
    st.info(f"""
        This analysis uses Earth Engine data to detect flooded areas (MNDWI) from Sentinel-2,
        focusing on the {area_choice}. The chosen season is {season_choice}.
    """)

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        run_btn = st.button("Run Flooding Detection", use_container_width=True, on_click=trigger_run_analysis)
    with col2:
        download_btn = st.download_button(
            "Download Results",
            data="data_placeholder",
            file_name="flood_detection_results.csv",
            use_container_width=True,
            disabled=True
        )
    with col3:
        clear_btn = st.button("Clear Results", use_container_width=True)

    # Some quick button styling
    st.markdown('''
    <style>
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
        background-color: #001A6E !important;
        color: white !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
        background-color: #FFF4B7 !important;
        color: black !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) button {
        background-color: #FFF4B7 !important;
        color: black !important;
    }
    </style>
    ''', unsafe_allow_html=True)

    # Run detection only if explicit run button clicked or if no df in session
    if st.session_state.run_analysis or "df_final" not in st.session_state:
        with st.spinner("Processing flood detection..."):
            prog_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                prog_bar.progress(i + 1)

            # This is your function from up_sentinel2_flooding_detection
            df_final, m = run_detection_flooding(
                aoi=selected_aoi,
                grid=grid,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                year=year
            )
            st.session_state.df_final = df_final
            st.session_state.m = m
            
            # Reset the run_analysis flag
            st.session_state.run_analysis = False

            output_file_name = f"floodingData_{year}.csv"
            try:
                _ = pd.read_csv(output_file_name)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # Clear results if requested
    if clear_btn:
        for k in ["df_final","m","rs_df","saed_df","combined_df"]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.run_analysis = False
        st.experimental_rerun()

    if "df_final" in st.session_state and "m" in st.session_state:
        total_area, avg_area, max_area, latest_date = create_statistics_summary(st.session_state.df_final)

        # 2-column layout
        col_map, col_stats = st.columns([3, 1])
        with col_map:
            st.markdown("<h3 align='center'>Spatial Map for Flooded Areas</h3>", unsafe_allow_html=True)
            st_folium(st.session_state.m, width=700)
        with col_stats:
            st.markdown(
                f"""
                <style>
                .stat-container {{
                    background-color: #F0F2F6;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 10px 0;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .stat-title {{
                    font-size: 1.1em;
                    font-weight: 600;
                    margin-bottom: 0.3em;
                    color: #333;
                }}
                .stat-value {{
                    font-size: 1.5em;
                    font-weight: 700;
                    color: #2c3e50;
                }}
                </style>

                <div style="text-align:center; margin-bottom: 20px;">
                    <h3>Flooding Stats to Date</h3>
                </div>

                <div class="stat-container">
                    <div class="stat-title">Latest Data Date</div>
                    <div class="stat-value">{latest_date}</div>
                </div>

                <div class="stat-container">
                    <div class="stat-title">Total Flooded Area</div>
                    <div class="stat-value">{total_area:.1f} ha</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # Tabular data & chart
        col_left, col_right = st.columns([2, 1])
        with col_right:
            st.subheader("Flooded Areas Tabular Data")
            st.dataframe(st.session_state.df_final)

        with col_left:
            st.subheader(f"SAED & Remote Sensing Comparison – {season_choice}")
            output_file_name = f"floodingData_{year}.csv"
            try:
                # Clear any existing matplotlib figures to prevent legend stacking
                plt.close('all')
                
                rs_df = pd.read_csv(output_file_name)
                rs_dff = rs_df.filter(regex=r"\d{4}-?\d{2}-?\d{2}$")
                area_sum = rs_dff.sum(axis=0)
                df_rs    = pd.DataFrame()
                df_rs["Date"]        = list(area_sum.index)
                df_rs["Area(ha)"]    = list(area_sum.values)
                df_rs["Data_source"] = f"Remote sensing {year}"
                df_rs["date"]        = pd.to_datetime(df_rs["Date"])

                # Suppose we have a local SAED CSV for 2025
                saed_csv_path = "saed_2025.csv"
                df_saed = pd.read_csv(saed_csv_path)
                df_saed["date"] = pd.to_datetime(df_saed["Date"])

                combined_df = pd.concat([df_rs, df_saed])
                combined_df["Day"]   = combined_df["date"].dt.day
                combined_df["Month"] = combined_df["date"].dt.month
                combined_df["Year"]  = combined_df["date"].dt.year
                combined_df["Days"]  = combined_df["date"].dt.dayofyear

                palette = {
                    f"Remote sensing {year}": "blue",
                    "SAED prepared area": "purple",
                    "SAED planted area": "orange"
                }

                # Store current figure parameters
                original_figsize = plt.rcParams["figure.figsize"].copy()
                scale_factor = 20/2
                plt.rcParams["figure.figsize"] = (scale_factor, scale_factor*0.6)

                fig, ax = plt.subplots()
                season_start = datetime(2025, 2, 15)
                season_end   = datetime(2025, 3, 15)

                sns.lineplot(
                    data=combined_df,
                    x="date",
                    y="Area(ha)",
                    hue="Data_source",
                    marker="o",
                    palette=palette,
                    ax=ax
                )

                ax.set_title(f"{year} {season_choice} Flooded Areas (ha)")
                ax.set_xlabel("Date")
                ax.set_ylabel("Area (ha)")

                # Planting period shading
                ax.axvspan(season_start, season_end, color="grey", alpha=0.3, label="Planting period")
                ax.axvline(season_start, color="black", linestyle="--")
                ax.axvline(season_end,   color="black", linestyle="--")

                mid_date = season_start + (season_end - season_start)/2
                ax.text(mid_date, 7000, "Planting period\n(15 FEB - 15 MAR)",
                        ha="center", va="center", color="black",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),rotation=90, rotation_mode="anchor")

                ax.legend(title="Data Source", bbox_to_anchor=(1.05,1), loc="upper left")
                ax.grid(True)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                st.pyplot(fig)
                
                # Restore original figure parameters
                plt.rcParams["figure.figsize"] = original_figsize
                plt.close(fig)

            except Exception as e:
                st.warning(f"No comparison CSV found or error reading data: {e}")


#####################################################
# 5) HISTORICAL COMPARISON
#####################################################
with tab_history:
    st.markdown(
        f"<h2 class='sub-header'>{season_choice} – {area_choice}: Historical Comparison</h2>", 
        unsafe_allow_html=True
    )

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        view_type = st.radio("View Type", ["Show history", "Re run the season"], horizontal=True)
    with filter_col2:
        st.write("Area chosen:", area_choice)

    if view_type == "Re run the season":
        st.warning("Work in progress – you will be able to rerun fresh flooding areas and maps for each year.")
    
    elif view_type == "Show history":
        #st.subheader("1) Cumulative Flooded-Area Curve (2019–2025)")

        selected_years = st.multiselect(
            "Select years for cumulative curve (including 2025 local data):",
            [2019, 2020, 2021, 2022, 2023, 2024, 2025],
            default=[2019, 2020, 2021, 2022,2023, 2024, 2025]
        )
        if selected_years:
            min_year = min(selected_years)
            max_year = max(selected_years)
            year_range = f"{min_year}–{max_year}"
        else:
            year_range = "No years selected"
        st.subheader("1) Cumulative Flooded-Area Curve (2019–2025)")

        # Clear any existing figures
        plt.close('all')
        
        all_dataframes = []
        for yr in selected_years:
            try:
                if yr == 2025:
                    # read local CSV
                    df_temp = pd.read_csv("floodingData_2025.csv").drop(
                        columns=["Est_flooding_date"], errors="ignore"
                    )
                else:
                    # read from GitHub
                    df_temp = load_remote_flood_data(yr, area_choice)
                all_dataframes.append(df_temp)
            except Exception as e:
                st.error(f"Error reading data for year {yr}: {e}")

        if all_dataframes:
            combined_df = pd.concat(all_dataframes, axis=1)
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

            processed_df = process_remote_sensing_data(combined_df)

            # For a custom color scheme:
            year_colors = {
                2019: "red",
                2020: "#1D1616",
                2021: "green",
                2022: "purple",
                2023: "orange",
                2024: "#00FF9C",
                2025: "blue"
            }

            fig, ax = plt.subplots(figsize=(10, 5))
            for y in sorted(processed_df["Year"].unique()):
                sub_df = processed_df[processed_df["Year"] == y]
                color  = year_colors.get(y, "black")
                ax.plot(sub_df["DOY"], sub_df["Area(ha)"], label=str(y), color=color)

            # Planting period
            (p_start, p_end) = PLANTING_DOY
            mid_date = p_start + (p_end - p_start)/2
            ax.text(mid_date, 25000, "Planting period\n(15 FEB - 15 MAR)",
                    ha="center", va="center", color="black",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                    rotation=90, rotation_mode="anchor")
            ax.axvspan(p_start, p_end, color="grey", alpha=0.3)
            ax.axvline(p_start, color="black", linestyle="--")
            ax.axvline(p_end,   color="black", linestyle="--")

            # Harvesting period
            (h_start, h_end) = HARVEST_DOY
            mid_dateh = h_start + (h_end - h_start)/2
            ax.text(mid_dateh, 25000, "Harvesting period\n(5 JUL - 16 SEP)",
                    ha="center", va="center", color="black",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                    rotation=90, rotation_mode="anchor")
            ax.axvspan(h_start, h_end, color="green", alpha=0.3)
            ax.axvline(h_start, color="black", linestyle="--")
            ax.axvline(h_end,   color="black", linestyle="--")

            ax.set_title(f"Historical Flooded Areas ({season_choice}, {area_choice}) - 2019–2025")
            ax.set_xlabel("Day of Year (DOY)")
            ax.set_ylabel("Area (ha)")
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.05,1), loc="upper left")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No data found for the chosen years in the cumulative curve.")

        # ------------------------------------------------------------
        # 2) TIF-based flooded maps for 2019–2024 (Up to 2 years)
        # ------------------------------------------------------------
        # st.subheader("2) Flooded-Map TIFFs (2019–2024)")

        # map_years = st.multiselect(
        #     "Select up to two years for TIF-based maps (2019–2024):",
        #     [ 2023, 2024],#2019, 2020, 2021, 2022,
        #     default=[]
        # )
        # if len(map_years) > 2:
        #     #st.warning("Please select up to 2 years only. Using the first two selected.")
        #     st.warning("Please dont take this option - work in progress because of limited resources in the streamlit community cloud.")
        #     map_years = map_years[:2]
        
        # for y in map_years:
        #     if area_choice == "Entire Dagana Region":
        #         boundary_path = f"https://github.com/janet6868/paddy_flooding_dashboard/blob/main/dagana_region.geojson"
        #         map_path = f"https://github.com/janet6868/paddy_flooding_dashboard/blob/main/flooding_map_Dagana{y}.tif"
        #         map_title = f"{season_choice} Dagana Flooding Map {y}"
        #     else:
        #         boundary_path =f"https://github.com/janet6868/paddy_flooding_dashboard/blob/main/agcelerant_plots.geojson"
        #         map_path = f"https://github.com/janet6868/paddy_flooding_dashboard/blob/main/flooding_map_agcelerant_{y}.tif"
        #         map_title = f"{season_choice} agCelerant Flooding Map {y}"

        #     fig_map = create_flooding_map_plot(map_path, boundary_path, map_title)
        #     if fig_map:
        #         st.pyplot(fig_map)
        #         plt.close(fig_map)  # Close figure after displaying
        #     else:
        #         st.warning(f"No TIF found (or error) for {y} – {area_choice}.")

        # ---------------------------
        # PNG-based Flooded Map Display
        # ---------------------------
        st.subheader("2) Flooded areas spatial maps(2019–2024)")
        map_years = st.multiselect(
            "Select up to two years for PNG-based maps (2019–2024):",
            [2023, 2024],
            default=[]
        )
        if len(map_years) > 2:
            st.warning("Please select up to 2 years only. Using the first two selected.")
            map_years = map_years[:2]
        for y in map_years:
            if area_choice == "Entire Dagana Region":
                image_path = f"https://github.com/janet6868/paddy_flooding_dashboard/blob/main/flooding_map_Dagana{y}.png"
                map_title = f"{season_choice} Dagana Flooding Map {y}"
            else:
                image_path = f"https://github.com/janet6868/paddy_flooding_dashboard/blob/main/flooding_map_agcelerant_{y}.png"
                map_title = f"{season_choice} agCelerant Flooding Map {y}"
            show_flooded_map_png(image_path, map_title)

#####################################
# 6) METHODOLOGY & FOOTER
#####################################
with st.expander("Methodology & Data Sources"):
    st.markdown("""
    ### Data Collection Methodology
    
    This monitoring uses the following data sources and processing steps:
    
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

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#7f8c8d;font-size:0.8rem">
    © 2025 Paddy Flooding Detection Project | Data Sources: Sentinel-2, SAED | Last updated: March 8, 2025
</div>
""", unsafe_allow_html=True)

