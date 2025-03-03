

"""
"""
#%%
## Import the libraries
import rasterio
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from io import StringIO

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import glob
import pandas as pd
# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")
# Title of the Streamlit App
st.title("Paddy Flooding Detection using Sentinel 2 Analysis (2019-2024)")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
START_PLANTING = 46  # 15 Feb
END_PLANTING = 74    # 15 Mar
START_HARVESTING = 186  # 5 Jul
END_HARVESTING = 259    # 16 Sep

def read_github_csv(url):
    """Read CSV from GitHub URL."""
    raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return pd.read_csv(raw_url)

def process_rs_data(df):
    rs_df = df.filter(regex=('\d{4}-?\d{2}-?\d{2}$'))
    area_rs = rs_df.sum(axis=0)
    rs_df_combined = pd.DataFrame({
        'Time': area_rs.index,
        'Area(ha)': area_rs.values
    })
    rs_df_combined['Year'] = rs_df_combined['Time'].str[:4]
    rs_df_combined['Class'] = 'RS_' + rs_df_combined['Year']
    rs_df_combined['Time'] = pd.to_datetime(rs_df_combined['Time'])
    rs_df_combined['DOY'] = rs_df_combined['Time'].dt.dayofyear
    return rs_df_combined

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

# Dagana Flooding Data
st.header("**1. Dagana Plots**")
dagana_urls = [f'https://github.com/ICRISAT-Senegal/Remote-sensing/blob/main/flooding_data_{year}.csv' for year in range(2019, 2025)]
process_data(dagana_urls, "2019-2024 Cumulative Flooded Areas using Dagana Plots")

# agCelerant Data
st.header("**2. agCelerant Plots**")
agcelerant_urls = [f'https://github.com/ICRISAT-Senegal/Remote-sensing/blob/main/combined_flooding_data_{year}.csv' for year in range(2019, 2025)]
process_data(agcelerant_urls, "2019-2024 Cumulative Flooded Areas using agCelerant Plots")


#___________________________________________________________credit information________________________-


# Header for Credit Information
st.header("3. Credit Information")

# Initial Hypotheses
st.subheader("Initial Hypotheses")
st.markdown("""
- **agCelerant Data**:
    - Increase in flooding → Early promise of credit.
    - Decrease in flooding → Delay in the promise of credit.
""")

# Learnings from Data Exploration
st.subheader("Learnings from Exploration of the Data")

st.subheader("First Spike Hypothesis")
st.markdown("""
- Some GIEs might have internal financial reserves that are used.
- Some GIEs have good creditworthiness and could use this to get a loan from the bank.
""")

st.subheader("Second Spike Hypothesis")
st.markdown("""
- Reaction by the second category of farmers (those who lack internal reserves and thin file GIEs) to the promise of credit.
- Or the same group has learned about the decisions of the banks on credit issuance.
""")

st.subheader("Early Flooding Hypothesis")
st.markdown("""
- Early flooding explains/reflects the credit disbursement process.
- Everyone starts the cropping cycle without any payment. The cycle is based on promises or commitments by the GIE to get credit.
- Some unions have reserves to start planting before funds are released.
- The early start does not necessarily mean the money has been released yet.
""")

# New Hypothesis
st.subheader("New Hypothesis")
st.markdown("""
- **Creditworthiness and Reserves of GIE Explain the Monotonic Pattern of Flooding**
    - Does the absence of a plateau imply reserves?
    - Is there a significant difference between the lengths of the plateaus? Does a significant difference in the length of the plateau reflect different access to credit? 
    - Extract the date of the credit committee for the Dry hot seasons from 2017-2024 years.
    - Explore the topology of GIE.
""")

# Action Points
st.subheader("Action Points")
st.markdown("""
- Consider **quantitative data analysis**: This approach is more process-based and accumulates evidence from many points to understand how the accumulation of events (like floods) reflects credit disbursement timelines.
- Zoom in to analyze **GIEs**:
    - See their performance in terms of their decision to go into production.
    - The commitment of credit & the farmers' response.
- Extract boundaries and calculate the total areas for confirmation with "Soule".
- Investigate the date of credit committee meetings and decisions for each year.
- Investigate when money starts flowing for them (GIE).
""")

st.markdown("General Information: The dataset contains 4,608 rows and 258 columns. "
            "The `parcel_cod` column has 596 unique values, indicating multiple parcels. "
            "The `operating_account` column contains 163 unique types of operations, "
            "with 'Assurance CNAAS' being the most frequent. "
            "There are missing data in `credit_auth_date` and `credit_exec_date`.")



#_____________________________________________________________uuuuuuuuuuuuuu____________________________
file_url ='https://github.com/ICRISAT-Senegal/Remote-sensing/blob/main/merged_cr_rs_data.csv'
#read_github_csv(url)
combined_df_credit = read_github_csv(file_url)#pd.read_csv(file, index_col=0,na_values=np.nan)

st.markdown("**Sample of the Credit Data:**")
st.dataframe(combined_df_credit.head(20))

# # Assuming 'unique_col' (e.g., 'farmer_id') is the column that identifies farmers
# unique_col = 'culture_id'  # Replace this with the actual farmer ID or unique column name

# # Step 1: Group by unique_col (farmers) and check if they have at least one complete row for credit details
# has_credit = combined_df_credit.groupby(unique_col).apply(
#     lambda group: group.dropna(subset=['credit_req_date', 'credit_auth_date', 'credit_exec_date']).shape[0] > 0
# ).reset_index(name='has_credit')

# # # Step 2: Separate farmers into two categories based on whether they have complete credit details or not
# # has_credit_details = has_credit[has_credit['has_credit'] == True][unique_col]
# # no_credit_details = has_credit[has_credit['has_credit'] == False][unique_col]

# # Step 2: Separate farmers into two categories based on whether they have complete credit details or not
# farmers_with_credit = has_credit[has_credit['has_credit'] == True][unique_col]
# farmers_without_credit = has_credit[has_credit['has_credit'] == False][unique_col]

# # Step 3: Filter the original dataframe for rows belonging to these two groups
# has_credit_details = combined_df_credit[combined_df_credit[unique_col].isin(farmers_with_credit)]
# st.write(has_credit_details.culture_id.value_counts())
# no_credit_details = combined_df_credit[combined_df_credit[unique_col].isin(farmers_without_credit)]
# st.write(no_credit_details.culture_id.value_counts())
# # Separate rows with and without credit details
# #has_credit_details = combined_df_credit.dropna(subset=['credit_req_date', 'credit_auth_date', 'credit_exec_date'])
# #no_credit_details = combined_df_credit[combined_df_credit[['credit_req_date', 'credit_auth_date', 'credit_exec_date']].isna().all(axis=1)]
# st.write(has_credit_details)
# st.write(no_credit_details)
# # Function to process the remote sensing data for cumulative area
# def process_rs_data(merged_df):
#     rs_df = merged_df.filter(regex=('\d{4}-?\d{2}-?\d{2}$'))  # Date columns
#     area_rs = rs_df.sum(axis=0)  # Summing values row-wise
#     rs_df_combined = pd.DataFrame()
#     rs_df_combined['Time'] = list(area_rs.index)
#     rs_df_combined['Area(ha)'] = list(area_rs.values)
#     rs_df_combined['Year'] = rs_df_combined['Time'].str[:4]
#     rs_df_combined['Class'] = 'RS_' + rs_df_combined['Year']
#     rs_df_combined['Time'] = pd.to_datetime(rs_df_combined['Time'])
#     rs_df_combined['DOY'] = rs_df_combined['Time'].apply(lambda x: x.timetuple().tm_yday)
#     return rs_df_combined

# # Process both datasets
# has_credit_df = process_rs_data(has_credit_details)
# no_credit_df = process_rs_data(no_credit_details)

# # Convert credit dates to DOY
# def convert_to_doy(date_string):
#     date = pd.to_datetime(date_string)
#     return date.dayofyear

# # Extract DOY for all credit request, authorization, and execution dates
# credit_req_doy = has_credit_details['credit_req_date'].apply(convert_to_doy).values
# credit_auth_doy = has_credit_details['credit_auth_date'].dropna().apply(convert_to_doy).values
# credit_exec_doy = has_credit_details['credit_exec_date'].dropna().apply(convert_to_doy).values
#     # Visualization: Lagged Correlation and Box Plot in same row but different columns
# fig, (ax_lag, ax_box) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# # Display dataframes in two columns
# col1, col2 = st.columns(2)
    
# with col1:
#     st.markdown("**Sample agCelerant data with Credit Details**")
#     st.dataframe(has_credit_df.head())

# with col2:
#     st.markdown("**Sample agCelerant data without Credit Details**")
#     st.dataframe(no_credit_df.head())

# # List of years to plot
# years = has_credit_df['Time'].dt.year.unique()

# # Visualization: Plot cumulative flooded areas by year with two categories
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
# axs = axs.flatten()  # Flatten the grid to iterate over each axis easily


# ## 5. Boxplot Before and After Credit Events

# window = 7  # Days before and after the event
# # Loop through each year and plot the cumulative flooded areas
# for i, year in enumerate(years):
#     if i >= len(axs):  # Avoid index out of bounds
#         break

#     ax = axs[i]

#     # Plot "With Credit" data
#     year_df_with_credit = has_credit_df[has_credit_df['Time'].dt.year == year]
#     ax.plot(year_df_with_credit['DOY'], year_df_with_credit['Area(ha)'], marker='o', linestyle='-', label=f'With Credit {year}', color='blue')

#     # Plot "Without Credit" data
#     year_df_without_credit = no_credit_df[no_credit_df['Time'].dt.year == year]
#     ax.plot(year_df_without_credit['DOY'], year_df_without_credit['Area(ha)'], marker='x', linestyle='--', label=f'Without Credit {year}', color='red')

#     # Customize each subplot
#     ax.set_title(f'Cumulative Flooded Areas {year}', fontsize=10)
#     ax.set_xlabel('Day of Year (DOY)')
#     ax.set_ylabel('Cumulative Flooded Area (ha)')
#     ax.legend()
#     ax.grid(True)


# st.subheader("Is there any difference in the flooded areas for plots with credit and without credit details?")

# # Hide unused subplots if there are fewer than 6 years
# for i in range(len(years), 6):
#     fig.delaxes(axs[i])
# # Adjust layout to avoid overlapping subplots
# plt.tight_layout()
# st.pyplot(fig)

# # Display the plot
# st.subheader("Is there any relationship between credit and flooding activities for all years?")
# st.markdown("""
# - There seems to be a clear short-term relationship between credit execution and flooding, especially within the first few days after credit is granted. This is more evident in some years, particularly 2023 and 2024.
# - The box plots reinforce the finding that, for some years (particularly 2023 and 2024), flooded areas tend to increase after credit execution, possibly indicating that these years experienced more significant events (such as floods) in response to financial interventions.
# """)

# # Define function to plot boxplots with colors and label adjustments
# def plot_boxplot_before_after(ax, df, credit_date, label):
#     before_date = df[df['DOY'] < credit_date]
#     after_date = df[df['DOY'] >= credit_date]
#     # Plot boxplots with colors and rotated labels
#     box = ax.boxplot([before_date['Area(ha)'], after_date['Area(ha)']], labels=[f'{label} Before', f'{label} After'], patch_artist=True)

#     # Customize the colors of the boxplots
#     colors = ['#1f77b4', '#ff7f0e']  # Blue for before, orange for after
#     for patch, color in zip(box['boxes'], colors):
#         patch.set_facecolor(color)

#     ax.set_title(f'Flooded Area Before and After {label}')
#     # ax.set_xticklabels([f'{label} Before', f'{label} After'], rotation=45, ha='right')  
#     ax.set_xticks([1, 2])  # Ensure 2 ticks for "Before" and "After"
#     #ax.set_xticklabels([f'{label} Before', f'{label} After'], rotation=45, ha='right') 
#     ax.set_xticklabels([f'Before Credit Execution', f'After Credit Execution'], rotation=45, ha='right') 
#     ax.tick_params(axis='x', labelsize=10)  # Set the font size for the labels


# # Boxplot Before and After Credit Events for each year (in 3 columns and 2 rows)
# fig_box, axs_box = plt.subplots(2, 3, figsize=(18, 12))
# axs_box = axs_box.flatten()  # Flatten to easily iterate over each subplot

# for i, year in enumerate(years):
#     if i >= len(axs_box):  # Avoid index out of bounds
#         break

#     year_df_with_credit = has_credit_df[has_credit_df['Time'].dt.year == year]

#     # Credit request DOY
#     if len(credit_req_doy) > 0:
#         plot_boxplot_before_after(axs_box[i], year_df_with_credit, credit_req_doy[0], f'{year} Credit Request')
#     # Credit authorization DOY
#     if len(credit_auth_doy) > 0:
#         plot_boxplot_before_after(axs_box[i], year_df_with_credit, credit_auth_doy[0], f'{year} Credit Authorization')
#     # Credit execution DOY
#     if len(credit_exec_doy) > 0:
#         plot_boxplot_before_after(axs_box[i], year_df_with_credit, credit_exec_doy[0], f'{year} Credit Execution')

# Adjust layout to avoid overlapping subplots
# st.markdown('Before and After Credit Events for each year')
# plt.tight_layout()
# #st.pyplot(fig_box) --can be reactivated later
# container = st.container()
# #ontainer.write("how significant increases in flooded areas after credit execution, which could point to a stronger relationship between credit and increased flooding activity in these years.")

# st.markdown("""
# **Box Plots for Flooded Area Before and After Credit Execution**:
# - **Visual Comparison**:
#     - In most years (2019, 2020, 2021, 2022), the median flooded area appears relatively stable before and after credit execution, but **2023** and **2024** stand out with larger differences.
#     - **2023** shows a much larger flooded area after credit execution, while **2024** also indicates a noticeable increase in flooded area after credit execution.
# - **Outliers**: Some outliers (like in 2023) suggest that there were exceptional cases of increased flooded areas after credit execution.
# - **Conclusion**: While credit execution doesn't seem to drastically change the median flooded area for all years, **2023** and **2024** show significant increases in flooded areas after credit execution, which could point to a stronger relationship between credit and increased flooding activity in these years.
# """)

# # Post hoc analysis
# st.subheader('Post hoc analysis for only 2023 data')
# st.markdown("""
#             - The linear regression suggests a positive relationship between days_since_credit and Area(ha), but this relationship is not statistically significant (p-value = 0.225).
#             - The model explains 60% of the variability in the Area(ha) data, which is moderate, but the overall significance of the model is weak.
#              - P>|t| for days_since_credit is 0.225, which is higher than the significance level of 0.05, meaning this predictor is not statistically significant. 
#             In simpler terms, we cannot conclude with high confidence that days_since_credit has a significant effect on the Area(ha).
            
#             """)

# sub_2023 = has_credit_df[has_credit_df['Year']=='2023']
# def linear_regression(df, credit_date):
#     df['days_since_credit'] = df['DOY'] - credit_date
#     X = sm.add_constant(df['days_since_credit'])  # Adding constant for intercept
#     y = df['Area(ha)']
#     model = sm.OLS(y, X).fit()
#     return model

# # Running linear regression for the period after credit execution
# regression_model = linear_regression(sub_2023[sub_2023['DOY'] >= credit_exec_doy[0]], credit_exec_doy[0])

# # Displaying regression results
# st.write(regression_model.summary())

# # _________________Anova analysis

# # Function to divide data into 'before' and 'after' credit execution
# def divide_before_after(df, credit_date):
#     before_credit = df[df['DOY'] < credit_date]['Area(ha)']
#     after_credit = df[df['DOY'] >= credit_date]['Area(ha)']
#     return before_credit, after_credit

# # Assuming you have 'has_credit_df' dataframe and 'credit_exec_doy' contains the execution dates for each year
# years = has_credit_df['Time'].dt.year.unique()

# # Initialize an empty list to store ANOVA results
# anova_results = []

# # Loop through each year and perform ANOVA
# for year in years:
#     #st.write(f"### Year: {year}")

#     # Filter data for the current year
#     year_df_with_credit = has_credit_df[has_credit_df['Time'].dt.year == year]

#     # Get the credit execution date for the current year
#     credit_exec_date = credit_exec_doy[0]  # Replace with actual execution date per year if available

#     # Divide the flooded area data into before and after credit execution
#     before_credit, after_credit = divide_before_after(year_df_with_credit, credit_exec_date)

#     # Perform one-way ANOVA
#     anova_result = stats.f_oneway(before_credit, after_credit)

#     # Store results in the list as a dictionary
#     anova_results.append({
#         'Year': year,
#         'F-statistic': round(anova_result.statistic, 3),
#         'p-value': round(anova_result.pvalue, 4),
#         'Significant': 'Yes' if anova_result.pvalue < 0.05 else 'no statistically significant difference between the flooded areas before and after credit execution for {year}'
#     })

#     # Write ANOVA result
#     # st.write(f"ANOVA Result for {year}: F-statistic = {anova_result.statistic}, p-value = {anova_result.pvalue}")
#     # if anova_result.pvalue < 0.05:
#     #     st.write(f"There is a statistically significant difference between the flooded areas before and after credit execution for {year}.")
#     # else:
#     #     st.write(f"There is no statistically significant difference between the flooded areas before and after credit execution for {year}.")

# # Convert the results into a DataFrame
# anova_df = pd.DataFrame(anova_results)

# # Display the results in a table format
# st.write("### ANOVA Results Table")
# st.dataframe(anova_df)

# # Adjust layout to avoid overlap
# plt.tight_layout()

# # Display the full grid of plots
# #st.pyplot(fig)

#_____________________________________GIE analysis________________________________________-_______________________________
st.header("4. GIE Analysis")
# Load the merged data
data = combined_df_credit.copy()
# Convert None to empty strings
data = data.fillna('')

# Filter out rows with empty strings in the specified columns
has_credit_details = data[
    (data['credit_req_date'] != '') &
    (data['credit_auth_date'] != '') &
    (data['credit_exec_date'] != '')
]
#data = data.replace({None: np.nan})
# Move filters to the sidebar
#st.sidebar.header("GIE Based Analysis-Has")
# Filter GIEs with all credit details (credit_req_date, credit_auth_date, credit_exec_date)
#has_credit_details = data.dropna(subset=['credit_req_date', 'credit_auth_date', 'credit_exec_date'],how='all')
gies_with_credit_details = data['gie_name'].unique()
#%%
# Display number of all unique GIEs
all_gies = data['gie_name'].unique()
st.write(f"- Total number of GIEs available for analysis: {len(all_gies)}")
# Filter GIEs with all credit details (credit_req_date, credit_auth_date, credit_exec_date)
#has_credit_details = data.dropna(subset=['credit_req_date', 'credit_auth_date', 'credit_exec_date'])
gies_with_credit_details = has_credit_details['gie_name'].unique()
# Display number of GIEs with complete credit details
st.write(f"- Number of GIEs with complete credit details: {len(gies_with_credit_details)}")
st.write('-Time to execution: the number of days from credit request to execution')
# Select GIE for analysis in the sidebar
selected_gie = st.selectbox('Select GIE:', has_credit_details['gie_name'].unique())
st.markdown(f"**Details for GIE: {selected_gie}**")
filtered_data = has_credit_details[has_credit_details['gie_name'] == selected_gie]

# Convert the 'credit_req_date', 'credit_auth_date', and 'credit_exec_date' columns to datetime for analysis
filtered_data['credit_req_date'] = pd.to_datetime(data['credit_req_date'], errors='coerce')
filtered_data['credit_auth_date'] = pd.to_datetime(data['credit_auth_date'], errors='coerce')
filtered_data['credit_exec_date'] = pd.to_datetime(data['credit_exec_date'], errors='coerce')

# Calculate the time between request and execution
filtered_data['time_to_execution'] = (filtered_data['credit_exec_date'] - filtered_data['credit_req_date']).dt.days

# Display basic statistics for the time to execution
time_to_execution_stats = filtered_data['time_to_execution'].describe()

# Plotting a histogram of time to execution
fig2 = plt.figure(figsize=(10, 6))
plt.hist(filtered_data['time_to_execution'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title(f'Distribution of Time to Execution for Credit Requests for GIE: {selected_gie}')
plt.xlabel('Days from Credit Request to Execution')
plt.ylabel('Frequency')
plt.grid(True)

# Show the plot
plt.show()
st.pyplot(fig2)

# Process RS data (you can define this function)
gie_rs_df = process_rs_data(filtered_data)

# Plotting the Cumulative Area for the selected GIE
years = gie_rs_df['Time'].dt.year.unique()
fig, ax = plt.subplots(figsize=(10, 6))

for year in years:
    year_df = gie_rs_df[gie_rs_df['Time'].dt.year == year]
    ax.plot(year_df['DOY'], year_df['Area(ha)'], marker='o', linestyle='-', label=f'RS Area {year}')

start_planting = datetime.strptime('2023-02-15', '%Y-%m-%d').timetuple().tm_yday
end_planting = datetime.strptime('2023-03-15', '%Y-%m-%d').timetuple().tm_yday
start_harvesting = datetime.strptime('2023-07-05', '%Y-%m-%d').timetuple().tm_yday
end_harvesting = datetime.strptime('2023-09-16', '%Y-%m-%d').timetuple().tm_yday

ax.axvline(start_planting, color='blue', linestyle='--', label='Start Planting (15 Feb)')
ax.axvline(end_planting, color='green', linestyle='--', label='End Planting (15 Mar)')
ax.axvline(start_harvesting, color='orange', linestyle='--', label='Start Harvesting (5 Jul)')
ax.axvline(end_harvesting, color='red', linestyle='--', label='End Harvesting (16 Sep)')

ax.set_title(f'2019-2024 Cumulative Flooded Areas for GIE: {selected_gie}')
ax.set_xlabel('Day of Year (DOY)')
ax.set_ylabel('Area (ha)')
ax.grid(True)
ax.legend()
st.pyplot(fig)


# Bar chart for Irrigation Type
st.subheader(f"Count of Irrigation Types for: {selected_gie}")
irrigation_type_counts = filtered_data['irrigation_type'].value_counts()
fig_irrigation, ax_irrigation = plt.subplots(figsize=(8, 4))
ax_irrigation.bar(irrigation_type_counts.index, irrigation_type_counts.values, color='skyblue')
ax_irrigation.set_title('Irrigation Type Count')
ax_irrigation.set_xlabel('Irrigation Type')
ax_irrigation.set_ylabel('Count')
ax_irrigation.grid(True)
st.pyplot(fig_irrigation)

st.title(f'Operations accounts for {selected_gie}')
st.write(' Visualizes the credit request, authorization, and execution process for different operating accounts ( eg gasoil irrigation) in a timeline format. This allows you to easily track the progress of each operating account’s credit process over time.')
# Prepare the data for timeline visualization
timeline_data = []
for idx, row in filtered_data.iterrows():
    timeline_data.append(dict(Task=row['operating_account'], Start=row['credit_req_date'], Finish=row['credit_auth_date'], Stage='Credit Requested'))
    timeline_data.append(dict(Task=row['operating_account'], Start=row['credit_auth_date'], Finish=row['credit_exec_date'], Stage='Credit Authorized'))
    timeline_data.append(dict(Task=row['operating_account'], Start=row['credit_exec_date'], Finish=row['credit_exec_date'], Stage='Credit Executed'))

# Convert the timeline data into a DataFrame
timeline_df = pd.DataFrame(timeline_data)

# Custom color map for stages
custom_colors = {
    'Credit Requested': 'dodgerblue',
    'Credit Authorized': 'orange',
    'Credit Executed': 'green'
}

# Create a Gantt-style timeline using Plotly with customizations
fig_timeline = px.timeline(
    timeline_df,
    x_start="Start",
    x_end="Finish",
    y="Task",
    color="Stage",
    color_discrete_map=custom_colors,  # Use custom colors for each stage
    title=f"Credit Request, Authorization, and Execution Timeline for GIE: {selected_gie}",
    hover_data={
        'Start': '|%B %d, %Y',  # Customize hover date format
        'Finish': '|%B %d, %Y',  # Customize hover date format
        'Stage': True,
        'Task': True
    },
    labels={"Task": "Operation Account", "Stage": "Credit Stage"}
)

# Customizing the layout
fig_timeline.update_layout(
    xaxis_title="Date",
    yaxis_title="Operation Account",
    showlegend=True,
    height=600,
    margin=dict(l=100, r=50, t=50, b=50),  # Adjust margins for readability
    font=dict(family="Arial, sans-serif", size=12)  # Customize font
)

# Improve x-axis readability
fig_timeline.update_xaxes(
    tickformat="%b %d, %Y",  # Date format on the x-axis
    tickangle=45,  # Rotate x-axis labels
    nticks=20  # Control number of ticks
)

# Display the timeline
st.plotly_chart(fig_timeline)

# Assuming 'melted_data' is your DataFrame and it has a 'geometry' column
# Load GeoDataFrame (GIE credit and RS data with geometries)
st.dataframe(filtered_data)

#________________________________________________________PPPP___________________________________
# Prediction
st.header('4. Estimation of the Flooded area using growth Models')
#st.subheader('**Logistics groth model**')
# Title and description
st.title("Growth Curve: Logistic")

# Equation
st.markdown("### Equation:")
st.latex(r"P(t) = \frac{K}{1 + \left( \frac{K - P_0}{P_0} \right) e^{-rt}}")

# Explanation of the terms
st.markdown("""
Where:
- **P(t)**: Population or size at time **t**,
- **K**: Carrying capacity (the maximum size the population can reach),
- **P₀**: Initial population or size,
- **r**: Growth rate,
- **t**: Time.
""")

# Shape explanation
st.markdown("""
### Shape:
The logistic curve is S-shaped (sigmoidal). It starts with an exponential growth phase, 
slows down as the population approaches the carrying capacity, and eventually levels off at the carrying capacity.
""")

# Growth dynamics
st.markdown("""
### Growth Dynamics:
- **Initial Phase**: Rapid growth (exponential).
- **Middle Phase**: Growth rate decreases as resources become limited.
- **Final Phase**: Growth slows down and asymptotically approaches the carrying capacity, **K**.
""")

#plot the prediction and the data
def process_rs_data(dff):
    rs_df = dff.filter(regex=('\d{4}-?\d{2}-?\d{2}$'))  # Date columns
    area_rs = rs_df.sum(axis=0)  # Summing values row-wise
    rs_df_combined = pd.DataFrame()
    rs_df_combined['Time'] = list(area_rs.index)
    rs_df_combined['Area(ha)'] = list(area_rs.values)
    rs_df_combined['Year'] = rs_df_combined['Time'].str[:4]
    rs_df_combined['Class'] = 'RS_' + rs_df_combined['Year']
    rs_df_combined['Time'] = pd.to_datetime(rs_df_combined['Time'])
    rs_df_combined['DOY'] = rs_df_combined['Time'].apply(lambda x: x.timetuple().tm_yday)
    return rs_df_combined

    # Process both datasets
dag_rs_df = process_rs_data(read_github_csv('https://github.com/ICRISAT-Senegal/Remote-sensing/blob/main/flooding_data_2024.csv'))
file_log ='https://github.com/ICRISAT-Senegal/Remote-sensing/blob/main/prediction_growth_ts_first_attempt.csv'
data_log = read_github_csv(file_log)
data_log['Time'] = pd.to_datetime(data_log['time_t'])
data_log['Area(ha)'] = data_log['area_t']
#data_log['Class'] = 'log_2024'
#data_log['Time'] = pd.to_datetime(data_log['Time'])
data_log['DOY'] = data_log['Time'].apply(lambda x: x.timetuple().tm_yday)

fig, ax = plt.subplots(figsize=(10, 6))

combined_df_g = pd.concat([dag_rs_df,data_log],axis = 0)
# Plot each class in the 'Class' column
for cls in combined_df_g['Class'].unique():
    class_data = combined_df_g[combined_df_g['Class'] == cls]
    ax.plot(class_data['DOY'], class_data['Area(ha)'], marker='x', linestyle='--', label=cls)

# Customize plot
ax.set_title(f'Cumulative Flooded Areas {year}', fontsize=10)
ax.set_xlabel('Day of Year (DOY)')
ax.set_ylabel('Cumulative Flooded Area (ha)')
ax.legend(title='Classes')
ax.grid(True)
plt.show()
st.pyplot(fig)

