import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Import graph_objects
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import statsmodels.api as sm # For potential future use if adding model summaries

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Hospital Financial Dashboard",
    layout="wide", # Use wide layout for better use of space
    initial_sidebar_state="expanded" # Keep sidebar expanded by default
)

# --- Custom CSS for Aesthetics (Blue/White) ---
st.markdown("""
<style>
    /* Main page background */
    .stApp {
        background-color: #FFFFFF; /* White background */
    }
    /* Sidebar background */
    .stsidebar {
        background-color: #F0F2F6; /* Light grey/white background */
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0047AB; /* Cobalt Blue for headers */
    }
    /* Markdown text */
    .markdown-text-container {
        color: #333333; /* Dark grey for text */
    }
    /* Metric labels */
    [data-testid="stMetricLabel"] > div {
        color: #0047AB; /* Cobalt Blue for metric labels */
    }
    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #1E90FF; /* Dodger Blue for metric values */
    }
    /* Info boxes */
    div[data-testid="stNotification"] div[data-testid="stMarkdownContainer"] {
        color: #0047AB !important; /* Blue for info box text */
    }
    /* Warnings */
     div[data-testid="stNotification"] div[data-testid="stMarkdownContainer"] {
        color: #FF4B4B !important; /* Keep default red for warnings */
     }
    /* Success messages */
    div[data-testid="stNotification"] div[data-testid="stMarkdownContainer"] {
        color: #008000 !important; /* Keep default green for success */
    }
    /* Error messages */
    div[data-testid="stNotification"] div[data-testid="stMarkdownContainer"] {
        color: #FF0000 !important; /* Keep default red for errors */
     }
</style>
""", unsafe_allow_html=True)


# --- Configuration ---
# Define the path to the final output file from your combined pipeline
DATA_FILE_PATH = r"C:\Users\yadla\OneDrive\Desktop\Hospital_Analysis_Project\Data\Hospital_Financial_Analysis_Subset_Integrated_Imputed.csv"

# --- Data Loading ---
# Use Streamlit's caching mechanism to load data only once
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """
    Loads the processed hospital financial data from the specified path.
    Includes basic type conversions for key columns.

    Args:
        path (str): The full path to the CSV data file.

    Returns:
        pd.DataFrame: The loaded DataFrame, or an empty DataFrame if loading fails.
    """
    try:
        df = pd.read_csv(path)

        # --- Basic Data Cleaning/Type Conversion after Loading ---
        # Ensure 'Year' is integer and handle potential NaNs
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64') # Use Int64 for nullable integer
        else:
            st.warning("Column 'Year' not found in data.")

        # Ensure 'State Code', 'County', and 'Rural Versus Urban' are strings for filtering/grouping
        if 'State Code' in df.columns:
            df['State Code'] = df['State Code'].astype(str).str.strip()
        else:
             st.warning("Column 'State Code' not found in data.")

        if 'County' in df.columns:
             df['County'] = df['County'].astype(str).str.strip()
        else:
             st.warning("Column 'County' not found in data.")


        if 'Rural Versus Urban' in df.columns:
            df['Rural Versus Urban'] = df['Rural Versus Urban'].astype(str).str.strip()
        else:
             st.warning("Column 'Rural Versus Urban' not found in data.")

        # Ensure facility_id is present and string type for grouping/plotting
        if 'facility_id' in df.columns:
             df['facility_id'] = df['facility_id'].astype(str).str.strip()
        else:
             st.warning("Column 'facility_id' not found in data.")

        # Ensure 'Hospital Name' is present and string type for grouping/plotting
        if 'Hospital Name' in df.columns:
             df['Hospital Name'] = df['Hospital Name'].astype(str).str.strip()
        else:
             st.warning("Column 'Hospital Name' not found in data.")


        # Convert specific key financial columns to numeric, coercing errors
        # Add other key columns you plan to use frequently here
        financial_cols_to_numeric = [
            'Net Income',
            'Cost To Charge Ratio',
            'Total Revenue',
            'Net Patient Revenue',
            "Contractual Allowance and Discounts on Patients' Accounts", # Using the renamed column
            'Total Patient Revenue', # Added for line plot
            'Less Total Operating Expense', # Added for potential future use
            'Total_Staffing_FTE', # Added for potential future use
            'Annual', # Added for potential future use
            'Total_Yearly_Medicare_Enrollment', # Corrected column name for Medicare enrollment
            'Number of Beds', # Added for potential use in correlation
            'Total Assets', # Added for potential use in correlation
            'Total Liabilities', # Added for potential use in correlation
            'Total Salaries From Worksheet A', # Added for potential use in correlation
            'Depreciation Cost', # Added for potential use in correlation
            'Total Bad Debt Expense', # Added for potential use in correlation
            'Cost of Charity Care', # Added for potential use in correlation
            'Inpatient Revenue', # Added for YoY analysis
            'Outpatient Revenue' # Added for YoY analysis

        ]
        for col in financial_cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                 st.warning(f"Financial column '{col}' not found in data.")

        # Convert all other numeric-like columns to numeric, coercing errors (fallback)
        for col in df.columns:
            if col not in financial_cols_to_numeric and col not in ['Year', 'State Code', 'Rural Versus Urban', 'facility_id', 'County', 'Hospital Name']: # Exclude already handled columns
                 if df[col].dtype != 'object' or (df[col].dtype == 'object' and pd.to_numeric(df[col], errors='coerce').notna().sum() / len(df) > 0.8):
                     df[col] = pd.to_numeric(df[col], errors='coerce')


        return df

    except FileNotFoundError:
        st.error(f"Error: Data file not found at {path}. Please check the file path and ensure the pipeline has been run successfully.")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        st.error(f"An unexpected error occurred while loading or initially processing data from {path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# Load the data into the main DataFrame
df_hospital = load_data(DATA_FILE_PATH)

# --- Check if data loaded successfully ---
if df_hospital.empty:
    st.error("Dashboard cannot load due to data loading errors. Please check the file path and pipeline output.")
    st.stop() # Stop the app execution if data is not loaded

# --- Dashboard Title and Introduction ---
st.title("Hospital Financial Performance Dashboard (2018-2022)")
st.markdown("Explore key financial metrics and trends for hospitals based on processed data.")

# --- Dashboard Controls and Filters (Sidebar) ---
st.sidebar.header("Global Filters")
st.sidebar.markdown("These filters apply to Summary Metrics, Geographic Overview, and Detailed Analysis tabs.")

# --- Apply Global Filters (Reordered) ---
# Start with a copy of the full data
filtered_df = df_hospital.copy()

# State Filter (Apply first)
# Ensure 'State Code' column exists before creating the filter
if 'State Code' in filtered_df.columns: # Use filtered_df here as it's the base for this section
    # Get unique states, drop NaNs, sort, and add 'All States' option
    state_options = ['All States'] + sorted(filtered_df['State Code'].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox(
        "Select State",
        options=state_options,
        help="Filter data by selecting a specific state or view all states."
    )
    # Apply State filter immediately
    if selected_state != 'All States':
        filtered_df = filtered_df[filtered_df['State Code'].astype(str) == selected_state]
else:
    st.sidebar.warning("'State Code' column not found in data. State filter disabled.")
    selected_state = 'All States' # Default to All States if column is missing


# County Filter (Determine options based on the state-filtered data)
selected_county_option = None # Initialize as None
if 'County' in filtered_df.columns: # Use filtered_df here to get counties within the selected state
    county_options = sorted(filtered_df['County'].dropna().unique().tolist())

    # Add an option to select all counties in the list
    county_options_with_all = ['All Counties'] + county_options

    selected_county_option = st.sidebar.selectbox(
        "Select County",
        options=county_options_with_all,
        help="Filter data by selecting a single county within the selected state."
    )
    # Apply County filter
    if selected_county_option != 'All Counties' and selected_county_option is not None:
         # Ensure the column used for filtering is string type and compare with selected county
         filtered_df = filtered_df[filtered_df['County'].astype(str) == selected_county_option]

else:
    st.sidebar.warning("'County' column not found in data. County filter disabled.")


# Year Multi-select Filter (Apply last)
# Ensure 'Year' column exists and is numeric before creating the multiselect
if 'Year' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['Year']):
    valid_years = sorted(filtered_df['Year'].dropna().unique().tolist()) # Get unique valid years from the already filtered data
    if valid_years:
        selected_years = st.sidebar.multiselect(
            "Select Year(s)",
            options=valid_years,
            default=valid_years, # Default to selecting all available years in the filtered data
            help="Select one or more years to include in the analysis."
        )
        # Convert selected years to a tuple for consistency in displaying range
        year_range = (min(selected_years), max(selected_years)) if selected_years else (None, None)

        # Apply Year filter
        filtered_df['Year'] = pd.to_numeric(filtered_df['Year'], errors='coerce')
        filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]

    else:
        st.sidebar.warning("'Year' column has no valid numeric data in the filtered subset. Year filter disabled.")
        selected_years = []
        year_range = (None, None)
        # If no valid years, the filtered_df should be empty for year-based analysis
        filtered_df = filtered_df[filtered_df['Year'].isin([])] # Filter to empty set

else:
    st.sidebar.warning("'Year' column not found or not numeric in data. Year filter disabled.")
    selected_years = []
    year_range = (None, None)
    # If Year column is missing, ensure filtered_df is handled appropriately if subsequent steps rely on Year
    # For now, let's assume subsequent steps check for 'Year' column existence.


# =============================================================================
# --- Dashboard Sections (Rearranged) ---
# Use filtered_df for all analysis and visualizations
# =============================================================================

# --- Section 1: Summary Metrics ---
st.header("Summary Metrics")
st.markdown(f"Overview of key metrics for **{selected_state}** ({selected_county_option if selected_county_option is not None else 'All Counties'}) from **{year_range[0] if year_range[0] is not None else 'N/A'}** to **{year_range[1] if year_range[1] is not None else 'N/A'}**.")

if not filtered_df.empty:
    # --- 1) Total Number of Hospitals ---
    if 'facility_id' in filtered_df.columns:
        num_hospitals = filtered_df['facility_id'].nunique()
        st.metric(label="Total Number of Hospitals", value=num_hospitals)
    else:
        st.info("Data not available to count hospitals.")

    # --- 2) Metrics Like: Net Income, Cost-Charge-Ratio ---
    # Calculate average Net Income and Cost To Charge Ratio for the filtered data
    # Ensure columns are numeric before calculating mean and handle potential NaNs
    avg_net_income = filtered_df['Net Income'].mean() if 'Net Income' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['Net Income']) else np.nan
    avg_cost_charge_ratio = filtered_df['Cost To Charge Ratio'].mean() if 'Cost To Charge Ratio' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['Cost To Charge Ratio']) else np.nan

    col1, col2 = st.columns(2) # Use columns to display metrics side-by-side

    with col1:
        if not np.isnan(avg_net_income): # Check if not NaN
            st.metric(label="Average Net Income", value=f"${avg_net_income:,.2f}")
        else:
            st.info("Net Income data not available for average calculation.")

    with col2:
        if not np.isnan(avg_cost_charge_ratio): # Check if not NaN
            st.metric(label="Average Cost To Charge Ratio", value=f"{avg_cost_charge_ratio:.4f}")
        else:
            st.info("Cost To Charge Ratio data not available for average calculation.")

else:
    st.info("Filtered data is empty, cannot display summary metrics.")

st.markdown("---") # Add a separator


# --- Section 2: Geographic Overview ---
st.header("Geographic Overview")
st.markdown(f"Visualize key metrics across states and counties for the filtered data ({selected_state}, {selected_county_option if selected_county_option is not None else 'All Counties'}) from **{year_range[0] if year_range[0] is not None else 'N/A'}** to **{year_range[1] if year_range[1] is not None else 'N/A'}**.")

# Metric Selection for Geographic Overview Map and Tables
# Identify potential numeric columns for the map color and tables
geographic_metric_options = [
    'Net Income',
    'Total Revenue',
    'Cost To Charge Ratio',
    'Total_Staffing_FTE',
    'Number of Beds',
    'Total_Yearly_Medicare_Enrollment' # Added Medicare enrollment
]
# Filter to include only metrics present and numeric in the filtered data
present_numeric_geographic_metrics = [
    metric for metric in filtered_df.columns
    if metric in geographic_metric_options and pd.api.types.is_numeric_dtype(filtered_df[metric])
]

selected_geographic_metric = None
if present_numeric_geographic_metrics:
    selected_geographic_metric = st.selectbox(
        "Select Metric for Geographic Analysis",
        options=present_numeric_geographic_metrics,
        help="Choose the financial metric to display on the map and in the tables."
    )
else:
    st.warning("No suitable numeric metrics available for geographic visualization and tables.")


# Ensure required columns for map and tables are present and data is not empty, and a metric is selected
if not filtered_df.empty and 'State Code' in filtered_df.columns and selected_geographic_metric and selected_geographic_metric in filtered_df.columns:
    # Aggregate data by State Code using the selected metric
    # Ensure selected_geographic_metric is numeric before aggregating
    if pd.api.types.is_numeric_dtype(filtered_df[selected_geographic_metric]) and filtered_df[selected_geographic_metric].notna().any():
        state_agg_data = filtered_df.groupby('State Code')[selected_geographic_metric].mean().reset_index()
        state_agg_data = state_agg_data.dropna(subset=[selected_geographic_metric]) # Drop states with no data for the selected metric

        if not state_agg_data.empty:
            try:
                # Create the Chloropleth map
                fig_map = px.choropleth(
                    state_agg_data,
                    locations='State Code', # Column with state abbreviations
                    locationmode="USA-states", # Set to plot US states
                    color=selected_geographic_metric, # Column determining the color intensity (the selected metric)
                    hover_name='State Code', # Display state code on hover
                    hover_data={selected_geographic_metric: ':.2f'}, # Format the selected metric in hover
                    scope="usa", # Focus the map on the USA
                    color_continuous_scale="Blues", # Changed color scale to a blue gradient
                    title=f"Average {selected_geographic_metric} by State ({year_range[0] if year_range[0] is not None else 'N/A'}–{year_range[1] if year_range[1] is not None else 'N/A'})"
                )
                fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}) # Adjust layout
                st.plotly_chart(fig_map, use_container_width=True)

                # --- Display State-level data in a table ---
                st.subheader(f"Average {selected_geographic_metric} by State")
                st.dataframe(state_agg_data.style.format({selected_geographic_metric: '{:,.2f}'})) # Format for readability


            except Exception as e:
                st.error(f"An error occurred while generating the Chloropleth map or State table: {e}")
                st.write("Please check the data and column names for the geographic visualization.")

        else:
            st.info(f"Aggregated state data is empty for map and table visualization with metric '{selected_geographic_metric}'. This might be because the selected filters resulted in no data for this metric.") # More specific message

    elif not pd.api.types.is_numeric_dtype(filtered_df[selected_geographic_metric]):
         st.warning(f"Selected metric '{selected_geographic_metric}' is not numeric in the filtered data. Cannot generate geographic analysis.")
    elif not filtered_df[selected_geographic_metric].notna().any():
         st.info(f"Selected metric '{selected_geographic_metric}' has no valid data for the current filters. Cannot generate geographic analysis.")


    # --- Display County-level data in a table (if a specific state is selected) ---
    if selected_state != 'All States' and 'County' in filtered_df.columns and selected_geographic_metric and selected_geographic_metric in filtered_df.columns:
        # Aggregate data by County within the selected state
        if pd.api.types.is_numeric_dtype(filtered_df[selected_geographic_metric]) and filtered_df[selected_geographic_metric].notna().any():
            county_agg_data = filtered_df.groupby('County')[selected_geographic_metric].mean().reset_index()
            county_agg_data = county_agg_data.dropna(subset=[selected_geographic_metric]) # Drop counties with no data

            if not county_agg_data.empty:
                st.subheader(f"Average {selected_geographic_metric} by County in {selected_state}")
                st.dataframe(county_agg_data.style.format({selected_geographic_metric: '{:,.2f}'})) # Format for readability
            else:
                 st.info(f"Aggregated county data is empty for metric '{selected_geographic_metric}' in {selected_state}. This might be because the selected filters resulted in no data for this metric.")

        elif not pd.api.types.is_numeric_dtype(filtered_df[selected_geographic_metric]):
             st.warning(f"Selected metric '{selected_geographic_metric}' is not numeric in the filtered data. Cannot generate county table.")
        elif not filtered_df[selected_geographic_metric].notna().any():
             st.info(f"Selected metric '{selected_geographic_metric}' has no valid data for the current filters. Cannot generate county table.")


elif not filtered_df.empty:
    missing_geographic_cols = [col for col in ['State Code'] if col not in filtered_df.columns]
    if selected_geographic_metric is None:
         st.warning("No suitable numeric metrics available for geographic analysis.")
    elif selected_geographic_metric not in filtered_df.columns:
         st.warning(f"Selected metric '{selected_geographic_metric}' not found in the filtered data.")
    elif missing_geographic_cols:
         st.warning(f"Required columns for geographic analysis ({missing_geographic_cols}) are missing in the filtered data.")
    else:
         st.warning("Filtered data is empty or required columns/metrics are missing for geographic analysis.")

else:
    st.info("Filtered data is empty, cannot generate Geographic Overview.")


st.markdown("---") # Add a separator


# --- Remaining Sections (using Tabs) ---
st.header("Detailed Analysis")
st.markdown("Explore specific trends and hospital rankings in the tabs below.")

# Added a new tab 'Metric Trend Drilldown'
tab1_detail, tab2_detail, tab3_compare, tab4_drilldown = st.tabs([
    "Key Financial Trends",
    "Top & Bottom Hospitals",
    "Comparison Analysis",
    "Metric Trend Drilldown" # New tab title
])

# --- Tab 1 (Detail): Key Financial Trends (Original Content) ---
with tab1_detail:
    st.header("Key Financial Metric Trends Over Time (Overview)")
    st.markdown(f"Average trends for selected metrics across the filtered data for **{selected_state}** ({selected_county_option if selected_county_option is not None else 'All Counties'}) from **{year_range[0] if year_range[0] is not None else 'N/A'}** to **{year_range[1] if year_range[1] is not None else 'N/A'}**.")

    metrics_to_plot_time = [
        'Net Patient Revenue',
        'Total Revenue',
        "Contractual Allowance and Discounts on Patients' Accounts",
        'Cost To Charge Ratio'
    ]

    # Filter to include only metrics present and numeric in the filtered data
    present_numeric_metrics = [
        metric for metric in filtered_df.columns
        if metric in metrics_to_plot_time and pd.api.types.is_numeric_dtype(filtered_df[metric])
    ]

    if not filtered_df.empty and 'Year' in filtered_df.columns and present_numeric_metrics:
        # Aggregate data by Year for plotting
        try:
            # Ensure Year is numeric before grouping
            filtered_df['Year'] = pd.to_numeric(filtered_df['Year'], errors='coerce')
            time_series_data = filtered_df.groupby('Year')[present_numeric_metrics].mean().reset_index() # Using mean for aggregation

            # Melt the DataFrame for Plotly Express
            time_series_melted = time_series_data.melt(
                id_vars='Year',
                value_vars=present_numeric_metrics,
                var_name='Metric',
                value_name='Value'
            )

            if not time_series_melted.empty:
                fig_time_series = px.line(
                    time_series_melted,
                    x='Year',
                    y='Value',
                    color='Metric',
                    title=f"Average Metric Values Over Time ({selected_state}, {selected_county_option if selected_county_option is not None else 'All Counties'}) from {year_range[0] if year_range[0] is not None else 'N/A'}–{year_range[1] if year_range[1] is not None else 'N/A'}",
                    markers=True,
                    labels={'Year': 'Year', 'Value': 'Average Value'},
                    color_discrete_sequence=px.colors.qualitative.Plotly # Use a Plotly color sequence (often includes blues)
                )
                fig_time_series.update_layout(xaxis=dict(dtick=1)) # Ensure integer ticks for years
                st.plotly_chart(fig_time_series, use_container_width=True)
            else:
                st.info("Aggregated data is empty for time series plot.")

        except Exception as e:
            st.error(f"An error occurred while generating the time series plot: {e}")
            st.write("Please check the data and column names for the time series analysis.")

    elif not present_numeric_metrics:
        st.warning(f"None of the required time series metrics ({metrics_to_plot_time}) are present or numeric in the filtered data.")
    else:
        st.info("Filtered data is empty, cannot generate time series plot.")


# --- Tab 2 (Detail): Top & Bottom Hospitals ---
with tab2_detail:
    st.header("Top & Bottom Performing Hospitals")
    st.markdown(f"Identify hospitals with the highest and lowest performance based on selected criteria for **{selected_state}** ({selected_county_option if selected_county_option is not None else 'All Counties'}) from **{year_range[0] if year_range[0] is not None else 'N/A'}** to **{year_range[1] if year_range[1] is not None else 'N/A'}**.")

    if not filtered_df.empty and 'Hospital Name' in filtered_df.columns and 'Net Income' in filtered_df.columns and 'Year' in filtered_df.columns and 'facility_id' in filtered_df.columns:

        # Controls for Top/Bottom Hospitals
        col_perf_metric, col_timeframe = st.columns(2)

        with col_perf_metric:
            performance_metric_option = st.radio(
                "Select Performance Metric:",
                ('Performance by numbers (Avg Net Income)', 'Performance by % change (Avg YoY % Change Net Income)')
            )

        with col_timeframe:
            timeframe_option = st.radio(
                "Select Time Frame:",
                ('Last 1 Year', 'Last 3 Years', 'Last 5 Years') # Corrected options to reflect available data
            )

        # Determine the number of years for filtering based on the radio button
        num_years_for_ranking = int(timeframe_option.split(' ')[1])

        # Get the most recent years available in the filtered data
        if 'Year' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['Year']):
            all_years_filtered = sorted(filtered_df['Year'].dropna().unique())
            if len(all_years_filtered) < num_years_for_ranking:
                st.warning(f"Not enough years ({len(all_years_filtered)}) in filtered data for analysis over {timeframe_option}. Using all available years.")
                recent_years_for_ranking = all_years_filtered # Use all available years
            else:
                recent_years_for_ranking = all_years_filtered[-num_years_for_ranking:]

            if recent_years_for_ranking:
                ranking_data = filtered_df[filtered_df['Year'].isin(recent_years_for_ranking)].copy()

                if not ranking_data.empty:
                    # Ensure Net Income is numeric for calculations
                    if 'Net Income' in ranking_data.columns:
                        ranking_data['Net Income'] = pd.to_numeric(ranking_data['Net Income'], errors='coerce')

                        # Calculate performance based on selected metric
                        if performance_metric_option == 'Performance by numbers (Avg Net Income)':
                            # Calculate average Net Income per hospital over the selected period
                            hospital_ranking_summary = ranking_data.groupby('Hospital Name')['Net Income'].mean().reset_index()
                            hospital_ranking_summary = hospital_ranking_summary.dropna(subset=['Net Income']) # Drop hospitals with no data
                            sort_col = 'Net Income'
                            title_suffix = "by Average Net Income"

                        elif performance_metric_option == 'Performance by % change (Avg YoY % Change Net Income)':
                            # Calculate YoY % Change in Net Income
                            # Need data for at least 2 years for YoY change
                            if len(recent_years_for_ranking) >= 2 and 'facility_id' in ranking_data.columns: # Added facility_id check
                                # Sort by facility and year for correct YoY calculation
                                ranking_data_sorted = ranking_data.sort_values(['facility_id', 'Year'])
                                ranking_data_sorted['Net Income YoY % Change'] = ranking_data_sorted.groupby('facility_id')['Net Income'].pct_change() * 100

                                # Calculate average YoY % Change per hospital over the selected period
                                # Exclude the first year of the period as YoY is NaN
                                ranking_data_yoy = ranking_data_sorted[ranking_data_sorted['Year'] != recent_years_for_ranking[0]].copy()
                                hospital_ranking_summary = ranking_data_yoy.groupby('Hospital Name')['Net Income YoY % Change'].mean().reset_index()
                                hospital_ranking_summary = hospital_ranking_summary.dropna(subset=['Net Income YoY % Change']) # Drop hospitals with no valid YoY data
                                sort_col = 'Net Income YoY % Change'
                                title_suffix = "by Average YoY % Change in Net Income"
                            else:
                                st.warning(f"Cannot calculate YoY % Change for {timeframe_option} as less than 2 years are available or 'facility_id' is missing in the filtered data.")
                                hospital_ranking_summary = pd.DataFrame() # Empty DataFrame if cannot calculate YoY


                        if not hospital_ranking_summary.empty and sort_col in hospital_ranking_summary.columns:
                            # Get Top 10 and Bottom 10 hospitals
                            top_10_hospitals = hospital_ranking_summary.sort_values(by=sort_col, ascending=False).head(10)
                            bottom_10_hospitals = hospital_ranking_summary.sort_values(by=sort_col, ascending=True).head(10)

                            # --- Display Top and Bottom side-by-side ---
                            col_top, col_bottom = st.columns(2)

                            with col_top:
                                st.subheader(f"Top 10 Hospitals")
                                # Format the display column
                                if sort_col == 'Net Income':
                                    top_10_hospitals['Average Net Income ($)'] = top_10_hospitals['Net Income'].apply(lambda x: f"${x:,.2f}")
                                    st.dataframe(top_10_hospitals[['Hospital Name', 'Average Net Income ($)']])
                                elif sort_col == 'Net Income YoY % Change':
                                     top_10_hospitals['Average YoY % Change (%)'] = top_10_hospitals['Net Income YoY % Change'].apply(lambda x: f"{x:.2f}%")
                                     st.dataframe(top_10_hospitals[['Hospital Name', 'Average YoY % Change (%)']])
                                else:
                                     # Fallback display if sort_col is unexpected
                                     st.dataframe(top_10_hospitals)


                            with col_bottom:
                                st.subheader(f"Bottom 10 Hospitals")
                                # Format the display column
                                if sort_col == 'Net Income':
                                    bottom_10_hospitals['Average Net Income ($)'] = bottom_10_hospitals['Net Income'].apply(lambda x: f"${x:,.2f}")
                                    st.dataframe(bottom_10_hospitals[['Hospital Name', 'Average Net Income ($)']])
                                elif sort_col == 'Net Income YoY % Change':
                                     bottom_10_hospitals['Average YoY % Change (%)'] = bottom_10_hospitals['Net Income YoY % Change'].apply(lambda x: f"{x:.2f}%")
                                     st.dataframe(bottom_10_hospitals[['Hospital Name', 'Average YoY % Change (%)']])
                                else:
                                     # Fallback display if sort_col is unexpected
                                     st.dataframe(bottom_10_hospitals)


                        elif not hospital_ranking_summary.empty:
                             st.warning(f"Sorting column '{sort_col}' not found after calculating ranking summary.")
                        else:
                             st.info(f"No complete data available in the last {num_years_for_ranking} years for the selected filters to rank hospitals.")

                    else:
                        st.warning("Net Income data not available or not numeric in the filtered data for ranking.")
                else:
                    st.info(f"No data available in the last {num_years_for_ranking} years for the selected filters.")
            else:
                st.info(f"No years available in filtered data to analyze the last {num_years_for_ranking} years.")

    else:
        st.info("Filtered data is empty or required columns (Hospital Name, Net Income, Year, facility_id) are missing for Top/Bottom Hospitals analysis.")


# --- Tab 3 (Detail): Comparison Analysis ---
with tab3_compare:
    st.header("Comparison Analysis")
    st.markdown("Compare selected financial metrics across different entities (Hospitals, Counties, or States).")

    # --- Flexible Entity/Metric Comparison (using multi-selects) ---
    st.subheader("Flexible Entity and Metric Comparison")
    st.markdown("Select entities and metrics below to compare their trends or aggregated values.")

    # Select Comparison Level - Single radio button with horizontal layout
    comparison_level = st.radio(
        "Select Comparison Level:",
        ('Hospital', 'County', 'State'),
        horizontal=True, # Display horizontally
        key='comp_level_flex', # Added a unique key
        help="Choose the level of granularity for comparison."
    )

    # Dynamic Entity Selection based on Comparison Level
    selected_entities_compare = [] # Initialize list of entities to compare
    # Use the full data for entity selection options, regardless of global filters
    comparison_data_tab_flex_options = df_hospital.copy()
    group_by_col_flex = None # Initialize group_by_col
    entity_type_flex = None # Initialize entity_type

    if comparison_level == 'Hospital':
        # Get unique hospital names from the full data
        hospital_options_compare = sorted(comparison_data_tab_flex_options['Hospital Name'].dropna().unique().tolist())
        selected_entities_compare = st.multiselect(
            "Select Hospitals to Compare",
            options=hospital_options_compare,
            key='select_hospitals_flex', # Added a unique key
            help="Choose two or more hospitals to compare their metrics."
        )
        group_by_col_flex = 'Hospital Name'
        entity_type_flex = 'Hospitals'

    elif comparison_level == 'County':
        # Get unique county names from the full data
        county_options_compare = sorted(comparison_data_tab_flex_options['County'].dropna().unique().tolist())
        selected_entities_compare = st.multiselect(
            "Select Counties to Compare",
            options=county_options_compare,
             key='select_counties_flex', # Added a unique key
            help="Choose two or more counties to compare their metrics."
        )
        group_by_col_flex = 'County'
        entity_type_flex = 'Counties'

    elif comparison_level == 'State':
        # Get unique state names from the full data
        state_options_compare = sorted(comparison_data_tab_flex_options['State Code'].dropna().unique().tolist())
        selected_entities_compare = st.multiselect(
            "Select States to Compare",
            options=state_options_compare,
             key='select_states_flex', # Added a unique key
            help="Choose two or more states to compare their metrics."
        )
        group_by_col_flex = 'State Code'
        entity_type_flex = 'States'


    # Metric Multi-select for Comparison (Flexible)
    # Identify potential numeric columns for comparison (excluding identifiers and grouping columns)
    numeric_cols_compare_flex = df_hospital.select_dtypes(include=np.number).columns.tolist() # Use full data for metric options
    exclude_cols_compare_flex = ['Year', 'facility_id'] # Always exclude these
    # Exclude the grouping column if it's numeric (e.g., Year is numeric but used for grouping)
    if group_by_col_flex and group_by_col_flex in numeric_cols_compare_flex: # Check if group_by_col is set and numeric
         selectable_metrics_compare_flex = [col for col in numeric_cols_compare_flex if col not in exclude_cols_compare_flex and col != group_by_col_flex]
    else:
         selectable_metrics_compare_flex = [col for col in numeric_cols_compare_flex if col not in exclude_cols_compare_flex]


    selected_metrics_compare_flex = st.multiselect(
        "Select Metrics to Compare",
        options=selectable_metrics_compare_flex,
        key='select_metrics_flex', # Added a unique key
        help="Choose one or more financial metrics to compare across selected entities."
    )

    # Add a radio button for chart type selection (Flexible)
    chart_type_compare_flex = st.radio(
        "Select Chart Type:",
        ('Line Chart', 'Bar Chart'),
        horizontal=True,
        key='chart_type_flex', # Added a unique key
        help="Choose the type of chart for visualization."
    )

    # Add Year filter specific to this flexible comparison sub-section
    # Use the full data for year options
    if 'Year' in df_hospital.columns and pd.api.types.is_numeric_dtype(df_hospital['Year']):
        valid_years_comp_flex_options = sorted(df_hospital['Year'].dropna().unique().tolist())
        if valid_years_comp_flex_options:
            selected_years_comp_flex = st.multiselect(
                f"Select Year(s) for {comparison_level} Comparison",
                options=valid_years_comp_flex_options,
                default=valid_years_comp_flex_options, # Default to all years available in full data
                key='select_years_flex', # Added a unique key
                help="Select one or more years to include in this comparison."
            )
        else:
             st.info("No valid years available for flexible comparison.")
             selected_years_comp_flex = [] # Ensure it's an empty list if no years
    else:
         st.warning("'Year' column not found or not numeric in data. Year filter disabled for flexible comparison.")
         selected_years_comp_flex = [] # Ensure it's an empty list if no years


    # Perform and display flexible comparison
    # Only proceed if entities and metrics are selected, and group_by_col is determined
    if selected_entities_compare and selected_metrics_compare_flex and group_by_col_flex and selected_years_comp_flex:
        # Filter data for selected entities and years based on the chosen level
        if group_by_col_flex in df_hospital.columns and 'Year' in df_hospital.columns:
            comparison_subset_data_flex = df_hospital[
                df_hospital[group_by_col_flex].isin(selected_entities_compare) &
                df_hospital['Year'].isin(selected_years_comp_flex)
            ].copy()

            if not comparison_subset_data_flex.empty:
                # Ensure selected metrics are numeric
                for metric in selected_metrics_compare_flex:
                    if metric in comparison_subset_data_flex.columns:
                        comparison_subset_data_flex[metric] = pd.to_numeric(comparison_subset_data_flex[metric], errors='coerce')

                st.subheader(f"Comparison Table ({entity_type_flex})")
                st.markdown(f"Average values of selected metrics for each chosen {comparison_level} over the selected period.")

                # Aggregate data by the selected comparison level and Year for the table
                # Use mean for aggregation
                aggregation_dict_flex = {metric: 'mean' for metric in selected_metrics_compare_flex}

                if aggregation_dict_flex:
                    comparison_table_data_flex = comparison_subset_data_flex.groupby([group_by_col_flex, 'Year'])[list(aggregation_dict_flex.keys())].agg(aggregation_dict_flex).reset_index()

                    if not comparison_table_data_flex.empty:
                        st.dataframe(comparison_table_data_flex)

                    else:
                        st.info(f"No data available for selected {entity_type_flex} and metrics in the selected comparison years for the table after aggregation.")
                else:
                     st.info("No metrics selected for table aggregation.")


                st.subheader(f"Comparison Visualization ({entity_type_flex})")
                st.markdown(f"Visualizing selected metrics over time for chosen {comparison_level}s.")

                # --- Generate plots based on chart type ---
                if not comparison_subset_data_flex.empty:
                     if chart_type_compare_flex == 'Line Chart':
                         # --- Generate a single Plotly Graph Objects figure for Line Chart ---
                         fig_go_flex = go.Figure()
                         has_cost_to_charge_ratio = 'Cost To Charge Ratio' in selected_metrics_compare_flex

                         # Iterate through each selected entity and each selected metric to add traces
                         for entity in selected_entities_compare:
                              entity_data = comparison_subset_data_flex[comparison_subset_data_flex[group_by_col_flex] == entity].copy()

                              if not entity_data.empty:
                                   # Aggregate by Year for the current entity
                                   # Use mean for aggregation
                                   entity_yearly_agg = entity_data.groupby('Year')[selected_metrics_compare_flex].mean().reset_index()

                                   if not entity_yearly_agg.empty:
                                       for metric in selected_metrics_compare_flex:
                                            if metric in entity_yearly_agg.columns and pd.api.types.is_numeric_dtype(entity_yearly_agg[metric]):
                                                # Determine y-axis based on metric
                                                y_axis = 'y' if metric != 'Cost To Charge Ratio' else 'y2'

                                                # Determine hover format
                                                hover_format = '.2f' # Default
                                                if 'Income' in metric or 'Revenue' in metric or 'Expense' in metric or 'Assets' in metric or 'Liabilities' in metric:
                                                    hover_format = '$,,.2f' # Currency with commas, two decimal places
                                                elif 'Ratio' in metric:
                                                     hover_format = '.4f' # Four decimal places for ratios
                                                elif 'Percent' in metric or '%' in metric:
                                                     hover_format = '.2f%' # Percentage with two decimal places


                                                fig_go_flex.add_trace(go.Scatter(
                                                    x=entity_yearly_agg['Year'],
                                                    y=entity_yearly_agg[metric],
                                                    mode='lines+markers',
                                                    name=f"{entity} - {metric}", # Name includes entity and metric
                                                    yaxis=y_axis,
                                                    hovertemplate=f"<b>{entity}</b><br>Year: %{{x}}<br>{metric}: %{{y:{hover_format}}}<extra></extra>" # Enhanced hover
                                                ))
                                            elif metric not in entity_yearly_agg.columns:
                                                 st.warning(f"Metric '{metric}' not found in aggregated data for {entity} for plotting.")
                                            elif not pd.api.types.is_numeric_dtype(entity_yearly_agg[metric]):
                                                 st.warning(f"Metric '{metric}' for {entity} is not numeric. Cannot plot.")
                                   else:
                                        st.info(f"No aggregated data available for {entity} for plotting.")
                              else:
                                   st.info(f"No data available for entity '{entity}' in the selected years for plotting.")


                         # Configure layout for the single Plotly GO figure
                         fig_go_flex.update_layout(
                             title=f"Trend of Selected Metrics Over Time for Selected {entity_type_flex}",
                             xaxis=dict(title="Year", dtick=1),
                             yaxis=dict(title="Amount ($)", side="left", tickformat='$,.0f'), # Primary Y-axis for dollar amounts
                             # Configure secondary y-axis only if Cost To Charge Ratio is selected
                             yaxis2=dict(title="Cost to Charge Ratio", overlaying="y", side="right", showgrid=False, tickformat='.2f') if has_cost_to_charge_ratio else None,
                             hovermode='closest',
                             template='plotly_white',
                             height=600,
                             margin=dict(t=50, r=50, l=50, b=50)
                         )

                         # Display the single Plotly GO figure
                         st.plotly_chart(fig_go_flex, use_container_width=True)


                     elif chart_type_compare_flex == 'Bar Chart':
                          # --- Generate separate Plotly Express bar charts for Bar Chart ---
                          for metric in selected_metrics_compare_flex:
                              if metric in comparison_subset_data_flex.columns and pd.api.types.is_numeric_dtype(comparison_subset_data_flex[metric]):
                                  # Create a temporary DataFrame for the current metric and selected entities/years
                                  plot_data_flex = comparison_subset_data_flex[[group_by_col_flex, 'Year', metric]].dropna().copy()

                                  if not plot_data_flex.empty:
                                      try:
                                          # For bar chart, aggregate by entity and take the mean over the selected years
                                          bar_data_flex = plot_data_flex.groupby(group_by_col_flex)[metric].mean().reset_index()

                                          # Determine y-axis format for bar chart
                                          y_format = None
                                          if 'Income' in metric or 'Revenue' in metric or 'Expense' in metric or 'Assets' in metric or 'Liabilities' in metric:
                                              y_format = '$,.0f' # Currency with commas, no decimal places
                                          elif 'Ratio' in metric:
                                               y_format = '.2f' # Two decimal places for ratios
                                          elif 'Percent' in metric or '%' in metric:
                                               y_format = '.2f%' # Percentage with two decimal places

                                          # Determine hover format for bar chart
                                          hover_format = None
                                          if 'Income' in metric or 'Revenue' in metric or 'Expense' in metric or 'Assets' in metric or 'Liabilities' in metric:
                                              hover_format = '$,,.2f' # Currency with commas, two decimal places for hover
                                          elif 'Ratio' in metric:
                                               hover_format = '.4f' # Four decimal places for ratios in hover
                                          elif 'Percent' in metric or '%' in metric:
                                               hover_format = '.2f%' # Percentage with two decimal places in hover
                                          else:
                                               hover_format = '.2f' # Default to two decimal places for others


                                          fig_flex = px.bar(
                                             bar_data_flex,
                                             x=group_by_col_flex,
                                             y=metric,
                                             color=group_by_col_flex, # Color by the selected grouping column
                                             title=f"Average {metric} by {comparison_level} ({min(selected_years_comp_flex) if selected_years_comp_flex else 'N/A'} - {max(selected_years_comp_flex) if selected_years_comp_flex else 'N/A'})",
                                             labels={group_by_col_flex: comparison_level, metric: metric},
                                             color_discrete_sequence=px.colors.qualitative.Plotly, # Use a Plotly color sequence
                                             # Explicitly set hover data format
                                              hover_data={
                                                 group_by_col_flex: True,
                                                 metric: hover_format, # Apply specific format to the metric value in hover
                                              }
                                          )
                                          # Rotate x-axis labels if there are many entities
                                          if len(bar_data_flex) > 5:
                                               fig_flex.update_layout(xaxis={'categoryorder':'total ascending', 'tickangle': 45})

                                          fig_flex.update_layout(
                                              yaxis=dict(title=metric, tickformat=y_format) # Set y-axis title and format
                                          )


                                          st.plotly_chart(fig_flex, use_container_width=True)

                                      except Exception as e:
                                          st.error(f"An error occurred while generating the plot for {metric}: {e}")
                                          st.write("Please check the data and column names for this metric.")

                                  else:
                                      st.info(f"No data available to plot {metric} for selected {entity_type_flex} and years.")

                              elif metric not in comparison_subset_data_flex.columns:
                                  st.warning(f"Selected metric '{metric}' not found in the data for plotting.")
                              elif not pd.api.types.is_numeric_dtype(comparison_subset_data_flex[metric]):
                                   st.warning(f"Selected metric '{metric}' is not numeric. Cannot plot.")

                else:
                    st.info(f"No data available for selected {entity_type_flex} and metrics to generate visualizations.")
            else:
                 st.info(f"No data available for selected {entity_type_flex} and metrics to generate visualizations.")


        else:
            st.info(f"No data available for the selected {entity_type_flex} in the selected comparison years.")


    elif selected_entities_compare and selected_metrics_compare_flex and group_by_col_flex:
         st.info("Please select year(s) for the comparison.")
    elif selected_entities_compare and selected_years_comp_flex and group_by_col_flex:
         st.info("Please select metrics to compare.")
    elif selected_metrics_compare_flex and selected_years_comp_flex:
         st.info(f"Please select {comparison_level}s to compare.")
    else:
         st.info(f"Select {comparison_level}s, metrics, and years to view flexible comparison.")


    # Removed the redundant else block for missing columns, handled at the top of the tab


# --- Tab 4 (New): Metric Trend Drilldown ---
with tab4_drilldown:
    st.header("Metric Trend Drilldown (Net Income & Cost to Charge Ratio)")
    st.markdown(f"Analyze trends for **Net Income** or **Cost To Charge Ratio** over time with drill-down capabilities, and visualize potential influencing factors, across the filtered data ({selected_state}, {selected_county_option if selected_county_option is not None else 'All Counties'}) from **{year_range[0] if year_range[0] is not None else 'N/A'}** to **{year_range[1] if year_range[1] is not None else 'N/A'}**.")

    # --- Controls for Metric Trend Drilldown ---
    col_metric_drilldown, col_level_drilldown = st.columns(2)

    with col_metric_drilldown:
        # Select the primary metric for trend analysis
        drilldown_metric_options = [
            'Net Income',
            'Cost To Charge Ratio'
        ]
        # Filter to include only metrics present and numeric in the filtered data
        present_numeric_drilldown_metrics = [
            metric for metric in filtered_df.columns
            if metric in drilldown_metric_options and pd.api.types.is_numeric_dtype(filtered_df[metric])
        ]
        selected_drilldown_metric = st.selectbox(
            "Select Primary Metric for Trend Analysis",
            options=present_numeric_drilldown_metrics,
            key='select_drilldown_metric',
            help="Choose the main financial metric to analyze over time."
        )

    with col_level_drilldown:
        # Select the level of analysis
        drilldown_level_options = ['All States', 'State', 'County', 'Hospital']
        selected_drilldown_level = st.radio(
            "Select Drilldown Level",
            options=drilldown_level_options,
            horizontal=True,
            key='select_drilldown_level',
            help="Choose the geographic level for trend analysis."
        )

    # Dynamic Hospital Selection if level is Hospital
    selected_hospital_drilldown = None
    if selected_drilldown_level == 'Hospital' and 'Hospital Name' in filtered_df.columns:
        # Get unique hospital names based on current global filters
        hospital_options_drilldown = sorted(filtered_df['Hospital Name'].dropna().unique().tolist())
        if hospital_options_drilldown:
            selected_hospital_drilldown = st.selectbox(
                "Select Hospital",
                options=hospital_options_drilldown,
                key='select_hospital_drilldown',
                help="Select a specific hospital for trend analysis."
            )
        else:
            st.info("No hospitals found for the current global filters to display at the Hospital level.")
            # Fallback if no hospitals available - maybe default to County or State?
            # For now, just display the info message and the plot will be empty.


    # --- Control for number of influential metrics to display (for correlation plot) ---
    st.subheader("Trend Visualization with Correlated Factors")
    st.markdown("Automatically identify and plot metrics most correlated with the primary metric at the selected level over the entire filtered period.")

    num_influential_metrics = st.slider(
        "Number of Most Correlated Metrics to Display (in Trend Plot)",
        min_value=0,
        max_value=10, # Limit to a reasonable number to keep the plot readable
        value=3, # Default to showing the top 3
        step=1,
        key='num_influential_metrics',
        help="Choose how many of the most correlated metrics to plot alongside the primary trend."
    )


    # --- Data Preparation and Plotting for Selected Trend and Auto-Identified Factors ---
    metrics_to_plot_drilldown = [] # Initialize list of metrics to plot

    if not filtered_df.empty and selected_drilldown_metric and selected_drilldown_metric in filtered_df.columns and 'Year' in filtered_df.columns:

        # Ensure the primary selected metric is numeric
        filtered_df[selected_drilldown_metric] = pd.to_numeric(filtered_df[selected_drilldown_metric], errors='coerce')

        # Determine the grouping column based on the selected level
        group_col_drilldown = None
        if selected_drilldown_level == 'All States':
            # For 'All States', we don't group by an entity column for the trend plot
            # We will aggregate across all data points for each year
            group_col_drilldown = 'Year' # Group by year for the trend
            plot_data_level = filtered_df.copy()
            plot_title_drilldown = f"Average Metric Trends Across All States"

        elif selected_drilldown_level == 'State' and 'State Code' in filtered_df.columns:
             group_col_drilldown = 'State Code'
             plot_data_level = filtered_df.copy()
             plot_title_drilldown = f"Average Metric Trends by State"

        elif selected_drilldown_level == 'County' and 'County' in filtered_df.columns:
             group_col_drilldown = 'County'
             plot_data_level = filtered_df.copy()
             plot_title_drilldown = f"Average Metric Trends by County"

        elif selected_drilldown_level == 'Hospital' and selected_hospital_drilldown and 'Hospital Name' in filtered_df.columns:
             group_col_drilldown = 'Hospital Name'
             # Filter data for the selected hospital
             plot_data_level = filtered_df[filtered_df['Hospital Name'] == selected_hospital_drilldown].copy()
             plot_title_drilldown = f"Metric Trends for {selected_hospital_drilldown}"

        else:
             st.info(f"Cannot generate trend plot for selected level '{selected_drilldown_level}'. Required columns might be missing or no hospital selected.")
             plot_data_level = pd.DataFrame() # Ensure empty DataFrame if level is invalid


        if not plot_data_level.empty and group_col_drilldown is not None:
            # Ensure Year is numeric in the level-filtered data
            plot_data_level['Year'] = pd.to_numeric(plot_data_level['Year'], errors='coerce')
            plot_data_level = plot_data_level.dropna(subset=['Year', selected_drilldown_metric]) # Drop rows with NaN Year or primary metric


            # --- Identify Influential Metrics using Correlation (for the trend plot) ---
            # Select only numeric columns for correlation analysis within the current level's data
            numeric_cols_level = plot_data_level.select_dtypes(include=np.number).columns.tolist()
            # Exclude identifier and grouping columns from correlation calculation
            exclude_from_corr = ['Year', 'facility_id']
            # Add group_col_drilldown to exclusion if it's not 'Year' (the grouping for All States)
            if group_col_drilldown != 'Year' and group_col_drilldown in numeric_cols_level:
                 exclude_from_corr.append(group_col_drilldown)


            # Ensure the primary metric is in the list of numeric columns before calculating correlation
            if selected_drilldown_metric in numeric_cols_level:
                cols_for_correlation = [col for col in numeric_cols_level if col not in exclude_from_corr]

                if len(cols_for_correlation) > 1: # Need at least the primary metric and one other numeric column
                    # Calculate the correlation matrix for the relevant columns
                    correlation_matrix = plot_data_level[cols_for_correlation].corr()

                    # Get correlations of the primary metric with all other columns
                    if selected_drilldown_metric in correlation_matrix.columns:
                        correlations_with_primary = correlation_matrix[selected_drilldown_metric].drop(selected_drilldown_metric)

                        # Get the top N metrics with the highest absolute correlation
                        # Handle cases where num_influential_metrics is greater than available correlated metrics
                        num_to_select = min(num_influential_metrics, len(correlations_with_primary.dropna()))

                        if num_to_select > 0:
                             top_correlated_metrics = correlations_with_primary.dropna().abs().sort_values(ascending=False).head(num_to_select).index.tolist()
                             metrics_to_plot_drilldown = [selected_drilldown_metric] + top_correlated_metrics # Always include the primary metric

                             st.markdown(f"**Metrics identified as most correlated with {selected_drilldown_metric} at the {selected_drilldown_level} level (for trend plot):**")
                             st.write(", ".join(top_correlated_metrics))
                             st.markdown(f"*(Based on Pearson correlation coefficient calculated over the filtered data from {year_range[0] if year_range[0] is not None else 'N/A'} to {year_range[1] if year_range[1] is not None else 'N/A'})*")

                        elif selected_drilldown_metric in filtered_df.columns: # If primary metric is valid but no other numeric cols or num_to_select is 0
                             metrics_to_plot_drilldown = [selected_drilldown_metric] # Only plot the primary metric
                             st.info(f"No additional influential metrics identified based on correlation at the {selected_drilldown_level} level or the number of metrics to display is set to 0.")

                        else:
                             st.warning(f"Primary metric '{selected_drilldown_metric}' not found in filtered data for correlation analysis.")

                    elif selected_drilldown_metric in filtered_df.columns: # If primary metric is valid but not in correlation matrix (e.g., only one numeric col)
                         metrics_to_plot_drilldown = [selected_drilldown_metric] # Only plot the primary metric
                         st.info(f"Not enough numeric columns available at the {selected_drilldown_level} level to calculate correlations for influential metrics. Plotting only the primary metric.")
                    else:
                         st.warning(f"Primary metric '{selected_drilldown_metric}' not found in filtered data for correlation analysis.")


                elif selected_drilldown_metric in filtered_df.columns: # If only one numeric column (the primary metric)
                     metrics_to_plot_drilldown = [selected_drilldown_metric] # Only plot the primary metric
                     st.info(f"Not enough numeric columns available at the {selected_drilldown_level} level to calculate correlations for influential metrics. Plotting only the primary metric.")
                else:
                     st.warning(f"Primary metric '{selected_drilldown_metric}' not found in filtered data.")

            elif selected_drilldown_metric in filtered_df.columns: # If no numeric columns other than potentially the primary metric
                 metrics_to_plot_drilldown = [selected_drilldown_metric] # Only plot the primary metric
                 st.info(f"No other numeric columns available in the filtered data at the {selected_drilldown_level} level to identify influential metrics. Plotting only the primary metric.")
            else:
                 st.warning(f"Primary metric '{selected_drilldown_metric}' not found in filtered data.")


            # --- Aggregate Data for Trend Plotting ---
            if metrics_to_plot_drilldown:
                # Ensure all metrics selected for plotting are numeric
                valid_metrics_to_plot = [
                    metric for metric in metrics_to_plot_drilldown
                    if metric in plot_data_level.columns and pd.api.types.is_numeric_dtype(plot_data_level[metric])
                ]

                if valid_metrics_to_plot:
                    # Aggregate by Year and the grouping column
                    aggregation_dict_drilldown = {metric: 'mean' for metric in valid_metrics_to_plot}

                    # Special handling for 'All States' level aggregation for plotting
                    if selected_drilldown_level == 'All States':
                         plot_data_aggregated = plot_data_level.groupby('Year')[list(aggregation_dict_drilldown.keys())].agg(aggregation_dict_drilldown).reset_index()
                         # Add a dummy group column for melting if needed, but Plotly Express might handle it
                         plot_data_aggregated['Group'] = 'All States'
                         group_col_for_melt = 'Group' # Use the dummy group column for melting
                    else:
                         # Aggregate by Year and the entity grouping column for other levels
                         if group_col_drilldown and group_col_drilldown in plot_data_level.columns:
                             plot_data_aggregated = plot_data_level.groupby(['Year', group_col_drilldown])[list(aggregation_dict_drilldown.keys())].agg(aggregation_dict_drilldown).reset_index()
                             pivot_index_col = group_col_drilldown # Use the actual group column for pivoting
                             group_col_for_melt = group_col_drilldown # Use the actual group column for melting
                         else:
                              st.warning(f"Grouping column '{group_col_drilldown}' not found in data for aggregation at level '{selected_drilldown_level}'.")
                              plot_data_aggregated = pd.DataFrame() # Ensure empty if grouping column is missing


                    if not plot_data_aggregated.empty and group_col_for_melt in plot_data_aggregated.columns:
                        # Melt the aggregated data for plotting
                        plot_data_melted = plot_data_aggregated.melt(
                            id_vars=['Year', group_col_for_melt], # Use the appropriate group column
                            value_vars=valid_metrics_to_plot,
                            var_name='Metric',
                            value_name='Value'
                        )

                        if not plot_data_melted.empty:
                            # Create the Plotly Express line chart
                            fig_drilldown = px.line(
                                plot_data_melted,
                                x='Year',
                                y='Value',
                                color='Metric', # Color distinguishes different metrics
                                line_dash=group_col_for_melt if selected_drilldown_level != 'All States' else None, # Line style distinguishes entities if not All States
                                title=plot_title_drilldown,
                                markers=True,
                                labels={'Year': 'Year', 'Value': 'Average Value', 'Metric': 'Metric', group_col_for_melt: selected_drilldown_level if selected_drilldown_level != 'All States' else 'Level'}, # Dynamic labels
                                color_discrete_sequence=px.colors.qualitative.Plotly, # Use a Plotly color sequence
                                hover_name=group_col_for_melt if selected_drilldown_level != 'All States' else None, # Show entity name on hover if applicable
                                hover_data={
                                    'Year': True,
                                    'Metric': True,
                                    'Value': ':.2f', # Default value format, can be customized per metric if needed
                                    group_col_for_melt: False if selected_drilldown_level == 'All States' else True # Show group in hover if applicable
                                }
                            )

                            fig_drilldown.update_layout(
                                xaxis=dict(title="Year", dtick=1), # Ensure integer ticks for years
                                yaxis=dict(title="Value"), # Generic Y-axis title as multiple metrics might have different units
                                hovermode='x unified' # Show hover info for all traces at a given x-coordinate
                            )

                            st.plotly_chart(fig_drilldown, use_container_width=True)

                            st.markdown("""
                            **Interpreting the Trend Plot:**
                            * Each line represents the trend of a specific metric for a specific entity (State, County, or Hospital).
                            * Compare the trends of the primary metric ('Net Income' or 'Cost To Charge Ratio') with the trends of the automatically identified most correlated metrics over the entire filtered period.
                            * Look for visual patterns where changes in other metrics seem to broadly align with changes in the primary metric's trend.
                            * **Note:** Metrics with vastly different scales are plotted on the same Y-axis. Focus on the *shape* and *direction* of the trends for visual comparison, rather than the absolute values unless the scales are similar.
                            """)

                        else:
                             st.info(f"No melted data available for plotting the selected metrics and level.")
                    elif not plot_data_aggregated.empty:
                         st.warning(f"Grouping column '{group_col_for_melt}' not found in aggregated data for melting.")
                    else:
                         st.info(f"No aggregated data available to plot the selected metrics for the selected level ({selected_drilldown_level}) and filters.")
                elif metrics_to_plot_drilldown:
                    st.warning("None of the selected metrics for plotting are present or numeric in the level-filtered data.")
                # If metrics_to_plot_drilldown is empty, the initial checks above would have shown a message.

            elif selected_drilldown_metric: # If primary metric is selected but no other metrics identified or available
                 st.info(f"No data available to plot the primary metric '{selected_drilldown_metric}' for the selected level ({selected_drilldown_level}) and filters.")
            else: # If no primary metric is selected
                 st.info("Please select a primary metric for trend analysis.")

        elif selected_drilldown_metric and not plot_data_level.empty:
             st.warning(f"Primary metric '{selected_drilldown_metric}' is not numeric in the filtered data for the selected level.")
        elif selected_drilldown_metric: # If primary metric is selected but plot_data_level is empty
             st.info(f"Filtered data is empty for the selected level ({selected_drilldown_level}). Cannot perform trend analysis.")
        else: # If no primary metric is selected and data is empty
             st.info("Filtered data is empty, cannot generate trend analysis.")


    elif not filtered_df.empty and not metrics_to_plot_drilldown:
         st.info("Please select a primary metric for trend analysis.")
    elif not filtered_df.empty:
         st.warning("Required columns (Primary Metric, Year) are missing in the filtered data for trend analysis.")
    else:
        st.info("Filtered data is empty, cannot generate trend analysis.")

    st.markdown("---") # Add a separator

    # --- New Section: Year-over-Year Change Analysis ---
    st.subheader("Year-over-Year Change Analysis")
    st.markdown("Analyze the change in the primary metric and key factors from the previous year to a selected analysis year at the chosen drilldown level.")

    # Select the year for YoY analysis
    # Get available years from the filtered data that have a previous year
    if 'Year' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['Year']):
        available_years_yoy = sorted(filtered_df['Year'].dropna().unique().tolist())
        # Only include years for which a previous year exists in the data
        years_with_previous = [year for year in available_years_yoy if year - 1 in available_years_yoy]

        if years_with_previous:
            selected_analysis_year = st.selectbox(
                "Select Analysis Year for YoY Change",
                options=years_with_previous,
                key='select_analysis_year_yoy',
                help="Choose the year to analyze the change from the previous year."
            )
        else:
            st.info("Not enough years available in the filtered data to perform Year-over-Year analysis.")
            selected_analysis_year = None # Ensure it's None if no years for YoY
    else:
        st.warning("'Year' column not found or not numeric in data. YoY analysis disabled.")
        selected_analysis_year = None # Ensure it's None if Year column is missing


    # Define the list of factors to analyze for YoY change
    # These should be potential drivers of change in the primary metric
    yoy_factors_to_analyze = [
        'Total Revenue',
        'Net Patient Revenue',
        'Inpatient Revenue',
        'Outpatient Revenue',
        'Less Total Operating Expense',
        'Total Salaries From Worksheet A',
        'Depreciation Cost',
        'Contractual Allowance and Discounts on Patients\' Accounts',
        'Total Bad Debt Expense',
        'Cost of Charity Care',
        'Total_Staffing_FTE',
        'Number of Beds',
        'Annual', # Inflation
        'Total_Yearly_Medicare_Enrollment' # Medicare Enrollment
    ]

    # Add the primary selected metric to the list of metrics for YoY analysis
    metrics_for_yoy_analysis = []
    if selected_drilldown_metric:
         metrics_for_yoy_analysis.append(selected_drilldown_metric)
    # Filter factors to include only those present and numeric in the filtered data
    present_numeric_yoy_factors = [
        metric for metric in filtered_df.columns
        if metric in yoy_factors_to_analyze and pd.api.types.is_numeric_dtype(filtered_df[metric])
    ]
    metrics_for_yoy_analysis.extend(present_numeric_yoy_factors)
    # Ensure uniqueness and remove the primary metric from factors if it was already in the factor list
    metrics_for_yoy_analysis = list(set(metrics_for_yoy_analysis))
    # Reorder to have the primary metric first if it's selected
    if selected_drilldown_metric and selected_drilldown_metric in metrics_for_yoy_analysis:
        metrics_for_yoy_analysis.remove(selected_drilldown_metric)
        metrics_for_yoy_analysis.insert(0, selected_drilldown_metric)


    # Perform YoY analysis if a year is selected and data is available
    # Also ensure that the determined group_col_drilldown is valid for the YoY analysis
    # For 'All States', group_col_drilldown was set to 'Year' for the trend plot,
    # but for YoY analysis, we need to handle 'All States' as a special case for aggregation.
    if selected_analysis_year is not None and not filtered_df.empty and metrics_for_yoy_analysis:

        previous_year = selected_analysis_year - 1

        # Filter data for the analysis year and the previous year, at the selected drilldown level
        # Use the same filtering logic as for the trend plot to get the relevant subset
        yoy_data_subset = pd.DataFrame() # Initialize empty
        yoy_group_col = None # Initialize group column for YoY analysis
        entity_display_name = selected_drilldown_level # Name to use in the summary

        if selected_drilldown_level == 'All States':
            # For 'All States', filter the globally filtered data for the two years
            yoy_data_subset = filtered_df[
                filtered_df['Year'].isin([previous_year, selected_analysis_year])
            ].copy()
            # No grouping column needed for aggregation across all data
            yoy_group_col = None # Indicate no grouping column for aggregation
            entity_display_name = 'All States' # Use 'All States' for the summary

        elif selected_drilldown_level == 'State' and 'State Code' in filtered_df.columns:
             yoy_data_subset = filtered_df[
                 filtered_df['Year'].isin([previous_year, selected_analysis_year])
             ].copy()
             yoy_group_col = 'State Code'
             entity_display_name = f"Selected State ({selected_state})" if selected_state != 'All States' else 'States'


        elif selected_drilldown_level == 'County' and 'County' in filtered_df.columns:
             yoy_data_subset = filtered_df[
                 filtered_df['Year'].isin([previous_year, selected_analysis_year])
             ].copy()
             yoy_group_col = 'County'
             entity_display_name = f"Selected County ({selected_county_option})" if selected_county_option != 'All Counties' and selected_county_option is not None else 'Counties'


        elif selected_drilldown_level == 'Hospital' and selected_hospital_drilldown and 'Hospital Name' in filtered_df.columns:
             yoy_data_subset = filtered_df[
                 (filtered_df['Hospital Name'] == selected_hospital_drilldown) &
                 (filtered_df['Year'].isin([previous_year, selected_analysis_year]))
             ].copy()
             yoy_group_col = 'Hospital Name' # Group by hospital name for this level
             entity_display_name = f"Selected Hospital ({selected_hospital_drilldown})"

        else:
            st.info(f"Cannot perform YoY analysis for selected level '{selected_drilldown_level}'. Required columns might be missing or no hospital selected.")
            yoy_data_subset = pd.DataFrame() # Ensure empty DataFrame


        if not yoy_data_subset.empty:
            # Ensure all metrics for YoY analysis are numeric
            for metric in metrics_for_yoy_analysis:
                if metric in yoy_data_subset.columns:
                    yoy_data_subset[metric] = pd.to_numeric(yoy_data_subset[metric], errors='coerce')
                else:
                     st.warning(f"Metric '{metric}' not found in data for YoY analysis.")
                     if metric in metrics_for_yoy_analysis:
                         metrics_for_yoy_analysis.remove(metric) # Remove missing metric


            # Aggregate data based on the drilldown level
            aggregation_dict_yoy = {metric: 'mean' for metric in metrics_for_yoy_analysis}

            if aggregation_dict_yoy:
                if selected_drilldown_level == 'All States':
                    # Aggregate only by Year for 'All States'
                    yoy_aggregated_data = yoy_data_subset.groupby('Year')[list(aggregation_dict_yoy.keys())].agg(aggregation_dict_yoy).reset_index()
                    # Add a dummy group column for consistency in the next step if needed, but pivot handles it
                    yoy_aggregated_data['Group'] = 'All States'
                    pivot_index_col = 'Group' # Use the dummy group column for pivoting
                else:
                    # Aggregate by Year and the entity grouping column for other levels
                    if yoy_group_col and yoy_group_col in yoy_data_subset.columns:
                        yoy_aggregated_data = yoy_data_subset.groupby(['Year', yoy_group_col])[list(aggregation_dict_yoy.keys())].agg(aggregation_dict_yoy).reset_index()
                        pivot_index_col = yoy_group_col # Use the actual group column for pivoting
                    else:
                         st.warning(f"Grouping column '{yoy_group_col}' not found in data for aggregation at level '{selected_drilldown_level}'.")
                         yoy_aggregated_data = pd.DataFrame() # Ensure empty if grouping column is missing


                if not yoy_aggregated_data.empty and pivot_index_col in yoy_aggregated_data.columns:
                    # Pivot the aggregated data to have years as columns
                    # Ensure the index column exists before pivoting
                    yoy_pivot = yoy_aggregated_data.pivot(index=pivot_index_col, columns='Year', values=list(aggregation_dict_yoy.keys())).reset_index()

                    # Flatten the multi-level columns created by pivot
                    yoy_pivot.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in yoy_pivot.columns]

                    # Calculate change and percentage change for each metric
                    yoy_results_list = []

                    # Iterate through each entity (row in the pivoted DataFrame)
                    for index, entity_row in yoy_pivot.iterrows():
                        entity_name_yoy = entity_row[pivot_index_col] # Get the entity name

                        for metric in list(aggregation_dict_yoy.keys()):
                            col_prev_year = f'{metric}_{previous_year}'
                            col_analysis_year = f'{metric}_{selected_analysis_year}'

                            # Check if both year columns exist for the current metric
                            if col_prev_year in entity_row.index and col_analysis_year in entity_row.index:
                                prev_year_value = entity_row[col_prev_year]
                                analysis_year_value = entity_row[col_analysis_year]

                                # Calculate Absolute Change
                                absolute_change = analysis_year_value - prev_year_value

                                # Calculate Percentage Change, handling division by zero
                                percentage_change = np.nan
                                if prev_year_value != 0 and pd.notna(prev_year_value):
                                    percentage_change = (absolute_change / prev_year_value) * 100
                                elif prev_year_value == 0 and absolute_change != 0:
                                     # Handle case where previous year was 0 and there was a change
                                     percentage_change = float('inf') * np.sign(absolute_change) # Represent as +/- infinity
                                elif prev_year_value == 0 and absolute_change == 0:
                                     percentage_change = 0 # No change from zero


                                # Add results to the list for display
                                yoy_results_list.append({
                                    selected_drilldown_level if selected_drilldown_level != 'All States' else 'Entity': entity_name_yoy, # Use dynamic column name
                                    'Metric': metric,
                                    f'Average Value in {previous_year}': prev_year_value,
                                    f'Average Value in {selected_analysis_year}': analysis_year_value,
                                    'Absolute Change': absolute_change,
                                    'Percentage Change (%)': percentage_change
                                })
                            else:
                                # Handle case where data for one of the years is missing for this metric/entity
                                yoy_results_list.append({
                                    selected_drilldown_level if selected_drilldown_level != 'All States' else 'Entity': entity_name_yoy,
                                    'Metric': metric,
                                    f'Average Value in {previous_year}': entity_row.get(col_prev_year, np.nan), # Use .get() to avoid error if column is missing
                                    f'Average Value in {selected_analysis_year}': entity_row.get(col_analysis_year, np.nan),
                                    'Absolute Change': np.nan,
                                    'Percentage Change (%)': np.nan,
                                    'Note': f"Data missing for {previous_year} or {selected_analysis_year}"
                                })


                    # Display the YoY results in a DataFrame
                    if yoy_results_list:
                        yoy_results_df = pd.DataFrame(yoy_results_list)
                        st.subheader(f"Change from {previous_year} to {selected_analysis_year} for {entity_display_name}:")

                        # --- Enhanced Interpretation and Summary ---
                        st.markdown("---") # Separator before summary
                        st.subheader("Key Insights from Year-over-Year Change")

                        # Find the row(s) for the primary metric
                        primary_metric_rows = yoy_results_df[yoy_results_df['Metric'] == selected_drilldown_metric]

                        if not primary_metric_rows.empty:
                            # If 'All States', there's only one row for the primary metric
                            # If State/County/Hospital, there might be multiple rows if the grouping didn't fully collapse (though aggregation should)
                            # Let's assume aggregation worked and take the first row for the summary focus
                            primary_metric_row = primary_metric_rows.iloc[0]

                            primary_abs_change = primary_metric_row['Absolute Change']
                            primary_pct_change = primary_metric_row['Percentage Change (%)']
                            primary_value_prev = primary_metric_row[f'Average Value in {previous_year}']
                            primary_value_analysis = primary_metric_row[f'Average Value in {selected_analysis_year}']

                            # Describe the change in the primary metric
                            change_description = ""
                            if pd.notna(primary_abs_change):
                                change_type = "increased" if primary_abs_change > 0 else ("decreased" if primary_abs_change < 0 else "remained unchanged")
                                change_amount_abs = f"${abs(primary_abs_change):,.2f}" if 'Income' in selected_drilldown_metric or 'Revenue' in selected_drilldown_metric or 'Expense' in selected_drilldown_metric else f"{abs(primary_abs_change):,.2f}"

                                change_amount_pct = ""
                                if pd.notna(primary_pct_change):
                                     if primary_pct_change == float('inf'):
                                         change_amount_pct = "infinitely"
                                     elif primary_pct_change == float('-inf'):
                                          change_amount_pct = "infinitely"
                                     else:
                                         change_amount_pct = f"{abs(primary_pct_change):.2f}%"


                                if change_amount_pct:
                                     change_description = f"The average **{selected_drilldown_metric}** {change_type} by **{change_amount_abs}** ({change_amount_pct}) from {previous_year} to {selected_analysis_year} for {entity_display_name}."
                                else:
                                     change_description = f"The average **{selected_drilldown_metric}** {change_type} by **{change_amount_abs}** from {previous_year} to {selected_analysis_year} for {entity_display_name}."


                            else:
                                change_description = f"Change in average **{selected_drilldown_metric}** from {previous_year} to {selected_analysis_year} for {entity_display_name} could not be calculated due to missing data."

                            st.markdown(change_description)

                            # Identify factors with significant changes (excluding the primary metric itself)
                            factors_yoy_df = yoy_results_df[yoy_results_df['Metric'] != selected_drilldown_metric].copy()
                            factors_yoy_df = factors_yoy_df.dropna(subset=['Absolute Change', 'Percentage Change (%)']) # Only consider factors with calculated changes

                            if not factors_yoy_df.empty:
                                # Sort by absolute change magnitude
                                factors_yoy_df['Abs_Abs_Change'] = factors_yoy_df['Absolute Change'].abs()
                                top_abs_change_factors = factors_yoy_df.sort_values(by='Abs_Abs_Change', ascending=False).head(3) # Top 3 by absolute change

                                # Sort by percentage change magnitude (handle inf values if necessary, or exclude)
                                # Let's exclude inf values for percentage change ranking for now to avoid skewing
                                factors_for_pct_ranking = factors_yoy_df[np.isfinite(factors_yoy_df['Percentage Change (%)'])].copy()
                                factors_for_pct_ranking['Abs_Pct_Change'] = factors_for_pct_ranking['Percentage Change (%)'].abs()
                                top_pct_change_factors = factors_for_pct_ranking.sort_values(by='Abs_Pct_Change', ascending=False).head(3) # Top 3 by percentage change

                                st.markdown(f"Here's a look at some key factors that saw significant changes in the same period:")

                                # Describe factors with largest absolute changes
                                if not top_abs_change_factors.empty:
                                    st.markdown("**Factors with the largest absolute changes:**")
                                    for index, row in top_abs_change_factors.iterrows():
                                        factor_metric = row['Metric']
                                        factor_abs_change = row['Absolute Change']
                                        factor_pct_change = row['Percentage Change (%)']

                                        factor_change_type = "increased" if factor_abs_change > 0 else ("decreased" if factor_abs_change < 0 else "remained unchanged")
                                        factor_change_amount_abs = f"${abs(factor_abs_change):,.2f}" if 'Income' in factor_metric or 'Revenue' in factor_metric or 'Expense' in factor_metric else f"{abs(factor_abs_change):,.2f}"

                                        factor_change_amount_pct = ""
                                        if pd.notna(factor_pct_change):
                                             if factor_pct_change == float('inf'):
                                                 factor_change_amount_pct = "infinitely"
                                             elif factor_pct_change == float('-inf'):
                                                  factor_change_amount_pct = "infinitely"
                                             else:
                                                 factor_change_amount_pct = f"{abs(factor_pct_change):.2f}%"


                                        summary_text = f"* **{factor_metric}** {factor_change_type} by **{factor_change_amount_abs}**"
                                        if factor_change_amount_pct:
                                             summary_text += f" ({factor_change_amount_pct})"
                                        summary_text += f"."

                                        # Add a note about alignment with primary metric change
                                        if pd.notna(primary_abs_change) and pd.notna(factor_abs_change):
                                             # Simple check for same/opposite direction
                                             if np.sign(primary_abs_change) == np.sign(factor_abs_change):
                                                  summary_text += f" This change moved in the same direction as {selected_drilldown_metric}."
                                             else:
                                                  summary_text += f" This change moved in the opposite direction of {selected_drilldown_metric}."


                                        st.markdown(summary_text)

                                # Describe factors with largest percentage changes (if different from absolute)
                                if not top_pct_change_factors.empty and not top_pct_change_factors.equals(top_abs_change_factors):
                                     st.markdown("**Factors with the largest percentage changes (excluding infinite changes):**")
                                     for index, row in top_pct_change_factors.iterrows():
                                         factor_metric = row['Metric']
                                         factor_abs_change = row['Absolute Change']
                                         factor_pct_change = row['Percentage Change (%)']

                                         factor_change_type = "increased" if factor_abs_change > 0 else ("decreased" if factor_abs_change < 0 else "remained unchanged")
                                         factor_change_amount_pct = f"{abs(factor_pct_change):.2f}%"

                                         summary_text = f"* **{factor_metric}** {factor_change_type} by **{factor_change_amount_pct}**"
                                         # Add a note about alignment with primary metric change
                                         if pd.notna(primary_abs_change) and pd.notna(factor_abs_change):
                                             # Simple check for same/opposite direction
                                             if np.sign(primary_abs_change) == np.sign(factor_abs_change):
                                                  summary_text += f" This change moved in the same direction as {selected_drilldown_metric}."
                                             else:
                                                  summary_text += f" This change moved in the opposite direction of {selected_drilldown_metric}."

                                         st.markdown(summary_text)


                                # --- Specific Decomposition for Net Income ---
                                if selected_drilldown_metric == 'Net Income':
                                     st.markdown("**Net Income Decomposition:**")
                                     total_revenue_row = yoy_results_df[(yoy_results_df['Metric'] == 'Total Revenue') & (yoy_results_df[selected_drilldown_level if selected_drilldown_level != 'All States' else 'Entity'] == entity_name_yoy)]
                                     total_expense_row = yoy_results_df[(yoy_results_df['Metric'] == 'Less Total Operating Expense') & (yoy_results_df[selected_drilldown_level if selected_drilldown_level != 'All States' else 'Entity'] == entity_name_yoy)]


                                     if not total_revenue_row.empty and not total_expense_row.empty:
                                         revenue_change = total_revenue_row['Absolute Change'].iloc[0]
                                         expense_change = total_expense_row['Absolute Change'].iloc[0]

                                         if pd.notna(revenue_change) and pd.notna(expense_change):
                                             calculated_net_income_change = revenue_change - expense_change
                                             st.markdown(f"The change in **Total Revenue** (${calculated_net_income_change:,.2f}) minus the change in **Less Total Operating Expense** (${expense_change:,.2f}) approximately equals the change in **Net Income** (${calculated_net_income_change:,.2f}). This highlights how changes in overall revenue and operating costs directly impact Net Income.")
                                         else:
                                             st.info("Cannot perform Net Income decomposition due to missing data for Total Revenue or Less Total Operating Expense.")
                                     else:
                                         st.info("Data for Total Revenue or Less Total Operating Expense is not available to perform Net Income decomposition.")


                            else:
                                st.info("No other factors with calculated Year-over-Year changes available to provide key insights.")

                        else:
                            st.warning(f"Year-over-Year change for the primary metric '{selected_drilldown_metric}' could not be calculated for the selected entity/level.")

                        st.markdown("---") # Separator after summary


                        # Format the numeric columns for better readability in the table
                        st.dataframe(yoy_results_df.style.format({
                            f'Average Value in {previous_year}': '{:,.2f}',
                            f'Average Value in {selected_analysis_year}': '{:,.2f}',
                            'Absolute Change': '{:,.2f}',
                            'Percentage Change (%)': '{:,.2f}%'
                        }))

                        st.markdown("""
                        **Interpreting the Year-over-Year Change Table:**
                        * This table shows how the average value of each metric changed from the year before the selected Analysis Year to the Analysis Year itself, at the chosen drilldown level.
                        * Use this table to see the exact numbers behind the key insights provided above.
                        """)

                    else:
                        st.info(f"No complete data available for YoY analysis for the selected metrics and level between {previous_year} and {selected_analysis_year}.")

                elif not yoy_aggregated_data.empty:
                     st.warning(f"Pivot index column '{pivot_index_col}' not found in aggregated data for YoY analysis.")
                else:
                    st.info(f"No aggregated data available for YoY analysis for the selected level between {previous_year} and {selected_analysis_year}.")
            else:
                 st.info("No metrics selected or available for Year-over-Year analysis.")

        elif selected_analysis_year is not None and not filtered_df.empty:
             st.warning("Required columns for YoY analysis (Metrics, Year, Grouping Column) are missing or not numeric in the filtered data.")
        elif selected_analysis_year is not None:
             st.info("Filtered data is empty. Cannot perform Year-over-Year analysis.")
        else:
             st.info("Select an Analysis Year to view Year-over-Year changes.")


# --- Tab 2 (Detail): Top & Bottom Hospitals ---
# This tab remains unchanged
# ... (code for Top & Bottom Hospitals tab) ...


# --- Tab 3 (Detail): Comparison Analysis ---
# This tab remains unchanged
# ... (code for Comparison Analysis tab) ...


# =============================================================================
# --- End of Dashboard Sections ---
# =============================================================================
