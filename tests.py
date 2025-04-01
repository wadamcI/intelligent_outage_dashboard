import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
import random
from datetime import datetime, timedelta
from folium.plugins import HeatMap

# Set up Streamlit page layout
st.set_page_config(layout="wide")

# Initialize session state for report view
if 'show_report' not in st.session_state:
    st.session_state.show_report = False


@st.cache_data
def load_data():
    np.random.seed(42)
    num_rows = 500  # Adjust the number of synthetic outage events

    data = {
        "id": np.arange(1, num_rows + 1),
        "latitude_x": np.random.uniform(32.5121, 42.0126, num_rows),
        "longitude_x": np.random.uniform(-124.6509, -114.1315, num_rows),
        "datetime": [datetime(2024, 1, 1) + timedelta(hours=random.randint(0, 1000)) for _ in range(num_rows)],
        "estcustaffected": np.random.randint(10, 5000, num_rows),
        "cause": np.random.choice(["Storm", "Equipment Failure", "Unknown", "Tree Contact", "Animal Contact"],
                                  num_rows),
        "regionname": np.random.choice(["Northern CA", "Central CA", "Southern CA"], num_rows),
    }
    return pd.DataFrame(data)


# Load data
df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Options")

# Convert datetime column to timestamps for use in slider
df["timestamp"] = df["datetime"].astype("int64") // 10 ** 9
min_time, max_time = df["timestamp"].min(), df["timestamp"].max()

# Date Slider
selected_timestamp = st.sidebar.slider("Select Date", min_time, max_time, min_time)
# Using fromtimestamp instead of utcfromtimestamp
selected_datetime = datetime.fromtimestamp(selected_timestamp)

# Interval Selection
interval_option = st.sidebar.selectbox("Select Interval", ["1 Day", "1 Month", "6 Months", "1 Year"])

# Compute the date range based on the selected interval
if interval_option == "1 Day":
    filtered_df = df[df["datetime"].dt.date == selected_datetime.date()]
elif interval_option == "1 Month":
    filtered_df = df[(df["datetime"] >= selected_datetime) & (df["datetime"] < selected_datetime + timedelta(days=30))]
elif interval_option == "6 Months":
    filtered_df = df[(df["datetime"] >= selected_datetime) & (df["datetime"] < selected_datetime + timedelta(days=180))]
elif interval_option == "1 Year":
    filtered_df = df[(df["datetime"] >= selected_datetime) & (df["datetime"] < selected_datetime + timedelta(days=365))]

# Chat Section
st.sidebar.header("Chat Interface")
chat_input = st.sidebar.text_input("Type location or ask question...")

# Define map center and zoom based on chat input
map_center = [39, -100]  # Default center (California)
zoom_level = 4  # Default zoom level

# Process chat input to control map and view
if chat_input:
    st.sidebar.write(f"You entered: {chat_input}")

    if "california" in chat_input.lower():
        map_center = [36.7783, -119.4179]  # California coordinates
        zoom_level = 6
        st.sidebar.success("Zooming to California")
    elif "northern ca" in chat_input.lower():
        map_center = [40.7899, -122.7894]  # Northern CA approximate center
        zoom_level = 7
        st.sidebar.success("Zooming to Northern California")
    elif "central ca" in chat_input.lower():
        map_center = [36.7783, -119.4179]  # Central CA approximate center
        zoom_level = 7
        st.sidebar.success("Zooming to Central California")
    elif "southern ca" in chat_input.lower():
        map_center = [33.7175, -117.8311]  # Southern CA approximate center
        zoom_level = 7
        st.sidebar.success("Zooming to Southern California")
    elif "report" in chat_input.lower() or "analysis" in chat_input.lower():
        st.session_state.show_report = True
        st.sidebar.success("Showing detailed outage report")
    elif "map" in chat_input.lower() and st.session_state.show_report:
        st.session_state.show_report = False
        st.sidebar.success("Returning to map view")
    elif "how many" in chat_input.lower() or "outages" in chat_input.lower():
        st.sidebar.info(f"There are {len(filtered_df)} outages in the current view.")
    elif "customers" in chat_input.lower() or "affected" in chat_input.lower():
        total = filtered_df["estcustaffected"].sum()
        st.sidebar.info(f"A total of {total:,} customers are affected in this period.")
    elif "cause" in chat_input.lower():
        causes = filtered_df["cause"].value_counts()
        most_common = causes.idxmax() if not causes.empty else "None"
        st.sidebar.info(f"The most common cause is {most_common}.")
    elif "help" in chat_input.lower():
        st.sidebar.info("""You can:
- Type 'California', 'Northern CA', etc. to zoom
- Type 'report' to see detailed analysis
- Type 'map' to return to the map view
- Ask about outages, customers, causes""")

# Conditional display based on session state
if not st.session_state.show_report:
    # Map View
    st.header("Outage Events Map")
    m = folium.Map(location=map_center, zoom_start=zoom_level)

    if interval_option == "1 Day":
        # Plot outage points for 1 day
        for _, row in filtered_df.iterrows():
            folium.CircleMarker(
                location=[row["latitude_x"], row["longitude_x"]],
                radius=3,
                color="red",
                fill=True,
                fill_opacity=0.7,
                popup=f"Region: {row['regionname']}<br>Cause: {row['cause']}<br>Customers Affected: {row['estcustaffected']}",
            ).add_to(m)
    else:
        # Create a heatmap for longer intervals
        heat_data = filtered_df[["latitude_x", "longitude_x"]].values.tolist()
        HeatMap(heat_data, radius=10, blur=15).add_to(m)

    # Display the map
    st_folium(m, width=900, height=600)

    # Display outage statistics
    st.header("Outage Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Outages", len(filtered_df))

    with col2:
        total_affected = filtered_df["estcustaffected"].sum()
        st.metric("Total Customers Affected", f"{total_affected:,}")

    with col3:
        causes = filtered_df["cause"].value_counts()
        most_common = causes.idxmax() if not causes.empty else "None"
        st.metric("Most Common Cause", most_common)

else:
    # Detailed Report View
    st.header("Detailed Outage Analysis Report")

    # Time period info
    if interval_option == "1 Day":
        period_text = f"Date: {selected_datetime.strftime('%Y-%m-%d')}"
    elif interval_option == "1 Month":
        end_date = selected_datetime + timedelta(days=30)
        period_text = f"Period: {selected_datetime.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    elif interval_option == "6 Months":
        end_date = selected_datetime + timedelta(days=180)
        period_text = f"Period: {selected_datetime.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    else:  # 1 Year
        end_date = selected_datetime + timedelta(days=365)
        period_text = f"Period: {selected_datetime.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

    st.subheader(period_text)

    # Summary metrics
    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Outages", len(filtered_df))
        st.metric("Average Customers Affected per Outage", int(filtered_df["estcustaffected"].mean()))

    with col2:
        total_affected = filtered_df["estcustaffected"].sum()
        st.metric("Total Customers Affected", f"{total_affected:,}")

        # Max outage
        max_outage = filtered_df.loc[filtered_df["estcustaffected"].idxmax()] if not filtered_df.empty else None
        if max_outage is not None:
            st.metric("Largest Outage", f"{int(max_outage['estcustaffected']):,} customers")

    with col3:
        causes = filtered_df["cause"].value_counts()
        most_common = causes.idxmax() if not causes.empty else "None"
        st.metric("Most Common Cause", most_common)

        # Add percentage
        if not causes.empty:
            percentage = (causes[most_common] / len(filtered_df)) * 100
            st.metric("Percentage of Total", f"{percentage:.1f}%")

    # Cause breakdown
    st.subheader("Outage Causes")
    cause_counts = filtered_df["cause"].value_counts().reset_index()
    cause_counts.columns = ["Cause", "Count"]
    cause_counts["Percentage"] = (cause_counts["Count"] / cause_counts["Count"].sum() * 100).round(1)
    st.bar_chart(cause_counts.set_index("Cause")["Count"])
    st.table(cause_counts)

    # Region analysis
    st.subheader("Regional Analysis")
    region_data = filtered_df.groupby("regionname").agg(
        outage_count=("id", "count"),
        total_affected=("estcustaffected", "sum"),
        avg_affected=("estcustaffected", "mean")
    ).reset_index()

    region_data["avg_affected"] = region_data["avg_affected"].round(0).astype(int)
    st.table(region_data)

    # Time analysis (daily distribution)
    st.subheader("Temporal Analysis")
    filtered_df["hour"] = filtered_df["datetime"].dt.hour
    hour_counts = filtered_df["hour"].value_counts().sort_index().reset_index()
    hour_counts.columns = ["Hour of Day", "Number of Outages"]
    st.line_chart(hour_counts.set_index("Hour of Day"))

    # Additional insights
    st.subheader("Key Insights")

    # Most affected region
    most_affected_region = region_data.loc[region_data["total_affected"].idxmax(), "regionname"]
    region_pct = region_data.loc[region_data["total_affected"].idxmax(), "total_affected"] / total_affected * 100

    # Peak outage hour
    peak_hour = hour_counts.loc[hour_counts["Number of Outages"].idxmax(), "Hour of Day"]

    # Insights bullets
    st.markdown(f"""
    * {most_affected_region} experienced the highest impact, accounting for {region_pct:.1f}% of all affected customers.
    * The most common cause was {most_common}, responsible for {percentage:.1f}% of all outages.
    * Outages peaked at hour {peak_hour} (24-hour format).
    * On average, each outage affected {int(filtered_df["estcustaffected"].mean())} customers.
    """)

    # Back to map button
    if st.button("Return to Map View"):
        st.session_state.show_report = False