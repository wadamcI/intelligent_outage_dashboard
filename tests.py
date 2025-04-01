import streamlit as st
import sqlite3
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime
import time

# Set up Streamlit page layout - MUST BE FIRST!
st.set_page_config(layout="wide")

# Load data from SQLite
@st.cache_data
def load_data():
    conn = sqlite3.connect("outage_data.db")
    query = "SELECT id, latitude_x, longitude_x, datetime, estcustaffected, cause, regionname FROM outage_events"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert Unix time to datetime
    df["datetime"] = pd.to_datetime(df["datetime"], unit='s')
    return df


df = load_data()

# Sidebar: Time slider
st.sidebar.header("Time Filter")
min_time, max_time = df["datetime"].min().to_pydatetime(), df["datetime"].max().to_pydatetime()

selected_time = st.sidebar.slider("Select Time", min_value=min_time, max_value=max_time, value=min_time)

# Filter data based on selected time
filtered_df = df[df["datetime"] <= selected_time]

# Main content: Map
st.header("Outage Events Map")

m = folium.Map(location=[df["latitude_x"].mean(), df["longitude_x"].mean()], zoom_start=6)

for _, row in filtered_df.iterrows():
    folium.CircleMarker(
        location=[row["latitude_x"], row["longitude_x"]],
        radius=2,
        color="red",
        fill=True,
        fill_opacity=0.7,
        popup=f"Region: {row['regionname']}<br>Cause: {row['cause']}<br>Customers Affected: {row['estcustaffected']}",
    ).add_to(m)

st_folium(m, width=700, height=500)

# Chat Section (Placeholder for LLM)
st.sidebar.header("Chat with AI (Coming Soon)")
chat_input = st.sidebar.text_input("Ask something...")
if chat_input:
    st.sidebar.write(f"You asked: {chat_input}")
    # Placeholder for LLM response