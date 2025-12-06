import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import folium
import joblib
from src.data import model
from streamlit_folium import st_folium

from src.data import (
    load_data,
    clean_data,
    peak_hours,
    busiest_stations,
    trip_duration_distribution,
    bike_usage,
    user_type_counts
)

st.set_page_config(page_title="Toronto Bike Sharing Dashboard", layout="wide")

# ===============================================================
# UI STYLING (Dark Theme)
# ===============================================================
st.markdown("""
<style>
    body { background-color: #0d1117; }
    .stApp { background-color: #0d1117; }
    h1, h2, h3, h4, h5 { color: #58a6ff !important; }
    .stMarkdown, p { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🚲 Toronto Bike Sharing — Advanced Analytics Dashboard")


# ===============================================================
# UPLOAD FILE
# ===============================================================
file = st.sidebar.file_uploader("📤 Upload CSV", type="csv")

if not file:
    st.info("Upload a CSV file from the left sidebar to continue.")
    st.stop()

df = clean_data(load_data(file))
st.success("Dataset Loaded Successfully!")


# ===============================================================
# SIDEBAR FILTERS
# ===============================================================
st.sidebar.header("🔍 Filters")

# Date filter
if "start_time" in df.columns:
    min_date, max_date = df["start_time"].min(), df["start_time"].max()
    date_range = st.sidebar.date_input("📅 Date range:", [min_date, max_date])
    if len(date_range) == 2:
        start, end = date_range
        df = df[(df["start_time"] >= pd.to_datetime(start)) &
                (df["start_time"] <= pd.to_datetime(end))]


# User type filter
if "user_type" in df.columns:
    selected_users = st.sidebar.multiselect(
        "👤 User Types:",
        df["user_type"].unique(),
        default=list(df["user_type"].unique())
    )
    df = df[df["user_type"].isin(selected_users)]


# ======================================================================================
# 1. DATA PREVIEW
# ======================================================================================
st.header("📊 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)


# ======================================================================================
# 2. PEAK HOUR & DAY ANALYTICS
# ======================================================================================
st.header("⏰ Peak Time Analytics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Peak Hours")
    try:
        ph = peak_hours(df)
        fig, ax = plt.subplots()
        ax.bar(ph['hour'], ph['count'])
        ax.set_xlabel("Hour")
        ax.set_ylabel("Trips")
        ax.set_title("Trips Per Hour of Day")
        st.pyplot(fig)
    except Exception as e:
        st.warning(str(e))

with col2:
    st.subheader("Peak Days")
    try:
        df["weekday"] = df["start_time"].dt.day_name()
        day_counts = df["weekday"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(day_counts.index, day_counts.values)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    except:
        st.warning("Cannot compute peak days.")


# ======================================================================================
# 3. STATION ANALYTICS
# ======================================================================================
st.header("🚉 Station Analytics")
colA, colB = st.columns(2)

with colA:
    st.subheader("Top 10 Busiest Start Stations")
    try:
        st.dataframe(busiest_stations(df))
    except Exception as e:
        st.warning(str(e))

with colB:
    st.subheader("Top 10 Busiest End Stations")
    try:
        end_stats = df["end_station_name"].value_counts().head(10)
        st.dataframe(end_stats)
    except:
        st.warning("End stations unavailable.")


# ======================================================================================
# 4. GEOLOCATION MAP (Folium)
# ======================================================================================
st.header("🗺 Geolocation Map")

if {"start_station_name", "start_station_id"}.issubset(df.columns):

    # Create a base map
    toronto_map = folium.Map(location=[43.651070, -79.347015], zoom_start=12)

    # Add station markers (if coordinates available — otherwise random example)
    if "lat" in df.columns and "lng" in df.columns:
        for _, row in df.iterrows():
            folium.Marker(
                [row["lat"], row["lng"]],
                popup=row["start_station_name"],
            ).add_to(toronto_map)
    else:
        folium.Marker([43.65, -79.38], popup="Sample Marker").add_to(toronto_map)

    st_folium(toronto_map, width=900, height=500)
else:
    st.warning("Geolocation columns (lat/lng) missing. Map shown with placeholder marker.")


# ======================================================================================
# 5. HEATMAP (Seaborn)
# ======================================================================================
st.header("🔥 Heatmap — Hour vs Weekday Activity")

try:
    df["weekday_num"] = df["start_time"].dt.weekday
    pivot = df.pivot_table(index="weekday_num", columns="hour", values="trip_id", aggfunc="count")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, cmap="viridis", ax=ax)
    st.pyplot(fig)
except:
    st.warning("Heatmap cannot be generated — missing datetime columns.")


# ======================================================================================
# 6. ML PREDICTION — TRIP DURATION
# ======================================================================================
st.header("🤖 Machine Learning — Predict Trip Duration")

if model:
    # --- User Inputs ---
    st.subheader("Enter Trip Details:")
    start_hour = st.number_input("Start hour (0-23)", min_value=0, max_value=23, value=8)
    distance_km = st.number_input("Trip distance (km)", min_value=0.0, value=5.0)
    day_of_week = st.selectbox("Day of the week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    
    # Convert categorical input to numerical if needed
    day_mapping = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
    day_num = day_mapping[day_of_week]

    # --- Prepare input for prediction ---
    input_data = pd.DataFrame({
        "start_hour":[start_hour],
        "distance_km":[distance_km],
        "day_of_week":[day_num]
    })

    # --- Predict on button click ---
    if st.button("Predict Trip Duration"):
        try:
            prediction = model.predict(input_data)
            st.success(f"Predicted Trip Duration: {prediction[0]:.2f} minutes")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.warning("ML model is not loaded. Cannot predict trip duration.")


# ======================================================================================
# 7. USER TYPE BREAKDOWN & BIKE USAGE
# ======================================================================================
st.header("🧍 User Demographics")

try:
    user_counts = user_type_counts(df)
    fig, ax = plt.subplots()
    ax.pie(user_counts.values, labels=user_counts.index, autopct="%1.1f%%")
    st.pyplot(fig)
except:
    st.warning("User type breakdown unavailable.")

st.header("🚲 Bike Usage")
try:
    st.dataframe(bike_usage(df))
except:
    st.warning("Bike usage unavailable.")
