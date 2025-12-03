import streamlit as st
import pandas as pd
from src.data import load_data, clean_data, peak_hours, busiest_stations
import matplotlib.pyplot as plt

st.title("Toronto Bike-Sharing Dashboard")
file = st.file_uploader("Upload CSV", type="csv")

if file:
    df = load_data(file)
    df = clean_data(df)
    st.write("Cleaned Data Preview", df.head())

    st.subheader("Peak Hours")
    ph = peak_hours(df)
    fig, ax = plt.subplots()
    ax.bar(ph['hour'], ph['count'])
    st.pyplot(fig)

    st.subheader("Busiest Stations")
    if "start_station_name" in df.columns:
        st.table(busiest_stations(df))
    else:
        st.info("start_station_name column missing")
