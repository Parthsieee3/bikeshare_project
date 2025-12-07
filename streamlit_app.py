"""
app.py

Streamlit dashboard for Toronto Bike-Sharing Analytics Tool.

Run:
    streamlit run app.py

Features:
- Upload CSV (or select the uploaded file path)
- Robust parsing using data.py
- Professional layout: sidebar controls, top KPIs, interactive charts (Plotly), pydeck map
- Insights panel that computes and displays textual observations
- Export filtered dataset
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
from data import load_data, clean_data, add_time_features, full_analytics, top_n_stations, station_flow, trips_over_time, user_type_summary, hourly_heatmap_data, get_station_coords, duration_stats

# --- Streamlit page config ---
st.set_page_config(page_title="Toronto Bike-Sharing Analytics", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar ---
st.sidebar.title("Data & Controls")
uploaded = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls", "parquet"])
use_sample = st.sidebar.checkbox("Load sample dataset (first 5k rows)", value=True)
agg_freq = st.sidebar.selectbox("Time aggregation for trend", options=["D", "W", "M", "H"], index=0,
                                format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly", "H": "Hourly"}[x])
top_n = st.sidebar.slider("Top N stations to show", min_value=5, max_value=25, value=10)
station_select = st.sidebar.text_input("Inspect station (type name)", value="")
download_button = st.sidebar.checkbox("Show download filtered dataset", value=True)

# --- Load data safely ---
@st.cache_data(show_spinner=False)
def load_and_prep(path_or_file, use_sample_flag: bool):
    if path_or_file is None:
        return None
    if hasattr(path_or_file, "read"):  # uploaded file-like object
        # read bytes into pandas (best attempt)
        try:
            df = pd.read_csv(path_or_file)
        except Exception:
            path_or_file.seek(0)
            df = pd.read_excel(path_or_file)
    else:
        df = load_data(path_or_file)
    df = clean_data(df)
    df = add_time_features(df)
    return df


if uploaded is None and use_sample:
    st.sidebar.info("No file uploaded — will attempt to load sample file from disk if available.")
    try:
        sample_path = "financial_transactions (2).csv"
        df = load_and_prep(sample_path, use_sample_flag=True)
   # 💥 DEBUG: SHOW COLUMN NAMES
        st.write("DEBUG COLUMNS:", df.columns.tolist())

    except Exception as e:
        st.sidebar.error(f"Could not load sample file  automatically: {e}")
        df = None
else:
    df = load_and_prep(uploaded, use_sample_flag=use_sample)

# --- Page header ---
st.title("Toronto Bike-Sharing Analytics")
st.markdown("""
Professional exploratory dashboard for the Bike Share Toronto dataset.
Use the sidebar to upload a CSV, adjust aggregation and inspect stations.
""")

if df is None or df.empty:
    st.warning("No data available yet. Upload a CSV or enable the sample dataset in the sidebar.")
    st.stop()

# --- KPIs row ---
left, middle, right = st.columns([1,1,1])
stats = duration_stats(df)
left.metric("Total Trips", f"{len(df):,}")
middle.metric("Median Trip (min)", f"{(stats.get('median_s',0)/60):.1f}")
right.metric("Long Trips (>1hr)", f"{stats.get('long_trips_count',0):,}")

# --- Top stations table & chart ---
st.subheader(f"Top {top_n} Start Stations")
top_st = top_n_stations(df, n=top_n)
col1, col2 = st.columns([1,1])
with col1:
    st.dataframe(top_st.reset_index(drop=True), use_container_width=True)
with col2:
    fig_bar = px.bar(top_st.sort_values("trips"), x="trips", y="start_station_name", orientation="h", labels={"start_station_name":"Station","trips":"Trips"}, title="Top Start Stations")
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Trips over time ---
st.subheader("Trips Over Time")
ts = trips_over_time(df, freq=agg_freq)
fig_ts = px.line(ts, x=ts.columns[0], y="trips", labels={ts.columns[0]:"Time", "trips":"Trips"}, title="Trips Over Time")
st.plotly_chart(fig_ts, use_container_width=True)

# --- Hourly heatmap (dayofweek x hour) ---
st.subheader("Weekday/Hour Heatmap")
heat = hourly_heatmap_data(df)
# Convert pivot to long form for plotting
heat_long = heat.reset_index().melt(id_vars="dayofweek", var_name="hour", value_name="trips")
# Map dayofweek numbers to names
dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
heat_long["day"] = heat_long["dayofweek"].map(dow_map)
fig_heat = px.density_heatmap(heat_long, x="hour", y="day", z="trips", histfunc="avg", title="Trips by Day of Week and Hour")
st.plotly_chart(fig_heat, use_container_width=True)

# --- User type summary ---
st.subheader("User Type Comparison")
ut = user_type_summary(df)
fig_ut = px.bar(ut, x="user_type", y="trips", text="avg_duration_s", title="Trips by User Type (text shows avg duration s)")
st.plotly_chart(fig_ut, use_container_width=True)
st.dataframe(ut, use_container_width=True)

# --- Station flow inspector ---
if station_select:
    st.subheader(f"Flows for station: {station_select}")
    try:
        out, inc = station_flow(df, station_select, top_k=10)
        c1, c2 = st.columns(2)
        c1.markdown("**Top outbound destinations**")
        c1.dataframe(out)
        c2.markdown("**Top inbound origins**")
        c2.dataframe(inc)
    except KeyError:
        st.info("Station flow requires `start_station_name` and `end_station_name` columns.")

# --- Map (if coords exist) ---
st.subheader("Station Map (if lat/lon exists)")
coords = get_station_coords(df)
if not coords.empty:
    st.write("Station sample (click to expand):")
    st.dataframe(coords.head(), use_container_width=True)
    # pydeck view
    view_state = pdk.ViewState(latitude=coords["lat"].mean(), longitude=coords["lon"].mean(), zoom=11, pitch=0)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=coords,
        get_position=["lon", "lat"],
        get_radius=70,
        pickable=True,
        auto_highlight=True
    )
    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text":"{start_station_name}"})
    st.pydeck_chart(deck)
else:
    st.info("No lat/lon columns found. If your dataset includes coordinates, ensure the column names contain 'lat' and 'lng' or 'lon'.")

# --- Insights / auto-observations ---
st.subheader("Automated Insights")
with st.expander("Click to expand computed insights"):
    ds = duration_stats(df)
    st.markdown(f"- Dataset contains **{len(df):,}** trips (after cleaning).")
    st.markdown(f"- Median trip length ≈ **{ds.get('median_s',0)/60:.1f} minutes**; mean ≈ **{ds.get('mean_s',0)/60:.1f} minutes**.")
    st.markdown(f"- Top station by trips: **{top_st.iloc[0]['start_station_name']}** with **{int(top_st.iloc[0]['trips']):,}** starts.")
    st.markdown(f"- Number of long trips (>1 hour): **{ds.get('long_trips_count',0):,}**.")
    # Peak hour
    peak_hour = df.groupby("hour").size().idxmax() if "hour" in df.columns else None
    if peak_hour is not None:
        st.markdown(f"- Peak starting hour: **{int(peak_hour)}:00**.")
    # user comparison
    if not ut.empty:
        top_user = ut.iloc[0]
        st.markdown(f"- Most active user type: **{top_user['user_type']}** with **{int(top_user['trips']):,}** trips and avg duration **{top_user['avg_duration_s']:.1f}s**.")
    st.markdown("**Suggested next analyses:** clustering of stations by flows, day-of-week seasonal decomposition, and building predictive model for demand by hour/station.")

# --- Download filtered dataset ---
if download_button:
    st.subheader("Export")
    st.write("Download the cleaned dataset for further analysis or submission.")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download cleaned CSV", csv, "cleaned_bikeshare.csv", "text/csv")

st.sidebar.markdown("---")


