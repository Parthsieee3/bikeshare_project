"""
data.py — FINAL VERSION (100% compatible with your Streamlit app)
"""

import pandas as pd
import numpy as np


# =========================
# Column Normalization Map
# =========================
_COLUMN_MAP = {
    "Trip Id": "trip_id",
    "Trip  Duration": "duration_seconds",
    "Start Station Id": "start_station_id",
    "Start Time": "start_time",
    "Start Station Name": "start_station_name",
    "End Station Id": "end_station_id",
    "End Time": "end_time",
    "End Station Name": "end_station_name",
    "Bike Id": "bike_id",
    "User Type": "user_type",
    "Model": "model",
}


# ================
# Loading
# ================
def load_data(path):
    return pd.read_csv(path)


# ================
# Cleaning
# ================
def clean_data(df):
    df = df.copy()

    # Apply renaming
    rename_map = {c: _COLUMN_MAP[c] for c in df.columns if c in _COLUMN_MAP}
    df = df.rename(columns=rename_map)

    # Auto-detect fallback names
    for col, key in [
        ("start_station_name", ["start station"]),
        ("end_station_name", ["end station"]),
        ("user_type", ["user"]),
    ]:
        if col not in df.columns:
            for raw in df.columns:
                if any(k in raw.lower() for k in key):
                    df = df.rename(columns={raw: col})

    # Convert datetime
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    if "end_time" in df.columns:
        df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")

    # Convert duration to numeric
    if "duration_seconds" in df.columns:
        df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")

    return df


# ========================
# Time-based features
# ========================
def add_time_features(df):
    if "start_time" not in df.columns:
        return df.copy()

    df = df.copy()
    df["year"] = df["start_time"].dt.year
    df["month"] = df["start_time"].dt.month
    df["day"] = df["start_time"].dt.day
    df["hour"] = df["start_time"].dt.hour
    df["dayofweek"] = df["start_time"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6])
    return df


# ========================
# Duration Statistics
# ========================
def duration_stats(df):
    """Return dict with required KPI statistics."""
    if "duration_seconds" not in df.columns:
        return {"median_s": 0, "mean_s": 0, "long_trips_count": 0}

    d = pd.to_numeric(df["duration_seconds"], errors="coerce").dropna()

    return {
        "median_s": d.median(),
        "mean_s": d.mean(),
        "long_trips_count": (d > 3600).sum(),
    }


# ========================
# Top Stations
# ========================
def top_n_stations(df, n=10):
    if "start_station_name" not in df.columns:
        return pd.DataFrame()

    df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")

    grp = df.groupby("start_station_name").agg(
        trips=("trip_id", "count"),
        avg_duration_seconds=("duration_seconds", "mean"),
    ).reset_index()

    grp["avg_duration_seconds"] = grp["avg_duration_seconds"].fillna(0).round(1)
    return grp.sort_values("trips", ascending=False).head(n)


# ========================
# User Type Summary
# ========================
def user_type_summary(df):
    if "user_type" not in df.columns:
        return pd.DataFrame()

    df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")

    group_key = "trip_id" if "trip_id" in df.columns else df.columns[0]

    out = df.groupby("user_type").agg(
        trips=(group_key, "count"),
        avg_duration_s=("duration_seconds", "mean"),
    ).reset_index()

    out["avg_duration_s"] = out["avg_duration_s"].fillna(0).round(1)
    return out.sort_values("trips", ascending=False)


# ========================
# Trips Over Time
# ========================
def trips_over_time(df, freq="D"):
    if "start_time" not in df.columns:
        return pd.DataFrame()
    return df.set_index("start_time").resample(freq).size().rename("trips").reset_index()


# ========================
# Hourly Heatmap Data
# ========================
def hourly_heatmap_data(df):
    """
    Returns a pivoted matrix of trips by day-of-week (rows) and hour (columns).
    This ensures Streamlit's melt() step works without errors.
    """
    if "hour" not in df.columns or "dayofweek" not in df.columns:
        return pd.DataFrame()

    # Group
    grp = df.groupby(["dayofweek", "hour"]).size().reset_index(name="trips")

    # Pivot to wide format
    pivot = grp.pivot(index="dayofweek", columns="hour", values="trips").fillna(0)

    return pivot



# ========================
# Station Flow (Outbound + Inbound)
# ========================
def station_flow(df, station_name, top_k=10):
    if "start_station_name" not in df.columns or "end_station_name" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    # Outbound
    outbound = (
        df[df["start_station_name"].str.lower() == station_name.lower()]
        .groupby("end_station_name")
        .size()
        .reset_index(name="trips")
        .sort_values("trips", ascending=False)
        .head(top_k)
    )

    # Inbound
    inbound = (
        df[df["end_station_name"].str.lower() == station_name.lower()]
        .groupby("start_station_name")
        .size()
        .reset_index(name="trips")
        .sort_values("trips", ascending=False)
        .head(top_k)
    )

    return outbound, inbound


# ========================
# Coordinates
# ========================
def get_station_coords(df):
    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lon_cols = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]

    if not lat_cols or not lon_cols:
        return pd.DataFrame(columns=["start_station_name", "lat", "lon"])

    lat = lat_cols[0]
    lon = lon_cols[0]
    out = df[["start_station_name", lat, lon]].dropna().drop_duplicates()
    return out.rename(columns={lat: "lat", lon: "lon"})


# ========================
# Full Analytics Bundle
# ========================
def full_analytics(df):
    return {
        "top_stations": top_n_stations(df),
        "user_summary": user_type_summary(df),
        "trends": trips_over_time(df),
        "heatmap": hourly_heatmap_data(df),
        "duration_stats": duration_stats(df),
    }
