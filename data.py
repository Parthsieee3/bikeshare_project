"""
data.py

Modular functions for loading, cleaning, feature-engineering and generating
analytical dataframes/aggregates for the Toronto Bike-Sharing dataset.

Design goals:
- Robust to minor schema changes (different column names).
- Clear, pure functions that return dataframes or small objects (plot-ready).
- Docstrings & typing for clarity and easy testing.
"""

from typing import Optional, Tuple, List
import pandas as pd
import numpy as np


"""
data.py - CLEAN, FINAL WORKING VERSION
"""

import pandas as pd
import numpy as np

# ====== CORRECT COLUMN MAP FOR YOUR DATASET ======
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


def load_data(path_or_file):
    """Load CSV safely."""
    df = pd.read_csv(path_or_file)
    return df


def clean_data(df):
    """Clean + rename columns + normalize types."""
    df = df.copy()

    # -------- APPLY FIXED COLUMN MAP --------
    rename_map = {}
    for col in df.columns:
        if col in _COLUMN_MAP:
            rename_map[col] = _COLUMN_MAP[col]
    df = df.rename(columns=rename_map)

    # -------- AUTO-DETECT missing station columns --------
    if "start_station_name" not in df.columns:
        poss = [c for c in df.columns if "start" in c.lower() and "station" in c.lower()]
        if poss:
            df = df.rename(columns={poss[0]: "start_station_name"})

    if "end_station_name" not in df.columns:
        poss = [c for c in df.columns if "end" in c.lower() and "station" in c.lower()]
        if poss:
            df = df.rename(columns={poss[0]: "end_station_name"})

    # -------- PARSE TIME COLUMN --------
    if "start_time" not in df.columns:
        poss = [c for c in df.columns if "start" in c.lower() and "time" in c.lower()]
        if poss:
            df = df.rename(columns={poss[0]: "start_time"})

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")

    # -------- DURATION MUST BE NUMERIC --------
    df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely add time-based features by auto-detecting the start time column.
    Works even if the original column name varies (Start Time, start_time, etc.)
    """
    df = df.copy()

    # STEP 1 — find ANY column containing both "start" and "time"
    time_cols = [c for c in df.columns if "start" in c.lower() and "time" in c.lower()]

    # If none found, show user EXACT column names to debug
    if not time_cols:
        raise KeyError(
            f"Could not find a start time column.\nAvailable columns are: {list(df.columns)}"
        )

    # STEP 2 — use the FIRST matching time column
    time_col = time_cols[0]

    # STEP 3 — parse it
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # STEP 4 — rename to standard column name
    df = df.rename(columns={time_col: "start_time"})

    # STEP 5 — add datetime features
    df["year"] = df["start_time"].dt.year
    df["month"] = df["start_time"].dt.month
    df["day"] = df["start_time"].dt.day
    df["hour"] = df["start_time"].dt.hour
    df["dayofweek"] = df["start_time"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6])

    return df


# ----- Analytics helper functions -----

def top_n_stations(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Auto-detect start station column and compute top N stations.
    Handles non-numeric duration values safely.
    """

    # Auto-detect start station column
    possible = [c for c in df.columns if "start" in c.lower() and "station" in c.lower()]
    if not possible:
        raise KeyError(f"Could not find start station column. Columns: {list(df.columns)}")

    station_col = possible[0]

    # Convert duration to numeric
    if "duration_seconds" in df.columns:
        df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")
    else:
        df["duration_seconds"] = None

    # Choose count column
    trip_col = "trip_id" if "trip_id" in df.columns else None

    # Compute group statistics
    if trip_col:
        grp = df.groupby(station_col).agg(
            trips=(trip_col, "count"),
            avg_duration_seconds=("duration_seconds", "mean")
        )
    else:
        grp = df.groupby(station_col).size().reset_index(name="trips")
        grp["avg_duration_seconds"] = None

    grp = grp.reset_index(drop=True)

    # Handle non-numeric duration
    grp["avg_duration_seconds"] = pd.to_numeric(grp["avg_duration_seconds"], errors="coerce")

    # Fill NaN with 0 for safety
    grp["avg_duration_seconds"] = grp["avg_duration_seconds"].fillna(0)

    # Round safely
    grp["avg_duration_seconds"] = grp["avg_duration_seconds"].round(1)

    # Standard column name
    grp = grp.rename(columns={station_col: "start_station_name"})

    # Sort and top N
    grp = grp.sort_values("trips", ascending=False).head(n)

    return grp


def station_flow(df: pd.DataFrame, station_name: str, top_k: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For a given station, returns:
    - outbound top destinations (from this station to others)
    - inbound top origins (from other stations to this station)
    """
    if "start_station_name" not in df.columns or "end_station_name" not in df.columns:
        raise KeyError("start_station_name and end_station_name required")
    out = df[df["start_station_name"] == station_name].groupby("end_station_name").size().reset_index(name="trips").sort_values("trips", ascending=False).head(top_k)
    inc = df[df["end_station_name"] == station_name].groupby("start_station_name").size().reset_index(name="trips").sort_values("trips", ascending=False).head(top_k)
    return out, inc

def duration_stats(df: pd.DataFrame) -> dict:
    """
    Basic trip duration stats: median, mean, percentiles, long-trip count.
    """
    if "duration_seconds" not in df.columns:
        return {}
    s = df["duration_seconds"].dropna().astype(float)
    stats = {
        "count": int(s.count()),
        "mean_s": float(s.mean()),
        "median_s": float(s.median()),
        "p90_s": float(s.quantile(0.9)),
        "p99_s": float(s.quantile(0.99)),
        "long_trips_count": int((s > 3600).sum()),  # >1 hour
    }
    return stats

def trips_over_time(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Returns time series of trip counts aggregated by given frequency (D, H, W, M).
    """
    if "start_time" not in df.columns:
        raise KeyError("start_time required")
    ts = df.set_index(pd.to_datetime(df["start_time"])).resample(freq).size().rename("trips").reset_index()
    return ts

def hourly_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns pivot table with dayofweek x hour counts for heatmap plotting.
    """
    if "hour" not in df.columns or "dayofweek" not in df.columns:
        df = add_time_features(df)
    pivot = df.groupby(["dayofweek", "hour"]).size().reset_index(name="trips")
    # Pivot to matrix form if a plotting library prefers it
    mat = pivot.pivot(index="dayofweek", columns="hour", values="trips").fillna(0).astype(int)
    return mat

def user_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary stats by user_type (counts, avg duration).
    Auto-detects user type column and duration column.
    """
    df = df.copy()

    # --- AUTO-DETECT user_type column ---
    if "user_type" not in df.columns:
        poss = [c for c in df.columns if "user" in c.lower()]
        if poss:
            df = df.rename(columns={poss[0]: "user_type"})

    # If still missing, return empty
    if "user_type" not in df.columns:
        return pd.DataFrame()

    # --- SAFE GROUPING ---
    trip_col = "trip_id" if "trip_id" in df.columns else df.columns[0]

    # Ensure duration numeric
    if "duration_seconds" in df.columns:
        df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")
    else:
        df["duration_seconds"] = np.nan

    out = df.groupby("user_type").agg(
        trips=(trip_col, "count"),
        avg_duration_s=("duration_seconds", "mean")
    ).reset_index()

    out["avg_duration_s"] = out["avg_duration_s"].fillna(0).round(1)
    out = out.sort_values("trips", ascending=False)

    return out

def get_station_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely returns station coordinates if available.
    If the dataset contains no lat/lon columns, returns an empty
    dataframe with the correct columns so that Streamlit does not break.
    """
    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lng_cols = [c for c in df.columns if "lng" in c.lower() or "lon" in c.lower()]

    # If no coords exist → return safe empty DF
    if not lat_cols or not lng_cols:
        return pd.DataFrame(columns=["start_station_name", "lat", "lon"])

    # Otherwise use the first matching pair
    lat = lat_cols[0]
    lng = lng_cols[0]

    s = df.loc[:, ["start_station_name", lat, lng]].dropna()
    s = s.drop_duplicates(subset=["start_station_name"])
    s = s.rename(columns={lat: "lat", lng: "lon"})

    return s


# A convenience function returning everything an app may want
def full_analytics(df: pd.DataFrame) -> dict:
    dfc = clean_data(df)
    dfc = add_time_features(dfc)
    return {
        "clean": dfc,
        "top_stations": top_n_stations(dfc, n=15),
        "duration_stats": duration_stats(dfc),
        "trips_daily": trips_over_time(dfc, freq="D"),
        "hourly_heatmap": hourly_heatmap_data(dfc),
        "user_summary": user_type_summary(dfc),
    }

# If executed as script, quick CLI preview
if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "financial_transactions (2).csv"
    df = load_data(p, nrows=2000)
    out = full_analytics(df)
    print("Top stations sample:")
    print(out["top_stations"].head())
    print("Duration stats:", out["duration_stats"])
