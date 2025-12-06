
import pickle
import pandas as pd
import streamlit as st
import joblib

# ================================
# LOAD ML MODEL SAFELY
# ================================
@st.cache_resource
def load_ml_model():
    try:
        return joblib.load("model.pkl")   # <-- change path if needed
    except:
        return None

model = load_ml_model()



# ================================================================
# LOAD DATA
# ================================================================
def load_data(path_or_buffer):
    return pd.read_csv(path_or_buffer)




# ================================================================
# CLEAN + STANDARDIZE COLUMN NAMES
# ================================================================
def clean_data(df: pd.DataFrame):
    df = df.copy()

    # Standardize column names (lowercase + replace spaces with underscores)
    df.columns = (
        df.columns.str.strip()
                         .str.lower()
                         .str.replace(" ", "_")
                         .str.replace("-", "_")
    )

    # Rename known columns from your dataset → standard names
    rename_map = {
        "trip_id": "trip_id",
        "trip_duration": "duration",
        "trip__duration": "duration",
        "start_station_id": "start_station_id",
        "start_station_name": "start_station_name",
        "start_time": "start_time",
        "started_at": "start_time",
        "end_station_id": "end_station_id",
        "end_station_name": "end_station_name",
        "end_time": "end_time",
        "bike_id": "bike_id",
        "user_type": "user_type",
        "model": "model",
    }

    df = df.rename(columns=rename_map)

    # Convert datetime columns
    datetime_cols = ["start_time", "end_time"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convert duration to numeric
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

        # Convert seconds → minutes (if needed)
        if df["duration"].max() > 120:
            df["duration"] = df["duration"] / 60.0

    # Create hour column
    if "start_time" in df.columns:
        df["hour"] = df["start_time"].dt.hour

    return df


# ================================================================
# PEAK HOURS
# ================================================================
def peak_hours(df: pd.DataFrame):
    if "start_time" not in df.columns:
        raise KeyError("start_time column missing.")

    hours = df["start_time"].dropna().dt.hour
    result = hours.value_counts().sort_index()
    return result.rename_axis("hour").reset_index(name="count")


# ================================================================
# PEAK DAYS
# ================================================================
def peak_days(df: pd.DataFrame):
    if "start_time" not in df.columns:
        raise KeyError("start_time column missing.")

    df["weekday"] = df["start_time"].dt.day_name()
    return df["weekday"].value_counts()


# ================================================================
# BUSIEST STATIONS
# ================================================================
def busiest_stations(df: pd.DataFrame, col="start_station_name", top_n=10):
    if col not in df.columns:
        raise KeyError(f"{col} column not found.")
    result = df[col].value_counts().head(top_n)
    return result.rename_axis(col).reset_index(name="count")


# ================================================================
# DURATION DIST
# ================================================================
def trip_duration_distribution(df: pd.DataFrame):
    if "duration" not in df.columns:
        raise KeyError("duration column missing.")

    durations = df["duration"].dropna()
    return durations.describe()


# ================================================================+
# BIKE USAGE
# ================================================================+
def bike_usage(df: pd.DataFrame):
    if "bike_id" not in df.columns:
        raise KeyError("bike_id column missing.")

    usage = df["bike_id"].value_counts().head(20)
    return usage.rename_axis("bike_id").reset_index(name="count")



# ================================================================+
# USER TYPE COUNTS
# ================================================================+
def user_type_counts(df: pd.DataFrame):
    if "user_type" not in df.columns:
        raise KeyError("user_type column missing.")
    return df["user_type"].value_counts()
