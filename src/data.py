import pandas as pd

def load_data(path_or_buffer):
    return pd.read_csv(path_or_buffer)

def clean_data(df: pd.DataFrame):
    df = df.dropna(how="all").copy()
    for col in ['started_at','ended_at','start_time','end_time']:
        if col in df.columns:
            df[col]=pd.to_datetime(df[col], errors='coerce')
    for col in df.columns:
        if any(k in col.lower() for k in ['duration','distance','trip']):
            df[col]=pd.to_numeric(df[col], errors='coerce')
    return df

def peak_hours(df, datetime_column=None):

    # List of possible datetime column names used in bike-share datasets
    possible_columns = ["started_at", "start_time", "Start Time", "STARTTIME"]

    # Auto-detect the column if not provided
    if datetime_column is None:
        for col in possible_columns:
            if col in df.columns:
                datetime_column = col
                break

    # If still not found, raise error
    if datetime_column is None:
        raise KeyError("No valid datetime column found. Expected one of: " +
                       ", ".join(possible_columns))

    # Convert to datetime
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')

    # Extract hour
    df["hour"] = df[datetime_column].dt.hour

    # Group by hour
    return df.groupby("hour").size().reset_index(name="trips")


def busiest_stations(df: pd.DataFrame, col='start_station_name', top_n=10):
    if col not in df.columns:
        raise KeyError(f"{col} missing")
    result=df[col].value_counts().head(top_n)
    return result.rename_axis(col).reset_index(name="count")
