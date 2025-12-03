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

def peak_hours(df: pd.DataFrame, datetime_column='started_at'):
    if datetime_column not in df.columns:
        raise KeyError(f"{datetime_column} not in DataFrame")
    hours=df[datetime_column].dropna().dt.hour
    result=hours.value_counts().sort_index()
    return result.rename_axis("hour").reset_index(name="count")

def busiest_stations(df: pd.DataFrame, col='start_station_name', top_n=10):
    if col not in df.columns:
        raise KeyError(f"{col} missing")
    result=df[col].value_counts().head(top_n)
    return result.rename_axis(col).reset_index(name="count")
