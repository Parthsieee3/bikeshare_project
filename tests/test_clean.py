import pandas as pd
from src.data import clean_data

def test_clean_parses_dates_and_numeric():
    df = pd.DataFrame({
        "started_at": ["2021-01-01 08:00:00", None],
        "duration_minutes": ["10","20"]
    })
    c = clean_data(df)
    assert str(c['started_at'].dtype).startswith("datetime")
    assert c['duration_minutes'].dtype.kind in ('i','f')
