import pandas as pd
from src.data import peak_hours

def test_peak_hours_counts():
    df = pd.DataFrame({
        "started_at": pd.to_datetime([
            "2021-01-01 08:00",
            "2021-01-01 08:30",
            "2021-01-01 09:00"
        ])
    })
    out = peak_hours(df)
    assert any((out['hour']==8)&(out['count']==2))
    assert any((out['hour']==9)&(out['count']==1))
