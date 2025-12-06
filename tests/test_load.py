import pytest
import pandas as pd
from src.data import load_data

def test_load_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        load_data("missing.csv")

def test_load_valid_csv(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_text("a,b\n1,2")
    df = load_data(str(p))
    assert isinstance(df, pd.DataFrame)
    
    assert df.shape == (1,2)
