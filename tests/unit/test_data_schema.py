import pytest
import pandas as pd
from scripts.config import RAW_DATA

EXPECTED_COLUMNS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "yr_built", "yr_renovated", "zipcode",
    "lat", "long", "sqft_living15", "sqft_lot15", "price"
]


@pytest.mark.unit
def test_raw_data_columns():
    df = pd.read_csv(RAW_DATA)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing columns in raw data: {missing}"
