import pytest
from unittest.mock import patch
import pandas as pd
import shutil
from tempfile import mkdtemp

from daytradeai.data import get_downloaded_data


@pytest.fixture
def temp_parquet_dir():
    """Creates a temporary directory with Parquet test files and yields its path."""
    temp_dir = mkdtemp()
    try:
        yield temp_dir  # Passes the temp directory path to the test
    finally:
        shutil.rmtree(temp_dir)  # Clean up after test


@pytest.fixture
def df1_test_data():
    """Creates sample test data with a DateTime index and XYZ column."""
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "XYZ": [1.1, 2.2, 3.3]
    }).set_index("Date")


@pytest.fixture
def df2_test_data():
    """Creates 2nd test data to combine with data1, has one overlap index."""
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
        "XYZ": [3.3, 4.0]
    }).set_index("Date")


@pytest.fixture
def df_expected_combined_data():
    """Returns the expected data after combining parquet_test_data1 and parquet_test_data2."""
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "XYZ": [1.1, 2.2, 3.3, 4.0]
    }).set_index("Date")


def test_get_download_data(temp_parquet_dir, df1_test_data, df2_test_data, df_expected_combined_data):
    """Tests the get_downloaded_data function with Parquet files."""
    # Create Parquet files
    df1_test_data.to_parquet(f"{temp_parquet_dir}/file1.parquet")
    df2_test_data.to_parquet(f"{temp_parquet_dir}/file2.parquet")

    with patch("daytradeai.data.get_stock_download_dir", return_value=temp_parquet_dir):
        df_result = get_downloaded_data(cfg=dict())
        if df_result is None:
            pytest.fail("get_downloaded_data returned None")
        pd.testing.assert_frame_equal(df_result, df_expected_combined_data)
