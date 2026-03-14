"""Tests for encode(method='datetime') —"""

import numpy as np
import pandas as pd

import ml


def test_encode_datetime_basic():
    """encode(method='datetime') extracts date components."""
    data = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-15", "2024-03-22", "2024-07-04", "2023-12-25"]),
        "value": [1, 2, 3, 4],
    })
    enc = ml.encode(data, columns=["date"], method="datetime")
    result = enc.transform(data)
    assert "date_year" in result.columns
    assert "date_month" in result.columns
    assert "date_day" in result.columns
    assert "date_dayofweek" in result.columns
    assert "date" not in result.columns  # original dropped


def test_encode_datetime_is_weekend():
    """datetime encoder creates is_weekend flag."""
    # 2024-01-06 = Saturday, 2024-01-07 = Sunday, 2024-01-08 = Monday
    data = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-06", "2024-01-07", "2024-01-08"]),
        "value": [1, 2, 3],
    })
    enc = ml.encode(data, columns=["date"], method="datetime")
    result = enc.transform(data)
    assert "date_is_weekend" in result.columns
    assert result.loc[result.index[0], "date_is_weekend"] == 1  # Saturday
    assert result.loc[result.index[1], "date_is_weekend"] == 1  # Sunday
    assert result.loc[result.index[2], "date_is_weekend"] == 0  # Monday


def test_encode_datetime_hour_extraction():
    """datetime encoder extracts hour when timestamps have non-midnight hours."""
    data = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-01 08:30:00", "2024-01-02 14:00:00", "2024-01-03 22:45:00"]),
        "value": [1, 2, 3],
    })
    enc = ml.encode(data, columns=["ts"], method="datetime")
    result = enc.transform(data)
    assert "ts_hour" in result.columns


def test_encode_datetime_no_hour_for_date_only():
    """datetime encoder omits hour column when all times are midnight."""
    data = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-15", "2024-03-22", "2024-07-04"]),
        "x": [1.0, 2.0, 3.0],
    })
    enc = ml.encode(data, columns=["date"], method="datetime")
    result = enc.transform(data)
    assert "date_hour" not in result.columns


def test_encode_datetime_numeric_types():
    """datetime encoder transform produces numeric columns."""
    data = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-15", "2024-03-22", "2024-07-04"]),
        "x": [1.0, 2.0, 3.0],
    })
    enc = ml.encode(data, columns=["date"], method="datetime")
    result = enc.transform(data)
    for col in ["date_year", "date_month", "date_day"]:
        assert result[col].dtype in [np.int32, np.int64, np.float64, "float64"]


def test_encode_datetime_save_load(tmp_path):
    """datetime encoder survives save/load round-trip."""
    data = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-15", "2024-03-22"]),
        "value": [1, 2],
    })
    enc = ml.encode(data, columns=["date"], method="datetime")
    path = str(tmp_path / "enc.ml")
    ml.save(enc, path)
    enc2 = ml.load(path)
    result = enc2.transform(data)
    assert "date_year" in result.columns
    assert "date_month" in result.columns
    assert "date" not in result.columns


def test_encode_datetime_correct_values():
    """datetime encoder extracts correct year/month/day values."""
    data = pd.DataFrame({
        "date": pd.to_datetime(["2024-07-04"]),
        "x": [1.0],
    })
    enc = ml.encode(data, columns=["date"], method="datetime")
    result = enc.transform(data)
    assert result["date_year"].iloc[0] == 2024.0
    assert result["date_month"].iloc[0] == 7.0
    assert result["date_day"].iloc[0] == 4.0
    # July 4, 2024 is a Thursday (dayofweek=3)
    assert result["date_dayofweek"].iloc[0] == 3.0


def test_encode_datetime_multiple_columns():
    """datetime encoder handles multiple datetime columns."""
    data = pd.DataFrame({
        "created": pd.to_datetime(["2024-01-01", "2024-06-15"]),
        "updated": pd.to_datetime(["2024-02-01", "2024-07-20"]),
        "value": [10, 20],
    })
    enc = ml.encode(data, columns=["created", "updated"], method="datetime")
    result = enc.transform(data)
    assert "created_year" in result.columns
    assert "updated_year" in result.columns
    assert "created" not in result.columns
    assert "updated" not in result.columns


def test_encode_datetime_in_valid_methods():
    """datetime is a valid method name for encode()."""
    data = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-15", "2024-03-22"]),
        "value": [1, 2],
    })
    # Should not raise ConfigError
    enc = ml.encode(data, columns=["date"], method="datetime")
    assert enc.method == "datetime"
