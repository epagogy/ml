"""Tests for ml.profile() — data profiling."""

import pandas as pd
import pytest

import ml


def test_profile_basic():
    """profile() returns dict with shape and columns."""
    data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    prof = ml.profile(data=data)

    assert isinstance(prof, dict)
    assert prof["shape"] == (3, 2)
    assert prof["target"] is None
    assert prof["task"] is None
    assert "columns" in prof
    assert "warnings" in prof
    assert len(prof["columns"]) == 2


def test_profile_numeric_stats():
    """Numeric columns get mean, std, min, max, median."""
    data = pd.DataFrame({"age": [20, 30, 40, 50, 60]})
    prof = ml.profile(data=data)

    col_stats = prof["columns"]["age"]
    assert col_stats["dtype"] == "int64"
    assert col_stats["missing"] == 0
    assert col_stats["missing_pct"] == 0.0
    assert col_stats["unique"] == 5
    assert col_stats["mean"] == 40.0
    assert col_stats["std"] == pytest.approx(15.811, abs=0.01)
    assert col_stats["min"] == 20.0
    assert col_stats["max"] == 60.0
    assert col_stats["median"] == 40.0


def test_profile_categorical_stats():
    """Categorical columns get top, top_freq, unique."""
    data = pd.DataFrame({"state": ["CA", "CA", "NY", "TX", "CA"]})
    prof = ml.profile(data=data)

    col_stats = prof["columns"]["state"]
    assert col_stats["dtype"] in ("object", "str", "string")
    assert col_stats["unique"] == 3
    assert col_stats["top"] == "CA"
    assert col_stats["top_freq"] == 3
    # No mean/std for categorical
    assert "mean" not in col_stats
    assert "std" not in col_stats


def test_profile_missing_values():
    """Missing values counted correctly."""
    data = pd.DataFrame(
        {
            "a": [1, 2, None, 4, 5],
            "b": [None, None, 3, 4, 5],
        }
    )
    prof = ml.profile(data=data)

    assert prof["columns"]["a"]["missing"] == 1
    assert prof["columns"]["a"]["missing_pct"] == 20.0
    assert prof["columns"]["b"]["missing"] == 2
    assert prof["columns"]["b"]["missing_pct"] == 40.0


def test_profile_with_target_classification():
    """With classification target, adds task, distribution, balance."""
    data = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": ["A", "A", "A", "A", "A", "A", "A", "B", "B", "B"],
        }
    )
    prof = ml.profile(data=data, target="y")

    assert prof["target"] == "y"
    assert prof["task"] == "classification"
    assert prof["target_distribution"] == {"A": 7, "B": 3}
    assert prof["n_classes"] == 2
    assert prof["target_balance"] == 0.3  # minority class B


def test_profile_with_target_regression():
    """With regression target, task=regression, no distribution."""
    data = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "price": [100.5, 200.3, 150.2, 300.1, 250.9],
        }
    )
    prof = ml.profile(data=data, target="price")

    assert prof["target"] == "price"
    assert prof["task"] == "regression"
    assert "target_distribution" not in prof
    assert "target_balance" not in prof


def test_profile_warnings_missing():
    """Warning generated for columns with missing data."""
    data = pd.DataFrame({"a": [1, 2, None, 4], "b": [5, 6, 7, 8]})
    prof = ml.profile(data=data)

    warnings = prof["warnings"]
    assert any("missing values in 'a'" in w for w in warnings)
    assert any("25.0%" in w for w in warnings)


def test_profile_warnings_constant():
    """Warning for constant columns."""
    data = pd.DataFrame({"a": [1, 1, 1, 1], "b": [2, 3, 4, 5]})
    prof = ml.profile(data=data)

    warnings = prof["warnings"]
    assert any("'a' is constant" in w for w in warnings)


def test_profile_warnings_high_cardinality():
    """Warning for high-cardinality categorical columns."""
    data = pd.DataFrame({"id": [f"ID{i}" for i in range(100)]})
    prof = ml.profile(data=data)

    warnings = prof["warnings"]
    assert any("'id' high cardinality" in w for w in warnings)


def test_profile_empty_data_error():
    """Empty data raises DataError."""
    data = pd.DataFrame()
    with pytest.raises(ml.DataError, match="Cannot profile empty data"):
        ml.profile(data=data)


def test_profile_bad_target_error():
    """Nonexistent target raises DataError."""
    data = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ml.DataError, match="target='missing' not found"):
        ml.profile(data=data, target="missing")


def test_profile_non_dataframe_error():
    """Non-DataFrame input raises DataError."""
    with pytest.raises(ml.DataError):
        ml.profile(data=[1, 2, 3])
