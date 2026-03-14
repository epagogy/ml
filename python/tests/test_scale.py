"""Tests for ml.scale() — numeric feature scaling."""

import numpy as np
import pandas as pd
import pytest

import ml
from ml._types import ConfigError, DataError
from ml.scale import Scaler, scale


@pytest.fixture
def num_df():
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "age": rng.normal(40, 10, 100),
        "income": rng.normal(60000, 15000, 100),
        "score": rng.uniform(0, 1, 100),
        "label": rng.choice(["yes", "no"], 100),
    })


# Return type
def test_scale_returns_scaler(num_df):
    scl = scale(num_df, columns=["age", "income"])
    assert isinstance(scl, Scaler)

def test_scale_via_ml_namespace(num_df):
    scl = ml.scale(num_df, columns=["age"])
    assert isinstance(scl, Scaler)

def test_scale_attributes(num_df):
    scl = scale(num_df, columns=["age", "income"], method="standard")
    assert scl.columns == ["age", "income"]
    assert scl.method == "standard"
    assert "age" in scl._scalers
    assert "income" in scl._scalers

# transform output
def test_scale_preserves_column_names(num_df):
    scl = scale(num_df, columns=["age", "income"])
    out = scl.transform(num_df)
    assert "age" in out.columns
    assert "income" in out.columns

def test_scale_preserves_non_scaled_columns(num_df):
    scl = scale(num_df, columns=["age"])
    out = scl.transform(num_df)
    assert "income" in out.columns
    assert "label" in out.columns

def test_scale_standard_mean_near_zero(num_df):
    scl = scale(num_df, columns=["age", "income"])
    out = scl.transform(num_df)
    assert abs(out["age"].mean()) < 0.01
    assert abs(out["income"].mean()) < 0.01

def test_scale_standard_std_near_one(num_df):
    scl = scale(num_df, columns=["age"])
    out = scl.transform(num_df)
    assert abs(out["age"].std() - 1.0) < 0.05

def test_scale_minmax_range_zero_one(num_df):
    scl = scale(num_df, columns=["age"], method="minmax")
    out = scl.transform(num_df)
    assert out["age"].min() >= -1e-9
    assert out["age"].max() <= 1 + 1e-9

def test_scale_robust_no_crash(num_df):
    scl = scale(num_df, columns=["age"], method="robust")
    out = scl.transform(num_df)
    assert "age" in out.columns

def test_scale_preserves_index(num_df):
    num_df = num_df.copy()
    num_df.index = range(100, 200)
    scl = scale(num_df, columns=["age"])
    out = scl.transform(num_df)
    assert list(out.index) == list(num_df.index)

def test_scale_transform_valid_differs_from_train(num_df):
    """Different data — different scaled values (not re-fitted)."""
    train = num_df.iloc[:80]
    valid = num_df.iloc[80:]
    scl = scale(train, columns=["age"])
    out_valid = scl.transform(valid)
    # Valid values use train's mean/std — not re-centered
    assert not out_valid["age"].equals(scale(valid, columns=["age"]).transform(valid)["age"])

# No leakage: fit on train, apply to valid
def test_scale_fit_on_train_apply_to_valid(num_df):
    train = num_df.iloc[:80]
    valid = num_df.iloc[80:]
    scl = scale(train, columns=["age"])
    out = scl.transform(valid)
    assert "age" in out.columns

# save/load
def test_scale_save_load_standard(tmp_path, num_df):
    scl = scale(num_df, columns=["age", "income"], method="standard")
    path = str(tmp_path / "scaler.pyml")
    ml.save(scl, path)
    scl2 = ml.load(path)
    assert isinstance(scl2, Scaler)
    pd.testing.assert_frame_equal(scl.transform(num_df), scl2.transform(num_df))

def test_scale_save_load_minmax(tmp_path, num_df):
    scl = scale(num_df, columns=["age"], method="minmax")
    path = str(tmp_path / "s.pyml")
    ml.save(scl, path)
    scl2 = ml.load(path)
    pd.testing.assert_frame_equal(scl.transform(num_df), scl2.transform(num_df))

def test_scale_save_load_robust(tmp_path, num_df):
    scl = scale(num_df, columns=["age"], method="robust")
    path = str(tmp_path / "s.pyml")
    ml.save(scl, path)
    scl2 = ml.load(path)
    pd.testing.assert_frame_equal(scl.transform(num_df), scl2.transform(num_df))

# Error handling
def test_scale_non_dataframe_raises():
    with pytest.raises(DataError):
        scale(pd.Series([1.0, 2.0]), columns=["x"])

def test_scale_empty_columns_raises(num_df):
    with pytest.raises(DataError):
        scale(num_df, columns=[])

def test_scale_missing_column_raises(num_df):
    with pytest.raises(DataError):
        scale(num_df, columns=["nonexistent"])

def test_scale_bad_method_raises(num_df):
    with pytest.raises(ConfigError):
        scale(num_df, columns=["age"], method="zscore")

def test_scale_transform_missing_column_raises(num_df):
    scl = scale(num_df, columns=["age"])
    with pytest.raises(DataError):
        scl.transform(num_df.drop(columns=["age"]))

# repr
def test_scale_repr(num_df):
    scl = scale(num_df, columns=["age"])
    assert "Scaler" in repr(scl)
    assert "standard" in repr(scl)
