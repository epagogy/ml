"""Tests for ml.impute() — missing value imputation."""

import numpy as np
import pandas as pd
import pytest

import ml
from ml._types import ConfigError, DataError
from ml.impute import Imputer, impute


@pytest.fixture
def df_with_nans():
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "income": rng.normal(60000, 15000, n),
        "city": rng.choice(["NYC", "LA", "Chicago"], n),
        "label": rng.choice(["yes", "no"], n),
    })
    df.loc[::5, "age"] = np.nan      # 20% NaN
    df.loc[::10, "income"] = np.nan  # 10% NaN
    df.loc[::7, "city"] = np.nan     # ~14% NaN
    return df


# ── Return type ───────────────────────────────────────────────────────────────

def test_impute_returns_imputer(df_with_nans):
    imp = impute(df_with_nans, columns=["age"])
    assert isinstance(imp, Imputer)


def test_impute_via_ml_namespace(df_with_nans):
    imp = ml.impute(df_with_nans, columns=["age"])
    assert isinstance(imp, Imputer)


def test_impute_attributes(df_with_nans):
    imp = impute(df_with_nans, columns=["age", "income"], strategy="median")
    assert imp.columns == ["age", "income"]
    assert imp.strategy == "median"
    assert "age" in imp._fill_values
    assert "income" in imp._fill_values


# ── Auto-detect columns ───────────────────────────────────────────────────────

def test_impute_auto_detects_numeric_nan_columns(df_with_nans):
    """Default strategy=median → numeric NaN columns only."""
    imp = impute(df_with_nans)
    assert "age" in imp.columns
    assert "income" in imp.columns
    assert "city" not in imp.columns   # categorical — excluded for median
    assert "label" not in imp.columns  # no NaN


def test_impute_auto_detects_categorical_with_mode(df_with_nans):
    """strategy=mode → includes categorical columns."""
    imp = impute(df_with_nans, strategy="mode")
    assert "age" in imp.columns
    assert "income" in imp.columns
    assert "city" in imp.columns


def test_impute_auto_returns_noop_for_clean_data():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    imp = impute(df)
    assert imp.columns == []


# ── transform — NaN removal ───────────────────────────────────────────────────

def test_impute_fills_all_nans(df_with_nans):
    imp = impute(df_with_nans, columns=["age", "income"])
    out = imp.transform(df_with_nans)
    assert not out["age"].isna().any()
    assert not out["income"].isna().any()


def test_impute_preserves_non_imputed_columns(df_with_nans):
    imp = impute(df_with_nans, columns=["age"])
    out = imp.transform(df_with_nans)
    assert "income" in out.columns
    assert "city" in out.columns


def test_impute_preserves_index(df_with_nans):
    df = df_with_nans.copy()
    df.index = range(100, 200)
    imp = impute(df, columns=["age"])
    out = imp.transform(df)
    assert list(out.index) == list(df.index)


def test_impute_preserves_row_count(df_with_nans):
    imp = impute(df_with_nans, columns=["age"])
    out = imp.transform(df_with_nans)
    assert len(out) == len(df_with_nans)


# ── Strategies ────────────────────────────────────────────────────────────────

def test_impute_median_fill_value(df_with_nans):
    imp = impute(df_with_nans, columns=["age"], strategy="median")
    expected = df_with_nans["age"].dropna().median()
    assert abs(imp._fill_values["age"] - expected) < 1e-9


def test_impute_mean_fill_value(df_with_nans):
    imp = impute(df_with_nans, columns=["age"], strategy="mean")
    expected = df_with_nans["age"].dropna().mean()
    assert abs(imp._fill_values["age"] - expected) < 1e-9


def test_impute_mode_fills_categorical(df_with_nans):
    imp = impute(df_with_nans, columns=["city"], strategy="mode")
    out = imp.transform(df_with_nans)
    assert not out["city"].isna().any()


def test_impute_mode_fill_value_is_most_frequent(df_with_nans):
    imp = impute(df_with_nans, columns=["city"], strategy="mode")
    most_frequent = df_with_nans["city"].dropna().mode().iloc[0]
    assert imp._fill_values["city"] == most_frequent


def test_impute_constant_fill_value(df_with_nans):
    imp = impute(df_with_nans, columns=["age"], strategy="constant", fill_value=0.0)
    out = imp.transform(df_with_nans)
    assert not out["age"].isna().any()
    was_nan = df_with_nans["age"].isna()
    assert (out.loc[was_nan, "age"] == 0.0).all()


# ── No leakage ────────────────────────────────────────────────────────────────

def test_impute_fit_on_train_apply_to_valid(df_with_nans):
    train = df_with_nans.iloc[:80]
    valid = df_with_nans.iloc[80:]
    imp = impute(train, columns=["age", "income"])
    out = imp.transform(valid)
    assert not out["age"].isna().any()
    assert not out["income"].isna().any()


def test_impute_uses_train_median_not_valid_median(df_with_nans):
    """Fill value comes from training data, not from the data being transformed."""
    train = df_with_nans.iloc[:80]
    valid = df_with_nans.iloc[80:].copy()
    imp = impute(train, columns=["age"])
    train_median = train["age"].dropna().median()
    out = imp.transform(valid)
    was_nan = valid["age"].isna()
    if was_nan.any():
        assert (out.loc[was_nan, "age"] == train_median).all()


# ── save / load ───────────────────────────────────────────────────────────────

def test_impute_save_load_median(tmp_path, df_with_nans):
    imp = impute(df_with_nans, columns=["age", "income"], strategy="median")
    path = str(tmp_path / "imp.pyml")
    ml.save(imp, path)
    imp2 = ml.load(path)
    assert isinstance(imp2, Imputer)
    pd.testing.assert_frame_equal(imp.transform(df_with_nans), imp2.transform(df_with_nans))


def test_impute_save_load_constant(tmp_path, df_with_nans):
    imp = impute(df_with_nans, columns=["age"], strategy="constant", fill_value=-1.0)
    path = str(tmp_path / "imp.pyml")
    ml.save(imp, path)
    imp2 = ml.load(path)
    pd.testing.assert_frame_equal(imp.transform(df_with_nans), imp2.transform(df_with_nans))


# ── Error handling ────────────────────────────────────────────────────────────

def test_impute_non_dataframe_raises():
    with pytest.raises(DataError):
        impute(pd.Series([1.0, np.nan]), columns=["x"])


def test_impute_missing_column_raises(df_with_nans):
    with pytest.raises(DataError):
        impute(df_with_nans, columns=["nonexistent"])


def test_impute_bad_strategy_raises(df_with_nans):
    with pytest.raises(ConfigError):
        impute(df_with_nans, columns=["age"], strategy="forward_fill")


def test_impute_constant_without_fill_value_raises(df_with_nans):
    with pytest.raises(ConfigError):
        impute(df_with_nans, columns=["age"], strategy="constant")


def test_impute_mean_on_categorical_raises(df_with_nans):
    with pytest.raises(ConfigError):
        impute(df_with_nans, columns=["city"], strategy="mean")


def test_impute_median_on_categorical_raises(df_with_nans):
    with pytest.raises(ConfigError):
        impute(df_with_nans, columns=["city"], strategy="median")


def test_impute_transform_missing_column_raises(df_with_nans):
    imp = impute(df_with_nans, columns=["age"])
    with pytest.raises(DataError):
        imp.transform(df_with_nans.drop(columns=["age"]))


# ── repr ──────────────────────────────────────────────────────────────────────

def test_impute_repr(df_with_nans):
    imp = impute(df_with_nans, columns=["age"])
    assert "Imputer" in repr(imp)
    assert "median" in repr(imp)
