"""Tests for ml.discretize() — continuous feature binning."""

import numpy as np
import pandas as pd
import pytest

import ml
from ml.bin import Binner, discretize


@pytest.fixture
def num_df():
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "age": rng.randint(18, 80, n).astype(float),
        "income": rng.exponential(50000, n),
        "score": rng.rand(n),
        "label": rng.choice([0, 1], n),
    })


def test_discretize_returns_binner(num_df):
    """discretize() returns a Binner object."""
    binner = discretize(num_df, columns=["age"])
    assert isinstance(binner, Binner)
    assert binner.method == "quantile"
    assert binner.n_bins == 5


def test_discretize_transform_integer_columns(num_df):
    """transform() replaces columns with integer bin indices."""
    binner = discretize(num_df, columns=["age", "income"])
    out = binner.transform(num_df)
    assert out["age"].dtype in (int, "int64", "int32")
    assert out["income"].dtype in (int, "int64", "int32")


def test_discretize_quantile_bins_in_range(num_df):
    """Quantile binning produces indices in 0..n_bins-1."""
    n_bins = 5
    binner = discretize(num_df, columns=["age"], method="quantile", n_bins=n_bins)
    out = binner.transform(num_df)
    assert out["age"].min() >= 0
    assert out["age"].max() <= n_bins - 1


def test_discretize_uniform_bins_in_range(num_df):
    """Uniform binning produces indices in 0..n_bins-1."""
    n_bins = 4
    binner = discretize(num_df, columns=["income"], method="uniform", n_bins=n_bins)
    out = binner.transform(num_df)
    assert out["income"].min() >= 0
    assert out["income"].max() <= n_bins - 1


def test_discretize_via_ml_namespace(num_df):
    """ml.discretize() accessible from public API."""
    binner = ml.discretize(num_df, columns=["age"])
    assert isinstance(binner, Binner)


def test_discretize_preserves_other_columns(num_df):
    """discretize() does not touch non-specified columns."""
    binner = discretize(num_df, columns=["age"])
    out = binner.transform(num_df)
    pd.testing.assert_series_equal(out["score"], num_df["score"])
    pd.testing.assert_series_equal(out["label"], num_df["label"])
