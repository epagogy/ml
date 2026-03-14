"""Tests for ml.prepare() — grammar primitive #2."""

import numpy as np
import pandas as pd
import pytest

import ml
from ml._types import PreparedData


@pytest.fixture
def clf_df():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "age": rng.integers(18, 65, n).astype(float),
        "income": rng.normal(50000, 15000, n),
        "category": rng.choice(["A", "B", "C"], n),
        "target": rng.choice([0, 1], n),
    })


@pytest.fixture
def reg_df():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(5, 2, n),
        "group": rng.choice(["X", "Y"], n),
        "y": rng.normal(10, 3, n),
    })


def test_prepare_returns_prepared_data(clf_df):
    p = ml.prepare(clf_df, "target")
    assert isinstance(p, PreparedData)


def test_prepared_data_fields(clf_df):
    p = ml.prepare(clf_df, "target")
    assert isinstance(p.data, pd.DataFrame)
    assert p.target == "target"
    assert p.task in ("classification", "regression")
    assert p.state is not None


def test_prepare_data_is_numeric(clf_df):
    p = ml.prepare(clf_df, "target")
    for col in p.data.columns:
        assert pd.api.types.is_numeric_dtype(p.data[col]), f"{col} is not numeric"


def test_prepare_target_not_in_output(clf_df):
    p = ml.prepare(clf_df, "target")
    assert "target" not in p.data.columns


def test_prepare_detects_classification(clf_df):
    p = ml.prepare(clf_df, "target")
    assert p.task == "classification"


def test_prepare_detects_regression(reg_df):
    p = ml.prepare(reg_df, "y")
    assert p.task == "regression"


def test_prepare_explicit_task(clf_df):
    p = ml.prepare(clf_df, "target", task="classification")
    assert p.task == "classification"


def test_prepare_state_transform_on_new_data(clf_df):
    """State from prepare() can encode a new DataFrame."""
    p = ml.prepare(clf_df, "target")
    X_new = clf_df.drop(columns=["target"]).iloc[:10]
    X_enc = p.state.transform(X_new)
    assert X_enc is not None
    assert len(X_enc) == 10


def test_prepare_row_count_preserved(clf_df):
    p = ml.prepare(clf_df, "target")
    assert len(p.data) == len(clf_df)


def test_prepare_algorithm_hint(clf_df):
    """algorithm= param accepted without error."""
    p = ml.prepare(clf_df, "target", algorithm="random_forest")
    assert isinstance(p, PreparedData)
    p2 = ml.prepare(clf_df, "target", algorithm="logistic")
    assert isinstance(p2, PreparedData)


def test_prepare_missing_target_raises(clf_df):
    with pytest.raises(Exception):
        ml.prepare(clf_df, "nonexistent_column")


def test_prepare_none_data_raises():
    with pytest.raises(Exception):
        ml.prepare(None, "target")


def test_prepare_type_exported():
    """PreparedData is accessible from ml namespace."""
    assert hasattr(ml, "PreparedData")
    assert hasattr(ml, "prepare")


def test_prepare_with_split_result(clf_df):
    """Typical grammar workflow: split -> prepare -> fit."""
    s = ml.split(clf_df, "target", seed=42)
    p = ml.prepare(s.train, "target")
    assert isinstance(p, PreparedData)
    assert len(p.data) == len(s.train)
