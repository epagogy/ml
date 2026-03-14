"""Tests for native logistic regression (_logistic.py + _engines.py integration)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

import ml
from ml._logistic import _LogisticModel
from ml._types import ConfigError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def iris_df():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = [iris.target_names[i] for i in iris.target]
    return df


@pytest.fixture
def iris_binary_df(iris_df):
    """Binary subset: setosa vs versicolor only."""
    return iris_df[iris_df["species"].isin(["setosa", "versicolor"])].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Unit tests on _LogisticModel directly
# ---------------------------------------------------------------------------

def test_logistic_binary():
    """Binary iris subset — accuracy > 0.80."""
    iris = load_iris()
    X = iris.data[:100]  # setosa + versicolor only
    y = iris.target[:100]
    model = _LogisticModel(C=1.0)
    model.fit(X, y)
    preds = model.predict(X)
    acc = np.mean(preds == y)
    assert acc > 0.80, f"Binary accuracy too low: {acc:.3f}"


def test_logistic_multiclass():
    """Full iris 3-class — accuracy > 0.70."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    model = _LogisticModel(C=1.0)
    model.fit(X, y)
    preds = model.predict(X)
    acc = np.mean(preds == y)
    assert acc > 0.70, f"Multiclass accuracy too low: {acc:.3f}"


def test_logistic_proba_sum():
    """predict_proba rows sum to 1.0 ±1e-6."""
    iris = load_iris()
    model = _LogisticModel()
    model.fit(iris.data, iris.target)
    proba = model.predict_proba(iris.data)
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"Row sums off: {row_sums.min():.8f}–{row_sums.max():.8f}"


def test_logistic_proba_range():
    """All probabilities in [0, 1]."""
    iris = load_iris()
    model = _LogisticModel()
    model.fit(iris.data, iris.target)
    proba = model.predict_proba(iris.data)
    assert proba.min() >= 0.0, f"Negative probability: {proba.min()}"
    assert proba.max() <= 1.0, f"Probability > 1: {proba.max()}"


def test_logistic_proba_shape_binary():
    """Binary: predict_proba shape == (n, 2)."""
    iris = load_iris()
    X = iris.data[:100]
    y = iris.target[:100]
    model = _LogisticModel()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (100, 2), f"Expected (100, 2), got {proba.shape}"


def test_logistic_proba_shape_multiclass():
    """Multiclass: predict_proba shape == (n, 3)."""
    iris = load_iris()
    model = _LogisticModel()
    model.fit(iris.data, iris.target)
    proba = model.predict_proba(iris.data)
    assert proba.shape == (150, 3), f"Expected (150, 3), got {proba.shape}"


def test_logistic_reproducible():
    """Two identical fits produce identical predictions (L-BFGS is deterministic)."""
    iris = load_iris()
    m1 = _LogisticModel(C=1.0)
    m1.fit(iris.data, iris.target)
    m2 = _LogisticModel(C=1.0)
    m2.fit(iris.data, iris.target)
    assert np.array_equal(m1.predict(iris.data), m2.predict(iris.data))


# ---------------------------------------------------------------------------
# Integration tests via ml.fit
# ---------------------------------------------------------------------------

def test_logistic_via_ml_fit(iris_df):
    """ml.fit(..., algorithm='logistic') returns a fitted model."""
    s = ml.split(iris_df, "species", seed=42)
    model = ml.fit(s.train, "species", algorithm="logistic", seed=42)
    preds = ml.predict(model, s.valid)
    assert len(preds) == len(s.valid)


def test_logistic_regression_raises(iris_df):
    """algorithm='logistic' on regression target raises ConfigError."""
    df = iris_df.copy()
    df["target"] = np.random.default_rng(0).random(len(df))
    s = ml.split(df, "target", seed=42)
    with pytest.raises(ConfigError, match="classification"):
        ml.fit(s.train, "target", algorithm="logistic", seed=42)


def test_logistic_l1_raises(iris_df):
    """penalty='l1' raises ConfigError (native L2 only)."""
    s = ml.split(iris_df, "species", seed=42)
    with pytest.raises(ConfigError, match="L1"):
        ml.fit(s.train, "species", algorithm="logistic", penalty="l1", seed=42)


def test_logistic_balance_weights(iris_df):
    """ml.fit(..., balance=True) with logistic doesn't crash."""
    s = ml.split(iris_df, "species", seed=42)
    model = ml.fit(s.train, "species", algorithm="logistic", balance=True, seed=42)
    preds = ml.predict(model, s.valid)
    assert len(preds) == len(s.valid)
