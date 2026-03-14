"""Tests for native linear (Ridge) regression (_linear.py + _engines.py integration)."""

import numpy as np
import pandas as pd
import pytest

import ml
from ml._linear import _LinearModel
from ml._types import ConfigError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_df():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 5))
    y = X[:, 0] * 2.0 + X[:, 1] * 1.5 + rng.standard_normal(n) * 0.3
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return df


# ---------------------------------------------------------------------------
# Unit tests on _LinearModel directly
# ---------------------------------------------------------------------------

def test_linear_basic():
    """Ridge on simple linear data — R² > 0.90."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    y = X[:, 0] * 2 + X[:, 1] * -1 + rng.standard_normal(200) * 0.2
    model = _LinearModel(alpha=0.01)
    model.fit(X[:150], y[:150])
    pred = model.predict(X[150:])
    ss_res = np.sum((pred - y[150:]) ** 2)
    ss_tot = np.sum((y[150:] - y[150:].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.90, f"R² too low: {r2:.3f}"


def test_linear_coef_shape():
    """coef_ shape == (n_features,)."""
    model = _LinearModel()
    X = np.random.default_rng(1).standard_normal((50, 4))
    y = X @ np.array([1, 2, -1, 0.5]) + 0.1
    model.fit(X, y)
    assert model.coef_.shape == (4,), f"Expected (4,), got {model.coef_.shape}"


def test_linear_predict_shape():
    """Predictions shape matches number of input rows."""
    model = _LinearModel()
    X_tr = np.random.default_rng(2).standard_normal((100, 3))
    y_tr = X_tr[:, 0] + 1.0
    model.fit(X_tr, y_tr)
    X_te = np.random.default_rng(3).standard_normal((25, 3))
    assert model.predict(X_te).shape == (25,)


def test_linear_reproducible():
    """Two identical fits produce identical predictions."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((100, 5))
    y = X[:, 0] + rng.standard_normal(100) * 0.1
    m1 = _LinearModel(alpha=1.0)
    m1.fit(X, y)
    m2 = _LinearModel(alpha=1.0)
    m2.fit(X, y)
    assert np.allclose(m1.predict(X), m2.predict(X))


def test_linear_regularization():
    """Higher alpha → smaller coefficient norms."""
    rng = np.random.default_rng(8)
    X = rng.standard_normal((100, 5))
    y = X @ np.ones(5) + rng.standard_normal(100) * 0.5
    m_low = _LinearModel(alpha=0.001)
    m_high = _LinearModel(alpha=100.0)
    m_low.fit(X, y)
    m_high.fit(X, y)
    assert np.linalg.norm(m_high.coef_) < np.linalg.norm(m_low.coef_)


def test_linear_zero_alpha_ols():
    """alpha=0 is pure OLS — predictions very close to true signal."""
    rng = np.random.default_rng(9)
    X = rng.standard_normal((200, 3))
    w_true = np.array([2.0, -1.0, 0.5])
    y = X @ w_true + rng.standard_normal(200) * 0.01
    model = _LinearModel(alpha=0.0)
    model.fit(X, y)
    assert np.allclose(model.coef_, w_true, atol=0.05), f"coef {model.coef_} ≠ {w_true}"


# ---------------------------------------------------------------------------
# Integration tests via ml.fit
# ---------------------------------------------------------------------------

def test_linear_via_ml_fit(regression_df):
    """ml.fit(..., algorithm='linear') returns a fitted model."""
    s = ml.split(regression_df, "target", seed=42)
    model = ml.fit(s.train, "target", algorithm="linear", seed=42)
    preds = ml.predict(model, s.valid)
    assert len(preds) == len(s.valid)


def test_linear_classification_raises(regression_df):
    """algorithm='linear' on classification target raises ConfigError."""
    df = regression_df.copy()
    df["cls"] = (df["target"] > 0).astype(str)
    s = ml.split(df, "cls", seed=42)
    with pytest.raises(ConfigError, match="regression"):
        ml.fit(s.train, "cls", algorithm="linear", seed=42)


def test_linear_metrics_reasonable(regression_df):
    """R² > 0.85 on a clean linear dataset."""
    s = ml.split(regression_df, "target", seed=42)
    model = ml.fit(s.train, "target", algorithm="linear", seed=42)
    metrics = ml.evaluate(model, s.valid)
    assert metrics["r2"] > 0.85, f"R² too low: {metrics['r2']:.3f}"
