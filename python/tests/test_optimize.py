"""Tests for A4: Threshold Optimization — ml.optimize()."""

import numpy as np
import pandas as pd
import pytest

import ml
from ml._types import ConfigError, ModelError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_binary_data(n=200, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "x1": rng.randn(n),
        "x2": rng.randn(n),
        "target": rng.choice([0, 1], n),
    })


def _make_imbalanced_data(n=500, pos_rate=0.05, seed=42):
    """Heavily imbalanced binary classification dataset."""
    rng = np.random.RandomState(seed)
    n_pos = int(n * pos_rate)
    n_neg = n - n_pos
    return pd.DataFrame({
        "x1": rng.randn(n),
        "x2": rng.randn(n),
        "target": [1] * n_pos + [0] * n_neg,
    })


# ---------------------------------------------------------------------------
# A4 tests
# ---------------------------------------------------------------------------

def test_optimize_basic():
    """optimize() returns model with _threshold set in [0, 1]."""
    data = _make_binary_data()
    s = ml.split(data, "target", seed=42)
    model = ml.fit(s.train, "target", algorithm="random_forest", seed=42)
    opt = ml.optimize(model, data=s.valid, metric="f1")
    assert opt._threshold is not None
    assert 0.0 <= opt._threshold <= 1.0
    # Original model unchanged
    assert model._threshold is None


def test_optimize_imbalanced():
    """min_threshold='auto' on imbalanced data uses 1/n_pos as lower bound."""
    data = _make_imbalanced_data()
    s = ml.split(data, "target", seed=42)
    model = ml.fit(s.train, "target", algorithm="random_forest", seed=42)
    opt = ml.optimize(model, data=s.valid, metric="f1", min_threshold="auto")
    n_pos = int((s.valid["target"] == 1).sum())
    expected_min = max(0.001, 1.0 / n_pos) if n_pos > 0 else 0.001
    assert opt._threshold >= expected_min - 1e-9


def test_predict_uses_threshold():
    """After optimize(), model.predict() applies the threshold instead of 0.5."""
    data = _make_binary_data()
    s = ml.split(data, "target", seed=42)
    model = ml.fit(s.train, "target", algorithm="random_forest", seed=42)
    # Force a very high threshold — almost everything should predict class 0
    opt = ml.optimize(model, data=s.valid, metric="f1", min_threshold=0.90)
    preds_opt = opt.predict(s.valid)
    # High threshold shifts predictions differently than default 0.5
    # (the optimised model may predict different distribution)
    assert set(preds_opt.unique()).issubset({0, 1})
    # Verify threshold is actually applied: force threshold=0.99
    import copy
    forced = copy.copy(model)
    forced._threshold = 0.99
    preds_forced = forced.predict(s.valid)
    # At threshold=0.99, nearly all should be negative class (0)
    assert (preds_forced == 0).sum() >= (preds_forced == 1).sum()


def test_ranking_metric_error():
    """optimize() raises ConfigError for ranking metrics (roc_auc, log_loss)."""
    data = _make_binary_data()
    s = ml.split(data, "target", seed=42)
    model = ml.fit(s.train, "target", algorithm="random_forest", seed=42)
    with pytest.raises(ConfigError, match="ranking metric"):
        ml.optimize(model, data=s.valid, metric="roc_auc")
    with pytest.raises(ConfigError, match="ranking metric"):
        ml.optimize(model, data=s.valid, metric="log_loss")


def test_regression_error():
    """optimize() raises ModelError on regression models."""
    rng = np.random.RandomState(42)
    reg_data = pd.DataFrame({
        "x1": rng.randn(100),
        "x2": rng.randn(100),
        "target": rng.randn(100),
    })
    s = ml.split(reg_data, "target", seed=42)
    model = ml.fit(s.train, "target", algorithm="random_forest", seed=42)
    with pytest.raises(ModelError, match="classification"):
        ml.optimize(model, data=s.valid, metric="f1")


def test_oof_based_threshold():
    """oof_predictions= uses pre-computed probabilities instead of predict_proba."""
    data = _make_binary_data()
    s = ml.split(data, "target", seed=42)
    model = ml.fit(s.train, "target", algorithm="random_forest", seed=42)
    # Compute probabilities up front
    proba_df = model.predict_proba(s.valid)
    positive_class = model.classes_[1]
    oof_proba = proba_df[positive_class]
    # optimize with OOF probabilities
    opt_oof = ml.optimize(model, data=s.valid, oof_predictions=oof_proba, metric="f1")
    # optimize without (should produce same threshold since same data and proba)
    opt_fresh = ml.optimize(model, data=s.valid, metric="f1")
    # Both should have a threshold set
    assert opt_oof._threshold is not None
    assert opt_fresh._threshold is not None
    # Thresholds should be close (same underlying data)
    assert abs(opt_oof._threshold - opt_fresh._threshold) < 0.1
