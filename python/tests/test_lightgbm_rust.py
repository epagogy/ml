"""Integration tests: algorithm='lightgbm' with engine='ml' (Rust GBT + GOSS).

Tests cover:
- Rust routing (engine='ml', engine='auto')
- GOSS param mapping from LightGBM aliases
- max_depth=-1 (unlimited) handling
- multiclass
- early stopping mapping
- GOSS gate at small n
- engine='ml' error when Rust unavailable
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ml
from ml._rust import HAS_RUST_GBT, _RustGBTClassifier, _RustGBTRegressor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clf_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = y
    s = ml.split(df, "target", seed=0)
    return s.train, s.valid


@pytest.fixture()
def reg_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 8))
    y = X[:, 0] * 2.0 + X[:, 1] + rng.standard_normal(300) * 0.1
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = y
    s = ml.split(df, "target", seed=1, task="regression")
    return s.train, s.valid


@pytest.fixture()
def multiclass_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((450, 8))
    y = (X[:, 0] + X[:, 1]).clip(-2, 2)
    y = np.digitize(y, bins=[-0.67, 0.67])  # 0, 1, 2
    y = y.astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = y
    s = ml.split(df, "target", seed=2)
    return s.train, s.valid


# ---------------------------------------------------------------------------
# Smoke: engine='ml'
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_RUST_GBT, reason="Rust GBT not installed")
def test_lightgbm_rust_clf(clf_data):
    train, valid = clf_data
    m = ml.fit(train, "target", algorithm="lightgbm", engine="ml", seed=0, n_estimators=50)
    assert isinstance(m._model, _RustGBTClassifier)
    preds = ml.predict(m, valid)
    assert len(preds) == len(valid)
    metrics = ml.evaluate(m, valid)
    assert metrics["roc_auc"] > 0.5


@pytest.mark.skipif(not HAS_RUST_GBT, reason="Rust GBT not installed")
def test_lightgbm_rust_reg(reg_data):
    train, valid = reg_data
    m = ml.fit(train, "target", algorithm="lightgbm", engine="ml", seed=1,
                n_estimators=50, task="regression")
    assert isinstance(m._model, _RustGBTRegressor)
    preds = ml.predict(m, valid)
    assert len(preds) == len(valid)
    metrics = ml.evaluate(m, valid)
    assert metrics["r2"] > 0.0


@pytest.mark.skipif(not HAS_RUST_GBT, reason="Rust GBT not installed")
def test_lightgbm_rust_multiclass(multiclass_data):
    train, valid = multiclass_data
    m = ml.fit(train, "target", algorithm="lightgbm", engine="ml", seed=2, n_estimators=50)
    assert isinstance(m._model, _RustGBTClassifier)
    preds = ml.predict(m, valid)
    assert set(preds.unique()).issubset({0, 1, 2})
    metrics = ml.evaluate(m, valid)
    assert "accuracy" in metrics


# ---------------------------------------------------------------------------
# engine='auto' uses Rust when available
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_RUST_GBT, reason="Rust GBT not installed")
def test_lightgbm_auto_uses_rust_when_available(clf_data):
    train, _ = clf_data
    m = ml.fit(train, "target", algorithm="lightgbm", engine="auto", seed=0, n_estimators=20)
    assert isinstance(m._model, _RustGBTClassifier), (
        "engine='auto' should use Rust when ml-py is installed"
    )


# ---------------------------------------------------------------------------
# engine='ml' error without Rust
# ---------------------------------------------------------------------------


def test_lightgbm_engine_ml_error_without_rust(clf_data, monkeypatch):
    """engine='ml' must raise ConfigError if Rust GBT not available."""
    import ml._rust as rust_mod

    monkeypatch.setattr(rust_mod, "HAS_RUST_GBT", False)
    train, _ = clf_data
    with pytest.raises(Exception, match="engine='ml'"):
        ml.fit(train, "target", algorithm="lightgbm", engine="ml", seed=0)


# ---------------------------------------------------------------------------
# Param mapping
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_RUST_GBT, reason="Rust GBT not installed")
def test_lightgbm_param_mapping_depth_minus1(clf_data):
    """max_depth=-1 must not crash (unlimited depth)."""
    train, valid = clf_data
    m = ml.fit(train, "target", algorithm="lightgbm", engine="ml", seed=0,
                n_estimators=20, max_depth=-1)
    assert isinstance(m._model, _RustGBTClassifier)
    preds = ml.predict(m, valid)
    assert len(preds) == len(valid)


@pytest.mark.skipif(not HAS_RUST_GBT, reason="Rust GBT not installed")
def test_lightgbm_param_mapping_num_leaves(clf_data):
    """num_leaves alias maps to max_leaves on Rust model."""
    train, _ = clf_data
    m = ml.fit(train, "target", algorithm="lightgbm", engine="ml", seed=0,
                n_estimators=10, num_leaves=63)
    assert isinstance(m._model, _RustGBTClassifier)
    assert m._model.max_leaves == 63


@pytest.mark.skipif(not HAS_RUST_GBT, reason="Rust GBT not installed")
def test_lightgbm_early_stopping_mapped(clf_data):
    """early_stopping_rounds alias maps to n_iter_no_change on Rust model."""
    train, valid = clf_data
    m = ml.fit(train, "target", algorithm="lightgbm", engine="ml", seed=0,
                n_estimators=200, early_stopping_rounds=5)
    assert isinstance(m._model, _RustGBTClassifier)
    assert m._model.n_iter_no_change == 5


# ---------------------------------------------------------------------------
# GOSS gate: small n must skip GOSS
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_RUST_GBT, reason="Rust GBT not installed")
def test_lightgbm_goss_small_n_skipped():
    """With n=500 < goss_min_n=50_000, GOSS weights should not be applied.

    We verify this indirectly: the model trains without error and the
    goss_min_n param is stored correctly on the Rust model.
    """
    rng = np.random.default_rng(99)
    X = rng.standard_normal((500, 5))
    y = (X[:, 0] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    s = ml.split(df, "target", seed=99)
    # Use explicit goss params with small n → GOSS gate should skip
    m = ml.fit(s.train, "target", algorithm="lightgbm", engine="ml", seed=99,
                n_estimators=10, goss_min_n=50_000)
    assert isinstance(m._model, _RustGBTClassifier)
    assert m._model.goss_min_n == 50_000
    preds = ml.predict(m, s.valid)
    assert len(preds) == len(s.valid)
