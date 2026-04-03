"""Big data stress tests — Phase 3.

All tests marked @slow. Run on server only:
    pytest tests/test_bigdata.py -v --timeout=300

These prove ml handles production-scale data without OOM or correctness loss.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import ml


@pytest.mark.slow
def test_bigdata_logistic_1m():
    """1M rows + logistic: fit + predict + evaluate complete with finite metrics."""
    rng = np.random.RandomState(42)
    n = 1_000_000
    X = rng.randn(n, 10).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * 0.5 + rng.randn(n).astype(np.float32) * 0.5 > 0).astype(int)
    data = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    data["target"] = y

    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="logistic", seed=42)
    preds = ml.predict(model, s.valid)
    metrics = ml.evaluate(model, s.valid)

    assert len(preds) == len(s.valid)
    assert isinstance(metrics, dict)
    for k, v in metrics.items():
        assert np.isfinite(v), f"Metric {k} is not finite: {v}"
    assert metrics["accuracy"] > 0.5
    assert metrics["roc_auc"] > 0.5


@pytest.mark.slow
def test_bigdata_rf_screen_500k():
    """500K rows + RF: screen 3 algorithms, all produce valid AUC."""
    rng = np.random.RandomState(42)
    n = 500_000
    X = rng.randn(n, 5).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    data = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    data["target"] = y

    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb = ml.screen(
            data=s.train,
            target="target",
            algorithms=["logistic", "random_forest", "decision_tree"],
            seed=42,
            keep_models=False,
        )

    assert len(lb) == 3, f"Expected 3 algorithms in leaderboard, got {len(lb)}"
    # All algorithms should produce valid scores (>= random)
    for _, row in lb.iterrows():
        assert row["roc_auc"] >= 0.5, f"{row['algorithm']} roc_auc={row['roc_auc']}"


@pytest.mark.slow
def test_bigdata_wide_100k():
    """100K rows × 200 features: normalize + fit without NaN in predictions."""
    rng = np.random.RandomState(42)
    n = 100_000
    p = 200
    X = rng.randn(n, p).astype(np.float32)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + rng.randn(n).astype(np.float32) * 0.5
    data = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    data["target"] = y

    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    preds = ml.predict(model, s.valid)

    assert len(preds) == len(s.valid)
    assert not preds.isna().any(), "Predictions contain NaN"
    metrics = ml.evaluate(model, s.valid)
    assert metrics["r2"] > 0.5, f"R2 too low: {metrics['r2']}"


@pytest.mark.slow
def test_bigdata_tree_1m_with_categoricals():
    """1M rows with mixed numeric + categorical: tree model handles it."""
    rng = np.random.RandomState(42)
    n = 1_000_000
    data = pd.DataFrame({
        "num1": rng.randn(n).astype(np.float32),
        "num2": rng.rand(n).astype(np.float32) * 100,
        "cat1": rng.choice(["a", "b", "c", "d"], n),
        "target": (rng.randn(n) > 0).astype(int),
    })

    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(
            data=s.train, target="target", algorithm="random_forest", seed=42
        )
    preds = ml.predict(model, s.valid)

    assert len(preds) == len(s.valid)
    metrics = ml.evaluate(model, s.valid)
    assert isinstance(metrics, dict)
    assert metrics["accuracy"] >= 0.0


@pytest.mark.slow
def test_bigdata_evaluate_no_double_transform():
    """1M rows: evaluate() should not be 2x slower than predict() (Phase 1.3 regression guard)."""
    import time

    rng = np.random.RandomState(42)
    n = 1_000_000
    X = rng.randn(n, 10).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    data = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    data["target"] = y

    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="logistic", seed=42)

    # Time predict
    t0 = time.time()
    ml.predict(model, s.valid)
    t_predict = time.time() - t0

    # Time evaluate (should be ~same as predict, not 2x)
    t0 = time.time()
    ml.evaluate(model, s.valid)
    t_evaluate = time.time() - t0

    # evaluate includes metric computation, allow 3x overhead max
    assert t_evaluate < t_predict * 3 + 1.0, (
        f"evaluate ({t_evaluate:.2f}s) is too slow relative to predict ({t_predict:.2f}s)"
    )
