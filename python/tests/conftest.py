"""Pytest configuration and fixtures.

Scope policy:
- session: immutable data — deterministic construction, safe to share across all tests
- module: shared fitted models that tests read but never mutate
- function: only when the test modifies the fixture or requires fresh state

Memory policy:
- ml.config(n_jobs=1) prevents fork-bomb from screen/tune/drift on constrained hardware
- gc.collect() + plt.close("all") after each test prevents accumulation
- BLAS thread caps (env vars) reduce per-worker memory multiplication
- Peak RSS ~250MB for full suite (759 tests). Verified at 128MB Docker limit.
"""

import os

# BLAS/OpenMP thread caps — must be set BEFORE numpy import.
# Belt-and-suspenders: even with n_jobs=1, caps any residual BLAS parallelism.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import gc

import numpy as np
import pandas as pd
import pytest

import ml

# Force single-threaded for tests — prevents fork-bomb on constrained hardware.
# screen/tune/drift default to n_jobs=-1 (all cores) for users, but honor this override.
# Guards off for algorithm tests — provenance tests enable explicitly.
ml.config(n_jobs=1, guards="off")

# Set matplotlib backend to non-interactive Agg for headless CI
try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass


@pytest.fixture(autouse=True)
def _gc_after_test():
    """Force garbage collection after every test.

    Prevents memory accumulation on constrained hardware (macbook Air 16GB,
    CI runners with 7GB). Cost: ~1ms per test. Prevents: OOM death spiral
    on systems without Linux OOM killer (macOS).

    Also resets guards to "off" — some tests set guards="strict" and state
    leaks into subsequent test files via the global _CONFIG dict.
    """
    ml.config(guards="off")
    yield
    # Close matplotlib figures (Agg buffer accumulation across tests)
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        pass
    gc.collect()


@pytest.fixture(scope="session")
def churn_data():
    """Churn dataset for testing (session-scoped — read-only across tests)."""
    return ml.dataset("churn")


@pytest.fixture(scope="session")
def small_classification_data():
    """Small classification dataset for fast tests."""
    rng = np.random.RandomState(123)
    n = 100
    return pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "x3": rng.choice(["a", "b", "c"], n),
        "target": rng.choice(["yes", "no"], n),
    })


@pytest.fixture(scope="session")
def small_regression_data():
    """Small regression dataset for fast tests."""
    rng = np.random.RandomState(123)
    n = 100
    return pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.rand(n) * 100,
    })


@pytest.fixture(scope="session")
def multiclass_data():
    """Multiclass classification dataset."""
    rng = np.random.RandomState(123)
    n = 200
    return pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice(["red", "green", "blue"], n),
    })


@pytest.fixture(scope="session")
def dirty_df():
    """Classification data with 5% NaN, mixed types, and a high-cardinality categorical.

    Use this to test that functions handle real-world messiness without crashing.
    """
    rng = np.random.RandomState(7)
    n = 200
    x1 = rng.randn(n).astype(float)
    x2 = rng.randn(n).astype(float)
    x3 = rng.choice(["cat", "dog", "bird", "fish"], n)
    # High-cardinality categorical (50 unique values)
    x4 = rng.choice([f"item_{i}" for i in range(50)], n)
    target = (x1 + x2 > 0).astype(int)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "target": target})

    # Inject ~5% NaN into numeric columns
    nan_idx_x1 = rng.choice(n, size=10, replace=False)
    nan_idx_x2 = rng.choice(n, size=10, replace=False)
    df.loc[nan_idx_x1, "x1"] = np.nan
    df.loc[nan_idx_x2, "x2"] = np.nan
    return df


@pytest.fixture(scope="session")
def imbalanced_df():
    """Binary classification with 95/5 class split.

    Use this to test imbalance warnings, roc_auc vs accuracy tradeoffs.
    """
    rng = np.random.RandomState(11)
    n = 120
    target = rng.choice([0, 1], n, p=[0.95, 0.05])
    return pd.DataFrame({
        "x1": rng.randn(n),
        "x2": rng.randn(n),
        "x3": rng.rand(n),
        "target": target,
    })


@pytest.fixture(scope="session")
def wide_df():
    """Wide dataset: p=50 features, n=100 rows (p > n/2).

    Use this to test regularization, collinearity, and screen() ranking.
    """
    rng = np.random.RandomState(17)
    n, p = 100, 50
    X = rng.randn(n, p)
    # Only first 3 features actually matter
    target = (X[:, 0] - X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)
    cols = {f"f{i:02d}": X[:, i] for i in range(p)}
    cols["target"] = target
    return pd.DataFrame(cols)


# ── Algorithm Registry (single source of truth for parametrized tests) ────────────
# Keys: tasks, proba, explain, deterministic, engine, needs_scaling,
#       handles_nan, handles_inf, min_samples, optional
#
# handles_nan: Rust CART/RF/ET do NOT handle NaN (cart.rs: undefined behaviour).
#              histgradient uses histogram binning — NaN goes to missing bin.
#              xgboost handles NaN natively (learned direction).
ALGORITHM_REGISTRY: dict = {
    "linear":             {"tasks": ["reg"],        "proba": False, "explain": True,  "deterministic": True,  "engine": "ml",      "needs_scaling": True,  "handles_nan": False, "handles_inf": False, "min_samples": 2},
    "logistic":           {"tasks": ["clf"],        "proba": True,  "explain": True,  "deterministic": True,  "engine": "ml",      "needs_scaling": True,  "handles_nan": False, "handles_inf": False, "min_samples": 2},
    "decision_tree":      {"tasks": ["clf", "reg"], "proba": True,  "explain": True,  "deterministic": True,  "engine": "ml",      "needs_scaling": False, "handles_nan": False, "handles_inf": False, "min_samples": 1},
    "random_forest":      {"tasks": ["clf", "reg"], "proba": True,  "explain": True,  "deterministic": False, "engine": "ml",      "needs_scaling": False, "handles_nan": False, "handles_inf": False, "min_samples": 1},
    "extra_trees":        {"tasks": ["clf", "reg"], "proba": True,  "explain": True,  "deterministic": False, "engine": "ml",      "needs_scaling": False, "handles_nan": False, "handles_inf": False, "min_samples": 1},
    "knn":                {"tasks": ["clf", "reg"], "proba": True,  "explain": False, "deterministic": True,  "engine": "ml",      "needs_scaling": True,  "handles_nan": False, "handles_inf": False, "min_samples": 2},
    "naive_bayes":        {"tasks": ["clf"],        "proba": True,  "explain": False, "deterministic": True,  "engine": "ml",      "needs_scaling": False, "handles_nan": False, "handles_inf": False, "min_samples": 2},
    "elastic_net":        {"tasks": ["reg"],        "proba": False, "explain": True,  "deterministic": True,  "engine": "ml",      "needs_scaling": True,  "handles_nan": False, "handles_inf": False, "min_samples": 2},
    "svm":                {"tasks": ["clf", "reg"], "proba": True,  "explain": False, "deterministic": True,  "engine": "ml",      "needs_scaling": True,  "handles_nan": False, "handles_inf": False, "min_samples": 2},
    "gradient_boosting":  {"tasks": ["clf", "reg"], "proba": True,  "explain": True,  "deterministic": False, "engine": "ml",      "needs_scaling": False, "handles_nan": False, "handles_inf": False, "min_samples": 2},
    "histgradient":       {"tasks": ["clf", "reg"], "proba": True,  "explain": True,  "deterministic": False, "engine": "ml",      "needs_scaling": False, "handles_nan": True,  "handles_inf": False, "min_samples": 2},
    "adaboost":           {"tasks": ["clf"],        "proba": True,  "explain": True,  "deterministic": False, "engine": "ml",      "needs_scaling": False, "handles_nan": False, "handles_inf": False, "min_samples": 2},
    "xgboost":            {"tasks": ["clf", "reg"], "proba": True,  "explain": True,  "deterministic": False, "engine": "sklearn", "needs_scaling": False, "handles_nan": True,  "handles_inf": False, "min_samples": 2, "optional": True},
}

# Derived lists — use these in @pytest.mark.parametrize
ALL_ALGOS        = list(ALGORITHM_REGISTRY)
ALL_CLF          = [k for k, v in ALGORITHM_REGISTRY.items() if "clf" in v["tasks"]]
ALL_REG          = [k for k, v in ALGORITHM_REGISTRY.items() if "reg" in v["tasks"]]
ALL_RUST         = [k for k, v in ALGORITHM_REGISTRY.items() if v["engine"] == "ml"]
ALL_RUST_CLF     = [k for k, v in ALGORITHM_REGISTRY.items() if v["engine"] == "ml" and "clf" in v["tasks"]]
ALL_RUST_REG     = [k for k, v in ALGORITHM_REGISTRY.items() if v["engine"] == "ml" and "reg" in v["tasks"]]
ALL_DETERMINISTIC = [k for k, v in ALGORITHM_REGISTRY.items() if v["deterministic"]]
ALL_STOCHASTIC   = [k for k, v in ALGORITHM_REGISTRY.items() if not v["deterministic"]]
ALL_TREE_BASED   = ["decision_tree", "random_forest", "extra_trees", "gradient_boosting", "histgradient", "adaboost"]
ALL_LINEAR       = ["linear", "logistic", "elastic_net", "svm", "knn"]
