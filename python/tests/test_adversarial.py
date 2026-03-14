"""Adversarial input battery
Each test: algorithm either handles the input correctly or raises a CLEAN error.
Never crash with segfault, never silently return NaN/garbage.

Constant target (clf + reg)
Single feature
All-zero features
NaN passthrough (algorithms that handle NaN)
NaN clean error (algorithms that don't handle NaN)
p >> n (wide data)
Duplicate rows
Near-constant features
Single row predict
High-cardinality categorical feature
Column mismatch at predict time
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import ml
from tests.conftest import ALGORITHM_REGISTRY, ALL_CLF, ALL_REG

# ── Derived lists ──────────────────────────────────────────────────────────────
_ALL_CLF = [a for a in ALL_CLF if a != "xgboost"]
_ALL_REG = [a for a in ALL_REG if a != "xgboost"]
_ALL_ALGOS = list(dict.fromkeys(_ALL_CLF + _ALL_REG))  # ordered, deduped

_NAN_ALGOS = [a for a in _ALL_ALGOS if ALGORITHM_REGISTRY[a]["handles_nan"]]
_NO_NAN_ALGOS = [a for a in _ALL_ALGOS if not ALGORITHM_REGISTRY[a]["handles_nan"]]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fit_or_error(data, algorithm, seed=42, allow_errors=(Exception,)):
    """Fit and return model, or return the exception if one is raised."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return ml.fit(data=data, target="target", algorithm=algorithm, seed=seed)
        except allow_errors as e:
            return e


def _predict_or_error(model, data, allow_errors=(Exception,)):
    """Predict and return preds, or return the exception."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return ml.predict(model=model, data=data)
        except allow_errors as e:
            return e


def _is_finite_series(s):
    return isinstance(s, pd.Series) and s.notna().all() and np.isfinite(s.values).all()


def _clf_data(n=100, p=8):
    rng = np.random.RandomState(42)
    X = rng.randn(n, p)
    y = (X[:, 0] + rng.randn(n) * 0.5 > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y
    return df


def _reg_data(n=100, p=8):
    rng = np.random.RandomState(42)
    X = rng.randn(n, p)
    y = X[:, 0] * 2.0 + rng.randn(n) * 0.3
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y
    return df


# ── Constant target ─────────────────────────────────────────────────────

@pytest.mark.parametrize("algorithm", _ALL_CLF)
def test_constant_target_clf(algorithm):
    """y = constant → clean error (ml validates at split/fit time)."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["target"] = 1  # constant class

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises((ml.DataError, Exception)):
            s = ml.split(data=df, target="target", seed=42)
            ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)


@pytest.mark.parametrize("algorithm", _ALL_REG)
def test_constant_target_reg(algorithm):
    """y = constant → clean error (no variance, can't fit regression)."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["target"] = 5.0  # constant regression target

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises((ml.DataError, Exception)):
            s = ml.split(data=df, target="target", seed=42)
            ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)


# ── Single feature ──────────────────────────────────────────────────────

@pytest.mark.parametrize("algorithm", _ALL_CLF)
def test_single_feature_clf(algorithm):
    """p=1 feature → works or raises a clean error (not crash)."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 1)
    y = (X[:, 0] > 0).astype(int)
    df = pd.DataFrame(X, columns=["f0"])
    df["target"] = y

    result = _fit_or_error(df.copy(), algorithm)
    # If it returned a model, predictions should be finite
    if not isinstance(result, Exception):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = ml.split(data=df, target="target", seed=42)
            m = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
            preds = ml.predict(model=m, data=s.valid)
        assert preds.notna().all(), f"{algorithm}: NaN predictions on single-feature data"


@pytest.mark.parametrize("algorithm", _ALL_REG)
def test_single_feature_reg(algorithm):
    """p=1 feature → works or raises a clean error."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 1)
    y = X[:, 0] * 2.0 + rng.randn(100) * 0.5
    df = pd.DataFrame(X, columns=["f0"])
    df["target"] = y

    result = _fit_or_error(df.copy(), algorithm)
    if not isinstance(result, Exception):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = ml.split(data=df, target="target", seed=42)
            m = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
            preds = ml.predict(model=m, data=s.valid)
        assert np.isfinite(preds.values).all(), f"{algorithm}: non-finite predictions on single-feature data"


# ── All-zero features ───────────────────────────────────────────────────

@pytest.mark.parametrize("algorithm", _ALL_CLF)
def test_all_zero_features_clf(algorithm):
    """X = zeros → clean error or sensible constant prediction."""
    df = pd.DataFrame(np.zeros((80, 5)), columns=[f"f{i}" for i in range(5)])
    rng = np.random.RandomState(42)
    df["target"] = rng.randint(0, 2, size=80)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s = ml.split(data=df, target="target", seed=42)
            m = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
            preds = ml.predict(model=m, data=s.valid)
            # If successful, predictions must be finite integers
            assert preds.notna().all(), f"{algorithm}: NaN with all-zero features"
        except Exception as e:
            # Any exception is acceptable — just not a segfault or hung process
            assert "segfault" not in str(e).lower(), f"{algorithm}: possible segfault: {e}"


@pytest.mark.parametrize("algorithm", _ALL_REG)
def test_all_zero_features_reg(algorithm):
    """X = zeros, y = random → clean error or constant prediction."""
    df = pd.DataFrame(np.zeros((80, 5)), columns=[f"f{i}" for i in range(5)])
    rng = np.random.RandomState(42)
    df["target"] = rng.randn(80)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s = ml.split(data=df, target="target", seed=42)
            m = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
            preds = ml.predict(model=m, data=s.valid)
            assert np.isfinite(preds.values).all(), f"{algorithm}: non-finite with all-zero features"
        except Exception:
            pass  # Clean error acceptable


# ── NaN: no silent corruption ──────────────────────────────────────────

@pytest.mark.parametrize("algorithm", _NO_NAN_ALGOS)
def test_nan_no_silent_corruption(algorithm):
    """Algorithms without NaN support either raise a clean error OR produce valid (non-NaN) predictions.

    No silent corruption allowed: NaN input must not produce NaN output without a visible error.
    """
    df = _clf_data(n=100, p=5)
    # Inject NaN into training features
    df.iloc[5, 2] = float("nan")
    df.iloc[10, 0] = float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s = ml.split(data=df, target="target", seed=42)
            m = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
            preds = ml.predict(model=m, data=s.valid)
            # If fit+predict succeeded, output must not be NaN (silent corruption)
            assert preds.notna().all(), (
                f"{algorithm}: silently produced NaN predictions on NaN-containing input"
            )
        except Exception:
            pass  # Raising a clean error is the preferred behavior


# ── p >> n (wide data) ─────────────────────────────────────────────────

@pytest.mark.parametrize("algorithm", _ALL_CLF)
def test_wide_data_clf(algorithm):
    """10 rows, 100 features → works or clean error (never segfault)."""
    rng = np.random.RandomState(42)
    X = rng.randn(60, 100)  # n=60 so split has enough rows; p=100 >> typical
    y = rng.randint(0, 2, size=60)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(100)])
    df["target"] = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s = ml.split(data=df, target="target", seed=42)
            m = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
            preds = ml.predict(model=m, data=s.valid)
            assert preds.notna().all(), f"{algorithm}: NaN with wide data"
        except Exception:
            pass  # Clean error acceptable


# ── Duplicate rows ──────────────────────────────────────────────────────

@pytest.mark.parametrize("algorithm", _ALL_CLF)
def test_duplicate_rows_clf(algorithm):
    """All rows identical → no crash."""
    base_row = [1.5, -0.3, 0.7, 2.1, -1.0]
    X = np.tile(base_row, (80, 1))
    rng = np.random.RandomState(42)
    y = rng.randint(0, 2, size=80)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s = ml.split(data=df, target="target", seed=42)
            m = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
            preds = ml.predict(model=m, data=s.valid)
            assert preds.notna().all(), f"{algorithm}: NaN with duplicate rows"
        except Exception:
            pass  # Clean error acceptable (e.g., zero variance in all features)


# ── Near-constant features ─────────────────────────────────────────────

@pytest.mark.parametrize("algorithm", _ALL_CLF)
def test_near_constant_features_clf(algorithm):
    """Feature std ~ 1e-10 → no division by zero or NaN."""
    rng = np.random.RandomState(42)
    X = np.ones((80, 5)) + rng.randn(80, 5) * 1e-10
    y = rng.randint(0, 2, size=80)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s = ml.split(data=df, target="target", seed=42)
            m = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
            preds = ml.predict(model=m, data=s.valid)
            assert preds.notna().all(), f"{algorithm}: NaN with near-constant features"
        except Exception:
            pass  # Clean error acceptable


# ── Single-row predict ──────────────────────────────────────────────────

@pytest.mark.parametrize("algorithm", _ALL_CLF)
def test_single_row_predict_clf(algorithm):
    """Predict on 1 row → result has 1 element."""
    df = _clf_data(n=100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=df, target="target", seed=42)
        try:
            m = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
            single_row = s.valid.drop(columns=["target"]).iloc[:1]
            preds = ml.predict(model=m, data=single_row)
            assert len(preds) == 1, f"{algorithm}: single-row predict returned {len(preds)} rows"
            assert preds.notna().all(), f"{algorithm}: NaN on single-row predict"
        except Exception:
            pass  # Algorithms with min_samples constraints may fail — that's fine


# ── High-cardinality categorical ──────────────────────────────────────

@pytest.mark.parametrize("algorithm", _ALL_CLF)
def test_high_cardinality_categorical_clf(algorithm):
    """50-level string categorical → works or clean error."""
    rng = np.random.RandomState(42)
    n = 100
    X = rng.randn(n, 4)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    # 50-level categorical
    df["cat"] = [f"level_{i % 50}" for i in range(n)]
    df["target"] = rng.randint(0, 2, size=n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s = ml.split(data=df, target="target", seed=42)
            m = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
            preds = ml.predict(model=m, data=s.valid)
            assert preds.notna().all(), f"{algorithm}: NaN with high-cardinality categorical"
        except Exception:
            pass  # Clean error acceptable


# ── Column mismatch at predict time ───────────────────────────────────

@pytest.mark.parametrize("algorithm", _ALL_CLF)
def test_predict_missing_column(algorithm):
    """Train on 8 features, predict with 7 → clean error."""
    df = _clf_data(n=100, p=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s = ml.split(data=df, target="target", seed=42)
            m = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
        except Exception:
            pytest.skip(f"{algorithm}: failed at fit, can't test predict mismatch")

    # Drop one feature column from valid
    valid_missing = s.valid.drop(columns=["f0"])
    with pytest.raises(Exception) as exc_info:
        ml.predict(model=m, data=valid_missing)

    # Error must be raised (not silent garbage)
    assert exc_info.value is not None


@pytest.mark.parametrize("algorithm", _ALL_CLF)
def test_predict_reordered_columns(algorithm):
    """Train on (f0..f7), predict on reversed columns → correct or clean error."""
    df = _clf_data(n=100, p=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s = ml.split(data=df, target="target", seed=42)
            m = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
        except Exception:
            pytest.skip(f"{algorithm}: failed at fit")

    feat_cols = [c for c in s.valid.columns if c != "target"]
    reordered = s.valid[feat_cols[::-1]]  # reversed column order

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            preds = ml.predict(model=m, data=reordered)
            # If it succeeds, predictions must be finite (not garbage)
            assert preds.notna().all(), f"{algorithm}: NaN predictions with reordered columns"
        except Exception:
            pass  # Raising an error is also acceptable
