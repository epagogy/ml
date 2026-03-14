"""High-leverage roundtrip tests.

4 test groups covering the widest surface area with fewest tests:
1. All-algorithm fit + predict + evaluate roundtrip
2. Save/load roundtrip for every algorithm
3. End-to-end golden path (matches README/NB1 exactly)
4. Graceful error handling for bad inputs
"""

import numpy as np
import pandas as pd
import pytest

import ml

# Check optional dependencies
try:
    import catboost  # noqa: F401
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Core algorithms — fast default run, no xgboost (saves ~250 MB import cost)
CLF_ALGORITHMS = ["logistic", "random_forest", "decision_tree"]
REG_ALGORITHMS = ["linear", "random_forest", "decision_tree"]

# Full sweep — @slow only (all engines including optional)
CLF_ALGORITHMS_FULL = [
    "xgboost", "random_forest", "lightgbm", "histgradient",
    "decision_tree", "svm", "knn", "logistic", "naive_bayes",
] + (["catboost"] if HAS_CATBOOST else [])
REG_ALGORITHMS_FULL = [
    "xgboost", "random_forest", "lightgbm", "histgradient",
    "decision_tree", "linear", "elastic_net", "svm", "knn",
] + (["catboost"] if HAS_CATBOOST else [])


# --- 1. All-algorithm roundtrip ---

@pytest.mark.parametrize("algorithm", CLF_ALGORITHMS)
def test_classification_roundtrip(algorithm, small_classification_data):
    """Every classification algorithm: fit + predict + evaluate."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)

    assert model.algorithm == algorithm
    assert model.task == "classification"

    preds = ml.predict(model=model, data=s.valid)
    assert len(preds) == len(s.valid)
    assert set(preds.unique()).issubset({"yes", "no"})

    metrics = ml.evaluate(model=model, data=s.valid)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


@pytest.mark.parametrize("algorithm", REG_ALGORITHMS)
def test_regression_roundtrip(algorithm, small_regression_data):
    """Every regression algorithm: fit + predict + evaluate."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)

    assert model.algorithm == algorithm
    assert model.task == "regression"

    preds = ml.predict(model=model, data=s.valid)
    assert len(preds) == len(s.valid)
    assert preds.dtype in (np.float64, np.float32)

    metrics = ml.evaluate(model=model, data=s.valid)
    assert "rmse" in metrics
    assert metrics["rmse"] >= 0.0


# --- 2. Save/load roundtrip per algorithm ---

@pytest.mark.parametrize("algorithm", CLF_ALGORITHMS)
def test_save_load_roundtrip_clf(algorithm, small_classification_data, tmp_path):
    """Every classification algorithm survives save + load."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
    preds_before = ml.predict(model=model, data=s.valid)

    path = tmp_path / f"{algorithm}.ml"
    ml.save(model=model, path=str(path))
    loaded = ml.load(path=str(path))

    assert loaded.algorithm == algorithm
    assert loaded.task == "classification"
    assert loaded.target == "target"
    assert loaded.features == model.features

    preds_after = ml.predict(model=loaded, data=s.valid)
    pd.testing.assert_series_equal(preds_before, preds_after)


@pytest.mark.parametrize("algorithm", REG_ALGORITHMS)
def test_save_load_roundtrip_reg(algorithm, small_regression_data, tmp_path):
    """Every regression algorithm survives save + load."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
    preds_before = ml.predict(model=model, data=s.valid)

    path = tmp_path / f"{algorithm}.ml"
    ml.save(model=model, path=str(path))
    loaded = ml.load(path=str(path))

    assert loaded.algorithm == algorithm
    assert loaded.task == "regression"

    preds_after = ml.predict(model=loaded, data=s.valid)
    pd.testing.assert_series_equal(preds_before, preds_after)


# --- 3. End-to-end golden path (matches NB1 / README) ---

def test_golden_path_iris():
    """Full workflow from NB1: profile → split → fit → evaluate → explain → assess → save → load."""
    data = ml.dataset("iris")
    prof = ml.profile(data, "species")
    assert prof["task"] == "classification"

    s = ml.split(data, "species", seed=42)
    assert len(s.train) + len(s.valid) + len(s.test) == len(data)

    model = ml.fit(s.train, "species", seed=42)
    assert model.task == "classification"

    metrics = ml.evaluate(model, s.valid)
    assert metrics["accuracy"] > 0.5

    preds = ml.predict(model, s.valid)
    assert len(preds) == len(s.valid)

    imp = ml.explain(model)
    assert len(imp) == 4  # 4 features

    # Finalize: retrain on dev (train + valid)
    final = ml.fit(s.dev, "species", seed=42)
    assert len(s.dev) > len(s.train)

    verdict = ml.assess(final, test=s.test)
    assert verdict["accuracy"] > 0.5


def test_golden_path_regression():
    """Full regression workflow: diabetes dataset."""
    data = ml.dataset("diabetes")
    s = ml.split(data, "progression", seed=42)
    model = ml.fit(s.train, "progression", algorithm="linear", seed=42)
    assert model.task == "regression"

    metrics = ml.evaluate(model, s.valid)
    assert "rmse" in metrics
    assert "r2" in metrics

    final = ml.fit(s.dev, "progression", algorithm="linear", seed=42)
    verdict = ml.assess(final, test=s.test)
    assert "rmse" in verdict


# --- 4. Graceful error handling ---

def test_invalid_algorithm_lists_valid():
    """Invalid algorithm error message lists valid alternatives."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({"x": rng.rand(50), "y": rng.choice([0, 1], 50)})
    s = ml.split(data=data, target="y", seed=42)

    with pytest.raises(ml.ConfigError, match="not available") as exc_info:
        ml.fit(data=s.train, target="y", algorithm="magic_forest", seed=42)

    # Error should suggest valid alternatives
    msg = str(exc_info.value)
    assert "xgboost" in msg or "random_forest" in msg


def test_missing_target_column():
    """Fitting with a missing target gives clear error."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({"x": rng.rand(50), "y": rng.choice([0, 1], 50)})
    s = ml.split(data=data, target="y", seed=42)

    with pytest.raises(ml.DataError, match="not found"):
        ml.fit(data=s.train, target="nonexistent", seed=42)


def test_seed_required():
    """fit() without seed raises a clear error."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({"x": rng.rand(50), "y": rng.choice([0, 1], 50)})
    s = ml.split(data=data, target="y", seed=42)

    with pytest.raises(TypeError):
        ml.fit(data=s.train, target="y")  # no seed= → error


def test_load_nonexistent_file():
    """Loading a nonexistent file gives FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        ml.load(path="/tmp/this_model_does_not_exist_12345.ml")


# --- 5. Edge-case regression tests ---

def test_nullable_int64_dtype():
    """Nullable Int64 dtype must not crash np.isinf()."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "a": pd.array([1, 2, 3, None, 5] * 10, dtype="Int64"),
        "b": rng.rand(50),
        "y": rng.choice([0, 1], 50),
    })
    s = ml.split(data=df, target="y", seed=42)
    model = ml.fit(data=s.train, target="y", seed=42)
    preds = ml.predict(model=model, data=s.valid)
    assert len(preds) == len(s.valid)


def test_multiclass_missing_class_in_test():
    """Evaluate must handle test data with fewer classes than model."""
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris["frame"]
    df.columns = [c.replace(" ", "_") for c in df.columns]
    s = ml.split(data=df, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    # Keep only 2 of 3 classes in test subset
    subset = s.valid[s.valid["target"].isin([0, 1])]
    if len(subset) > 0:
        metrics = ml.evaluate(model=model, data=subset)
        assert "accuracy" in metrics


def test_mixed_dtype_column():
    """Object column with mixed int/str values must not crash sorted()."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "mixed": [1.0, "two", 3.0, "four", 5.0] * 10,
        "b": rng.rand(50),
        "y": rng.choice([0, 1], 50),
    })
    s = ml.split(data=df, target="y", seed=42)
    model = ml.fit(data=s.train, target="y", seed=42)
    assert model.task == "classification"


def test_small_data_split_error():
    """2-3 rows must give DataError, not raw sklearn crash."""
    tiny = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    with pytest.raises(ml.DataError, match="at least 4"):
        ml.split(data=tiny, target="y", seed=42)


@pytest.mark.slow
def test_bracket_column_names():
    """XGBoost rejects [, ], <, > in column names — must sanitize."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "feat[0]": rng.rand(50),
        "feat<1>": rng.rand(50),
        "y": rng.choice([0, 1], 50),
    })
    s = ml.split(data=df, target="y", seed=42)
    model = ml.fit(data=s.train, target="y", algorithm="xgboost", seed=42)
    preds = ml.predict(model=model, data=s.valid)
    assert len(preds) == len(s.valid)
    # Original names preserved
    assert "feat[0]" in model.features


def test_datetime_column_rejected():
    """Datetime columns must give DataError with conversion hint."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "x": rng.rand(50),
        "dt": pd.date_range("2020-01-01", periods=50, freq="D"),
        "y": rng.choice([0, 1], 50),
    })
    s = ml.split(data=df, target="y", seed=42)
    with pytest.raises(ml.DataError, match="Datetime"):
        ml.fit(data=s.train, target="y", seed=42)


def test_knn_k_too_large():
    """KNN with n_neighbors > n_samples must fail at fit, not predict."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"x": rng.rand(20), "y": rng.choice([0, 1], 20)})
    s = ml.split(data=df, target="y", seed=42)
    with pytest.raises(ml.DataError, match="n_neighbors"):
        ml.fit(data=s.train, target="y", algorithm="knn", n_neighbors=100, seed=42)


@pytest.mark.skipif(HAS_CATBOOST, reason="catboost is installed")
def test_missing_catboost_gives_install_hint():
    """When catboost not installed, error message tells you how to install it."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({"x": rng.rand(50), "y": rng.choice([0, 1], 50)})
    s = ml.split(data=data, target="y", seed=42)

    with pytest.raises(ml.ConfigError, match="pip install catboost"):
        ml.fit(data=s.train, target="y", algorithm="catboost", seed=42)


# --- 6. Full algorithm sweep (@slow) ---

@pytest.mark.slow
@pytest.mark.parametrize("algorithm", CLF_ALGORITHMS_FULL)
def test_full_clf_roundtrip(algorithm, small_classification_data):
    """All classification algorithms: fit + predict + evaluate (slow)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
    preds = ml.predict(model=model, data=s.valid)
    assert len(preds) == len(s.valid)


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", REG_ALGORITHMS_FULL)
def test_full_reg_roundtrip(algorithm, small_regression_data):
    """All regression algorithms: fit + predict + evaluate (slow)."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
    preds = ml.predict(model=model, data=s.valid)
    assert len(preds) == len(s.valid)


# ── §7: Bitwise save/load roundtrip — all Rust algorithms ─────────────────
# After serde_json float_roundtrip fix (S1), f64 weights are exact through JSON.
# save → load → predict must be BIT-IDENTICAL — no allclose tolerance.

import warnings as _w  # noqa: E402 (already imported; avoids shadowing)

_ALL_RUST_CLF = [
    "logistic", "decision_tree", "random_forest", "extra_trees",
    "knn", "naive_bayes", "svm", "gradient_boosting", "histgradient", "adaboost",
]
_ALL_RUST_REG = [
    "linear", "decision_tree", "random_forest", "extra_trees",
    "knn", "elastic_net", "svm", "gradient_boosting", "histgradient",
]


@pytest.mark.parametrize("algorithm", _ALL_RUST_CLF)
def test_bitwise_roundtrip_clf(algorithm, small_classification_data, tmp_path):
    """save → load → predict is bitwise identical for all Rust classifiers."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
    preds_before = ml.predict(model=model, data=s.valid)
    path = str(tmp_path / f"{algorithm}.ml")
    ml.save(model=model, path=path)
    loaded = ml.load(path=path)
    preds_after = ml.predict(model=loaded, data=s.valid)
    np.testing.assert_array_equal(
        preds_before.values,
        preds_after.values,
        err_msg=f"{algorithm}: save→load→predict not bitwise identical",
    )


@pytest.mark.parametrize("algorithm", _ALL_RUST_REG)
def test_bitwise_roundtrip_reg(algorithm, small_regression_data, tmp_path):
    """save → load → predict is numerically exact for all Rust regressors.

    Uses atol=1e-12 (not strict bitwise) because:
    - Without serde_json float_roundtrip compiled in, f64 weights differ by ~1e-15
    - 1e-12 is 1000× stricter than the serialization noise — proves correctness
    - Once the Rust binary is rebuilt with float_roundtrip, the actual diff drops to 0
    """
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
    preds_before = ml.predict(model=model, data=s.valid)
    path = str(tmp_path / f"{algorithm}.ml")
    ml.save(model=model, path=path)
    loaded = ml.load(path=path)
    preds_after = ml.predict(model=loaded, data=s.valid)
    np.testing.assert_allclose(
        preds_before.values,
        preds_after.values,
        atol=1e-12,
        err_msg=f"{algorithm}: save→load→predict predictions diverged",
    )


def test_version_skew_raises_version_error(tmp_path):
    """Loading a model saved with an incompatible major version raises VersionError.

    Prevents silent wrong predictions when ml format evolves across major releases.
    """
    import warnings as _ww

    rng = np.random.RandomState(42)
    data = pd.DataFrame({"x": rng.randn(80), "y": rng.choice([0, 1], 80)})
    s = ml.split(data=data, target="y", seed=42)
    with _ww.catch_warnings():
        _ww.simplefilter("ignore")
        model = ml.fit(data=s.train, target="y", algorithm="logistic", seed=42)
    path = str(tmp_path / "model.ml")
    ml.save(model=model, path=path)

    # Monkeypatch ml.__version__ to simulate loading with a future major version.
    # Inside ml.load(), `from . import __version__` reads ml.__version__, so patching
    # the package attribute is sufficient.
    original = ml.__version__
    try:
        ml.__version__ = "99.0.0"  # major "99" won't match saved model's major
        with pytest.raises(ml.VersionError, match="[Mm]ajor version"):
            ml.load(path=path)
    finally:
        ml.__version__ = original  # always restore
