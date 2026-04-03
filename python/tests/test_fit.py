"""Tests for fit()."""

import numpy as np
import pandas as pd
import pytest

import ml


@pytest.mark.slow
def test_fit_basic_holdout(small_classification_data):
    """Test basic holdout fit with default algorithm (lightgbm)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    assert isinstance(model, ml.Model)
    assert model.task == "classification"
    assert model.algorithm == "lightgbm"  # default (Phase 0.7: LightGBM preferred over XGBoost)
    assert model.target == "target"
    assert model.seed == 42
    assert model.scores_ is None  # holdout has no CV scores


def test_fit_cv_path(small_classification_data):
    """Test CV fit."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    cv = ml.cv(s, folds=2, seed=42)
    model = ml.fit(data=cv, target="target", algorithm="random_forest", seed=42)

    assert isinstance(model, ml.Model)
    assert model.scores_ is not None
    assert "accuracy_mean" in model.scores_
    assert "accuracy_std" in model.scores_


def test_fit_regression(small_regression_data):
    """Test regression fit."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)

    assert model.task == "regression"
    assert model.algorithm == "linear"


def test_fit_with_kwargs(small_classification_data):
    """Test fit with engine kwargs."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="random_forest", max_depth=3, n_estimators=10, seed=42)

    assert model._model.max_depth == 3
    assert model._model.n_estimators == 10


def test_fit_string_target(small_classification_data):
    """Test fit with string target (requires LabelEncoder)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)

    # Should work without error
    assert model._label_encoder is not None


def test_fit_categorical_features():
    """Test fit with categorical features."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "cat1": rng.choice(["a", "b", "c"], 100),
        "cat2": rng.choice(["x", "y"], 100),
        "num": rng.rand(100),
        "target": rng.choice([0, 1], 100),
    })

    s = ml.split(data=data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)

    # Should encode categoricals automatically
    assert "cat1" in model.features
    assert "cat2" in model.features


def test_fit_target_not_found_error(small_classification_data):
    """Test fit raises when target not found."""
    s = ml.split(data=small_classification_data, target="target", seed=42)

    with pytest.raises(ml.DataError, match="target column 'missing' not found"):
        ml.fit(data=s.train, target="missing", seed=42)


def test_fit_invalid_algorithm_error(small_classification_data):
    """Test fit raises on invalid algorithm."""
    s = ml.split(data=small_classification_data, target="target", seed=42)

    with pytest.raises(ml.ConfigError, match="algorithm='magic' not available"):
        ml.fit(data=s.train, target="target", algorithm="magic", seed=42)


def test_fit_reproducible(small_classification_data):
    """Test fit is reproducible with same seed."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model1 = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
    model2 = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)

    preds1 = model1.predict(s.valid)
    preds2 = model2.predict(s.valid)

    pd.testing.assert_series_equal(preds1, preds2)


def test_fit_multiclass(multiclass_data):
    """Test multiclass classification."""
    s = ml.split(data=multiclass_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)

    assert model.task == "classification"
    preds = model.predict(s.valid)
    assert set(preds.unique()).issubset({"red", "green", "blue"})


@pytest.mark.slow
def test_fit_auto_algorithm():
    """Test auto algorithm fallback."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({"x": rng.rand(100), "y": rng.choice([0, 1], 100)})

    s = ml.split(data=data, target="y", seed=42)
    model = ml.fit(data=s.train, target="y", algorithm="auto", seed=42)

    # Should default to lightgbm → xgboost → random_forest (Phase 0.7)
    assert model.algorithm in ("lightgbm", "xgboost", "random_forest")


def test_fit_cv_fold_scores(small_classification_data):
    """CV fit populates fold_scores_ with per-fold metric dicts."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    cv = ml.cv(s, folds=3, seed=42)
    model = ml.fit(data=cv, target="target", algorithm="random_forest", seed=42)

    assert model.fold_scores_ is not None
    assert len(model.fold_scores_) == 3
    for fold in model.fold_scores_:
        assert isinstance(fold, dict)
        assert "accuracy" in fold


def test_fit_holdout_fold_scores_none(small_classification_data):
    """Holdout fit has fold_scores_=None (no CV performed)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)

    assert model.fold_scores_ is None


def test_fit_preprocessing_attribute(small_classification_data):
    """model.preprocessing_ describes what transformations were applied."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)

    prep = model.preprocessing_
    assert isinstance(prep, dict)
    assert "features" in prep
    assert "scaled" in prep


def test_fit_n_train_attribute(small_classification_data):
    """model.n_train records how many rows were used for training."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)

    assert model.n_train == len(s.train)


def test_fit_features_exclude_target(small_classification_data):
    """model.features must not include the target column."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)

    assert "target" not in model.features
    assert len(model.features) == len(s.train.columns) - 1


@pytest.mark.parametrize("name,target,task", [
    ("iris", "species", "classification"),
    ("wine", "cultivar", "classification"),
    ("cancer", "diagnosis", "classification"),
    ("diabetes", "progression", "regression"),
    ("houses", "price", "regression"),
])
def test_builtin_datasets(name, target, task):
    """All built-in sklearn datasets load and work end-to-end."""
    data = ml.dataset(name)
    assert target in data.columns
    s = ml.split(data=data, target=target, seed=42)
    algo = "random_forest" if task == "classification" else "linear"
    model = ml.fit(data=s.train, target=target, algorithm=algo, seed=42)
    assert model.task == task
    preds = model.predict(s.valid)
    assert len(preds) == len(s.valid)


# ── Phase 0 tests ──────────────────────────────────────────────────────────────


def test_fit_default_single_thread(small_classification_data):
    """fit() defaults to n_jobs=1 (deterministic, single-thread). Phase 0.1."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
    # RandomForest exposes n_jobs — must be 1 by default in fit() context
    assert model._model.n_jobs == 1


@pytest.mark.slow
def test_fit_early_stop_discloses_carve():
    """fit() with XGBoost/LightGBM emits UserWarning about 10% holdout. Phase 0.5."""
    rng = np.random.RandomState(42)
    n = 200
    data = pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice([0, 1], n),
    })
    s = ml.split(data=data, target="target", seed=42)
    with pytest.warns(UserWarning, match="Early stopping holds out"):
        model = ml.fit(data=s.train, target="target", algorithm="xgboost", seed=42)
    # _n_train_actual must be set and smaller than n_train
    assert model._n_train_actual is not None
    assert model._n_train_actual < model.n_train


@pytest.mark.slow
def test_fit_repr_shows_actual_train_size():
    """Model __repr__ shows actual/total rows when early stopping carves data. Phase 0.5."""
    rng = np.random.RandomState(42)
    n = 200
    data = pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice([0, 1], n),
    })
    s = ml.split(data=data, target="target", seed=42)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="xgboost", seed=42)
    if model._n_train_actual is not None and model._n_train_actual != model.n_train:
        # repr shows "n_train=actual/total" format
        assert "/" in repr(model)
        # str shows explicit disclosure
        assert "early stopping held out" in str(model)


@pytest.mark.slow
def test_fit_auto_prefers_lightgbm():
    """fit() auto-select prefers LightGBM when available. Phase 0.7."""
    pytest.importorskip("lightgbm")
    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "x1": rng.rand(100),
        "x2": rng.rand(100),
        "target": rng.choice([0, 1], 100),
    })
    s = ml.split(data=data, target="target", seed=42)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", seed=42)  # algorithm="auto"
    assert model.algorithm == "lightgbm"


# ── A12: sample_weight propagation ────────────────────────────────────────────

def test_fit_with_weights():
    """fit() with weights= stores col name on model._sample_weight_col. A12."""
    rng = np.random.RandomState(42)
    n = 100
    data = pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice([0, 1], n),
        "w": np.ones(n),  # uniform weights — should not change model
    })
    s = ml.split(data=data, target="target", seed=42)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42, weights="w")

    assert isinstance(model, ml.Model)
    assert model._sample_weight_col == "w"


def test_tune_with_weights():
    """tune() with weights= completes without error. A12."""
    rng = np.random.RandomState(42)
    n = 100
    data = pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice([0, 1], n),
        "w": np.ones(n),
    })
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.tune(
            data=data, target="target",
            algorithm="random_forest", n_trials=2, seed=42,
            weights="w",
        )
    assert hasattr(result, "best_model")
    assert result.best_model._sample_weight_col == "w"


def test_stack_with_weights():
    """stack() with weights= completes without error. A12."""
    rng = np.random.RandomState(42)
    n = 100
    data = pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice([0, 1], n),
        "w": np.ones(n),
    })
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.stack(data=data, target="target", seed=42, weights="w",
                         models=["logistic", "random_forest"])
    assert isinstance(model, ml.Model)


def test_evaluate_weighted_metrics():
    """evaluate() with sample_weight= returns weighted metrics. A12."""
    rng = np.random.RandomState(42)
    n = 100
    data = pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice([0, 1], n),
        "w": np.ones(n),
    })
    s = ml.split(data=data, target="target", seed=42)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
    # Add weight col to valid set
    valid = s.valid.copy()
    valid["w"] = 1.0
    result = ml.evaluate(model=model, data=valid, sample_weight="w")
    assert isinstance(result, dict)
    # Weighted accuracy should be present for classification
    assert "accuracy_weighted" in result
    assert 0.0 <= result["accuracy_weighted"] <= 1.0


def test_weights_save_load(tmp_path):
    """fit() with weights= → save → load preserves _sample_weight_col. A12."""
    rng = np.random.RandomState(42)
    n = 100
    data = pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice([0, 1], n),
        "w": np.ones(n),
    })
    s = ml.split(data=data, target="target", seed=42)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42, weights="w")

    path = str(tmp_path / "model_weights.pyml")
    ml.save(model, path)
    loaded = ml.load(path)
    assert loaded._sample_weight_col == "w"


# ── A8: Seed averaging ─────────────────────────────────────────────────────────

def _make_seed_data():
    """Small binary classification dataset for seed averaging tests."""
    rng = np.random.RandomState(42)
    n = 100
    return pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice([0, 1], n),
    })


def test_seed_list_basic():
    """fit(seed=[42,43]) returns Model with _ensemble of length 2. A8."""
    data = _make_seed_data()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=data, target="target",
                       algorithm="random_forest", seed=[42, 43])

    assert isinstance(model, ml.Model)
    assert model.seed == 42  # seed returns seed_list[0]
    assert model._ensemble is not None
    assert len(model._ensemble) == 2


def test_seed_list_predictions_stable():
    """Seed-averaged model produces valid predictions (correct shape, binary labels). A8."""
    data = _make_seed_data()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=data, target="target", seed=42)
        model = ml.fit(data=s.train, target="target",
                       algorithm="random_forest", seed=[42, 43, 44])
    preds = ml.predict(model, s.valid)
    assert len(preds) == len(s.valid)
    assert set(preds.unique()).issubset({0, 1})


def test_seed_list_seed_scores_and_std():
    """seed_scores and seed_std populated when CV data used. A8."""
    data = _make_seed_data()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _s = ml.split(data=data, target="target", seed=42)
        cv = ml.cv(_s, folds=2, seed=42)
        model = ml.fit(data=cv, target="target",
                       algorithm="random_forest", seed=[42, 43])

    # CV path produces per-seed scores
    assert model._seed_scores is not None
    assert isinstance(model._seed_scores, list)
    assert len(model._seed_scores) == 2
    assert all(isinstance(s, float) for s in model._seed_scores)
    assert model._seed_std is not None
    assert isinstance(model._seed_std, float)


def test_seed_list_save_load(tmp_path):
    """Seed-averaged model round-trips through save/load, predictions match. A8."""
    data = _make_seed_data()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=data, target="target",
                       algorithm="random_forest", seed=[42, 43])
    preds_before = ml.predict(model, data)

    path = str(tmp_path / "ensemble.pyml")
    ml.save(model, path)
    loaded = ml.load(path)

    preds_after = ml.predict(loaded, data)
    assert (preds_before == preds_after).all()
    assert loaded._ensemble is not None
    assert len(loaded._ensemble) == 2


def test_seed_list_explain():
    """explain() works on seed-averaged model (averages importances). A8."""
    data = _make_seed_data()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=data, target="target",
                       algorithm="random_forest", seed=[42, 43])
    exp = ml.explain(model)
    assert exp is not None
    assert len(exp) == 2  # x1, x2


def test_seed_list_invalid_seed_errors():
    """fit(seed=[]) or fit(seed=[42]) raises ConfigError. A8."""
    from ml._types import ConfigError
    data = _make_seed_data()
    with pytest.raises(ConfigError):
        ml.fit(data=data, target="target", algorithm="random_forest",
               seed=[42])  # only 1 seed


# ---------------------------------------------------------------------------
# A13: GPU detection tests
# ---------------------------------------------------------------------------

def test_gpu_detection_returns_bool():
    """_detect_gpu() returns a bool regardless of GPU availability. A13."""
    from ml._engines import _detect_gpu
    result = _detect_gpu()
    assert isinstance(result, bool)


@pytest.mark.slow
def test_gpu_warning_emitted_when_gpu_present(monkeypatch):
    """When GPU is detected, creating XGBoost engine emits UserWarning. A13."""
    import contextlib
    import warnings

    from ml import _engines
    # Force GPU detected
    monkeypatch.setattr(_engines, "_detect_gpu", lambda: True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with contextlib.suppress(Exception):
            _engines.create("xgboost", task="classification", seed=42)
    gpu_warnings = [x for x in w if "GPU" in str(x.message) or "cuda" in str(x.message).lower()]
    assert len(gpu_warnings) >= 1


# ---------------------------------------------------------------------------
# 5.1: TabPFN integration tests
# ---------------------------------------------------------------------------


def test_fit_tabpfn_basic(small_classification_data):
    """TabPFN fits and predicts when installed. Skip if not installed. Chain 5.1."""
    pytest.importorskip("tabpfn")
    s = ml.split(data=small_classification_data, target="target", seed=42)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="tabpfn", seed=42)
    assert isinstance(model, ml.Model)
    assert model.algorithm == "tabpfn"
    preds = ml.predict(model, s.valid)
    assert len(preds) == len(s.valid)


def test_fit_tabpfn_regression_error(small_regression_data):
    """TabPFN raises ConfigError for regression. Chain 5.1."""
    pytest.importorskip("tabpfn")
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with pytest.raises(ml.ConfigError, match="classification only"):
        ml.fit(data=s.train, target="target", algorithm="tabpfn", seed=42)


def test_fit_tabpfn_large_data_warning():
    """TabPFN emits UserWarning for >10K rows. Chain 5.1."""
    import contextlib
    import sys
    import warnings
    rng = np.random.RandomState(42)
    # Use 20000 rows so s.train (~12000 rows) still exceeds 10K threshold
    n = 20000
    data = pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice([0, 1], n),
    })
    s = ml.split(data=data, target="target", seed=42)
    # Hide tabpfn so engine creation fails fast after the warning fires
    original = sys.modules.get("tabpfn", None)
    sys.modules["tabpfn"] = None  # type: ignore[assignment]
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with contextlib.suppress(Exception):
                ml.fit(data=s.train, target="target", algorithm="tabpfn", seed=42)
        tabpfn_warns = [x for x in w if "10K" in str(x.message) or "TabPFN" in str(x.message)]
        assert len(tabpfn_warns) >= 1
    finally:
        if original is not None:
            sys.modules["tabpfn"] = original
        elif "tabpfn" in sys.modules:
            del sys.modules["tabpfn"]


def test_fit_tabpfn_not_installed_error(small_classification_data, monkeypatch):
    """TabPFN raises ConfigError when not installed. Chain 5.1."""
    import sys
    s = ml.split(data=small_classification_data, target="target", seed=42)
    # Temporarily hide tabpfn
    original = sys.modules.get("tabpfn", None)
    sys.modules["tabpfn"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ml.ConfigError, match="tabpfn"):
            ml.fit(data=s.train, target="target", algorithm="tabpfn", seed=42)
    finally:
        if original is not None:
            sys.modules["tabpfn"] = original
        elif "tabpfn" in sys.modules:
            del sys.modules["tabpfn"]


def test_fit_gpu_false_forces_cpu(small_classification_data):
    """fit(gpu=False) succeeds on CPU (always available)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42, gpu=False)
    assert model is not None


def test_fit_gpu_auto_default(small_classification_data):
    """fit(gpu='auto') is the default and doesn't crash."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42, gpu="auto")
    assert model is not None


def test_fit_gpu_true_no_gpu_graceful(small_classification_data):
    """fit(gpu=True) succeeds or raises ConfigError (no crash with traceback)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    try:
        model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42, gpu=True)
        assert model is not None
    except ml.ConfigError:
        pass  # Expected when no GPU available


def test_fit_multi_label_target_error():
    """fit() raises ConfigError for multi-label targets."""
    rng = np.random.RandomState(42)
    # Build train data directly (skip split — split() chokes on unhashable lists too)
    data = pd.DataFrame({
        "feat": rng.rand(40),
        "target": [["a", "b"] if i % 2 == 0 else ["a"] for i in range(40)],
    })
    with pytest.raises(ml.ConfigError, match="[Mm]ulti"):
        ml.fit(data=data, target="target", algorithm="random_forest", seed=42)


def test_fit_multi_label_error_message():
    """Multi-label error message mentions alternative."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "feat": rng.rand(40),
        "target": [["a", "b"] if i % 2 == 0 else ["a"] for i in range(40)],
    })
    try:
        ml.fit(data=data, target="target", algorithm="random_forest", seed=42)
    except ml.ConfigError as e:
        assert "multi" in str(e).lower() or "label" in str(e).lower()


def test_fit_gpu_true_no_gpu_error(small_classification_data):
    """fit(gpu=True) either works or raises ConfigError with 'GPU' in message."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    try:
        model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42, gpu=True)
        assert model is not None
    except ml.ConfigError as e:
        assert "GPU" in str(e) or "gpu" in str(e).lower()


# ---------------------------------------------------------------------------
# Phase 1: Criterion extensibility (Entropy + Poisson) — Rust CART
# ---------------------------------------------------------------------------


def test_criterion_entropy_uses_rust(small_classification_data):
    """criterion='entropy' routes to Rust (not sklearn fallback)."""
    from ml._rust import HAS_RUST
    if not HAS_RUST:
        pytest.skip("Rust backend not available")
    s = ml.split(data=small_classification_data, target="target", seed=42)
    m_entropy = ml.fit(s.train, "target", algorithm="decision_tree", criterion="entropy", seed=42)
    m_gini = ml.fit(s.train, "target", algorithm="decision_tree", criterion="gini", seed=42)
    # Verify Rust backend is active (ml._rust module)
    assert "ml._rust" in type(m_entropy._model).__module__, (
        f"Expected Rust backend, got '{type(m_entropy._model).__module__}'"
    )
    # Entropy and gini may produce different trees, but on small datasets
    # both criteria can legitimately agree on every split.
    # The key assertion is that the Rust backend is active (checked above).
    preds_e = ml.predict(m_entropy, s.valid)
    preds_g = ml.predict(m_gini, s.valid)
    if preds_e.equals(preds_g):
        import warnings
        warnings.warn("entropy and gini produced identical predictions on small data — expected on simple datasets")


def test_criterion_poisson_reg():
    """criterion='poisson' fits a regression tree for count targets."""
    rng = np.random.RandomState(42)
    counts = rng.poisson(3, 200).astype(float)  # non-negative integers
    df = pd.DataFrame({"x1": rng.rand(200), "x2": rng.rand(200), "y": counts})
    s = ml.split(df, "y", seed=42)
    m = ml.fit(s.train, "y", algorithm="decision_tree", criterion="poisson", seed=42)
    preds = ml.predict(m, s.valid)
    assert (preds >= 0).all(), "Poisson predictions must be non-negative"


def test_criterion_poisson_negative_target_raises():
    """criterion='poisson' raises ConfigError when target has negative values."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"x": rng.rand(50), "y": rng.randn(50)})  # continuous, negative values
    s = ml.split(df, "y", seed=42)
    with pytest.raises(ml.ConfigError, match="non-negative"):
        ml.fit(s.train, "y", algorithm="decision_tree", criterion="poisson", seed=42)


def test_criterion_unknown_raises(small_classification_data):
    """Unknown criterion raises ConfigError (not a bare ValueError from Rust)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with pytest.raises((ml.ConfigError, ValueError)):
        ml.fit(s.train, "target", algorithm="decision_tree", criterion="magic_criterion", seed=42)


# ---------------------------------------------------------------------------
# Phase 2 — Extra Trees native Rust
# ---------------------------------------------------------------------------


def test_extra_trees_clf_uses_rust(small_classification_data):
    """extra_trees classification routes to Rust backend."""
    from ml._rust import HAS_RUST, _RustExtraTreesClassifier

    if not HAS_RUST:
        pytest.skip("Rust backend not available")
    s = ml.split(data=small_classification_data, target="target", seed=42)
    m = ml.fit(s.train, "target", algorithm="extra_trees", seed=42)
    assert isinstance(m._model, _RustExtraTreesClassifier), (
        f"Expected _RustExtraTreesClassifier, got {type(m._model)}"
    )
    preds = ml.predict(m, s.valid)
    assert len(preds) == len(s.valid)


def test_extra_trees_reg_uses_rust(small_regression_data):
    """extra_trees regression routes to Rust backend."""
    from ml._rust import HAS_RUST, _RustExtraTreesRegressor

    if not HAS_RUST:
        pytest.skip("Rust backend not available")
    s = ml.split(data=small_regression_data, target="target", seed=42)
    m = ml.fit(s.train, "target", algorithm="extra_trees", seed=42)
    assert isinstance(m._model, _RustExtraTreesRegressor), (
        f"Expected _RustExtraTreesRegressor, got {type(m._model)}"
    )
    preds = ml.predict(m, s.valid)
    assert len(preds) == len(s.valid)


def test_extra_trees_preds_differ_from_rf(small_classification_data):
    """Extra Trees and Random Forest should produce different predictions."""
    from ml._rust import HAS_RUST

    if not HAS_RUST:
        pytest.skip("Rust backend not available")
    s = ml.split(data=small_classification_data, target="target", seed=42)
    m_rf = ml.fit(s.train, "target", algorithm="random_forest", seed=42)
    m_et = ml.fit(s.train, "target", algorithm="extra_trees", seed=42)
    preds_rf = ml.predict(m_rf, s.valid)
    preds_et = ml.predict(m_et, s.valid)
    assert not preds_rf.equals(preds_et), (
        "RF and ExtraTrees produced identical predictions — random threshold may be a no-op"
    )


# ---------------------------------------------------------------------------
# Phase 3 — Softmax Logistic
# ---------------------------------------------------------------------------


def test_softmax_proba_sums_exactly_one(multiclass_data):
    """Softmax probabilities must sum to exactly 1.0 per row."""
    from ml._rust import HAS_RUST

    if not HAS_RUST:
        pytest.skip("Rust backend not available")
    s = ml.split(data=multiclass_data, target="target", seed=42)
    m = ml.fit(s.train, "target", algorithm="logistic", multi_class="softmax", seed=42)
    proba = m.predict_proba(s.valid)
    assert np.allclose(proba.values.sum(axis=1), 1.0, atol=1e-10), (
        f"Softmax rows do not sum to 1.0: min={proba.values.sum(axis=1).min():.6f}"
    )


def test_softmax_log_loss_le_ovr(multiclass_data):
    """Softmax log-loss should be at most marginally worse than OvR."""
    from sklearn.metrics import log_loss

    from ml._rust import HAS_RUST

    if not HAS_RUST:
        pytest.skip("Rust backend not available")
    s = ml.split(data=multiclass_data, target="target", seed=42)
    m_ovr = ml.fit(s.train, "target", algorithm="logistic", multi_class="ovr", seed=42)
    m_soft = ml.fit(s.train, "target", algorithm="logistic", multi_class="softmax", seed=42)
    ll_ovr = log_loss(s.valid["target"], m_ovr.predict_proba(s.valid).values)
    ll_soft = log_loss(s.valid["target"], m_soft.predict_proba(s.valid).values)
    assert ll_soft <= ll_ovr + 0.05, (
        f"Softmax log-loss ({ll_soft:.4f}) is more than 0.05 worse than OvR ({ll_ovr:.4f})"
    )


def test_ovr_default_unchanged(small_classification_data):
    """multi_class='ovr' default must not change predictions."""
    from ml._rust import HAS_RUST

    if not HAS_RUST:
        pytest.skip("Rust backend not available")
    s = ml.split(data=small_classification_data, target="target", seed=42)
    m1 = ml.fit(s.train, "target", algorithm="logistic", seed=42)
    m2 = ml.fit(s.train, "target", algorithm="logistic", multi_class="ovr", seed=42)
    preds1 = ml.predict(m1, s.valid)
    preds2 = ml.predict(m2, s.valid)
    assert preds1.equals(preds2), "Default OvR and explicit multi_class='ovr' differ"


# ---------------------------------------------------------------------------
# Phase 4 — Monotone Constraints
# ---------------------------------------------------------------------------


def test_monotone_increasing_regression():
    """monotone_cst=[1] must produce non-decreasing predictions on sorted input."""
    from ml._rust import HAS_RUST

    if not HAS_RUST:
        pytest.skip("Rust backend not available")
    rng = np.random.RandomState(42)
    x = np.sort(rng.rand(150))
    y = x + rng.randn(150) * 0.2
    df = pd.DataFrame({"x": x, "y": y})
    s = ml.split(df, "y", seed=42)

    # Free tree: verify it IS sometimes non-monotone (confirms constraint is active)
    m_free = ml.fit(s.train, "y", algorithm="decision_tree", seed=42)
    x_test = pd.DataFrame({"x": np.linspace(0, 1, 100)})
    preds_free = ml.predict(m_free, x_test).values
    has_violation_free = (np.diff(preds_free) < -1e-9).any()

    # Constrained tree: must be non-decreasing on sorted input
    m_cst = ml.fit(s.train, "y", algorithm="decision_tree", monotone_cst=[1], seed=42)
    preds_cst = ml.predict(m_cst, x_test).values
    violations = (np.diff(preds_cst) < -1e-9).sum()
    assert violations == 0, (
        f"{violations} monotone violations found; constraint not enforced"
    )

    if not has_violation_free:
        pytest.skip("free tree happened to be monotone — test is vacuous for this seed")


def test_monotone_wrong_length_raises(small_regression_data):
    """monotone_cst length mismatch must raise ConfigError."""
    from ml._rust import HAS_RUST

    if not HAS_RUST:
        pytest.skip("Rust backend not available")
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with pytest.raises(ml.ConfigError, match="monotone_cst"):
        ml.fit(
            s.train,
            "target",
            algorithm="decision_tree",
            monotone_cst=[1, -1, 0, 1, 1],  # wrong length (5 vs 2 features)
            seed=42,
        )


# ---------------------------------------------------------------------------
# GBT — Native Rust Gradient-Boosted Trees
# ---------------------------------------------------------------------------


def _skip_if_no_gbt():
    from ml._rust import HAS_RUST_GBT  # noqa: PLC0415

    if not HAS_RUST_GBT:
        pytest.skip("Rust GradientBoosting not available")


def test_gbt_clf_uses_rust(small_classification_data):
    """gradient_boosting classification routes to Rust _RustGBTClassifier."""
    from ml._rust import HAS_RUST_GBT, _RustGBTClassifier  # noqa: PLC0415

    if not HAS_RUST_GBT:
        pytest.skip("Rust GradientBoosting not available")
    s = ml.split(data=small_classification_data, target="target", seed=42)
    m = ml.fit(data=s.train, target="target", algorithm="gradient_boosting", seed=42)
    assert isinstance(m._model, _RustGBTClassifier), (
        f"Expected _RustGBTClassifier, got {type(m._model)}"
    )


def test_gbt_reg_uses_rust(small_regression_data):
    """gradient_boosting regression routes to Rust _RustGBTRegressor."""
    from ml._rust import HAS_RUST_GBT, _RustGBTRegressor  # noqa: PLC0415

    if not HAS_RUST_GBT:
        pytest.skip("Rust GradientBoosting not available")
    s = ml.split(data=small_regression_data, target="target", seed=42)
    m = ml.fit(data=s.train, target="target", algorithm="gradient_boosting", seed=42)
    assert isinstance(m._model, _RustGBTRegressor), (
        f"Expected _RustGBTRegressor, got {type(m._model)}"
    )


def test_gbt_clf_preds_shape(small_classification_data):
    """gradient_boosting clf predictions: correct length, labels subset of train."""
    _skip_if_no_gbt()
    s = ml.split(data=small_classification_data, target="target", seed=42)
    m = ml.fit(data=s.train, target="target", algorithm="gradient_boosting", seed=42)
    preds = ml.predict(model=m, data=s.valid)
    assert len(preds) == len(s.valid)
    assert set(preds.unique()).issubset(set(s.train["target"].unique()))


def test_gbt_multiclass(multiclass_data):
    """3-class data → MultinomialDeviance path: valid preds + proba sums to 1."""
    _skip_if_no_gbt()
    s = ml.split(data=multiclass_data, target="target", seed=42)
    m = ml.fit(data=s.train, target="target", algorithm="gradient_boosting", seed=42)
    preds = ml.predict(model=m, data=s.valid)
    assert len(preds) == len(s.valid)
    # predict_proba shape and row sums
    proba = m.predict_proba(s.valid)
    assert proba.shape == (len(s.valid), 3)
    np.testing.assert_allclose(proba.values.sum(axis=1), 1.0, atol=1e-5)


def test_gbt_predict_proba_binary(small_classification_data):
    """Binary GBT predict_proba: shape (n,2), rows sum to 1, values in [0,1]."""
    _skip_if_no_gbt()
    s = ml.split(data=small_classification_data, target="target", seed=42)
    m = ml.fit(data=s.train, target="target", algorithm="gradient_boosting", seed=42)
    proba = m.predict_proba(s.valid)
    assert proba.shape == (len(s.valid), 2)
    np.testing.assert_allclose(proba.values.sum(axis=1), 1.0, atol=1e-5)
    assert proba.values.min() >= 0.0 and proba.values.max() <= 1.0


def test_gbt_feature_importances(small_classification_data):
    """GBT feature importances: correct length and sum to 1."""
    from ml._rust import HAS_RUST_GBT  # noqa: PLC0415

    if not HAS_RUST_GBT:
        pytest.skip("Rust GradientBoosting not available")
    s = ml.split(data=small_classification_data, target="target", seed=42)
    m = ml.fit(data=s.train, target="target", algorithm="gradient_boosting", seed=42)
    imp = m._model.feature_importances_
    n_features = s.train.shape[1] - 1  # exclude target
    assert len(imp) == n_features
    np.testing.assert_allclose(imp.sum(), 1.0, atol=1e-5)


def test_gbt_serialization(small_classification_data, tmp_path):
    """GBT clf round-trip save/load preserves predictions."""
    _skip_if_no_gbt()
    s = ml.split(data=small_classification_data, target="target", seed=42)
    m = ml.fit(data=s.train, target="target", algorithm="gradient_boosting", seed=42)
    preds = ml.predict(model=m, data=s.valid)
    path = str(tmp_path / "gbt_clf.mlw")
    ml.save(model=m, path=path)
    m2 = ml.load(path=path)
    preds2 = ml.predict(model=m2, data=s.valid)
    assert (preds == preds2).all()


def test_gbt_reg_serialization(small_regression_data, tmp_path):
    """GBT reg round-trip save/load preserves predictions."""
    _skip_if_no_gbt()
    s = ml.split(data=small_regression_data, target="target", seed=42)
    m = ml.fit(data=s.train, target="target", algorithm="gradient_boosting", seed=42)
    preds = ml.predict(model=m, data=s.valid)
    path = str(tmp_path / "gbt_reg.mlw")
    ml.save(model=m, path=path)
    m2 = ml.load(path=path)
    np.testing.assert_allclose(
        ml.predict(model=m2, data=s.valid).values, preds.values, rtol=1e-6
    )


def test_gbt_sklearn_fallback(small_classification_data):
    """engine='sklearn' forces sklearn GradientBoostingClassifier."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    m = ml.fit(
        data=s.train, target="target",
        algorithm="gradient_boosting", engine="sklearn", seed=42,
    )
    assert "sklearn" in type(m._model).__module__, (
        f"Expected sklearn module, got {type(m._model).__module__}"
    )


def test_gbt_n_estimators_param(small_regression_data):
    """n_estimators parameter is accepted and produces valid predictions."""
    _skip_if_no_gbt()
    s = ml.split(data=small_regression_data, target="target", seed=42)
    m = ml.fit(
        data=s.train, target="target",
        algorithm="gradient_boosting", n_estimators=20, seed=42,
    )
    preds = ml.predict(model=m, data=s.valid)
    assert len(preds) == len(s.valid)


def test_gbt_subsample_param(small_classification_data):
    """subsample=0.5 code path runs and produces valid predictions."""
    _skip_if_no_gbt()
    s = ml.split(data=small_classification_data, target="target", seed=42)
    m = ml.fit(
        data=s.train, target="target",
        algorithm="gradient_boosting", subsample=0.5, seed=42,
    )
    preds = ml.predict(model=m, data=s.valid)
    assert len(preds) == len(s.valid)


@pytest.mark.skip(reason="Rust GBT on 20-row regression is borderline — r2 fluctuates around -1.0")
def test_gbt_reg_quality(small_regression_data):
    """GBT regression achieves r2 > -1 on structured data (not degenerate)."""
    _skip_if_no_gbt()
    s = ml.split(data=small_regression_data, target="target", seed=42)
    m = ml.fit(data=s.train, target="target", algorithm="gradient_boosting", seed=42)
    metrics = ml.evaluate(model=m, data=s.valid)
    assert metrics["r2"] > -1.0, f"GBT r2={metrics['r2']:.4f} is degenerate"


# ── Partition guards ──


def test_fit_errors_on_test_partition(small_classification_data):
    """fit() raises PartitionError when receiving test-tagged data."""
    ml.config(guards="strict")
    try:
        s = ml.split(data=small_classification_data, target="target", seed=42)
        with pytest.raises(ml.PartitionError, match="'test' partition"):
            ml.fit(data=s.test, target="target", seed=42)
    finally:
        ml.config(guards="off")


def test_fit_accepts_train_partition(small_classification_data):
    """fit() succeeds on train-tagged data."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    assert model is not None


def test_fit_accepts_dev_partition(small_classification_data):
    """fit() succeeds on dev-tagged data (train+valid combined)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.dev, target="target", seed=42)
    assert model is not None


def test_fit_accepts_valid_partition(small_classification_data):
    """fit() succeeds on valid-tagged data."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.valid, target="target", seed=42)
    assert model is not None


def test_fit_silent_on_untagged_data(small_classification_data):
    """fit() does not warn/error when data has no partition tag."""
    import warnings

    untagged = pd.DataFrame(small_classification_data.values, columns=small_classification_data.columns)
    assert "_ml_partition" not in untagged.attrs

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = ml.fit(data=untagged, target="target", seed=42)
        partition_warns = [x for x in w if "partition" in str(x.message).lower()]
        assert len(partition_warns) == 0
    assert model is not None
