"""Tests for ml.tune() — hyperparameter tuning via random search."""

import warnings

import numpy as np
import pandas as pd
import pytest

import ml

pytestmark = pytest.mark.slow  # tune() runs Optuna trials — 49s server, 463 MB peak


@pytest.fixture(scope="session")
def clf_data():
    """Classification dataset for tune tests."""
    rng = np.random.RandomState(42)
    n = 80
    data = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "x3": rng.normal(0, 1, n),
        "target": rng.choice(["yes", "no"], n),
    })
    return data


@pytest.fixture(scope="session")
def reg_data():
    """Regression dataset for tune tests."""
    rng = np.random.RandomState(42)
    n = 80
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    data = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": rng.normal(0, 1, n),
        "price": 3.0 * x1 + 2.0 * x2 + rng.normal(0, 0.5, n),
    })
    return data


@pytest.fixture(scope="module")
def xgb_tuned(clf_data):
    """Pre-tuned XGBoost on clf_data — shared across tests."""
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.tune(
            data=s.train, target="target", algorithm="xgboost", seed=42, n_trials=3
        )
    return s, result


def test_tune_basic_xgboost(xgb_tuned):
    """tune() with xgboost returns a TuningResult."""
    _, tuned = xgb_tuned
    assert isinstance(tuned, ml.TuningResult)
    assert tuned.algorithm == "xgboost"
    assert tuned.task == "classification"


def test_tune_basic_random_forest(clf_data):
    """tune() with random forest returns a TuningResult."""
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train, target="target", algorithm="random_forest", seed=42, n_trials=3
        )
    assert isinstance(tuned, ml.TuningResult)
    assert tuned.algorithm == "random_forest"


def test_tune_returns_tuning_result(clf_data):
    """TuningResult delegates Model-like attributes."""
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train, target="target", algorithm="xgboost", seed=42, n_trials=3
        )
    assert isinstance(tuned, ml.TuningResult)
    assert isinstance(tuned.best_model, ml.Model)
    assert tuned.task == "classification"
    assert tuned.algorithm == "xgboost"
    assert len(tuned.features) > 0
    assert tuned.target == "target"
    assert tuned.seed == 42


def test_tune_tuning_history(xgb_tuned):
    """model.tuning_history_ is DataFrame with correct columns."""
    _, tuned = xgb_tuned
    assert tuned.tuning_history_ is not None
    assert isinstance(tuned.tuning_history_, pd.DataFrame)
    assert "trial" in tuned.tuning_history_.columns
    assert "score" in tuned.tuning_history_.columns
    assert len(tuned.tuning_history_) == 3  # n_trials=3 in module fixture


def test_tune_best_params(xgb_tuned):
    """model.best_params_ is dict."""
    _, tuned = xgb_tuned
    assert tuned.best_params_ is not None
    assert isinstance(tuned.best_params_, dict)
    assert len(tuned.best_params_) > 0


def test_tune_best_score(clf_data):
    """TuningResult has .best_score property (float)."""
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train, target="target", algorithm="xgboost", seed=42, n_trials=3
        )
    assert tuned.best_score is not None
    assert isinstance(tuned.best_score, float)
    assert 0.0 <= tuned.best_score <= 1.0


def test_tune_seed_reproducible(clf_data):
    """Same seed → same result."""
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned1 = ml.tune(
            data=s.train, target="target", algorithm="xgboost", seed=42, n_trials=3
        )
        tuned2 = ml.tune(
            data=s.train, target="target", algorithm="xgboost", seed=42, n_trials=3
        )
    assert tuned1.best_params_ == tuned2.best_params_
    assert list(tuned1.tuning_history_["score"]) == list(tuned2.tuning_history_["score"])


def test_tune_custom_params_range(clf_data):
    """Custom params with numeric ranges work."""
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train,
            target="target",
            algorithm="xgboost",
            seed=42,
            n_trials=3,
            params={"max_depth": (3, 8)},
        )
    assert "max_depth" in tuned.best_params_
    assert 3 <= tuned.best_params_["max_depth"] <= 8


def test_tune_predict_works(xgb_tuned):
    """Tuned model can predict on new data."""
    s, tuned = xgb_tuned
    preds = tuned.predict(s.valid)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(s.valid)
    assert set(preds.unique()).issubset({"yes", "no"})


def test_tune_no_model_no_algorithm_error():
    """Passing neither model nor algorithm raises ConfigError."""
    data = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    with pytest.raises(ml.ConfigError, match="requires either"):
        ml.tune(data=data, target="y", seed=42)


def test_tune_bad_target_error(clf_data):
    """Missing target raises DataError."""
    s = ml.split(data=clf_data, target="target", seed=42)
    with pytest.raises(ml.DataError, match="not found"):
        ml.tune(data=s.train, target="missing", algorithm="xgboost", seed=42)


def test_tune_evaluate_accepts_tuning_result(clf_data):
    """evaluate() accepts TuningResult directly."""
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train, target="target", algorithm="xgboost", seed=42, n_trials=3
        )
    metrics = ml.evaluate(tuned, s.valid)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics


def test_tune_save_load_round_trip(clf_data):
    """TuningResult survives save/load cycle."""
    import tempfile
    from pathlib import Path

    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train, target="target", algorithm="xgboost", seed=42, n_trials=3
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "tuned.pyml")
        ml.save(tuned, path)
        loaded = ml.load(path)

    assert isinstance(loaded, ml.TuningResult)
    assert loaded.best_params_ == tuned.best_params_
    assert loaded.algorithm == tuned.algorithm

    # Predictions match
    preds_orig = tuned.predict(s.valid)
    preds_loaded = loaded.predict(s.valid)
    pd.testing.assert_series_equal(preds_orig, preds_loaded)


def test_tune_grid_basic(clf_data):
    """method='grid' runs exhaustive search over all combinations."""
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train,
            target="target",
            algorithm="xgboost",
            seed=42,
            method="grid",
            params={"max_depth": [3, 5], "n_estimators": [50, 100]},
        )
    assert isinstance(tuned, ml.TuningResult)
    assert tuned.algorithm == "xgboost"
    assert tuned.best_params_ is not None


def test_tune_regression(reg_data):
    """tune() works for regression tasks (reg_data fixture)."""
    s = ml.split(data=reg_data, target="price", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train, target="price", algorithm="xgboost", seed=42, n_trials=3
        )
    assert isinstance(tuned, ml.TuningResult)
    assert tuned.task == "regression"
    assert tuned.algorithm == "xgboost"
    preds = tuned.predict(s.valid)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(s.valid)


def test_tune_predict_proba(clf_data):
    """TuningResult.predict_proba() delegates to best_model for classification."""
    import numpy as np

    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train, target="target", algorithm="xgboost", seed=42, n_trials=3
        )
    probs = tuned.predict_proba(s.valid)
    assert isinstance(probs, pd.DataFrame)
    assert probs.shape == (len(s.valid), 2)
    assert all(np.isclose(probs.sum(axis=1), 1.0, atol=0.01))


# ── Phase 0 additions ─────────────────────────────────────────────────────────


def test_tune_zero_bounded_param_sampling():
    """Zero-bounded params use linear sampling — no log-uniform pathology. Phase 0.2."""
    import numpy as np

    from ml.tune import _sample_params

    # reg_alpha: (0.0, 10.0) — zero-bounded, must use linear sampling
    # If log-uniform were applied (log(0) = -inf), samples would cluster near 0
    search_space = {"reg_alpha": (0.0, 10.0)}
    rng = np.random.RandomState(42)
    samples = [_sample_params(search_space, rng)["reg_alpha"] for _ in range(200)]

    # With linear sampling: mean ≈ 5.0, and values > 5.0 should be ~half
    # With log-uniform (broken): mean << 1.0, values > 5.0 would be rare
    above_five = sum(1 for s in samples if s > 5.0)
    assert above_five > 60, (
        f"Expected ~50% of samples above 5.0 with linear sampling, got {above_five}/200. "
        "Suggests log-uniform pathology for zero-bounded params."
    )
    assert min(samples) >= 0.0
    assert max(samples) <= 10.0


def test_tune_different_seeds_per_fold(clf_data):
    """tune() uses unique seed per trial×fold combination. Phase 0.3."""
    from unittest.mock import patch

    captured_seeds = []
    original_create = ml._engines.create

    def mock_create(algo, *, task, seed, **kwargs):
        captured_seeds.append(seed)
        return original_create(algo, task=task, seed=seed, **kwargs)

    s = ml.split(data=clf_data, target="target", seed=42)
    with patch("ml._engines.create", side_effect=mock_create), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.tune(
            data=s.train, target="target", algorithm="xgboost", seed=42, n_trials=2, cv_folds=3
        )

    # 2 trials × 3 folds = 6 CV engine creations (+ 1 final refit = 7 total)
    # The 6 CV seeds must all be unique — final refit may reuse base seed
    assert len(captured_seeds) >= 6
    cv_seeds = captured_seeds[:6]  # first n_trials * cv_folds calls are the CV loop
    assert len(set(cv_seeds)) == len(cv_seeds), (
        "Duplicate seeds detected across trial×fold combinations. "
        "Per-trial-fold seed variation is broken."
    )


# ── A2: Bayesian HPO via Optuna ───────────────────────────────────────────────


def test_tune_bayesian_basic(clf_data):
    """tune(method='bayesian') returns TuningResult with valid best_params_. A2."""
    pytest.importorskip("optuna")
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train, target="target", algorithm="xgboost",
            method="bayesian", n_trials=2, seed=42,
        )
    assert isinstance(tuned, ml.TuningResult)
    assert isinstance(tuned.best_params_, dict)
    assert len(tuned.best_params_) > 0


def test_tune_bayesian_seed_reproducible(clf_data):
    """Two bayesian runs with same seed return same best_params_. A2."""
    pytest.importorskip("optuna")
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t1 = ml.tune(data=s.train, target="target", algorithm="random_forest",
                     method="bayesian", n_trials=2, seed=99)
        t2 = ml.tune(data=s.train, target="target", algorithm="random_forest",
                     method="bayesian", n_trials=2, seed=99)
    assert t1.best_params_ == t2.best_params_


def test_tune_bayesian_custom_params(clf_data):
    """tune(method='bayesian') respects custom params search space. A2."""
    pytest.importorskip("optuna")
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train, target="target", algorithm="xgboost",
            method="bayesian", n_trials=2, seed=42,
            params={"max_depth": (3, 6), "n_estimators": (50, 150)},
        )
    assert "max_depth" in tuned.best_params_
    assert 3 <= tuned.best_params_["max_depth"] <= 6


def test_tune_bayesian_timeout(clf_data):
    """tune(method='bayesian', timeout=2) terminates within timeout. A2."""
    pytest.importorskip("optuna")
    import time
    s = ml.split(data=clf_data, target="target", seed=42)
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train, target="target", algorithm="random_forest",
            method="bayesian", n_trials=9999, timeout=3, seed=42,
        )
    elapsed = time.time() - start
    # Should finish well within 10 seconds (timeout=3 + overhead)
    assert elapsed < 15, f"Bayesian tuning took {elapsed:.1f}s, expected < 15s"
    assert isinstance(tuned, ml.TuningResult)


def test_tune_bayesian_not_installed(clf_data, monkeypatch):
    """tune(method='bayesian') without optuna raises ConfigError. A2."""
    import sys
    s = ml.split(data=clf_data, target="target", seed=42)
    original = sys.modules.get("optuna", None)
    sys.modules["optuna"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ml.ConfigError, match="optuna"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ml.tune(data=s.train, target="target", algorithm="xgboost",
                    method="bayesian", n_trials=3, seed=42)
    finally:
        if original is not None:
            sys.modules["optuna"] = original
        elif "optuna" in sys.modules:
            del sys.modules["optuna"]


def test_tune_bayesian_regression(reg_data):
    """tune(method='bayesian') works for regression tasks. A2."""
    pytest.importorskip("optuna")
    s = ml.split(data=reg_data, target="price", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train, target="price", algorithm="random_forest",
            method="bayesian", n_trials=2, seed=42,
        )
    assert isinstance(tuned, ml.TuningResult)
    assert len(tuned.best_params_) > 0


def test_tune_bayesian_per_fold_normalization(clf_data):
    """Bayesian tune uses per-fold normalization (no leakage). A2."""
    pytest.importorskip("optuna")
    from unittest.mock import patch
    norm_calls = []
    original_prepare = ml._normalize.prepare

    def mock_prepare(X, y, algorithm, task):
        norm_calls.append(1)
        return original_prepare(X, y, algorithm=algorithm, task=task)

    s = ml.split(data=clf_data, target="target", seed=42)
    with patch("ml._normalize.prepare", side_effect=mock_prepare), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.tune(data=s.train, target="target", algorithm="xgboost",
                method="bayesian", n_trials=3, cv_folds=3, seed=42)

    # Bayesian path pre-computes fold cache: 3 folds + 1 final refit = 4 calls.
    # (Fold cache avoids redundant re-normalization across trials — correct behavior.)
    assert len(norm_calls) >= 4, (
        f"Expected >= 4 per-fold normalization calls, got {len(norm_calls)}. "
        "Per-fold normalization may be broken in Bayesian mode."
    )


def test_tune_custom_metric_callable(clf_data):
    """tune(metric=callable) uses custom scorer throughout CV. A2 (Conort C2)."""
    def my_f1(y_true, y_pred):
        from sklearn.metrics import f1_score
        try:
            return float(f1_score(y_true, y_pred, average="binary", pos_label="yes"))
        except Exception:
            return 0.0
    my_f1.greater_is_better = True
    my_f1.needs_proba = False

    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train, target="target", algorithm="random_forest",
            n_trials=2, seed=42, metric=my_f1,
        )
    assert isinstance(tuned, ml.TuningResult)
    # History score column should have float values
    assert tuned.tuning_history_["score"].dtype.kind == "f"
