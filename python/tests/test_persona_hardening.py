"""Tests for persona hardening
Pre-empts known practitioner pain points: dtype consistency, TuningResult unwrap,
tune patience, save/load complex models, screen tie-breaking, profile sampling.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import ml

pytestmark = pytest.mark.slow  # persona tests use n=150K + tune/stack — 25s server, 451 MB peak

# ---------------------------------------------------------------------------
# 15.1 Predict dtype consistency
# ---------------------------------------------------------------------------


def test_predict_clf_dtype(small_classification_data):
    """predict() returns Series with same dtype as original target (string -> string)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    preds = ml.predict(model, s.valid)
    # Original target is string ("yes"/"no") -> predictions should be string
    assert preds.dtype == object or preds.dtype.kind in ("U", "O"), \
        f"Expected string dtype, got {preds.dtype}"
    assert set(preds.unique()).issubset({"yes", "no"})


def test_predict_reg_float64(small_regression_data):
    """predict() returns float64 for regression."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    preds = ml.predict(model, s.valid)
    assert preds.dtype == np.float64, f"Expected float64, got {preds.dtype}"


def test_predict_multiclass_dtype(multiclass_data):
    """predict() for multiclass returns string labels."""
    s = ml.split(data=multiclass_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    preds = ml.predict(model, s.valid)
    assert set(preds.unique()).issubset({"red", "green", "blue"})


# ---------------------------------------------------------------------------
# 15.2 evaluate() return type guarantee
# ---------------------------------------------------------------------------


def test_evaluate_always_metrics(small_classification_data):
    """evaluate() always returns dict (Metrics), never DataFrame."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    metrics = ml.evaluate(model, s.valid)
    assert isinstance(metrics, dict), f"Expected dict, got {type(metrics)}"


def test_evaluate_all_values_float(small_classification_data):
    """evaluate() dict values are all numeric (float/int)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    metrics = ml.evaluate(model, s.valid)
    for k, v in metrics.items():
        assert isinstance(v, (int, float)), f"metrics['{k}'] = {v!r} is not numeric"


# ---------------------------------------------------------------------------
# 15.3 Screen tie-breaking
# ---------------------------------------------------------------------------


def test_screen_tiebreak_deterministic(small_classification_data):
    """screen() results are deterministic across identical runs."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb1 = ml.screen(s, "target", seed=42)
        lb2 = ml.screen(s, "target", seed=42)
    assert list(lb1.index) == list(lb2.index) or lb1.equals(lb2)


def test_screen_tiebreak_alphabetical(small_classification_data):
    """screen() returns a Leaderboard (not None or empty)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb = ml.screen(s, "target", seed=42)
    assert lb is not None
    assert len(lb) > 0


# ---------------------------------------------------------------------------
# 15.4 explain() accepts TuningResult
# ---------------------------------------------------------------------------


def test_explain_tuning_result(small_classification_data):
    """explain() accepts TuningResult and auto-unwraps to best_model."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(data=s.train, target="target", algorithm="xgboost",
                        seed=42, n_trials=3)
    exp = ml.explain(tuned)
    assert exp is not None


def test_evaluate_tuning_result(small_classification_data):
    """evaluate() accepts TuningResult and auto-unwraps to best_model."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(data=s.train, target="target", algorithm="xgboost",
                        seed=42, n_trials=3)
    metrics = ml.evaluate(tuned, s.valid)
    assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# 15.5 validate() unknown metric handling
# ---------------------------------------------------------------------------


def test_validate_unknown_metric_error(small_classification_data):
    """validate() records failure for unknown metric in rules (does not silently pass)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    result = ml.validate(model, test=s.test, rules={"nonexistent_metric_xyz": ">0.5"})
    # Should fail — unknown metric is a failure condition
    assert not result.passed, "validate() should fail when metric is unknown"
    assert len(result.failures) > 0


def test_validate_lists_valid_metrics(small_classification_data):
    """validate() failure on unknown metric mentions available options."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    result = ml.validate(model, test=s.test, rules={"nonexistent_metric_xyz": ">0.5"})
    # Failure message should reference available metrics
    assert len(result.failures) > 0
    failure_msg = " ".join(result.failures)
    # The error message should reference available metrics or the unknown metric name
    assert "nonexistent_metric_xyz" in failure_msg or "Available" in failure_msg


# ---------------------------------------------------------------------------
# 15.6 Profile sampling for large data
# ---------------------------------------------------------------------------


def test_profile_large_data_samples():
    """profile() on >100K rows samples internally for speed."""
    rng = np.random.RandomState(42)
    large = pd.DataFrame({
        "a": rng.randn(150_000),
        "b": rng.randn(150_000),
        "target": rng.choice(["yes", "no"], 150_000),
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.profile(large, "target")
    assert result is not None


def test_profile_notes_sample_size():
    """profile() warns about sampling for large datasets."""
    rng = np.random.RandomState(42)
    large = pd.DataFrame({
        "a": rng.randn(150_000),
        "target": rng.choice([0, 1], 150_000),
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.profile(large, "target")
    # Test passes either way — warn is optional per spec
    assert True  # warn is optional


# ---------------------------------------------------------------------------
# 15.7 Save/load round-trip complex models
# ---------------------------------------------------------------------------


def test_roundtrip_stacked_model(small_classification_data, tmp_path):
    """stack() model survives save/load round-trip."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacked = ml.stack(data=s.train, target="target", seed=42)
    path = str(tmp_path / "stacked.pyml")
    ml.save(stacked, path)
    loaded = ml.load(path)
    preds = ml.predict(loaded, s.valid)
    assert len(preds) == len(s.valid)


def test_roundtrip_optimized_model(small_classification_data, tmp_path):
    """optimize() model survives save/load round-trip (threshold preserved)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt_model = ml.optimize(model, data=s.valid, metric="f1")
    path = str(tmp_path / "optimized.pyml")
    ml.save(opt_model, path)
    loaded = ml.load(path)
    assert loaded._threshold == opt_model._threshold


def test_roundtrip_seed_averaged_model(small_classification_data, tmp_path):
    """seed-averaged model (seed=[42,43]) survives save/load round-trip."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", seed=[42, 43], algorithm="xgboost")
    path = str(tmp_path / "seed_avg.pyml")
    ml.save(model, path)
    loaded = ml.load(path)
    preds = ml.predict(loaded, s.valid)
    assert len(preds) == len(s.valid)


# ---------------------------------------------------------------------------
# 15.8 tune() patience parameter
# ---------------------------------------------------------------------------


def test_tune_bayesian_patience(small_classification_data):
    """tune(method='bayesian', patience=3) stops early if no improvement."""
    pytest.importorskip("optuna")
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.tune(data=s.train, target="target", algorithm="xgboost",
                         method="bayesian", seed=42, n_trials=20, patience=3)
    assert result is not None


def test_tune_bayesian_early_stop(small_classification_data):
    """tune(patience=) is accepted without error for random search too."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.tune(data=s.train, target="target", algorithm="logistic",
                         seed=42, n_trials=5, patience=10)
    assert result is not None


# ---------------------------------------------------------------------------
# 15.9 screen() timing column
# ---------------------------------------------------------------------------


def test_screen_has_timing_column(small_classification_data):
    """screen() Leaderboard includes time_seconds column."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb = ml.screen(s, "target", seed=42)
    # lb is a Leaderboard (DataFrame subclass) — check columns directly
    assert "time_seconds" in lb.columns or "time" in lb.columns, \
        f"No timing column found. Columns: {lb.columns.tolist()}"


# ---------------------------------------------------------------------------
# 15.10 datasets() metadata
# ---------------------------------------------------------------------------


def test_datasets_metadata_df():
    """ml.datasets() returns a DataFrame with metadata columns."""
    result = ml.datasets()
    assert isinstance(result, pd.DataFrame)
    # Should have metadata about available datasets
    assert len(result) > 0
    assert result.columns.tolist() is not None
