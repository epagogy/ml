"""Tests for tune() time_budget parameter —"""

import time
import warnings

import pytest

import ml

pytestmark = pytest.mark.slow  # time_budget tests run real Optuna — 26s server, 452 MB peak


def test_tune_time_budget_basic(small_classification_data):
    """tune() accepts time_budget= parameter."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.tune(data=s.train, target="target", algorithm="xgboost",
                         seed=42, time_budget=5)
    assert result is not None


def test_tune_time_budget_at_least_one_trial(small_classification_data):
    """tune() completes at least 1 trial even with time_budget."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 5s is enough to complete at least 1 trial (was 60s — caused OOM on 8GB machines)
        result = ml.tune(data=s.train, target="target", algorithm="logistic",
                         seed=42, time_budget=5)
    assert result is not None
    assert hasattr(result, "best_params_") or hasattr(result, "best_model")


def test_tune_time_budget_stops_early(small_classification_data):
    """tune() stops within reasonable time when time_budget is small."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.tune(data=s.train, target="target", algorithm="xgboost",
                         seed=42, time_budget=3, n_trials=10000)
    elapsed = time.time() - start
    assert result is not None
    assert elapsed < 60  # should stop well within a minute


def test_tune_time_budget_with_bayesian(small_classification_data):
    """tune() time_budget= works with method='bayesian'."""
    pytest.importorskip("optuna")
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.tune(data=s.train, target="target", algorithm="xgboost",
                         seed=42, time_budget=5, method="bayesian")
    assert result is not None


def test_tune_time_budget_returns_tuning_result(small_classification_data):
    """tune() with time_budget= returns a proper TuningResult."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.tune(data=s.train, target="target", algorithm="xgboost",
                         seed=42, time_budget=8)
    assert isinstance(result, ml.TuningResult)


def test_tune_time_budget_without_n_trials(small_classification_data):
    """tune() with time_budget= works without explicitly setting n_trials."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.tune(data=s.train, target="target", algorithm="logistic",
                         seed=42, time_budget=10)
    assert result is not None
    assert hasattr(result, "best_params_")


def test_tune_time_budget_none_defaults_to_n_trials(small_classification_data):
    """tune() with time_budget=None uses n_trials as before."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.tune(data=s.train, target="target", algorithm="logistic",
                         seed=42, n_trials=2, time_budget=None)
    assert result is not None
    # Should complete exactly n_trials=2 trials (was 3 — reduce test time)
    assert len(result.tuning_history_) == 2
