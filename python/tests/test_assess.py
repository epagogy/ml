"""Tests for assess()."""

import warnings

import pytest

import ml


def test_assess_basic(small_classification_data):
    """Test basic assess on test set."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    verdict = ml.assess(model=model, test=s.test)

    assert isinstance(verdict, dict)
    assert "accuracy" in verdict


def test_assess_keyword_only(small_classification_data):
    """Test assess requires test= keyword."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    # This should work
    ml.assess(model=model, test=s.test)

    # Positional should fail (test is keyword-only)
    with pytest.raises(TypeError):
        ml.assess(model, s.test)  # type: ignore


def test_assess_repeat_raises(small_classification_data):
    """Test assess raises ModelError on repeat calls (not warn — Jupyter swallows UserWarning)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    # First call succeeds
    ml.assess(model=model, test=s.test)

    # Second call raises ModelError
    with pytest.raises(ml.ModelError, match="called 2 times"):
        ml.assess(model=model, test=s.test)


def test_assess_same_metrics_as_evaluate(small_classification_data):
    """Test assess returns same metric keys as evaluate."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    eval_metrics = ml.evaluate(model=model, data=s.valid)
    assess_metrics = ml.assess(model=model, test=s.test)

    assert set(eval_metrics.keys()) == set(assess_metrics.keys())


# ── Gate 3 additions ──────────────────────────────────────────────────────────

def test_assess_wrong_type_raises_config_error():
    """assess() raises ConfigError when passed a non-Model object."""
    from ml._types import ConfigError
    with pytest.raises(ConfigError):
        ml.assess(model="not_a_model", test=pytest.importorskip("pandas").DataFrame())


@pytest.mark.slow
def test_assess_tuning_result_unwrap(small_classification_data):
    """assess() should accept TuningResult and unwrap to best_model."""
    from ml._types import TuningResult
    s = ml.split(data=small_classification_data, target="target", seed=42)
    tuned = ml.tune(data=s.train, target="target", algorithm="xgboost", seed=42)
    assert isinstance(tuned, TuningResult)
    # Must not raise — TuningResult should be unwrapped
    verdict = ml.assess(model=tuned, test=s.test)
    assert isinstance(verdict, dict)
    assert "accuracy" in verdict


def test_assess_regression_model(small_regression_data):
    """assess() works for regression tasks."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    verdict = ml.assess(model=model, test=s.test)
    assert isinstance(verdict, dict)
    assert "rmse" in verdict or "mae" in verdict


def test_assess_custom_metrics(small_classification_data):
    """assess() accepts custom metrics dict."""
    import numpy as np

    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    def my_metric(y_true, y_pred):
        return float(np.mean(y_true == y_pred))

    verdict = ml.assess(model=model, test=s.test, metrics={"my_acc": my_metric})
    assert isinstance(verdict, dict)
    assert "my_acc" in verdict


@pytest.mark.slow
def test_assess_intervals_true(small_classification_data):
    """assess(intervals=True) should return confidence interval keys."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    verdict = ml.assess(model=model, test=s.test, intervals=True)
    assert isinstance(verdict, dict)
    # With intervals=True, should have lower/upper keys
    interval_keys = [k for k in verdict if k.endswith("_lower") or k.endswith("_upper")]
    assert len(interval_keys) > 0


def test_assess_errors_on_valid_partition(small_classification_data):
    """assess() raises PartitionError when receiving 'valid' partition."""
    ml.config(guards="strict")
    try:
        s = ml.split(data=small_classification_data, target="target", seed=42)
        model = ml.fit(data=s.train, target="target", seed=42)

        with pytest.raises(ml.PartitionError, match="'valid' partition"):
            ml.assess(model=model, test=s.valid)
    finally:
        ml.config(guards="off")


def test_assess_no_partition_warning_on_test(small_classification_data):
    """assess() does NOT warn when receiving actual test partition."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ml.assess(model=model, test=s.test)
        partition_warns = [x for x in w if "partition" in str(x.message).lower()]
        assert len(partition_warns) == 0


def test_assess_errors_on_train_partition(small_classification_data):
    """assess() raises PartitionError when receiving 'train' partition."""
    ml.config(guards="strict")
    try:
        s = ml.split(data=small_classification_data, target="target", seed=42)
        model = ml.fit(data=s.train, target="target", seed=42)

        with pytest.raises(ml.PartitionError, match="'train' partition"):
            ml.assess(model=model, test=s.train)
    finally:
        ml.config(guards="off")


def test_assess_partition_error_doesnt_burn_counter(small_classification_data):
    """PartitionError on wrong partition does not consume the one-shot counter."""
    ml.config(guards="strict")
    try:
        s = ml.split(data=small_classification_data, target="target", seed=42)
        model = ml.fit(data=s.train, target="target", seed=42)

        # Wrong partition → PartitionError, counter should NOT increment
        with pytest.raises(ml.PartitionError):
            ml.assess(model=model, test=s.train)

        # Correct partition → should succeed (counter was not burned)
        verdict = ml.assess(model=model, test=s.test)
        assert isinstance(verdict, dict)
        assert len(verdict) > 0
    finally:
        ml.config(guards="off")


def test_assess_rejects_untagged_data(small_classification_data):
    """assess() rejects data without split provenance."""
    import pandas as pd

    ml.config(guards="strict")
    try:
        s = ml.split(data=small_classification_data, target="target", seed=42)
        model = ml.fit(data=s.train, target="target", seed=42)

        # Create untagged DataFrame (simulates user-constructed data)
        untagged = pd.DataFrame(s.test.values, columns=s.test.columns)

        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.assess(model=model, test=untagged)
    finally:
        ml.config(guards="off")


# ── K metering (validation selection pressure) ───────────────────────────────


def test_assess_K_tracks_validation_evaluations(small_classification_data):
    """Evidence._K counts evaluate() calls on the valid partition, not the model."""
    s = ml.split(data=small_classification_data, target="target", seed=42)

    # Two different models evaluate on same s.valid → K=2
    m1 = ml.fit(data=s.train, target="target", algorithm="logistic", seed=42)
    m2 = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
    ml.evaluate(m1, s.valid)
    ml.evaluate(m2, s.valid)

    final = ml.fit(data=s.dev, target="target", seed=42)
    evidence = ml.assess(final, test=s.test)

    assert isinstance(evidence, ml.Evidence)
    assert evidence._K == 2


def test_assess_K_zero_without_evaluate(small_classification_data):
    """Evidence._K is 0 when no evaluate() was called."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.dev, target="target", seed=42)
    evidence = ml.assess(model, test=s.test)

    assert evidence._K == 0


def test_assess_K_ignores_train_evaluations(small_classification_data):
    """Evaluating on s.train does not inflate K for the valid partition."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    # Evaluate on train (overfitting diagnostic) — should NOT count as K
    ml.evaluate(model, s.train)
    # Evaluate on valid — this is K=1
    ml.evaluate(model, s.valid)

    final = ml.fit(data=s.dev, target="target", seed=42)
    evidence = ml.assess(final, test=s.test)

    assert evidence._K == 1


def test_assess_K_per_partition_not_per_model(small_classification_data):
    """K tracks the partition, not the model. Same valid, 3 models → K=3."""
    s = ml.split(data=small_classification_data, target="target", seed=42)

    for algo in ["logistic", "random_forest", "naive_bayes"]:
        m = ml.fit(data=s.train, target="target", algorithm=algo, seed=42)
        ml.evaluate(m, s.valid)

    final = ml.fit(data=s.dev, target="target", seed=42)
    evidence = ml.assess(final, test=s.test)

    assert evidence._K == 3


def test_assess_K_different_splits_independent(small_classification_data):
    """K from one split does not leak into Evidence from a different split."""
    # Split 1: evaluate twice
    s1 = ml.split(data=small_classification_data, target="target", seed=42)
    m1 = ml.fit(data=s1.train, target="target", seed=42)
    ml.evaluate(m1, s1.valid)
    ml.evaluate(m1, s1.valid)

    # Split 2: no evaluations
    s2 = ml.split(data=small_classification_data, target="target", seed=99)
    m2 = ml.fit(data=s2.dev, target="target", seed=99)
    evidence = ml.assess(m2, test=s2.test)

    assert evidence._K == 0


def test_evidence_str_shows_K(small_classification_data):
    """Evidence.__str__ includes K when K > 0."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    ml.evaluate(model, s.valid)

    final = ml.fit(data=s.dev, target="target", seed=42)
    evidence = ml.assess(final, test=s.test)

    assert "K=1" in str(evidence)


def test_assess_internal_callers_dont_inflate_K(small_classification_data):
    """Internal evaluate calls (from compare, validate) should NOT inflate K.
    Only user-facing evaluate() calls count."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    m1 = ml.fit(data=s.train, target="target", algorithm="logistic", seed=42)
    m2 = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)

    # compare() calls evaluate internally with _guard=False
    ml.compare([m1, m2], s.valid)

    # No user-facing evaluate() was called → K should be 0
    final = ml.fit(data=s.dev, target="target", seed=42)
    evidence = ml.assess(final, test=s.test)
    assert evidence._K == 0
