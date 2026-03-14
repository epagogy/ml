"""Tests for validate()."""

import pytest

import ml


def test_validate_pass(small_classification_data):
    """validate() passes when rules are met."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    gate = ml.validate(model, test=s.test, rules={"accuracy": ">0.0"})
    assert gate.passed is True
    assert len(gate.failures) == 0
    assert "accuracy" in gate.metrics


def test_validate_fail(small_classification_data):
    """validate() fails when rules are not met."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    gate = ml.validate(model, test=s.test, rules={"accuracy": ">0.999"})
    assert gate.passed is False
    assert len(gate.failures) > 0


def test_validate_violations_alias(small_classification_data):
    """gate.violations is alias for gate.failures."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    gate = ml.validate(model, test=s.test, rules={"accuracy": ">0.999"})
    assert gate.violations == gate.failures
    assert len(gate.violations) > 0


def test_validate_multiple_rules(small_classification_data):
    """validate() checks all rules."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    gate = ml.validate(
        model, test=s.test,
        rules={"accuracy": ">0.0", "f1": ">=0.0"}
    )
    assert gate.passed is True


def test_validate_display_output(small_classification_data):
    """validate() display includes PASSED or FAILED."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    gate = ml.validate(model, test=s.test, rules={"accuracy": ">0.0"})
    output = str(gate)
    assert "PASSED" in output


def test_validate_keyword_only(small_classification_data):
    """validate() requires keyword arguments."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    # Positional should fail
    with pytest.raises(TypeError):
        ml.validate(model, s.test, {"accuracy": ">0.0"})  # type: ignore


def test_validate_then_assess_no_warning(small_classification_data):
    """validate() then assess() should NOT trigger double-peek warning."""
    import warnings

    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    # validate() should not count as a peek
    ml.validate(model, test=s.test, rules={"accuracy": ">0.0"})

    # assess() should be the FIRST peek — no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ml.assess(model, test=s.test)
        peek_warns = [x for x in w if "peeking" in str(x.message).lower()
                      or "assess" in str(x.message).lower()]
        assert len(peek_warns) == 0, (
            f"validate() + assess() should not trigger double-peek warning. "
            f"Got: {[str(x.message) for x in peek_warns]}"
        )


def test_validate_regression_rules(small_regression_data):
    """validate() works for regression with numeric rules like rmse < threshold."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)

    gate = ml.validate(model, test=s.test, rules={"rmse": "<1000.0"})
    assert gate.passed is True
    assert "rmse" in gate.metrics

    gate_fail = ml.validate(model, test=s.test, rules={"rmse": "<0.0"})
    assert gate_fail.passed is False
    assert len(gate_fail.failures) > 0


def test_validate_baseline_comparison(small_classification_data):
    """validate() with baseline= populates improvements/degradations lists."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    baseline = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="logistic", seed=42)

    gate = ml.validate(model, test=s.test, baseline=baseline)
    assert gate.baseline_metrics is not None
    assert isinstance(gate.improvements, list)
    assert isinstance(gate.degradations, list)
    # improvements + degradations + unchanged should cover all shared metrics
    total = len(gate.improvements) + len(gate.degradations) + len(gate.unchanged)
    assert total > 0


# ── Partition guards ──


def test_validate_errors_on_train_partition(small_classification_data):
    """validate() raises PartitionError when receiving train-tagged data."""
    ml.config(guards="strict")
    try:
        s = ml.split(data=small_classification_data, target="target", seed=42)
        model = ml.fit(data=s.train, target="target", seed=42)
        with pytest.raises(ml.PartitionError, match="'train' partition"):
            ml.validate(model, test=s.train, rules={"accuracy": ">0.0"})
    finally:
        ml.config(guards="off")


def test_validate_errors_on_valid_partition(small_classification_data):
    """validate() raises PartitionError when receiving valid-tagged data."""
    ml.config(guards="strict")
    try:
        s = ml.split(data=small_classification_data, target="target", seed=42)
        model = ml.fit(data=s.train, target="target", seed=42)
        with pytest.raises(ml.PartitionError, match="'valid' partition"):
            ml.validate(model, test=s.valid, rules={"accuracy": ">0.0"})
    finally:
        ml.config(guards="off")


def test_validate_rejects_untagged_data(small_classification_data):
    """validate() raises PartitionError on untagged data in strict mode."""
    ml.config(guards="strict")
    try:
        s = ml.split(data=small_classification_data, target="target", seed=42)
        model = ml.fit(data=s.train, target="target", seed=42)
        plain = small_classification_data.copy()
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.validate(model, test=plain, rules={"accuracy": ">0.0"})
    finally:
        ml.config(guards="off")


def test_validate_accepts_test_partition(small_classification_data):
    """validate() succeeds on test-tagged data."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    gate = ml.validate(model, test=s.test, rules={"accuracy": ">0.0"})
    assert gate is not None
