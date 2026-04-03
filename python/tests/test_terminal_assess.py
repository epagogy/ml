"""Adversarial tests for the terminal assess constraint.

Tests the paper's conformance condition 4:
"Reject a second assess call on the same test holdout set
 regardless of which model."

These tests try to BREAK the constraint via:
1. Same model, second call (should raise ModelError)
2. Different model, same test partition (should raise PartitionError)
3. Different seed, same data → same partition (should raise PartitionError)
4. copy.deepcopy bypass (known flank — model counter resets)
5. pickle/unpickle bypass (known flank — model counter resets)
6. Serialization roundtrip (save/load bypass)
7. New split with different seed → fresh budget (should work)
8. config(guards="off") explicit escape (should work)
9. Unhashable columns (list-valued) — can't fingerprint, passes silently
10. Empty test set edge case
11. Concurrent assess from threads
"""

import copy
import threading

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def fresh_registry():
    """Clear the provenance registry between tests."""
    import ml
    from ml._provenance import _registry
    _registry.clear()
    ml.config(guards="strict")
    yield
    _registry.clear()
    ml.config(guards="strict")


@pytest.fixture
def data():
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "x1": rng.randn(n),
        "x2": rng.randn(n),
        "y": (rng.randn(n) > 0).astype(int),
    })


# ── 1. Same model, second call → ModelError ──

def test_same_model_second_assess_raises(data):
    """Same model, second call. The partition guard fires first (PartitionError)
    because the partition is already marked. ModelError would fire second if
    the partition guard didn't catch it."""
    import ml
    s = ml.split(data, "y", seed=42)
    model = ml.fit(s.dev, "y", seed=42)
    ml.assess(model, test=s.test)  # first call OK
    with pytest.raises(ml.PartitionError, match="already been assessed"):
        ml.assess(model, test=s.test)  # partition guard catches it first


# ── 2. Different model, same test partition → PartitionError ──

def test_cross_model_same_test_raises(data):
    """The core conformance condition 4 test."""
    import ml
    s = ml.split(data, "y", seed=42)
    model_a = ml.fit(s.dev, "y", algorithm="logistic", seed=42)
    model_b = ml.fit(s.dev, "y", algorithm="random_forest", seed=42)

    ml.assess(model_a, test=s.test)  # first model OK
    with pytest.raises(ml.PartitionError):
        ml.assess(model_b, test=s.test)  # different model, same test → REJECT


# ── 3. Same seed same data → same partition → still rejected ──

def test_resplit_same_seed_same_data_still_rejected(data):
    """Re-splitting with the same seed on the same data produces the
    same content → same fingerprint → already assessed."""
    import ml
    s1 = ml.split(data, "y", seed=42)
    model1 = ml.fit(s1.dev, "y", seed=42)
    ml.assess(model1, test=s1.test)

    # Re-split with same seed and data
    s2 = ml.split(data, "y", seed=42)
    model2 = ml.fit(s2.dev, "y", seed=42)
    with pytest.raises(ml.PartitionError):
        ml.assess(model2, test=s2.test)  # same content → same fp → rejected


# ── 4. New split different seed → fresh budget → works ──

def test_new_split_different_seed_fresh_budget(data):
    """A new split with a different seed produces different partitions,
    so the test fingerprint is different → fresh assess budget."""
    import ml
    s1 = ml.split(data, "y", seed=42)
    model1 = ml.fit(s1.dev, "y", seed=42)
    ml.assess(model1, test=s1.test)

    # New split with different seed
    s2 = ml.split(data, "y", seed=99)
    model2 = ml.fit(s2.dev, "y", seed=42)
    # Should work — different partition content
    evidence = ml.assess(model2, test=s2.test)
    assert "accuracy" in evidence


# ── 5. copy.deepcopy bypass attempt ──

def test_deepcopy_model_does_not_bypass_partition_guard(data):
    """deepcopy copies model._assess_count (it's an int field on a dataclass),
    but even if someone manually reset it, the PARTITION is still marked as
    assessed in the registry. Cross-model guard catches this."""
    import ml
    s = ml.split(data, "y", seed=42)
    model = ml.fit(s.dev, "y", seed=42)
    ml.assess(model, test=s.test)

    # deepcopy copies the model (including assess_count)
    model_copy = copy.deepcopy(model)
    # Manually reset the model counter to simulate an attacker
    model_copy._assess_count = 0

    # But the partition registry still knows this test was assessed
    with pytest.raises(ml.PartitionError, match="already been assessed"):
        ml.assess(model_copy, test=s.test)


# ── 6. Three models on same test → only first succeeds ──

def test_three_models_one_test_only_first_succeeds(data):
    import ml
    s = ml.split(data, "y", seed=42)
    m1 = ml.fit(s.dev, "y", algorithm="logistic", seed=42)
    m2 = ml.fit(s.dev, "y", algorithm="random_forest", seed=42)
    m3 = ml.fit(s.dev, "y", algorithm="decision_tree", seed=42)

    ml.assess(m1, test=s.test)  # OK
    with pytest.raises(ml.PartitionError):
        ml.assess(m2, test=s.test)
    with pytest.raises(ml.PartitionError):
        ml.assess(m3, test=s.test)


# ── 7. config(guards="off") explicit escape ──

def test_guards_off_allows_multiple_assess(data):
    """config(guards='off') is the Rust unsafe — explicit, visible bypass."""
    import ml
    s = ml.split(data, "y", seed=42)
    m1 = ml.fit(s.dev, "y", seed=42)
    ml.assess(m1, test=s.test)

    ml.config(guards="off")
    m2 = ml.fit(s.dev, "y", seed=42)
    # Should not raise — guards are off
    evidence = ml.assess(m2, test=s.test)
    assert "accuracy" in evidence


# ── 8. config(guards="warn") downgrades to warning ──

def test_guards_warn_issues_warning_not_error(data):
    import ml
    s = ml.split(data, "y", seed=42)
    m1 = ml.fit(s.dev, "y", seed=42)

    ml.config(guards="warn")
    ml.assess(m1, test=s.test)

    m2 = ml.fit(s.dev, "y", seed=42)
    with pytest.warns(UserWarning, match="already been assessed"):
        ml.assess(m2, test=s.test)


# ── 9. Partition guard fires BEFORE model counter ──

def test_partition_guard_fires_before_model_counter(data):
    """If the test partition is already assessed, the PartitionError fires
    before the model's _assess_count is incremented. The model remains
    usable on a fresh partition (if trained on that split's dev)."""
    import ml
    s1 = ml.split(data, "y", seed=42)
    m1 = ml.fit(s1.dev, "y", seed=42)
    ml.assess(m1, test=s1.test)  # burns s1.test

    # Build a second model on s1 but try the same test → rejected
    m2 = ml.fit(s1.dev, "y", seed=99)
    with pytest.raises(ml.PartitionError, match="already been assessed"):
        ml.assess(m2, test=s1.test)

    # m2's assess_count should NOT have been incremented (guard fired first)
    assert m2._assess_count == 0

    # Train a model on fresh split and assess its own test → works
    s2 = ml.split(data, "y", seed=99)
    m3 = ml.fit(s2.dev, "y", seed=42)
    evidence = ml.assess(m3, test=s2.test)
    assert "accuracy" in evidence


# ── 10. Thread safety ──

def test_concurrent_assess_only_one_succeeds(data):
    """Two threads racing to assess the same test partition.
    Exactly one should succeed, one should raise."""
    import ml
    s = ml.split(data, "y", seed=42)
    m1 = ml.fit(s.dev, "y", algorithm="logistic", seed=42)
    m2 = ml.fit(s.dev, "y", algorithm="random_forest", seed=42)

    results = {"m1": None, "m2": None}

    def assess_m1():
        try:
            ml.assess(m1, test=s.test)
            results["m1"] = "ok"
        except Exception as e:
            results["m1"] = type(e).__name__

    def assess_m2():
        try:
            ml.assess(m2, test=s.test)
            results["m2"] = "ok"
        except Exception as e:
            results["m2"] = type(e).__name__

    t1 = threading.Thread(target=assess_m1)
    t2 = threading.Thread(target=assess_m2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Exactly one OK, one error
    outcomes = [results["m1"], results["m2"]]
    assert "ok" in outcomes, f"Neither succeeded: {results}"
    assert outcomes.count("ok") == 1, f"Both succeeded (race condition): {results}"


# ── 11. Registry clear gives fresh budget ──

def test_registry_clear_resets_assessed(data):
    """Explicit registry clear + breadcrumb removal resets all state.

    With breadcrumb persistence, registry.clear() alone is not enough —
    breadcrumb files must also be removed. This mirrors the real workflow:
    `rm -rf .ml_assessed` is the explicit, visible, auditable reset.
    """
    import shutil

    import ml
    from ml._provenance import _BREADCRUMB_DIR, _registry

    s = ml.split(data, "y", seed=42)
    m = ml.fit(s.dev, "y", seed=42)
    ml.assess(m, test=s.test)

    _registry.clear()
    if _BREADCRUMB_DIR.exists():
        shutil.rmtree(_BREADCRUMB_DIR, ignore_errors=True)

    # Need to re-register partitions (split again)
    s2 = ml.split(data, "y", seed=42)
    m2 = ml.fit(s2.dev, "y", seed=42)
    evidence = ml.assess(m2, test=s2.test)
    assert "accuracy" in evidence
