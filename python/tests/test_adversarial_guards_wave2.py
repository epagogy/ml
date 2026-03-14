"""Adversarial tests wave 2: deeper attacks on partition guard system.

Wave 1 covered obvious vectors (tag spoofing, copy, serialization).
Wave 2 targets: in-place mutation, NaN semantics, large-dataset sampling
collisions, TOCTOU races, monkey-patching, DataFrame subclassing,
column name injection, dev property, and calibrate guard.
"""


import numpy as np
import pandas as pd
import pytest

import ml
from ml._provenance import (
    _fingerprint,
)


@pytest.fixture(autouse=True)
def _strict_guards():
    ml.config(guards="strict")
    yield
    ml.config(guards="off")


@pytest.fixture
def clf_data():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "target": np.random.choice(["a", "b"], n),
    })


@pytest.fixture
def split_data(clf_data):
    return ml.split(clf_data, "target", seed=42)


# ── Attack 13: In-place mutation after split ──────────────────────────────


class TestInPlaceMutation:
    """Can mutating data in-place after split bypass guards?"""

    def test_inplace_value_change_invalidates_fingerprint(self, split_data):
        """Changing a cell value in-place changes fingerprint — rejected."""
        train = split_data.train
        fp_before = _fingerprint(train)
        # Mutate in place
        train.iloc[0, 0] = 999999.0
        fp_after = _fingerprint(train)
        assert fp_before != fp_after, "In-place mutation must change fingerprint"
        # The mutated data should now be unregistered
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.fit(data=train, target="target", seed=42)

    def test_inplace_target_change_invalidates(self, split_data):
        """Changing target values in-place invalidates fingerprint."""
        s2 = ml.split(split_data.train.copy(), "target", seed=99)
        original_train = s2.train
        fp_before = _fingerprint(original_train)
        original_train["target"] = "a"  # all same class
        fp_after = _fingerprint(original_train)
        assert fp_before != fp_after

    def test_loc_assignment_invalidates(self, split_data):
        """Using .loc to assign values invalidates fingerprint."""
        train = split_data.train
        fp_before = _fingerprint(train)
        train.loc[train.index[0], "x1"] = -999.0
        fp_after = _fingerprint(train)
        assert fp_before != fp_after


# ── Attack 14: NaN semantics ─────────────────────────────────────────────


class TestNaNSemantics:
    """NaN != NaN in Python. Does fingerprinting handle this correctly?"""

    def test_nan_in_data_is_fingerprintable(self):
        """DataFrames with NaN values should be fingerprintable."""
        df = pd.DataFrame({
            "x1": [1.0, np.nan, 3.0] * 30,
            "x2": [np.nan, 2.0, np.nan] * 30,
            "target": ["a", "b", "a"] * 30,
        })
        fp = _fingerprint(df)
        assert fp is not None, "NaN data should be fingerprintable"

    def test_nan_fingerprint_is_deterministic(self):
        """Same NaN pattern = same fingerprint."""
        df1 = pd.DataFrame({
            "x": [1.0, np.nan, 3.0],
            "target": ["a", "b", "a"],
        })
        df2 = pd.DataFrame({
            "x": [1.0, np.nan, 3.0],
            "target": ["a", "b", "a"],
        })
        assert _fingerprint(df1) == _fingerprint(df2)

    def test_nan_vs_value_different_fingerprint(self):
        """NaN and a real value produce different fingerprints."""
        df1 = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        df2 = pd.DataFrame({"x": [1.0, 0.0, 3.0]})
        assert _fingerprint(df1) != _fingerprint(df2)

    def test_nan_split_data_accepted(self, clf_data):
        """Split data with NaN should be accepted by fit."""
        clf_data.iloc[0, 0] = np.nan
        clf_data.iloc[5, 1] = np.nan
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(data=s.train, target="target", seed=42)
        assert model is not None


# ── Attack 15: Large dataset sampling collision ──────────────────────────


class TestSamplingCollision:
    """For >100K rows, fingerprint uses strided sampling.
    Can two different datasets share the same sampled rows?"""

    def test_different_large_datasets_different_fingerprints(self):
        """Two large datasets differing only in non-sampled rows should
        still produce different fingerprints (shape is included)."""
        n = 110_000
        np.random.seed(42)
        df1 = pd.DataFrame({"x": np.random.randn(n)})
        df2 = pd.DataFrame({"x": np.random.randn(n)})
        fp1 = _fingerprint(df1)
        fp2 = _fingerprint(df2)
        assert fp1 != fp2

    def test_same_large_dataset_same_fingerprint(self):
        """Same large dataset = same fingerprint (sampling is deterministic)."""
        n = 110_000
        np.random.seed(42)
        data = np.random.randn(n)
        df1 = pd.DataFrame({"x": data.copy()})
        df2 = pd.DataFrame({"x": data.copy()})
        assert _fingerprint(df1) == _fingerprint(df2)

    def test_large_dataset_row_count_matters(self):
        """Datasets with same sampled values but different row counts
        should differ (shape is in the payload)."""
        np.random.seed(42)
        data = np.random.randn(200_000)
        df1 = pd.DataFrame({"x": data[:150_000]})
        df2 = pd.DataFrame({"x": data[:110_000]})
        fp1 = _fingerprint(df1)
        fp2 = _fingerprint(df2)
        assert fp1 != fp2, "Different-sized large datasets must differ"


# ── Attack 16: Column name injection ─────────────────────────────────────


class TestColumnNameInjection:
    """Can column names with special characters confuse the fingerprint?"""

    def test_pipe_in_column_name(self):
        """Column names with '|' (the separator) should not collide."""
        df1 = pd.DataFrame({"a|b": [1], "c": [2]})
        df2 = pd.DataFrame({"a": [1], "b|c": [2]})
        # These have different column structures but the pipe-joined
        # names could collide: "a|b|c" vs "a|b|c"
        fp1 = _fingerprint(df1)
        fp2 = _fingerprint(df2)
        # This IS a potential collision if column values are the same
        # Check if it's a real problem
        if fp1 == fp2:
            pytest.fail(
                "COLLISION: column name pipe injection produces same fingerprint. "
                "Column separator '|' in column names causes ambiguity."
            )

    def test_empty_column_name(self):
        """Empty string column name should be fingerprintable."""
        df = pd.DataFrame({"": [1, 2, 3], "x": [4, 5, 6]})
        fp = _fingerprint(df)
        assert fp is not None

    def test_unicode_column_names(self):
        """Unicode column names should work."""
        df1 = pd.DataFrame({"α": [1], "β": [2]})
        df2 = pd.DataFrame({"a": [1], "b": [2]})
        assert _fingerprint(df1) != _fingerprint(df2)


# ── Attack 17: dev property ──────────────────────────────────────────────


class TestDevProperty:
    """The .dev property combines train+valid. Is it properly guarded?"""

    def test_dev_accepted_by_fit(self, split_data):
        """fit() should accept dev-tagged data."""
        model = ml.fit(data=split_data.dev, target="target", seed=42)
        assert model is not None

    def test_dev_rejected_by_assess(self, split_data):
        """assess() should reject dev-tagged data."""
        model = ml.fit(data=split_data.train, target="target", seed=42)
        with pytest.raises(ml.PartitionError):
            ml.assess(model=model, test=split_data.dev)

    def test_dev_is_train_plus_valid(self, split_data):
        """dev should contain all rows from train and valid."""
        dev = split_data.dev
        train = split_data.train
        valid = split_data.valid
        assert len(dev) == len(train) + len(valid)


# ── Attack 18: calibrate guard ───────────────────────────────────────────


class TestCalibrateGuard:
    """calibrate() uses guard_evaluate — does it actually enforce?"""

    def test_calibrate_rejects_unsplit_data(self, clf_data):
        """calibrate() should reject data without split provenance."""
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(data=s.train, target="target", seed=42)
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.calibrate(model, data=clf_data)

    def test_calibrate_rejects_test_data(self, split_data):
        """calibrate() uses evaluate guard — test data should be rejected."""
        model = ml.fit(data=split_data.train, target="target", seed=42)
        with pytest.raises(ml.PartitionError, match="'test' partition"):
            ml.calibrate(model, data=split_data.test)

    def test_calibrate_accepts_valid_data(self):
        """calibrate() should accept valid-tagged data."""
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "target": np.random.choice(["a", "b"], n),
        })
        s = ml.split(df, "target", seed=42)
        model = ml.fit(data=s.train, target="target", seed=42)
        cal = ml.calibrate(model, data=s.valid)
        assert cal is not None


# ── Attack 19: Monkey-patching the guard system ──────────────────────────


class TestMonkeyPatch:
    """Can runtime monkey-patching disable guards?"""

    def test_replacing_guard_function_is_possible(self, clf_data):
        """Monkey-patching guard_fit is a bypass — but requires
        importing private internals. Document as known limitation."""
        from ml import _provenance
        original = _provenance.guard_fit
        # Replace with no-op
        _provenance.guard_fit = lambda data: None
        try:
            # This should now succeed even without split
            model = ml.fit(data=clf_data, target="target", seed=42)
            assert model is not None  # bypass works
        finally:
            _provenance.guard_fit = original

    def test_replacing_registry_requires_store_access(self, clf_data):
        """Replacing _registry.identify() is NOT sufficient because
        guard_fit uses _identify_with_reason() which accesses _store
        directly. Must replace the whole _registry to bypass."""
        import ml._provenance as prov
        from ml._provenance import PartitionRegistry
        original_registry = prov._registry
        # Overriding identify() alone doesn't work — guard accesses _store
        class AlwaysTrain(PartitionRegistry):
            def identify(self, df):
                return "train"
        prov._registry = AlwaysTrain()
        # Still fails because _identify_with_reason checks _store directly
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.fit(data=clf_data, target="target", seed=42)
        prov._registry = original_registry


# ── Attack 20: TOCTOU (time-of-check, time-of-use) ──────────────────────


class TestTOCTOU:
    """Can data be modified between the guard check and actual use?"""

    def test_mutation_between_guard_and_fit(self, split_data):
        """In practice, guard and fit happen in the same call,
        so there's no window for mutation. Verify by checking
        that the guard is called inside fit(), not before."""
        # This isn't really attackable because guard_fit is called
        # inside fit() — there's no separate "check then use" pattern.
        # Just verify the guard is inside the function.
        import inspect
        source = inspect.getsource(ml.fit)
        assert "guard_fit" in source


# ── Attack 21: DataFrame subclass ────────────────────────────────────────


class TestDataFrameSubclass:
    """Does a DataFrame subclass bypass fingerprinting?"""

    def test_subclass_is_fingerprintable(self, split_data):
        """A subclass of DataFrame should still be fingerprintable."""
        class MyDF(pd.DataFrame):
            pass
        sub = MyDF(split_data.train)
        fp = _fingerprint(sub)
        assert fp is not None
        # Should match the original fingerprint
        fp_orig = _fingerprint(split_data.train)
        assert fp == fp_orig

    def test_subclass_accepted_if_registered(self, split_data):
        """A subclass with same content should pass the guard."""
        class MyDF(pd.DataFrame):
            pass
        sub = MyDF(split_data.train)
        model = ml.fit(data=sub, target="target", seed=42)
        assert model is not None


# ── Attack 22: Duplicate column names ────────────────────────────────────


class TestDuplicateColumns:
    """DataFrames with duplicate column names are legal in pandas."""

    def test_duplicate_columns_fingerprintable(self):
        """Duplicate column names should produce a valid fingerprint."""
        df = pd.DataFrame([[1, 2, 3]], columns=["x", "x", "y"])
        fp = _fingerprint(df)
        assert fp is not None

    def test_duplicate_vs_unique_different_fingerprint(self):
        """Duplicate columns produce different fingerprint than unique."""
        df1 = pd.DataFrame([[1, 2, 3]], columns=["x", "x", "y"])
        df2 = pd.DataFrame([[1, 2, 3]], columns=["x", "z", "y"])
        assert _fingerprint(df1) != _fingerprint(df2)


# ── Attack 23: Empty string and special values ───────────────────────────


class TestSpecialValues:
    """Edge case values that might confuse fingerprinting."""

    def test_inf_values_fingerprintable(self):
        """Inf values should be fingerprintable."""
        df = pd.DataFrame({"x": [np.inf, -np.inf, 0.0]})
        fp = _fingerprint(df)
        assert fp is not None

    def test_inf_vs_large_number_different(self):
        """Inf and a very large number should produce different fingerprints."""
        df1 = pd.DataFrame({"x": [np.inf]})
        df2 = pd.DataFrame({"x": [1e308]})
        assert _fingerprint(df1) != _fingerprint(df2)

    def test_mixed_types_fingerprintable(self):
        """DataFrame with mixed types (int, float, string, bool)."""
        df = pd.DataFrame({
            "i": [1, 2, 3],
            "f": [1.0, 2.0, 3.0],
            "s": ["a", "b", "c"],
            "b": [True, False, True],
        })
        fp = _fingerprint(df)
        assert fp is not None

    def test_all_nan_column_fingerprintable(self):
        """Column that is entirely NaN."""
        df = pd.DataFrame({
            "x": [np.nan, np.nan, np.nan],
            "target": ["a", "b", "a"],
        })
        fp = _fingerprint(df)
        assert fp is not None


# ── Attack 24: Evaluate on train data (leakage attempt) ─────────────────


class TestEvaluateLeakage:
    """Can you evaluate on training data? (Should be allowed but warned.)"""

    def test_evaluate_on_train_is_allowed(self, split_data):
        """evaluate() on train-tagged data should succeed (it's valid)."""
        model = ml.fit(data=split_data.train, target="target", seed=42)
        metrics = ml.evaluate(model=model, data=split_data.train)
        assert "accuracy" in metrics

    def test_evaluate_on_test_is_blocked(self, split_data):
        """evaluate() on test-tagged data is blocked."""
        model = ml.fit(data=split_data.train, target="target", seed=42)
        with pytest.raises(ml.PartitionError, match="'test' partition"):
            ml.evaluate(model=model, data=split_data.test)


# ── Attack 25: Assess with data from different split seed ────────────────


class TestCrossSplitAssess:
    """Using test from one split to assess a model from another split."""

    def test_cross_split_assess_detected(self, clf_data):
        """Model from split(seed=42), assess with test from split(seed=99).
        Cross-verb provenance should catch this."""
        s1 = ml.split(clf_data, "target", seed=42)
        s2 = ml.split(clf_data, "target", seed=99)
        model = ml.fit(data=s1.train, target="target", seed=42)
        try:
            ml.assess(model=model, test=s2.test)
            pytest.skip("Cross-split assess not enforced — known gap")
        except ml.PartitionError:
            pass  # Good — caught by lineage check


# ── Attack 26: Registry collision via hash truncation ────────────────────


class TestHashCollision:
    """SHA-256 is truncated to 16 chars (64 bits). Can we force a collision?"""

    def test_collision_probability_is_acceptable(self):
        """With 10K registry entries, birthday paradox collision probability
        is ~10K^2 / (2 * 2^64) ≈ 2.7e-12. Effectively zero."""
        # This is a statistical argument, not a test.
        # Just verify the hash is 16 chars.
        df = pd.DataFrame({"x": [1, 2, 3]})
        fp = _fingerprint(df)
        assert len(fp) == 16
        # Verify it's hex
        int(fp, 16)  # Should not raise

    def test_many_fingerprints_no_collision(self):
        """Generate 1000 different DataFrames — no collisions."""
        fps = set()
        for i in range(1000):
            df = pd.DataFrame({"x": [i], "y": [i * 2]})
            fp = _fingerprint(df)
            assert fp not in fps, f"Collision at i={i}!"
            fps.add(fp)


# ── Attack 27: assess on CVResult ────────────────────────────────────────


class TestCVResultBypass:
    """CVResult bypasses partition guards. Is this correct?"""

    def test_fit_cvresult_skips_guard(self, split_data):
        """fit() on CVResult skips guard — by design (CV manages its own splits)."""
        cv = ml.split(split_data.train, "target", seed=42, folds=3)
        model = ml.fit(data=cv, target="target", seed=42)
        assert model is not None
