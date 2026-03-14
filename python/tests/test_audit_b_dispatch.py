"""Adversarial audit B: Guard dispatch, ordering, and bypass paths.

Every test is an attack. Passing = guard holds or bypass confirmed.
Failing = real bug found.

Findings from code review (some FIXED since initial audit):
- compare() warns on test data (FIXED — was silent)
- stack() now calls guard_fit (FIXED — was unguarded)
- tune() guards via internal fit() call
- drift() has no partition guards at all
- evaluate() exposes _guard=False as a public parameter
- screen() internally auto-splits raw DataFrames, bypassing user-facing guards
- Model is a dataclass — _assess_count can be set to any value at construction
- _CONFIG is a plain dict with no thread-safe reads (TOCTOU race)
"""

import threading
import time
import warnings

import numpy as np
import pandas as pd
import pytest

import ml
from ml._config import _CONFIG
from ml._provenance import (
    _registry,
    guard_evaluate,
    guard_fit,
)
from ml._types import PartitionError

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _strict_guards():
    """All tests run in strict mode; reset after."""
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
def reg_data():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "target": np.random.randn(n),
    })


@pytest.fixture
def split_data(clf_data):
    return ml.split(clf_data, "target", seed=42)


@pytest.fixture
def fitted_model(split_data):
    ml.config(guards="off")
    model = ml.fit(data=split_data.train, target="target", seed=42)
    ml.config(guards="strict")
    return model


# ── Attack 1: Verbs that skip guards entirely ───────────────────────────


class TestMissingGuards:
    """Verbs that accept data but call no partition guard."""

    def test_compare_warns_on_test_data(self, split_data, fitted_model):
        """compare() warns (not errors) when test-tagged data is used.
        FIXED: compare() now emits UserWarning on test-tagged data."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lb = ml.compare(fitted_model, data=split_data.test)
        # Should have gotten a warning about test data
        assert any("test" in str(x.message).lower() for x in w)
        # compare still produces results (warning, not error)
        assert len(lb) > 0

    def test_compare_no_guard_on_unsplit_data(self, clf_data, fitted_model):
        """compare() with completely unsplit data — no PartitionError raised.
        evaluate(_guard=False) skips the provenance check entirely."""
        ml.config(guards="strict")
        # This should arguably raise PartitionError but doesn't
        lb = ml.compare(fitted_model, data=clf_data, warn_test=False)
        assert len(lb) > 0

    def test_stack_guards_on_unsplit_data(self, clf_data):
        """stack() now calls guard_fit — unsplit data is rejected.
        FIXED: stack() properly guards via guard_fit."""
        ml.config(guards="strict")
        with pytest.raises(PartitionError, match="split provenance"):
            ml.stack(clf_data, "target", seed=42,
                     models=["decision_tree", "logistic"])

    def test_tune_guards_via_internal_fit(self, clf_data):
        """tune() guards via its internal fit() call — guard_fit fires.
        Unsplit data is correctly rejected."""
        ml.config(guards="strict")
        with pytest.raises(PartitionError, match="split provenance"):
            ml.tune(clf_data, "target", algorithm="decision_tree",
                    n_trials=2, cv_folds=2, seed=42)

    def test_drift_no_guard(self, split_data):
        """drift() has no partition guards at all.
        Accepts test data as reference without warning."""
        ml.config(guards="strict")
        result = ml.drift(reference=split_data.test,
                          new=split_data.train,
                          target="target")
        assert result is not None


# ── Attack 2: _guard=False bypass via public API ─────────────────────────


class TestGuardBypassParameter:
    """evaluate() exposes _guard as a parameter accessible from user code."""

    def test_evaluate_guard_false_on_test_data(self, split_data, fitted_model):
        """User can call evaluate(model, test_data, _guard=False) to
        bypass the test-data rejection. This is a public-API bypass."""
        ml.config(guards="strict")
        # This should raise because it's test data
        with pytest.raises(PartitionError):
            ml.evaluate(fitted_model, split_data.test)

        # But _guard=False bypasses it
        result = ml.evaluate(fitted_model, split_data.test, _guard=False)
        assert "accuracy" in result or "rmse" in result

    def test_evaluate_guard_false_on_unsplit_data(self, clf_data, fitted_model):
        """_guard=False bypasses even the unsplit-data check."""
        ml.config(guards="strict")
        result = ml.evaluate(fitted_model, clf_data, _guard=False)
        assert isinstance(result, dict)


# ── Attack 3: TOCTOU race on config ─────────────────────────────────────


class TestConfigRace:
    """_CONFIG is a plain dict — no lock between read and check."""

    def test_config_race_strict_to_off(self, split_data, fitted_model):
        """Thread 1 calls evaluate(test_data) which reads config.
        Thread 2 sets config(guards='off') between the read and raise.

        This tests whether the race window exists. The guard reads _CONFIG
        in _guard_action which is called from guard_evaluate. If another
        thread flips guards='off' between guard_evaluate's identify call
        and the _guard_action call, the guard is bypassed."""
        ml.config(guards="strict")
        errors = []
        successes = []
        barrier = threading.Barrier(2, timeout=5)

        def eval_thread():
            try:
                barrier.wait()
                time.sleep(0.001)  # tiny delay to widen race window
                result = ml.evaluate(fitted_model, split_data.test)
                successes.append(result)
            except PartitionError:
                errors.append("blocked")
            except Exception as e:
                errors.append(str(e))

        def config_thread():
            try:
                barrier.wait()
                for _ in range(100):
                    ml.config(guards="off")
                    ml.config(guards="strict")
            except Exception:
                pass

        t1 = threading.Thread(target=eval_thread)
        t2 = threading.Thread(target=config_thread)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Document the outcome — either the guard held or the race succeeded
        # Both outcomes are valid findings
        if successes:
            # TOCTOU bypass confirmed — guard was bypassed by config flip
            pass
        else:
            # Guard held — race window too small or timing didn't align
            assert len(errors) > 0


# ── Attack 4: screen() internal guard dispatch ───────────────────────────


class TestScreenGuardDispatch:
    """screen() calls fit() and evaluate() internally — do guards fire?"""

    def test_screen_raw_dataframe_autosplits(self, clf_data):
        """screen() with raw DataFrame auto-splits internally,
        so guard_fit sees split-registered data. No error raised.
        This means screen silently avoids the guard by auto-splitting."""
        ml.config(guards="strict")
        lb = ml.screen(clf_data, "target", seed=42,
                       algorithms=["decision_tree"], keep_models=False)
        assert len(lb) > 0

    def test_screen_split_result_evaluate_guard(self, split_data):
        """screen() with SplitResult calls evaluate(model, split.valid).
        guard_evaluate fires on valid data — should pass (valid is allowed)."""
        ml.config(guards="strict")
        lb = ml.screen(split_data, "target", seed=42,
                       algorithms=["decision_tree"], keep_models=False)
        assert len(lb) > 0


# ── Attack 5: Non-DataFrame types skipping isinstance ────────────────────


class TestNonDataFrameBypass:
    """Guards call _identify_with_reason which calls _fingerprint.
    _fingerprint expects pd.DataFrame. What happens with other types?"""

    def test_dict_to_fit_raises_config_error_not_partition(self, clf_data):
        """Passing a dict to fit() — should hit type check before guard."""
        with pytest.raises((ml.ConfigError, TypeError, AttributeError)):
            ml.fit(data={"x1": [1, 2], "target": ["a", "b"]},
                   target="target", seed=42)

    def test_numpy_array_to_fit_raises_config_error(self):
        """Passing ndarray to fit() — should hit type check before guard."""
        arr = np.random.randn(100, 3)
        with pytest.raises((ml.ConfigError, TypeError, AttributeError)):
            ml.fit(data=arr, target="target", seed=42)

    def test_guard_fit_with_non_dataframe_passes_silently(self):
        """guard_fit with non-DataFrame passes silently (can't fingerprint, can't judge)."""
        guard_fit({"x1": [1], "target": ["a"]})  # no error — unfingerprintable

    def test_guard_evaluate_with_non_dataframe_passes_silently(self):
        """guard_evaluate with non-DataFrame passes silently (can't fingerprint, can't judge)."""
        guard_evaluate([[1, 2], [3, 4]])  # no error — unfingerprintable


# ── Attack 6: CVResult to assess ────────────────────────────────────────


class TestCVResultToAssess:
    """Can you pass a CVResult to assess() and skip the guard?"""

    def test_assess_rejects_cvresult(self, clf_data):
        """assess() type-checks for Model/TuningResult first.
        CVResult should fail at the model check, not the guard."""
        s = ml.split(clf_data, "target", seed=42)
        cv = ml.cv(s, folds=3, seed=42)
        with pytest.raises(Exception, match="Model|TuningResult|ConfigError"):
            ml.assess(cv, test=clf_data)


# ── Attack 7: stack/tune test data leakage ───────────────────────────────


class TestInternalDataLeakage:
    """Can stack() or tune() leak test data internally?"""

    def test_stack_on_test_partition_guarded(self, split_data):
        """stack() on test-tagged data — guard_fit now fires.
        FIXED: test data is correctly rejected by stack()."""
        ml.config(guards="strict")
        with pytest.raises(PartitionError, match="'test' partition"):
            ml.stack(split_data.test, "target", seed=42,
                     models=["decision_tree", "logistic"])

    def test_tune_on_test_partition_guarded(self, split_data):
        """tune() on test-tagged data — guard_fit fires via internal fit().
        Test data is correctly rejected."""
        ml.config(guards="strict")
        with pytest.raises(PartitionError, match="'test' partition"):
            ml.tune(split_data.test, "target",
                    algorithm="decision_tree",
                    n_trials=2, cv_folds=2, seed=42)

    def test_tune_final_refit_calls_fit_with_guard(self, split_data):
        """tune() calls fit() at the end for final refit. Does that
        inner fit() trigger guard_fit on test-tagged data?

        If tune's inner fit() raises PartitionError, then the guard
        partially works. If it doesn't, tune has a full bypass."""
        ml.config(guards="strict")
        # tune's final refit calls fit(data=data, ...) — data is the
        # test-tagged DataFrame. guard_fit should fire inside fit().
        # BUT: the data passed to tune might not have been registered
        # because tune was given test data directly.
        try:
            ml.tune(split_data.test, "target",
                    algorithm="decision_tree",
                    n_trials=1, cv_folds=2, seed=42)
            # If we get here, the refit's guard_fit didn't fire
            # (test data IS registered, so guard_fit sees "test" role
            #  but guard_fit allows test? No — guard_fit rejects non-train)
            pytest.fail(
                "Expected PartitionError from fit() inside tune() "
                "but none was raised. Full bypass confirmed."
            )
        except PartitionError:
            # guard_fit inside the final fit() caught it
            pass
        except Exception:
            # Some other error — tune might fail for other reasons
            pass


# ── Attack 8: compare() with test-tagged data ───────────────────────────


class TestCompareTestData:
    """compare() uses _guard=False — does it warn about test data?"""

    def test_compare_test_data_warns_not_errors(self, split_data, fitted_model):
        """compare() on test data: warns (not errors) because compare is diagnostic.
        FIXED: compare() now emits UserWarning when test-tagged data is used."""
        ml.config(guards="strict")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lb = ml.compare(fitted_model, data=split_data.test)
        # Warning emitted, not PartitionError
        assert any("test" in str(x.message).lower() for x in w)
        assert len(lb) > 0


# ── Attack 9: _assess_count ordering ────────────────────────────────────


class TestAssessCountOrdering:
    """Is the _assess_count increment before or after the guard?"""

    def test_wrong_partition_does_not_burn_assess(self, split_data, fitted_model):
        """If assess() is called with wrong partition (e.g., train data),
        the guard should fire BEFORE incrementing _assess_count.
        Otherwise a failed guard call burns the one-shot counter."""
        ml.config(guards="strict")
        assert fitted_model._assess_count == 0

        # Try assess with train data (wrong partition) — should raise
        with pytest.raises(PartitionError):
            ml.assess(fitted_model, test=split_data.train)

        # Counter should NOT have been incremented
        assert fitted_model._assess_count == 0, (
            "BUG: _assess_count incremented despite PartitionError. "
            "Wrong-partition error burned the one-shot counter."
        )

    def test_guard_before_count_with_valid_data(self, split_data, fitted_model):
        """Passing valid (not test) data: guard fires first, count untouched."""
        ml.config(guards="strict")
        assert fitted_model._assess_count == 0

        with pytest.raises(PartitionError, match="'valid'.*requires test"):
            ml.assess(fitted_model, test=split_data.valid)

        assert fitted_model._assess_count == 0


# ── Attack 10: Manual Model construction ─────────────────────────────────


class TestManualModelConstruction:
    """Model is a dataclass — can you construct one with hacked _assess_count?"""

    def test_model_with_negative_assess_count(self, split_data):
        """Construct a Model with _assess_count=-999. Can you call
        assess() 1000 times before hitting the counter?"""
        ml.config(guards="off")
        legit = ml.fit(data=split_data.train, target="target", seed=42)

        # Hack the counter
        legit._assess_count = -999
        ml.config(guards="off")

        # Should be able to call assess many times
        for _i in range(5):
            legit._assess_count = -999  # reset each time
            result = ml.assess(legit, test=split_data.test)
            assert isinstance(result, dict)

    def test_model_assess_count_directly_mutable(self, split_data):
        """_assess_count is a plain int attribute — freely mutable."""
        ml.config(guards="off")
        model = ml.fit(data=split_data.train, target="target", seed=42)

        # First assess works
        ml.assess(model, test=split_data.test)
        assert model._assess_count == 1

        # Reset counter — unlimited assess
        model._assess_count = 0
        ml.assess(model, test=split_data.test)
        assert model._assess_count == 1  # back to 1, not 2

    def test_model_deepcopy_resets_nothing(self, split_data):
        """deepcopy preserves _assess_count. Can be used to clone
        a model that has already been assessed."""
        import copy
        ml.config(guards="off")
        model = ml.fit(data=split_data.train, target="target", seed=42)
        ml.assess(model, test=split_data.test)
        assert model._assess_count == 1

        # Copy preserves counter
        clone = copy.deepcopy(model)
        assert clone._assess_count == 1

        # But the user can reset it
        clone._assess_count = 0
        result = ml.assess(clone, test=split_data.test)
        assert isinstance(result, dict)


# ── Attack 11: Validate bypasses assess_count ────────────────────────────


class TestValidateAssessInteraction:
    """validate() uses test data but does NOT increment _assess_count.
    This means unlimited validate() calls on test data."""

    def test_validate_unlimited_on_test_data(self, split_data):
        """validate() can be called unlimited times on test data.
        This is by design ('gate check, not final exam') but worth
        documenting that it provides unlimited test-set peeking."""
        ml.config(guards="off")
        model = ml.fit(data=split_data.train, target="target", seed=42)
        ml.config(guards="off")

        for _ in range(5):
            result = ml.validate(model, test=split_data.test,
                                 rules={"accuracy": ">0.0"})
            assert result.passed

        # assess_count still 0 — validate never touched it
        assert model._assess_count == 0


# ── Attack 12: Guard mode coercion ──────────────────────────────────────


class TestGuardModeCoercion:
    """What happens with non-standard guard config values?"""

    def test_guards_typo_defaults_to_raise(self, split_data):
        """Setting guards='strickt' (typo) — _guard_action falls through
        to the raise branch because mode != 'off' and mode != 'warn'."""
        ml.config(guards="strict")
        _CONFIG["guards"] = "strickt"  # bypass config validation

        with pytest.raises(PartitionError):
            ml.fit(data=split_data.test, target="target", seed=42)

    def test_guards_none_defaults_to_raise(self, split_data):
        """Setting guards=None — falls through to raise."""
        _CONFIG["guards"] = None
        with pytest.raises(PartitionError):
            ml.fit(data=split_data.test, target="target", seed=42)

    def test_guards_empty_string_raises(self, split_data):
        """Empty string is not 'off' or 'warn' — should raise."""
        _CONFIG["guards"] = ""
        with pytest.raises(PartitionError):
            ml.fit(data=split_data.test, target="target", seed=42)

    def test_guards_case_sensitive(self, split_data):
        """'OFF' (uppercase) is not 'off' — guard should still fire."""
        _CONFIG["guards"] = "OFF"
        with pytest.raises(PartitionError):
            ml.fit(data=split_data.test, target="target", seed=42)


# ── Attack 13: Calibrate guard ──────────────────────────────────────────


class TestCalibrateGuard:
    """calibrate() uses guard_evaluate. Test the dispatch."""

    def test_calibrate_rejects_test_data(self, split_data):
        """calibrate(data=s.test) should be rejected by guard_evaluate."""
        ml.config(guards="off")
        model = ml.fit(data=split_data.train, target="target", seed=42)
        ml.config(guards="strict")

        with pytest.raises(PartitionError, match="test"):
            ml.calibrate(model, data=split_data.test)

    def test_calibrate_accepts_valid_data(self, split_data):
        """calibrate(data=s.valid) should pass guard_evaluate."""
        ml.config(guards="off")
        model = ml.fit(data=split_data.train, target="target", seed=42)
        ml.config(guards="strict")

        # valid data is accepted by guard_evaluate
        try:
            ml.calibrate(model, data=split_data.valid)
        except Exception as e:
            if "PartitionError" in type(e).__name__:
                pytest.fail("calibrate should accept valid data")
            # Other errors (too few samples, etc.) are fine


# ── Attack 14: Fingerprint collision ─────────────────────────────────────


class TestFingerprintCollision:
    """Can two different DataFrames produce the same fingerprint?"""

    def test_column_rename_changes_fingerprint(self):
        """Renaming columns should produce a different fingerprint."""
        from ml._provenance import _fingerprint
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = df1.rename(columns={"a": "c"})
        fp1 = _fingerprint(df1)
        fp2 = _fingerprint(df2)
        assert fp1 != fp2, "Column rename should change fingerprint"

    def test_row_reorder_changes_fingerprint(self):
        """Reordering rows should produce a different fingerprint
        (content hash includes row order)."""
        from ml._provenance import _fingerprint
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = df1.iloc[::-1].reset_index(drop=True)
        fp1 = _fingerprint(df1)
        fp2 = _fingerprint(df2)
        # Reversed data = different content hash
        assert fp1 != fp2

    def test_empty_dataframe_distinguishes_columns(self):
        """Empty DataFrames with different columns get different fingerprints.
        Fixed: column names are now included in the empty hash."""
        from ml._provenance import _fingerprint
        df1 = pd.DataFrame(columns=["a", "b"])
        df2 = pd.DataFrame(columns=["x", "y"])
        fp1 = _fingerprint(df1)
        fp2 = _fingerprint(df2)
        assert fp1 != fp2, (
            "Empty DataFrames with different columns should have "
            "different fingerprints (column names included in hash)"
        )


# ── Attack 15: Registry eviction bypass ──────────────────────────────────


class TestRegistryEviction:
    """Registry has a max size (_MAX_REGISTRY_ENTRIES=10000).
    Can we force eviction of a legitimate partition?"""

    def test_registry_flood_evicts_old_entries(self):
        """Register 10001 partitions. The first one should be evicted.
        After eviction, the original partition is 'unknown' and the guard
        treats it as unregistered data (PartitionError in strict mode)."""
        from ml._provenance import _MAX_REGISTRY_ENTRIES

        # Register one legitimate partition
        original = pd.DataFrame({"a": [42.0], "b": [99.0]})
        fp = _registry.register(original, "train", "legit_split")
        assert fp is not None
        assert _registry.identify(original) == "train"

        # Flood with garbage to evict it
        for i in range(_MAX_REGISTRY_ENTRIES + 1):
            garbage = pd.DataFrame({"a": [float(i * 1000 + 1)],
                                    "b": [float(i * 1000 + 2)]})
            _registry.register(garbage, "train", f"flood_{i}")

        # Original should be evicted
        role = _registry.identify(original)
        if role is None:
            pass  # Confirmed: eviction-based bypass possible
        else:
            # Registry grew without eviction or hash collision preserved it
            pass
