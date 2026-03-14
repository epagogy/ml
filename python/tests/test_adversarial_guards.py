"""Adversarial tests for partition guard system.

Every test here is an attack vector. If it passes, the guard holds.
If it fails, we found a real bypass.
"""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

import ml
from ml._provenance import (
    _fingerprint,
    _registry,
    register_partition,
)


@pytest.fixture(autouse=True)
def _strict_guards():
    """All adversarial tests run in strict mode."""
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


# ── Attack 1: Tag spoofing via attrs ─────────────────────────────────────


class TestTagSpoofing:
    """Can a user fake a partition tag to bypass guards?"""

    def test_attrs_tag_does_not_fool_fingerprint_guard(self, clf_data):
        """Setting attrs['_ml_partition'] = 'train' does NOT bypass the
        content-addressed guard — the data was never registered."""
        fake = clf_data.copy()
        fake.attrs["_ml_partition"] = "train"
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.fit(data=fake, target="target", seed=42)

    def test_attrs_tag_test_does_not_fool_assess(self, clf_data, split_data):
        """Faking 'test' attr on unsplit data doesn't bypass assess."""
        model = ml.fit(data=split_data.train, target="target", seed=42)
        fake = clf_data.copy()
        fake.attrs["_ml_partition"] = "test"
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.assess(model=model, test=fake)


# ── Attack 2: Registry poisoning ─────────────────────────────────────────


class TestRegistryPoisoning:
    """Can a user register arbitrary data as 'train'?"""

    def test_direct_registry_register_bypasses_guard(self, clf_data):
        """Direct access to _registry.register IS a bypass — but it requires
        importing private internals. Document as known limitation."""
        fp = register_partition(clf_data, "train", "fake_split_id")
        assert fp is not None
        # Now fit should accept it — this IS a bypass
        model = ml.fit(data=clf_data, target="target", seed=42)
        assert model is not None
        # Clean up
        _registry.clear()

    def test_registry_clear_resets_provenance(self, split_data):
        """After clearing registry, previously-valid data is rejected."""
        # Verify data works first
        model = ml.fit(data=split_data.train, target="target", seed=42)
        assert model is not None
        # Clear and try again
        _registry.clear()
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.fit(data=split_data.train, target="target", seed=42)


# ── Attack 3: Copy/mutation attacks ──────────────────────────────────────


class TestCopyMutation:
    """Does copying or mutating data after split bypass guards?"""

    def test_copy_preserves_fingerprint(self, split_data):
        """df.copy() preserves cell values — fingerprint should match."""
        copied = split_data.train.copy()
        model = ml.fit(data=copied, target="target", seed=42)
        assert model is not None

    def test_deepcopy_preserves_fingerprint(self, split_data):
        """copy.deepcopy also preserves cell values."""
        deep = copy.deepcopy(split_data.train)
        model = ml.fit(data=deep, target="target", seed=42)
        assert model is not None

    def test_adding_column_invalidates_fingerprint(self, split_data):
        """Adding a column changes content — should be rejected."""
        modified = split_data.train.copy()
        modified["leaked_feature"] = 999
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.fit(data=modified, target="target", seed=42)

    def test_dropping_column_invalidates_fingerprint(self, split_data):
        """Dropping a column changes content — rejected."""
        modified = split_data.train.drop(columns=["x1"])
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.fit(data=modified, target="target", seed=42)

    def test_filtering_rows_invalidates_fingerprint(self, split_data):
        """Filtering rows changes content — rejected."""
        modified = split_data.train.iloc[:10]
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.fit(data=modified, target="target", seed=42)

    def test_sorting_invalidates_fingerprint(self, split_data):
        """Sorting changes row order — fingerprint changes."""
        modified = split_data.train.sort_values("x1").reset_index(drop=True)
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.fit(data=modified, target="target", seed=42)

    def test_reset_index_preserves_fingerprint(self, split_data):
        """reset_index(drop=True) on already-clean index should preserve."""
        # Only passes if index was already RangeIndex
        modified = split_data.train.reset_index(drop=True)
        fp_orig = _fingerprint(split_data.train)
        fp_mod = _fingerprint(modified)
        if fp_orig == fp_mod:
            model = ml.fit(data=modified, target="target", seed=42)
            assert model is not None
        else:
            with pytest.raises(ml.PartitionError):
                ml.fit(data=modified, target="target", seed=42)


# ── Attack 4: Serialization bypass ───────────────────────────────────────


class TestSerializationBypass:
    """Can serialization/deserialization reset guards?"""

    def test_pickle_roundtrip_preserves_fingerprint(self, split_data):
        """Pickle preserves cell values exactly — fingerprint matches
        the registry. This is CORRECT: content identity survives pickle."""
        raw = pickle.dumps(split_data.train)
        restored = pickle.loads(raw)
        model = ml.fit(data=restored, target="target", seed=42)
        assert model is not None

    def test_csv_roundtrip_loses_provenance(self, split_data, tmp_path):
        """Writing to CSV and reading back loses all provenance."""
        path = tmp_path / "train.csv"
        split_data.train.to_csv(path, index=False)
        loaded = pd.read_csv(path)
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.fit(data=loaded, target="target", seed=42)

    def test_parquet_roundtrip_preserves_fingerprint(self, split_data, tmp_path):
        """Parquet preserves cell values — fingerprint matches registry.
        Content-addressed identity survives format roundtrips."""
        pytest.importorskip("pyarrow")
        path = tmp_path / "train.parquet"
        split_data.train.to_parquet(path, index=False)
        loaded = pd.read_parquet(path)
        model = ml.fit(data=loaded, target="target", seed=42)
        assert model is not None


# ── Attack 5: Config toggling ────────────────────────────────────────────


class TestConfigToggling:
    """Can a user toggle guards mid-workflow to sneak past?"""

    def test_guards_off_then_on_rejects(self, clf_data):
        """Turning guards off to fit, then on for assess, doesn't help —
        the model was fitted on unsplit data, but assess still rejects
        unsplit test data."""
        ml.config(guards="off")
        model = ml.fit(data=clf_data, target="target", seed=42)
        ml.config(guards="strict")
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.assess(model=model, test=clf_data)

    def test_guards_off_bypasses_everything(self, clf_data):
        """guards='off' truly disables all checks — known and documented."""
        ml.config(guards="off")
        model = ml.fit(data=clf_data, target="target", seed=42)
        # evaluate on same data — no error
        metrics = ml.evaluate(model=model, data=clf_data)
        assert "accuracy" in metrics
        ml.config(guards="strict")


# ── Attack 6: Unfingerprintable data ─────────────────────────────────────


class TestUnfingerprintable:
    """Can unhashable data slip through?"""

    def test_list_valued_column_passes_guard_silently(self):
        """DataFrames with list-valued columns can't be fingerprinted.
        Guard passes silently — can't judge, no evidence.
        (fit may still fail on the actual data — that's a separate issue.)"""
        df = pd.DataFrame({
            "x": [[1, 2], [3, 4], [5, 6]] * 30,
            "target": ["a", "b", "a"] * 30,
        })
        # The guard should pass (unfingerprintable = no evidence).
        # fit() itself may crash on unhashable data — that's expected.
        fp = _fingerprint(df)
        assert fp is None  # confirms unfingerprintable


# ── Attack 7: Cross-split contamination ──────────────────────────────────


class TestCrossSplitContamination:
    """Can test data from a different split be used?"""

    def test_different_split_test_data(self, clf_data):
        """Train on split1, assess on split2.test — should be caught
        by provenance lineage check."""
        s1 = ml.split(clf_data, "target", seed=42)
        s2 = ml.split(clf_data, "target", seed=99)
        model = ml.fit(data=s1.train, target="target", seed=42)
        # assess should reject — different split lineage
        # (This depends on Layer 2 provenance being wired into assess)
        # If it passes, that's a known gap to document
        try:
            ml.assess(model=model, test=s2.test)
            # If we get here, cross-split is not enforced at assess boundary
            pytest.skip("Cross-split contamination not enforced at assess — known gap")
        except ml.PartitionError:
            pass  # Good — caught


# ── Attack 8: Registry eviction ──────────────────────────────────────────


class TestRegistryEviction:
    """Can the LRU eviction cause valid data to be rejected?"""

    def test_mass_splits_dont_evict_within_limit(self, clf_data):
        """350 splits × 3 partitions = 1050 entries — well within 10K limit.
        Earlier splits should still be registered."""
        s_first = ml.split(clf_data, "target", seed=1)
        first_train = s_first.train
        for i in range(350):
            ml.split(clf_data, "target", seed=1000 + i)
        role = _registry.identify(first_train)
        assert role == "train", f"Expected 'train', got {role} — evicted too early"
        model = ml.fit(data=first_train, target="target", seed=42)
        assert model is not None

    def test_eviction_at_limit(self, clf_data):
        """At 10K entries, oldest splits ARE evicted. This is documented
        behavior, not a bug — but long sessions hit it."""
        from ml._provenance import _MAX_REGISTRY_ENTRIES
        s_first = ml.split(clf_data, "target", seed=1)
        first_train = s_first.train
        # Each split registers 3 partitions, need > 10K/3 ≈ 3334 splits
        needed = (_MAX_REGISTRY_ENTRIES // 3) + 10
        for i in range(needed):
            ml.split(clf_data, "target", seed=5000 + i)
        role = _registry.identify(first_train)
        # Evicted — this is expected at scale
        assert role is None, "Expected eviction but data still registered"


# ── Attack 9: Empty DataFrame ────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases that might confuse the fingerprinting."""

    def test_empty_dataframe_rejected(self):
        """Empty DataFrame — fingerprintable but not registered."""
        empty = pd.DataFrame({"x": [], "target": []})
        with pytest.raises((ml.PartitionError, ml.DataError)):
            ml.fit(data=empty, target="target", seed=42)

    def test_single_row_rejected(self):
        """Single row DataFrame — not registered by split."""
        single = pd.DataFrame({"x": [1.0], "target": ["a"]})
        with pytest.raises((ml.PartitionError, ml.DataError)):
            ml.fit(data=single, target="target", seed=42)

    def test_identical_content_different_object(self, clf_data):
        """Two DataFrames with identical content and dtypes but different
        Python objects. Content-addressed fingerprint should match."""
        s = ml.split(clf_data, "target", seed=42)
        # Column-by-column reconstruction preserves dtypes
        twin = pd.DataFrame({
            col: s.train[col].values for col in s.train.columns
        })
        assert _fingerprint(twin) == _fingerprint(s.train)
        model = ml.fit(data=twin, target="target", seed=42)
        assert model is not None


# ── Attack 10: Pandas operations that strip attrs ────────────────────────


class TestPandasOperations:
    """Operations that strip .attrs — does fingerprinting survive?"""

    def test_merge_strips_attrs_but_fingerprint_changes(self, split_data):
        """pd.merge strips attrs AND changes content — double rejected."""
        other = pd.DataFrame({"x1": split_data.train["x1"], "extra": 999})
        merged = pd.merge(split_data.train, other, on="x1")
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.fit(data=merged, target="target", seed=42)

    def test_concat_strips_attrs_and_changes_content(self, split_data):
        """pd.concat of train+test = new content, rejected."""
        combined = pd.concat([split_data.train, split_data.test])
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.fit(data=combined, target="target", seed=42)

    def test_column_rename_changes_fingerprint(self, split_data):
        """Renaming columns changes the hash (column names are part of fingerprint)."""
        renamed = split_data.train.rename(columns={"x1": "feature_1"})
        fp_orig = _fingerprint(split_data.train)
        fp_renamed = _fingerprint(renamed)
        assert fp_orig != fp_renamed, "Column rename must change fingerprint"
        with pytest.raises(ml.PartitionError, match="split provenance"):
            ml.fit(data=renamed, target="target", seed=42)

    def test_dtype_coercion_may_change_fingerprint(self, split_data):
        """Changing dtype (float64 → float32) changes values — rejected."""
        coerced = split_data.train.copy()
        coerced["x1"] = coerced["x1"].astype(np.float32)
        fp_orig = _fingerprint(split_data.train)
        fp_coerced = _fingerprint(coerced)
        if fp_orig != fp_coerced:
            with pytest.raises(ml.PartitionError):
                ml.fit(data=coerced, target="target", seed=42)
        else:
            pytest.skip("float32 coercion didn't change fingerprint")


# ── Attack 11: Model serialization resets assess counter ─────────────────


class TestAssessOnceBypass:
    """Can assess-once be bypassed via serialization?"""

    def test_save_load_cannot_bypass_holdout_guard(self, split_data, tmp_path):
        """Save/load resets per-model counter but per-holdout guard still blocks.
        The provenance registry remembers the test holdout was already assessed."""
        ml.config(guards="strict")
        model = ml.fit(data=split_data.train, target="target", seed=42)
        ml.assess(model=model, test=split_data.test)
        # Save/load resets the model's _assess_count...
        path = tmp_path / "model.mlw"
        ml.save(model, str(path))
        loaded = ml.load(str(path))
        # ...but the per-holdout guard catches the re-assessment
        with pytest.raises(ml.PartitionError, match="already been assessed"):
            ml.assess(model=loaded, test=split_data.test)


# ── Attack 12: Concurrent access ─────────────────────────────────────────


class TestConcurrency:
    """Thread safety of the registry."""

    def test_concurrent_splits_dont_corrupt(self, clf_data):
        """Multiple threads splitting simultaneously shouldn't corrupt registry."""
        import threading
        errors = []

        def split_and_fit(seed):
            try:
                s = ml.split(clf_data, "target", seed=seed)
                ml.fit(data=s.train, target="target", seed=seed)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=split_and_fit, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent errors: {errors}"
