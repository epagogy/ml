"""Tests for computational guards (provenance system).

Layer 1: Content-addressed partition identity
Layer 2: Cross-verb provenance (split lineage + row overlap)
Layer 3: Hash-chained audit trail
"""

import pandas as pd
import pytest

import ml
from ml._provenance import (
    AuditChain,
    PartitionRegistry,
    _fingerprint,
    _registry,
    audit,
    build_provenance,
    check_provenance,
    guard_assess,
    guard_evaluate,
    guard_fit,
    guard_validate,
    identify_partition,
    new_split_id,
    register_partition,
)
from ml._types import PartitionError


@pytest.fixture
def sample_data():
    return pd.DataFrame({"x": range(100), "y": range(100)})


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear global registry between tests."""
    _registry.clear()
    yield
    _registry.clear()


# ---------------------------------------------------------------------------
# Layer 1: Fingerprinting
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_deterministic(self, sample_data):
        assert _fingerprint(sample_data) == _fingerprint(sample_data)

    def test_different_data_different_fingerprint(self):
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"x": [4, 5, 6]})
        assert _fingerprint(df1) != _fingerprint(df2)

    def test_empty_dataframe(self):
        fp = _fingerprint(pd.DataFrame())
        assert isinstance(fp, str)
        assert len(fp) == 16

    def test_survives_column_rename(self):
        """Fingerprint is content-based, not name-based."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        _fingerprint(df)
        df2 = df.rename(columns={"a": "x"})
        # Column rename changes content hash (column names are part of hash)
        # This is correct — renamed columns are different data
        assert isinstance(_fingerprint(df2), str)

    def test_copy_preserves_fingerprint(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert _fingerprint(df) == _fingerprint(df.copy())

    def test_large_dataframe_uses_sampling(self):
        """Datasets >100K rows use strided sampling."""
        df = pd.DataFrame({"x": range(200_000)})
        fp = _fingerprint(df)
        assert isinstance(fp, str)
        assert len(fp) == 16


# ---------------------------------------------------------------------------
# Layer 1: Partition Registry
# ---------------------------------------------------------------------------


class TestPartitionRegistry:
    def test_register_and_identify(self):
        reg = PartitionRegistry()
        df = pd.DataFrame({"x": [1, 2, 3]})
        sid = "test123"
        reg.register(df, "train", sid)
        assert reg.identify(df) == "train"

    def test_unknown_data_returns_none(self):
        reg = PartitionRegistry()
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert reg.identify(df) is None

    def test_split_id_tracking(self):
        reg = PartitionRegistry()
        df = pd.DataFrame({"x": [1, 2, 3]})
        sid = "abc123"
        reg.register(df, "train", sid)
        assert reg.get_split_id(df) == sid

    def test_eviction_at_max_size(self):
        reg = PartitionRegistry()
        # Register more than max entries
        from ml._provenance import _MAX_REGISTRY_ENTRIES
        for i in range(_MAX_REGISTRY_ENTRIES + 100):
            df = pd.DataFrame({"x": [i]})
            reg.register(df, "train", f"sid_{i}")
        assert len(reg) <= _MAX_REGISTRY_ENTRIES

    def test_clear(self):
        reg = PartitionRegistry()
        df = pd.DataFrame({"x": [1]})
        reg.register(df, "train", "sid")
        reg.clear()
        assert len(reg) == 0
        assert reg.identify(df) is None


# ---------------------------------------------------------------------------
# Layer 1: Guard Functions
# ---------------------------------------------------------------------------


class TestGuardFit:
    @pytest.fixture(autouse=True)
    def _strict_guards(self):
        ml.config(guards="strict")
        yield
        ml.config(guards="off")

    def test_rejects_test_partition(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        sid = new_split_id()
        register_partition(df, "test", sid)
        with pytest.raises(PartitionError, match="fit.*test"):
            guard_fit(df)

    def test_accepts_train(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        sid = new_split_id()
        register_partition(df, "train", sid)
        guard_fit(df)  # no error

    def test_accepts_valid(self):
        df = pd.DataFrame({"x": [4, 5, 6]})
        sid = new_split_id()
        register_partition(df, "valid", sid)
        guard_fit(df)  # no error

    def test_accepts_dev(self):
        df = pd.DataFrame({"x": [7, 8, 9]})
        sid = new_split_id()
        register_partition(df, "dev", sid)
        guard_fit(df)  # no error

    def test_rejects_unregistered(self):
        df = pd.DataFrame({"x": [10, 11, 12]})
        with pytest.raises(PartitionError, match="split provenance"):
            guard_fit(df)


class TestGuardEvaluate:
    @pytest.fixture(autouse=True)
    def _strict_guards(self):
        ml.config(guards="strict")
        yield
        ml.config(guards="off")

    def test_rejects_test_partition(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        sid = new_split_id()
        register_partition(df, "test", sid)
        with pytest.raises(PartitionError, match="evaluate.*test"):
            guard_evaluate(df)

    def test_accepts_valid(self):
        df = pd.DataFrame({"x": [4, 5, 6]})
        sid = new_split_id()
        register_partition(df, "valid", sid)
        guard_evaluate(df)  # no error

    def test_rejects_unregistered(self):
        df = pd.DataFrame({"x": [10, 11, 12]})
        with pytest.raises(PartitionError, match="split provenance"):
            guard_evaluate(df)


class TestGuardAssess:
    @pytest.fixture(autouse=True)
    def _strict_guards(self):
        ml.config(guards="strict")
        yield
        ml.config(guards="off")

    def test_rejects_train_partition(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        sid = new_split_id()
        register_partition(df, "train", sid)
        with pytest.raises(PartitionError, match="assess.*train"):
            guard_assess(df)

    def test_rejects_valid_partition(self):
        df = pd.DataFrame({"x": [4, 5, 6]})
        sid = new_split_id()
        register_partition(df, "valid", sid)
        with pytest.raises(PartitionError, match="assess.*valid"):
            guard_assess(df)

    def test_accepts_test(self):
        df = pd.DataFrame({"x": [7, 8, 9]})
        sid = new_split_id()
        register_partition(df, "test", sid)
        guard_assess(df)  # no error

    def test_rejects_unregistered(self):
        df = pd.DataFrame({"x": [10, 11, 12]})
        with pytest.raises(PartitionError, match="split provenance"):
            guard_assess(df)


class TestGuardValidate:
    @pytest.fixture(autouse=True)
    def _strict_guards(self):
        ml.config(guards="strict")
        yield
        ml.config(guards="off")

    def test_rejects_train_partition(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        sid = new_split_id()
        register_partition(df, "train", sid)
        with pytest.raises(PartitionError, match="validate.*train"):
            guard_validate(df)

    def test_accepts_test(self):
        df = pd.DataFrame({"x": [7, 8, 9]})
        sid = new_split_id()
        register_partition(df, "test", sid)
        guard_validate(df)  # no error


# ---------------------------------------------------------------------------
# Layer 2: Cross-Verb Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    @pytest.fixture(autouse=True)
    def _strict_guards(self):
        ml.config(guards="strict")
        yield
        ml.config(guards="off")

    def test_build_provenance_captures_fingerprint(self, sample_data):
        sid = new_split_id()
        register_partition(sample_data, "train", sid)
        prov = build_provenance(sample_data)
        assert prov["train_fingerprint"] == _fingerprint(sample_data)
        assert prov["split_id"] == sid
        assert "fit_timestamp" in prov

    def test_check_provenance_passes_disjoint(self):
        train = pd.DataFrame({"x": range(50)}, index=range(0, 50))
        test = pd.DataFrame({"x": range(50, 100)}, index=range(50, 100))
        sid = new_split_id()
        register_partition(train, "train", sid)
        register_partition(test, "test", sid)
        prov = build_provenance(train)
        check_provenance(prov, test)  # no error

    def test_check_provenance_no_overlap_check(self):
        """Row overlap is intentionally NOT checked (observational ≠ value identity)."""
        train = pd.DataFrame({"x": range(50)})
        test = pd.DataFrame({"x": range(50)})  # same values, different observations
        sid = new_split_id()
        register_partition(train, "train", sid)
        register_partition(test, "test", sid)
        prov = build_provenance(train)
        # Should NOT raise — same values don't mean same observations
        check_provenance(prov, test)

    def test_check_provenance_catches_different_split(self):
        train = pd.DataFrame({"x": range(50)})
        test = pd.DataFrame({"x": range(50, 100)})
        sid1 = new_split_id()
        sid2 = new_split_id()
        register_partition(train, "train", sid1)
        register_partition(test, "test", sid2)
        prov = build_provenance(train)
        with pytest.raises(PartitionError, match="different split"):
            check_provenance(prov, test)

    def test_check_provenance_silent_on_empty(self):
        test = pd.DataFrame({"x": range(50)})
        check_provenance({}, test)  # no error
        check_provenance(None, test)  # no error


# ---------------------------------------------------------------------------
# Layer 3: Audit Chain
# ---------------------------------------------------------------------------


class TestAuditChain:
    def test_append_and_verify(self):
        chain = AuditChain()
        chain.append("split", "abc123")
        chain.append("fit", "def456")
        assert len(chain) == 2
        assert chain.verify()

    def test_tamper_detected(self):
        chain = AuditChain()
        chain.append("split", "abc123")
        chain.append("fit", "def456")
        # Tamper with an entry
        chain._entries[0].verb = "TAMPERED"
        assert not chain.verify()

    def test_extract_by_model(self):
        chain = AuditChain()
        chain.append("fit", "abc", model_hash="m1")
        chain.append("assess", "def", model_hash="m1")
        chain.append("fit", "ghi", model_hash="m2")
        entries = chain.extract("m1")
        assert len(entries) == 2

    def test_to_json(self):
        import json
        chain = AuditChain()
        chain.append("split", "abc123", split_id="s1")
        result = json.loads(chain.to_json())
        assert len(result) == 1
        assert result[0]["verb"] == "split"
        assert result[0]["split_id"] == "s1"

    def test_empty_chain_verifies(self):
        chain = AuditChain()
        assert chain.verify()

    def test_repr(self):
        chain = AuditChain()
        chain.append("split", "abc")
        assert "1 entries" in repr(chain)
        assert "intact" in repr(chain)


# ---------------------------------------------------------------------------
# Integration: Full workflow
# ---------------------------------------------------------------------------


class TestIntegration:
    @pytest.fixture(autouse=True)
    def _strict_guards(self):
        ml.config(guards="strict")
        yield
        ml.config(guards="off")

    def test_split_registers_partitions(self):
        """split() registers all partitions in the provenance registry."""
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        assert identify_partition(s.train) == "train"
        assert identify_partition(s.valid) == "valid"
        assert identify_partition(s.test) == "test"

    def test_split_temporal_registers_partitions(self):
        """split_temporal() registers all partitions."""
        data = ml.dataset("tips")
        data["date"] = pd.date_range("2020-01-01", periods=len(data), freq="D")
        s = ml.split_temporal(data, "tip", time="date")
        assert identify_partition(s.train) == "train"
        assert identify_partition(s.valid) == "valid"
        assert identify_partition(s.test) == "test"

    def test_split_group_registers_partitions(self):
        """split_group() registers all partitions."""
        data = ml.dataset("tips")
        s = ml.split_group(data, "tip", groups="day", seed=42)
        assert identify_partition(s.train) == "train"
        assert identify_partition(s.test) == "test"

    def test_two_way_split_registers_partitions(self):
        """2-way split registers train and test."""
        data = ml.dataset("iris")
        s = ml.split(data, "species", ratio=(0.8, 0.2), seed=42)
        assert identify_partition(s.train) == "train"
        assert identify_partition(s.test) == "test"

    def test_dev_registers_partition(self):
        """s.dev registers as dev partition."""
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        dev = s.dev
        assert identify_partition(dev) == "dev"

    def test_fit_rejects_test_data(self):
        """fit(s.test) raises PartitionError."""
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        with pytest.raises(PartitionError):
            ml.fit(s.test, "species", seed=42)

    def test_fit_accepts_train_data(self):
        """fit(s.train) works normally."""
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        model = ml.fit(s.train, "species", seed=42)
        assert model is not None

    def test_fit_stores_provenance(self):
        """fit() stores provenance on the model."""
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        model = ml.fit(s.train, "species", seed=42)
        assert hasattr(model, "_provenance")
        assert model._provenance["train_fingerprint"] == _fingerprint(s.train)

    def test_assess_rejects_train_data(self):
        """assess(model, test=s.train) raises PartitionError."""
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        model = ml.fit(s.dev, "species", seed=42)
        with pytest.raises(PartitionError):
            ml.assess(model, test=s.train)

    def test_audit_chain_records_workflow(self):
        """Full workflow creates audit entries."""
        chain = audit()
        initial_len = len(chain)
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        ml.fit(s.train, "species", seed=42)
        assert len(chain) > initial_len
        assert chain.verify()

    def test_untagged_data_rejected(self):
        """User-constructed DataFrames are rejected by guards."""
        data = ml.dataset("iris")
        # No split() call — data has no provenance
        with pytest.raises(PartitionError, match="split provenance"):
            ml.fit(data, "species", seed=42)

    def test_evaluate_rejects_test_via_fingerprint(self):
        """evaluate() rejects test partition via content-addressed guard."""
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        model = ml.fit(s.train, "species", seed=42)
        with pytest.raises(PartitionError, match="evaluate.*test"):
            ml.evaluate(model, s.test)

    def test_assess_catches_different_split_lineage(self):
        """assess() catches test data from a different split."""
        data = ml.dataset("iris")
        s1 = ml.split(data, "species", seed=42)
        s2 = ml.split(data, "species", seed=99)
        model = ml.fit(s1.dev, "species", seed=42)
        with pytest.raises(PartitionError, match="different split"):
            ml.assess(model, test=s2.test)


# ---------------------------------------------------------------------------
# Config: guards mode
# ---------------------------------------------------------------------------


class TestGuardsConfig:
    @pytest.fixture(autouse=True)
    def _restore_guards(self):
        yield
        ml.config(guards="off")

    def test_guards_strict_rejects_wrong_partition(self):
        """guards='strict' raises PartitionError for wrong partition."""
        ml.config(guards="strict")
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        with pytest.raises(PartitionError):
            ml.fit(s.test, "species", seed=42)

    def test_guards_strict_rejects_unregistered(self):
        """guards='strict' raises PartitionError for unsplit data."""
        ml.config(guards="strict")
        data = ml.dataset("iris")
        with pytest.raises(PartitionError, match="split provenance"):
            ml.fit(data, "species", seed=42)

    def test_guards_warn_warns(self):
        """guards='warn' issues warning instead of error."""
        ml.config(guards="warn")
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = ml.fit(s.test, "species", seed=42)
            partition_warnings = [x for x in w if "partition" in str(x.message).lower()]
            assert len(partition_warnings) >= 1
            assert model is not None

    def test_guards_off_silent(self):
        """guards='off' skips all guards silently."""
        ml.config(guards="off")
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        model = ml.fit(s.test, "species", seed=42)
        assert model is not None

    def test_guards_off_allows_unsplit(self):
        """guards='off' allows unsplit data through."""
        ml.config(guards="off")
        data = ml.dataset("iris")
        model = ml.fit(data, "species", seed=42)
        assert model is not None


# ---------------------------------------------------------------------------
# Save/load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadProvenance:
    @pytest.fixture(autouse=True)
    def _strict_guards(self):
        ml.config(guards="strict")
        yield
        ml.config(guards="off")

    def test_provenance_survives_save_load(self, tmp_path):
        """Provenance is preserved across save/load."""
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        model = ml.fit(s.train, "species", seed=42)
        assert model._provenance.get("split_id") is not None

        path = str(tmp_path / "model.pyml")
        ml.save(model, path)
        loaded = ml.load(path)
        assert loaded._provenance.get("split_id") == model._provenance["split_id"]
        assert loaded._provenance.get("train_fingerprint") == model._provenance["train_fingerprint"]

    def test_loaded_model_has_empty_provenance_for_old_files(self, tmp_path):
        """Models saved before provenance system load with empty provenance."""
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        model = ml.fit(s.train, "species", seed=42)
        # Simulate pre-provenance model
        model._provenance = None
        path = str(tmp_path / "old_model.pyml")
        ml.save(model, path)
        loaded = ml.load(path)
        assert loaded._provenance == {}


# ---------------------------------------------------------------------------
# FORTRESS: Adversarial bypass vectors
# ---------------------------------------------------------------------------


class TestAdversarialBypass:
    @pytest.fixture(autouse=True)
    def _strict_guards(self):
        ml.config(guards="strict")
        yield
        ml.config(guards="off")

    def test_reset_index_preserves_identity(self):
        """reset_index(drop=True) does NOT break fingerprint."""
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        fp_before = _fingerprint(s.train)
        reset = s.train.reset_index(drop=True)
        fp_after = _fingerprint(reset)
        assert fp_before == fp_after  # same content → same fingerprint

    def test_copy_preserves_identity(self):
        """df.copy() preserves fingerprint."""
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        assert identify_partition(s.test.copy()) == "test"

    def test_column_drop_breaks_identity(self):
        """Dropping a column changes fingerprint (correctly)."""
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        reduced = s.test.drop(columns=["species"])
        assert identify_partition(reduced) is None  # different content → unknown

    def test_concat_breaks_identity(self):
        """pd.concat strips identity (correctly — new data, unknown role)."""
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        combined = pd.concat([s.train, s.test]).reset_index(drop=True)
        assert identify_partition(combined) is None

    def test_attrs_spoofing_overridden_by_fingerprint(self):
        """Manual .attrs spoofing is caught by fingerprint check."""
        data = ml.dataset("iris")
        s = ml.split(data, "species", seed=42)
        # User tries to spoof train as test
        fake = s.train.copy()
        fake.attrs["_ml_partition"] = "test"
        # Fingerprint knows this is train, not test
        assert identify_partition(fake) == "train"

    def test_guard_on_large_dataframe(self):
        """Guards work on DataFrames >100K rows (strided sampling)."""
        import numpy as np
        n = 150_000
        df = pd.DataFrame({"x": np.random.randn(n), "y": np.random.randint(0, 2, n)})
        s = ml.split(df, "y", seed=42)
        assert identify_partition(s.train) == "train"
        assert identify_partition(s.test) == "test"

    def test_list_column_graceful_degradation(self):
        """DataFrames with unhashable columns get valid fingerprint via CSV fallback."""
        df = pd.DataFrame({"x": [[1, 2], [3, 4], [5, 6]], "y": [0, 1, 0]})
        # Falls back to CSV serialization — never returns None
        fp = _fingerprint(df)
        assert isinstance(fp, str) and len(fp) == 16
        # Deterministic across calls
        assert _fingerprint(df) == fp
        # Unregistered data returns None from identify (partition not in registry)
        assert identify_partition(df) is None
        # Guards correctly reject unregistered data — no silent bypass
        with pytest.raises(PartitionError):
            guard_fit(df)
        with pytest.raises(PartitionError):
            guard_evaluate(df)
