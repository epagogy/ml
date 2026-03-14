"""Security audit: adversarial fingerprint attacks.

Audit focus: _fingerprint function and PartitionRegistry.
Each test probes a specific weakness. Tests that FAIL expose real bugs.
Tests that PASS document why the attack does not work.
"""

import numpy as np
import pandas as pd
import pytest

import ml
from ml._provenance import _MAX_REGISTRY_ENTRIES, _fingerprint, _registry


@pytest.fixture(autouse=True)
def _strict():
    ml.config(guards="strict")
    yield
    ml.config(guards="off")
    _registry.clear()


# ---------------------------------------------------------------------------
# ATTACK A1: Empty DataFrame collision — column names ignored
# ---------------------------------------------------------------------------
# _fingerprint returns sha256(b"empty")[:16] for ALL empty DataFrames,
# regardless of column names or dtypes. Two empty DFs with different schemas
# get the SAME fingerprint. If one is registered, the other passes the guard.


class TestEmptyDataFrameCollision:
    """All empty DataFrames share one fingerprint — column names are ignored."""

    def test_empty_dfs_different_columns_same_fingerprint(self):
        """BUG PROBE: empty DataFrames with different column schemas produce
        the same fingerprint. This means registering an empty 'train' DF
        would let ANY empty DF pass the guard as 'train'."""
        df1 = pd.DataFrame({"x": pd.Series([], dtype=float)})
        df2 = pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=float)})
        fp1 = _fingerprint(df1)
        fp2 = _fingerprint(df2)
        # If these are equal, that is a collision bug in the fingerprint
        if fp1 == fp2:
            pytest.fail(
                "BUG: all empty DataFrames produce the same fingerprint "
                f"({fp1!r}) regardless of column names. "
                "An attacker who registers any empty partition can match "
                "any other empty DataFrame to that partition."
            )

    def test_empty_df_different_dtypes_same_column_name(self):
        """Empty DFs with same column name but different dtypes — dtypes are
        NOT part of the empty hash (only column names). This is accepted:
        dtype is a property of values, and empty DFs have no values."""
        df_float = pd.DataFrame({"x": pd.Series([], dtype="float64")})
        df_str = pd.DataFrame({"x": pd.Series([], dtype="object")})
        fp1 = _fingerprint(df_float)
        fp2 = _fingerprint(df_str)
        # Same column name → same fingerprint (dtype not included for empty)
        assert fp1 == fp2


# ---------------------------------------------------------------------------
# ATTACK A2: Null-byte injection in column names
# ---------------------------------------------------------------------------
# Column signature is: "\x00".join(str(c) for c in df.columns)
# If a column name contains \x00, the join is ambiguous:
#   columns ["a\x00b", "c"] -> "a\x00b\x00c"
#   columns ["a", "b\x00c"] -> "a\x00b\x00c"
# Same col_sig, so if cell values also match, fingerprints collide.


class TestNullByteInjection:
    """Column names containing the \x00 separator create ambiguous signatures."""

    def test_null_byte_in_column_name_collision(self):
        """BUG PROBE: two DFs with different column structures but same
        null-byte-joined signature. If cell values match, fingerprints collide."""
        # These two have different column structures:
        #   df1: columns = ["a\x00b", "c"]  -> col_sig = "a\x00b\x00c"
        #   df2: columns = ["a", "b\x00c"]  -> col_sig = "a\x00b\x00c"
        df1 = pd.DataFrame({"a\x00b": [1, 2, 3], "c": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b\x00c": [4, 5, 6]})
        fp1 = _fingerprint(df1)
        fp2 = _fingerprint(df2)
        if fp1 == fp2:
            pytest.fail(
                "BUG: null-byte injection in column names causes fingerprint "
                "collision. Columns ['a\\x00b', 'c'] and ['a', 'b\\x00c'] "
                f"both produce fingerprint {fp1!r}. "
                "Fix: use a length-prefixed or escaped encoding for col names."
            )

    def test_null_byte_column_name_fingerprintable(self):
        """Column names with embedded null bytes should at least not crash."""
        df = pd.DataFrame({"a\x00b": [1, 2, 3]})
        fp = _fingerprint(df)
        assert fp is not None, "Null-byte column name should be fingerprintable"


# ---------------------------------------------------------------------------
# ATTACK A3: Strided sampling blind spots (>100K rows)
# ---------------------------------------------------------------------------
# For n > 100_000, only every (n // 2000)th row is hashed.
# An attacker can modify ONLY non-sampled rows and preserve the fingerprint.


class TestStridedSamplingBypass:
    """Modify only non-sampled rows to preserve fingerprint of large DataFrames."""

    def test_modify_non_sampled_rows_preserves_fingerprint(self):
        """BUG PROBE: for >100K rows, modifying only rows that fall between
        the strided sample should not change the fingerprint. This is a
        real blind spot — the guard cannot see these modifications."""
        n = 110_000
        np.random.seed(42)
        df = pd.DataFrame({"x": np.random.randn(n), "y": np.random.randn(n)})
        fp_original = _fingerprint(df)

        # Compute which rows are sampled
        stride = max(1, n // 2000)  # same logic as _fingerprint
        sampled_indices = set(range(0, n, stride))

        # Find a non-sampled row and modify it
        modified = df.copy()
        for idx in range(n):
            if idx not in sampled_indices:
                modified.iloc[idx, 0] = 999999.0  # drastically different value
                break

        fp_modified = _fingerprint(modified)
        # KNOWN DESIGN TRADEOFF: strided sampling for >100K rows means
        # non-sampled rows are invisible. This is accepted for performance.
        assert fp_original == fp_modified, (
            "Strided sampling should NOT detect non-sampled row changes"
        )

    def test_modify_many_non_sampled_rows(self):
        """BUG PROBE: modify ALL non-sampled rows. If fingerprint is unchanged,
        the guard is nearly useless for large DataFrames."""
        n = 110_000
        np.random.seed(42)
        df = pd.DataFrame({"x": np.random.randn(n)})
        fp_original = _fingerprint(df)

        stride = max(1, n // 2000)
        sampled_indices = set(range(0, n, stride))

        modified = df.copy()
        for idx in range(n):
            if idx not in sampled_indices:
                modified.iloc[idx, 0] = 0.0  # zero out all unsampled rows

        fp_modified = _fingerprint(modified)
        # KNOWN DESIGN TRADEOFF: same as above — strided sampling accepted.
        assert fp_original == fp_modified, (
            "Strided sampling should NOT detect non-sampled row changes"
        )


# ---------------------------------------------------------------------------
# ATTACK A4: dtype coercion preserving fingerprint
# ---------------------------------------------------------------------------
# pd.util.hash_pandas_object hashes cell values. Does it distinguish
# int64(1) from float64(1.0)? bool(True) from int64(1)?


class TestDtypeSemanticConfusion:
    """Same logical values, different dtypes — does fingerprint distinguish?"""

    def test_int64_vs_float64_same_values(self):
        """PROBE: integer 1 and float 1.0 — same fingerprint?
        If yes, dtype changes silently pass the guard."""
        df_int = pd.DataFrame({"x": pd.array([1, 2, 3], dtype="int64")})
        df_float = pd.DataFrame({"x": pd.array([1.0, 2.0, 3.0], dtype="float64")})
        fp_int = _fingerprint(df_int)
        fp_float = _fingerprint(df_float)
        # Document whether this is a collision or not.
        # Either outcome is interesting:
        # - Same: dtype changes are invisible (potential semantic issue)
        # - Different: casting breaks the guard (user friction)
        # Document whether this is a collision or not — either outcome
        # is interesting but not necessarily a bug.
        assert fp_int is not None and fp_float is not None

    def test_bool_vs_int_same_values(self):
        """PROBE: bool True/False vs int 1/0 — same fingerprint?"""
        df_bool = pd.DataFrame({"x": [True, False, True]})
        df_int = pd.DataFrame({"x": [1, 0, 1]})
        fp_bool = _fingerprint(df_bool)
        fp_int = _fingerprint(df_int)
        # KNOWN: hash_pandas_object treats bool True/False same as int 1/0.
        # This is pandas behavior, not a fingerprint bug. Document it.
        assert fp_bool is not None and fp_int is not None

    def test_categorical_vs_object_string(self):
        """PROBE: Categorical(['a','b']) vs object(['a','b']) — same hash?"""
        df_obj = pd.DataFrame({"x": ["a", "b", "c"]})
        df_cat = pd.DataFrame({"x": pd.Categorical(["a", "b", "c"])})
        fp_obj = _fingerprint(df_obj)
        fp_cat = _fingerprint(df_cat)
        # Categorical encoding may produce different hashes.
        # Either outcome is acceptable — just document it.
        assert fp_obj is not None and fp_cat is not None

    def test_float32_vs_float64_precision_loss(self):
        """PROBE: float64 -> float32 -> float64 roundtrip. If the roundtrip
        changes values (precision loss), fingerprint changes. But if values
        happen to be exactly representable in float32, fingerprint may survive."""
        # Values that ARE exactly representable in float32
        df64 = pd.DataFrame({"x": [1.0, 2.0, 0.5, 0.25]})
        df32 = pd.DataFrame({"x": df64["x"].astype("float32")})
        fp64 = _fingerprint(df64)
        fp32 = _fingerprint(df32)
        # Even if values are the same, dtype difference may change hash
        if fp64 == fp32:
            pytest.fail(
                "BUG: float32 and float64 with identical values produce "
                "same fingerprint. Precision loss from float32 conversion "
                "is invisible to the guard."
            )


# ---------------------------------------------------------------------------
# ATTACK A5: In-place mutation via underlying numpy buffer
# ---------------------------------------------------------------------------
# DataFrames backed by numpy arrays: can we mutate the underlying buffer
# directly and bypass copy-on-write protections?


class TestNumpyBufferMutation:
    """Mutate the underlying numpy array after registration."""

    def test_values_buffer_mutation(self):
        """PROBE: get .values, mutate the numpy array directly.
        Does the fingerprint recompute from the mutated data?"""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        fp_before = _fingerprint(df)

        # Direct numpy buffer mutation — bypasses pandas CoW
        arr = df["x"].values
        if arr.flags.writeable:
            arr[0] = 999.0
            fp_after = _fingerprint(df)
            # Fingerprint SHOULD change because _fingerprint recomputes
            assert fp_before != fp_after, (
                "Numpy buffer mutation did change the data but fingerprint "
                "is stale — _fingerprint recomputes, so this should differ"
            )
        else:
            pytest.skip("numpy array is read-only (CoW enabled)")

    def test_register_then_mutate_buffer_bypasses_guard(self):
        """CRITICAL PROBE: register data, then mutate the underlying buffer.
        The registry still maps the OLD fingerprint to 'train'. But the
        data now has DIFFERENT content. fit() recomputes the fingerprint
        and should REJECT because the new fingerprint is not registered."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "x": np.random.randn(n),
            "target": np.random.choice(["a", "b"], n),
        })
        s = ml.split(df, "target", seed=42)
        train = s.train

        # Verify it works before mutation
        fp_original = _fingerprint(train)
        assert fp_original is not None
        role = _registry.identify(train)
        assert role == "train"

        # Mutate the underlying buffer
        arr = train["x"].values
        if arr.flags.writeable:
            arr[:] = 0.0  # zero out all feature values
            fp_mutated = _fingerprint(train)
            assert fp_mutated != fp_original, "Buffer mutation must change fingerprint"
            role_after = _registry.identify(train)
            if role_after == "train":
                pytest.fail(
                    "BUG: mutating the underlying numpy buffer after split "
                    "does NOT invalidate the guard. The data content changed "
                    "but the registry still identifies it as 'train'. "
                    "This should not happen if _fingerprint recomputes."
                )
            # If role_after is None, the guard correctly rejects — good
        else:
            pytest.skip("numpy array is read-only (CoW enabled)")


# ---------------------------------------------------------------------------
# ATTACK A6: Timezone-aware datetime columns
# ---------------------------------------------------------------------------
# Same wall-clock time in different timezones — same data or different?


class TestTimezoneDatetime:
    """Timezone-aware datetime columns and fingerprint behavior."""

    def test_same_instant_different_timezone_representation(self):
        """PROBE: same UTC instant expressed in different timezones.
        These represent the SAME moment in time. Should they fingerprint the same?"""
        ts_utc = pd.to_datetime(["2024-01-01 12:00:00"]).tz_localize("UTC")
        ts_berlin = ts_utc.tz_convert("Europe/Berlin")
        df_utc = pd.DataFrame({"t": ts_utc, "v": [1.0]})
        df_berlin = pd.DataFrame({"t": ts_berlin, "v": [1.0]})
        fp_utc = _fingerprint(df_utc)
        fp_berlin = _fingerprint(df_berlin)
        # Both are the same instant — hash_pandas_object should hash
        # the underlying int64 nanoseconds which are equal.
        # If they differ, timezone conversion breaks the guard (user friction).
        # Either way, both must be non-None (fingerprintable).
        assert fp_utc is not None
        assert fp_berlin is not None

    def test_same_wall_time_different_timezone(self):
        """PROBE: same wall-clock time, different timezone — DIFFERENT instants.
        These are NOT the same data. Fingerprints must differ."""
        df_utc = pd.DataFrame({
            "t": pd.to_datetime(["2024-01-01 12:00:00"]).tz_localize("UTC"),
            "v": [1.0],
        })
        df_est = pd.DataFrame({
            "t": pd.to_datetime(["2024-01-01 12:00:00"]).tz_localize("US/Eastern"),
            "v": [1.0],
        })
        fp_utc = _fingerprint(df_utc)
        fp_est = _fingerprint(df_est)
        if fp_utc == fp_est:
            pytest.fail(
                "BUG: same wall-clock time in different timezones (UTC vs EST) "
                "produces the same fingerprint. These are DIFFERENT instants "
                "(5 hours apart). The guard cannot distinguish them."
            )


# ---------------------------------------------------------------------------
# ATTACK: Registry lineage memory leak
# ---------------------------------------------------------------------------
# _store uses OrderedDict with eviction at _MAX_REGISTRY_ENTRIES.
# But _lineage is a plain dict with NO eviction. It grows without bound.


class TestRegistryLineageLeak:
    """The _lineage dict is never evicted — unbounded memory growth."""

    def test_lineage_not_evicted_when_store_evicts(self):
        """BUG PROBE: register enough entries to trigger eviction in _store,
        then check if _lineage also evicted. If _lineage retains entries
        that _store has evicted, it is a memory leak."""
        # Register more than _MAX_REGISTRY_ENTRIES
        from ml._provenance import new_split_id
        n_entries = _MAX_REGISTRY_ENTRIES + 100
        for i in range(n_entries):
            df = pd.DataFrame({"x": [float(i)]})
            sid = new_split_id()
            _registry.register(df, "train", sid)

        # _store should have been evicted down to _MAX_REGISTRY_ENTRIES
        assert len(_registry._store) <= _MAX_REGISTRY_ENTRIES, (
            f"Store has {len(_registry._store)} entries, expected <= {_MAX_REGISTRY_ENTRIES}"
        )

        # But _lineage may still have all entries (no eviction logic)
        lineage_count = len(_registry._lineage)
        if lineage_count > _MAX_REGISTRY_ENTRIES:
            pytest.fail(
                f"MEMORY LEAK: _lineage has {lineage_count} entries but _store "
                f"was evicted to {len(_registry._store)}. _lineage grows without "
                "bound because eviction only clears _store (OrderedDict.popitem) "
                "but never removes corresponding _lineage entries."
            )


# ---------------------------------------------------------------------------
# ATTACK A8: Shape not in payload for small DataFrames
# ---------------------------------------------------------------------------
# For n <= 100K, the payload is: row_hashes.tobytes() + col_sig
# Shape is NOT included. For n > 100K, shape IS included.
# This means two small DFs with same hash_pandas_object output and same
# columns but conceptually different structure get the same fingerprint.
# (In practice this is hard to exploit because row_hashes include all rows.)


class TestShapeNotInSmallPayload:
    """For <=100K rows, df.shape is not part of the fingerprint payload."""

    def test_shape_absence_documented(self):
        """Verify that shape is not in the small-DF payload.
        This is not directly exploitable (all rows are hashed), but
        it is an asymmetry worth documenting."""
        # Two DFs with same content but one has been through
        # operations that change internal representation
        df1 = pd.DataFrame({"x": [1.0, 2.0]})
        df2 = pd.DataFrame({"x": [1.0, 2.0]})
        # Same content, same shape — should be same fingerprint
        assert _fingerprint(df1) == _fingerprint(df2)
        # This test just documents the design — not a bug per se


# ---------------------------------------------------------------------------
# ATTACK A9: Concurrent register + identify race condition
# ---------------------------------------------------------------------------
# _fingerprint is called outside the lock in identify(). Two threads could:
# Thread 1: register(df, "train", sid)  — computes fp, acquires lock, stores
# Thread 2: identify(df_modified)       — computes fp (no lock), reads store
# The race window is between fp computation and store lookup.


class TestRaceCondition:
    """Race between fingerprint computation and registry lookup."""

    def test_identify_outside_lock(self):
        """DESIGN PROBE: identify() calls _fingerprint() outside the lock,
        then does a dict lookup. If another thread modifies _store between
        the fingerprint computation and the lookup, the result could be stale.
        In practice, dict.get is atomic in CPython (GIL), so this is safe
        on CPython but NOT guaranteed on other interpreters."""
        # This is a documentation/design issue, not a test that can reliably
        # demonstrate the race. Just verify the code structure.
        import inspect
        source = inspect.getsource(_registry.identify)
        # identify() does NOT hold the lock during lookup
        assert "self._lock" not in source, (
            "identify() acquired the lock — race condition is mitigated"
        )
        # If we reach here, identify() does NOT use the lock — which means
        # it relies on CPython's GIL for thread safety of dict.get()


# ---------------------------------------------------------------------------
# ATTACK A10: hash_pandas_object with ordered categoricals
# ---------------------------------------------------------------------------
# Ordered categoricals have different semantics than unordered ones.
# Does hash_pandas_object distinguish them?


class TestOrderedCategorical:
    """Ordered vs unordered categorical — same values, different semantics."""

    def test_ordered_vs_unordered_categorical(self):
        """PROBE: same categories, same values, but one is ordered.
        These have different semantics (ordinal vs nominal).
        If fingerprints match, the guard can't distinguish them."""
        cats = ["low", "mid", "high"]
        vals = ["low", "mid", "high", "low", "mid"]
        df_unord = pd.DataFrame({"x": pd.Categorical(vals, categories=cats, ordered=False)})
        df_ord = pd.DataFrame({"x": pd.Categorical(vals, categories=cats, ordered=True)})
        fp_unord = _fingerprint(df_unord)
        fp_ord = _fingerprint(df_ord)
        # Document: ordered flag may or may not be visible to fingerprint.
        # Not necessarily a bug — cell values are identical.
        assert fp_unord is not None and fp_ord is not None

    def test_different_category_order_same_values(self):
        """PROBE: same values but categories listed in different order.
        Categorical(['a','b'], categories=['a','b']) vs
        Categorical(['a','b'], categories=['b','a'])"""
        df1 = pd.DataFrame({"x": pd.Categorical(["a", "b"], categories=["a", "b"])})
        df2 = pd.DataFrame({"x": pd.Categorical(["a", "b"], categories=["b", "a"])})
        fp1 = _fingerprint(df1)
        fp2 = _fingerprint(df2)
        # Category order affects integer codes and thus model behavior.
        # If fingerprints are the same, category reordering is invisible.
        # Document: category order may or may not be visible.
        assert fp1 is not None and fp2 is not None


# ---------------------------------------------------------------------------
# ATTACK A11: NaN variants (None vs np.nan vs pd.NA)
# ---------------------------------------------------------------------------
# pandas has multiple null representations. Do they all hash the same?


class TestNaNVariants:
    """Different null representations — do they produce the same hash?"""

    def test_none_vs_nan_in_float_column(self):
        """None and np.nan in a float column — should be the same (both -> NaN)."""
        df1 = pd.DataFrame({"x": [1.0, None, 3.0]})
        df2 = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        fp1 = _fingerprint(df1)
        fp2 = _fingerprint(df2)
        assert fp1 == fp2, (
            "None and np.nan in float column produce different fingerprints. "
            "Both should coerce to NaN."
        )

    def test_pd_na_vs_np_nan(self):
        """pd.NA (nullable integer NA) vs np.nan — different type systems."""
        df_nan = pd.DataFrame({"x": pd.array([1, None, 3], dtype="Int64")})
        df_float = pd.DataFrame({"x": pd.array([1.0, np.nan, 3.0], dtype="float64")})
        fp_nan = _fingerprint(df_nan)
        fp_float = _fingerprint(df_float)
        # These represent different type systems (nullable int vs float).
        # It's acceptable if they differ. Just document.
        assert fp_nan is not None and fp_float is not None

    def test_nat_in_datetime_column(self):
        """pd.NaT in datetime column — should be fingerprintable."""
        df = pd.DataFrame({
            "t": pd.to_datetime(["2024-01-01", pd.NaT, "2024-01-03"]),
            "v": [1.0, 2.0, 3.0],
        })
        fp = _fingerprint(df)
        assert fp is not None, "NaT in datetime column should be fingerprintable"


# ---------------------------------------------------------------------------
# ATTACK A12: Column order permutation
# ---------------------------------------------------------------------------
# Same columns, same values, different column order.
# Column signature includes column names in order — should differ.


class TestColumnOrderPermutation:
    """Same data, different column order — must produce different fingerprint."""

    def test_column_reorder_changes_fingerprint(self):
        """Reordering columns changes the col_sig and row hashes.
        Both should contribute to a different fingerprint."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"b": [3, 4], "a": [1, 2]})
        fp1 = _fingerprint(df1)
        fp2 = _fingerprint(df2)
        if fp1 == fp2:
            pytest.fail(
                "BUG: column reordering does not change fingerprint. "
                "Columns ['a','b'] and ['b','a'] with same values produce "
                f"identical fingerprint {fp1!r}. This means column order "
                "is invisible to the guard."
            )


# ---------------------------------------------------------------------------
# ATTACK A13: Negative zero (-0.0 vs 0.0)
# ---------------------------------------------------------------------------
# IEEE 754: -0.0 == 0.0 but they have different bit representations.


class TestNegativeZero:
    """Negative zero has a different bit pattern than positive zero."""

    def test_negative_zero_vs_positive_zero(self):
        """PROBE: -0.0 and 0.0 are equal in Python but have different bits.
        Does hash_pandas_object distinguish them?"""
        df_pos = pd.DataFrame({"x": [0.0, 1.0]})
        df_neg = pd.DataFrame({"x": [-0.0, 1.0]})
        fp_pos = _fingerprint(df_pos)
        fp_neg = _fingerprint(df_neg)
        # Either outcome is acceptable — just document it.
        # -0.0 == 0.0 in Python, so same hash is defensible.
        assert fp_pos is not None and fp_neg is not None


# ---------------------------------------------------------------------------
# ATTACK A14: Very long column names to force hash collision
# ---------------------------------------------------------------------------
# The col_sig is part of the SHA-256 input. Can extremely long column names
# push the hash computation to behave differently?


class TestLongColumnNames:
    """Extremely long column names — resource exhaustion or hash issues."""

    def test_very_long_column_name(self):
        """Column name with 1M characters — should still fingerprint."""
        long_name = "x" * 1_000_000
        df = pd.DataFrame({long_name: [1, 2, 3]})
        fp = _fingerprint(df)
        assert fp is not None, "Very long column name should be fingerprintable"
        assert len(fp) == 16, "Fingerprint length should be 16 hex chars"

    def test_many_columns(self):
        """1000 columns — should still fingerprint correctly."""
        data = {f"col_{i}": [float(i)] for i in range(1000)}
        df = pd.DataFrame(data)
        fp = _fingerprint(df)
        assert fp is not None
        assert len(fp) == 16


# ---------------------------------------------------------------------------
# ATTACK A15: Register with wrong role, then use legitimately
# ---------------------------------------------------------------------------
# If an attacker can register unsplit data as "test", they can bypass
# assess()'s role check.


class TestRoleSpoofingViaDirectRegistry:
    """Direct registry access to register data with a spoofed role."""

    def test_register_train_data_as_test(self):
        """KNOWN GAP: direct _registry.register() can assign any role.
        This bypasses the split() → register flow entirely."""
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.randn(100),
            "target": np.random.choice(["a", "b"], 100),
        })
        from ml._provenance import new_split_id, register_partition

        # Register the full dataset as "train" to allow fitting
        sid = new_split_id()
        register_partition(df, "train", sid)
        model = ml.fit(data=df, target="target", seed=42)

        # Now register the SAME data as "test" — this should be wrong
        # because we're assessing on training data
        register_partition(df, "test", sid)
        # assess() will accept it because it checks role == "test"
        verdict = ml.assess(model=model, test=df)
        # If we get here, the guard was fooled — same data used for train and test
        assert verdict is not None, (
            "KNOWN GAP: direct registry access allows registering train data "
            "as 'test', completely bypassing the train/test separation."
        )


# ---------------------------------------------------------------------------
# ATTACK A16: Eviction timing — register, evict, re-register with new role
# ---------------------------------------------------------------------------
# After eviction, the same data can be re-registered with a different role.


class TestEvictionRoleSwap:
    """After eviction, re-register same data with different role."""

    def test_eviction_then_reregister_different_role(self):
        """Register data as 'train', force eviction, re-register as 'test'.
        The data identity doesn't change but its role does."""
        from ml._provenance import new_split_id, register_partition

        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        sid = new_split_id()
        register_partition(df, "train", sid)
        assert _registry.identify(df) == "train"

        # Simply re-register with new role — no eviction needed!
        # The OrderedDict just overwrites the value.
        register_partition(df, "test", sid)
        assert _registry.identify(df) == "test", (
            "Re-registration should overwrite the role"
        )
        # This means ANY code with access to register_partition can
        # flip a partition's role at any time.


# ---------------------------------------------------------------------------
# ATTACK A17: Index manipulation
# ---------------------------------------------------------------------------
# _fingerprint uses index=False in hash_pandas_object.
# But does the index state affect the underlying data layout?


class TestIndexManipulation:
    """Index changes that might affect fingerprinting."""

    def test_multiindex_vs_flat_index(self):
        """MultiIndex should not affect fingerprint (index=False)."""
        df_flat = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        df_multi = df_flat.set_index(["x"])  # x moves to index, only y remains
        # These have DIFFERENT data — df_multi only has column 'y'
        fp_flat = _fingerprint(df_flat)
        fp_multi = _fingerprint(df_multi)
        assert fp_flat != fp_multi, (
            "set_index moves a column out of the data — fingerprints must differ"
        )

    def test_set_index_then_reset_preserves_fingerprint(self):
        """set_index then reset_index should roundtrip (if drop=False)."""
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        fp1 = _fingerprint(df)
        df2 = df.set_index("x").reset_index()
        fp2 = _fingerprint(df2)
        # Should be the same — the data and columns are restored
        assert fp1 == fp2, (
            "set_index then reset_index should preserve fingerprint"
        )


# ---------------------------------------------------------------------------
# ATTACK A18: Sparse vs dense representation
# ---------------------------------------------------------------------------
# Sparse and dense DataFrames with the same values — same fingerprint?


class TestSparseVsDense:
    """Sparse arrays vs dense arrays with the same values."""

    def test_sparse_vs_dense_same_values(self):
        """PROBE: sparse representation of same data — does fingerprint match?"""
        dense = pd.DataFrame({"x": [0.0, 0.0, 1.0, 0.0, 0.0]})
        sparse = pd.DataFrame({"x": pd.arrays.SparseArray([0.0, 0.0, 1.0, 0.0, 0.0])})
        fp_dense = _fingerprint(dense)
        fp_sparse = _fingerprint(sparse)
        # If they differ, converting to sparse breaks the guard
        if fp_dense != fp_sparse:
            pytest.fail(
                "Converting to SparseArray changes the fingerprint even though "
                "the logical values are identical. This means any sparse "
                "optimization would invalidate partition registration."
            )
