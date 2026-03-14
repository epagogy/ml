"""Computational guards for ML partition integrity.

Content-addressed partition identity + cross-verb provenance + audit chain.

Layer 1: Partition Fingerprinting
    Every split() registers partitions by content hash. Every verb checks identity.
    Survives column operations. Silent on unknown data. PartitionError on wrong role.

Layer 2: Cross-Verb Provenance
    fit() stores training fingerprint + split lineage. assess() verifies zero
    row overlap and same-split lineage.

Layer 3: Audit Chain
    Every verb call produces a hash-chained entry. Verifiable after the fact.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Layer 1: Content-Addressed Partition Identity
# ---------------------------------------------------------------------------

_MAX_REGISTRY_ENTRIES = 10_000
_SAMPLE_ROWS = 2000


def _fingerprint(df: pd.DataFrame) -> str:
    """Deterministic content-addressed fingerprint of a DataFrame.

    Uses pd.util.hash_pandas_object on cell values (index=False for
    index-agnosticism), plus column names as part of the hash. This
    ensures: (1) same content = same fingerprint regardless of index,
    (2) column renames produce different fingerprints, (3) strided
    sampling for large datasets (>100K rows).

    Returns a hex string (first 16 chars of SHA-256), or None if
    unhashable (e.g., list-valued columns) or not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        return None
    n = len(df)
    if n == 0:
        col_sig = b"".join(
            len(s).to_bytes(4, "big") + s
            for c in df.columns
            for s in [str(c).encode()]
        )
        return hashlib.sha256(b"empty" + col_sig).hexdigest()[:16]

    try:
        # Column names are part of identity — renaming = different data
        # Length-prefix each name to avoid injection (e.g. "a\x00b"+"c" vs "a"+"b\x00c")
        col_sig = b"".join(
            len(s).to_bytes(4, "big") + s
            for c in df.columns
            for s in [str(c).encode()]
        )
        if n > 100_000:
            stride = max(1, n // _SAMPLE_ROWS)
            sample = df.iloc[::stride]
            row_hashes = pd.util.hash_pandas_object(sample, index=False)
            payload = row_hashes.values.tobytes() + col_sig + f"|{df.shape}".encode()
        else:
            row_hashes = pd.util.hash_pandas_object(df, index=False)
            payload = row_hashes.values.tobytes() + col_sig

        return hashlib.sha256(payload).hexdigest()[:16]
    except TypeError:
        # Unhashable column types (e.g., list-valued columns) — can't fingerprint
        return None


class PartitionRegistry:
    """Session-scoped registry of known partitions.

    Maps content fingerprints to partition roles and split lineage.
    Thread-safe via a lock on mutations.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._store: OrderedDict[str, str] = OrderedDict()      # fp → role
        self._lineage: dict[str, str] = {}                       # fp → split_id
        self._splits: dict[str, dict] = {}                       # split_id → metadata
        self._assessed: set[str] = set()                         # fp → assessed test partitions

    def register(self, df: pd.DataFrame, role: str, split_id: str) -> str | None:
        """Register a partition. Returns its fingerprint, or None if unhashable."""
        fp = _fingerprint(df)
        if fp is None:
            return None
        with self._lock:
            self._store[fp] = role
            self._lineage[fp] = split_id
            # Evict oldest entries if registry is too large
            while len(self._store) > _MAX_REGISTRY_ENTRIES:
                evicted_fp, _ = self._store.popitem(last=False)
                self._lineage.pop(evicted_fp, None)
        return fp

    def identify(self, df: pd.DataFrame) -> str | None:
        """Identify the partition role of a DataFrame, or None if unknown."""
        fp = _fingerprint(df)
        if fp is None:
            return None
        return self._store.get(fp)

    def get_split_id(self, df: pd.DataFrame) -> str | None:
        """Get the split_id that produced this partition, or None."""
        fp = _fingerprint(df)
        if fp is None:
            return None
        return self._lineage.get(fp)

    def register_split(self, split_id: str, **metadata):
        """Record metadata for a split operation."""
        with self._lock:
            self._splits[split_id] = {
                "timestamp": time.time(),
                **metadata,
            }

    def mark_assessed(self, df: pd.DataFrame) -> None:
        """Mark a test partition's fingerprint as assessed (spent)."""
        fp = _fingerprint(df)
        if fp is not None:
            with self._lock:
                self._assessed.add(fp)

    def is_assessed(self, df: pd.DataFrame) -> bool:
        """Check if a test partition has already been assessed by any model."""
        fp = _fingerprint(df)
        if fp is None:
            return False
        return fp in self._assessed

    def clear(self):
        """Clear the registry (e.g., between experiments)."""
        with self._lock:
            self._store.clear()
            self._lineage.clear()
            self._splits.clear()
            self._assessed.clear()

    def __len__(self):
        return len(self._store)


# Module-level singleton
_registry = PartitionRegistry()


def new_split_id() -> str:
    """Generate a unique split identifier."""
    return uuid.uuid4().hex[:12]


def register_partition(df: pd.DataFrame, role: str, split_id: str) -> str:
    """Register a partition in the global registry. Returns fingerprint."""
    return _registry.register(df, role, split_id)


def identify_partition(df: pd.DataFrame) -> str | None:
    """Identify the partition role of a DataFrame, or None if unknown."""
    return _registry.identify(df)


def get_split_id(df: pd.DataFrame) -> str | None:
    """Get the split_id that produced this partition."""
    return _registry.get_split_id(df)


# ---------------------------------------------------------------------------
# Layer 2: Cross-Verb Provenance
# ---------------------------------------------------------------------------


def build_provenance(data: pd.DataFrame) -> dict:
    """Build provenance metadata from training data for storage in a Model.

    Row overlap detection is intentionally NOT included: two observations with
    identical feature values are NOT the same observation (observational identity
    ≠ value identity). Content-based overlap detection produces false positives
    on any dataset with categorical features or small value ranges. Split lineage
    (Check 1) catches the real danger: test data from a different split.
    """
    fp = _fingerprint(data)
    split_id = _registry._lineage.get(fp)
    return {
        "train_fingerprint": fp,
        "split_id": split_id,
        "fit_timestamp": time.time(),
    }


def check_provenance(model_provenance: dict, test: pd.DataFrame) -> None:
    """Check cross-verb provenance. Raises PartitionError on violation.

    Checks:
    1. Split lineage — test data from same split as training data
    2. Row overlap — no shared rows between train and test
    """
    if not model_provenance:
        return

    # Check 1: Split lineage
    test_split = get_split_id(test)
    train_split = model_provenance.get("split_id")
    if test_split and train_split and test_split != train_split:
        _guard_action(
            "assess() test data comes from a different split than fit() training data. "
            "This enables test-set shopping. Use test data from the same split: "
            "s = ml.split(...); model = ml.fit(s.train, ...); ml.assess(model, test=s.test)"
        )

    # Note: Row overlap detection intentionally omitted.
    # Content-based overlap (hash each row) is unsound: independent observations
    # with identical feature values (common in categorical/small data) trigger
    # false positives. Observational identity ≠ value identity.
    # Split lineage (Check 1) catches the real threat.


# ---------------------------------------------------------------------------
# Layer 3: Audit Chain
# ---------------------------------------------------------------------------


@dataclass
class AuditEntry:
    """Single entry in the audit chain."""
    verb: str
    data_fingerprint: str
    partition_role: str | None
    model_hash: str | None
    split_id: str | None
    timestamp: float
    prev_hash: str

    @property
    def hash(self) -> str:
        payload = (
            f"{self.verb}:{self.data_fingerprint}:{self.partition_role}:"
            f"{self.model_hash}:{self.split_id}:{self.timestamp}:{self.prev_hash}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class AuditChain:
    """Append-only, hash-chained audit trail of ML verb invocations.

    Each entry includes a SHA-256 hash of the previous entry, creating
    a tamper-evident chain (same integrity model as git).

    NOTE: This supports but does NOT satisfy regulatory requirements
    (21 CFR Part 11, EU AI Act). It provides no persistence, no
    cryptographic signatures, and no access control.
    """

    def __init__(self):
        self._entries: list[AuditEntry] = []
        self._lock = threading.Lock()

    def append(
        self,
        verb: str,
        data_fingerprint: str,
        partition_role: str | None = None,
        model_hash: str | None = None,
        split_id: str | None = None,
    ) -> AuditEntry:
        """Append a new entry to the chain."""
        with self._lock:
            prev_hash = self._entries[-1].hash if self._entries else "genesis"
            entry = AuditEntry(
                verb=verb,
                data_fingerprint=data_fingerprint,
                partition_role=partition_role,
                model_hash=model_hash,
                split_id=split_id,
                timestamp=time.time(),
                prev_hash=prev_hash,
            )
            self._entries.append(entry)
        return entry

    def verify(self) -> bool:
        """Verify the integrity of the chain.

        Returns True if every entry's prev_hash matches the previous
        entry's computed hash. Returns True for empty chains.
        """
        if len(self._entries) <= 1:
            return True
        for i in range(1, len(self._entries)):
            if self._entries[i].prev_hash != self._entries[i - 1].hash:
                return False
        return True

    def extract(self, model_hash: str) -> list[AuditEntry]:
        """Extract entries related to a specific model."""
        return [e for e in self._entries if e.model_hash == model_hash]

    def to_json(self) -> str:
        """Export the chain as JSON for review."""
        import json
        entries = []
        for e in self._entries:
            entries.append({
                "verb": e.verb,
                "data_fingerprint": e.data_fingerprint,
                "partition_role": e.partition_role,
                "model_hash": e.model_hash,
                "split_id": e.split_id,
                "timestamp": e.timestamp,
                "prev_hash": e.prev_hash,
                "hash": e.hash,
            })
        return json.dumps(entries, indent=2)

    def __len__(self):
        return len(self._entries)

    def __repr__(self):
        n = len(self._entries)
        ok = self.verify()
        status = "intact" if ok else "BROKEN"
        return f"AuditChain({n} entries, {status})"


# Module-level singleton
_chain = AuditChain()


def audit_log(
    verb: str,
    data_fingerprint: str,
    partition_role: str | None = None,
    model_hash: str | None = None,
    split_id: str | None = None,
) -> AuditEntry:
    """Append an entry to the global audit chain."""
    return _chain.append(verb, data_fingerprint, partition_role, model_hash, split_id)


def audit() -> AuditChain:
    """Return the current session's audit chain."""
    return _chain


# ---------------------------------------------------------------------------
# Guard helpers (used by fit, evaluate, assess, validate, calibrate)
# ---------------------------------------------------------------------------


def _guard_action(message: str) -> None:
    """Raise or warn based on ml.config(guards=...) setting.

    - "strict" (default): raise PartitionError
    - "warn": issue UserWarning
    - "off": silently pass
    """
    import warnings

    from ._config import _CONFIG
    from ._types import PartitionError

    mode = _CONFIG.get("guards", "strict")
    if mode == "off":
        return
    if mode == "warn":
        warnings.warn(message, UserWarning, stacklevel=3)
        return
    raise PartitionError(message)


def _identify_with_reason(data: pd.DataFrame) -> tuple[str | None, bool]:
    """Identify partition with fingerprintability info.

    Returns (role, fingerprintable). role is None if unregistered or
    unfingerprintable. fingerprintable is False for unhashable data.
    """
    fp = _fingerprint(data)
    if fp is None:
        return None, False
    role = _registry._store.get(fp)
    return role, True


def guard_fit(data: pd.DataFrame) -> None:
    """Layer 1 guard for fit(): require split provenance."""
    role, fingerprintable = _identify_with_reason(data)
    if not fingerprintable:
        return  # can't judge — pass silently
    if role is None:
        _guard_action(
            "fit() received data without split provenance. "
            "Split your data first: s = ml.split(df, target, seed=42), "
            "then ml.fit(s.train, target, seed=42). "
            "To disable: ml.config(guards='off')"
        )
    elif role not in ("train", "valid", "dev"):
        _guard_action(
            f"fit() received data identified as '{role}' partition. "
            f"fit() accepts train, valid, or dev data. "
            f"Use: ml.fit(s.train, ...) or ml.fit(s.dev, ...)"
        )


def guard_evaluate(data: pd.DataFrame) -> None:
    """Layer 1 guard for evaluate(): require split provenance, reject test data."""
    role, fingerprintable = _identify_with_reason(data)
    if not fingerprintable:
        return  # can't judge — pass silently
    if role is None:
        _guard_action(
            "evaluate() received data without split provenance. "
            "Split your data first: s = ml.split(df, target, seed=42), "
            "then ml.evaluate(model, s.valid). "
            "To disable: ml.config(guards='off')"
        )
    elif role == "test":
        _guard_action(
            "evaluate() received data identified as 'test' partition. "
            "evaluate() is the practice exam — use validation data. "
            "For the final exam, use ml.assess(model, test=s.test)."
        )


def guard_assess(data: pd.DataFrame) -> None:
    """Layer 1 guard for assess(): require split provenance, reject non-test data,
    reject already-assessed test holdouts (per-holdout enforcement)."""
    role, fingerprintable = _identify_with_reason(data)
    if not fingerprintable:
        return  # can't judge — pass silently
    if role is None:
        _guard_action(
            "assess() received data without split provenance. "
            "Split your data first: s = ml.split(df, target, seed=42), "
            "then ml.assess(model, test=s.test). "
            "To disable: ml.config(guards='off')"
        )
    elif role != "test":
        _guard_action(
            f"assess() received data identified as '{role}' partition. "
            "assess() requires test data (s.test). "
            "For validation iterations, use ml.evaluate(model, s.valid)."
        )
    elif _registry.is_assessed(data):
        _guard_action(
            "assess() called on a test holdout that has already been assessed. "
            "The test set gets one assessment per holdout, regardless of which model. "
            "Use ml.evaluate(model, s.valid) for model comparison. "
            "To get a fresh holdout: s = ml.split(df, target, seed=NEW_SEED). "
            "To override: ml.config(guards='off')"
        )


def guard_validate(data: pd.DataFrame) -> None:
    """Layer 1 guard for validate(): require split provenance, reject non-test data."""
    role, fingerprintable = _identify_with_reason(data)
    if not fingerprintable:
        return  # can't judge — pass silently
    if role is None:
        _guard_action(
            "validate() received data without split provenance. "
            "Split your data first: s = ml.split(df, target, seed=42), "
            "then ml.validate(model, test=s.test). "
            "To disable: ml.config(guards='off')"
        )
    elif role != "test":
        _guard_action(
            f"validate() received data identified as '{role}' partition. "
            "validate() requires test data (s.test). "
            "For validation iterations, use ml.evaluate(model, s.valid)."
        )
