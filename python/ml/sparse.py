"""sparse() — sparse TF-IDF features for high-dimensional text pipelines.

Returns a SparseFrame wrapping scipy CSR + feature names. Flows through
fit/predict without densification for sparse-capable algorithms (logistic,
linear, elastic_net, naive_bayes, svm). Tree-based algorithms auto-densify.

Usage:
    >>> tok = ml.tokenize(s.train, columns=["review"], max_features=5000)
    >>> sf = ml.sparse(tok, s.train)
    >>> model = ml.fit(sf, "target", seed=42)
    >>> preds = ml.predict(model, ml.sparse(tok, s.valid))
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    import scipy.sparse as sp

    HAS_SCIPY = True
except ImportError:
    sp = None  # type: ignore[assignment]
    HAS_SCIPY = False

try:
    from ml_py import fit_tfidf_vocabulary as _rust_fit_vocab
    from ml_py import fit_transform_tfidf_sparse as _rust_fit_transform
    from ml_py import transform_tfidf_sparse as _rust_transform
    from ml_py import transform_tfidf_sparse_bytes as _rust_transform_bytes

    HAS_RUST_TFIDF = True
except ImportError:
    _rust_transform = None  # type: ignore[assignment]
    _rust_transform_bytes = None  # type: ignore[assignment]
    _rust_fit_vocab = None  # type: ignore[assignment]
    _rust_fit_transform = None  # type: ignore[assignment]
    HAS_RUST_TFIDF = False


# Algorithms that can consume scipy CSR directly via sklearn API
# naive_bayes excluded: GaussianNB requires dense (variance computation)
SPARSE_ALGORITHMS = frozenset({
    "logistic", "linear", "elastic_net", "svm",
})

# Tree-based algorithms must densify — they scan all values per feature
DENSE_ONLY_ALGORITHMS = frozenset({
    "random_forest", "extra_trees", "decision_tree",
    "gradient_boosting", "histgradient", "adaboost",
    "xgboost", "lightgbm", "catboost", "knn",
})


@dataclass
class SparseFrame:
    """Sparse feature matrix with metadata.

    Wraps a scipy CSR matrix with feature names, numeric passthrough columns,
    and target column. Acts as a drop-in for DataFrame in ml.fit/predict.
    """

    data: Any  # scipy.sparse.csr_matrix
    feature_names: list[str]
    numeric: pd.DataFrame
    target: str | None = None
    index: pd.Index = field(default_factory=lambda: pd.RangeIndex(0))
    attrs: dict = field(default_factory=dict)  # DataFrame.attrs passthrough (provenance)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.data.shape[0], self.data.shape[1] + self.numeric.shape[1])

    @property
    def columns(self) -> list[str]:
        return list(self.numeric.columns) + self.feature_names

    def to_dense(self) -> pd.DataFrame:
        """Convert to dense DataFrame. Falls back for tree-based algorithms."""
        n_num = self.numeric.shape[1]
        n_sparse = self.data.shape[1]

        # Fast path: allocate one contiguous array, fill both sides
        out = np.empty((len(self), n_num + n_sparse), dtype=np.float64)

        if n_num > 0:
            num_vals = self.numeric.values
            if num_vals.dtype != np.float64:
                # Object columns (target might be int/str) — copy col-by-col
                for j in range(n_num):
                    try:
                        out[:, j] = num_vals[:, j].astype(np.float64)
                    except (ValueError, TypeError):
                        # Non-numeric column (e.g. target with strings) — defer to pandas
                        return self._to_dense_pandas()
            else:
                out[:, :n_num] = num_vals

        # Sparse → dense into right half of pre-allocated array
        # toarray() returns C-contiguous float64 — single memcpy
        out[:, n_num:] = self.data.toarray()

        col_names = list(self.numeric.columns) + self.feature_names
        result = pd.DataFrame(out, columns=col_names, index=self.index)
        if self.attrs:
            result.attrs.update(self.attrs)
        return result

    def _to_dense_pandas(self) -> pd.DataFrame:
        """Fallback dense conversion via pandas concat (handles mixed dtypes)."""
        sparse_df = pd.DataFrame(
            self.data.toarray(),
            columns=self.feature_names,
            index=self.index,
        )
        result = pd.concat(
            [self.numeric.reset_index(drop=True), sparse_df], axis=1
        )
        if self.attrs:
            result.attrs.update(self.attrs)
        return result

    def to_X_sparse(self, algorithm: str | None = None) -> Any:
        """Return feature matrix for engine consumption.

        For sparse-capable algorithms: returns scipy CSR (with dense columns
        hstacked as sparse if any numeric features exist).
        For tree-based: returns dense ndarray.
        """
        # Exclude target from numeric features
        if self.target:
            num_cols = [c for c in self.numeric.columns if c != self.target]
        else:
            num_cols = list(self.numeric.columns)

        use_sparse = algorithm is None or algorithm in SPARSE_ALGORITHMS

        if use_sparse:
            if num_cols:
                # hstack dense numeric as CSR + sparse TF-IDF
                num_arr = self.numeric[num_cols].values.astype(np.float64)
                num_csr = sp.csr_matrix(num_arr)
                return sp.hstack([num_csr, self.data], format="csr")
            return self.data
        else:
            # Dense path for trees — single allocation
            n_num = len(num_cols)
            n_sparse = self.data.shape[1]
            out = np.empty((len(self), n_num + n_sparse), dtype=np.float64)
            if n_num > 0:
                out[:, :n_num] = self.numeric[num_cols].values.astype(np.float64)
            out[:, n_num:] = self.data.toarray()
            return out

    def feature_names_for_engine(self) -> list[str]:
        """Feature column names (excluding target)."""
        if self.target:
            num_feats = [c for c in self.numeric.columns if c != self.target]
        else:
            num_feats = list(self.numeric.columns)
        return num_feats + self.feature_names

    def __len__(self) -> int:
        return self.data.shape[0]

    def __repr__(self) -> str:
        nnz = self.data.nnz
        density = nnz / max(1, self.data.shape[0] * self.data.shape[1])
        n_num = self.numeric.shape[1]
        target_str = f", target='{self.target}'" if self.target else ""
        return (
            f"SparseFrame({self.data.shape[0]} rows, "
            f"{len(self.feature_names)} sparse + {n_num} dense features, "
            f"nnz={nnz}, density={density:.4f}{target_str})"
        )


def sparse(
    tokenizer: Any,
    data: pd.DataFrame,
    *,
    target: str | None = None,
) -> SparseFrame:
    """Build sparse TF-IDF features from a fitted tokenizer.

    Takes a Tokenizer (from ml.tokenize) and a DataFrame, returns a SparseFrame
    with sparse TF-IDF columns and dense numeric passthrough.

    Parameters
    ----------
    tokenizer : Tokenizer
        Fitted tokenizer from ml.tokenize().
    data : pd.DataFrame
        DataFrame with text columns matching the tokenizer + any other columns.
    target : str, optional
        Target column name. If present, excluded from features and stored
        in the SparseFrame for downstream use by fit().

    Returns
    -------
    SparseFrame
        Sparse feature matrix ready for ml.fit() or ml.predict().

    Examples
    --------
    >>> tok = ml.tokenize(s.train, columns=["review"], max_features=5000)
    >>> sf = ml.sparse(tok, s.train)
    >>> model = ml.fit(sf, "target", seed=42)
    >>> preds = ml.predict(model, ml.sparse(tok, s.valid))
    """
    from ._types import ConfigError, DataError
    from .tokenize import Tokenizer

    if not HAS_SCIPY:
        raise ConfigError(
            "sparse() requires scipy. Install with: pip install scipy"
        )

    if not isinstance(tokenizer, Tokenizer):
        raise DataError(
            f"sparse() expects a Tokenizer from ml.tokenize(), "
            f"got {type(tokenizer).__name__}"
        )

    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"sparse() expects DataFrame, got {type(data).__name__}"
        )

    # Check text columns exist
    missing = [c for c in tokenizer.columns if c not in data.columns]
    if missing:
        raise DataError(
            f"Text columns missing from data: {missing}. "
            f"Expected columns: {tokenizer.columns}"
        )

    # Build sparse TF-IDF matrices per text column
    sparse_blocks = []
    all_feat_names: list[str] = []

    for col in tokenizer.columns:
        vec = tokenizer._vectorizers[col]
        texts = data[col].fillna("").astype(str)
        mat = _transform_sparse(vec, texts)
        n_features = mat.shape[1]
        all_feat_names.extend(f"{col}_tfidf_{i}" for i in range(n_features))
        sparse_blocks.append(mat)

    # Horizontal stack sparse blocks
    if len(sparse_blocks) == 1:
        sparse_matrix = sparse_blocks[0]
    else:
        sparse_matrix = sp.hstack(sparse_blocks, format="csr")

    # Separate numeric columns (passthrough)
    text_cols = set(tokenizer.columns)
    if target:
        exclude = text_cols | {target}
    else:
        exclude = text_cols

    numeric_cols = [c for c in data.columns if c not in exclude]
    numeric_df = data[numeric_cols].copy() if numeric_cols else pd.DataFrame(
        index=data.index
    )

    # Include target in numeric for fit() to extract
    if target and target in data.columns:
        numeric_df[target] = data[target]

    return SparseFrame(
        data=sparse_matrix,
        feature_names=all_feat_names,
        numeric=numeric_df,
        target=target,
        index=data.index,
        attrs=dict(data.attrs) if data.attrs else {},
    )


# ---------- tokenizer ----------
# Compiled once at module import. C-level regex, GIL released during match.
_WORD_RE = re.compile(r"(?u)\b\w\w+\b")


def _transform_sparse(vectorizer: Any, texts) -> Any:
    """Transform texts to sparse CSR — maximum throughput.

    Uses Rust backend (ml-py) when available: parallel tokenization via rayon,
    fused IDF weighting + L2 normalization, zero-copy numpy output.
    Falls back to Python single-pass CSR construction otherwise.
    """
    docs = list(texts)
    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_
    n_features = len(vocab)
    lowercase = vectorizer.lowercase

    # Rust fast path: parallel tokenize + fused IDF*TF + L2 norm
    if HAS_RUST_TFIDF:
        return _transform_sparse_rust(docs, vocab, idf, n_features, lowercase)

    return _transform_sparse_full(docs, vocab, idf, n_features, lowercase)


def _transform_sparse_rust(docs, vocab, idf, n_features, lowercase):
    """Rust-accelerated sparse TF-IDF via ml-py.

    Uses zero-copy bytes path: concatenates docs into one contiguous buffer
    with byte offsets. Rust tokenizes directly from &[u8] — no per-string
    allocation across the Python/Rust boundary.
    """
    from .tokenize import _encode_docs

    text_bytes, offsets = _encode_docs(docs)

    indices, indptr, values, n_cols = _rust_transform_bytes(
        text_bytes, offsets, vocab, idf, lowercase
    )
    return sp.csr_matrix(
        (values, indices, indptr), shape=(len(docs), n_cols)
    )


def _transform_sparse_full(docs, vocab, idf, n_features, lowercase):
    """Single-pass CSR construction with vectorized L2 normalization."""
    n_docs = len(docs)
    _findall = _WORD_RE.findall
    _get = vocab.get

    # Geometric growth — start at avg 10 nnz/doc
    cap = max(n_docs * 10, 256)
    indices_buf = np.empty(cap, dtype=np.int32)
    values_buf = np.empty(cap, dtype=np.float64)
    indptr = np.empty(n_docs + 1, dtype=np.int32)
    indptr[0] = 0
    pos = 0

    idf_arr = idf  # numpy array

    for i in range(n_docs):
        text = docs[i]
        if not isinstance(text, str):
            text = str(text)
        if lowercase:
            text = text.lower()

        tokens = _findall(text)

        # TF accumulation
        tf: dict[int, int] = {}
        for tok in tokens:
            idx = _get(tok)
            if idx is not None:
                if idx in tf:
                    tf[idx] += 1
                else:
                    tf[idx] = 1

        n = len(tf)
        if n > 0:
            # Ensure capacity (geometric doubling)
            need = pos + n
            if need > cap:
                while cap < need:
                    cap *= 2
                indices_buf = _grow(indices_buf, cap)
                values_buf = _grow_f64(values_buf, cap)

            # Write sorted (CSR canonical) with TF*IDF values
            sorted_items = sorted(tf.items())
            for j, (idx, count) in enumerate(sorted_items):
                indices_buf[pos + j] = idx
                values_buf[pos + j] = count * idf_arr[idx]
            pos += n

        indptr[i + 1] = pos

    # Trim
    indices = indices_buf[:pos]
    values = values_buf[:pos]

    # Build CSR with in-place L2 row-normalization.
    # Normalize values array directly (preserves sorted indices, no matrix multiply).
    for i in range(n_docs):
        start = indptr[i]
        end = indptr[i + 1]
        if start == end:
            continue
        row = values[start:end]
        norm = np.sqrt(np.dot(row, row))
        if norm > 0.0:
            row /= norm

    mat = sp.csr_matrix((values, indices, indptr), shape=(n_docs, n_features))

    return mat


def _grow(arr: np.ndarray, new_cap: int) -> np.ndarray:
    """Grow int32 array to new_cap, preserving existing data."""
    new = np.empty(new_cap, dtype=np.int32)
    new[:len(arr)] = arr
    return new


def _grow_f64(arr: np.ndarray, new_cap: int) -> np.ndarray:
    """Grow float64 array to new_cap, preserving existing data."""
    new = np.empty(new_cap, dtype=np.float64)
    new[:len(arr)] = arr
    return new
