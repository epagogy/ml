"""tokenize() — explicit text preprocessing for ML pipelines.

Standalone stateful transformer. Fits TF-IDF on training text columns,
replaces them with numeric features at transform time. No model attachment.

Usage:
    >>> prep = ml.tokenize(s.train, columns=["review"], max_features=100)
    >>> model = ml.fit(prep.transform(s.train), "label", seed=42)
    >>> preds = ml.predict(model, prep.transform(s.valid))
    >>> ml.save(prep, "prep.pyml")
    >>> prep2 = ml.load("prep.pyml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    from ml_py import fit_tfidf_vocabulary as _rust_fit_vocab
    from ml_py import fit_transform_tfidf_sparse as _rust_fit_transform
    from ml_py import fit_transform_tfidf_strings as _rust_fit_transform_strings

    _HAS_RUST_FIT = True
except ImportError:
    _rust_fit_vocab = None  # type: ignore[assignment]
    _rust_fit_transform = None  # type: ignore[assignment]
    _rust_fit_transform_strings = None  # type: ignore[assignment]
    _HAS_RUST_FIT = False


def _encode_docs(texts) -> tuple[bytes, np.ndarray]:
    """Encode text series to contiguous byte buffer + offsets for Rust.

    Optimized: single pass with cumsum for offsets, avoids Python loop.
    """
    docs = list(texts)
    # Fast encode — list comp is faster than generator for CPython
    encoded = [d.encode("utf-8") if isinstance(d, str) else d for d in docs]
    # Vectorized offsets via numpy cumsum (avoids Python for-loop)
    lengths = np.fromiter((len(e) for e in encoded), dtype=np.int64, count=len(encoded))
    offsets = np.empty(len(encoded) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    return b"".join(encoded), offsets


class _RustVectorizer:
    """Lightweight duck-type for sklearn/native TfidfVectorizer.

    Stores vocab dict + IDF array from Rust fit. Provides the same
    `.vocabulary_`, `.idf_`, `.lowercase` interface that sparse.py expects.
    No transform method — sparse.py handles transform via Rust backend.
    """

    __slots__ = ("vocabulary_", "idf_", "lowercase")

    def __init__(self, vocabulary: dict, idf: np.ndarray, lowercase: bool = True):
        self.vocabulary_ = vocabulary
        self.idf_ = idf
        self.lowercase = lowercase

    def transform(self, texts) -> np.ndarray:
        """Dense transform fallback for Tokenizer.transform() path."""
        from .sparse import _transform_sparse

        import scipy.sparse as sp

        mat = _transform_sparse(self, texts)
        return mat.toarray()


@dataclass
class Tokenizer:
    """Fitted text preprocessor.

    Stores one TF-IDF vectorizer per text column. At transform time,
    replaces each text column with numeric ``{col}_tfidf_{i}`` features.

    Attributes
    ----------
    columns : list[str]
        Text column names that were fitted.
    max_features : int
        Maximum TF-IDF features per column.
    _vectorizers : dict[str, Any]
        Internal: fitted TfidfVectorizer per column.

    Examples
    --------
    >>> prep = ml.tokenize(s.train, columns=["review"], max_features=50)
    >>> df_ready = prep.transform(s.train)
    >>> df_ready.columns  # review replaced by review_tfidf_0 .. review_tfidf_49
    """

    columns: list[str]
    max_features: int
    _vectorizers: dict[str, Any] = field(repr=False)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Replace text columns with TF-IDF numeric features.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the fitted text columns (may also contain
            other numeric columns, which are passed through unchanged).

        Returns
        -------
        pd.DataFrame
            New DataFrame with text columns replaced by ``{col}_tfidf_{i}``
            features. Column order: non-text columns first, then tfidf blocks
            in the order columns were specified.

        Raises
        ------
        DataError
            If any fitted column is missing from data.

        Examples
        --------
        >>> df_ready = prep.transform(s.valid)
        >>> df_ready.shape[1]  # original columns - text + tfidf features
        """

        from ._types import DataError
        if not isinstance(data, pd.DataFrame):
            raise DataError(
                f"transform() expects DataFrame, got {type(data).__name__}."
            )

        missing = [c for c in self.columns if c not in data.columns]
        if missing:
            raise DataError(
                f"Text columns missing from data: {missing}. "
                f"Expected columns: {self.columns}"
            )

        # Start with non-text columns
        other_cols = [c for c in data.columns if c not in self.columns]
        out = data[other_cols].copy()

        # Accumulate dense matrices from our TfidfVectorizer
        dense_blocks = []
        all_feat_names: list[str] = []
        for col in self.columns:
            vec = self._vectorizers[col]
            texts = data[col].fillna("").astype(str)
            mat = vec.transform(texts)  # dense ndarray
            n_features = mat.shape[1]
            all_feat_names.extend(f"{col}_tfidf_{i}" for i in range(n_features))
            dense_blocks.append(mat)

        if dense_blocks:
            import numpy as np
            combined = np.hstack(dense_blocks)
            for i, name in enumerate(all_feat_names):
                out[name] = combined[:, i]

        return out

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Alias for transform() — allows using Tokenizer as a callable."""
        return self.transform(data)

    def __repr__(self) -> str:
        col_summary = ", ".join(self.columns)
        # Show actual vocabulary size (may be < max_features for small corpora)
        actual = [len(self._vectorizers[col].vocabulary_) for col in self.columns]
        feat_str = str(actual[0]) if len(actual) == 1 else str(actual)
        return (
            f"Tokenizer(columns=[{col_summary}], "
            f"features={feat_str}, max_features={self.max_features})"
        )


def tokenize(
    data: pd.DataFrame,
    *,
    columns: list[str],
    max_features: int = 100,
) -> Tokenizer:
    """Fit a text tokenizer on training data.

    Fits one TF-IDF vectorizer per text column. Returns a ``Tokenizer``
    that can transform any DataFrame (train, valid, new data) by replacing
    the text columns with numeric TF-IDF features.

    The tokenizer is fitted on ``data`` only — no leakage into validation
    or test sets.

    Parameters
    ----------
    data : pd.DataFrame
        Training DataFrame containing the text columns.
    columns : list[str]
        Names of text columns to tokenize.
    max_features : int, default=100
        Maximum TF-IDF vocabulary size per column (number of features generated).

    Returns
    -------
    Tokenizer
        Fitted text preprocessor with ``.transform(df)`` method.

    Raises
    ------
    DataError
        If data is not a DataFrame, columns list is empty, or any column
        is missing from data or has a non-string dtype with no string values.
    ConfigError
        If max_features < 1.

    Examples
    --------
    >>> import ml
    >>> prep = ml.tokenize(s.train, columns=["review"], max_features=50)
    >>> model = ml.fit(prep.transform(s.train), "label", seed=42)
    >>> preds = ml.predict(model, prep.transform(s.valid))

    Multiple text columns:
    >>> prep = ml.tokenize(s.train, columns=["title", "body"], max_features=100)
    >>> df_ready = prep.transform(s.train)  # title_tfidf_* + body_tfidf_* added

    Save and reload:
    >>> ml.save(prep, "prep.pyml")
    >>> prep2 = ml.load("prep.pyml")
    >>> preds = ml.predict(model, prep2.transform(new_df))
    """
    from ._compat import to_pandas
    from ._transforms import TfidfVectorizer
    from ._types import ConfigError, DataError

    data = to_pandas(data)

    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"tokenize() expects DataFrame, got {type(data).__name__}."
        )

    if not columns:
        raise DataError(
            "tokenize() requires at least one column. "
            "Pass columns=['col1', 'col2', ...]."
        )

    if max_features < 1:
        raise ConfigError(
            f"max_features must be >= 1, got {max_features}."
        )

    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise DataError(
            f"Columns not found in data: {missing}. "
            f"Available columns: {list(data.columns)}"
        )

    vectorizers: dict[str, Any] = {}
    for col in columns:
        texts = data[col].fillna("").astype(str)

        if _HAS_RUST_FIT:
            # Rust fast path: parallel vocab fit via rayon
            text_bytes, offsets = _encode_docs(texts)
            vocab_dict, idf_arr = _rust_fit_vocab(
                text_bytes, offsets, max_features, True
            )
            vec = _RustVectorizer(vocab_dict, idf_arr, lowercase=True)
        else:
            vec = TfidfVectorizer(max_features=max_features, lowercase=True)
            vec.fit(texts)

        vectorizers[col] = vec

    return Tokenizer(
        columns=list(columns),
        max_features=max_features,
        _vectorizers=vectorizers,
    )


def tokenize_sparse(
    data: pd.DataFrame,
    *,
    columns: list[str],
    max_features: int = 100,
    target: str | None = None,
) -> tuple[Any, Any]:
    """Fused tokenize + sparse in one call.

    When Rust backend is available, fits vocabulary and transforms to sparse
    TF-IDF in a single Rust call — avoids encoding text twice and eliminates
    one Python↔Rust boundary crossing.

    Returns (Tokenizer, SparseFrame) — equivalent to:
        tok = ml.tokenize(data, columns=columns, max_features=max_features)
        sf = ml.sparse(tok, data, target=target)

    Parameters
    ----------
    data : pd.DataFrame
        Training DataFrame with text columns.
    columns : list[str]
        Text column names.
    max_features : int
        Max TF-IDF vocabulary size per column.
    target : str, optional
        Target column name.

    Returns
    -------
    tuple[Tokenizer, SparseFrame]
    """
    from .sparse import SparseFrame

    try:
        import scipy.sparse as sp_mod
    except ImportError:
        # Fall back to separate calls
        tok = tokenize(data, columns=columns, max_features=max_features)
        from .sparse import sparse as sparse_fn
        sf = sparse_fn(tok, data, target=target)
        return tok, sf

    if not _HAS_RUST_FIT or len(columns) != 1:
        # Multi-column or no Rust — fall back to separate calls
        tok = tokenize(data, columns=columns, max_features=max_features)
        from .sparse import sparse as sparse_fn
        sf = sparse_fn(tok, data, target=target)
        return tok, sf

    # --- Fused Rust path: list[str] directly, no Python encode ---
    col = columns[0]
    texts = data[col].fillna("").astype(str)
    docs = list(texts)

    indices, indptr, values, n_cols, vocab_dict, idf_arr = _rust_fit_transform_strings(
        docs, max_features, True
    )

    vec = _RustVectorizer(vocab_dict, idf_arr, lowercase=True)
    tok = Tokenizer(
        columns=list(columns),
        max_features=max_features,
        _vectorizers={col: vec},
    )

    sparse_matrix = sp_mod.csr_matrix(
        (values, indices, indptr), shape=(len(data), n_cols)
    )
    n_features = sparse_matrix.shape[1]
    feat_names = [f"{col}_tfidf_{i}" for i in range(n_features)]

    # Separate numeric columns
    text_cols = set(columns)
    exclude = text_cols | ({target} if target else set())
    numeric_cols = [c for c in data.columns if c not in exclude]
    numeric_df = data[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=data.index)
    if target and target in data.columns:
        numeric_df[target] = data[target]

    sf = SparseFrame(
        data=sparse_matrix,
        feature_names=feat_names,
        numeric=numeric_df,
        target=target,
        index=data.index,
        attrs=dict(data.attrs) if data.attrs else {},
    )

    return tok, sf
