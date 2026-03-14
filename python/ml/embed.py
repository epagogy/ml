"""embed() — text to features (internal module, not part of public API).

Explicit, stateful text embedding. Returns an Embedder that stores
the fitted tokenizer/vocabulary. Use ml.tokenize() for the public API.

Direct import:
    >>> from ml.embed import embed, Embedder
    >>> emb = embed(texts, method="tfidf")
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Embedder:
    """Fitted text embedder.

    Stores the fitted tokenizer/vectorizer and provides transform() for new texts.

    Attributes
    ----------
    vectors : pd.DataFrame
        Embedded feature vectors from training texts (n_samples x vocab_size)
    method : str
        Embedding method used ("tfidf", "sbert", "bilstm")
    vocab_size : int
        Vocabulary size (number of features)
    _vectorizer : Any
        Internal: fitted sklearn vectorizer or embedding model
    """

    vectors: pd.DataFrame
    method: str
    vocab_size: int
    _vectorizer: object

    def transform(self, texts: pd.Series | list) -> pd.DataFrame:
        """Transform new texts using the fitted embedder.

        Parameters
        ----------
        texts : pd.Series or list
            New texts to embed (uses stored vocabulary/tokenizer)

        Returns
        -------
        pd.DataFrame
            Embedded feature vectors (n_samples x vocab_size)

        Examples
        --------
        >>> import ml
        >>> import pandas as pd
        >>> texts = pd.Series(["good product", "bad service", "great value"])
        >>> emb = ml.embed(texts, method="tfidf")
        >>> new_texts = pd.Series(["excellent quality"])
        >>> new_vectors = emb.transform(new_texts)
        >>> new_vectors.shape
        (1, vocab_size)
        """
        from ._types import DataError

        # Convert list to Series if needed
        if isinstance(texts, list):
            texts = pd.Series(texts)

        # Validate
        if not isinstance(texts, pd.Series):
            raise DataError(f"Expected pd.Series or list, got {type(texts).__name__}")

        # Transform using stored vectorizer
        if self.method == "tfidf":
            embeddings = self._vectorizer.transform(texts)
            # Convert to DataFrame (our TfidfVectorizer returns dense ndarray)
            df = pd.DataFrame(
                embeddings, columns=[f"tfidf_{i}" for i in range(self.vocab_size)]
            )
            return df
        else:
            # Other methods (sbert, bilstm) would go here
            raise NotImplementedError(
                f"method='{self.method}' not yet implemented. Use method='tfidf'."
            )


def embed(
    texts: pd.Series | list,
    *,
    method: str = "tfidf",
    max_features: int = 100,
) -> Embedder:
    """Embed texts into numeric features.

    Explicit, stateful embedding. Returns an Embedder that stores the fitted
    tokenizer/vocabulary for use at prediction time.

    Parameters
    ----------
    texts : pd.Series or list
        Texts to embed
    method : str, default="tfidf"
        Embedding method. Options:
        - "tfidf": TF-IDF (sklearn, fast, default)
        - "sbert": Sentence-BERT (semantic, optional dependency)
        - "bilstm": Learned BiLSTM (optional dependency)
    max_features : int, default=100
        Maximum vocabulary size (number of features to generate)

    Returns
    -------
    Embedder
        Fitted embedder with .vectors (DataFrame), .transform(new_texts),
        .method, .vocab_size attributes.

    Raises
    ------
    DataError
        If texts is not pd.Series or list, or if empty
    ConfigError
        If method is not recognized

    Examples
    --------
    >>> import ml
    >>> import pandas as pd
    >>> texts = pd.Series(["good product", "bad service", "great value"])
    >>> emb = ml.embed(texts, method="tfidf")
    >>> emb.vectors.shape
    (3, 100)
    >>> emb.method
    'tfidf'
    >>> emb.vocab_size
    100

    At prediction time:
    >>> new_texts = pd.Series(["excellent quality"])
    >>> new_vectors = emb.transform(new_texts)
    >>> new_vectors.shape
    (1, 100)

    Save and load:
    >>> ml.save(emb, "embedder.ml")
    >>> emb_loaded = ml.load("embedder.ml")
    """
    from ._types import ConfigError, DataError

    # Convert list to Series if needed
    if isinstance(texts, list):
        texts = pd.Series(texts)

    # Validation
    if not isinstance(texts, pd.Series):
        raise DataError(f"Expected pd.Series or list, got {type(texts).__name__}")

    if len(texts) == 0:
        raise DataError("Cannot embed empty texts (0 samples)")

    # Supported methods (only tfidf in v1.0)
    if method not in ("tfidf",):
        raise ConfigError(
            f"method='{method}' not recognized. Choose from: ['tfidf']"
        )

    # TF-IDF implementation (Gate 1)
    if method == "tfidf":
        from ._transforms import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=max_features, lowercase=True)
        embeddings = vectorizer.fit_transform(texts)

        # Convert to DataFrame (our TfidfVectorizer returns dense ndarray)
        actual_vocab_size = embeddings.shape[1]
        vectors_df = pd.DataFrame(
            embeddings,
            columns=[f"tfidf_{i}" for i in range(actual_vocab_size)],
        )

        return Embedder(
            vectors=vectors_df,
            method=method,
            vocab_size=actual_vocab_size,
            _vectorizer=vectorizer,
        )

    else:
        # Should not reach here after validator — future methods deferred
        raise NotImplementedError(f"method='{method}' not yet implemented.")
