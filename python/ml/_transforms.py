"""Pure-numpy preprocessing transforms. No sklearn dependency.

Drop-in replacements for sklearn's LabelEncoder, OneHotEncoder,
StandardScaler, SimpleImputer, MinMaxScaler, RobustScaler.

All classes follow the sklearn fit/transform API but do not inherit from
any sklearn base class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class LabelEncoder:
    """Encode target labels as integers 0..n_classes-1."""

    def __init__(self):
        self.classes_: np.ndarray | None = None
        self._mapping: dict | None = None
        self._inverse: dict | None = None

    def fit(self, y):
        y = np.asarray(y)
        self.classes_ = np.sort(np.unique(y))
        self._mapping = {c: i for i, c in enumerate(self.classes_)}
        self._inverse = {i: c for c, i in self._mapping.items()}
        return self

    def transform(self, y):
        y = np.asarray(y)
        try:
            return np.array([self._mapping[v] for v in y], dtype=np.int64)
        except KeyError as exc:
            unseen = sorted(set(y) - set(self.classes_), key=str)
            raise ValueError(
                f"y contains previously unseen labels: {unseen}"
            ) from exc

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        y = np.asarray(y)
        return np.array([self._inverse[int(v)] for v in y])


class OneHotEncoder:
    """One-hot encode categorical columns.

    Parameters:
        handle_unknown: "error" or "ignore". If "ignore", unseen categories
            produce all-zeros rows.
        sparse_output: Ignored (always dense). Kept for API compat.
        drop: Ignored. Kept for API compat.
    """

    def __init__(self, *, handle_unknown="error", sparse_output=False, drop=None):
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        self.drop = drop
        self.categories_: list[np.ndarray] | None = None
        self._col_names: list[str] | None = None

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            self.categories_ = []
            self._col_names = list(X.columns)
            for col in X.columns:
                cats = sorted(X[col].dropna().unique(), key=str)
                self.categories_.append(np.array(cats))
        else:
            X = np.asarray(X)
            self.categories_ = []
            self._col_names = None
            for j in range(X.shape[1]):
                cats = sorted(set(X[:, j]) - {None}, key=str)
                self.categories_.append(np.array(cats))
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
            arrays = []
            for i, col in enumerate(cols):
                cats = self.categories_[i]
                cat_to_idx = {v: j for j, v in enumerate(cats)}
                col_data = X[col].values
                block = np.zeros((len(col_data), len(cats)), dtype=np.float64)
                indices = np.array([cat_to_idx.get(v, -1) for v in col_data])
                # Check for unknown categories (not NaN, not known)
                if self.handle_unknown == "error":
                    for row_idx, val in enumerate(col_data):
                        if indices[row_idx] < 0 and val is not None and not (isinstance(val, float) and np.isnan(val)):
                            raise ValueError(f"Unknown category '{val}' in column '{col}'")
                valid = indices >= 0
                if valid.any():
                    block[np.where(valid)[0], indices[valid]] = 1.0
                arrays.append(block)
            return np.hstack(arrays) if arrays else np.empty((len(X), 0))
        else:
            raise NotImplementedError("Only DataFrame input supported")

    def get_feature_names_out(self, input_features=None):
        names = []
        cols = input_features if input_features is not None else self._col_names
        if cols is None:
            cols = [f"x{i}" for i in range(len(self.categories_))]
        for i, col in enumerate(cols):
            for cat in self.categories_[i]:
                names.append(f"{col}_{cat}")
        return names


class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.var_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.n_samples_seen_: int = 0

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        self.mean_ = np.nanmean(X, axis=0)
        self.var_ = np.nanvar(X, axis=0)
        self.scale_ = np.sqrt(self.var_)
        # Avoid division by zero: if std=0, set scale=1 (pass through)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            vals = X.values.astype(np.float64)
            result = (vals - self.mean_) / self.scale_
            return result
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X * self.scale_ + self.mean_


class SimpleImputer:
    """Impute missing values using column statistics.

    Parameters:
        strategy: "median" (default), "mean", or "most_frequent".
    """

    def __init__(self, strategy="median"):
        self.strategy = strategy
        self.statistics_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        self.n_features_in_ = X_arr.shape[1]

        stats = np.zeros(X_arr.shape[1])
        for j in range(X_arr.shape[1]):
            col = X_arr[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                stats[j] = 0.0
            elif self.strategy == "median":
                stats[j] = np.median(valid)
            elif self.strategy == "mean":
                stats[j] = np.mean(valid)
            elif self.strategy == "most_frequent":
                vals, counts = np.unique(valid, return_counts=True)
                stats[j] = vals[np.argmax(counts)]
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        self.statistics_ = stats
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            result = X.values.astype(np.float64).copy()
        else:
            result = np.asarray(X, dtype=np.float64).copy()
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        for j in range(result.shape[1]):
            mask = np.isnan(result[:, j])
            result[mask, j] = self.statistics_[j]
        return result

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Scale features to [0, 1] range."""

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.data_range_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_in_ = X.shape[1]
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        # Avoid division by zero: if range=0, set to 1 (pass through)
        safe_range = self.data_range_.copy()
        safe_range[safe_range == 0] = 1.0
        f_min, f_max = self.feature_range
        self.scale_ = (f_max - f_min) / safe_range
        self.min_ = f_min - self.data_min_ * self.scale_
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            vals = X.values.astype(np.float64)
        else:
            vals = np.asarray(X, dtype=np.float64)
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        return vals * self.scale_ + self.min_

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (X - self.min_) / self.scale_


class RobustScaler:
    """Scale features using median and IQR (robust to outliers)."""

    def __init__(self):
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_in_ = X.shape[1]
        self.center_ = np.nanmedian(X, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        q25 = np.nanpercentile(X, 25, axis=0)
        self.scale_ = q75 - q25
        # Avoid division by zero: if IQR=0, set to 1 (pass through)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            vals = X.values.astype(np.float64)
        else:
            vals = np.asarray(X, dtype=np.float64)
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        return (vals - self.center_) / self.scale_

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X * self.scale_ + self.center_


class TfidfVectorizer:
    """TF-IDF vectorizer. Dense output, no scipy/sklearn dependency.

    IDF formula (smooth_idf=True, matching sklearn default):
        idf(t) = log((1 + n) / (1 + df(t))) + 1

    TF: raw term frequency within each document.
    Rows are L2-normalized after TF*IDF weighting.

    Parameters:
        max_features: Keep top N features by corpus-wide term frequency.
        lowercase: Convert text to lowercase before tokenizing.
    """

    def __init__(self, max_features: int | None = None, lowercase: bool = True):
        self.max_features = max_features
        self.lowercase = lowercase
        self.vocabulary_: dict[str, int] | None = None
        self.idf_: np.ndarray | None = None

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words (2+ alphanumeric chars, matching sklearn)."""
        import re

        if self.lowercase:
            text = text.lower()
        return re.findall(r"(?u)\b\w\w+\b", str(text))

    def fit(self, texts):
        """Learn vocabulary and IDF weights from texts."""
        docs = list(texts)
        n = len(docs)

        # Tokenize and count document frequency
        df_counts: dict[str, int] = {}
        tf_totals: dict[str, int] = {}
        for doc in docs:
            tokens = self._tokenize(str(doc))
            seen = set()
            for tok in tokens:
                tf_totals[tok] = tf_totals.get(tok, 0) + 1
                if tok not in seen:
                    df_counts[tok] = df_counts.get(tok, 0) + 1
                    seen.add(tok)

        # Select top features by corpus frequency, then alphabetical index order
        if self.max_features is not None:
            top = sorted(tf_totals.keys(), key=lambda t: (-tf_totals[t], t))
            top = sorted(top[:self.max_features])  # alphabetical for indices
        else:
            top = sorted(tf_totals.keys())

        self.vocabulary_ = {term: i for i, term in enumerate(top)}

        # IDF: log((1+n)/(1+df)) + 1  (smooth_idf=True)
        idf = np.zeros(len(self.vocabulary_))
        for term, idx in self.vocabulary_.items():
            df = df_counts.get(term, 0)
            idf[idx] = np.log((1 + n) / (1 + df)) + 1
        self.idf_ = idf
        return self

    def transform(self, texts) -> np.ndarray:
        """Transform texts to TF-IDF dense matrix."""
        docs = list(texts)
        n_features = len(self.vocabulary_)
        result = np.zeros((len(docs), n_features), dtype=np.float64)

        for i, doc in enumerate(docs):
            tokens = self._tokenize(str(doc))
            # Raw term frequency
            for tok in tokens:
                idx = self.vocabulary_.get(tok)
                if idx is not None:
                    result[i, idx] += 1.0
            # TF * IDF
            result[i] *= self.idf_
            # L2 normalize
            norm = np.sqrt(np.sum(result[i] ** 2))
            if norm > 0:
                result[i] /= norm

        return result

    def fit_transform(self, texts) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)
