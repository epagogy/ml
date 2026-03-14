"""Native K-Nearest Neighbors — zero sklearn dependency.

Brute-force: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b, np.argpartition for O(n) partial sort.
For n_train > 50K consider sklearn (KD-tree/ball-tree).
"""

from __future__ import annotations

import numpy as np


def _pairwise_sq_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of A and B.

    Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b to avoid explicit loops.
    Returns shape (n_A, n_B).
    """
    A_sq = np.sum(A ** 2, axis=1, keepdims=True)  # (n_A, 1)
    B_sq = np.sum(B ** 2, axis=1)                  # (n_B,)
    return A_sq + B_sq - 2.0 * A @ B.T


class _KNNClassifier:
    """K-Nearest Neighbors classifier (sklearn-compatible protocol).

    Brute-force, O(n*m) memory. Deterministic (no randomness).
    Tie-breaking: lowest class index wins.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors. Default 5.
    n_jobs : int
        Accepted for API compatibility, ignored (no parallelism).
    """

    def __init__(self, *, n_neighbors: int = 5, n_jobs: int = 1):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.classes_: np.ndarray | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def get_params(self, deep: bool = True) -> dict:
        return {"n_neighbors": self.n_neighbors, "n_jobs": self.n_jobs}

    def set_params(self, **params) -> _KNNClassifier:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> _KNNClassifier:
        self._X_train = np.asarray(X, dtype=np.float64)
        self._y_train = np.asarray(y)
        self.classes_ = np.unique(self._y_train)
        return self

    def _get_neighbors(self, X: np.ndarray) -> np.ndarray:
        """Return indices of k nearest neighbors for each row in X."""
        X = np.asarray(X, dtype=np.float64)
        dists = _pairwise_sq_dist(X, self._X_train)
        k = self.n_neighbors
        if k >= dists.shape[1]:
            return np.argsort(dists, axis=1)[:, :k]
        # argpartition is O(n), then sort the k smallest
        idx = np.argpartition(dists, k, axis=1)[:, :k]
        # Sort within the k neighbors by distance (for deterministic ordering)
        rows = np.arange(X.shape[0])[:, None]
        sorted_within = np.argsort(dists[rows, idx], axis=1)
        return idx[rows, sorted_within]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        neigh_idx = self._get_neighbors(X)
        neigh_labels = self._y_train[neigh_idx]
        proba = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            proba[:, i] = np.mean(neigh_labels == c, axis=1)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        # Tie-breaking: argmax returns first (lowest index) max
        return self.classes_[np.argmax(proba, axis=1)]


class _KNNRegressor:
    """K-Nearest Neighbors regressor (sklearn-compatible protocol).

    Brute-force, O(n*m) memory. Deterministic.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors. Default 5.
    n_jobs : int
        Accepted for API compatibility, ignored.
    """

    def __init__(self, *, n_neighbors: int = 5, n_jobs: int = 1):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def get_params(self, deep: bool = True) -> dict:
        return {"n_neighbors": self.n_neighbors, "n_jobs": self.n_jobs}

    def set_params(self, **params) -> _KNNRegressor:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> _KNNRegressor:
        self._X_train = np.asarray(X, dtype=np.float64)
        self._y_train = np.asarray(y, dtype=np.float64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        dists = _pairwise_sq_dist(X, self._X_train)
        k = self.n_neighbors
        if k >= dists.shape[1]:
            idx = np.argsort(dists, axis=1)[:, :k]
        else:
            idx = np.argpartition(dists, k, axis=1)[:, :k]
        return np.mean(self._y_train[idx], axis=1)
