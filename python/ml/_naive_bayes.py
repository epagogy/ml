"""Native Gaussian Naive Bayes — zero sklearn dependency.

Closed-form: per-class mean/variance + class priors, log-space prediction.
var_smoothing follows sklearn convention: epsilon = var_smoothing * max(all variances).
"""

from __future__ import annotations

import numpy as np


class _NaiveBayesModel:
    """Gaussian Naive Bayes classifier (sklearn-compatible protocol).

    Parameters
    ----------
    var_smoothing : float
        Portion of the largest variance of all features added to variances
        for numerical stability.  Default 1e-9 (matches sklearn).
    """

    def __init__(self, *, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self._theta: np.ndarray | None = None  # (n_classes, n_features) means
        self._var: np.ndarray | None = None    # (n_classes, n_features) variances
        self._prior: np.ndarray | None = None  # (n_classes,) log-priors
        self._n_features: int = 0

    # -- sklearn protocol --

    def get_params(self, deep: bool = True) -> dict:
        return {"var_smoothing": self.var_smoothing}

    def set_params(self, **params) -> _NaiveBayesModel:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(
        self, X: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray | None = None
    ) -> _NaiveBayesModel:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self._n_features = n_features

        self._theta = np.zeros((n_classes, n_features))
        self._var = np.zeros((n_classes, n_features))
        class_counts = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            mask = y == c
            X_c = X[mask]
            if sample_weight is not None:
                w = sample_weight[mask]
                w_sum = w.sum()
                class_counts[i] = w_sum
                # Weighted mean
                self._theta[i] = np.average(X_c, axis=0, weights=w)
                # Weighted variance
                diff = X_c - self._theta[i]
                self._var[i] = np.average(diff ** 2, axis=0, weights=w)
            else:
                class_counts[i] = X_c.shape[0]
                self._theta[i] = X_c.mean(axis=0)
                self._var[i] = X_c.var(axis=0)

        # Audit condition C1: epsilon = var_smoothing * max(ALL variances globally)
        epsilon = self.var_smoothing * np.max(np.var(X, axis=0))
        self._var += epsilon

        # Log-priors from class frequencies
        self._prior = np.log(class_counts / class_counts.sum())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        # Log-likelihood: -0.5 * (log(2*pi*var) + (x-mu)^2/var)
        # Summed over features per class
        log_proba = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            log_var = np.log(self._var[i])
            diff = X - self._theta[i]
            log_proba[:, i] = (
                self._prior[i]
                - 0.5 * np.sum(log_var)
                - 0.5 * np.sum(diff ** 2 / self._var[i], axis=1)
            )

        # Log-sum-exp trick for numerical stability
        max_log = log_proba.max(axis=1, keepdims=True)
        log_proba -= max_log
        np.exp(log_proba, out=log_proba)
        log_proba /= log_proba.sum(axis=1, keepdims=True)
        return log_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
