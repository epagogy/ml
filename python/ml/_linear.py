"""Native Ridge regression — numpy closed-form solution.

No sklearn dependency. Solves:
    min ||y - (X w + b)||^2 + alpha * ||w||^2
via the normal equations:
    (X_aug^T X_aug + Lambda) w_aug = X_aug^T y

where X_aug = [1 | X] (bias first) and Lambda = diag([0, alpha, ..., alpha])
(bias term not regularised).

Supports sample_weight via the weighted normal equations without materialising
an n×n weight matrix (O(n p^2) cost, same as unweighted).
"""

from __future__ import annotations

import numpy as np


class _LinearModel:
    """Native Ridge regression (closed-form).

    sklearn-compatible interface: fit / predict.
    Single alpha (regularisation strength). Bias not regularised.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self._coef: np.ndarray = np.array([])
        self._intercept: float = 0.0

    # ---- sklearn compatibility ----

    def get_params(self, deep: bool = True) -> dict:
        return {"alpha": self.alpha}

    def set_params(self, **params) -> _LinearModel:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    @property
    def coef_(self) -> np.ndarray:
        """Feature weights, shape (n_features,) — mirrors sklearn Ridge.coef_."""
        return self._coef

    # ---- core ----

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> _LinearModel:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape

        # Augmented design matrix: [bias | features]
        X_aug = np.column_stack([np.ones(n), X])

        # Penalty matrix: bias (index 0) not regularised
        lam = np.zeros(p + 1)
        lam[1:] = self.alpha
        Lambda = np.diag(lam)

        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64)
            sw = sw / sw.sum() * n          # normalise to sum = n
            sw_sqrt = np.sqrt(sw)
            # Scale rows — avoids materialising n×n weight matrix
            X_w = X_aug * sw_sqrt[:, None]
            y_w = y * sw_sqrt
            A = X_w.T @ X_w + Lambda
            b = X_w.T @ y_w
        else:
            A = X_aug.T @ X_aug + Lambda
            b = X_aug.T @ y

        w = np.linalg.solve(A, b)
        self._intercept = float(w[0])
        self._coef = w[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self._coef + self._intercept
