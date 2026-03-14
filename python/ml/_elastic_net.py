"""Native Elastic Net — zero sklearn dependency.

Coordinate descent with soft-thresholding for L1 + ridge L2 term.
Convergence uses relative tolerance (audit condition C3).
"""

from __future__ import annotations

import warnings

import numpy as np


class _ElasticNetModel:
    """Elastic Net regression (sklearn-compatible protocol).

    Minimizes: (1/(2*n)) * ||y - Xw||^2 + alpha * l1_ratio * ||w||_1
                                          + alpha * (1-l1_ratio) * 0.5 * ||w||^2

    Parameters
    ----------
    alpha : float
        Regularization strength. Default 1.0.
    l1_ratio : float
        L1/L2 mix. 0 = Ridge, 1 = Lasso. Default 0.5.
    max_iter : int
        Maximum coordinate descent iterations. Default 1000.
    tol : float
        Convergence tolerance (relative). Default 1e-4.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    def get_params(self, deep: bool = True) -> dict:
        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    def set_params(self, **params) -> _ElasticNetModel:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(
        self, X: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray | None = None
    ) -> _ElasticNetModel:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape

        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64)
            sw = sw * (n / sw.sum())  # normalize so sum = n
        else:
            sw = np.ones(n)

        # Center data (weighted)
        X_mean = np.average(X, axis=0, weights=sw)
        y_mean = np.average(y, weights=sw)
        Xc = X - X_mean
        yc = y - y_mean

        # Apply sqrt(weights) for weighted least squares
        sw_sqrt = np.sqrt(sw)
        Xw = Xc * sw_sqrt[:, None]
        yw = yc * sw_sqrt

        # Pre-compute ||X_j||^2 (loop invariant)
        col_norms_sq = np.sum(Xw ** 2, axis=0)

        # L1/L2 penalties
        l1_pen = self.alpha * self.l1_ratio * n
        l2_pen = self.alpha * (1 - self.l1_ratio) * n

        # Relative convergence tolerance (audit C3)
        # sklearn uses: tol * ||y||^2 / n
        y_norm_sq = np.dot(yw, yw)
        abs_tol = self.tol * y_norm_sq / n

        # Initialize coefficients
        coef = np.zeros(p)
        residual = yw.copy()

        for iteration in range(1, self.max_iter + 1):
            max_update = 0.0

            for j in range(p):
                if col_norms_sq[j] == 0:
                    continue

                # Add back j-th contribution
                residual += Xw[:, j] * coef[j]

                # Raw update (OLS without penalty)
                rho = np.dot(Xw[:, j], residual)

                # Soft-thresholding for L1
                coef_new = np.sign(rho) * max(abs(rho) - l1_pen, 0.0)
                coef_new /= (col_norms_sq[j] + l2_pen)

                # Track convergence
                update = abs(coef_new - coef[j])
                if update > max_update:
                    max_update = update

                # Update residual
                residual -= Xw[:, j] * coef_new
                coef[j] = coef_new

            self.n_iter_ = iteration
            if max_update < abs_tol:
                break
        else:
            warnings.warn(
                f"ElasticNet did not converge after {self.max_iter} iterations. "
                f"Consider increasing max_iter or decreasing tol.",
                stacklevel=2,
            )

        self.coef_ = coef
        self.intercept_ = float(y_mean - X_mean @ coef)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_
