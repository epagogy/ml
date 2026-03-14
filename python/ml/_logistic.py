"""Native logistic regression — numpy L-BFGS (Nocedal-Wright Alg 7.4).

No sklearn dependency. Supports:
- Binary classification (sigmoid + cross-entropy + L2)
- Multiclass via one-vs-rest (K independent binary classifiers)
- L2 regularisation (C parameter, bias not regularised)
- Strong Wolfe line search (Armijo + curvature conditions)
"""

from __future__ import annotations

from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Line search — strong Wolfe conditions
# ---------------------------------------------------------------------------

def _wolfe_line_search(
    f_grad,
    x: np.ndarray,
    p: np.ndarray,
    f0: float,
    g0: np.ndarray,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_ls: int = 20,
) -> tuple[float, np.ndarray, float]:
    """Backtracking line search satisfying Armijo + curvature (weak Wolfe).

    Returns (alpha, new_grad, new_f).
    Falls back to alpha=1e-6 if no satisfactory step is found.
    """
    alpha = 1.0
    slope0 = float(g0 @ p)

    for _ in range(max_ls):
        x_new = x + alpha * p
        f_new, g_new = f_grad(x_new)
        # Armijo (sufficient decrease)
        if f_new <= f0 + c1 * alpha * slope0:
            # Curvature condition (weak Wolfe)
            if float(g_new @ p) >= c2 * slope0:
                return alpha, g_new, f_new
        alpha *= 0.5

    # Fallback — take a tiny step to ensure progress
    alpha = 1e-6
    x_new = x + alpha * p
    f_new, g_new = f_grad(x_new)
    return alpha, g_new, f_new


# ---------------------------------------------------------------------------
# L-BFGS optimizer — Nocedal & Wright Algorithm 7.4
# ---------------------------------------------------------------------------

def _lbfgs(
    f_grad,
    x0: np.ndarray,
    m: int = 10,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> np.ndarray:
    """L-BFGS minimiser.

    Parameters
    ----------
    f_grad : callable
        Returns (scalar loss, gradient array) given parameter vector.
    x0 : np.ndarray
        Initial parameters (flat 1-D).
    m : int
        History size (number of (s, y) pairs stored).
    max_iter : int
        Maximum iterations.
    tol : float
        Gradient norm convergence tolerance.

    Returns
    -------
    np.ndarray
        Optimised parameter vector.
    """
    x = x0.copy().astype(np.float64)
    f, g = f_grad(x)

    s_list: deque = deque(maxlen=m)
    y_list: deque = deque(maxlen=m)
    rho_list: deque = deque(maxlen=m)

    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            break

        # Two-loop L-BFGS recursion to compute search direction H * g
        q = g.copy()
        alphas = []
        for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            a = rho * float(s @ q)
            alphas.append(a)
            q = q - a * y

        # Initial Hessian approximation: scale by last (s·y)/(y·y)
        if s_list:
            s_last = s_list[-1]
            y_last = y_list[-1]
            yy = float(y_last @ y_last)
            if yy > 0:
                gamma = float(s_last @ y_last) / yy
            else:
                gamma = 1.0
        else:
            gamma = 1.0

        r = gamma * q

        for s, y, rho, a in zip(s_list, y_list, rho_list, reversed(alphas)):
            beta = rho * float(y @ r)
            r = r + s * (a - beta)

        p = -r  # descent direction

        # Line search
        alpha, g_new, f_new = _wolfe_line_search(f_grad, x, p, f, g)

        s = alpha * p
        y = g_new - g

        sy = float(s @ y)
        if sy > 1e-10:
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / sy)

        x = x + s
        f = f_new
        g = g_new

    return x


# ---------------------------------------------------------------------------
# Loss + gradient for one binary classifier
# ---------------------------------------------------------------------------

def _binary_loss_grad(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    sample_weight: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    """Cross-entropy loss + L2 regularisation (bias not regularised).

    Parameters layout: w[0] = bias, w[1:] = feature weights.
    """
    z = X @ w[1:] + w[0]
    z = np.clip(z, -500.0, 500.0)
    p = 1.0 / (1.0 + np.exp(-z))
    p_safe = np.clip(p, 1e-15, 1.0 - 1e-15)

    n = len(y)
    if sample_weight is not None:
        sw = sample_weight / sample_weight.sum() * n
        loss = -np.sum(sw * (y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe))) / n
        err = sw * (p - y)
    else:
        loss = -np.mean(y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe))
        err = p - y

    # L2 penalty — match sklearn's C convention: C multiplies data term,
    # so effective regularization per sample is 1/(C*n).
    loss += 0.5 / (C * n) * np.sum(w[1:] ** 2)

    g_bias = np.mean(err)
    g_w = X.T @ err / n + w[1:] / (C * n)

    return loss, np.concatenate([[g_bias], g_w])


# ---------------------------------------------------------------------------
# Public estimator class
# ---------------------------------------------------------------------------

class _LogisticModel:
    """Native logistic regression with L-BFGS optimiser.

    sklearn-compatible interface: fit / predict / predict_proba.
    Multiclass via one-vs-rest. L2 regularisation only.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self._coefs: list[np.ndarray] = []   # one coef vector per class (binary: length 1)
        self.classes_: np.ndarray = np.array([])
        self._n_features: int = 0

    # ---- sklearn compatibility ----

    @property
    def coef_(self) -> np.ndarray:
        """Feature weights as 2-D array, mimicking sklearn's coef_ interface.

        Binary:     shape (1, n_features)
        Multiclass: shape (n_classes, n_features)
        Bias excluded (index 0 of each stored coef vector).
        """
        if len(self.classes_) == 2:
            return self._coefs[0][1:].reshape(1, -1)
        return np.array([c[1:] for c in self._coefs])

    def get_params(self, deep: bool = True) -> dict:
        return {"C": self.C, "max_iter": self.max_iter}

    def set_params(self, **params) -> _LogisticModel:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ---- core ----

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> _LogisticModel:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self._n_features = X.shape[1]
        self._coefs = []

        if n_classes == 2:
            # Binary: single classifier, positive class = classes_[1]
            y_bin = (y == self.classes_[1]).astype(np.float64)
            w0 = np.zeros(X.shape[1] + 1)

            def fg(w):
                return _binary_loss_grad(w, X, y_bin, self.C, sample_weight)

            self._coefs.append(_lbfgs(fg, w0, max_iter=self.max_iter))
        else:
            # Multiclass OvR: K binary classifiers
            for k_idx in range(n_classes):
                y_bin = (y == self.classes_[k_idx]).astype(np.float64)
                w0 = np.zeros(X.shape[1] + 1)

                def fg(w, y_b=y_bin):
                    return _binary_loss_grad(w, X, y_b, self.C, sample_weight)

                self._coefs.append(_lbfgs(fg, w0, max_iter=self.max_iter))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_classes = len(self.classes_)

        if n_classes == 2:
            w = self._coefs[0]
            z = np.clip(X @ w[1:] + w[0], -500.0, 500.0)
            p1 = 1.0 / (1.0 + np.exp(-z))
            proba = np.column_stack([1.0 - p1, p1])
        else:
            # OvR: compute raw scores for each class, then normalise
            scores = np.empty((X.shape[0], n_classes))
            for k_idx, w in enumerate(self._coefs):
                z = np.clip(X @ w[1:] + w[0], -500.0, 500.0)
                scores[:, k_idx] = 1.0 / (1.0 + np.exp(-z))
            row_sums = scores.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            proba = scores / row_sums

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]
