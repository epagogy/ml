"""calibrate() — post-hoc probability calibration.

ml.calibrate(model, data) → Model (with calibrated predict_proba)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ._types import Model, TuningResult


def calibrate(
    model: Model | TuningResult,
    *,
    data: pd.DataFrame,
    method: str = "auto",
) -> Model:
    """Calibrate a classification model's probabilities.

    Returns a NEW Model with calibrated predict_proba. The original model
    is unchanged. Use held-out data (e.g., s.valid) — never training data.

    Works for binary and multiclass classification. For multiclass, each
    class is calibrated independently (one-vs-rest).

    Args:
        model: Trained Model or TuningResult (classification only)
        data: Calibration data with target column (held-out, e.g. s.valid).
            Needs >= 100 rows (raises DataError otherwise).
        method: Calibration method.
            "auto" (default) — sigmoid for <1000 samples, isotonic for >=1000.
            "sigmoid" — Platt scaling (2 parameters, stable on small data).
            "isotonic" — non-parametric (flexible, needs >=1000 samples).

    Returns:
        New Model with calibrated probabilities

    Raises:
        ModelError: If model is regression or lacks predict_proba
        ConfigError: If method is invalid
        DataError: If target missing, too few samples (<100), or single class

    Example:
        >>> model = ml.fit(s.train, "churn", seed=42)
        >>> calibrated = ml.calibrate(model, data=s.valid)
        >>> probs = calibrated.predict(s.test, proba=True)
    """
    from ._types import ConfigError, DataError, Model, ModelError, TuningResult

    # Unwrap TuningResult
    if isinstance(model, TuningResult):
        model = model.best_model

    if not isinstance(model, Model):
        raise ConfigError(
            f"calibrate() expects a Model or TuningResult, got {type(model).__name__}."
        )

    # DFA state transition: calibrate is idempotent (state unchanged)
    import contextlib

    from ._types import check_workflow_transition
    with contextlib.suppress(Exception):
        model._workflow_state = check_workflow_transition(
            model._workflow_state, "calibrate"
        )

    # Partition guard — calibrate uses validation data, reject test
    from ._provenance import guard_evaluate
    guard_evaluate(data)

    # Classification only
    if model._task != "classification":
        raise ModelError(
            "calibrate() is for classification models only. "
            f"This model's task is '{model._task}'."
        )

    # Must have predict_proba
    if not hasattr(model._model, "predict_proba"):
        raise ModelError(
            f"Model ({model._algorithm}) does not support predict_proba. "
            "Calibration requires a probabilistic classifier."
        )

    # Normalise aliases before validation
    if method == "platt":
        method = "sigmoid"  # 'platt' is the canonical statistics name for sigmoid calibration

    # Validate method
    valid_methods = {"auto", "sigmoid", "isotonic"}
    if method not in valid_methods:
        raise ConfigError(
            f"method='{method}' is not valid. "
            f"Choose from: {sorted(valid_methods)} (or 'platt' as alias for 'sigmoid')"
        )

    # Validate data
    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"data= must be a DataFrame, got {type(data).__name__}."
        )

    if len(data) == 0:
        raise DataError("Calibration data is empty (0 rows).")

    if model._target not in data.columns:
        available = data.columns.tolist()
        raise DataError(
            f"Target '{model._target}' not found in calibration data. "
            f"Available columns: {available}"
        )

    # Minimum sample size guard
    n = len(data)
    if n < 100:
        raise DataError(
            f"Calibration requires >= 100 samples, got {n}. "
            "Use a larger held-out set or skip calibration."
        )

    # Resolve method
    if method == "auto":
        resolved = "isotonic" if n >= 1000 else "sigmoid"
    else:
        resolved = method

    if resolved == "isotonic" and n < 1000:
        warnings.warn(
            f"Isotonic calibration with {n} samples may overfit. "
            "Consider method='sigmoid' for <1000 calibration samples.",
            UserWarning,
            stacklevel=2,
        )

    # Warn on double calibration
    if model._calibrated:
        warnings.warn(
            "This model is already calibrated. Re-calibrating stacks "
            "transformations and may distort probabilities. "
            "Use the original uncalibrated model instead.",
            UserWarning,
            stacklevel=2,
        )

    # Extract X and y
    y = data[model._target]
    X = data.drop(columns=[model._target])

    # Guard: need at least 2 classes for calibration
    n_classes = y.nunique()
    if n_classes < 2:
        raise DataError(
            f"Calibration data contains only {n_classes} class. "
            "Need at least 2 classes in the target. "
            "Check your calibration split for class imbalance."
        )

    # Feature alignment (same guards as predict.py)
    missing_features = set(model._features) - set(X.columns)
    if missing_features:
        raise DataError(
            f"Missing features in calibration data: {sorted(missing_features)}. "
            f"Expected: {model._features}"
        )
    X = X[model._features]  # reorder to match training

    # Transform features using stored normalization state
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*NaN.*")
        warnings.filterwarnings("ignore", message=".*Auto-scaling.*")
        X_clean = model._feature_encoder.transform(X)

    # Apply custom preprocessor if present
    if model._preprocessor is not None:
        X_clean = model._preprocessor(X_clean)

    # Encode target (handles None encoder path)
    y_enc = model._feature_encoder.encode_target(y)

    # Build calibrated wrapper (no sklearn dependency)
    import numpy as np

    proba = model._model.predict_proba(X_clean)
    if isinstance(proba, pd.DataFrame):
        proba = proba.values
    proba = np.asarray(proba, dtype=np.float64)

    ccv = _CalibratedModel(model._model, resolved)
    ccv.fit(proba, np.asarray(y_enc))

    # Return new Model with calibrated estimator
    return Model(
        _model=ccv,
        _task=model._task,
        _algorithm=model._algorithm,
        _features=model._features,
        _target=model._target,
        _seed=model._seed,
        _label_encoder=model._label_encoder,
        _feature_encoder=model._feature_encoder,
        _preprocessor=model._preprocessor,
        _n_train=model._n_train,
        scores_=model.scores_,
        fold_scores_=model.fold_scores_,
        _time=model._time,
        _balance=model._balance,
        _calibrated=True,
    )


# ---------------------------------------------------------------------------
# Calibration internals — Platt scaling, isotonic (PAV), OVR wrapper
# ---------------------------------------------------------------------------

def _fit_platt(p, y):
    """Fit Platt scaling: P(y=1|p) = 1 / (1 + exp(a*p + b)).

    Returns (a, b) minimizing log-loss via gradient descent (numpy-only).
    """
    import numpy as np

    p = np.asarray(p, dtype=np.float64).clip(1e-8, 1 - 1e-8)
    y = np.asarray(y, dtype=np.float64)

    a, b = 1.0, 0.0
    lr = 0.1
    for _ in range(1000):
        logit = a * p + b
        prob = np.where(
            logit >= 0,
            1.0 / (1.0 + np.exp(-logit)),
            np.exp(logit) / (1.0 + np.exp(logit)),
        )
        prob = np.clip(prob, 1e-8, 1 - 1e-8)
        err = prob - y
        a -= lr * np.mean(err * p)
        b -= lr * np.mean(err)
    return float(a), float(b)


def _apply_platt(p, a, b):
    """Apply Platt scaling: 1 / (1 + exp(a*p + b))."""
    import numpy as np

    logit = a * np.asarray(p, dtype=np.float64) + b
    return np.where(
        logit >= 0,
        1.0 / (1.0 + np.exp(-logit)),
        np.exp(logit) / (1.0 + np.exp(logit)),
    )


def _fit_isotonic(p, y):
    """Fit isotonic regression via Pool Adjacent Violators (PAV).

    Returns (x_thresholds, y_values) for stepwise interpolation.
    """
    import numpy as np

    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Sort by predicted probability
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]

    # PAV algorithm
    n = len(y_sorted)
    blocks = [[i] for i in range(n)]
    values = list(y_sorted.astype(float))
    weights = [1.0] * n

    i = 0
    while i < len(values) - 1:
        if values[i] > values[i + 1]:
            # Merge blocks
            new_w = weights[i] + weights[i + 1]
            new_v = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / new_w
            values[i] = new_v
            weights[i] = new_w
            blocks[i] = blocks[i] + blocks[i + 1]
            del values[i + 1]
            del weights[i + 1]
            del blocks[i + 1]
            # Check previous block
            if i > 0:
                i -= 1
        else:
            i += 1

    # Build thresholds: average p for each block
    x_thresh = np.array([np.mean(p_sorted[b]) for b in blocks])
    y_vals = np.array(values)
    return x_thresh, y_vals


def _apply_isotonic(p, x_thresh, y_vals):
    """Apply isotonic calibration via linear interpolation."""
    import numpy as np

    return np.interp(np.asarray(p, dtype=np.float64), x_thresh, y_vals).clip(0, 1)


class _CalibratedModel:
    """OVR calibration wrapper. Wraps a fitted estimator with per-class calibration.

    For binary: calibrate class-1 probability.
    For multiclass: calibrate each class independently, renormalize to sum=1.
    """

    def __init__(self, base_estimator, method="sigmoid"):
        self.base_estimator = base_estimator
        self.method = method
        self.calibrators_ = []  # list of (method, params) per class
        self.classes_ = None

    def fit(self, proba, y):
        """Fit calibrators from base model's predict_proba output and true labels."""
        import numpy as np

        classes = np.unique(y)
        self.classes_ = classes
        n_classes = len(classes)

        if n_classes <= 2:
            # Binary: calibrate P(class=1) only
            p1 = proba[:, 1] if proba.ndim > 1 else proba
            y_bin = (y == classes[-1]).astype(float) if n_classes == 2 else y.astype(float)
            if self.method == "sigmoid":
                a, b = _fit_platt(p1, y_bin)
                self.calibrators_ = [("sigmoid", (a, b))]
            else:
                x_t, y_v = _fit_isotonic(p1, y_bin)
                self.calibrators_ = [("isotonic", (x_t, y_v))]
        else:
            # Multiclass OVR: calibrate each class independently
            self.calibrators_ = []
            for i, cls in enumerate(classes):
                p_cls = proba[:, i]
                y_bin = (y == cls).astype(float)
                if self.method == "sigmoid":
                    a, b = _fit_platt(p_cls, y_bin)
                    self.calibrators_.append(("sigmoid", (a, b)))
                else:
                    x_t, y_v = _fit_isotonic(p_cls, y_bin)
                    self.calibrators_.append(("isotonic", (x_t, y_v)))

    def predict_proba(self, X):
        """Calibrated predict_proba."""
        import numpy as np

        proba = self.base_estimator.predict_proba(X)
        if isinstance(proba, pd.DataFrame):
            proba = proba.values
        proba = np.asarray(proba, dtype=np.float64)

        n_classes = len(self.classes_)

        if n_classes <= 2 and len(self.calibrators_) == 1:
            # Binary
            p1 = proba[:, 1] if proba.ndim > 1 else proba
            method, params = self.calibrators_[0]
            if method == "sigmoid":
                cal_p1 = _apply_platt(p1, *params)
            else:
                cal_p1 = _apply_isotonic(p1, *params)
            cal_p1 = np.clip(cal_p1, 0, 1)
            return np.column_stack([1 - cal_p1, cal_p1])
        else:
            # Multiclass OVR
            cal = np.zeros_like(proba)
            for i, (method, params) in enumerate(self.calibrators_):
                if method == "sigmoid":
                    cal[:, i] = _apply_platt(proba[:, i], *params)
                else:
                    cal[:, i] = _apply_isotonic(proba[:, i], *params)
            # Renormalize rows to sum=1
            row_sums = cal.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            return cal / row_sums

    def predict(self, X):
        """Predict class labels from calibrated probabilities."""
        import numpy as np

        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        """Accuracy score (for compatibility with model.score())."""
        import numpy as np

        preds = self.predict(X)
        return float(np.mean(preds == y))
