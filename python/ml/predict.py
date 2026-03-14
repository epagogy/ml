"""Prediction implementation.

Model.predict() is defined in _types.py but uses a late import to call _predict() here.
This avoids circular dependency: _types ↔ predict.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ._types import Model


def predict(
    model: Model, data: pd.DataFrame, *,
    proba: bool = False,
    augment: int | None = None,
    noise_scale: float = 0.01,
    seed: int | None = None,
    intervals: bool = False,
    confidence: float = 0.90,
) -> pd.Series | pd.DataFrame:
    """Predict on new data.

    Top-level convenience: ``ml.predict(model, data)`` is equivalent
    to ``model.predict(data)``.

    Args:
        model: Fitted Model or TuningResult
        data: DataFrame with same features as training data
        proba: If True, return class probabilities (classification only)
        augment: Number of test-time augmentation (TTA) passes. When set,
            adds small Gaussian noise to numeric features on each pass and
            averages the predictions. Reduces prediction variance, especially
            for unstable models or noisy features.
            Example: augment=10 makes 10 noisy passes, then averages.
            Default: None (standard single-pass prediction).
        noise_scale: Gaussian noise std for TTA (as fraction of feature std).
            Default: 0.01 (1% of each feature's standard deviation).
            Ignored when augment=None.
        seed: Random seed for TTA noise. Required when augment is not None.
            Also required when intervals=True.
            Ignored when augment=None and intervals=False.
        intervals: If True, return a DataFrame with prediction/lower/upper columns
            (regression only). Uses bootstrap noise augmentation for uncertainty.
        confidence: Confidence level for prediction intervals (default 0.90 = 90%).
            Ignored when intervals=False.

    Returns:
        Series of predictions, or DataFrame of class probabilities when proba=True,
        or DataFrame with prediction/lower/upper when intervals=True (regression).
    """
    from ._compat import to_pandas
    from ._types import ConfigError, TuningResult
    from ._types import Model as ModelType

    # Auto-convert Polars/other DataFrames to pandas
    data = to_pandas(data)

    from ._types import OptimizeResult
    if isinstance(model, TuningResult):
        model = model.best_model
    elif isinstance(model, OptimizeResult):
        model = model.model  # unwrap to underlying calibrated Model
    elif not isinstance(model, ModelType):
        raise ConfigError(
            f"predict() requires a Model, TuningResult, or OptimizeResult. Got {type(model).__name__}. "
            "Use ml.fit() first, then ml.predict(model, data)."
        )

    # DFA state transition: predict is idempotent (state unchanged)
    import contextlib

    from ._types import check_workflow_transition
    with contextlib.suppress(Exception):
        model._workflow_state = check_workflow_transition(
            model._workflow_state, "predict"
        )

    # Warn when confidence= is passed but intervals=False (F2 W33)
    _DEFAULT_CONFIDENCE = 0.90
    if not intervals and confidence != _DEFAULT_CONFIDENCE:
        warnings.warn(
            f"confidence={confidence!r} has no effect unless intervals=True. "
            "Use predict(intervals=True, confidence="
            f"{confidence!r}) to get prediction intervals.",
            UserWarning,
            stacklevel=2,
        )

    # Prediction intervals path (regression only)
    if intervals:
        if seed is None:
            raise ConfigError(
                "predict() with intervals=True requires seed=. "
                "Example: ml.predict(model, data, intervals=True, seed=42)"
            )
        if model._task == "classification":
            raise ConfigError(
                "intervals=True is only for regression. For classification uncertainty, "
                "use ml.predict(model, data, proba=True) instead."
            )
        return _predict_intervals(model, data, confidence=confidence, seed=seed)

    # Test-time augmentation path
    if augment is not None:
        if seed is None:
            raise ConfigError(
                "predict() with augment= requires seed=. "
                "Example: ml.predict(model, data, augment=10, seed=42)"
            )
        if model._task == "classification" and noise_scale > 0:
            warnings.warn(
                f"augment=True with noise_scale={noise_scale} on a classification model "
                "adds Gaussian noise to input features before predicting, which changes "
                "predicted class labels. For classification, noise augmentation is "
                "semantically different from regression TTA — class boundaries may shift. "
                "Use noise_scale=0 to disable noise, or proba=True for probability smoothing.",
                UserWarning,
                stacklevel=2,
            )
        return _predict_tta(model, data, proba=proba, augment=augment,
                            noise_scale=noise_scale, seed=seed)

    return _predict_impl(model, data, proba=proba)


def _predict_impl(
    model: Model, data: pd.DataFrame, *, proba: bool = False,
) -> pd.Series | pd.DataFrame:
    """Predict on new data.

    Called by Model.predict() via late import.

    Args:
        model: Fitted Model
        data: DataFrame with same features as training data
        proba: If True, return class probabilities instead of predictions.
            Classification only — raises ModelError for regression.

    Returns:
        Series of predictions (proba=False), or DataFrame of class
        probabilities (proba=True, classification only). DataFrame
        columns are class labels matching model.classes_.

    Raises:
        DataError: If features don't match training data
        ModelError: If proba=True on a regression model

    Example:
        >>> preds = model.predict(s.valid)
        >>> preds.shape
        (1000,)
        >>> probs = model.predict(s.valid, proba=True)  # classification
        >>> probs["yes"]  # probability of class "yes"
        0    0.87
        1    0.12
    """
    if proba:
        return _predict_proba(model, data)
    from ._types import DataError

    # Reject non-DataFrame input (Series, ndarray, etc.)
    if isinstance(data, pd.Series):
        raise DataError(
            f"predict() expects DataFrame, got Series (name='{data.name}'). "
            "Use df[[col1, col2]] (double brackets) instead of df[col]."
        )
    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"predict() expects DataFrame, got {type(data).__name__}. "
            "Convert to DataFrame first."
        )

    if len(data) == 0:
        raise DataError("Data has 0 rows. Cannot predict on an empty DataFrame.")

    # Extract features (drop target if present)
    if model._target in data.columns:
        X = data.drop(columns=[model._target])
    else:
        X = data

    # Validate features match
    missing_features = set(model._features) - set(X.columns)
    if missing_features:
        raise DataError(
            f"Missing features: {sorted(missing_features)}. "
            f"Expected features: {model._features}"
        )

    extra_features = set(X.columns) - set(model._features)
    if extra_features:
        warnings.warn(
            f"Extra features will be ignored: {sorted(extra_features)}",
            UserWarning,
            stacklevel=3
        )

    # Select + reorder columns to match training order (single operation)
    X = X[model._features]

    # Warn on all-NaN columns — tree models pass through silently, causing garbage predictions
    all_nan_cols = [c for c in model._features if X[c].isna().all()]
    if all_nan_cols:
        warnings.warn(
            f"Columns {all_nan_cols} are entirely NaN in prediction data. "
            "Predictions may be unreliable. Fill or drop these columns before predicting.",
            UserWarning,
            stacklevel=3,
        )

    # Transform features using stored normalization state
    # Suppress NaN/scaling warnings — fit() already warned once
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*NaN.*")
        warnings.filterwarnings("ignore", message=".*Auto-scaling.*")
        X_clean = model._feature_encoder.transform(X)

    # Apply custom preprocessor if present (Hook 4: Lego)
    if model._preprocessor is not None:
        X_clean = model._preprocessor(X_clean)

    # Predict using underlying model
    # A4: Apply custom threshold if set (binary classification only)
    if getattr(model, "_threshold", None) is not None and model._task == "classification":
        try:
            proba = model._model.predict_proba(X_clean).astype(np.float64)
        except AttributeError as exc:
            raise DataError(
                f"Backend '{model._algorithm}' does not support predict_proba(), "
                "required for threshold-based prediction."
            ) from exc
        # Normalize per-row to sum=1
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        proba = proba / row_sums
        # Positive class = last column; apply threshold
        pos_proba = proba[:, -1]
        raw_classes = model._model.classes_
        predictions = np.where(
            pos_proba >= model._threshold,
            raw_classes[-1],
            raw_classes[0],
        )
    else:
        try:
            predictions = model._model.predict(X_clean)
        except ValueError as exc:
            raise DataError(
                f"Prediction failed due to dtype mismatch: {exc}. "
                "Ensure all feature columns have numeric or categorical dtypes."
            ) from exc

    # Decode predictions back to original labels
    decoded = model._feature_encoder.decode(predictions)

    # Return as Series with original index
    result = pd.Series(decoded.values, index=data.index, name=model._target)

    # Ensure float64 for regression (XGBoost returns float32, but
    # pandas/scipy expect float64 for downstream computations)
    if model._task == "regression" and result.dtype != np.float64:
        result = result.astype(np.float64)

    return result


def _predict_proba(model: Model, data: pd.DataFrame) -> pd.DataFrame:
    """Predict class probabilities (classification only).

    Called by Model.predict_proba() via late import.

    Args:
        model: Fitted Model (must be classification task)
        data: DataFrame with same features as training data

    Returns:
        DataFrame of shape (n_samples, n_classes). Columns are class
        labels matching model.classes_. All values float64.

    Raises:
        ModelError: If task is regression
        DataError: If features don't match training data
    """
    from ._types import DataError, ModelError

    if model._task != "classification":
        raise ModelError(
            f"predict_proba() only works for classification. "
            f"This model's task is '{model._task}'."
        )


    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"predict_proba() expects DataFrame, got {type(data).__name__}. "
            "Convert to DataFrame first."
        )

    # Extract features (drop target if present)
    if model._target in data.columns:
        X = data.drop(columns=[model._target])
    else:
        X = data

    # Validate features match
    missing_features = set(model._features) - set(X.columns)
    if missing_features:
        raise DataError(
            f"Missing features: {sorted(missing_features)}. "
            f"Expected features: {model._features}"
        )

    extra_features = set(X.columns) - set(model._features)
    if extra_features:
        warnings.warn(
            f"Extra features will be ignored: {sorted(extra_features)}",
            UserWarning,
            stacklevel=3
        )

    # Select + reorder columns to match training order (single operation)
    X = X[model._features]

    # Transform features using stored normalization state
    # Suppress NaN/scaling warnings — fit() already warned once
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*NaN.*")
        warnings.filterwarnings("ignore", message=".*Auto-scaling.*")
        X_clean = model._feature_encoder.transform(X)

    # Apply custom preprocessor if present (Hook 4: Lego)
    if model._preprocessor is not None:
        X_clean = model._preprocessor(X_clean)

    # Get probabilities from underlying model
    try:
        proba = model._model.predict_proba(X_clean)
    except AttributeError as exc:
        raise ModelError(
            f"Algorithm '{model._algorithm}' does not support predict_proba(). "
            "If using a custom backend, ensure the estimator has a "
            ".predict_proba() method (e.g., pass probability=True for SVM)."
        ) from exc

    # Normalize to sum=1.0 per row (XGBoost can have floating-point drift)
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid division by zero
    proba = proba / row_sums

    # Upcast to float64 (XGBoost returns float32; scipy/sklearn expect float64)
    if proba.dtype != np.float64:
        proba = proba.astype(np.float64)

    # Wrap in labeled DataFrame (consistent with predict(proba=True))
    classes = model.classes_ or list(range(proba.shape[1]))
    return pd.DataFrame(proba, columns=classes, index=data.index)


def predict_proba(model: Model, data: pd.DataFrame) -> pd.DataFrame:
    """Predict class probabilities (classification only).

    Convenience alias for ``model.predict_proba(data)`` and
    ``predict(model, data, proba=True)``.

    Args:
        model: Fitted Model (classification task)
        data: DataFrame with same features as training data

    Returns:
        DataFrame of shape (n_samples, n_classes). Columns are class
        labels. All values are float64 and rows sum to 1.0.

    Example::

        probs = ml.predict_proba(model, s.valid)
        probs["yes"]  # probability of class "yes"
    """
    return _predict_proba(model, data)


# ---------------------------------------------------------------------------
# Prediction intervals
# ---------------------------------------------------------------------------


def _predict_intervals(
    model: Model,
    data: pd.DataFrame,
    *,
    confidence: float = 0.90,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap prediction intervals for regression.

    Generates N bootstrap passes by adding small noise to numeric features,
    then takes percentiles to form lower/upper bounds.

    Args:
        model: Fitted regression Model
        data: DataFrame with features (target column optional)
        confidence: Confidence level (default 0.90 = 90% interval)
        n_bootstrap: Number of bootstrap passes (default 50)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: prediction, lower, upper
    """
    alpha = (1.0 - confidence) / 2.0
    all_preds = []
    rng = np.random.RandomState(seed)

    # Strip target column if present
    if model._target in data.columns:
        X = data.drop(columns=[model._target])
    else:
        X = data

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        col_stds = X[numeric_cols].std().values.astype(np.float64)
        col_stds = np.where(col_stds == 0, 1.0, col_stds)
    else:
        col_stds = np.array([], dtype=np.float64)

    # Pre-extract numeric template once (avoids full DataFrame copy per pass)
    numeric_template = X[numeric_cols].values.astype(np.float64) if numeric_cols else None
    noisy = X.copy()  # single copy, reused each pass

    for _ in range(n_bootstrap):
        if numeric_template is not None:
            noise = rng.normal(0, 0.05, size=numeric_template.shape)
            noisy[numeric_cols] = numeric_template + noise * col_stds
        pred = _predict_impl(model, noisy, proba=False)
        all_preds.append(pred.values.astype(np.float64))

    arr = np.array(all_preds)  # shape: (n_bootstrap, n_samples)
    center = arr.mean(axis=0)
    lower = np.percentile(arr, alpha * 100.0, axis=0)
    upper = np.percentile(arr, (1.0 - alpha) * 100.0, axis=0)

    return pd.DataFrame(
        {"prediction": center, "lower": lower, "upper": upper},
        index=data.index,
    )


# ---------------------------------------------------------------------------
# Test-time augmentation
# ---------------------------------------------------------------------------


def _predict_tta(
    model: Model,
    data: pd.DataFrame,
    *,
    proba: bool,
    augment: int,
    noise_scale: float,
    seed: int,
) -> pd.Series | pd.DataFrame:
    """Test-time augmentation: N noisy passes → averaged predictions.

    Adds small Gaussian noise to numeric features on each pass.
    Noise magnitude = noise_scale * feature_std (so 0.01 = 1% of each feature's std).
    Averages all N predictions for a smoother, lower-variance result.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    # Pre-compute feature stds for noise scaling; replace 0 with 1 to avoid scaling issues
    if numeric_cols:
        col_stds = data[numeric_cols].std().values
        col_stds = np.where(col_stds == 0, 1.0, col_stds)
    else:
        col_stds = np.array([], dtype=np.float64)

    rng = np.random.RandomState(seed)
    passes: list[pd.Series | pd.DataFrame] = []

    # Pre-extract numeric template once (avoids full DataFrame copy per pass)
    numeric_template = data[numeric_cols].values.astype(np.float64) if numeric_cols else None
    noisy_data = data.copy() if numeric_cols else data  # single copy, reused

    for _ in range(augment):
        if numeric_template is not None:
            noise = rng.normal(0, noise_scale, size=numeric_template.shape)
            noisy_data[numeric_cols] = numeric_template + noise * col_stds

        preds = _predict_impl(model, noisy_data, proba=proba)
        passes.append(preds)

    return _average_predictions(passes, proba=proba)


def _average_predictions(
    passes: list[pd.Series | pd.DataFrame],
    *,
    proba: bool,
) -> pd.Series | pd.DataFrame:
    """Average TTA predictions across all passes.

    For proba=True (DataFrames of class probabilities): numeric mean.
    For proba=False:
        - Regression (numeric Series): mean of values.
        - Classification (label Series): mode (most common label).
    """
    first = passes[0]

    if proba:
        # Stack DataFrames, compute mean per cell
        stacked = np.mean([p.values for p in passes], axis=0)
        return pd.DataFrame(stacked, columns=first.columns, index=first.index)

    # Series path
    if pd.api.types.is_numeric_dtype(first):
        # Regression: mean of numeric predictions
        stacked = np.mean(
            [p.values.astype(np.float64) for p in passes], axis=0
        )
        return pd.Series(stacked, index=first.index, name=first.name)
    else:
        # Classification: majority vote using np.unique (scipy.stats.mode
        # dropped support for non-numeric arrays in 1.11+)
        preds_arr = np.array([p.values for p in passes])  # (n_passes, n_samples)
        n_samples = preds_arr.shape[1]
        majority = np.empty(n_samples, dtype=preds_arr.dtype)
        for i in range(n_samples):
            vals, counts = np.unique(preds_arr[:, i], return_counts=True)
            majority[i] = vals[np.argmax(counts)]
        return pd.Series(majority, index=first.index, name=first.name)
