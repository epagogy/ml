"""blend() — blend multiple model predictions.

Rank averaging, geometric mean, and power mean blending for competition use.
Rank blending is invariant to miscalibrated probabilities — robust for competition and production use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._types import ConfigError


def blend(
    predictions: list[pd.Series | pd.DataFrame | np.ndarray],
    *,
    method: str = "mean",
    weights: list[float] | None = None,
    power: float = 1.0,
) -> pd.Series:
    """Blend multiple model predictions into a single prediction.

    Why rank averaging wins:
        Models may be miscalibrated (probabilities don't match true frequencies).
        Rank-transforming predictions makes blending invariant to calibration.
        If model A outputs [0.1, 0.2, 0.9] and model B outputs [0.001, 0.003, 0.8],
        arithmetic mean is dominated by A. Rank average treats them equally.

    Methods:
        "mean":      Weighted arithmetic mean (default)
        "rank":      Rank-transform each model's predictions, then weighted mean of ranks
        "geometric": Weighted geometric mean — naturally handles log-space (good for AUC)
        "power":     Generalized mean: M_p = (sum(w_i * x_i^p))^(1/p)
                     power=1 = arithmetic, power=0 → geometric, power=-1 = harmonic

    Args:
        predictions: List of 1D prediction arrays, pd.Series, or proba DataFrames
            (all same length). When a proba DataFrame is passed (e.g. from
            ``ml.predict(proba=True)``), the positive-class column (index 1) is
            extracted automatically for binary classification.
        method: Blending method ("mean", "rank", "geometric", "power").
        weights: Per-model weights (default: equal). Must sum to 1.
        power: Exponent for power mean (only used when method="power").

    Returns:
        pd.Series: Blended 1D predictions with the index from the first input.

    Raises:
        ConfigError: If weights don't sum to 1, method is unknown, or lengths differ.

    Example:
        >>> p1 = ml.predict(model_a, test, proba=True)  # (n, 2) DataFrame
        >>> p2 = ml.predict(model_b, test, proba=True)  # (n, 2) DataFrame
        >>> blended = ml.blend([p1, p2], method="rank")  # pd.Series, shape (n,)
    """
    if not predictions:
        raise ConfigError("blend() requires at least one prediction array.")

    # Capture index from first input for output alignment
    first_index: pd.Index | None = None
    if isinstance(predictions[0], (pd.Series, pd.DataFrame)):
        first_index = predictions[0].index

    # Validate prediction types — catch string class-label predictions early
    for i, p in enumerate(predictions):
        sample = p if not isinstance(p, (pd.Series, pd.DataFrame)) else (
            p.iloc[0] if isinstance(p, pd.Series) else p.iloc[0, 0]
        )
        if isinstance(sample, str):
            raise ConfigError(
                f"predictions[{i}] contains string class labels (e.g. '{sample}'). "
                "blend() requires numeric probabilities, not class predictions. "
                "Use ml.predict(model, data, proba=True) to get probabilities."
            )

    # Normalize to 1D numpy arrays.
    # For proba DataFrames (2D, columns = classes): extract positive class (col index 1).
    # This handles the common pattern of passing ml.predict(proba=True) directly.
    arrays: list[np.ndarray] = []
    for p in predictions:
        if isinstance(p, pd.DataFrame):
            if p.ndim == 2 and p.shape[1] >= 2:
                # Binary classification proba DataFrame: extract positive-class column
                arr = p.iloc[:, 1].to_numpy(dtype=np.float64)
            else:
                # Single-column DataFrame or unexpected shape: flatten
                arr = p.to_numpy(dtype=np.float64).ravel()
        elif isinstance(p, pd.Series):
            arr = p.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(p, dtype=np.float64)
            if arr.ndim != 1:
                arr = arr.ravel()
        arrays.append(arr)

    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        sizes = [len(a) for a in arrays]
        raise ConfigError(
            f"All prediction arrays must have the same length, got {sizes}."
        )

    n_models = len(arrays)

    # Resolve weights
    if weights is None:
        w = [1.0 / n_models] * n_models
    else:
        w = list(weights)
        if len(w) != n_models:
            raise ConfigError(
                f"len(weights)={len(w)} but {n_models} predictions provided."
            )
        total = sum(w)
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                f"weights must sum to 1.0, got {total:.6f}. "
                "Example: weights=[0.7, 0.3] for two models."
            )

    valid_methods = ("mean", "weighted", "rank", "geometric", "gmean", "power")
    if method not in valid_methods:
        raise ConfigError(
            f"method='{method}' not recognised. Choose from: {valid_methods}"
        )

    if method in ("mean", "weighted"):
        result_arr = _arithmetic_mean(arrays, w)
    elif method == "rank":
        result_arr = _rank_average(arrays, w)
    elif method in ("geometric", "gmean"):
        result_arr = _geometric_mean(arrays, w)
    else:  # power
        result_arr = _power_mean(arrays, w, power)

    # Return pd.Series to preserve index alignment
    return pd.Series(result_arr, index=first_index)


def _arithmetic_mean(
    predictions: list[np.ndarray], weights: list[float]
) -> np.ndarray:
    blended = np.zeros(len(predictions[0]), dtype=np.float64)
    for w, p in zip(weights, predictions):
        blended += w * p
    return blended


def _rank_average(
    predictions: list[np.ndarray], weights: list[float]
) -> np.ndarray:
    blended = np.zeros(len(predictions[0]), dtype=np.float64)
    for w, p in zip(weights, predictions):
        sorter = np.argsort(p, kind="mergesort")
        ranks = np.empty(len(p), dtype=np.float64)
        ranks[sorter] = np.arange(1, len(p) + 1, dtype=np.float64)
        blended += w * (ranks / len(p))
    return blended


def _geometric_mean(
    predictions: list[np.ndarray], weights: list[float]
) -> np.ndarray:
    log_sum = np.zeros(len(predictions[0]), dtype=np.float64)
    for w, p in zip(weights, predictions):
        log_sum += w * np.log(np.clip(p, 1e-15, 1 - 1e-15))
    return np.exp(log_sum)


def _power_mean(
    predictions: list[np.ndarray], weights: list[float], power: float
) -> np.ndarray:
    if abs(power) < 1e-10:  # geometric limit
        return _geometric_mean(predictions, weights)
    powered_sum = np.zeros(len(predictions[0]), dtype=np.float64)
    for w, p in zip(weights, predictions):
        powered_sum += w * np.power(np.clip(p, 1e-15, None), power)
    return np.power(powered_sum, 1.0 / power)
