"""Threshold optimization for binary classification — A4.

Finds the optimal decision threshold by sweeping from min_threshold to 0.95
using a two-phase search (coarse then fine). Returns a copy of the model
with ``_threshold`` set; subsequent ``model.predict()`` calls use it.

Usage:
    >>> optimized = ml.optimize(model, data=s.valid, metric="f1")
    >>> optimized._threshold
    0.37
    >>> preds = optimized.predict(s.test)   # uses 0.37 instead of 0.5
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd

from ._scoring import METRIC_REGISTRY, make_scorer
from ._types import ConfigError, DataError, ModelError


def optimize(
    model,
    *,
    data: pd.DataFrame | None = None,
    oof_predictions: pd.Series | pd.DataFrame | None = None,
    metric: str | callable = "f1",
    min_threshold: float | str = "auto",
) -> OptimizeResult:  # noqa: F821 — forward ref resolved at runtime
    """Find optimal prediction threshold for binary classification.

    Sweeps thresholds from *min_threshold* to 0.95 in two phases (coarse 0.05
    steps, then fine 0.005 steps around the coarse best) and returns a copy of
    *model* with ``_threshold`` set.  Subsequent ``model.predict()`` calls apply
    this threshold to the positive-class probability instead of using 0.5.

    Args:
        model: Fitted binary classification Model (2 classes only).
        data: DataFrame containing the target column used for true labels.
            Required unless *oof_predictions* already contains aligned labels.
        oof_predictions: Pre-computed positive-class probabilities aligned with
            *data* (same row order and index).  When provided, these are used
            instead of calling ``model.predict_proba(data)``, which avoids
            optimistic selection bias when *data* was used during training.
            Accepts a ``pd.Series`` or single-column ``pd.DataFrame``.
        metric: Optimisation objective.  String key from the built-in registry
            (``"f1"``, ``"f1_weighted"``, ``"accuracy"``, ``"precision"``,
            ``"recall"``) or a callable ``scorer(y_true, y_pred) -> float``
            with optional ``scorer.greater_is_better`` attribute.
            Ranking metrics (``"roc_auc"``, ``"log_loss"``) are rejected
            because they are threshold-independent.
        min_threshold: Lower bound of the sweep.  ``"auto"`` (default) computes
            ``max(0.001, 1 / n_positives)`` — the minimum meaningful threshold
            for imbalanced data.  Pass a float to override.

    Returns:
        Model — a copy of the input model with threshold tuned. The original is
        unchanged. Use this returned model for all subsequent predictions; do NOT
        extract a float and apply it manually.

        To inspect the threshold value: ``optimized.threshold`` (public property).
        To predict with the threshold applied: ``ml.predict(optimized, data)``.
        The threshold is baked in — every ``predict()`` call uses it automatically.

    Raises:
        ModelError:  If model task is not classification.
        ModelError:  If model has more than 2 classes (multiclass not supported).
        ConfigError: If *metric* is a ranking metric (``roc_auc``, ``log_loss``).
        ConfigError: If neither *data* nor *oof_predictions* is provided.
        DataError:   If *oof_predictions* has an unsupported type.

    Example:
        >>> import ml
        >>> data = ml.dataset("churn")
        >>> s = ml.split(data, "churn", seed=42)
        >>> model = ml.fit(s.dev, "churn", seed=42)
        >>> optimized = ml.optimize(model, data=s.valid, metric="f1")
        >>> optimized.threshold       # inspect the threshold found
        0.4
        >>> preds = ml.predict(optimized, s.test)  # threshold applied automatically
    """
    from ._types import Model as _Model
    from ._types import TuningResult as _TuningResult

    if isinstance(model, _TuningResult):
        model = model.best_model
    elif not isinstance(model, _Model):
        raise ModelError(
            f"optimize() requires a Model or TuningResult, got {type(model).__name__}. "
            "Use ml.fit() or ml.tune() first."
        )

    # Task guard
    if model._task != "classification":
        raise ModelError(
            "optimize() only works for binary classification models. "
            f"This model's task is '{model._task}'."
        )

    # Binary-only guard
    classes = model.classes_
    if classes is None or len(classes) != 2:
        n = len(classes) if classes is not None else 0
        raise ModelError(
            f"optimize() only works for binary classification (2 classes). "
            f"This model has {n} class(es). For multiclass, use per-class thresholds."
        )

    # Ranking metric guard
    if isinstance(metric, str):
        registry_entry = METRIC_REGISTRY.get(metric)
        if registry_entry is not None and registry_entry.needs_proba:
            raise ConfigError(
                f"metric='{metric}' is a ranking metric and cannot be threshold-optimised. "
                "Use a label-based metric instead: 'f1', 'f1_weighted', 'accuracy', "
                "'precision', or 'recall'."
            )

    # Data requirement guard
    if data is None:
        raise ConfigError(
            "data= is required to provide true labels. "
            "Pass the validation or OOF dataset you want to optimise on."
        )

    scorer = make_scorer(metric)

    # Guard: scorer must not need probabilities (those are ranking metrics)
    if scorer.needs_proba:
        raise ConfigError(
            f"metric='{scorer.name}' is a probability-based ranking metric. "
            "Threshold optimisation requires a label-based metric ('f1', 'accuracy', etc.)."
        )

    # True labels
    if model._target not in data.columns:
        raise DataError(
            f"Target column '{model._target}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )
    y_true = data[model._target].values

    # Positive-class probabilities
    if oof_predictions is not None:
        if isinstance(oof_predictions, pd.Series):
            pos_proba = oof_predictions.values.astype(np.float64)
        elif isinstance(oof_predictions, pd.DataFrame):
            if oof_predictions.shape[1] < 1:
                raise DataError("oof_predictions DataFrame has no columns.")
            # Take the last column (positive class when 2 columns)
            pos_proba = oof_predictions.iloc[:, -1].values.astype(np.float64)
        else:
            raise DataError(
                f"oof_predictions must be pd.Series or pd.DataFrame, "
                f"got {type(oof_predictions).__name__}."
            )
    else:
        # Compute probabilities fresh from data
        proba_df = model.predict_proba(data)
        pos_col = classes[1]
        pos_proba = proba_df[pos_col].values.astype(np.float64)

    # Compute min threshold
    if min_threshold == "auto":
        n_pos = int((y_true == classes[1]).sum())
        min_thresh = max(0.001, 1.0 / n_pos) if n_pos > 0 else 0.001
    else:
        min_thresh = float(min_threshold)

    # Phase 1: coarse scan (0.05 steps from min_thresh to 0.95)
    coarse_thresholds = np.arange(min_thresh, 0.95 + 1e-9, 0.05)
    best_score: float = -np.inf if scorer.greater_is_better else np.inf
    best_thresh: float = float(coarse_thresholds[0]) if len(coarse_thresholds) > 0 else 0.5

    for thresh in coarse_thresholds:
        y_pred = np.where(pos_proba >= thresh, classes[1], classes[0])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            score = scorer(y_true, y_pred)
        if scorer.greater_is_better:
            if score > best_score:
                best_score = score
                best_thresh = float(thresh)
        else:
            if score < best_score:
                best_score = score
                best_thresh = float(thresh)

    # Phase 2: fine scan ±0.1 around coarse best (0.005 steps)
    fine_min = max(min_thresh, best_thresh - 0.1)
    fine_max = min(0.95, best_thresh + 0.1)
    fine_thresholds = np.arange(fine_min, fine_max + 1e-9, 0.005)

    for thresh in fine_thresholds:
        y_pred = np.where(pos_proba >= thresh, classes[1], classes[0])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            score = scorer(y_true, y_pred)
        if scorer.greater_is_better:
            if score > best_score:
                best_score = score
                best_thresh = float(thresh)
        else:
            if score < best_score:
                best_score = score
                best_thresh = float(thresh)

    # Return OptimizeResult — wraps the calibrated model + exposes threshold as float
    from ._types import OptimizeResult
    optimized = copy.copy(model)
    optimized._threshold = best_thresh
    return OptimizeResult(threshold=best_thresh, model=optimized)
