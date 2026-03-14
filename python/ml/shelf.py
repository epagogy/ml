"""shelf() — label-required model freshness check.

Compares a model's current performance on new labeled data against its
original training-time performance. Detects model degradation when ground
truth labels become available.

Usage:
    >>> result = ml.shelf(model, new=labeled_batch, target="churn")
    >>> result.fresh                    # True
    >>> result.metrics_then             # {"accuracy": 0.88, "f1": 0.84}
    >>> result.metrics_now              # {"accuracy": 0.85, "f1": 0.81}
    >>> result.degradation              # {"accuracy": -0.03, "f1": -0.03}
    >>> result.recommendation           # "Model stable. Minor degradation within tolerance."
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ._types import Model, TuningResult


@dataclass
class ShelfResult:
    """Result of a model freshness check.

    Attributes
    ----------
    fresh : bool
        True if model is still performing within tolerance of original metrics.
    metrics_then : dict[str, float]
        Metrics at training time (from model.scores_).
    metrics_now : dict[str, float]
        Current metrics on the new labeled batch.
    degradation : dict[str, float]
        Per-metric change (metrics_now - metrics_then). Negative = worse.
    recommendation : str
        Human-readable summary and action recommendation.
    n_new : int
        Number of rows in the new labeled dataset.
    tolerance : float
        Allowed degradation threshold used (default 0.05).
    """

    fresh: bool | None
    metrics_then: dict[str, float]
    metrics_now: dict[str, float]
    degradation: dict[str, float]
    recommendation: str
    n_new: int
    tolerance: float = 0.05

    @property
    def stale(self) -> bool | None:
        """True if model is no longer performing within tolerance. Inverse of fresh."""
        return not self.fresh if self.fresh is not None else None

    @property
    def past_shelf_life(self) -> bool | None:
        """True if model should be retrained. Alias for stale."""
        return self.stale

    def __repr__(self) -> str:
        fresh_str = f"fresh={self.fresh}"
        rec_short = self.recommendation[:60] + "..." if len(self.recommendation) > 60 else self.recommendation
        return f"ShelfResult({fresh_str}, '{rec_short}')"


def shelf(
    model: Model | TuningResult,
    *,
    new: pd.DataFrame,
    target: str,
    tolerance: float = 0.05,
) -> ShelfResult:
    """Check if a model is past its shelf life.

    Evaluates the model on new labeled data and compares performance to
    the model's original training metrics. Requires ground truth labels.

    Run this when outcome labels become available (daily/weekly batch
    scoring → wait for outcomes → run shelf()). Pair with drift() for a
    complete monitoring strategy:
    - drift() detects input distribution shift (label-free, run always)
    - shelf() detects performance degradation (needs labels, run periodically)

    Parameters
    ----------
    model : Model or TuningResult
        Fitted model with training scores (from ml.fit() or ml.tune()).
    new : pd.DataFrame
        New labeled dataset including the target column.
    target : str
        Name of the target column in new.
    tolerance : float, default=0.05
        Allowed degradation per metric. Degradation beyond tolerance on
        any key metric marks the model as stale.

    Returns
    -------
    ShelfResult
        - ``.fresh``: True if model performance is within tolerance
        - ``.metrics_then``: Original training metrics
        - ``.metrics_now``: Current metrics on new data
        - ``.degradation``: Per-metric delta (negative = worse)
        - ``.recommendation``: Human-readable guidance

    Raises
    ------
    DataError
        If new is not a DataFrame, target column is missing, or n < 5 rows.
    ModelError
        If model has no training metrics (scores_) to compare against.
    ConfigError
        If target doesn't match model's trained target.

    Examples
    --------
    >>> result = ml.shelf(model, new=labeled_batch, target="churn")
    >>> result.fresh
    True
    >>> result.degradation
    {'accuracy': -0.02, 'f1': -0.01}
    >>> result.recommendation
    'Model stable. Minor degradation within tolerance.'

    Stale model:
    >>> result.fresh
    False
    >>> result.recommendation
    'Model stale: accuracy degraded by 0.12 (tolerance 0.05). Retrain recommended.'
    """
    import pandas as pd

    from ._types import ConfigError, DataError
    from .evaluate import evaluate

    # F3: unwrap TuningResult — shelf() was missing this, causing AttributeError
    if isinstance(model, TuningResult):
        model = model.best_model

    if not isinstance(new, pd.DataFrame):
        raise DataError(
            f"shelf() new must be a DataFrame, got {type(new).__name__}."
        )
    if target not in new.columns:
        raise DataError(
            f"Target column '{target}' not found in new data. "
            f"Available columns: {list(new.columns)}"
        )
    if len(new) < 5:
        raise DataError(
            f"shelf() requires at least 5 labeled rows, got {len(new)}. "
            "Collect more labeled data before running shelf()."
        )

    if target != model._target:
        raise ConfigError(
            f"Target mismatch: model was trained on '{model._target}', "
            f"but shelf() called with target='{target}'. "
            "Ensure you're using the same target column."
        )

    # Get original training metrics
    # scores_ is only populated for CV fits (folds=k). Holdout fits have scores_=None.
    # CV scores have _mean/_std suffixes — normalize to plain metric names using _mean only.
    has_prior_metrics = bool(model.scores_)
    if has_prior_metrics:
        metrics_then = {
            k.replace("_mean", ""): float(v)
            for k, v in model.scores_.items()
            if k.endswith("_mean")
        }
        if not metrics_then:
            # Fallback: scores_ exists but no _mean keys (shouldn't happen)
            metrics_then = {k: float(v) for k, v in model.scores_.items()}
    elif model.cv_score is not None:
        # Holdout fit: cv_score is a single float (accuracy for clf, r2 for reg).
        # Use it as a single-metric baseline so shelf() can detect degradation.
        # W51-F1: Without this, metrics_then == {} and degradation detection is
        # silently disabled for the canonical ml.fit(s.train, target, seed=42) workflow.
        primary = "accuracy" if model._task == "classification" else "r2"
        metrics_then = {primary: float(model.cv_score)}
        has_prior_metrics = True
    else:
        metrics_then = {}

    # Evaluate on new labeled data
    metrics_now_raw = evaluate(model=model, data=new)
    metrics_now = dict(metrics_now_raw)

    # Compute degradation on shared metrics (lower is worse for accuracy/f1/r2;
    # higher is worse for rmse/mae)
    _lower_is_better = {"rmse", "mae"}
    degradation: dict[str, float] = {}
    for metric in metrics_then:
        if metric in metrics_now:
            degradation[metric] = float(metrics_now[metric] - metrics_then[metric])

    # Freshness: any key metric degraded beyond tolerance?
    # Key metrics: primary score per task
    _key_metrics = {"accuracy", "f1", "roc_auc", "rmse", "mae", "r2",
                    "f1_weighted", "roc_auc_ovr"}

    worst_degradation: float = 0.0
    worst_metric: str = ""
    for metric, delta in degradation.items():
        if metric not in _key_metrics:
            continue
        if metric in _lower_is_better:
            # Positive delta = worse (error went up)
            if delta > worst_degradation:
                worst_degradation = delta
                worst_metric = metric
        else:
            # Negative delta = worse (score went down)
            if -delta > worst_degradation:
                worst_degradation = -delta
                worst_metric = metric

    # If no prior metrics available (holdout fit, no CV scores):
    # The model was just fitted — it has not had a chance to degrade yet.
    # Return fresh=True with a note that no historical comparison was possible.
    # This makes shelf() useful in the canonical workflow: fit(s.train) → shelf(new).
    # W30-F6: returning fresh=None made shelf() unusable for standard holdout fits.
    if not has_prior_metrics:
        fresh = True
        metrics_summary = ", ".join(
            f"{k}={v:.3f}" for k, v in list(metrics_now.items())[:4]
        )
        recommendation = (
            f"Model freshly deployed (no CV baseline available — fitted without folds). "
            f"Current metrics on {len(new)} labeled rows: {metrics_summary}. "
            "To enable degradation comparison in future shelf() calls, refit with CV: "
            "cv = ml.split(data, target, seed=seed, folds=5); ml.fit(cv, target, seed=seed)."
        )
    elif worst_degradation == 0.0:
        fresh = True
        recommendation = "Model stable. No degradation detected."
    elif worst_degradation <= tolerance:
        fresh = True
        recommendation = (
            f"Model stable. Minor degradation within tolerance "
            f"(worst: {worst_metric} degraded by {worst_degradation:.3f}, tolerance {tolerance})."
        )
    else:
        fresh = False
        recommendation = (
            f"Model stale: {worst_metric} degraded by {worst_degradation:.3f} "
            f"(tolerance {tolerance}). Retrain recommended."
        )

    return ShelfResult(
        fresh=fresh,
        metrics_then=metrics_then,
        metrics_now=metrics_now,
        degradation=degradation,
        recommendation=recommendation,
        n_new=len(new),
        tolerance=tolerance,
    )
