"""enough() — learning curve analysis.

Answers "do I need more data?". Trains at increasing data sizes and reports
train vs validation performance at each step.

Usage:
    >>> result = ml.enough(data, "churn", seed=42)
    >>> result.saturated       # True if more data won't help
    >>> result.curve           # DataFrame: n_samples, train_score, val_score
    >>> result.recommendation  # "Collect more data" or "Model is saturated"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EnoughResult:
    """Result of learning curve analysis.

    Attributes
    ----------
    saturated : bool
        True if adding more data is unlikely to improve performance (< 1%
        gain over last half of the curve).
    curve : pd.DataFrame
        Learning curve data. Columns: n_samples, train_score, val_score.
        Scores are the primary metric (accuracy for classification, r2 for
        regression).
    metric : str
        Name of the metric used (e.g., "accuracy", "r2").
    n_current : int
        Number of training samples in the full dataset.
    recommendation : str
        Human-readable action recommendation.
    """

    saturated: bool
    curve: pd.DataFrame
    metric: str
    n_current: int
    recommendation: str

    def __repr__(self) -> str:
        sat_str = "saturated" if self.saturated else "still learning"
        last_score = self.curve["val_score"].iloc[-1] if len(self.curve) > 0 else float("nan")
        return (
            f"EnoughResult({sat_str}, {self.metric}={last_score:.3f} "
            f"at n={self.n_current})"
        )


def enough(
    data: pd.DataFrame,
    target: str,
    *,
    seed: int,
    algorithm: str = "auto",
    steps: int = 8,
    cv: int = 3,
) -> EnoughResult:
    """Analyze whether more data would improve model performance.

    Trains the model at increasing fractions of the data and measures
    train vs validation performance. The resulting learning curve shows
    whether the model is still learning (more data helps) or saturated
    (more data unlikely to help).

    Parameters
    ----------
    data : pd.DataFrame
        Full labeled dataset including target column.
    target : str
        Name of the target column.
    seed : int
        Random seed for reproducible splits (required, no default).
    algorithm : str, default="auto"
        Algorithm to use. "auto" selects based on task detection.
        Any algorithm supported by ml.fit() works here.
    steps : int, default=8
        Number of data-size steps to evaluate (evenly spaced from
        10% to 100% of training data).
    cv : int, default=3
        Number of cross-validation folds for validation score.

    Returns
    -------
    EnoughResult
        - ``.saturated``: True if curve plateaus
        - ``.curve``: DataFrame with n_samples, train_score, val_score
        - ``.recommendation``: action to take

    Raises
    ------
    DataError
        If data has fewer than 50 rows or target is missing.
    ConfigError
        If steps < 2 or cv < 2.

    Examples
    --------
    >>> result = ml.enough(data, "churn", seed=42)
    >>> result.saturated
    False
    >>> result.recommendation
    'Still learning: val accuracy improved 4.2% in last 3 steps. Collect more data.'
    >>> result.curve
       n_samples  train_score  val_score
    0         50        0.912      0.731
    1        100        0.898      0.778
    2        200        0.887      0.812
    ...
    """
    from ._compat import to_pandas
    from ._types import ConfigError, DataError
    from .evaluate import evaluate
    from .fit import fit
    from .split import _detect_task
    data = to_pandas(data)
    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"data= must be a DataFrame, got {type(data).__name__}."
        )

    if target not in data.columns:
        raise DataError(
            f"Target '{target}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )

    if len(data) < 50:
        raise DataError(
            f"enough() requires at least 50 rows, got {len(data)}. "
            "Collect more data or use ml.validate() for small datasets."
        )

    if steps < 2:
        raise ConfigError(f"steps must be >= 2, got {steps}.")

    if cv < 2:
        raise ConfigError(f"cv must be >= 2, got {cv}.")

    task = _detect_task(data[target])
    metric = "accuracy" if task == "classification" else "r2"

    # Shuffle data for reproducible random subsampling
    shuffled = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_total = len(shuffled)
    # Steps from ~10% to 100%, at least 20 samples minimum per step
    min_n = max(20, n_total // (steps * 2))
    fractions = np.linspace(min_n, n_total, steps, dtype=int)
    fractions = sorted(set(fractions.tolist()))  # deduplicate

    records = []

    for n in fractions:
        subset = shuffled.iloc[:n]

        # Train score: fit on subset, evaluate on same subset (no CV)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_model = fit(
                    data=subset,
                    target=target,
                    algorithm=algorithm,
                    seed=seed,
                )
                train_eval = evaluate(train_model, data=subset)
            train_score = float(train_eval.get(metric, 0.0))
        except Exception:
            train_score = float("nan")

        # Validation score: k-fold CV on subset
        try:
            from .split import _detect_task, _kfold, _stratified_kfold

            X = subset.drop(columns=[target])
            y = subset[target]

            ttype = _detect_task(y)
            n_folds = min(cv, len(subset) // 2)
            if ttype == "classification":
                cv_iter = _stratified_kfold(y.values, k=n_folds, seed=seed)
            else:
                cv_iter = _kfold(len(X), k=n_folds, seed=seed)

            fold_scores = []
            for train_idx, val_idx in cv_iter:
                fold_train = subset.iloc[train_idx]
                fold_val = subset.iloc[val_idx]
                if len(fold_train) < 5 or len(fold_val) < 2:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fold_model = fit(
                        data=fold_train,
                        target=target,
                        algorithm=algorithm,
                        seed=seed,
                    )
                fold_eval = evaluate(fold_model, data=fold_val)
                fold_scores.append(float(fold_eval.get(metric, 0.0)))

            val_score = float(np.mean(fold_scores)) if fold_scores else float("nan")
        except Exception:
            val_score = float("nan")

        records.append({
            "n_samples": int(n),
            "train_score": round(train_score, 4),
            "val_score": round(val_score, 4),
        })

    curve = pd.DataFrame(records)

    # Saturation detection: val_score improvement in last half of steps
    val_scores = curve["val_score"].dropna().values
    saturated = False
    gain_str = ""

    if len(val_scores) >= 4:
        # Compare first half max vs second half max
        mid = len(val_scores) // 2
        first_half_max = float(np.nanmax(val_scores[:mid]))
        second_half_max = float(np.nanmax(val_scores[mid:]))

        if metric in {"accuracy", "r2", "f1", "roc_auc"}:
            # Higher is better
            gain = second_half_max - first_half_max
        else:
            # Lower is better (rmse, mae)
            gain = first_half_max - second_half_max

        # F1: don't use abs() — negative gain means degrading curve, not "still learning"
        # gain < 0: model is getting worse with more data → saturated/degrading
        # 0 <= gain < 1%: plateau → saturated
        # gain >= 1%: still improving → not saturated
        gain_pct = max(gain, 0.0) * 100
        gain_str = f"{gain_pct:.1f}%"
        saturated = gain <= 0 or gain_pct < 1.0

    last_val = float(val_scores[-1]) if len(val_scores) > 0 else float("nan")
    last_n = int(curve["n_samples"].iloc[-1])

    if saturated:
        recommendation = (
            f"Model is saturated: {metric} improved < 1% adding more data. "
            "Focus on feature engineering or a more powerful algorithm instead."
        )
    elif len(val_scores) < 4:
        recommendation = (
            "Insufficient data to determine saturation. "
            "Collect more labeled examples."
        )
    else:
        recommendation = (
            f"Still learning: {metric} improved {gain_str} in last half of data. "
            f"More data likely helps. Current {metric}={last_val:.3f} at n={last_n}."
        )

    return EnoughResult(
        saturated=saturated,
        curve=curve,
        metric=metric,
        n_current=n_total,
        recommendation=recommendation,
    )
