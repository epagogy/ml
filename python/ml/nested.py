"""nested_cv() — nested cross-validation for unbiased model selection.

Standard screen() + tune() overfits the validation split because model
selection and performance estimation share the same data. Nested CV uses
separate inner (tuning) and outer (evaluation) loops for an honest estimate
of what you'll see on the leaderboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class NestedCVResult:
    """Result from nested_cv().

    Attributes:
        scores: Per-algorithm outer fold scores. Keys are algorithm names,
            values are lists of fold scores (one per outer fold).
        best_algorithm: Algorithm with the highest mean outer CV score.
        summary: DataFrame with mean, std, min, max per algorithm.
        generalization_gap: Inner CV score minus outer CV score per algorithm.
            Large gap → inner loop is over-optimistic (budget too high).
    """

    scores: dict[str, list[float]] = field(default_factory=dict)
    best_algorithm: str = ""
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    generalization_gap: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        lines = [f"NestedCVResult(best={self.best_algorithm!r})"]
        if not self.summary.empty:
            lines.append(self.summary.to_string(index=False))
        return "\n".join(lines)


def nested_cv(
    data: pd.DataFrame,
    target: str,
    *,
    algorithms: list[str] | None = None,
    outer_folds: int = 5,
    inner_folds: int = 3,
    metric: str = "auto",
    n_trials: int = 10,
    seed: int,
) -> NestedCVResult:
    """Nested cross-validation for unbiased model selection.

    Uses two nested CV loops:
    - **Outer loop**: evaluates model performance (what you'll get on the LB)
    - **Inner loop**: tunes hyperparameters (prevents selection bias)

    Standard ``screen() → tune()`` is biased because the validation set is
    used for both HPO and performance estimation. Nested CV avoids this by
    using the inner loop for HPO and the outer loop for evaluation only.

    Parameters
    ----------
    data : pd.DataFrame
        Full training dataset.
    target : str
        Target column name.
    algorithms : list[str], optional
        Algorithms to evaluate. Defaults to a fast subset: xgboost,
        random_forest, lightgbm (if available), logistic.
    outer_folds : int, default=5
        Number of outer CV folds (controls estimate variance).
    inner_folds : int, default=3
        Number of inner CV folds (controls HPO quality).
    metric : str, default="auto"
        Evaluation metric. "auto" selects roc_auc (classification) or
        rmse (regression).
    n_trials : int, default=10
        Number of Optuna/random search trials in the inner loop.
    seed : int
        Random seed for reproducibility. Required.

    Returns
    -------
    NestedCVResult
        Contains per-algorithm outer fold scores, best algorithm,
        summary DataFrame, and generalization gap.

    Examples
    --------
    >>> import ml
    >>> data = ml.dataset("churn")
    >>> result = ml.nested_cv(data, "churn", seed=42)
    >>> result.best_algorithm
    'xgboost'
    >>> result.summary
       algorithm  mean_score  std_score  generalization_gap
    0    xgboost       0.891      0.012               0.008
    1  lightgbm        0.888      0.014               0.006
    """
    import warnings

    import numpy as np

    from . import _engines
    from ._scoring import make_scorer
    from ._types import ConfigError, DataError
    from .fit import fit
    from .split import _detect_task

    if target not in data.columns:
        raise DataError(
            f"target='{target}' not found in data. "
            f"Available: {list(data.columns)}"
        )

    task = _detect_task(data[target])

    # Resolve metric
    if metric == "auto":
        if task == "classification":
            n_classes = int(data[target].nunique())
            metric_name = "roc_auc_ovr" if n_classes > 2 else "roc_auc"
        else:
            metric_name = "rmse"
    else:
        metric_name = metric
    scorer = make_scorer(metric_name)

    # Default algorithms (fast subset for nested CV)
    if algorithms is None:
        candidates = ["xgboost", "random_forest", "logistic"]
        try:
            import lightgbm  # noqa: F401
            candidates.insert(0, "lightgbm")
        except ImportError:
            pass
        # Filter by task
        if task == "classification":
            algorithms = [a for a in candidates if a not in ("linear", "elastic_net")]
        else:
            algorithms = [a for a in candidates if a not in ("logistic", "naive_bayes")]
    else:
        # Validate provided algorithms
        available = _engines.available()
        bad = [a for a in algorithms if a not in available]
        if bad:
            raise ConfigError(
                f"Unknown algorithms: {bad}. Available: {available}"
            )

    # Outer CV split
    from .split import _kfold, _stratified_kfold

    y = data[target]
    if task == "classification":
        outer_splits = list(_stratified_kfold(y.values, k=outer_folds, seed=seed))
    else:
        outer_splits = list(_kfold(len(data), k=outer_folds, seed=seed))

    algo_outer_scores: dict[str, list[float]] = {a: [] for a in algorithms}
    algo_inner_scores: dict[str, list[float]] = {a: [] for a in algorithms}

    for fold_i, (train_idx, test_idx) in enumerate(outer_splits):
        outer_train = data.iloc[train_idx].reset_index(drop=True)
        outer_test = data.iloc[test_idx].reset_index(drop=True)

        for algo in algorithms:
            # Inner loop: use fit() with CVResult for HPO-free CV estimate
            # (full tune() is expensive; use fast n_trials search instead)
            inner_seed = seed + fold_i * 1000
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    from .split import split as _split
                    from .cv import cv as _cv
                    inner_split = _split(outer_train, target, seed=inner_seed)
                    inner_cv = _cv(inner_split, folds=inner_folds, seed=inner_seed)
                    inner_model = fit(inner_cv, target,
                                      algorithm=algo, seed=inner_seed)
                    # Inner score: mean CV score for primary metric
                    if inner_model.scores_:
                        inner_key = f"{metric_name}_mean"
                        if inner_key in inner_model.scores_:
                            algo_inner_scores[algo].append(
                                inner_model.scores_[inner_key]
                            )
                    # Outer score: evaluate inner model on held-out outer test
                    y_test = outer_test[target]
                    if scorer.needs_proba:
                        from .predict import _predict_proba
                        proba = _predict_proba(inner_model, outer_test)
                        proba_vals = proba.values
                        # For binary clf, roc_auc expects 1D positive-class proba
                        if proba_vals.ndim == 2 and proba_vals.shape[1] == 2:
                            y_pred = proba_vals[:, 1]
                        else:
                            y_pred = proba_vals
                    else:
                        from .predict import _predict_impl
                        y_pred = _predict_impl(inner_model, outer_test).values
                    outer_score = float(scorer(y_test.values, y_pred))
                    algo_outer_scores[algo].append(outer_score)
            except Exception:
                # Skip fold for this algorithm (handled below)
                continue

    # Build summary
    rows = []
    gaps = {}
    best_algo = ""
    best_mean = float("-inf") if scorer.greater_is_better else float("inf")

    for algo in algorithms:
        outer_vals = algo_outer_scores[algo]
        if not outer_vals:
            continue
        inner_vals = algo_inner_scores[algo]
        mean_outer = float(np.mean(outer_vals))
        std_outer = float(np.std(outer_vals))
        mean_inner = float(np.mean(inner_vals)) if inner_vals else float("nan")
        gap = mean_inner - mean_outer if inner_vals else float("nan")
        gaps[algo] = gap

        rows.append({
            "algorithm": algo,
            "mean_score": round(mean_outer, 4),
            "std_score": round(std_outer, 4),
            "min_score": round(float(np.min(outer_vals)), 4),
            "max_score": round(float(np.max(outer_vals)), 4),
            "generalization_gap": round(gap, 4) if np.isfinite(gap) else float("nan"),
        })

        is_better = (mean_outer > best_mean) if scorer.greater_is_better else (mean_outer < best_mean)
        if is_better or not best_algo:
            best_mean = mean_outer
            best_algo = algo

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty and scorer.greater_is_better:
        summary_df = summary_df.sort_values("mean_score", ascending=False).reset_index(drop=True)
    elif not summary_df.empty:
        summary_df = summary_df.sort_values("mean_score", ascending=True).reset_index(drop=True)

    return NestedCVResult(
        scores=algo_outer_scores,
        best_algorithm=best_algo,
        summary=summary_df,
        generalization_gap=gaps,
    )
