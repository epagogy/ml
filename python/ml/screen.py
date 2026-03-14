"""screen() — quick algorithm screening.

Screen multiple algorithms on your data with defaults. Returns a sorted DataFrame.
Not a fair comparison — just a quick filter to identify candidates for tuning.
The top-ranked algorithm's score is optimistic due to multiple comparison bias.

For fair comparison of tuned models, use ml.compare().
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ._types import CVResult, Leaderboard, SplitResult


_COST_ORDER = [
    "naive_bayes", "decision_tree", "logistic", "knn",
    "histgradient", "random_forest", "lightgbm",
    "xgboost", "catboost", "svm",
    "linear", "elastic_net",
]


def _screen_one_algo(
    algo: str,
    data,
    target: str,
    task: str,
    seed: int,
    metrics,
    keep_models: bool,
    screen_kwargs: dict,
) -> tuple:
    """Fit and score one algorithm. Returns (algo_metrics, model_or_None, elapsed_secs).

    Extracted so both sequential and parallel code paths share one implementation.
    """
    import warnings as _warnings

    import numpy as np

    from ._types import CVResult
    from .evaluate import evaluate
    from .fit import fit

    t0 = time.time()
    algo_model = None
    try:
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=FutureWarning)
            _warnings.filterwarnings("ignore", category=RuntimeWarning)
            _warnings.filterwarnings("ignore", message=".*DataConversion.*")
            _warnings.filterwarnings("ignore", message=".*NaN features.*")
            _warnings.filterwarnings("ignore", message=".*NaN.*")
            _warnings.filterwarnings("ignore", message=".*Auto-scaling.*")
            _warnings.filterwarnings("ignore", message=".*auto-imputed.*")
            _warnings.filterwarnings("ignore", message=".*NaN target.*")
            _warnings.filterwarnings("ignore", message=".*imbalance.*")
            _warnings.filterwarnings("ignore", message=".*same as training data.*")
            if isinstance(data, CVResult):
                algo_model = fit(data, target, algorithm=algo, seed=seed, **screen_kwargs)
                algo_metrics = {
                    k.replace("_mean", ""): v
                    for k, v in algo_model.scores_.items()
                    if "_mean" in k
                }
                if algo_model.fold_scores_ and len(algo_model.fold_scores_) > 1:
                    metric_keys = list(algo_model.fold_scores_[0].keys())
                    for mk in metric_keys:
                        fold_vals = [fs[mk] for fs in algo_model.fold_scores_ if mk in fs]
                        if fold_vals:
                            algo_metrics[f"{mk}_cv_std"] = float(np.std(fold_vals))
                            algo_metrics[f"{mk}_cv_min"] = float(np.min(fold_vals))
                            algo_metrics[f"{mk}_cv_max"] = float(np.max(fold_vals))
            else:
                algo_model = fit(data.train, target, algorithm=algo, seed=seed, **screen_kwargs)
                algo_metrics = dict(evaluate(algo_model, data.valid))

            if metrics:
                from ._scoring import make_scorer as _make_scorer
                from .predict import _predict_impl, _predict_proba
                eval_data = data._data if isinstance(data, CVResult) else data.valid
                y_true_extra = eval_data[target]
                for extra_metric in metrics:
                    if extra_metric in algo_metrics:
                        continue
                    scorer = _make_scorer(extra_metric)
                    try:
                        if scorer.needs_proba:
                            y_p = _predict_proba(algo_model, eval_data).values
                        else:
                            y_p = _predict_impl(algo_model, eval_data).values
                        algo_metrics[extra_metric] = float(scorer(y_true_extra.values, y_p))
                    except Exception:
                        pass
    except Exception as e:
        error_msg = str(e).split("\n")[0][:120]
        algo_metrics = {"error": error_msg}

    elapsed = time.time() - t0
    return algo_metrics, (algo_model if keep_models else None), elapsed


def screen(
    data: SplitResult | CVResult | pd.DataFrame,
    target: str | None = None,
    *,
    algorithms: list[str] | None = None,
    seed: int,
    sort_by: str = "auto",
    metrics: list[str] | None = None,
    time_budget: float | None = None,
    keep_models: bool = True,
    parallel: bool = False,
    engine: str = "auto",
    **kwargs,
) -> Leaderboard:
    """Screen multiple algorithms on your data.

    Fits each algorithm with default hyperparameters and evaluates on validation.
    Returns a DataFrame with algorithm names, metrics, and timing information,
    sorted by performance.

    This is a quick filter — not a fair comparison. Use ml.tune() on the
    top candidates, then ml.compare() to fairly evaluate tuned models.

    .. note:: **Multiple comparison bias.** Picking the best of N algorithms on
       one validation set produces an optimistic estimate — the winner's score is
       inflated by selection. The more algorithms you screen, the larger the bias.
       Always tune the top 2-3 candidates with ml.tune(), then evaluate them with
       ml.compare() on held-out data before trusting the ranking.

    .. note:: **Sort metric matters.** Default sort uses ``roc_auc`` for
       classification. This is threshold-free and ranks models well overall, but
       hides minority-class failures (weighted average) and doesn't reflect
       performance at a specific decision threshold. For imbalanced data, try
       ``sort_by='f1'`` or ``sort_by='f1_macro'``. For deployment at a fixed
       threshold, check precision/recall directly with ml.evaluate().

    Parameters
    ----------
    split : SplitResult or CVResult
        Split data from ml.split(). Pass the split object, not the raw DataFrame.
        Example: s = ml.split(df, "target", seed=42); ml.screen(s, "target", seed=42)
    target : str
        Target column name
    algorithms : list[str], optional
        Algorithms to screen. If None, uses all available for the task.
    seed : int
        Random seed for reproducibility. Required — no default.
    sort_by : str, default="auto"
        Metric to sort by. "auto" selects roc_auc (binary clf), roc_auc_ovr
        (multiclass), or rmse (regression). Use ``sort_by='f1_macro'`` to
        rank by balanced F1 across all classes.
    metrics : list[str], optional
        Additional metrics to compute per algorithm. E.g. ``["roc_auc", "f1", "mcc"]``.
        All metrics are shown in results; ``sort_by`` controls ranking order.
    time_budget : float, optional
        Maximum seconds for the entire screen. Algorithms are tried cheapest-first.
        Screening stops between algorithms (not mid-fit) when budget is exceeded.
        Returns a partial leaderboard with however many algorithms completed.
    keep_models : bool, default=True
        If False, discard fitted models after scoring to save memory.
        Useful for large datasets where 10 models in memory would cause OOM.
    parallel : bool, default=False
        If True, fit algorithms concurrently using up to 4 threads. Threading
        backend shares memory (no pickling). Algorithms that release the GIL
        during fitting (RF, boosting) benefit most. Not compatible with
        time_budget (raises ConfigError). Increases memory usage proportionally
        to the number of concurrent fits.
    **kwargs
        Additional parameters passed to all fit() calls

    Returns
    -------
    pd.DataFrame
        Leaderboard with columns: algorithm, metrics..., time.
        Sorted by performance (descending for most metrics, ascending for error metrics).

    Raises
    ------
    ConfigError
        If split is not a SplitResult or CVResult
    DataError
        If target not found (propagated from fit)

    Examples
    --------
    >>> import ml
    >>> data = ml.dataset("churn")
    >>> s = ml.split(data, "churn", seed=42)
    >>> leaderboard = ml.screen(s, "churn")
    >>> leaderboard.head()
       algorithm  accuracy    f1  roc_auc  time
    0   xgboost      0.87  0.84     0.91   1.2
    1   random_forest 0.85  0.82     0.89   0.8
    2   logistic     0.83  0.80     0.87   0.1

    For imbalanced targets, sort by F1 instead of ROC AUC:

    >>> leaderboard = ml.screen(s, "churn", sort_by="f1")
    """
    from . import _engines
    from ._types import ConfigError, CVResult, Leaderboard, SplitResult

    # Infer target from SplitResult or CVResult when not provided
    if target is None:
        if isinstance(data, CVResult) and data.target is not None:
            target = data.target
        elif isinstance(data, SplitResult) and data._target is not None:
            target = data._target
        elif isinstance(data, pd.DataFrame):
            raise ConfigError(
                "target= is required when passing a raw DataFrame. "
                "Example: ml.screen(df, 'target_col', seed=42)"
            )

    # Convenience: accept raw DataFrame — auto-split internally
    if isinstance(data, pd.DataFrame):
        from .split import split as _split_fn
        data = _split_fn(data, target, seed=seed)

    # parallel=True is incompatible with time_budget (can't stop mid-parallel)
    if parallel and time_budget is not None:
        raise ConfigError(
            "parallel=True is not compatible with time_budget=. "
            "Use one or the other: parallel for speed, time_budget for limits."
        )

    # Validation
    if not isinstance(data, (SplitResult, CVResult)):
        raise ConfigError(
            "screen() requires SplitResult, CVResult, or DataFrame. "
            "Use: ml.screen(df, 'target', seed=42) or ml.screen(ml.split(df, 'target', seed=42), 'target', seed=42)"
        )

    # Detect task from target column (no wasted model fit)
    from ._types import DataError
    from .split import _detect_task

    if isinstance(data, CVResult):
        src = data._data
    else:
        src = data.train

    if target not in src.columns:
        raise DataError(
            f"target='{target}' not found in data. "
            f"Available columns: {list(src.columns)}"
        )
    task = _detect_task(src[target])

    # Validate algorithms type
    if algorithms is not None and isinstance(algorithms, str):
        raise ConfigError(
            f"algorithms= must be a list, not a string. "
            f"Use: algorithms=['{algorithms}']"
        )

    # Get available algorithms
    if algorithms is None:

        # Get all algorithms for this task
        all_algos = _engines.available()
        # Filter by task
        algorithms_to_try = []
        for algo in all_algos:
            if algo == "auto":
                continue
            # Filter by task compatibility
            if task == "classification" and algo in ("linear", "elastic_net"):
                continue
            if task == "regression" and algo in ("logistic", "naive_bayes"):
                continue
            # Skip optional deps that aren't installed
            if algo == "catboost":
                try:
                    import catboost  # noqa: F401
                except ImportError:
                    continue
            if algo == "lightgbm":
                try:
                    import lightgbm  # noqa: F401
                except ImportError:
                    continue
            algorithms_to_try.append(algo)

        # Cost-frugal ordering: cheapest algorithms first for maximum signal per second
        cost_rank = {a: i for i, a in enumerate(_COST_ORDER)}
        algorithms_to_try.sort(key=lambda a: cost_rank.get(a, len(_COST_ORDER)))
    else:
        algorithms_to_try = algorithms

    # Validate extra metrics list
    if metrics is not None:
        from ._scoring import make_scorer as _make_scorer
        for m in metrics:
            _make_scorer(m)  # raises ConfigError for unknown names

    # Warn when dataset is too small for a reliable screen
    n_algos = len(algorithms_to_try)
    n_rows = len(src)
    if n_rows < n_algos * 20:
        warnings.warn(
            f"Screening {n_algos} algorithms on only {n_rows} rows. "
            f"With <{n_algos * 20} rows, performance differences may be noise. "
            "Consider reducing the algorithm list or using folds= for more stable estimates.",
            UserWarning,
            stacklevel=2,
        )

    # Screen each algorithm (NaN auto-imputed for non-tree algorithms)

    results = []
    fitted_models: list = []  # parallel to results — None for failed algorithms
    screen_start = time.time()

    if parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        n_workers = min(4, len(algorithms_to_try))
        futures_map: dict = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for algo in algorithms_to_try:
                screen_kwargs = (
                    {"n_jobs": _engines._screen_n_jobs(), "engine": engine, **kwargs}
                    if algo in _engines.PARALLEL_ALGORITHMS
                    else {"engine": engine, **kwargs}
                )
                fut = pool.submit(
                    _screen_one_algo,
                    algo, data, target, task, seed, metrics, keep_models, screen_kwargs,
                )
                futures_map[fut] = algo
            # Collect in completion order; store in algo order for consistent output
            algo_results: dict = {}
            for n_done, fut in enumerate(as_completed(futures_map), 1):
                algo = futures_map[fut]
                algo_metrics, algo_model, elapsed = fut.result()
                algo_results[algo] = (algo_metrics, algo_model, elapsed)
                print(
                    f"  [{n_done}/{len(algorithms_to_try)}] {algo}: "
                    f"{elapsed:.1f}s",
                    flush=True,
                )
        # Reassemble in original cost order
        for algo in algorithms_to_try:
            algo_metrics, algo_model, elapsed = algo_results[algo]
            fitted_models.append(algo_model)
            results.append({"algorithm": algo, **algo_metrics, "time_seconds": round(elapsed, 2)})
    else:
        for i, algo in enumerate(algorithms_to_try):
            # Time budget: stop between algorithms (not mid-fit)
            if time_budget is not None and (time.time() - screen_start) > time_budget:
                break
            print(
                f"  [{i + 1}/{len(algorithms_to_try)}] {algo}...",
                end=" ", flush=True,
            )
            screen_kwargs = (
                {"n_jobs": _engines._screen_n_jobs(), "engine": engine, **kwargs}
                if algo in _engines.PARALLEL_ALGORITHMS
                else {"engine": engine, **kwargs}
            )
            algo_metrics, algo_model, elapsed = _screen_one_algo(
                algo, data, target, task, seed, metrics, keep_models, screen_kwargs,
            )
            print(f"{elapsed:.1f}s", flush=True)
            fitted_models.append(algo_model)
            results.append({"algorithm": algo, **algo_metrics, "time_seconds": round(elapsed, 2)})

    # Warn about failed algorithms (Principle 4: Fail Loudly)
    n_failed = sum(1 for r in results if "error" in r)
    if n_failed > 0:
        n_total = len(results)
        failed_names = [r["algorithm"] for r in results if "error" in r]
        warnings.warn(
            f"{n_failed}/{n_total} algorithms failed: {failed_names}. "
            "See 'error' column in results for details.",
            UserWarning,
            stacklevel=2,
        )

    # Build DataFrame
    df = pd.DataFrame(results)

    # Sort by metric
    if sort_by == "auto":
        # Auto-select sort column
        if task == "classification":
            # Prefer roc_auc, fallback to accuracy
            if "roc_auc" in df.columns:
                sort_col = "roc_auc"
            elif "roc_auc_ovr" in df.columns:
                sort_col = "roc_auc_ovr"
            elif "accuracy" in df.columns:
                sort_col = "accuracy"
            else:
                # No metrics available, don't sort
                return df
        else:  # regression
            # Prefer rmse
            if "rmse" in df.columns:
                sort_col = "rmse"
            elif "mae" in df.columns:
                sort_col = "mae"
            else:
                # No metrics available, don't sort
                return df
    else:
        sort_col = sort_by

    # Sort: ascending for error metrics (rmse, mae, mape, smape, rmsle, log_cosh),
    # descending for everything else
    _error_metrics = {"rmse", "mae", "mape", "smape", "rmsle", "log_cosh"}
    sorted_models = fitted_models  # default: unsorted
    if sort_col in df.columns:
        ascending = sort_col in _error_metrics
        sort_idx = df.sort_values(sort_col, ascending=ascending).index.tolist()
        df = df.loc[sort_idx].reset_index(drop=True)
        sorted_models = [fitted_models[i] for i in sort_idx]
    elif sort_by != "auto":
        metric_cols = [c for c in df.columns
                       if c not in ("algorithm", "error", "time_seconds")
                       and not c.endswith(("_cv_std", "_cv_min", "_cv_max"))]
        raise ConfigError(
            f"sort_by={sort_by!r} not found in results. "
            f"Available metrics: {metric_cols}"
        )

    # Round numeric columns for clean display
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(4)

    n = len(algorithms_to_try)
    return Leaderboard(df, title=f"Screen [{task} · {n} algorithms]", models=sorted_models)
