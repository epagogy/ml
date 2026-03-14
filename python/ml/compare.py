"""compare() — fair comparison of pre-fitted models.

Evaluate multiple pre-fitted/tuned models on the same data.
Unlike screen(), this does NOT fit anything — it only evaluates.

Use screen() first to identify candidates, tune() to optimize them,
then compare() to fairly evaluate the tuned models side by side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ._types import Leaderboard

# Module-level set to track unique warning contexts.
# Throttles the peeking warning to once per unique (n_rows, n_cols, columns) context.
# Prevents 20 identical warnings in a dev loop while letting tests with different
# data shapes see the warning. W30-F4.
_warned_contexts: set[tuple] = set()


def compare(
    *args,
    data: pd.DataFrame | None = None,
    sort_by: str = "auto",
    metric: str | callable | None = None,
    warn_test: bool = True,
) -> Leaderboard:
    """Compare pre-fitted models on the same data.

    Evaluates each model and returns a sorted leaderboard.
    No fitting happens — models must already be trained.

    Args:
        *args: Pre-fitted models to compare. Pass as list or individual args:
            ``ml.compare([m1, m2], data=s.valid)`` or ``ml.compare(m1, m2, data=s.valid)``
        data: Evaluation data (e.g., s.valid or s.test). Must contain the target column.
        sort_by: Metric to sort by. "auto" selects roc_auc (clf) or rmse (reg).
        metric: Custom metric callable or string name (A10 — Conort C2).
            Accepts same interface as tune(metric=). When provided, the custom
            metric is computed alongside standard metrics and used for sorting.
            Example: ``metric=my_gini`` or ``metric="f1_weighted"``.
        warn_test: When True (default), emits a UserWarning reminding that
            using compare() for model selection on test data constitutes
            implicit data peeking. Set warn_test=False when evaluating on
            validation data or to suppress the reminder.

    Returns:
        Leaderboard with columns: algorithm, metrics... Sorted by performance.

    Raises:
        ConfigError: If models is empty or not a list of Model objects
        DataError: If target column not found in data

    Example:
        >>> import ml
        >>> data = ml.dataset("churn")
        >>> s = ml.split(data, "churn", seed=42)
        >>> tuned_xgb = ml.tune(s.train, "churn", algorithm="xgboost", seed=42)
        >>> tuned_rf = ml.tune(s.train, "churn", algorithm="random_forest", seed=42)
        >>> leaderboard = ml.compare([tuned_xgb, tuned_rf], s.valid)
    """
    import warnings

    from ._compat import to_pandas
    from ._types import ConfigError, DataError, Leaderboard, TuningResult
    from ._types import Model as ModelType
    from .evaluate import evaluate

    # A10 warning — emitted after data is resolved from positional/keyword args (see below)

    # A10: Custom metric scorer
    custom_scorer = None
    if metric is not None:
        from ._scoring import make_scorer
        custom_scorer = make_scorer(metric)

    # Auto-convert Polars/other DataFrames to pandas
    if data is not None:
        data = to_pandas(data)

    # Parse flexible args: compare([m1,m2], data) OR compare(m1, m2, data=...)
    models: list = []
    _data = data
    for arg in args:
        if isinstance(arg, list):
            models.extend(arg)
        elif isinstance(arg, pd.DataFrame):
            if _data is not None:
                raise ConfigError(
                    "compare() got data as both positional and keyword argument. "
                    "Use: ml.compare([m1, m2], data=s.valid) or ml.compare([m1, m2], s.valid)"
                )
            _data = arg
        elif isinstance(arg, (ModelType, TuningResult)):
            models.append(arg)
        else:
            raise ConfigError(
                f"compare() unexpected argument type: {type(arg).__name__}. "
                "Pass Model/TuningResult objects and a DataFrame."
            )
    data = _data

    # A10: Warn about potential test-data peeking — throttled per session.
    # Uses first-row values for robust dedup. Runs AFTER positional arg parsing
    # so data is always resolved (W30-F4).
    if warn_test:
        try:
            if data is not None:
                _cols = tuple(data.columns.tolist())
                _first = tuple(data.iloc[0].values.tolist()) if len(data) > 0 else ()
                _ctx: tuple = (data.shape[1], _cols, _first)
            else:
                _ctx = (None,)
            _should_warn = _ctx not in _warned_contexts
            if _should_warn:
                _warned_contexts.add(_ctx)
        except Exception:
            _should_warn = True
        if _should_warn:
            warnings.warn(
                "compare() evaluates all models on the same data. "
                "If this is held-out test data, comparing multiple models constitutes "
                "implicit model selection on the test set — choose one model before "
                "evaluating on test, or use ml.assess() for a single final model.",
                UserWarning,
                stacklevel=2,
            )

    # Validation
    if len(models) == 0:
        raise ConfigError(
            "compare() requires at least one pre-fitted model. "
            "Use: ml.compare([model_a, model_b], data=s.valid)"
        )

    # Auto-unwrap TuningResult → best_model, but remember provenance
    unwrapped = []
    is_tuned = []
    for i, m in enumerate(models):
        if isinstance(m, TuningResult):
            unwrapped.append(m.best_model)
            is_tuned.append(True)
        elif isinstance(m, ModelType):
            unwrapped.append(m)
            is_tuned.append(False)
        else:
            raise ConfigError(
                f"compare() item {i} is not a Model or TuningResult. "
                f"Got {type(m).__name__}. Use ml.fit() or ml.tune() first."
            )
    models = unwrapped


    if not isinstance(data, pd.DataFrame):
        raise ConfigError(
            f"compare() requires a DataFrame for evaluation data. "
            f"Got {type(data).__name__}. "
            "Use: ml.compare([model_a, model_b], s.valid)"
        )

    # Partition guard — warn (not error) when test-tagged data is used
    from ._provenance import identify_partition
    _role = identify_partition(data)
    if _role == "test":
        warnings.warn(
            "compare() received test-tagged data. Comparing models on test data "
            "constitutes implicit model selection on the test set. "
            "Use validation data for comparison: ml.compare([m1, m2], data=s.valid). "
            "Reserve test data for ml.assess().",
            UserWarning,
            stacklevel=2,
        )

    # Detect task and target from first model, validate consistency
    task = models[0]._task
    target = models[0]._target

    for i, m in enumerate(models[1:], 1):
        if m._task != task:
            raise ConfigError(
                f"compare() item {i} is {m._task} but item 0 is {task}. "
                "Cannot compare classifiers with regressors."
            )
        if m._target != target:
            raise ConfigError(
                f"compare() item {i} has target={m._target!r} but item 0 "
                f"has target={target!r}. All models must share the same target."
            )

    # Check target exists
    if target not in data.columns:
        raise DataError(
            f"target='{target}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )

    # Evaluate each model
    import time as _time

    all_preds = []   # store predictions per model for significance testing
    results = []
    for i, model in enumerate(models):
        try:
            import warnings as _w
            t0 = _time.perf_counter()
            with _w.catch_warnings():
                # Suppress false-positive "same as training data" warning in compare():
                # compare() is explicitly designed to evaluate on held-out data, and the
                # heuristic (len(data) == n_train) triggers false positives when val size
                # happens to equal train size.
                _w.filterwarnings("ignore", message=".*same as training data.*")
                metrics = evaluate(model, data, _guard=False)
            elapsed = round(_time.perf_counter() - t0, 2)
            algo = model._algorithm
            if is_tuned[i]:
                algo = f"{algo} (tuned)"
            elif getattr(model, "_calibrated", False):
                algo = f"{algo} (calibrated)"
            elif getattr(model, "_balance", False):
                algo = f"{algo} (balanced)"
            # A10: custom metric (Conort C2)
            if custom_scorer is not None:
                try:
                    from .predict import _predict_impl, _predict_proba
                    if custom_scorer.needs_proba:
                        y_pred = _predict_proba(model, data).values
                    else:
                        y_pred = _predict_impl(model, data).values
                    y_true = data[target].values
                    custom_score = custom_scorer(y_true, y_pred)
                    metrics[custom_scorer.name] = float(custom_score)
                except Exception:
                    pass  # Custom scorer errors are non-fatal

            # Store raw predictions for significance testing (2.6)
            try:
                from .predict import _predict_impl as _pi
                all_preds.append(_pi(model, data).values)
            except Exception:
                all_preds.append(None)

            results.append({"algorithm": algo, **metrics, "time": elapsed})
        except DataError:
            raise  # Data errors are real — don't mask them
        except Exception as e:
            error_msg = str(e).split("\n")[0][:120]
            algo = getattr(model, "_algorithm", "unknown")
            if is_tuned[i]:
                algo = f"{algo} (tuned)"
            elif getattr(model, "_calibrated", False):
                algo = f"{algo} (calibrated)"
            elif getattr(model, "_balance", False):
                algo = f"{algo} (balanced)"
            all_preds.append(None)
            results.append({"algorithm": algo, "error": error_msg})

    # Warn about failed models (Principle 4: Fail Loudly)
    import warnings

    n_failed = sum(1 for r in results if "error" in r)
    if n_failed > 0:
        n_total = len(results)
        failed_names = [r["algorithm"] for r in results if "error" in r]
        warnings.warn(
            f"{n_failed}/{n_total} models failed evaluation: {failed_names}. "
            "See 'error' column in results for details.",
            UserWarning,
            stacklevel=2,
        )

    # Build DataFrame
    df = pd.DataFrame(results)

    # Statistical significance vs best model (2.6)
    # McNemar's test for classification, paired t-test for regression.
    # Best = first element in all_preds (highest-ranked by sort below).
    # We compute significance after sorting, so skip here and add after sort.

    # Disambiguate duplicate algorithm labels (e.g., two "stacked" models)
    if "algorithm" in df.columns:
        algos = df["algorithm"].tolist()
        from collections import Counter
        counts = Counter(algos)
        seen: dict[str, int] = {}
        for i, a in enumerate(algos):
            if counts[a] > 1:
                seen[a] = seen.get(a, 0) + 1
                algos[i] = f"{a}_{seen[a]}"
        df["algorithm"] = algos

    # Sort by metric
    n = len(models)
    sorted_indices = list(range(len(models)))

    # A10: When custom metric provided, sort by it in "auto" mode
    if sort_by == "auto" and custom_scorer is not None and custom_scorer.name in df.columns:
        sort_col = custom_scorer.name
        ascending = not custom_scorer.greater_is_better
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=ascending)
            sorted_indices = df.index.tolist()
            df = df.reset_index(drop=True)
            sorted_models = [models[i] for i in sorted_indices]
        else:
            sorted_models = list(models)
    elif sort_by == "auto":
        if task == "classification":
            if "roc_auc" in df.columns:
                sort_col = "roc_auc"
            elif "roc_auc_ovr" in df.columns:
                sort_col = "roc_auc_ovr"
            elif "accuracy" in df.columns:
                sort_col = "accuracy"
            else:
                sort_col = None
        else:
            if "rmse" in df.columns:
                sort_col = "rmse"
            elif "mae" in df.columns:
                sort_col = "mae"
            else:
                sort_col = None
        if sort_col and sort_col in df.columns:
            ascending = sort_col in ("rmse", "mae")
            df = df.sort_values(sort_col, ascending=ascending)
            sorted_indices = df.index.tolist()
            df = df.reset_index(drop=True)
            sorted_models = [models[i] for i in sorted_indices]
        else:
            sorted_models = list(models)
    else:
        sort_col = sort_by
        if sort_col in df.columns:
            ascending = sort_col in ("rmse", "mae")
            df = df.sort_values(sort_col, ascending=ascending)
            sorted_indices = df.index.tolist()
            df = df.reset_index(drop=True)
            sorted_models = [models[i] for i in sorted_indices]
        else:
            metric_cols = [c for c in df.columns if c not in ("algorithm", "error", "time")]
            raise ConfigError(
                f"sort_by={sort_by!r} not found in results. "
                f"Available metrics: {metric_cols}"
            )

    # Statistical significance vs best model (2.6 — McNemar / paired t-test)
    df = _add_significance_column(df, all_preds, sorted_indices, task,
                                  data[target].values)

    # Round numeric columns for clean display
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(4)

    return Leaderboard(df, title=f"Compare [{task} · {n} models]", models=sorted_models)


def _add_significance_column(
    df: pd.DataFrame,
    all_preds: list,
    sorted_indices: list,
    task: str,
    y_true,
) -> pd.DataFrame:
    """Add significant_vs_best column using McNemar (clf) or paired t-test (reg).

    p < 0.05 → True (different from best). Best model vs itself is always False.
    Falls back gracefully when scipy is absent or predictions are unavailable.
    """
    import numpy as np

    from . import _stats

    if not sorted_indices or all_preds[sorted_indices[0]] is None:
        return df

    best_idx = sorted_indices[0]
    best_preds = all_preds[best_idx]
    sig_col = []

    for orig_idx in sorted_indices:
        preds = all_preds[orig_idx]
        if orig_idx == best_idx or preds is None or best_preds is None:
            sig_col.append(False)
            continue
        try:
            if task == "classification":
                # McNemar's test on disagreement table
                best_correct = (best_preds == y_true).astype(int)
                model_correct = (preds == y_true).astype(int)
                b = int(np.sum((best_correct == 1) & (model_correct == 0)))
                c = int(np.sum((best_correct == 0) & (model_correct == 1)))
                if b + c == 0:
                    sig_col.append(False)
                else:
                    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                    p = float(_stats.chi2_sf(chi2, 1))
                    sig_col.append(bool(p < 0.05))
            else:
                # Paired t-test on absolute errors
                best_errs = np.abs(best_preds - y_true)
                model_errs = np.abs(preds - y_true)
                _, p = _stats.ttest_rel(best_errs, model_errs)
                sig_col.append(bool(p < 0.05))
        except Exception:
            sig_col.append(False)

    df = df.copy()
    df["significant_vs_best"] = sig_col
    return df
