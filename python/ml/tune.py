"""Hyperparameter tuning via random or grid search.

Progressive disclosure: zero-config smart defaults, or custom param ranges.
No external dependencies (Optuna deferred to Gate 3).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import _engines, _normalize
from ._types import ConfigError, DataError, Model, TuningResult
from .fit import _compute_metrics

# Smart defaults per algorithm — (low, high) for numeric, list for categorical
TUNE_DEFAULTS: dict[str, dict] = {
    "xgboost": {
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.3),
        "n_estimators": (100, 1000),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.3, 1.0),
        "min_child_weight": (1, 20),
        "reg_alpha": (0.0, 10.0),
        "reg_lambda": (0.0, 10.0),
        "gamma": (0.0, 5.0),                 # min loss reduction for a split
        "max_bin": (64, 254),                # histogram bin count (u8 limit: 255 reserved for NaN)
        "grow_policy": ["depthwise", "lossguide"],
    },
    "gradient_boosting": {
        "n_estimators": (50, 300),
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 8),
        "subsample": (0.5, 1.0),
        "min_samples_split": (2, 10),
        "min_samples_leaf": (1, 5),
        "reg_lambda": (0.001, 10.0),            # L2 regularization (log-uniform)
        "gamma": (0.0, 5.0),                    # min split gain
        "colsample_bytree": (0.3, 1.0),         # column subsampling per tree
        "min_child_weight": (1.0, 10.0),         # min hessian per leaf (log-uniform)
    },
    "random_forest": {
        "max_depth": (5, 30),
        "n_estimators": (50, 500),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
        "max_features": ["sqrt", "log2"],
        "min_impurity_decrease": (0.0, 0.1),     # min impurity decrease for split
        "ccp_alpha": (0.0, 0.05),                # cost-complexity pruning
    },
    "logistic": {
        "C": (0.001, 100.0),
    },
    "svm": {
        "C": (0.01, 100.0),
        "gamma": ["scale", "auto"],
        "kernel": ["linear", "rbf"],             # rbf only viable for n <= 5000
    },
    "knn": {
        "n_neighbors": (3, 21),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "linear": {},
    "lightgbm": {
        "num_leaves": (20, 150),             # THE critical LightGBM param — leaf-wise splitting
        "max_depth": (-1, 12),               # -1 = unlimited (let num_leaves control complexity)
        "learning_rate": (0.01, 0.3),
        "n_estimators": (100, 1000),         # higher budget with early stopping
        "subsample": (0.5, 1.0),            # aka bagging_fraction
        "colsample_bytree": (0.3, 1.0),     # wider range for feature subsampling
        "min_child_samples": (5, 100),       # aka min_data_in_leaf
        "reg_alpha": (0.0, 10.0),
        "reg_lambda": (0.0, 10.0),
        "min_split_gain": (0.0, 1.0),       # minimum gain to make a split
        "path_smooth": (0.0, 10.0),         # smoothing for leaf values
    },
    "lightgbm_dart": {
        "num_leaves": (20, 150),
        "max_depth": (-1, 12),
        "learning_rate": (0.01, 0.3),
        "n_estimators": (100, 500),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.3, 1.0),
        "min_child_samples": (5, 100),
        "reg_alpha": (0.0, 10.0),
        "reg_lambda": (0.0, 10.0),
        "drop_rate": (0.05, 0.3),           # fraction of trees to drop per iteration
        "skip_drop": (0.3, 0.7),            # probability of skipping dropout
    },
    "catboost": {
        "depth": (4, 10),
        "learning_rate": (0.01, 0.3),
        "iterations": (100, 1000),
        "l2_leaf_reg": (1.0, 10.0),         # L2 regularisation on leaves
        "random_strength": (0.0, 10.0),      # randomness for scoring splits
        "bagging_temperature": (0.0, 10.0),  # Bayesian bootstrap temperature
        "border_count": (32, 255),           # number of splits for numerical features
        "min_data_in_leaf": (1, 50),
        "grow_policy": ["SymmetricTree", "Lossguide", "Depthwise"],
    },
    "histgradient": {
        "max_depth": (3, 12),
        "learning_rate": (0.01, 0.3),
        "max_iter": (100, 500),
        "min_samples_leaf": (5, 50),
        "max_leaf_nodes": (20, 150),
        "l2_regularization": (0.0, 10.0),
        "max_bins": (64, 255),
        "reg_lambda": (0.001, 10.0),             # L2 regularization (log-uniform)
        "gamma": (0.0, 5.0),                     # min split gain
        "colsample_bytree": (0.3, 1.0),          # column subsampling per tree
        "min_child_weight": (1.0, 10.0),          # min hessian per leaf (log-uniform)
    },
    "decision_tree": {
        "max_depth": (3, 20),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
        "min_impurity_decrease": (0.0, 0.1),     # min impurity decrease for split
        "ccp_alpha": (0.0, 0.05),                # cost-complexity pruning
    },
    "extra_trees": {
        "max_depth": (5, 30),
        "n_estimators": (50, 500),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
        "max_features": ["sqrt", "log2"],
        "min_impurity_decrease": (0.0, 0.1),     # min impurity decrease for split
        "ccp_alpha": (0.0, 0.05),                # cost-complexity pruning
    },
    "naive_bayes": {},  # GaussianNB has no hyperparams to tune
    "elastic_net": {
        "alpha": (0.001, 10.0),
        "l1_ratio": (0.1, 0.9),
    },
}


def tune(
    data: pd.DataFrame,
    target: str,
    *,
    model: Model | None = None,
    algorithm: str | None = None,
    n_trials: int = 20,
    cv_folds: int = 3,
    method: str = "random",
    seed: int,
    params: dict | None = None,
    task: str = "auto",
    timeout: int | None = None,
    time_budget: int | None = None,
    metric: str | object = "auto",
    weights: str | None = None,
    balance: bool = False,
    patience: int | None = None,
    engine: str = "auto",
) -> TuningResult:
    """Tune hyperparameters via random or grid search with cross-validation.

    Provide either a fitted model (defines algorithm) or an algorithm name.
    Searches for better hyperparameters using cross-validation.

    Args:
        data: Training data (same format as ml.fit()).
        target: Target column name.
        model: Fitted Model from ml.fit(). Defines the algorithm to tune.
        algorithm: Algorithm name (e.g. "xgboost"). Alternative to passing model.
        n_trials: Number of random search trials (default: 20). Ignored for grid search.
        cv_folds: Number of cross-validation folds (default: 3).
        method: Search method — "random" (default), "grid" (exhaustive),
            or "bayesian" (Optuna TPE, requires ``pip install optuna``).
            Grid search requires params= with lists of values (not ranges).
        seed: Random seed for reproducibility.
        params: Custom parameter ranges. Dict of param_name → (low, high) for
            numeric, or param_name → [val1, val2, ...] for categorical.
            For method="grid", all values must be lists (not tuples).
            If None, uses algorithm-specific smart defaults.
        timeout: Maximum search time in seconds (method="bayesian" only).
            If None, runs until n_trials is reached.
        time_budget: Maximum wall-clock time in seconds for ALL methods.
            When set, the search stops after this many seconds regardless of
            n_trials. Works with random, grid, and bayesian methods.
            For bayesian: passed as timeout= to study.optimize().
            For random/grid: checks elapsed time between trials.
        metric: Scoring metric. ``"auto"`` uses roc_auc for classification
            and rmse for regression. Also accepts metric strings (``"f1"``,
            ``"rmse"``, ``"accuracy"``) or a callable with
            ``scorer.greater_is_better: bool``.
        balance: Auto-weight classes for imbalanced data (classification only).
            Same as ml.fit(balance=True) — applied per fold during HPO search
            and to the final refit. Cannot combine with weights=.

            .. warning::
                After many trials, tune() CV score is optimistically biased.
                Always evaluate on held-out data (s.valid) for honest estimation.

    Returns:
        TuningResult with:
        - .best_model: Model fitted with best hyperparameters
        - .best_params_: dict of best hyperparameter values
        - .tuning_history_: DataFrame with trial results
        Delegates .predict() and .predict_proba() to best_model.

    Raises:
        ConfigError: If neither model nor algorithm provided, or params are invalid
        DataError: If data/target invalid

    Example:
        >>> s = ml.split(data, "churn", seed=42)
        >>> tuned = ml.tune(s.train, "churn", algorithm="xgboost", seed=42)
        >>> ml.evaluate(tuned, s.valid)

        >>> # Grid search — exhaustive over all combinations
        >>> tuned = ml.tune(s.train, "churn", algorithm="xgboost", seed=42,
        ...     method="grid", params={"max_depth": [3, 5, 7], "n_estimators": [100, 200]})

        >>> # Tune with class balancing (imbalanced data)
        >>> tuned = ml.tune(s.train, "fraud", algorithm="xgboost", seed=42, balance=True)
    """
    # Resolve algorithm from model or algorithm param
    if model is not None and not isinstance(model, Model):
        raise ConfigError(
            "tune() model= must be a fitted Model from ml.fit(). "
            f"Got {type(model).__name__}."
        )

    if model is None and algorithm is None:
        raise ConfigError(
            "tune() requires either model= (fitted Model) or algorithm= (string). "
            "Example: ml.tune(data, 'target', algorithm='xgboost', seed=42)"
        )

    if model is not None and algorithm is not None:
        warnings.warn(
            f"Both model= and algorithm= provided. Using algorithm='{algorithm}'.",
            UserWarning,
            stacklevel=2,
        )

    # Validate seed (same guard as fit() and split())
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(
            f"seed must be an integer, got {type(seed).__name__}: {seed!r}. "
            "Example: seed=42"
        )
    if seed < 0 or seed > 2**32 - 1:
        raise ConfigError(
            f"seed must be between 0 and {2**32 - 1}, got {seed}. "
            "Example: seed=42"
        )

    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"Expected DataFrame for data, got {type(data).__name__}"
        )

    # Validate n_trials
    if n_trials < 1:
        raise ConfigError(
            f"n_trials must be >= 1, got {n_trials}. "
            "Example: ml.tune(data, 'target', algorithm='xgboost', n_trials=20, seed=42)"
        )

    if target not in data.columns:
        available = data.columns.tolist()
        raise DataError(
            f"target='{target}' not found in data. Available: {available}"
        )

    # Resolve algorithm and task
    if algorithm is not None:
        algo = algorithm
    else:
        algo = model._algorithm

    # Resolve task
    if task != "auto":
        detected_task = task
    elif model is not None:
        detected_task = model._task
    else:
        # Detect task from target column (no wasteful probe fit)
        from .split import _detect_task
        detected_task = _detect_task(data[target])
    task = detected_task

    # Resolve parameter ranges
    search_space = params if params is not None else TUNE_DEFAULTS.get(algo, {})

    # SVM kernel conditional: rbf is O(n^2) — only viable for small datasets
    if algo == "svm" and params is None and "kernel" in search_space:
        n_rows = len(data)
        if n_rows > 5000:
            search_space = dict(search_space)
            search_space["kernel"] = ["linear"]

    if not search_space:
        warnings.warn(
            f"No hyperparameters to tune for algorithm='{algo}'. "
            "Returning model with default parameters.",
            UserWarning,
            stacklevel=2
        )
        if model is not None:
            base = model
        else:
            from .fit import fit as _fit
            base = _fit(data=data, target=target, algorithm=algo, seed=seed)
        return TuningResult(
            best_model=base,
            best_params_={},
            tuning_history_=pd.DataFrame(columns=["trial", "score"]),
        )

    # Validate method
    if method not in ("random", "grid", "bayesian"):
        raise ConfigError(
            f"method='{method}' not available. Choose from: ['random', 'grid', 'bayesian']"
        )

    # Validate search space
    for name, spec in search_space.items():
        if isinstance(spec, (list, tuple)):
            continue
        raise ConfigError(
            f"Invalid param spec for '{name}': {spec}. "
            "Expected (low, high) tuple or [val1, val2, ...] list."
        )

    # Grid search: auto-expand tuples to 5-point grids, then cap at 1000 combinations
    if method == "grid":
        expanded: dict = {}
        for name, spec in search_space.items():
            if isinstance(spec, tuple) and len(spec) == 2:
                low, high = spec
                if isinstance(low, int) and isinstance(high, int):
                    expanded[name] = [int(v) for v in np.linspace(low, high, 5)]
                else:
                    expanded[name] = [float(v) for v in np.linspace(float(low), float(high), 5)]
            elif isinstance(spec, list):
                expanded[name] = spec
            else:
                raise ConfigError(
                    f"Grid search requires lists or (low, high) tuples. "
                    f"Got '{name}': {spec!r}."
                )
        search_space = expanded

        # Hard cap: prevent combinatorial explosion
        total = 1
        for v in search_space.values():
            total *= len(v) if v else 1
            if total > 1000:
                break
        if total > 1000:
            raise ConfigError(
                f"Grid search would require {total:,}+ combinations (max 1000). "
                "Reduce parameter space or use method='random' with n_trials=20."
            )

    # Validate cv_folds
    if cv_folds < 2:
        raise ConfigError("cv_folds must be >= 2")
    if cv_folds > len(data):
        raise ConfigError(
            f"cv_folds={cv_folds} exceeds data rows ({len(data)}). "
            f"Use cv_folds={min(5, len(data))} or fewer."
        )

    # Validate balance + weights conflict (same as fit())
    if balance and weights is not None:
        raise ConfigError(
            "Cannot use balance=True and weights= together. "
            "Choose one: balance=True for auto class balancing, "
            "or weights= for custom per-sample weights."
        )

    # Validate weights column
    if weights is not None:
        if weights not in data.columns:
            raise DataError(
                f"weights='{weights}' not found in data. Available columns: {data.columns.tolist()}"
            )
        if weights == target:
            raise DataError("weights= column cannot be the same as the target column.")

    # Set up data
    drop_cols = [target] + ([weights] if weights else [])
    X = data.drop(columns=drop_cols)
    y = data[target]
    rng = np.random.RandomState(seed)

    # Set up CV
    from .split import _kfold, _stratified_kfold

    if task == "classification":
        cv_splits = list(_stratified_kfold(y.values, k=cv_folds, seed=seed))
    else:
        cv_splits = list(_kfold(len(X), k=cv_folds, seed=seed))

    # Determine primary metric (aligned with screen/compare defaults)
    if task == "classification":
        n_classes = data[target].nunique()
        primary_metric = "roc_auc" if n_classes == 2 else "roc_auc_ovr"
    else:
        primary_metric = "rmse"
    higher_is_better = primary_metric != "rmse"

    # Custom metric override (A11, Conort C2)
    _custom_scorer = None
    if metric != "auto":
        from ._scoring import make_scorer as _make_scorer
        _custom_scorer = _make_scorer(metric)
        primary_metric = _custom_scorer.name
        higher_is_better = _custom_scorer.greater_is_better

    # Warn once about NaN before the search loop (not per fold per trial)
    n_nan = X.isna().any(axis=1).sum()
    if n_nan:
        warnings.warn(
            f"{n_nan} rows contain NaN features. "
            "NaN is passed through for tree-based models but may cause "
            "errors for linear/SVM/KNN.",
            UserWarning,
            stacklevel=2,
        )

    # Bayesian search path (Optuna TPE)
    # Conflict guard: both timeout= and time_budget= cannot be set simultaneously.
    # They both cap compute time and were added at different times for the same purpose.
    # Force the user to be explicit rather than silently picking one.
    if timeout is not None and time_budget is not None:
        raise ConfigError(
            "Use time_budget= (seconds). timeout= is deprecated — please remove it. "
            f"Got timeout={timeout} and time_budget={time_budget}."
        )

    # W35-F3: Deprecation warning when timeout= is used alone (time_budget= is None)
    if timeout is not None and time_budget is None:
        warnings.warn(
            "timeout= is deprecated. Use time_budget= instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Resolve time_budget: overrides timeout for all methods
    _effective_timeout = time_budget if time_budget is not None else timeout

    if method == "bayesian":
        trials = _bayesian_search(
            data=data, target=target, algo=algo, task=task,
            search_space=search_space, cv_splits=cv_splits,
            seed=seed, n_trials=n_trials, timeout=_effective_timeout,
            primary_metric=primary_metric, higher_is_better=higher_is_better,
            custom_scorer=_custom_scorer, engine=engine,
        )
    else:
        # Generate parameter candidates
        if method == "grid":
            param_candidates = _grid_combinations(search_space)
            total = len(param_candidates)
            if total > 100:
                warnings.warn(
                    f"Grid search has {total} combinations (>100). "
                    "This may take a long time. Consider using method='random' "
                    f"with n_trials={min(50, total)}.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            param_candidates = [_sample_params(search_space, rng) for _ in range(n_trials)]

        # Search loop (suppress per-fold NaN warnings)
        import time as _time
        _search_start = _time.time()
        trials = []
        # For random search with time_budget: iterate lazily using counter
        if method == "random" and _effective_timeout is not None:
            _random_iter = _random_iter_with_budget(search_space, rng, n_trials, _effective_timeout, _search_start)
        else:
            _random_iter = None

        # Pre-compute per-fold normalization + transformed data (fold data and
        # algorithm are identical across trials — only hyperparams change).
        _fold_cache: dict[int, tuple] = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*rows contain NaN.*")
            for fold_idx, (train_idx, valid_idx) in enumerate(cv_splits):
                fold_train = data.iloc[train_idx]
                fold_valid = data.iloc[valid_idx]
                fold_drop = [target] + ([weights] if weights else [])
                X_train = fold_train.drop(columns=fold_drop)
                y_train = fold_train[target]
                X_valid = fold_valid.drop(columns=fold_drop)
                y_valid = fold_valid[target]
                fold_sw = None
                if weights is not None:
                    fold_sw = fold_train[weights].values.astype(np.float64)
                norm_state = _normalize.prepare(X_train, y_train, algorithm=algo, task=task)
                X_train_clean = norm_state.pop_train_data()
                if X_train_clean is None:
                    X_train_clean = norm_state.transform(X_train)
                y_train_clean = norm_state.encode_target(y_train)
                X_valid_clean = norm_state.transform(X_valid)
                _fold_cache[fold_idx] = (
                    norm_state, X_train_clean, y_train_clean, X_valid_clean, y_valid, fold_sw
                )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*rows contain NaN.*")
            _iter_source = _random_iter if _random_iter is not None else enumerate(param_candidates)
            for trial_idx, trial_params in _iter_source:
                if _effective_timeout is not None and _time.time() - _search_start >= _effective_timeout:
                    break

                fold_scores = []
                valid_trial = True
                for fold_idx, (_train_idx, _valid_idx) in enumerate(cv_splits):  # Phase 0.3: track fold_idx
                    norm_state, X_train_clean, y_train_clean, X_valid_clean, y_valid, fold_sw = _fold_cache[fold_idx]

                    try:
                        # Use all cores for parallel-capable algos, unless user set ml.config(n_jobs=N).
                        # Phase 0.3: vary seed per trial+fold to reduce HPO evaluation bias.
                        fold_seed = seed + trial_idx * 1000 + fold_idx
                        tune_n_jobs = {"n_jobs": _engines._screen_n_jobs()} if algo in _engines.PARALLEL_ALGORITHMS else {}

                        # Apply class balancing per fold (same logic as fit())
                        fold_balance_params = dict(trial_params)
                        balance_sw = fold_sw  # user weights (or None)
                        if balance:
                            from .fit import _prepare_balance
                            fold_balance_params, computed_sw = _prepare_balance(
                                algo, y_train_clean, fold_balance_params
                            )
                            if computed_sw is not None:
                                balance_sw = computed_sw

                        engine_kwargs = {**tune_n_jobs, **fold_balance_params}
                        estimator = _engines.create(
                            algo, task=task, seed=fold_seed, engine=engine, **engine_kwargs
                        )
                        fit_kwargs = {"sample_weight": balance_sw} if balance_sw is not None else {}
                        estimator.fit(X_train_clean, y_train_clean, **fit_kwargs)

                        y_pred = estimator.predict(X_valid_clean)
                        y_pred_decoded = norm_state.decode(y_pred)

                        if _custom_scorer is not None:
                            if _custom_scorer.needs_proba and hasattr(estimator, "predict_proba"):
                                y_score = estimator.predict_proba(X_valid_clean)
                                if hasattr(y_score, "ndim") and y_score.ndim > 1:
                                    y_score = y_score[:, 1]
                                fold_score = _custom_scorer(y_valid, y_score)
                            else:
                                fold_score = _custom_scorer(y_valid, y_pred_decoded)
                        else:
                            metrics = _compute_metrics(
                                y_valid, y_pred_decoded, task, estimator, X_valid_clean
                            )
                            fold_score = metrics.get(primary_metric, 0.0)

                        fold_scores.append(fold_score)
                    except Exception:
                        # Invalid param combination — skip this trial
                        valid_trial = False
                        break

                if valid_trial and fold_scores:
                    mean_score = float(np.mean(fold_scores))
                else:
                    mean_score = float("-inf") if higher_is_better else float("inf")

                trials.append({
                    "trial": trial_idx,
                    "params": trial_params,
                    "score": mean_score,
                })

    # Warn about failed trials (Principle 4: Fail Loudly)
    n_failed = sum(
        1 for t in trials
        if t["score"] == float("-inf") or t["score"] == float("inf")
    )
    if n_failed > 0:
        n_total = len(trials)
        warnings.warn(
            f"{n_failed}/{n_total} trials failed (invalid parameter combinations "
            "or fitting errors). Failed trials are excluded from selection.",
            UserWarning,
            stacklevel=2,
        )
        if n_failed == n_total:
            raise ConfigError(
                f"All {n_total} trials failed. Check your data and algorithm. "
                "Common causes: all-NaN features, incompatible parameter ranges, "
                "or too few samples for the algorithm."
            )

    # Find best trial
    if higher_is_better:
        best_trial = max(trials, key=lambda t: t["score"])
    else:
        best_trial = min(trials, key=lambda t: t["score"])

    # Refit on ALL data with best params (suppress duplicate NaN warning)
    from .fit import fit
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*rows contain NaN.*")
        tuned_model = fit(
            data=data, target=target, algorithm=algo, seed=seed,
            task=task, weights=weights, balance=balance, **best_trial["params"]
        )

    # Build tuning history DataFrame
    history_records = []
    for t in trials:
        record = {"trial": t["trial"], "score": t["score"]}
        record.update(t["params"])
        history_records.append(record)

    tuning_history = pd.DataFrame(history_records).sort_values(
        "score", ascending=not higher_is_better
    ).reset_index(drop=True)

    return TuningResult(
        best_model=tuned_model,
        best_params_=dict(best_trial["params"]),
        tuning_history_=tuning_history,
        metric_=primary_metric,
    )


def _random_iter_with_budget(
    search_space: dict,
    rng: np.random.RandomState,
    n_trials: int,
    time_budget: float,
    start_time: float,
) -> None:  # Generator
    """Lazily generate (trial_idx, params) pairs until n_trials or time_budget hit."""
    import time
    trial_idx = 0
    while trial_idx < n_trials:
        if time.time() - start_time >= time_budget:
            return
        yield trial_idx, _sample_params(search_space, rng)
        trial_idx += 1



def _grid_combinations(search_space: dict) -> list[dict]:
    """Generate all parameter combinations for grid search.

    Args:
        search_space: Dict of param_name → [val1, val2, ...] lists.

    Returns:
        List of dicts, each a unique parameter combination.
    """
    import itertools

    names = list(search_space.keys())
    value_lists = [search_space[n] for n in names]

    combos = []
    for values in itertools.product(*value_lists):
        combos.append(dict(zip(names, values)))
    return combos


def _sample_params(search_space: dict, rng: np.random.RandomState) -> dict:
    """Sample one set of hyperparameters from search space."""
    params = {}
    for name, spec in search_space.items():
        if isinstance(spec, list):
            params[name] = spec[rng.randint(len(spec))]
        elif isinstance(spec, tuple) and len(spec) == 2:
            low, high = spec
            if isinstance(low, int) and isinstance(high, int):
                params[name] = int(rng.randint(low, high + 1))
            else:
                low_f, high_f = float(low), float(high)
                if low_f > 0 and high_f / low_f > 100:
                    # Log-uniform for very wide ranges (e.g. C: 0.001–100)
                    params[name] = float(
                        np.exp(rng.uniform(np.log(low_f), np.log(high_f)))
                    )
                else:
                    params[name] = float(rng.uniform(low_f, high_f))
    return params


# ---------------------------------------------------------------------------
# Bayesian search (Optuna TPE)
# ---------------------------------------------------------------------------

# Params where log-scale is physically meaningful
_LOG_SCALE_PARAMS = frozenset({"learning_rate", "reg_alpha", "reg_lambda", "C", "gamma"})


def _bayesian_search(
    *,
    data: pd.DataFrame,
    target: str,
    algo: str,
    task: str,
    search_space: dict,
    cv_splits: list,
    seed: int,
    n_trials: int,
    timeout: int | None,
    primary_metric: str,
    higher_is_better: bool,
    custom_scorer: object | None,
    engine: str = "auto",
) -> list:
    """Optuna TPE-based hyperparameter search with MedianPruner.

    Returns list of trial dicts with keys: trial, params, score.
    """
    try:
        import optuna
    except ImportError:
        from ._types import ConfigError
        raise ConfigError(
            "Bayesian HPO requires optuna. Install it with: pip install optuna\n"
            "Or: pip install 'ml[optuna]'"
        ) from None

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pre-compute per-fold normalization + transformed data (fold data and
    # algorithm are identical across trials — only hyperparams change).
    _bayes_fold_cache: dict[int, tuple] = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*rows contain NaN.*")
        for _fi, (_ti, _vi) in enumerate(cv_splits):
            _ft = data.iloc[_ti]
            _fv = data.iloc[_vi]
            _xt = _ft.drop(columns=[target])
            _yt = _ft[target]
            _xv = _fv.drop(columns=[target])
            _yv = _fv[target]
            _ns = _normalize.prepare(_xt, _yt, algorithm=algo, task=task)
            _xtc = _ns.pop_train_data()
            if _xtc is None:
                _xtc = _ns.transform(_xt)
            _ytc = _ns.encode_target(_yt)
            _xvc = _ns.transform(_xv)
            _bayes_fold_cache[_fi] = (_ns, _xtc, _ytc, _xvc, _yv)

    trial_counter = [0]
    trials_out: list = []

    def objective(trial: optuna.Trial) -> float:
        trial_idx = trial_counter[0]
        trial_counter[0] += 1

        # Suggest params using Optuna trial API
        trial_params: dict = {}
        for name, spec in search_space.items():
            if isinstance(spec, list):
                trial_params[name] = trial.suggest_categorical(name, spec)
            elif isinstance(spec, tuple) and len(spec) == 2:
                low, high = spec
                if isinstance(low, int) and isinstance(high, int):
                    trial_params[name] = trial.suggest_int(name, int(low), int(high))
                else:
                    use_log = name in _LOG_SCALE_PARAMS and float(low) > 0
                    trial_params[name] = trial.suggest_float(
                        name, float(low), float(high), log=use_log
                    )

        fold_scores: list = []
        for fold_idx, (_train_idx, _valid_idx) in enumerate(cv_splits):
            norm_state, X_train_clean, y_train_clean, X_valid_clean, y_valid = _bayes_fold_cache[fold_idx]

            fold_seed = seed + trial_idx * 1000 + fold_idx
            tune_n_jobs = {"n_jobs": _engines._screen_n_jobs()} if algo in _engines.PARALLEL_ALGORITHMS else {}
            estimator = _engines.create(
                algo, task=task, seed=fold_seed, engine=engine, **{**tune_n_jobs, **trial_params}
            )

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    estimator.fit(X_train_clean, y_train_clean)
            except Exception:
                return float("-inf") if higher_is_better else float("inf")

            y_pred = estimator.predict(X_valid_clean)
            y_pred_decoded = norm_state.decode(y_pred)

            if custom_scorer is not None:
                if getattr(custom_scorer, "needs_proba", False) and hasattr(estimator, "predict_proba"):
                    y_score = estimator.predict_proba(X_valid_clean)
                    if hasattr(y_score, "ndim") and y_score.ndim > 1:
                        y_score = y_score[:, 1]
                    fold_score = float(custom_scorer(y_valid, y_score))
                else:
                    fold_score = float(custom_scorer(y_valid, y_pred_decoded))
            else:
                metrics = _compute_metrics(y_valid, y_pred_decoded, task, estimator, X_valid_clean)
                fold_score = float(metrics.get(primary_metric, 0.0))

            fold_scores.append(fold_score)

            # Report intermediate value for MedianPruner
            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores)) if fold_scores else (
            float("-inf") if higher_is_better else float("inf")
        )
        trials_out.append({"trial": trial_idx, "params": trial_params, "score": mean_score})
        # Optuna always maximizes: return score directly if higher_is_better, else negate
        return mean_score if higher_is_better else -mean_score

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return trials_out
