"""Fit models on data.

Single verb for all modeling: fit(DataFrame) or fit(CVResult).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from . import _engines, _normalize
from ._logging import logger
from ._types import CVResult, DataError, Model


def fit(
    data: pd.DataFrame | CVResult,
    target: str | None = None,
    *,
    algorithm: str = "auto",
    backend=None,
    preprocessor=None,
    seed: int,
    task: str = "auto",
    balance: bool = False,
    weights=None,
    early_stopping: bool | int = True,
    eval_fraction: float = 0.1,
    monotone: dict | None = None,
    gpu: bool | str = "auto",
    engine: str = "auto",
    **kwargs,
) -> Model:
    """Fit a model on data.

    Handles both holdout and cross-validation paths.

    Args:
        data: DataFrame (holdout) or CVResult (cross-validation)
        target: Target column name
        algorithm: "auto", "xgboost", "random_forest", "svm", "knn", "logistic",
            "linear" (Ridge regression with L2 regularization, not plain OLS).
            When backend= is used, controls normalization strategy and display name.
        backend: Custom estimator. Must have .fit(X, y) and .predict(X) methods.
            Optionally .predict_proba(X) for classification. When provided, skips
            built-in engine creation. Cannot combine with **kwargs.
        engine: Backend selection. "auto" prefers Rust (ml) > native > sklearn.
            "ml" forces Rust. "sklearn" forces sklearn. "native" forces
            Rust or pure-numpy (no sklearn).
        preprocessor: Custom feature transform applied after normalization, before
            engine.fit(). Callable: X_df → X_df. Applied during both fit and predict.
            Stateless (no fitting). For stateful transforms, pre-fit and pass .transform.
        seed: Random seed for reproducibility (required)
        task: "auto", "classification", or "regression" — override heuristic
        balance: Auto-weight classes for imbalanced data (classification only).
            Sets class_weight="balanced" for tree/linear models, scale_pos_weight for
            XGBoost binary, is_unbalance for LightGBM, auto_class_weights for CatBoost.
        weights: Per-sample weights. Either a column name (str) in data, or an
            array-like (pd.Series, np.ndarray) with one value per row.
            Passed to engine.fit(sample_weight=...). Cannot combine with balance=True.
        early_stopping: Controls early stopping for XGBoost/LightGBM.
            True (default): patience=10, eval_fraction carve. False: no carve, all
            data used for training. int (e.g. 50): custom patience rounds with
            eval_fraction carve.
        eval_fraction: Fraction of training data carved for early stopping eval set
            (default: 0.1 = 10%). Only used when early_stopping != False.
        monotone: Monotonicity constraints per feature. Dict of feature_name → int,
            where 1 = increasing, -1 = decreasing, 0 = unconstrained.
            Supported algorithms: xgboost, lightgbm, catboost. Raises ConfigError
            for other algorithms.
            Example: monotone={"age": 1, "discount": -1}
        **kwargs: Passed to underlying engine (e.g., max_depth=5, n_estimators=100)

    Returns:
        Model with .scores_ (CV) or .scores_=None (holdout)

    Raises:
        DataError: If target not found or data invalid
        ConfigError: If algorithm not available, kwargs invalid, or backend invalid

    Example:
        >>> model = ml.fit(s.train, "churn", seed=42)
        >>> model.algorithm
        'xgboost'

        >>> model = ml.fit(s.train, "churn", seed=42, balance=True)

        >>> # Monotonicity constraints (credit scoring, insurance)
        >>> model = ml.fit(s.train, "risk", algorithm="xgboost", seed=42,
        ...     monotone={"age": 1, "income": 1, "loan_amount": -1})

        >>> # Disable early stopping to use all training data
        >>> model = ml.fit(s.train, "churn", seed=42, early_stopping=False)
    """
    # Validate seed early (before sklearn sees it and raises opaque errors)
    # A8: seed may be a list for seed averaging
    if isinstance(seed, list):
        return _fit_seed_average(
            data=data, target=target, algorithm=algorithm, seed_list=seed,
            backend=backend, preprocessor=preprocessor, task=task,
            balance=balance, weights=weights,
            early_stopping=early_stopping, eval_fraction=eval_fraction,
            monotone=monotone, **kwargs,
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        from ._types import ConfigError
        raise ConfigError(
            f"seed must be an integer or list of integers, got {type(seed).__name__}: {seed!r}. "
            "Example: seed=42 or seed=[42, 43, 44]"
        )
    if seed < 0 or seed > 2**32 - 1:
        from ._types import ConfigError
        raise ConfigError(
            f"seed must be between 0 and {2**32 - 1}, got {seed}. "
            "Example: seed=42"
        )

    # Auto-convert Polars/other DataFrames to pandas

    # Infer target from CVResult or SplitResult partition when not provided
    if target is None:
        if isinstance(data, CVResult) and data.target is not None:
            target = data.target
        elif hasattr(data, '_target') and data._target is not None:
            target = data._target
        elif isinstance(data, pd.DataFrame) and data.attrs.get("_ml_target") is not None:
            target = data.attrs["_ml_target"]
        elif isinstance(data, pd.DataFrame):
            from ._types import ConfigError
            raise ConfigError(
                "target= is required when passing a raw DataFrame. "
                "Example: ml.fit(data, 'target_col', seed=42)"
            )

    # Validate data type early
    if not isinstance(data, (pd.DataFrame, CVResult)):
        from ._types import ConfigError
        raise ConfigError(
            f"data= must be a DataFrame or CVResult. Got {type(data).__name__}. "
            "Use: ml.fit(data=df, target='col') or ml.fit(data=cv, target='col')."
        )

    # Partition guard — reject test data in fit()
    if isinstance(data, pd.DataFrame):
        from ._provenance import guard_fit
        guard_fit(data)

    # Validate preprocessor
    if preprocessor is not None and not callable(preprocessor):
        from ._types import ConfigError
        raise ConfigError(
            f"preprocessor= must be callable (X_df → X_df). "
            f"Got {type(preprocessor).__name__}."
        )

    # Validate weights + balance conflict
    if weights is not None and balance:
        from ._types import ConfigError
        raise ConfigError(
            "Cannot use weights= and balance=True together. "
            "Choose one: weights= for custom per-sample weights, "
            "or balance=True for auto class balancing."
        )

    # Debug logging
    try:
        if isinstance(data, CVResult):
            _n_rows = len(data) if hasattr(data, "__len__") else 0
            _n_features = 0
        else:
            _n_rows = len(data)
            _n_features = max(0, len(data.columns) - 1)
        logger.debug(
            "fit() called: algorithm=%s, task=%s, n_rows=%d, n_features=%d",
            algorithm, task, _n_rows, _n_features,
        )
    except Exception:
        pass  # logging must never break fit()

    # Route to CV or holdout
    if isinstance(data, CVResult):
        return _fit_cv(data, target, algorithm, seed, task, backend=backend,
                       preprocessor=preprocessor, balance=balance, weights=weights,
                       early_stopping=early_stopping, eval_fraction=eval_fraction,
                       monotone=monotone, gpu=gpu, engine=engine, **kwargs)
    else:
        return _fit_holdout(data, target, algorithm, seed, task, backend=backend,
                            preprocessor=preprocessor, balance=balance, weights=weights,
                            early_stopping=early_stopping, eval_fraction=eval_fraction,
                            monotone=monotone, gpu=gpu, engine=engine, **kwargs)


def _validate_backend(backend) -> None:
    """Validate custom backend has required duck-type methods."""
    from ._types import ConfigError

    if not hasattr(backend, "fit") or not callable(backend.fit):
        raise ConfigError(
            f"backend must have a .fit(X, y) method. "
            f"{type(backend).__name__} does not."
        )
    if not hasattr(backend, "predict") or not callable(backend.predict):
        raise ConfigError(
            f"backend must have a .predict(X) method. "
            f"{type(backend).__name__} does not."
        )


def _auto_select_algorithm(n: int, task: str) -> tuple[str, str | None]:
    """Algorithm selection for algorithm='auto'.

    Grammar: gradient boosting (lightgbm/xgboost) is the modern tabular default.
    Falls back to random_forest when neither is installed.
    For small n (<500): consider algorithm='logistic' (clf) or 'linear' (reg) —
    parametric models generalize better than boosting at limited sample sizes.

    Returns (algorithm_name, optional_warning_message).
    """
    try:
        import lightgbm  # noqa: F401
        return "lightgbm", None
    except ImportError:
        pass
    try:
        import xgboost  # noqa: F401
        return "xgboost", None
    except ImportError:
        pass
    return "random_forest", (
        "lightgbm and xgboost not installed — using random_forest as default. "
        'Install for better defaults: pip install "mlw[xgboost]"'
    )


def _fit_holdout(
    data: pd.DataFrame,
    target: str,
    algorithm: str,
    seed: int,
    task: str,
    backend=None,
    preprocessor=None,
    balance: bool = False,
    weights=None,
    early_stopping: bool | int = True,
    eval_fraction: float = 0.1,
    monotone: dict | None = None,
    gpu: bool | str = "auto",
    engine: str = "auto",
    **kwargs,
) -> Model:
    """Fit on a single DataFrame (holdout path)."""
    from ._types import ConfigError

    # Validate backend
    if backend is not None:
        _validate_backend(backend)
        if kwargs:
            raise ConfigError(
                "backend= and **kwargs cannot be used together. "
                "Configure your estimator before passing it as backend=."
            )

    # Validate data
    if len(data) == 0:
        raise DataError("Data has 0 rows. Cannot fit a model on an empty DataFrame.")

    if data.columns.duplicated().any():
        dupes = data.columns[data.columns.duplicated()].tolist()
        raise DataError(f"Duplicate column names: {dupes}")

    if target not in data.columns:
        cols = list(data.columns)
        shown = cols[:10]
        extra = len(cols) - 10
        suffix = f" ... and {extra} more" if extra > 0 else ""
        raise DataError(
            f"target column '{target}' not found in data. "
            f"Available columns: {shown}{suffix}"
        )

    # Drop rows with NaN target
    n_before = len(data)
    data = data.dropna(subset=[target]).reset_index(drop=True)
    n_after = len(data)
    if n_after < n_before:
        warnings.warn(
            f"Dropped {n_before - n_after} rows with NaN target.",
            UserWarning,
            stacklevel=3
        )

    if n_after == 0:
        raise DataError(f"Target column '{target}' is entirely NaN")

    # Multi-label detection: target column contains lists/tuples/sets
    # (must run BEFORE nunique() which crashes on unhashable types)
    if data[target].dtype == object:
        _non_null = data[target].dropna()
        if len(_non_null) > 0:
            _sample = _non_null.iloc[0]
            if isinstance(_sample, (list, tuple, set)):
                from ._types import ConfigError
                raise ConfigError(
                    "Multi-label targets detected (target column contains lists). "
                    "ml supports binary and multiclass classification only. "
                    "Flatten to single-label or use one-vs-rest encoding."
                )

    # Check single-class target — can't train on this
    if data[target].nunique() == 1:
        only_val = data[target].iloc[0]
        raise DataError(
            f"Target '{target}' has only 1 unique value ({only_val}). "
            "Need at least 2 classes for classification or variance for regression."
        )

    # Extract user-provided sample weights before X/y split.
    # weights= accepts either a column name (str) or an array-like (pd.Series, np.ndarray, list).
    user_weights = None
    _weights_col: str | None = None  # column to drop from X, only when weights is a str
    if weights is not None:
        if isinstance(weights, str):
            if weights not in data.columns:
                available = data.columns.tolist()
                raise DataError(
                    f"weights='{weights}' not found in data. Available columns: {available}"
                )
            if weights == target:
                raise DataError("weights= column cannot be the same as the target column.")
            user_weights = data[weights].values.astype(np.float64)
            _weights_col = weights
        else:
            # Array-like: pd.Series, np.ndarray, list, etc.
            user_weights = np.asarray(weights, dtype=np.float64).ravel()
            if len(user_weights) != len(data):
                raise DataError(
                    f"weights array length {len(user_weights)} != data length {len(data)}. "
                    "weights= must have one value per row in data."
                )
        if np.isnan(user_weights).any():
            raise DataError(
                "weights contains NaN values. All sample weights must be numeric and non-null."
            )
        if (user_weights < 0).any():
            raise DataError(
                "weights contains negative values. All sample weights must be non-negative."
            )

    # Separate X and y BEFORE normalization
    drop_cols = [target]
    if _weights_col is not None:
        drop_cols.append(_weights_col)
    X = data.drop(columns=drop_cols)
    y = data[target]

    # Guard: zero features after dropping target
    if len(X.columns) == 0:
        raise DataError(
            f"No features found — data only has the target column '{target}'. "
            "Need at least one feature column to train a model."
        )

    # Guard: Inf check removed — _normalize.prepare() checks Inf canonically.

    # Guard: datetime/timedelta columns — not meaningful as numeric features
    dt_cols = list(X.select_dtypes(include=["datetime", "datetimetz", "timedelta"]).columns)
    if dt_cols:
        raise DataError(
            f"Datetime/timedelta columns found: {dt_cols[:5]}. "
            "Convert to numeric features first (e.g., .dt.year, .dt.dayofweek, "
            "or total_seconds())."
        )

    # Warn about zero-variance features (carry no predictive signal)
    _nuniques = X.nunique()
    zero_var = _nuniques[_nuniques <= 1].index.tolist()
    if zero_var:
        names = ", ".join(str(c) for c in zero_var[:5])
        suffix = f" (and {len(zero_var) - 5} more)" if len(zero_var) > 5 else ""
        warnings.warn(
            f"{len(zero_var)} feature(s) have zero variance: {names}{suffix}. "
            "These carry no predictive signal.",
            UserWarning,
            stacklevel=3,
        )

    # Detect task if auto
    detected_task = task
    if task == "auto":
        detected_task = _detect_task_from_target(y)

    # Guard: task='regression' on string/categorical target is a misconfiguration
    if detected_task == "regression" and (
        pd.api.types.is_object_dtype(y)
        or pd.api.types.is_string_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
    ):
        raise ConfigError(
            f"task='regression' but target '{target}' has dtype '{y.dtype}' "
            "(string/categorical). Regression requires numeric targets. "
            "Use task='classification' or convert target to numeric."
        )

    # Resolve short aliases before anything else
    from ._engines import ALGORITHM_ALIASES
    algorithm = ALGORITHM_ALIASES.get(algorithm, algorithm)

    # Resolve algorithm and estimator
    if backend is not None:
        # Custom backend: when algorithm="auto", use "_custom" to force NaN
        # imputation — we don't know if the custom backend handles NaN natively
        # (e.g. GradientBoosting doesn't, unlike XGBoost/LightGBM).
        resolved_algorithm = algorithm if algorithm != "auto" else "_custom"
        display_name = algorithm if algorithm != "auto" else type(backend).__name__
        estimator = backend
    else:
        # Built-in path: resolve algorithm
        resolved_algorithm = algorithm
        if algorithm == "auto":
            resolved_algorithm, _auto_warn = _auto_select_algorithm(n_after, detected_task)
            if _auto_warn:
                warnings.warn(_auto_warn, UserWarning, stacklevel=3)
        display_name = resolved_algorithm

    # Prepare normalization state
    norm_state = _normalize.prepare(X, y, algorithm=resolved_algorithm, task=detected_task)

    # Use cached training data from prepare() if available (avoids redundant transform)
    X_clean = norm_state.pop_train_data()
    if X_clean is None:
        X_clean = norm_state.transform(X)

    # Apply custom preprocessor (Hook 4: Lego)
    if preprocessor is not None:
        X_clean = preprocessor(X_clean)

    y_clean = norm_state.encode_target(y)

    # Apply class balancing or user-provided weights
    sample_weight = user_weights  # None unless weights= was specified
    if balance:
        if detected_task != "classification":
            from ._types import ConfigError as _CE
            raise _CE(
                "balance=True only works for classification tasks, not "
                "regression. For regression with skewed targets, consider "
                "transforming the target (e.g., log) or using weights= "
                "for per-sample weights."
            )
        kwargs, sample_weight = _prepare_balance(resolved_algorithm, y_clean, kwargs)

    # LightGBM: set force_col_wise only for wide data (>500 features)
    # Unconditional force_col_wise degrades performance on narrow datasets
    if resolved_algorithm == "lightgbm" and X_clean.shape[1] > 500:
        kwargs = {**kwargs, "force_col_wise": True}

    # Monotonicity constraints — convert feature-name dict to engine-native format
    if monotone is not None:
        kwargs = _apply_monotone_constraints(resolved_algorithm, monotone, X_clean.columns, kwargs)

    # Guard: criterion="poisson" requires non-negative target values
    if kwargs.get("criterion") == "poisson" and (y_clean < 0).any():
        from ._types import ConfigError as _ConfigError
        raise _ConfigError(
            "criterion='poisson' requires non-negative target values. "
            "Got negative values in target. Transform the target or use a different criterion."
        )

    # Guard: TabPFN works best with <10K rows (warn before engine creation for fast fail)
    if resolved_algorithm == "tabpfn" and len(X_clean) > 10000:
        warnings.warn(
            f"TabPFN works best with <10K rows. Got {len(X_clean)}. "
            "Consider algorithm='lightgbm' for larger datasets, "
            "or use TabPFN in an ensemble for diversity.",
            UserWarning,
            stacklevel=3,
        )

    # Create estimator (only for built-in path)
    if backend is None:
        estimator = _engines.create(resolved_algorithm, task=detected_task, seed=seed, gpu=gpu, engine=engine, **kwargs)

    # Guard: KNN performance degrades in high dimensions (curse of dimensionality)
    if resolved_algorithm == "knn" and X_clean.shape[1] > 50:
        warnings.warn(
            f"KNN with {X_clean.shape[1]} features: distance metrics degrade in high "
            "dimensions. Consider algorithm='random_forest' or 'logistic', "
            "or apply PCA first.",
            UserWarning,
            stacklevel=3,
        )

    # Guard: KNN n_neighbors must not exceed training samples
    if resolved_algorithm == "knn":
        k = getattr(estimator, "n_neighbors", 5)
        if k > len(X_clean):
            raise DataError(
                f"n_neighbors={k} but training data has only {len(X_clean)} rows. "
                f"Use n_neighbors<={len(X_clean)} or choose a different algorithm."
            )

    # Resolve early stopping patience
    # early_stopping=True → patience=10 (legacy default)
    # early_stopping=False → no carve
    # early_stopping=N (int) → patience=N
    if early_stopping is False:
        es_patience = None  # disabled
    elif early_stopping is True:
        es_patience = 10
    else:
        if not isinstance(early_stopping, int) or isinstance(early_stopping, bool):
            from ._types import ConfigError
            raise ConfigError(
                f"early_stopping must be True, False, or an int (patience rounds). "
                f"Got {early_stopping!r}."
            )
        es_patience = int(early_stopping)

    # Prepare early stopping eval set (XGBoost/LightGBM)
    fit_kwargs = {}
    n_train_actual = None  # Phase 0.5: track actual rows used for final fit
    X_eval, y_eval = None, None  # W35-F1: eval set for holdout score estimate
    if es_patience is not None and _wants_early_stopping(estimator) and len(X_clean) >= 50:
        if sample_weight is not None:
            X_fit, X_eval, y_fit, y_eval, sw_fit = _early_stop_split(
                X_clean, y_clean, detected_task, seed,
                sample_weight=sample_weight,
                eval_fraction=eval_fraction,
            )
            fit_kwargs["sample_weight"] = sw_fit
        else:
            X_fit, X_eval, y_fit, y_eval = _early_stop_split(
                X_clean, y_clean, detected_task, seed,
                eval_fraction=eval_fraction,
            )
        fit_kwargs["eval_set"] = [(X_eval, y_eval)]
        estimator.set_params(early_stopping_rounds=es_patience)
        _silence_early_stop(estimator, fit_kwargs)
        # Phase 0.5: disclose the carve to user
        n_train_actual = len(X_fit)
        warnings.warn(
            f"Early stopping holds out {len(X_eval)} rows ({len(X_eval)/len(X_clean):.0%}) "
            f"for evaluation. Final model trains on {n_train_actual} of {len(X_clean)} rows. "
            "Use model._n_train_actual to see the actual training size.",
            UserWarning,
            stacklevel=3,
        )
    else:
        X_fit, y_fit = X_clean, y_clean
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

    t0 = time.perf_counter()
    # Suppress sklearn internal RuntimeWarnings for linear models
    # (matmul divide-by-zero / overflow — known sklearn artifact, not a user bug)
    _suppress_rt = resolved_algorithm in ("linear", "elastic_net", "logistic")
    with warnings.catch_warnings():
        if _suppress_rt:
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        try:
            estimator.fit(X_fit, y_fit, **fit_kwargs)
        except ValueError as e:
            if "NaN" in str(e):
                from ._types import DataError as _DE
                raise _DE(
                    f"{display_name} still has NaN after auto-imputation. "
                    "This may indicate all-NaN columns. Use data.dropna(axis=1) "
                    "to remove fully-missing columns."
                ) from e
            raise
    fit_time = time.perf_counter() - t0

    # W35-F1: Compute holdout score estimate for cv_score property.
    # Gradient boosting with early stopping: use the eval set score directly.
    # Other algorithms: quick train/eval split estimate (no new data needed).
    holdout_score = _compute_holdout_score(
        engine=estimator,
        resolved_algorithm=resolved_algorithm,
        detected_task=detected_task,
        X_fit=X_fit,
        y_fit=y_fit,
        X_eval=X_eval,
        y_eval=y_eval,
        seed=seed,
    )

    # Build Model
    _model = Model(
        _model=estimator,
        _task=detected_task,
        _algorithm=display_name,
        _features=list(X.columns),
        _target=target,
        _seed=seed,
        _label_encoder=norm_state.label_encoder,
        _feature_encoder=norm_state,
        _preprocessor=preprocessor,
        _n_train=len(data),
        _n_train_actual=n_train_actual,  # Phase 0.5: actual rows used (< n_train when early stopping)
        scores_=None,  # holdout has no CV scores
        _holdout_score=holdout_score,  # W35-F1: holdout estimate for cv_score
        _time=fit_time,
        _balance=balance,
        _sample_weight_col=weights,  # A12: store for reference
    )
    # Layer 2: Store provenance for cross-verb checks
    from ._provenance import _fingerprint, audit_log, build_provenance
    _model._provenance = build_provenance(data)
    audit_log("fit", _fingerprint(data), partition_role=data.attrs.get("_ml_partition"))
    return _model


def _fit_cv(
    cv: CVResult,
    target: str,
    algorithm: str,
    seed: int,
    task: str,
    backend=None,
    preprocessor=None,
    balance: bool = False,
    weights=None,
    early_stopping: bool | int = True,
    eval_fraction: float = 0.1,
    monotone: dict | None = None,
    gpu: bool | str = "auto",
    engine: str = "auto",
    **kwargs,
) -> Model:
    """Fit with cross-validation (per-fold normalization).

    CRITICAL: Normalization statistics MUST be recomputed per fold.
    Global normalization is textbook preprocessing leakage.
    """
    from ._types import ConfigError

    # Validate backend
    if backend is not None:
        _validate_backend(backend)
        if kwargs:
            raise ConfigError(
                "backend= and **kwargs cannot be used together. "
                "Configure your estimator before passing it as backend=."
            )

    data = cv._data

    # Validate target
    if target not in data.columns:
        cols = list(data.columns)
        shown = cols[:10]
        extra = len(cols) - 10
        suffix = f" ... and {extra} more" if extra > 0 else ""
        raise DataError(
            f"target column '{target}' not found in data. "
            f"Available columns: {shown}{suffix}"
        )

    # Drop rows with NaN target (same cleaning as holdout)
    n_before = len(data)
    data = data.dropna(subset=[target]).reset_index(drop=True)
    n_after = len(data)
    if n_after < n_before:
        warnings.warn(
            f"Dropped {n_before - n_after} rows with NaN target.",
            UserWarning,
            stacklevel=3
        )

    if n_after == 0:
        raise DataError(f"Target column '{target}' is entirely NaN")

    # Detect task if auto
    y_full = data[target]
    detected_task = task
    if task == "auto":
        detected_task = _detect_task_from_target(y_full)

    # Guard: task='regression' on string/categorical target is a misconfiguration
    if detected_task == "regression" and (
        pd.api.types.is_object_dtype(y_full)
        or pd.api.types.is_string_dtype(y_full)
        or isinstance(y_full.dtype, pd.CategoricalDtype)
    ):
        raise ConfigError(
            f"task='regression' but target '{target}' has dtype '{y_full.dtype}' "
            "(string/categorical). Regression requires numeric targets. "
            "Use task='classification' or convert target to numeric."
        )

    # Resolve short aliases before anything else
    from ._engines import ALGORITHM_ALIASES
    algorithm = ALGORITHM_ALIASES.get(algorithm, algorithm)

    # Resolve algorithm and display name
    if backend is not None:
        resolved_algorithm = algorithm if algorithm != "auto" else "_custom"
        display_name = algorithm if algorithm != "auto" else type(backend).__name__
    else:
        resolved_algorithm = algorithm
        if algorithm == "auto":
            resolved_algorithm, _auto_warn = _auto_select_algorithm(n_after, detected_task)
            if _auto_warn:
                warnings.warn(_auto_warn, UserWarning, stacklevel=3)
        display_name = resolved_algorithm

    # Validate weights column
    if weights is not None:
        if weights not in data.columns:
            available = data.columns.tolist()
            raise DataError(
                f"weights='{weights}' not found in data. Available columns: {available}"
            )
        if weights == target:
            raise DataError("weights= column cannot be the same as the target column.")
        _w = data[weights].values.astype(np.float64)
        if np.isnan(_w).any():
            raise DataError(
                f"weights column '{weights}' contains NaN values. "
                "All sample weights must be numeric and non-null."
            )

    # Validate balance
    if balance and detected_task != "classification":
        from ._types import ConfigError as _CE
        raise _CE(
                "balance=True only works for classification tasks, not "
                "regression. For regression with skewed targets, consider "
                "transforming the target (e.g., log) or using weights= "
                "for per-sample weights."
            )

    # Memory warning for large CV (R audit H8)
    n_features = len(data.columns) - 1
    n_folds = len(cv.folds)
    n_cells = len(data) * n_features * n_folds
    if n_cells > 1e8:
        warnings.warn(
            f"CV will process {n_cells / 1e6:.0f}M cells "
            f"({len(data)} rows x {n_features} features x {n_folds} folds). "
            "Consider reducing folds or sampling data to manage memory.",
            UserWarning,
            stacklevel=3,
        )

    # Guard: KNN performance degrades in high dimensions (curse of dimensionality)
    if resolved_algorithm == "knn" and n_features > 50:
        warnings.warn(
            f"KNN with {n_features} features: distance metrics degrade in high "
            "dimensions. Consider algorithm='random_forest' or 'logistic', "
            "or apply PCA first.",
            UserWarning,
            stacklevel=3,
        )

    # Per-fold training and evaluation
    # Suppress per-fold scaling/imputation warnings (fire once, not N times)
    t0 = time.perf_counter()
    fold_metrics = []
    # OOF prediction collectors
    oof_preds_parts: list[pd.Series] = []
    oof_proba_parts: list[tuple[np.ndarray, np.ndarray]] = []  # (indices, proba)
    _suppress_rt = resolved_algorithm in ("linear", "elastic_net", "logistic")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Auto-scaling.*")
        warnings.filterwarnings("ignore", message=".*auto-imputed.*")
        if _suppress_rt:
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        for fold_train, fold_valid in cv.folds:

            # Separate X and y for fold train
            drop_cols_cv = [target] + ([weights] if weights else [])
            X_train = fold_train.drop(columns=drop_cols_cv)
            y_train = fold_train[target]

            # Separate X and y for fold valid
            X_valid = fold_valid.drop(columns=drop_cols_cv)
            y_valid = fold_valid[target]

            # Extract per-fold weights
            fold_user_weights = None
            if weights is not None:
                fold_user_weights = fold_train[weights].values.astype(np.float64)

            # PER-FOLD normalization (CRITICAL)
            norm_state = _normalize.prepare(X_train, y_train, algorithm=resolved_algorithm, task=detected_task)
            X_train_clean = norm_state.pop_train_data()
            if X_train_clean is None:
                X_train_clean = norm_state.transform(X_train)
            y_train_clean = norm_state.encode_target(y_train)

            # Transform valid using fold-train stats
            X_valid_clean = norm_state.transform(X_valid)

            # Apply custom preprocessor (Hook 4: Lego)
            if preprocessor is not None:
                X_train_clean = preprocessor(X_train_clean)
                X_valid_clean = preprocessor(X_valid_clean)

            # Apply class balancing or user weights per fold
            fold_kwargs = dict(kwargs)
            fold_sw = fold_user_weights  # None unless weights= was specified
            if balance:
                fold_kwargs, fold_sw = _prepare_balance(resolved_algorithm, y_train_clean, fold_kwargs)

            # LightGBM: force_col_wise only for wide data (>500 features)
            if resolved_algorithm == "lightgbm" and X_train_clean.shape[1] > 500:
                fold_kwargs["force_col_wise"] = True

            # Monotonicity constraints
            if monotone is not None:
                fold_kwargs = _apply_monotone_constraints(
                    resolved_algorithm, monotone, X_train_clean.columns, fold_kwargs
                )

            # Fit estimator on fold (clone backend for fresh instance)
            if backend is not None:
                from ._engines import clone as _clone
                estimator = _clone(backend)
            else:
                estimator = _engines.create(resolved_algorithm, task=detected_task, seed=seed, gpu=gpu, engine=engine, **fold_kwargs)

            # Build fit kwargs: sample_weight + early stopping eval_set
            fold_fit_kwargs = {}
            if fold_sw is not None:
                fold_fit_kwargs["sample_weight"] = fold_sw
            # Encode validation target for early stopping eval_set.
            # May fail in temporal CV when validation has labels unseen in
            # training (temporal class drift). Skip early stopping for that fold.
            try:
                y_valid_clean = norm_state.encode_target(y_valid)
            except ValueError:
                y_valid_clean = None
                warnings.warn(
                    "Validation fold has labels unseen in training "
                    "(temporal class drift). Early stopping skipped for this fold.",
                    UserWarning,
                    stacklevel=3,
                )
            # Resolve patience for this fold
            if early_stopping is False:
                _fold_es_patience = None
            elif early_stopping is True:
                _fold_es_patience = 10
            else:
                _fold_es_patience = int(early_stopping)
            if _fold_es_patience is not None and _wants_early_stopping(estimator) and y_valid_clean is not None:
                fold_fit_kwargs["eval_set"] = [(X_valid_clean, y_valid_clean)]
                estimator.set_params(early_stopping_rounds=_fold_es_patience)
                _silence_early_stop(estimator, fold_fit_kwargs)
            estimator.fit(X_train_clean, y_train_clean, **fold_fit_kwargs)

            # Predict on fold valid
            y_pred = estimator.predict(X_valid_clean)

            # Decode predictions back to original labels
            y_pred_decoded = norm_state.decode(y_pred)

            # Precompute proba once for metrics (avoids redundant predict_proba call)
            fold_proba = None
            if detected_task == "classification" and hasattr(estimator, "predict_proba"):
                import contextlib
                with contextlib.suppress(Exception):
                    fold_proba = estimator.predict_proba(X_valid_clean)

            # Compute metrics
            metrics = _compute_metrics(y_valid, y_pred_decoded, detected_task, estimator, X_valid_clean, proba=fold_proba)
            fold_metrics.append(metrics)

            # Collect OOF predictions
            oof_preds_parts.append(pd.Series(y_pred_decoded.values, index=fold_valid.index))
            if fold_proba is not None:
                oof_proba_parts.append((fold_valid.index.to_numpy(), fold_proba))

    # After all folds: refit on ALL data (warnings visible here — single fit)
    drop_cols_final = [target] + ([weights] if weights else [])
    X_all = data.drop(columns=drop_cols_final)
    y_all = data[target]

    final_norm = _normalize.prepare(X_all, y_all, algorithm=resolved_algorithm, task=detected_task)
    X_all_clean = final_norm.pop_train_data()
    if X_all_clean is None:
        X_all_clean = final_norm.transform(X_all)

    # Apply custom preprocessor (Hook 4: Lego)
    if preprocessor is not None:
        X_all_clean = preprocessor(X_all_clean)

    y_all_clean = final_norm.encode_target(y_all)

    # Apply class balancing or user weights for final refit
    final_kwargs = dict(kwargs)
    final_sw = data[weights].values.astype(np.float64) if weights else None
    if balance:
        final_kwargs, final_sw = _prepare_balance(resolved_algorithm, y_all_clean, final_kwargs)

    # LightGBM: force_col_wise only for wide data
    if resolved_algorithm == "lightgbm" and X_all_clean.shape[1] > 500:
        final_kwargs["force_col_wise"] = True

    # Monotonicity constraints for final refit
    if monotone is not None:
        final_kwargs = _apply_monotone_constraints(
            resolved_algorithm, monotone, X_all_clean.columns, final_kwargs
        )

    if backend is not None:
        from ._engines import clone as _clone
        final_engine = _clone(backend)
    else:
        final_engine = _engines.create(resolved_algorithm, task=detected_task, seed=seed, gpu=gpu, engine=engine, **final_kwargs)

    # Resolve patience for final refit
    if early_stopping is False:
        _final_es_patience = None
    elif early_stopping is True:
        _final_es_patience = 10
    else:
        _final_es_patience = int(early_stopping)

    # Build final fit kwargs
    final_fit_kwargs = {}
    _is_temporal = getattr(cv, "_temporal", False)
    n_train_actual_cv = None  # Phase 0.5: track actual rows for CV final refit
    if _final_es_patience is not None and _wants_early_stopping(final_engine) and len(X_all_clean) >= 50:
        if final_sw is not None:
            X_fit, X_eval, y_fit, y_eval, sw_fit = _early_stop_split(
                X_all_clean, y_all_clean, detected_task, seed,
                sample_weight=final_sw,
                temporal=_is_temporal,
                eval_fraction=eval_fraction,
            )
            final_fit_kwargs["sample_weight"] = sw_fit
        else:
            X_fit, X_eval, y_fit, y_eval = _early_stop_split(
                X_all_clean, y_all_clean, detected_task, seed,
                temporal=_is_temporal,
                eval_fraction=eval_fraction,
            )
        final_fit_kwargs["eval_set"] = [(X_eval, y_eval)]
        final_engine.set_params(early_stopping_rounds=_final_es_patience)
        _silence_early_stop(final_engine, final_fit_kwargs)
        # Phase 0.5: disclose the carve in the final refit
        n_train_actual_cv = len(X_fit)
        warnings.warn(
            f"Early stopping holds out {len(X_eval)} rows ({len(X_eval)/len(X_all_clean):.0%}) "
            f"for evaluation. Final model trains on {n_train_actual_cv} of {len(X_all_clean)} rows. "
            "Use model._n_train_actual to see the actual training size.",
            UserWarning,
            stacklevel=3,
        )
    else:
        X_fit, y_fit = X_all_clean, y_all_clean
        if final_sw is not None:
            final_fit_kwargs["sample_weight"] = final_sw
    with warnings.catch_warnings():
        if _suppress_rt:
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        final_engine.fit(X_fit, y_fit, **final_fit_kwargs)
    fit_time = time.perf_counter() - t0

    # Build scores_ dict: mean + std per metric
    scores_dict = {}
    for metric_name in fold_metrics[0]:
        vals = [fold[metric_name] for fold in fold_metrics]
        scores_dict[f"{metric_name}_mean"] = float(np.mean(vals))
        scores_dict[f"{metric_name}_std"] = float(np.std(vals, ddof=1))

    # Assemble OOF predictions
    _cv_preds = None
    _cv_proba = None
    if oof_preds_parts:
        _cv_preds = pd.concat(oof_preds_parts).sort_index()
    if oof_proba_parts:
        # Assemble full probability matrix ordered by original index
        n_total = len(data)
        n_classes = oof_proba_parts[0][1].shape[1]
        _cv_proba = np.empty((n_total, n_classes), dtype=np.float64)
        for indices, proba in oof_proba_parts:
            _cv_proba[indices] = proba

    # Build Model with scores_ and per-fold metrics
    _model = Model(
        _model=final_engine,
        _task=detected_task,
        _algorithm=display_name,
        _features=list(X_all.columns),
        _target=target,
        _seed=seed,
        _label_encoder=final_norm.label_encoder,
        _feature_encoder=final_norm,
        _preprocessor=preprocessor,
        _n_train=len(data),
        _n_train_actual=n_train_actual_cv,  # Phase 0.5: actual rows in final refit
        scores_=scores_dict,
        fold_scores_=fold_metrics,
        _time=fit_time,
        _balance=balance,
        _sample_weight_col=weights,  # A12: store for reference
        cv_predictions_=_cv_preds,
        cv_probabilities_=_cv_proba if detected_task == "classification" else None,
    )
    # Layer 2: Store provenance for cross-verb checks
    from ._provenance import _fingerprint, audit_log, build_provenance
    _model._provenance = build_provenance(data)
    audit_log("fit", _fingerprint(data), partition_role=data.attrs.get("_ml_partition"))
    return _model


_EARLY_STOP_ENGINES = {"XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor"}


def _wants_early_stopping(engine) -> bool:
    """Check if engine supports early stopping via eval_set."""
    return type(engine).__name__ in _EARLY_STOP_ENGINES


def _compute_holdout_score(
    engine, resolved_algorithm: str, detected_task: str,
    X_fit, y_fit, X_eval, y_eval, seed: int,
) -> float | None:
    """Compute a quick holdout performance estimate for cv_score property.

    W35-F1: Provides a non-None cv_score for holdout-fitted models.

    Strategy:
    - Gradient boosting with early stopping (XGBoost/LightGBM/CatBoost):
      Use the best eval-set score recorded during training (no extra compute).
    - Random forest: Use OOB score if available, else train/eval split.
    - All others: Quick train/eval split (20% holdout, same seed).

    Returns a float in [0, 1] for classification (higher is better),
    or a negative RMSE for regression (higher magnitude = worse).
    Returns None if estimation fails or dataset is too small.
    """
    from ._scoring import _acc as _accuracy_score
    from ._scoring import _r2 as _r2_score

    try:
        engine_type = type(engine).__name__

        # ── Gradient boosting with early stopping ──────────────────────────
        # Use accuracy/r2 on eval set (same metric as other algorithms) so
        # cv_score is comparable across all algorithms. Predict on X_eval if
        # available (early-stopping already carved it out); fall through to
        # generic eval below if not.
        if engine_type in (
            "XGBClassifier", "XGBRegressor",
            "LGBMClassifier", "LGBMRegressor",
            "CatBoostClassifier", "CatBoostRegressor",
        ):
            # Fall through to the generic X_eval path below; don't return early
            pass

        # ── Random forest / Rust RF: OOB score ──────────────────────────
        # Duck typing: any engine with oob_score_ (sklearn RF, Rust RF)
        oob = getattr(engine, "oob_score_", None)
        if oob is not None:
            return float(oob)

        # ── Fallback: train/eval split using 20% of X_fit ─────────────────
        # Skip if we already have a dedicated eval set (early stopping already
        # computed a score but wasn't captured above — defensive guard only).
        if X_eval is not None and y_eval is not None and len(y_eval) >= 5:
            preds = engine.predict(X_eval)
            if detected_task == "classification":
                return float(_accuracy_score(y_eval, preds))
            else:
                return float(_r2_score(y_eval, preds))

        # If no eval set, clone+refit on a 80/20 split for a quick estimate.
        # Only for small datasets (<=10K rows) where cost is ~10-25ms.
        # Large datasets: skip — too expensive (200ms+ at 100K), users call
        # evaluate() directly. shelf() handles None cv_score gracefully.
        if X_fit is not None and 50 <= len(X_fit) <= 10_000:
            from .split import _train_test_split
            Xtr, Xval, ytr, yval = _train_test_split(
                X_fit, y_fit, test_size=0.2, random_state=seed,
                stratify=y_fit if detected_task == "classification" else None,
            )
            # Clone and refit for the estimate (avoid contaminating the real model)
            try:
                from ._engines import clone as _clone
                quick = _clone(engine)
                # Remove early stopping for clone (clean refit)
                if hasattr(quick, "early_stopping_rounds"):
                    quick.set_params(early_stopping_rounds=None)
                quick.fit(Xtr, ytr)
                preds = quick.predict(Xval)
                if detected_task == "classification":
                    return float(_accuracy_score(yval, preds))
                else:
                    return float(_r2_score(yval, preds))
            except Exception:
                return None

    except Exception:
        return None

    return None


def _early_stop_split(
    X: pd.DataFrame, y, task: str, seed: int,
    sample_weight=None,
    temporal: bool = False,
    eval_fraction: float = 0.1,
) -> tuple:
    """Split off eval_fraction for early stopping eval set.

    When temporal=True, uses positional last eval_fraction (most recent data) instead
    of random split. This preserves chronological ordering for temporal CV.

    Returns (X_train, X_eval, y_train, y_eval) when sample_weight is None,
    or (X_train, X_eval, y_train, y_eval, sw_train) when sample_weight given.
    """
    if temporal:
        # Positional split: last eval_fraction as eval (data sorted chronologically)
        split_idx = int(len(X) * (1.0 - eval_fraction))
        X_train, X_eval = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_eval = y[:split_idx], y[split_idx:]
        if sample_weight is not None:
            sw_train = sample_weight[:split_idx]
            return X_train, X_eval, y_train, y_eval, sw_train
        return X_train, X_eval, y_train, y_eval

    from .split import _train_test_split

    stratify = y if task == "classification" else None
    arrays = [X, y] if sample_weight is None else [X, y, sample_weight]
    try:
        parts = _train_test_split(
            *arrays, test_size=eval_fraction, random_state=seed, stratify=stratify
        )
    except ValueError:
        parts = _train_test_split(
            *arrays, test_size=eval_fraction, random_state=seed
        )
    if sample_weight is None:
        return parts  # (X_train, X_eval, y_train, y_eval)
    # (X_train, X_eval, y_train, y_eval, sw_train, sw_eval)
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def _silence_early_stop(engine, fit_kwargs: dict) -> None:
    """Suppress per-round output during early stopping."""
    name = type(engine).__name__
    if "LGBM" in name and hasattr(engine, "set_params"):
        engine.set_params(verbose=-1)
    if "XGB" in name:
        # verbosity=0 (constructor) only suppresses C++ warnings.
        # verbose=False (fit kwarg) suppresses per-round eval printing.
        fit_kwargs["verbose"] = False


def _prepare_balance(
    algorithm: str,
    y: np.ndarray | pd.Series,
    kwargs: dict,
) -> tuple[dict, np.ndarray | None]:
    """Inject class-balancing parameters for the given algorithm.

    Returns (modified_kwargs, sample_weight_or_None).
    sample_weight is non-None only for algorithms without native class_weight.
    """
    kwargs = dict(kwargs)  # defensive copy

    # Algorithms with native class_weight="balanced"
    _HAS_CLASS_WEIGHT = {"random_forest", "logistic", "svm"}

    if algorithm in _HAS_CLASS_WEIGHT:
        kwargs.setdefault("class_weight", "balanced")
        return kwargs, None

    if algorithm == "xgboost":
        # Binary: scale_pos_weight = n_negative / n_positive
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) == 2:
            # Encoded as 0/1 — class 0 is negative, class 1 is positive
            n_neg = counts[0]
            n_pos = counts[1]
            if n_pos > 0:
                kwargs.setdefault("scale_pos_weight", float(n_neg / n_pos))
            return kwargs, None
        else:
            # Multiclass: use sample_weight (no native multi-class balancing)
            return kwargs, _compute_sample_weight(y)

    if algorithm == "lightgbm":
        kwargs.setdefault("is_unbalance", True)
        return kwargs, None

    if algorithm == "catboost":
        kwargs.setdefault("auto_class_weights", "Balanced")
        return kwargs, None

    # Algorithms that accept sample_weight in fit()
    _SUPPORTS_SAMPLE_WEIGHT = {"naive_bayes"}
    if algorithm in _SUPPORTS_SAMPLE_WEIGHT:
        return kwargs, _compute_sample_weight(y)

    # Algorithms that don't support any class balancing (knn, etc.)
    # Warn and skip — better than crashing
    _NO_BALANCE = {"knn"}
    if algorithm in _NO_BALANCE:
        warnings.warn(
            f"balance=True has no effect for algorithm='{algorithm}'. "
            "This algorithm does not support class balancing. "
            "Try algorithm='xgboost' or 'random_forest' instead.",
            UserWarning,
            stacklevel=4,
        )
        return kwargs, None

    # Unknown algorithm or custom backend: try sample_weight as fallback
    return kwargs, _compute_sample_weight(y)


def _compute_sample_weight(y: np.ndarray | pd.Series) -> np.ndarray:
    """Compute balanced sample weights from class frequencies.

    Each sample's weight = n_samples / (n_classes * n_samples_for_class).
    Equivalent to sklearn's compute_sample_weight("balanced", y).
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {c: n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[v] for v in y], dtype=np.float64)


def _apply_monotone_constraints(
    algorithm: str,
    monotone: dict,
    feature_columns,
    kwargs: dict,
) -> dict:
    """Translate feature-name monotone dict to engine-native constraint format.

    Supported: xgboost (tuple), lightgbm (list), catboost (dict).
    Raises ConfigError for unsupported algorithms.

    Args:
        algorithm: Resolved algorithm name.
        monotone: Dict of feature_name → int (1=inc, -1=dec, 0=none).
        feature_columns: Ordered feature column names after normalization.
        kwargs: Current engine kwargs (not mutated — returns new dict).

    Returns:
        Updated kwargs dict with monotone_constraints set.
    """
    from ._types import ConfigError
    kwargs = dict(kwargs)

    if algorithm == "xgboost":
        # XGBoost expects tuple of ints, one per feature column
        constraints = tuple(monotone.get(col, 0) for col in feature_columns)
        kwargs["monotone_constraints"] = constraints
    elif algorithm == "lightgbm":
        # LightGBM expects list of ints, one per feature column
        constraints = [monotone.get(col, 0) for col in feature_columns]
        kwargs["monotone_constraints"] = constraints
    elif algorithm == "catboost":
        # CatBoost accepts dict directly (feature name → constraint)
        kwargs["monotone_constraints"] = dict(monotone)
    else:
        raise ConfigError(
            f"monotone= constraints are not supported for algorithm='{algorithm}'. "
            "Supported: 'xgboost', 'lightgbm', 'catboost'."
        )
    return kwargs


def _compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    task: str,
    engine: any,
    X_clean: pd.DataFrame,
    proba: any = None,
) -> dict[str, float]:
    """Compute metrics for a fold or evaluation.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        task: "classification" or "regression"
        engine: Fitted estimator (for predict_proba)
        X_clean: Cleaned features (for predict_proba)
        proba: Pre-computed predict_proba output. If provided, skips the
            internal predict_proba call (avoids redundant computation in
            CV loops and bootstrap iterations).

    Returns:
        Dict of metric_name → value
    """
    from ._scoring import _acc as _accuracy_score
    from ._scoring import _brier, _precision_recall_f1, _rmse, _roc_auc
    from ._scoring import _mae as _mean_absolute_error
    from ._scoring import _r2 as _r2_score
    from ._scoring import _roc_auc_ovr as _roc_auc_ovr_score

    if task == "classification":
        metrics = {
            "accuracy": _accuracy_score(y_true, y_pred),
        }

        # Check if binary or multiclass
        # Use max of engine classes and test-set unique — test data may have
        # fewer classes (rare class absent in split) OR more (unseen class)
        n_classes_test = y_true.nunique()
        n_classes_model = len(engine.classes_) if hasattr(engine, "classes_") else n_classes_test
        n_classes = max(n_classes_test, n_classes_model)

        if n_classes <= 2:
            # Binary classification
            # For string labels, pos_label is the second sorted class
            classes = sorted(y_true.unique())
            pos_label = classes[1] if len(classes) == 2 else classes[-1]

            p, r, f = _precision_recall_f1(y_true, y_pred, average="binary", pos_label=pos_label)
            metrics["f1"] = f
            metrics["precision"] = p
            metrics["recall"] = r

            # ROC AUC and Brier score (binary) — needs >=2 samples, both classes,
            # AND model must know >=2 classes (temporal CV may train on 1 class)
            if (hasattr(engine, "predict_proba") and len(y_true) >= 2
                    and y_true.nunique() >= 2
                    and hasattr(engine, "classes_") and len(engine.classes_) >= 2):
                _proba_full = proba if proba is not None else engine.predict_proba(X_clean)
                _proba_pos = _proba_full[:, 1]
                metrics["roc_auc"] = _roc_auc(y_true, _proba_pos)
                import contextlib
                with contextlib.suppress(Exception):
                    metrics["brier_score"] = _brier(
                        y_true, _proba_pos, pos_label=pos_label
                    )

        else:
            # Multiclass classification
            pw, rw, f1w = _precision_recall_f1(y_true, y_pred, average="weighted")
            pm, rm, f1m = _precision_recall_f1(y_true, y_pred, average="macro")
            metrics["f1_weighted"] = f1w
            metrics["f1_macro"] = f1m
            metrics["precision_weighted"] = pw
            metrics["precision_macro"] = pm
            metrics["recall_weighted"] = rw
            metrics["recall_macro"] = rm

            # ROC AUC (multiclass one-vs-rest) — needs >=2 samples and all classes present
            if hasattr(engine, "predict_proba") and len(y_true) >= 2 and y_true.nunique() >= 2:
                _proba_full = proba if proba is not None else engine.predict_proba(X_clean)
                # Normalize probabilities to sum to 1.0 (XGBoost can have floating-point drift)
                _proba_full = _proba_full / _proba_full.sum(axis=1, keepdims=True)
                import contextlib
                with contextlib.suppress(ValueError):
                    metrics["roc_auc_ovr"] = _roc_auc_ovr_score(y_true, _proba_full)
                # Brier score (multiclass): mean brier across one-vs-rest binary problems
                with contextlib.suppress(Exception):
                    classes_list = sorted(y_true.unique())
                    brier_scores = []
                    for k, cls in enumerate(classes_list):
                        y_bin = (y_true == cls).astype(int)
                        brier_scores.append(_brier(y_bin, _proba_full[:, k]))
                    metrics["brier_score"] = float(np.mean(brier_scores))

    else:
        # Regression
        metrics = {
            "rmse": _rmse(y_true, y_pred),
            "mae": _mean_absolute_error(y_true, y_pred),
            "r2": _r2_score(y_true, y_pred),
        }

    # Ensure all values are Python float (not np.float64) for JSON serialization
    return {k: float(v) for k, v in metrics.items()}


def _detect_task_from_target(target: pd.Series) -> str:
    """Detect classification vs regression from target column.

    Delegates to split._detect_task (single source of truth).
    """
    from .split import _detect_task

    return _detect_task(target)


def _fit_seed_average(
    *,
    data,
    target: str,
    algorithm: str,
    seed_list: list,
    backend=None,
    preprocessor=None,
    task: str = "auto",
    balance: bool = False,
    weights=None,
    early_stopping: bool | int = True,
    eval_fraction: float = 0.1,
    monotone: dict | None = None,
    **kwargs,
) -> Model:
    """Fit N models with different seeds, returning an averaged ensemble. A8.

    When seed= is a list, creates one model per seed using the same data and
    algorithm, then combines them into a single Model that averages predictions.
    High _seed_std indicates an unstable model needing more regularization.

    Returns a Model with _ensemble, _seed_scores (CV scores per seed), and
    _seed_std (stability diagnostic). model.seed returns seed_list[0] for
    backwards compatibility.
    """
    from ._types import ConfigError

    if len(seed_list) < 2:
        raise ConfigError(
            f"seed= list must have at least 2 seeds, got {seed_list!r}. "
            "Example: seed=[42, 43, 44]"
        )
    for s in seed_list:
        if not isinstance(s, int) or isinstance(s, bool):
            raise ConfigError(
                f"All seeds in seed= list must be integers. Got {s!r} ({type(s).__name__}). "
                "Example: seed=[42, 43, 44]"
            )

    # Suppress per-seed warnings (duplicate NaN, etc.) to avoid noise
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*rows contain NaN.*")
        warnings.filterwarnings("ignore", message=".*early stopping.*", category=UserWarning)
        sub_models = []
        for s in seed_list:
            m = fit(
                data=data, target=target, algorithm=algorithm, seed=s,
                backend=backend, preprocessor=preprocessor, task=task,
                balance=balance, weights=weights,
                early_stopping=early_stopping, eval_fraction=eval_fraction,
                monotone=monotone, **kwargs,
            )
            sub_models.append(m)

    # Compute seed-level CV scores (if available) and stability diagnostic
    seed_scores = []
    for m in sub_models:
        if m.scores_ is not None:
            # Use primary metric: roc_auc (classification) or rmse (regression)
            task_detected = sub_models[0]._task
            if task_detected == "classification":
                score = m.scores_.get("roc_auc_mean", m.scores_.get("accuracy_mean", 0.0))
            else:
                score = m.scores_.get("rmse_mean", 0.0)
            seed_scores.append(float(score))

    seed_std = float(np.std(seed_scores, ddof=1)) if len(seed_scores) >= 2 else None

    # Create ensemble wrapper engine that averages predictions
    first = sub_models[0]
    ensemble_engine = _SeedAverageEnsemble(sub_models)

    return Model(
        _model=ensemble_engine,
        _task=first._task,
        _algorithm=first._algorithm,
        _features=first._features,
        _target=target,
        _seed=seed_list[0],  # amendment #17: seed returns seed_list[0]
        _label_encoder=first._label_encoder,
        _feature_encoder=first._feature_encoder,
        _preprocessor=preprocessor,
        _n_train=first._n_train,
        _balance=balance,
        _sample_weight_col=weights,
        _ensemble=sub_models,
        _seed_scores=seed_scores if seed_scores else None,
        _seed_std=seed_std,
    )


class _SeedAverageEnsemble:
    """Thin wrapper that averages predictions from multiple models. A8.

    Implements the sklearn estimator interface (predict, predict_proba, classes_)
    so the parent Model works with evaluate(), explain(), etc.
    """

    def __init__(self, models: list):
        self._models = models
        # Mirror classes_ from first model for compatibility
        first_engine = models[0]._model
        if hasattr(first_engine, "classes_"):
            self.classes_ = first_engine.classes_
        # Average feature importances for explain() compatibility
        imp_arrays = [
            getattr(m._model, "feature_importances_", None) for m in models
        ]
        if all(x is not None for x in imp_arrays):
            self.feature_importances_ = np.mean(imp_arrays, axis=0)

    def predict(self, X):
        """Average predictions.

        For classification: majority vote.
        For regression: mean of continuous predictions.
        """
        # Each sub_model's _model is already fitted; call raw engine predict
        preds = []
        for m in self._models:
            p = m._model.predict(X)
            preds.append(p)
        preds_arr = np.array(preds)
        first = self._models[0]
        if first._task == "regression":
            return preds_arr.mean(axis=0)
        else:
            # Majority vote for classification (mode across seeds)
            n_samples = preds_arr.shape[1]
            result = np.empty(n_samples, dtype=preds_arr.dtype)
            for i in range(n_samples):
                vals, counts = np.unique(preds_arr[:, i], return_counts=True)
                result[i] = vals[np.argmax(counts)]
            return result

    def predict_proba(self, X):
        """Average predicted probabilities across seeds."""
        probas = []
        for m in self._models:
            if hasattr(m._model, "predict_proba"):
                p = m._model.predict_proba(X)
                probas.append(p)
        if not probas:
            raise AttributeError("None of the ensemble models support predict_proba.")
        return np.mean(probas, axis=0)
