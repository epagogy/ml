"""Ensemble stacking — combine multiple algorithms.

Thin wrapper around sklearn StackingClassifier/StackingRegressor.
OOF-based meta-learning. Returns a Model that works with evaluate/assess/save/load.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import _engines, _normalize
from ._types import ConfigError, DataError, Model


def stack(
    data: pd.DataFrame,
    target: str,
    *,
    models: list[str] | None = None,
    meta: str | None = None,
    levels: int = 1,
    cv_folds: int = 5,
    seed: int,
    passthrough: bool = False,
    weights: str | None = None,
    balance: bool = False,
    engine: str = "auto",
) -> Model:
    """Build a stacked ensemble from multiple algorithms.

    Trains base models via out-of-fold predictions, then fits a meta-learner
    on top. Returns a single Model that works with evaluate(), assess(), etc.

    Args:
        data: Training data (DataFrame with target column).
        target: Target column name.
        models: List of algorithm names for base models.
            Default: ["xgboost", "random_forest"] for classification,
                     ["xgboost", "random_forest"] for regression.
        meta: Meta-learner algorithm. Default: "logistic" (classification)
            or "linear" (regression). Must be compatible with task. Can be
            any algorithm — "xgboost", "lightgbm", etc. for non-linear meta.
        levels: Number of stacking levels (default: 1).
            levels=1: base models → meta-learner (standard stacking).
            levels=2: base models → level-1 models → meta-learner.
        cv_folds: Number of CV folds for OOF predictions (default: 5).
        seed: Random seed for reproducibility.
        passthrough: Include original features alongside OOF predictions
            in the level-1 meta-learner input (default: False). Useful when
            base models miss simple linear patterns.
        weights: Optional column name in data for sample weights.

    Returns:
        Model with algorithm="stacked". Works with evaluate(), assess(),
        explain(), save(), load(). explain() returns base model weights.

    Raises:
        ConfigError: If invalid algorithm names or incompatible task/algorithm
        DataError: If data/target invalid

    Example:
        >>> s = ml.split(data, "churn", seed=42)
        >>> model = ml.stack(s.train, "churn", seed=42)
        >>> ml.evaluate(model, s.valid)
        {'accuracy': 0.89, ...}

        >>> model = ml.stack(s.train, "churn", seed=42,
        ...     models=["xgboost", "random_forest", "logistic"],
        ...     meta="xgboost", levels=2, passthrough=True)
    """
    # Validate inputs
    if not isinstance(data, pd.DataFrame):
        raise DataError(f"Expected DataFrame for data, got {type(data).__name__}")

    # Partition guard — stack is a training verb
    from ._provenance import guard_fit
    guard_fit(data)

    if target not in data.columns:
        available = data.columns.tolist()
        raise DataError(
            f"target='{target}' not found in data. Available: {available}"
        )

    # Drop NaN targets
    y_raw = data[target]
    nan_mask = y_raw.isna()
    if nan_mask.all():
        raise DataError(f"Target '{target}' is entirely NaN.")
    if nan_mask.any():
        n_dropped = nan_mask.sum()
        warnings.warn(
            f"Dropped {n_dropped} rows with NaN target.",
            UserWarning, stacklevel=2,
        )
        data = data[~nan_mask].reset_index(drop=True)

    # Handle weights= sentinel strings before column lookup
    # 'cv_score' and 'equal' are weighting strategies, not column names.
    _base_weights_strategy: str | None = None
    if weights in ("cv_score", "equal"):
        if weights != "equal":
            _base_weights_strategy = "cv_score"
        weights = None  # clear so column-lookup path is skipped

    # Detect task
    from .split import _detect_task
    task = _detect_task(data[target])

    # Validate balance= parameter
    if balance and task == "regression":
        raise ConfigError("balance=True requires classification task.")

    # Validate cv_folds range
    if cv_folds < 2:
        raise ConfigError(
            f"cv_folds must be >= 2, got {cv_folds}. "
            "Example: ml.stack(data, 'target', cv_folds=5, seed=42)"
        )
    if cv_folds >= len(data):
        raise ConfigError(
            f"cv_folds={cv_folds} must be less than data rows ({len(data)}). "
            f"Try cv_folds={min(5, len(data) // 2)}."
        )
    # Validate that each fold will have at least 2 samples — prevents IndexError in OOF loop
    n_train_samples = len(data)
    if cv_folds > n_train_samples // 2:
        raise ConfigError(
            f"cv_folds={cv_folds} exceeds training data size {n_train_samples} // 2 = "
            f"{n_train_samples // 2}. Each fold would have fewer than 2 samples. "
            f"Reduce cv_folds to at most {min(5, n_train_samples // 2)}."
        )

    # Default base models — include a linear model for ensemble diversity
    if models is None:
        if task == "classification":
            models = ["xgboost", "random_forest", "logistic"]
        else:
            models = ["xgboost", "random_forest", "linear"]

    if len(models) < 2:
        raise ConfigError(
            "stack() requires at least 2 base algorithms. "
            f"Got only: {models}. "
            "Add more algorithms: ml.stack(data, target, algorithms=['xgboost', 'random_forest'], seed=42)"
        )

    # Accept fitted Model objects — extract algorithm name for re-fitting
    # Users familiar with sklearn StackingClassifier naturally pass fitted models.
    resolved: list[str] = []
    for m in models:
        if isinstance(m, Model):
            resolved.append(m.algorithm)
        elif isinstance(m, str):
            resolved.append(m)
        else:
            raise ConfigError(
                f"stack() models= accepts algorithm name strings or fitted Model objects. "
                f"Got {type(m).__name__}. "
                "Example: ml.stack(data, target, models=['xgboost', 'random_forest'], seed=42)"
            )
    models = resolved

    # Validate algorithm names against registry
    valid_algos = _engines.available()
    for m in models:
        if m not in valid_algos:
            raise ConfigError(
                f"algorithm='{m}' not available. Choose from: {valid_algos}"
            )

    # Default meta-learner
    if meta is None:
        meta = "logistic" if task == "classification" else "linear"

    if meta not in valid_algos:
        raise ConfigError(
            f"meta='{meta}' not available. Choose from: {valid_algos}"
        )

    # Prepare data
    drop_cols = [target] + ([weights] if weights else [])
    X = data.drop(columns=drop_cols)
    y = data[target]

    # Extract sample weights for meta-learner (A12)
    meta_sw = None
    if weights is not None:
        if weights not in data.columns:
            raise DataError(
                f"weights='{weights}' not found in data. Available columns: {data.columns.tolist()}"
            )
        meta_sw = data[weights].values.astype(np.float64)

    # Normalize features: if ANY base model is linear, use linear encoding (safe for all).
    # Trees work fine with one-hot + scaling; linear models NEED it.
    any_linear_base = any(m in _normalize.LINEAR_ALGORITHMS for m in models)
    norm_algo = "logistic" if any_linear_base else "xgboost"

    # Full-data normalization — used for final base model refitting and predictions.
    # Target encoding is derived from this state for consistent label mapping.
    norm_state = _normalize.prepare(X, y, algorithm=norm_algo, task=task)
    X_clean = norm_state.pop_train_data()
    if X_clean is None:
        X_clean = norm_state.transform(X)
    y_clean = norm_state.encode_target(y)

    # Validate algorithm names (after preparing norm_state to catch early)
    for algo in models:
        if algo not in _engines.available():
            raise ConfigError(
                f"algorithm='{algo}' not available. Choose from: {_engines.available()}"
            )

    # Build meta-learner
    try:
        meta_est = _engines.create(meta, task=task, seed=seed, engine=engine)
    except ConfigError as e:
        raise ConfigError(
            f"Cannot use meta='{meta}' as meta-learner for "
            f"{'classification' if task == 'classification' else 'regression'}. {e}"
        ) from e

    # Apply balance=True: set class_weight='balanced' on LogisticRegression meta-learner
    if balance and task == "classification":
        if hasattr(meta_est, "class_weight"):
            meta_est.set_params(class_weight="balanced")

    # NaN detection (on post-normalization features, before OOF loop)
    has_nan = X_clean.isna().any().any()
    if has_nan:
        n_nan = int(X_clean.isna().any(axis=1).sum())
        warnings.warn(
            f"{n_nan} rows have NaN features — imputing per fold for stacking. "
            "Base models in non-tree algorithms require complete features.",
            UserWarning, stacklevel=2,
        )

    # --- OOF loop with per-fold normalization (prevents preprocessing leakage) ---
    # Global normalization before OOF would leak test-fold statistics into training.
    # Each fold independently normalizes only on its training subset.
    from .split import _kfold, _stratified_kfold

    use_proba = task == "classification"
    n_classes = len(np.unique(y_clean)) if use_proba else 0

    if use_proba:
        meta_width = len(models) if n_classes == 2 else len(models) * n_classes
    else:
        meta_width = len(models)

    oof_meta = np.zeros((len(X), meta_width))
    # Track per-model OOF predictions for cv_score weighting (F1 W33)
    _model_oof_preds: list[list] = [[] for _ in models]  # list of (val_idx, preds)

    # A5: Build OOF column names upfront.
    # Duplicate algorithm names use {algo}_{i}_oof to distinguish them.
    # (amendment #16)
    from collections import Counter
    algo_counts = Counter(models)
    algo_seen: Counter = Counter()
    oof_col_names: list[str] = []
    for algo in models:
        prefix = f"{algo}_{algo_seen[algo]}" if algo_counts[algo] > 1 else algo
        algo_seen[algo] += 1
        if use_proba and n_classes > 2:
            for k in range(n_classes):
                oof_col_names.append(f"{prefix}_oof_class_{k}")
        else:
            oof_col_names.append(f"{prefix}_oof")

    if task == "classification":
        cv_splits = list(_stratified_kfold(y_clean, k=cv_folds, seed=seed))
    else:
        cv_splits = list(_kfold(len(X), k=cv_folds, seed=seed))

    import warnings as _warn

    # Track per-model fold failures for 6.8 silent failure handling
    _model_fail_counts: list[int] = [0] * len(models)

    with _warn.catch_warnings():
        _warn.filterwarnings("ignore", category=RuntimeWarning)
        _warn.filterwarnings("ignore", message=".*NaN.*")
        _warn.filterwarnings("ignore", message=".*Auto-scaling.*")
        _warn.filterwarnings("ignore", message=".*imbalance.*")

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_fold_train = X.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_clean = y_clean[train_idx]

            # Per-fold feature normalization — statistics from fold train only
            fold_norm = _normalize.prepare(
                X_fold_train, y.iloc[train_idx], algorithm=norm_algo, task=task,
            )
            Xft = fold_norm.pop_train_data()
            if Xft is None:
                Xft = fold_norm.transform(X_fold_train)
            Xfv = fold_norm.transform(X_fold_val)

            # Per-fold median imputation when NaN present
            # (sklearn RandomForest can't handle NaN; XGBoost can but works fine imputed)
            if has_nan and Xft.isna().any().any():
                from ._transforms import SimpleImputer
                fold_imp = SimpleImputer(strategy="median")
                Xft = pd.DataFrame(
                    fold_imp.fit_transform(Xft), columns=Xft.columns, index=Xft.index,
                )
                Xfv = pd.DataFrame(
                    fold_imp.transform(Xfv), columns=Xfv.columns, index=Xfv.index,
                )

            fold_parts = []
            for _mi, algo in enumerate(models):
                try:
                    fold_est = _engines.create(algo, task=task, seed=seed, engine=engine)
                    fold_est.fit(Xft, y_fold_clean)
                    if use_proba and hasattr(fold_est, "predict_proba"):
                        p = fold_est.predict_proba(Xfv)
                        # Store hard predictions for OOF scoring (cv_score weighting)
                        _model_oof_preds[_mi].append((val_idx, fold_est.predict(Xfv)))
                        if n_classes == 2:
                            fold_parts.append(p[:, 1:2])
                        else:
                            fold_parts.append(p)
                    else:
                        raw_pred = fold_est.predict(Xfv)
                        _model_oof_preds[_mi].append((val_idx, raw_pred))
                        fold_parts.append(raw_pred.reshape(-1, 1))
                except Exception as _fold_exc:
                    # 6.8: emit UserWarning per fold failure (not silent)
                    _warn.warn(
                        f"{algo} failed on fold {fold_idx}: {str(_fold_exc)}",
                        UserWarning,
                        stacklevel=2,
                    )
                    _model_fail_counts[_mi] += 1
                    _model_oof_preds[_mi].append((val_idx, np.zeros(len(val_idx))))
                    w = 1 if (n_classes == 2 or not use_proba) else n_classes
                    fold_parts.append(np.zeros((len(val_idx), w)))

            oof_meta[np.array(val_idx)] = np.hstack(fold_parts)

    # -----------------------------------------------------------------------
    # 6.8: Post-OOF failure analysis — drop mostly-failed models, raise if all fail
    # -----------------------------------------------------------------------
    from ._types import ModelError
    _surviving_model_indices: list[int] = []
    for _mi, algo in enumerate(models):
        fail_rate = _model_fail_counts[_mi] / cv_folds if cv_folds > 0 else 0.0
        if fail_rate > 0.5:
            warnings.warn(
                f"Base model '{algo}' failed on {_model_fail_counts[_mi]}/{cv_folds} folds "
                f"(>{50}%). Dropping from ensemble.",
                UserWarning,
                stacklevel=2,
            )
        else:
            _surviving_model_indices.append(_mi)

    if not _surviving_model_indices:
        raise ModelError(
            "All base models failed during OOF stacking. "
            "Check your data and algorithm choices."
        )

    # Rebuild models list and oof_meta columns using surviving models only
    if len(_surviving_model_indices) < len(models):
        _surviving_models = [models[i] for i in _surviving_model_indices]
        # Recompute oof column indices for surviving models
        if use_proba and n_classes > 2:
            _cols_per = n_classes
        else:
            _cols_per = 1
        _keep_cols = []
        for _mi in _surviving_model_indices:
            _start = _mi * _cols_per
            _keep_cols.extend(range(_start, _start + _cols_per))
        oof_meta = oof_meta[:, _keep_cols]
        models = _surviving_models
        _model_oof_preds = [_model_oof_preds[i] for i in _surviving_model_indices]
        # Recompute meta_width
        if use_proba:
            meta_width = len(models) if n_classes == 2 else len(models) * n_classes
        else:
            meta_width = len(models)
        # Rebuild oof_col_names for surviving models
        from collections import Counter as _Counter
        _algo_counts2 = _Counter(models)
        _algo_seen2: _Counter = _Counter()
        oof_col_names = []
        for _algo in models:
            _prefix = f"{_algo}_{_algo_seen2[_algo]}" if _algo_counts2[_algo] > 1 else _algo
            _algo_seen2[_algo] += 1
            if use_proba and n_classes > 2:
                for _k in range(n_classes):
                    oof_col_names.append(f"{_prefix}_oof_class_{_k}")
            else:
                oof_col_names.append(f"{_prefix}_oof")

    # -----------------------------------------------------------------------
    # cv_score weighting (F1 W33): scale OOF columns by base model OOF performance
    # -----------------------------------------------------------------------
    if _base_weights_strategy == "cv_score":
        _cv_weights: list[float] = []
        for _mi, _fold_list in enumerate(_model_oof_preds):
            if not _fold_list:
                _cv_weights.append(1.0)
                continue
            _all_idx = np.concatenate([np.array(v) for v, _ in _fold_list])
            _all_pred = np.concatenate([p for _, p in _fold_list])
            _sort_order = np.argsort(_all_idx)
            _all_idx = _all_idx[_sort_order]
            _all_pred = _all_pred[_sort_order]
            _y_oof = y_clean[_all_idx]
            if task == "classification":
                _score = float((_all_pred == _y_oof).mean())
            else:
                _ss_res = float(np.sum((_y_oof - _all_pred) ** 2))
                _ss_tot = float(np.sum((_y_oof - _y_oof.mean()) ** 2))
                _score = max(0.0, 1.0 - _ss_res / _ss_tot) if _ss_tot > 0 else 0.0
            _cv_weights.append(_score)

        _cv_weights_arr = np.array(_cv_weights, dtype=np.float64)
        _total = _cv_weights_arr.sum()
        if _total <= 0:
            warnings.warn(
                "stack(weights='cv_score'): all base models scored 0 — "
                "falling back to equal weights.",
                UserWarning, stacklevel=2,
            )
            _cv_weights_arr = np.ones(len(models), dtype=np.float64) / len(models)
        else:
            _cv_weights_arr = _cv_weights_arr / _total

        # Scale each model's OOF column block by its weight
        _col_per_model = meta_width // len(models)
        for _mi, _w in enumerate(_cv_weights_arr):
            _start = _mi * _col_per_model
            _end = _start + _col_per_model
            oof_meta[:, _start:_end] *= _w

    # -----------------------------------------------------------------------
    # Multi-level stacking (3.2): optional second OOF generation phase
    # -----------------------------------------------------------------------
    fitted_level1_bases: list | None = None

    if levels >= 2:
        # Level-1 features = level-0 OOF predictions (+ original features if passthrough)
        X_l1 = oof_meta.copy()
        if passthrough:
            X_l1 = np.hstack([X_l1, X_clean.values])

        # Column names for level-1 feature DataFrame
        l1_col_names = [f"oof_{i}" for i in range(oof_meta.shape[1])]
        if passthrough:
            l1_col_names = l1_col_names + list(X.columns)
        X_l1_df = pd.DataFrame(X_l1, columns=l1_col_names)

        oof_meta2 = np.zeros((len(X), meta_width))

        with _warn.catch_warnings():
            _warn.filterwarnings("ignore", category=RuntimeWarning)
            _warn.filterwarnings("ignore", message=".*NaN.*")
            _warn.filterwarnings("ignore", message=".*Auto-scaling.*")
            _warn.filterwarnings("ignore", message=".*imbalance.*")

            for train_idx2, val_idx2 in cv_splits:
                Xl1_fold_train = X_l1_df.iloc[train_idx2]
                Xl1_fold_val = X_l1_df.iloc[val_idx2]
                y_fold2_clean = y_clean[train_idx2]

                fold_parts2 = []
                for algo in models:
                    try:
                        fold_est2 = _engines.create(algo, task=task, seed=seed + 1000, engine=engine)
                        fold_est2.fit(Xl1_fold_train, y_fold2_clean)
                        if use_proba and hasattr(fold_est2, "predict_proba"):
                            p2 = fold_est2.predict_proba(Xl1_fold_val)
                            if n_classes == 2:
                                fold_parts2.append(p2[:, 1:2])
                            else:
                                fold_parts2.append(p2)
                        else:
                            fold_parts2.append(
                                fold_est2.predict(Xl1_fold_val).reshape(-1, 1)
                            )
                    except Exception:
                        w2 = 1 if (n_classes == 2 or not use_proba) else n_classes
                        fold_parts2.append(np.zeros((len(val_idx2), w2)))

                oof_meta2[np.array(val_idx2)] = np.hstack(fold_parts2)

        # Refit level-1 models on full level-1 data
        fitted_level1_bases = []
        with _warn.catch_warnings():
            _warn.filterwarnings("ignore", category=RuntimeWarning)
            _warn.filterwarnings("ignore", message=".*NaN.*")
            _warn.filterwarnings("ignore", message=".*Auto-scaling.*")
            for algo in models:
                try:
                    est2 = _engines.create(algo, task=task, seed=seed + 1000, engine=engine)
                    est2.fit(X_l1_df, y_clean)
                    fitted_level1_bases.append((algo, est2))
                except ConfigError as e:
                    raise ConfigError(
                        f"Cannot use algorithm='{algo}' as level-1 model. {e}"
                    ) from e

        # Meta-learner trains on level-1 OOF; store level-1 OOF as oof_df
        meta_input = oof_meta2
        oof_df = pd.DataFrame(oof_meta2, columns=oof_col_names, index=X.index)

    elif passthrough:
        # levels=1 + passthrough: concat OOF with original features for meta
        meta_input = np.hstack([oof_meta, X_clean.values])
        oof_df = pd.DataFrame(oof_meta, columns=oof_col_names, index=X.index)

    else:
        # Standard levels=1 (default)
        meta_input = oof_meta
        # A5: Build OOF DataFrame with named columns, aligned to training index.
        oof_df = pd.DataFrame(oof_meta, columns=oof_col_names, index=X.index)

    # Train meta-learner on meta_input (A12: pass sample weights if provided)
    with _warn.catch_warnings():
        _warn.filterwarnings("ignore", category=RuntimeWarning)
        meta_fit_kwargs = {"sample_weight": meta_sw} if meta_sw is not None else {}
        meta_est.fit(meta_input, y_clean, **meta_fit_kwargs)

    # Refit all base models on FULL training data (full-data/global normalization).
    #
    # Normalization strategy note (Phase 0.4):
    # - OOF loop above uses per-fold normalization → unbiased meta-learner training.
    # - Final refit here uses global normalization (same X_clean used at predict time).
    # - For scale-sensitive base models (SVM/KNN/logistic), the OOF predictions
    #   come from per-fold normalized models while inference uses globally normalized
    #   models. This distribution mismatch is a known limitation. The meta-learner
    #   learns to compensate partially. For minimal mismatch, prefer tree-based base
    #   models (xgboost, random_forest) which are scale-invariant.
    fitted_bases = []
    with _warn.catch_warnings():
        _warn.filterwarnings("ignore", category=RuntimeWarning)
        _warn.filterwarnings("ignore", message=".*NaN.*")
        _warn.filterwarnings("ignore", message=".*Auto-scaling.*")
        for algo in models:
            try:
                est = _engines.create(algo, task=task, seed=seed, engine=engine)
                est.fit(X_clean, y_clean)
                fitted_bases.append((algo, est))
            except ConfigError as e:
                raise ConfigError(
                    f"Cannot use algorithm='{algo}' as base model for "
                    f"{'classification' if task == 'classification' else 'regression'} "
                    f"stacking. {e}"
                ) from e

    # Fit a full-data imputer for NaN handling at predict time
    nan_imputer = None
    if has_nan:
        from ._transforms import SimpleImputer
        nan_imputer = SimpleImputer(strategy="median")
        nan_imputer.fit(X_clean)  # SimpleImputer computes medians ignoring NaN natively

    # Build custom stacking model
    fitted_model = _StackingModel(
        estimators=fitted_bases,
        final_estimator_=meta_est,
        use_proba=use_proba,
        n_classes=n_classes,
        nan_imputer=nan_imputer,
        level1_estimators=fitted_level1_bases,
        passthrough=passthrough,
    )

    # Compute OOF cv_score: meta-learner predictions on OOF meta-features.
    # meta_input rows are derived from base-model OOF predictions (each row
    # was predicted by models trained without that row), so this is a fair
    # out-of-fold estimate of ensemble performance — consistent with how
    # Model.cv_score is populated for regular fit() CV path.
    oof_cv_score: float | None = None
    try:
        oof_preds = meta_est.predict(meta_input)
        if task == "classification":
            oof_cv_score = float((oof_preds == y_clean).mean())
        else:
            ss_res = float(np.sum((y_clean - oof_preds) ** 2))
            ss_tot = float(np.sum((y_clean - y_clean.mean()) ** 2))
            oof_cv_score = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    except Exception:
        oof_cv_score = None

    return Model(
        _model=fitted_model,
        _task=task,
        _algorithm="stacked",
        _features=list(X.columns),
        _target=target,
        _seed=seed,
        _label_encoder=norm_state.label_encoder,
        _feature_encoder=norm_state,
        _n_train=len(data),
        _oof_predictions=oof_df,
        _base_models=fitted_bases,
        _holdout_score=oof_cv_score,
    )


class _StackingModel:
    """Custom stacking estimator with per-fold normalization in OOF.

    Replaces sklearn StackingClassifier/StackingRegressor to prevent
    preprocessing leakage: normalization statistics are computed per fold
    during OOF, not on the full training set.

    At predict time, receives X_clean (already normalized by
    Model._feature_encoder). Stacks base model outputs → meta-learner.

    `.estimators` and `.final_estimator_` match sklearn's StackingClassifier
    attribute names so explain() works without changes.
    """

    def __init__(self, estimators, final_estimator_, use_proba, n_classes,
                 nan_imputer=None, level1_estimators=None, passthrough=False):
        self.estimators = estimators        # list of (name, fitted_est)
        self.final_estimator_ = final_estimator_
        self.use_proba = use_proba
        self.n_classes = n_classes
        self.nan_imputer = nan_imputer      # SimpleImputer or None
        self.level1_estimators = level1_estimators  # list or None (levels=2)
        self.passthrough = passthrough
        # Expose classes_ and _estimator_type so CalibratedClassifierCV works
        if use_proba:
            self._estimator_type = "classifier"
            if hasattr(final_estimator_, "classes_"):
                self.classes_ = final_estimator_.classes_
        else:
            self._estimator_type = "regressor"

    def _make_meta(self, X):
        # Impute NaN if needed (tree models pass NaN, RF can't handle it)
        if self.nan_imputer is not None and hasattr(X, "isna") and X.isna().any().any():
            is_df = isinstance(X, pd.DataFrame)
            cols = X.columns if is_df else None
            idx = X.index if is_df else None
            arr = self.nan_imputer.transform(X)
            X = pd.DataFrame(arr, columns=cols, index=idx) if is_df else arr

        # Level-0: compute base model predictions
        parts = []
        for _, est in self.estimators:
            if self.use_proba and hasattr(est, "predict_proba"):
                p = est.predict_proba(X)
                if self.n_classes == 2:
                    parts.append(p[:, 1:2])
                else:
                    parts.append(p)
            else:
                parts.append(est.predict(X).reshape(-1, 1))
        level0_meta = np.hstack(parts) if parts else np.zeros((len(X), 0))

        if self.level1_estimators is not None:
            # levels=2: apply level-1 models on level-0 meta (+ X if passthrough)
            X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            X_l1 = np.hstack([level0_meta, X_arr]) if self.passthrough else level0_meta

            parts1 = []
            for _, est1 in self.level1_estimators:
                if self.use_proba and hasattr(est1, "predict_proba"):
                    p1 = est1.predict_proba(X_l1)
                    if self.n_classes == 2:
                        parts1.append(p1[:, 1:2])
                    else:
                        parts1.append(p1)
                else:
                    parts1.append(est1.predict(X_l1).reshape(-1, 1))
            return np.hstack(parts1) if parts1 else level0_meta

        elif self.passthrough:
            # levels=1 + passthrough: concat level-0 meta with original features
            X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            return np.hstack([level0_meta, X_arr])

        else:
            return level0_meta

    def fit(self, X, y=None):
        """No-op fit() — _StackingModel is already fitted during stack().

        Required by sklearn's CalibratedClassifierCV parameter validation
        even when cv='prefit'. Returns self to satisfy sklearn API contract.
        """
        return self

    def predict(self, X):
        return self.final_estimator_.predict(self._make_meta(X))

    def predict_proba(self, X):
        return self.final_estimator_.predict_proba(self._make_meta(X))
