"""Shared prediction-and-metrics logic used by evaluate and assess.

Private module — not part of the public API.
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .fit import _compute_metrics

if TYPE_CHECKING:
    from ._types import Metrics, Model


def _score(
    model: Model,
    data: pd.DataFrame,
    *,
    metrics: dict | None = None,
    intervals: bool = False,
    se: bool = False,
    sample_weight: str | None = None,
) -> Metrics:
    """Predict on data and compute metrics. No guards — caller is responsible.

    Parameters
    ----------
    model : Model
        Fitted model (already unwrapped from TuningResult by caller).
    data : pd.DataFrame
        DataFrame with target column present.
    metrics : dict or None
        Custom scoring functions.
    intervals : bool
        Bootstrap confidence intervals.
    se : bool
        Bootstrap standard errors.
    sample_weight : str or None
        Column name for sample weights.

    Returns
    -------
    Metrics
        Dict of metric_name -> value.
    """
    from ._types import ConfigError, DataError, Metrics, TuningResult

    # Unwrap TuningResult → Model
    if isinstance(model, TuningResult):
        model = model.best_model

    # Normalize metrics= parameter:
    # - list of strings → filter built-in results to those keys after compute
    # - dict → custom callables added to results (existing behaviour)
    _metrics_filter: list[str] | None = None
    if metrics is not None:
        if isinstance(metrics, list):
            _metrics_filter = list(metrics)
            metrics = None  # no custom callables; filtering applied at the end
        elif not isinstance(metrics, dict):
            raise ConfigError(
                f"metrics= must be a dict of name → callable, or a list of metric names. "
                f"Got {type(metrics).__name__}. "
                f"Example: ml.evaluate(model, data, metrics=['accuracy', 'roc_auc'])"
            )
        else:
            for name, fn in metrics.items():
                if not callable(fn):
                    raise ConfigError(
                        f"metrics['{name}'] must be callable. Got {type(fn).__name__}."
                    )

    # Validate target present
    if model._target not in data.columns:
        available = data.columns.tolist()
        raise DataError(
            f"target column '{model._target}' not found in data. Available: {available}"
        )

    # Guard: warn if data looks like training data
    if hasattr(model, "_n_train") and model._n_train is not None:
        if len(data) == model._n_train:
            warnings.warn(
                f"Data has {len(data)} rows — same as training data. "
                "Are you evaluating on train instead of valid/test? "
                "Metrics on training data are inflated.",
                UserWarning,
                stacklevel=3,
            )

    t0 = time.perf_counter()

    y_true = data[model._target]

    # Transform features ONCE — used for both predictions and proba
    X = data.drop(columns=[model._target])
    X = X[model._features]

    # Warn if any feature columns are entirely NaN (data pipeline bug)
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if len(all_nan_cols) > 0:
        if len(all_nan_cols) == len(X.columns):
            warnings.warn(
                f"All {len(X.columns)} feature columns are entirely NaN. "
                "Metrics will be meaningless. Check your data pipeline.",
                UserWarning,
                stacklevel=3,
            )
        else:
            warnings.warn(
                f"{len(all_nan_cols)} feature column(s) entirely NaN: "
                f"{sorted(all_nan_cols)[:5]}. Check your data pipeline.",
                UserWarning,
                stacklevel=3,
            )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*NaN.*")
        warnings.filterwarnings("ignore", message=".*Auto-scaling.*")
        warnings.filterwarnings("ignore", message=".*auto-imputed.*")
        X_clean = model._feature_encoder.transform(X)

        # Apply custom preprocessor if present (Hook 4: Lego)
        if model._preprocessor is not None:
            X_clean = model._preprocessor(X_clean)

    # Predict from transformed features (avoids double transform via model.predict())
    if getattr(model, "_threshold", None) is not None and model._task == "classification":
        import contextlib
        with contextlib.suppress(AttributeError):
            _proba = model._model.predict_proba(X_clean).astype(np.float64)
            row_sums = _proba.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            _proba = _proba / row_sums
            pos_proba = _proba[:, -1]
            raw_classes = model._model.classes_
            predictions = np.where(
                pos_proba >= model._threshold, raw_classes[-1], raw_classes[0]
            )
    else:
        predictions = model._model.predict(X_clean)

    decoded = model._feature_encoder.decode(predictions)
    y_pred = pd.Series(decoded.values, index=data.index, name=model._target)
    if model._task == "regression" and y_pred.dtype != np.float64:
        y_pred = y_pred.astype(np.float64)

    # Warn on single-class validation set (P0-B fix) — metrics will be 0 or absent
    if model._task == "classification":
        n_unique = y_true.nunique()
        n_model_classes = len(model._model.classes_) if hasattr(model._model, "classes_") else 2
        if n_unique < n_model_classes:
            warnings.warn(
                f"Validation set contains only {n_unique} class(es) out of "
                f"{n_model_classes} the model was trained on. "
                "Metrics like roc_auc and f1 will be 0.0 or absent. "
                "Check class balance or use stratify=True in ml.split().",
                UserWarning,
                stacklevel=3,
            )

    # Precompute proba once (used by metrics, SE bootstrap, and CI bootstrap)
    _eval_proba = None
    if model._task == "classification" and hasattr(model._model, "predict_proba"):
        import contextlib
        with contextlib.suppress(Exception):
            _eval_proba = model._model.predict_proba(X_clean)

    # Compute built-in metrics (reuse fit.py logic)
    # Suppress RuntimeWarnings from sklearn's logistic regression forward pass
    # (stacked models use LogisticRegression meta-learner → matmul overflow/divide-by-zero)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            result = _compute_metrics(y_true, y_pred, model._task, model._model, X_clean, proba=_eval_proba)
        except TypeError as exc:
            raise DataError(
                f"Label type mismatch: model predicts "
                f"{type(y_pred.iloc[0]).__name__} but test data has "
                f"{type(y_true.iloc[0]).__name__} labels for target "
                f"'{model._target}'. Ensure test data uses the same label "
                f"types as training data."
            ) from exc

    # Weighted metrics (A12: sample_weight= param)
    if sample_weight is not None:
        if not isinstance(sample_weight, str):
            raise ConfigError(
                f"sample_weight must be a column name string in data, "
                f"got {type(sample_weight).__name__}. "
                "Example: ml.evaluate(model, data, sample_weight='weight_col')"
            )
        if sample_weight not in data.columns:
            raise DataError(
                f"sample_weight='{sample_weight}' not found in data. "
                f"Available columns: {data.columns.tolist()}"
            )
        sw = data[sample_weight].values.astype(np.float64)
        _yt = np.asarray(y_true, dtype=np.float64 if model._task != "classification" else None)
        _yp = np.asarray(y_pred, dtype=np.float64 if model._task != "classification" else None)
        if model._task == "classification":
            result["accuracy_weighted"] = float(np.average(np.asarray(y_true) == np.asarray(y_pred), weights=sw))
        else:
            _diff = _yt - _yp
            result["rmse_weighted"] = float(np.sqrt(np.average(_diff ** 2, weights=sw)))
            result["mae_weighted"] = float(np.average(np.abs(_diff), weights=sw))
            _ss_res_w = np.average(_diff ** 2, weights=sw)
            _ss_tot_w = np.average((_yt - np.average(_yt, weights=sw)) ** 2, weights=sw)
            result["r2_weighted"] = float(1 - _ss_res_w / _ss_tot_w) if _ss_tot_w > 0 else 0.0

    # Custom metrics (Hook 2: Lego)
    if metrics:
        for name, fn in metrics.items():
            try:
                result[name] = float(fn(y_true, y_pred))
            except Exception as e:
                err_str = str(e)
                hint = ""
                if "float" in err_str and model._task == "classification":
                    hint = (
                        " Metrics like log_loss need probabilities, not class labels. "
                        "Use: model.predict_proba(data) to get probabilities."
                    )
                warnings.warn(
                    f"Custom metric '{name}' failed: {e}.{hint}",
                    UserWarning,
                    stacklevel=3,
                )

    # Lightweight standard error (opt-in, 100 bootstrap iterations)
    if se:
        n = len(y_true)
        n_boot_se = 100
        rng_se = np.random.RandomState(getattr(model, "_seed", 42) + 1)
        se_results: dict[str, list[float]] = {k: [] for k in result}
        for _ in range(n_boot_se):
            idx = rng_se.randint(0, n, size=n)
            yt_b = y_true.iloc[idx].reset_index(drop=True)
            yp_b = y_pred.iloc[idx].reset_index(drop=True)
            if model._task == "classification" and yt_b.nunique() < 2:
                continue
            try:
                _boot_proba = _eval_proba[idx] if _eval_proba is not None else None
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    boot_m = _compute_metrics(yt_b, yp_b, model._task, model._model,
                                              X_clean.iloc[idx].reset_index(drop=True),
                                              proba=_boot_proba)
                for k in result:
                    if k in boot_m:
                        se_results[k].append(boot_m[k])
            except Exception:
                continue
        for k, vals in se_results.items():
            if len(vals) >= 10:
                result[f"{k}_se"] = float(np.std(vals))

    # Bootstrap confidence intervals (opt-in)
    if intervals:
        n = len(y_true)
        n_boot = 1000
        rng = np.random.RandomState(getattr(model, "_seed", 42))
        boot_results: dict[str, list[float]] = {k: [] for k in result}
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            yt_b = y_true.iloc[idx].reset_index(drop=True)
            yp_b = y_pred.iloc[idx].reset_index(drop=True)
            # Skip degenerate samples (e.g., only one class)
            if model._task == "classification" and yt_b.nunique() < 2:
                continue
            try:
                _boot_proba_ci = _eval_proba[idx] if _eval_proba is not None else None
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    boot_m = _compute_metrics(yt_b, yp_b, model._task, model._model, X_clean.iloc[idx].reset_index(drop=True), proba=_boot_proba_ci)
                for k in result:
                    if k in boot_m:
                        boot_results[k].append(boot_m[k])
            except Exception:
                continue
        for k, vals in boot_results.items():
            if len(vals) >= 100:
                arr = np.array(vals)
                result[f"{k}_lower"] = float(np.percentile(arr, 2.5))
                result[f"{k}_upper"] = float(np.percentile(arr, 97.5))

    # Coverage warning for uncertainty estimates (V3 AO: fold-SE achieves ~56% coverage)
    if se or intervals:
        warnings.warn(
            "Bootstrap CI/SE is more reliable than fold-level SE, but CV uncertainty "
            "estimates typically achieve ~56-70% actual coverage at nominal 95% "
            "(Bengio & Grandvalet 2004, Bates et al. 2023). Interpret with caution.",
            UserWarning,
            stacklevel=3,
        )

    elapsed = time.perf_counter() - t0

    # Warn if accuracy is misleading due to class imbalance
    balanced = getattr(model, "_balance", False)
    if model._task == "classification" and "accuracy" in result and not balanced:
        counts = y_true.value_counts()
        if len(counts) >= 2:
            ratio = counts.iloc[0] / counts.iloc[-1]
            if ratio >= 5.0:
                warnings.warn(
                    f"Class imbalance ({ratio:.0f}:1). Accuracy ({result['accuracy']:.2f}) "
                    f"may be misleading — a model predicting only the majority class "
                    f"scores {counts.iloc[0]/len(y_true):.0%}. "
                    f"Prefer roc_auc or f1 for imbalanced data. "
                    f"Consider balance=True in fit().",
                    UserWarning,
                    stacklevel=3,
                )

    # Filter to requested metric names if metrics= was a list of strings
    if _metrics_filter is not None:
        unknown = [k for k in _metrics_filter if k not in result]
        if unknown:
            warnings.warn(
                f"metrics= requested keys not found in results: {unknown}. "
                f"Available: {list(result)}",
                UserWarning,
                stacklevel=3,
            )
        result = {k: v for k, v in result.items() if k in _metrics_filter}

    return Metrics(
        {k: round(v, 4) for k, v in result.items()},
        task=model._task,
        time=round(elapsed, 2),
    )
