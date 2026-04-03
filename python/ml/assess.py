"""Assess model performance on held-out test data.

The final exam — do once.

No call relationship with evaluate(). Both call _compute_metrics for
scoring, but assess and evaluate are independent primitives.
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ._types import Evidence, Model


def assess(
    model: Model,
    *,
    test: pd.DataFrame,
    metrics: dict | None = None,
    intervals: bool = False,
) -> Evidence:
    """Assess model on held-out test data.

    Uses keyword-only test= — intentional friction for the final exam.
    The final verdict (final exam — do once).

    Separate verb from evaluate() to force conscious choice:
    "This is the final verdict."

    Parameters
    ----------
    model : Model or TuningResult
        Fitted model to assess.
    test : pd.DataFrame
        Test DataFrame (keyword-only, deliberate friction).
    metrics : dict or None
        Custom scoring functions. Dict of name -> callable(y_true, y_pred) -> float.
        Merged with auto-selected built-in metrics.
    intervals : bool, default=False
        If True, compute 95% bootstrap confidence intervals for each metric.

    Returns
    -------
    Evidence
        Sealed dict of metric_name -> value. Not substitutable for Metrics.
        When intervals=True, also includes metric_lower/metric_upper.

    Raises
    ------
    PartitionError
        If test partition has already been assessed (regardless of which model).
        Each test holdout gets one assessment. This is the terminal constraint.
    ModelError
        If assess() called 2+ times on same model.
    ConfigError
        If model is not a fitted Model or TuningResult.

    Examples
    --------
    >>> verdict = ml.assess(model, test=s.test)
    >>> verdict
    {'accuracy': 0.83, 'f1': 0.80, 'precision': 0.81, 'recall': 0.79, 'roc_auc': 0.86}
    """
    from ._compat import to_pandas
    from ._types import ConfigError, DataError, Model, ModelError, TuningResult

    # Auto-convert Polars/other DataFrames to pandas
    test = to_pandas(test)

    # Validate model type before any attribute access
    if not isinstance(model, (Model, TuningResult)):
        raise ConfigError(
            f"assess() requires a fitted Model or TuningResult, got {type(model).__name__}. "
            "Use: ml.assess(model, test=s.test) where model = ml.fit(s.dev, 'target', seed=42)"
        )

    # Unwrap TuningResult → Model
    if isinstance(model, TuningResult):
        model = model.best_model

    # Partition guard — reject non-test data AND already-assessed partitions.
    # Must fire BEFORE assess_count increment, so a wrong-partition error
    # doesn't burn the one-shot counter.
    from ._provenance import check_provenance, guard_assess
    guard_assess(test)
    # Layer 2: Cross-verb provenance check (same split lineage)
    check_provenance(getattr(model, "_provenance", {}), test)

    # DFA state transition: FITTED/EVALUATED → ASSESSED (terminal)
    import contextlib

    from ._types import check_workflow_transition
    with contextlib.suppress(Exception):
        model._workflow_state = check_workflow_transition(
            model._workflow_state, "assess"
        )

    # Per-model assess count (same model, second call → raise)
    model._assess_count += 1
    if model._assess_count > 1:
        raise ModelError(
            f"assess() called {model._assess_count} times on same model. "
            "Repeated peeking at test data inflates apparent performance and "
            "violates the final-exam contract. "
            "Use ml.evaluate() for iteration. assess() is a one-time ceremony."
        )

    # Warn on small test sets (unreliable estimates)
    if len(test) < 30:
        warnings.warn(
            f"Test set has only {len(test)} rows. "
            "Statistical estimates (accuracy, AUC) may be unreliable with <30 samples. "
            "Consider using cross-validation for small datasets.",
            UserWarning,
            stacklevel=2,
        )

    # Validate target present
    if model._target not in test.columns:
        available = test.columns.tolist()
        raise DataError(
            f"target column '{model._target}' not found in test data. Available: {available}"
        )

    # Guard: empty test set
    if len(test) == 0:
        raise DataError(
            "Cannot assess on empty test data (0 rows). "
            "Check your data splitting."
        )

    t0 = time.perf_counter()

    # ── Scoring (same _compute_metrics as evaluate, no call to evaluate) ──

    y_true = test[model._target]

    # Transform features
    X = test.drop(columns=[model._target])
    X = X[model._features]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*NaN.*")
        warnings.filterwarnings("ignore", message=".*Auto-scaling.*")
        warnings.filterwarnings("ignore", message=".*auto-imputed.*")
        X_clean = model._feature_encoder.transform(X)

        if model._preprocessor is not None:
            X_clean = model._preprocessor(X_clean)

    # Predict
    if getattr(model, "_threshold", None) is not None and model._task == "classification":
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
    y_pred = pd.Series(decoded.values, index=test.index, name=model._target)
    if model._task == "regression" and y_pred.dtype != np.float64:
        y_pred = y_pred.astype(np.float64)

    # Precompute proba
    _eval_proba = None
    if model._task == "classification" and hasattr(model._model, "predict_proba"):
        with contextlib.suppress(Exception):
            _eval_proba = model._model.predict_proba(X_clean)

    # Compute metrics
    from .fit import _compute_metrics

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            result = _compute_metrics(y_true, y_pred, model._task, model._model,
                                      X_clean, proba=_eval_proba)
        except TypeError as exc:
            raise DataError(
                f"Label type mismatch: model predicts "
                f"{type(y_pred.iloc[0]).__name__} but test data has "
                f"{type(y_true.iloc[0]).__name__} labels for target "
                f"'{model._target}'. Ensure test data uses the same label "
                f"types as training data."
            ) from exc

    # Custom metrics
    if metrics:
        for name, fn in metrics.items():
            try:
                result[name] = float(fn(y_true, y_pred))
            except Exception as e:
                warnings.warn(
                    f"Custom metric '{name}' failed: {e}.",
                    UserWarning,
                    stacklevel=2,
                )

    # Bootstrap confidence intervals
    if intervals:
        n = len(y_true)
        n_boot = 1000
        rng = np.random.RandomState(getattr(model, "_seed", 42))
        boot_results: dict[str, list[float]] = {k: [] for k in result}
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
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
                        boot_results[k].append(boot_m[k])
            except Exception:
                continue
        for k, vals in boot_results.items():
            if len(vals) >= 100:
                arr = np.array(vals)
                result[f"{k}_lower"] = float(np.percentile(arr, 2.5))
                result[f"{k}_upper"] = float(np.percentile(arr, 97.5))

    elapsed = time.perf_counter() - t0

    # K = number of evaluate() calls on the valid partition from the same split.
    # Measures selection pressure: how many times the practitioner steered on
    # validation data before committing to this assessment.
    from ._provenance import _registry, get_split_id
    _test_split_id = get_split_id(test)
    _K = 0
    if _test_split_id is not None:
        _K = _registry.get_eval_count_by_role(_test_split_id, "valid")

    # Wrap in sealed Evidence type — not substitutable for Metrics
    from ._types import Evidence
    return Evidence(
        {k: round(v, 4) for k, v in result.items()},
        task=model._task,
        time=round(elapsed, 2),
        K=_K,
    )
