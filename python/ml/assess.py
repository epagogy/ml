"""Assess model performance on held-out test data.

The final exam — do once.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd

from .evaluate import evaluate

if TYPE_CHECKING:
    from ._types import Metrics, Model

# Module-level tracking: detect multi-model assessment on same test data.
# Key = (n_rows, n_cols, first_value_hash), Value = count of assess calls.
# Addresses known design gap: assess counter is per-model, not per-test-set.
_test_set_tracker: dict[tuple, int] = {}


def assess(
    model: Model,
    *,
    test: pd.DataFrame,
    metrics: dict | None = None,
    intervals: bool = False,
) -> Metrics:
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
        Merged with auto-selected built-in metrics. Same as evaluate(metrics=).
    intervals : bool, default=False
        If True, compute 95% bootstrap confidence intervals for each metric.

    Returns
    -------
    Metrics
        Dict of metric_name -> value (same metrics as evaluate, plus custom).
        When intervals=True, also includes metric_lower/metric_upper for each metric.

    Raises
    ------
    ModelError
        If assess() called 2+ times on same model. Repeated peeking at test
        data inflates apparent performance. Use ml.evaluate() for iteration.
    ConfigError
        If model is not a fitted Model or TuningResult.

    Examples
    --------
    >>> verdict = ml.assess(model, test=s.test)
    >>> verdict
    {'accuracy': 0.83, 'f1': 0.80, 'precision': 0.81, 'recall': 0.79, 'roc_auc': 0.86}
    """
    from ._compat import to_pandas
    from ._types import ConfigError, Model, ModelError, TuningResult

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

    # Partition guard — reject non-test data (must fire BEFORE assess_count
    # increment, so a wrong-partition error doesn't burn the one-shot counter)
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

    # Raise on repeat calls — warning is swallowed by Jupyter>=7 (P0-A fix)
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

    # Cross-model test-set tracking: detect cherry-picking across seeds.
    # Seed inflation scales as log(K) without bound (Roth 2026a, Exp AP).
    # Reporting best-of-K seeds inflates AUC by 0.003*log(K) on average.
    try:
        _fingerprint = (len(test), len(test.columns), hash(test.iloc[0].values.tobytes()))
    except Exception:
        _fingerprint = (len(test), len(test.columns))
    _test_set_tracker[_fingerprint] = _test_set_tracker.get(_fingerprint, 0) + 1
    if _test_set_tracker[_fingerprint] > 1:
        n_assessments = _test_set_tracker[_fingerprint]
        warnings.warn(
            f"assess() called {n_assessments} times on same test set (different models). "
            f"Seed cherry-picking inflates AUC by ~{0.003 * __import__('math').log(n_assessments):.4f} "
            "on average (Roth 2026a). Report mean across seeds, not best.",
            UserWarning,
            stacklevel=2,
        )

    # Use evaluate() for metrics (same logic, different intent)
    result = evaluate(model, test, metrics=metrics, intervals=intervals, _guard=False)
    # Wrap in sealed Evidence type — not substitutable for Metrics (Codd condition 7)
    from ._types import Evidence
    return Evidence(result, task=result._task, time=result._time)
