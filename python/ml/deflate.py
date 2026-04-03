"""deflate() — multiple testing correction for model selection.

You screen() 14 algorithms, compare() the best 5 — that's 5 hypothesis
tests on the same data.  The "winner" may just be lucky.

deflate() applies Benjamini-Hochberg (or Bonferroni) correction to
compare() output and tells you which models are *significantly* better
than the baseline after accounting for the number of models tested.

Usage:
    >>> lb = ml.compare([m1, m2, m3, m4], data=s.valid)
    >>> deflated = ml.deflate(lb, data=s.valid)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._types import ConfigError, DataError


def deflate(
    leaderboard,
    *,
    data: pd.DataFrame,
    method: str = "bh",
    alpha: float = 0.05,
    baseline: str = "best",
) -> pd.DataFrame:
    """Adjust model comparison for multiple testing.

    Re-evaluates models from a compare() Leaderboard on the given data,
    computes per-model p-values vs the baseline, and applies correction.

    Args:
        leaderboard: Output from ml.compare(). Must have .models attribute.
        data: Evaluation data (same as used in compare, or held-out).
        method: Correction method.
            "bh" — Benjamini-Hochberg (controls FDR, less conservative).
            "bonferroni" — Bonferroni (controls FWER, more conservative).
            "holm" — Holm-Bonferroni (step-down, tighter than Bonferroni).
        alpha: Significance level (default 0.05).
        baseline: Which model to test against.
            "best" — compare each model to the top-ranked model (default).
            "worst" — compare each model to the worst-ranked model.

    Returns:
        DataFrame with original metrics plus columns:
            p_value: raw p-value vs baseline
            p_adjusted: corrected p-value
            significant: bool — survives correction at given alpha

    Raises:
        ConfigError: If leaderboard has no .models or method is unknown.
        DataError: If data doesn't contain the target column.
    """
    from . import _stats
    from .predict import _predict_impl

    # Validate inputs
    if not hasattr(leaderboard, "_models") or not leaderboard._models:
        raise ConfigError(
            "deflate() requires a Leaderboard from ml.compare() with .models attached."
        )

    models = leaderboard._models
    if len(models) < 2:
        raise ConfigError("deflate() requires at least 2 models to compare.")

    if method not in ("bh", "bonferroni", "holm"):
        raise ConfigError(
            f"method={method!r} not recognized. Use 'bh', 'bonferroni', or 'holm'."
        )

    target = models[0]._target
    if target not in data.columns:
        raise DataError(
            f"target={target!r} not found in data. "
            f"Available: {list(data.columns)}"
        )

    task = models[0]._task
    y_true = data[target].values

    # Get predictions for each model
    preds = []
    for m in models:
        try:
            preds.append(_predict_impl(m, data).values)
        except Exception:
            preds.append(None)

    # Select baseline index
    if baseline == "best":
        base_idx = 0
    elif baseline == "worst":
        base_idx = len(models) - 1
    else:
        raise ConfigError(f"baseline={baseline!r} not recognized. Use 'best' or 'worst'.")

    base_preds = preds[base_idx]
    if base_preds is None:
        raise ConfigError("Baseline model failed prediction — cannot compute p-values.")

    # Compute raw p-values: each model vs baseline
    raw_pvals = []
    for i, pred in enumerate(preds):
        if i == base_idx or pred is None:
            raw_pvals.append(np.nan)
            continue
        try:
            if task == "classification":
                # McNemar's test on disagreement table
                best_correct = (base_preds == y_true).astype(int)
                model_correct = (pred == y_true).astype(int)
                b = int(np.sum((best_correct == 1) & (model_correct == 0)))
                c = int(np.sum((best_correct == 0) & (model_correct == 1)))
                if b + c == 0:
                    raw_pvals.append(1.0)  # identical predictions
                else:
                    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                    raw_pvals.append(float(_stats.chi2_sf(chi2, 1)))
            else:
                # Paired t-test on absolute errors
                best_errs = np.abs(base_preds.astype(float) - y_true.astype(float))
                model_errs = np.abs(pred.astype(float) - y_true.astype(float))
                _, p = _stats.ttest_rel(best_errs, model_errs)
                raw_pvals.append(float(p))
        except Exception:
            raw_pvals.append(np.nan)

    # Apply multiple testing correction
    adjusted = _correct_pvalues(raw_pvals, method=method)

    # Build result DataFrame
    df = leaderboard._df.copy() if hasattr(leaderboard, "_df") else pd.DataFrame(leaderboard)
    df["p_value"] = [round(p, 6) if not np.isnan(p) else np.nan for p in raw_pvals]
    df["p_adjusted"] = [round(p, 6) if not np.isnan(p) else np.nan for p in adjusted]
    df["significant"] = [
        bool(p <= alpha) if not np.isnan(p) else False for p in adjusted
    ]

    return df


def _correct_pvalues(
    pvals: list[float], *, method: str = "bh"
) -> list[float]:
    """Apply multiple testing correction to p-values.

    Handles NaN values (baseline model, failed predictions) by passing through.
    """
    arr = np.array(pvals, dtype=float)
    valid_mask = ~np.isnan(arr)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        return pvals

    valid_pvals = arr[valid_mask]
    adjusted = np.full_like(arr, np.nan)

    if method == "bonferroni":
        adj = np.minimum(valid_pvals * n_valid, 1.0)
        adjusted[valid_mask] = adj

    elif method == "holm":
        # Holm step-down: sort p-values, multiply by (m - rank + 1), enforce monotonicity
        order = np.argsort(valid_pvals)
        sorted_p = valid_pvals[order]
        adj = np.empty(n_valid)
        for i in range(n_valid):
            adj[i] = sorted_p[i] * (n_valid - i)
        # Enforce monotonicity (cumulative max)
        for i in range(1, n_valid):
            adj[i] = max(adj[i], adj[i - 1])
        adj = np.minimum(adj, 1.0)
        # Unsort
        result = np.empty(n_valid)
        result[order] = adj
        adjusted[valid_mask] = result

    elif method == "bh":
        # Benjamini-Hochberg: sort, multiply by m/rank, enforce monotonicity (reverse)
        order = np.argsort(valid_pvals)
        sorted_p = valid_pvals[order]
        adj = np.empty(n_valid)
        for i in range(n_valid):
            rank = i + 1
            adj[i] = sorted_p[i] * n_valid / rank
        # Enforce monotonicity from bottom up (cumulative min in reverse)
        for i in range(n_valid - 2, -1, -1):
            adj[i] = min(adj[i], adj[i + 1])
        adj = np.minimum(adj, 1.0)
        # Unsort
        result = np.empty(n_valid)
        result[order] = adj
        adjusted[valid_mask] = result

    return adjusted.tolist()
