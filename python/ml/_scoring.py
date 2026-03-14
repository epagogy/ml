"""Unified scoring interface for tune(), screen(), compare(), and optimize().

A11 — Conort C2. Centralises metric handling so custom competition metrics
(Gini, QWK, business KPIs) work consistently throughout the pipeline.

Usage:
    >>> scorer = make_scorer("roc_auc")
    >>> scorer(y_true, y_pred_proba)
    0.87

    >>> def my_gini(y_true, y_prob):
    ...     auc = ml.evaluate(model, data)["roc_auc"]
    ...     return 2 * auc - 1
    >>> my_gini.greater_is_better = True
    >>> scorer = make_scorer(my_gini)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ._types import ConfigError


@dataclass(frozen=True)
class Scorer:
    """A scoring function with metadata used throughout the ml pipeline.

    Attributes:
        name: Human-readable metric name (e.g. "roc_auc", "rmse").
        fn: Callable with signature (y_true, y_pred) -> float.
        greater_is_better: True when higher score = better (accuracy, roc_auc).
            False for error metrics (rmse, mae, log_loss).
        needs_proba: True when fn expects probability scores rather than
            class labels. Callers must pass predict_proba() output.
            Default: False (fn expects label/value output from predict()).
    """

    name: str
    fn: Callable
    greater_is_better: bool
    needs_proba: bool = field(default=False)

    def __call__(self, y_true, y_pred) -> float:
        """Score predictions. Returns float (higher = better when greater_is_better)."""
        return float(self.fn(y_true, y_pred))


# ---------------------------------------------------------------------------
# Pure-numpy metric implementations (no sklearn dependency)
# ---------------------------------------------------------------------------


def _confusion_matrix(y_true, y_pred, labels=None):
    """Confusion matrix from arrays. Returns (n_labels, n_labels) numpy array."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.int64)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((n, n), dtype=np.int64)
    ti = np.array([label_to_idx.get(t, -1) for t in y_true])
    pi = np.array([label_to_idx.get(p, -1) for p in y_pred])
    valid = (ti >= 0) & (pi >= 0)
    np.add.at(cm, (ti[valid], pi[valid]), 1)
    return cm


def _acc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def _infer_pos_label(y_true):
    """Return pos_label suitable for binary metrics.

    When labels are strings (e.g. "yes"/"no", "fraud"/"legit"),
    binary metrics default pos_label=1 and crash. We infer the minority class
    as the positive label (or alphabetically last if equal counts).
    """
    unique = np.unique(y_true)
    if len(unique) != 2:
        return 1  # multiclass / regression: fall back to default
    # If numeric, use 1 if present, else max
    if np.issubdtype(np.array(unique).dtype, np.integer):
        return 1 if 1 in unique else int(unique.max())
    # String labels: use minority class as positive; ties -> alphabetically last
    counts = {u: int((np.asarray(y_true) == u).sum()) for u in unique}
    return min(counts, key=lambda k: (counts[k], k))


def _precision_recall_f1(y_true, y_pred, average="binary", pos_label=None):
    """Compute precision, recall, f1 from confusion matrix.

    average: "binary", "macro", "weighted"
    Returns (precision, recall, f1).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = _confusion_matrix(y_true, y_pred, labels=labels)

    if average == "binary":
        if pos_label is None:
            pos_label = _infer_pos_label(y_true)
        idx = None
        for i, lab in enumerate(labels):
            if lab == pos_label:
                idx = i
                break
        if idx is None:
            return 0.0, 0.0, 0.0
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return float(prec), float(rec), float(f1)

    # Per-class metrics
    n = len(labels)
    precs = np.zeros(n)
    recs = np.zeros(n)
    f1s = np.zeros(n)
    supports = np.zeros(n)
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        precs[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recs[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s[i] = 2 * precs[i] * recs[i] / (precs[i] + recs[i]) if (precs[i] + recs[i]) > 0 else 0.0
        supports[i] = support

    if average == "macro":
        return float(precs.mean()), float(recs.mean()), float(f1s.mean())
    elif average == "weighted":
        total = supports.sum()
        if total == 0:
            return 0.0, 0.0, 0.0
        w = supports / total
        return float((precs * w).sum()), float((recs * w).sum()), float((f1s * w).sum())
    return float(precs.mean()), float(recs.mean()), float(f1s.mean())


def _f1(y_true, y_pred):
    _, _, f1 = _precision_recall_f1(y_true, y_pred, average="binary",
                                    pos_label=_infer_pos_label(y_true))
    return f1


def _f1_weighted(y_true, y_pred):
    _, _, f1 = _precision_recall_f1(y_true, y_pred, average="weighted")
    return f1


def _f1_macro(y_true, y_pred):
    _, _, f1 = _precision_recall_f1(y_true, y_pred, average="macro")
    return f1


def _precision(y_true, y_pred):
    p, _, _ = _precision_recall_f1(y_true, y_pred, average="binary",
                                   pos_label=_infer_pos_label(y_true))
    return p


def _recall(y_true, y_pred):
    _, r, _ = _precision_recall_f1(y_true, y_pred, average="binary",
                                   pos_label=_infer_pos_label(y_true))
    return r


def _roc_curve_impl(y_true, y_score):
    """Compute ROC curve (FPR, TPR, thresholds) via trapezoidal method."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)

    # Sort by score descending
    desc = np.argsort(y_score)[::-1]
    y_score = y_score[desc]
    y_true_sorted = y_true[desc]

    # Distinct thresholds
    distinct = np.concatenate([[True], np.diff(y_score) != 0])
    tps = np.cumsum(y_true_sorted)[distinct]
    fps = np.cumsum(1 - y_true_sorted)[distinct]
    thresholds = y_score[distinct]

    # Prepend (0, 0) point
    tps = np.concatenate([[0], tps])
    fps = np.concatenate([[0], fps])
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds])

    # Normalize
    tpr = tps / tps[-1] if tps[-1] > 0 else tps
    fpr = fps / fps[-1] if fps[-1] > 0 else fps

    return fpr, tpr, thresholds


def _auc_trapz(x, y):
    """Trapezoidal AUC from sorted x, y arrays."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # Sort by x
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    # Manual trapezoid: works on all numpy versions (np.trapz removed in 2.x)
    dx = np.diff(x)
    return float(np.sum((y[:-1] + y[1:]) / 2.0 * dx))


def _roc_auc(y_true, y_score):
    """Binary ROC AUC via trapezoidal rule on sorted scores."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    # Handle 2D prob array (take column 1 for binary)
    if y_score.ndim == 2:
        y_score = y_score[:, 1]
    # Encode string labels to 0/1
    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError(f"roc_auc requires exactly 2 classes, got {len(classes)}")
    # Map to 0/1: second class (sorted) = positive
    pos = classes[1]
    y_bin = (y_true == pos).astype(np.int64)
    fpr, tpr, _ = _roc_curve_impl(y_bin, y_score)
    return _auc_trapz(fpr, tpr)


def _roc_auc_ovr(y_true, y_prob):
    """One-vs-rest ROC AUC for multiclass. Weighted average by class support."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    classes = np.unique(y_true)
    if y_prob.ndim == 1:
        return _roc_auc(y_true, y_prob)
    weighted_auc = 0.0
    total_weight = 0.0
    for k, cls in enumerate(sorted(classes)):
        y_bin = (y_true == cls).astype(np.int64)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue  # skip classes absent from fold
        col = k if k < y_prob.shape[1] else y_prob.shape[1] - 1
        fpr, tpr, _ = _roc_curve_impl(y_bin, y_prob[:, col])
        auc_k = _auc_trapz(fpr, tpr)
        weight = float(y_bin.sum())
        weighted_auc += auc_k * weight
        total_weight += weight
    if total_weight == 0:
        return 0.5
    return float(weighted_auc / total_weight)


def _log_loss(y_true, y_prob):
    """Log loss (cross-entropy). Handles binary (1D) and multiclass (2D)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    eps = 1e-15

    if y_prob.ndim == 1:
        # Binary: y_prob is P(positive class)
        classes = np.unique(y_true)
        pos = classes[-1] if len(classes) == 2 else classes[0]
        y_bin = (y_true == pos).astype(np.float64)
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return float(-np.mean(y_bin * np.log(y_prob) + (1 - y_bin) * np.log(1 - y_prob)))

    # Multiclass: y_prob is (n_samples, n_classes)
    y_prob = np.clip(y_prob, eps, 1 - eps)
    # Renormalize after clipping
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    classes = sorted(np.unique(y_true))
    n = len(y_true)
    loss = 0.0
    for k, cls in enumerate(classes):
        mask = (y_true == cls)
        col = k if k < y_prob.shape[1] else y_prob.shape[1] - 1
        loss -= np.sum(np.log(y_prob[mask, col]))
    return float(loss / n)


def _brier(y_true, y_prob, pos_label=None):
    """Brier score: mean((y_prob - y_true_binary)**2). Lower is better."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    # Encode to 0/1
    if pos_label is not None:
        y_bin = (y_true == pos_label).astype(np.float64)
    else:
        classes = sorted(np.unique(y_true))
        pos = classes[-1] if len(classes) >= 2 else classes[0]
        y_bin = (y_true == pos).astype(np.float64)
    return float(np.mean((y_prob - y_bin) ** 2))


def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Competition metrics
# ---------------------------------------------------------------------------


def _cohen_kappa(y_true, y_pred, weights=None):
    """Cohen's kappa coefficient. weights: None, 'linear', or 'quadratic'."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_labels = len(labels)
    cm = _confusion_matrix(y_true, y_pred, labels=labels)
    n = cm.sum()
    if n == 0:
        return 0.0

    if weights is None:
        # Unweighted kappa
        p_o = np.diag(cm).sum() / n
        p_e = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (n * n)
        if p_e == 1.0:
            return 1.0
        return float((p_o - p_e) / (1 - p_e))

    # Weighted kappa
    w = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            if weights == "linear":
                w[i, j] = abs(i - j)
            else:  # quadratic
                w[i, j] = (i - j) ** 2
    if w.max() > 0:
        w = w / w.max()

    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    num = (w * cm).sum()
    den = (w * expected).sum()
    if den == 0:
        return 1.0
    return float(1 - num / den)


def _qwk(y_true, y_pred):
    """Quadratic weighted kappa -- ordinal agreement metric used in competitions."""
    return _cohen_kappa(y_true, np.asarray(y_pred).round().astype(int), weights="quadratic")


def _gini(y_true, y_prob):
    """Gini coefficient = 2*AUC - 1. Standard in insurance competitions."""
    return 2 * _roc_auc(y_true, y_prob) - 1


def _mcc(y_true, y_pred):
    """Matthews correlation coefficient -- best single metric for imbalanced binary."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = _confusion_matrix(y_true, y_pred, labels=labels)
    # MCC from confusion matrix: general formula for any number of classes
    n = cm.sum()
    if n == 0:
        return 0.0
    # Binary shortcut (numerically stable)
    if len(labels) == 2:
        tp, fp = cm[1, 1], cm[0, 1]
        fn, tn = cm[1, 0], cm[0, 0]
        num = tp * tn - fp * fn
        den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        if den == 0:
            return 0.0
        return float(num / den)
    # Multiclass MCC (general formula)
    t_k = cm.sum(axis=1)
    p_k = cm.sum(axis=0)
    c = np.trace(cm)
    s = n
    s2 = s * s
    num = c * s - float(np.dot(t_k, p_k))
    den = np.sqrt(float(s2 - np.dot(p_k, p_k)) * float(s2 - np.dot(t_k, t_k)))
    if den == 0:
        return 0.0
    return float(num / den)


def _mape(y_true, y_pred):
    """Mean absolute percentage error. Returns inf when y_true has zeros."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true != 0
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _smape(y_true, y_pred):
    """Symmetric mean absolute percentage error (SMAPE)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def _rmsle(y_true, y_pred):
    """Root mean squared log error. Clips negative predictions to 0."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred_clip = np.maximum(y_pred, 0)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred_clip)) ** 2)))


def _log_cosh(y_true, y_pred):
    """Log-cosh loss -- smooth approximation to MAE, robust to outliers."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.log(np.cosh(y_pred - y_true))))


#: Registry of built-in metrics by name.
METRIC_REGISTRY: dict[str, Scorer] = {
    # Classification — label-based
    "accuracy":     Scorer("accuracy",     _acc,          greater_is_better=True),
    "f1":           Scorer("f1",           _f1,           greater_is_better=True),
    "f1_weighted":  Scorer("f1_weighted",  _f1_weighted,  greater_is_better=True),
    "f1_macro":     Scorer("f1_macro",     _f1_macro,     greater_is_better=True),
    "precision":    Scorer("precision",    _precision,    greater_is_better=True),
    "recall":       Scorer("recall",       _recall,       greater_is_better=True),
    "qwk":          Scorer("qwk",          _qwk,          greater_is_better=True),
    "mcc":          Scorer("mcc",          _mcc,          greater_is_better=True),
    # Classification — probability-based
    "roc_auc":      Scorer("roc_auc",      _roc_auc,      greater_is_better=True,  needs_proba=True),
    "roc_auc_ovr":  Scorer("roc_auc_ovr",  _roc_auc_ovr,  greater_is_better=True,  needs_proba=True),
    "log_loss":     Scorer("log_loss",     _log_loss,     greater_is_better=False, needs_proba=True),
    "gini":         Scorer("gini",         _gini,         greater_is_better=True,  needs_proba=True),
    # Regression
    "rmse":         Scorer("rmse",         _rmse,         greater_is_better=False),
    "mae":          Scorer("mae",          _mae,          greater_is_better=False),
    "r2":           Scorer("r2",           _r2,           greater_is_better=True),
    "mape":         Scorer("mape",         _mape,         greater_is_better=False),
    "smape":        Scorer("smape",        _smape,        greater_is_better=False),
    "rmsle":        Scorer("rmsle",        _rmsle,        greater_is_better=False),
    "log_cosh":     Scorer("log_cosh",     _log_cosh,     greater_is_better=False),
}


def make_scorer(metric) -> Scorer:
    """Create a Scorer from a string name or callable.

    Args:
        metric: One of:
            - A string naming a built-in metric. Available:
              classification: "accuracy", "f1", "f1_weighted", "f1_macro",
                              "precision", "recall", "roc_auc", "roc_auc_ovr",
                              "log_loss"
              regression:     "rmse", "mae", "r2"
            - A callable with signature ``(y_true, y_pred) -> float``.
              Optional attributes on the callable:
                ``greater_is_better`` (bool): defaults to True if absent.
                ``name`` (str): defaults to ``callable.__name__`` if absent.
                ``needs_proba`` (bool): defaults to False if absent.

    Returns:
        Scorer instance ready for use in tune(), screen(), compare().

    Raises:
        ConfigError: If metric string is not in METRIC_REGISTRY.
        ConfigError: If metric is neither a string nor callable.

    Examples:
        >>> scorer = make_scorer("roc_auc")
        >>> scorer(y_true, y_prob)
        0.87

        >>> import numpy as np
        >>> def my_gini(y_true, y_prob):
        ...     # Sort by predicted probability, measure cumulative gain
        ...     order = np.argsort(y_prob)
        ...     y_sorted = np.array(y_true)[order]
        ...     return float(2 * np.mean(y_sorted) - 1)
        >>> my_gini.greater_is_better = True
        >>> my_gini.needs_proba = True
        >>> scorer = make_scorer(my_gini)
        >>> scorer.name
        'my_gini'
    """
    if isinstance(metric, str):
        if metric not in METRIC_REGISTRY:
            import difflib
            available = sorted(METRIC_REGISTRY.keys())
            matches = difflib.get_close_matches(metric, available, n=1, cutoff=0.6)
            hint = f" Did you mean '{matches[0]}'?" if matches else ""
            raise ConfigError(
                f"metric='{metric}' not recognised.{hint} "
                f"Available: {available}. "
                "Or pass a callable: scorer(y_true, y_pred) -> float"
            )
        return METRIC_REGISTRY[metric]

    if callable(metric):
        name = getattr(metric, "name", None) or getattr(metric, "__name__", "custom_scorer")
        greater_is_better = getattr(metric, "greater_is_better", True)
        needs_proba = getattr(metric, "needs_proba", False)
        return Scorer(
            name=str(name),
            fn=metric,
            greater_is_better=bool(greater_is_better),
            needs_proba=bool(needs_proba),
        )

    raise ConfigError(
        f"metric must be a string or callable, got {type(metric).__name__}. "
        "Example: metric='roc_auc' or metric=my_scorer_function"
    )
