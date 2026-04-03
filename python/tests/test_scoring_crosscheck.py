"""Cross-check ml's own scorers against sklearn equivalents.

Safety net for sklearn-independence branch: ensures our hand-rolled metrics
match sklearn's output to known tolerances. Uses pytest.importorskip to
gracefully skip when sklearn is not installed.
"""

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ml._scoring import (  # noqa: E402
    _acc,
    _brier,
    _cohen_kappa,
    _f1,
    _f1_macro,
    _f1_weighted,
    _log_loss,
    _mae,
    _mape,
    _mcc,
    _precision,
    _r2,
    _recall,
    _rmse,
    _roc_auc,
)


@pytest.fixture
def binary_data():
    """Binary classification data with integer labels (avoids pos_label issues)."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 200)
    y_pred = y_true.copy()
    # Flip ~20% for imperfect predictions
    flip = rng.choice(200, 40, replace=False)
    y_pred[flip] = 1 - y_pred[flip]
    y_prob = rng.rand(200)
    # Make probabilities correlated with truth
    y_prob = 0.3 * y_prob + 0.7 * y_true.astype(float)
    return y_true, y_pred, y_prob


@pytest.fixture
def multiclass_data_scorer():
    """Multiclass classification data."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 4, 200)
    y_pred = y_true.copy()
    flip = rng.choice(200, 40, replace=False)
    y_pred[flip] = rng.randint(0, 4, 40)
    return y_true, y_pred


@pytest.fixture
def regression_data():
    """Regression data."""
    rng = np.random.RandomState(42)
    y_true = rng.randn(200)
    y_pred = y_true + rng.randn(200) * 0.3
    return y_true, y_pred


# ── Exact-match metrics (tolerance: 1e-12) ──────────────────────────────────


def test_accuracy_matches_sklearn(binary_data):
    y_true, y_pred, _ = binary_data
    ours = _acc(y_true, y_pred)
    theirs = accuracy_score(y_true, y_pred)
    assert abs(ours - theirs) < 1e-12


def test_f1_binary_matches_sklearn(binary_data):
    y_true, y_pred, _ = binary_data
    ours = _f1(y_true, y_pred)
    theirs = f1_score(y_true, y_pred, pos_label=1)
    assert abs(ours - theirs) < 1e-12


def test_f1_weighted_matches_sklearn(binary_data):
    y_true, y_pred, _ = binary_data
    ours = _f1_weighted(y_true, y_pred)
    theirs = f1_score(y_true, y_pred, average="weighted")
    assert abs(ours - theirs) < 1e-12


def test_f1_macro_matches_sklearn(binary_data):
    y_true, y_pred, _ = binary_data
    ours = _f1_macro(y_true, y_pred)
    theirs = f1_score(y_true, y_pred, average="macro")
    assert abs(ours - theirs) < 1e-12


def test_f1_macro_multiclass(multiclass_data_scorer):
    y_true, y_pred = multiclass_data_scorer
    ours = _f1_macro(y_true, y_pred)
    theirs = f1_score(y_true, y_pred, average="macro")
    assert abs(ours - theirs) < 1e-12


def test_f1_weighted_multiclass(multiclass_data_scorer):
    y_true, y_pred = multiclass_data_scorer
    ours = _f1_weighted(y_true, y_pred)
    theirs = f1_score(y_true, y_pred, average="weighted")
    assert abs(ours - theirs) < 1e-12


def test_precision_matches_sklearn(binary_data):
    y_true, y_pred, _ = binary_data
    ours = _precision(y_true, y_pred)
    theirs = precision_score(y_true, y_pred, pos_label=1)
    assert abs(ours - theirs) < 1e-12


def test_recall_matches_sklearn(binary_data):
    y_true, y_pred, _ = binary_data
    ours = _recall(y_true, y_pred)
    theirs = recall_score(y_true, y_pred, pos_label=1)
    assert abs(ours - theirs) < 1e-12


def test_rmse_matches_sklearn(regression_data):
    y_true, y_pred = regression_data
    ours = _rmse(y_true, y_pred)
    theirs = np.sqrt(mean_squared_error(y_true, y_pred))
    assert abs(ours - theirs) < 1e-12


def test_mae_matches_sklearn(regression_data):
    y_true, y_pred = regression_data
    ours = _mae(y_true, y_pred)
    theirs = mean_absolute_error(y_true, y_pred)
    assert abs(ours - theirs) < 1e-12


def test_r2_matches_sklearn(regression_data):
    y_true, y_pred = regression_data
    ours = _r2(y_true, y_pred)
    theirs = r2_score(y_true, y_pred)
    assert abs(ours - theirs) < 1e-12


def test_mcc_binary_matches_sklearn(binary_data):
    y_true, y_pred, _ = binary_data
    ours = _mcc(y_true, y_pred)
    theirs = matthews_corrcoef(y_true, y_pred)
    assert abs(ours - theirs) < 1e-12


def test_mcc_multiclass_matches_sklearn(multiclass_data_scorer):
    """Multiclass MCC cross-check."""
    y_true, y_pred = multiclass_data_scorer
    ours = _mcc(y_true, y_pred)
    theirs = matthews_corrcoef(y_true, y_pred)
    assert abs(ours - theirs) < 1e-12


def test_brier_matches_sklearn(binary_data):
    """Brier score cross-check."""
    y_true, _, y_prob = binary_data
    ours = _brier(y_true, y_prob, pos_label=1)
    theirs = brier_score_loss(y_true, y_prob, pos_label=1)
    assert abs(ours - theirs) < 1e-12


def test_kappa_matches_sklearn(binary_data):
    y_true, y_pred, _ = binary_data
    ours = _cohen_kappa(y_true, y_pred)
    theirs = cohen_kappa_score(y_true, y_pred)
    assert abs(ours - theirs) < 1e-8


def test_kappa_quadratic_matches_sklearn(binary_data):
    y_true, y_pred, _ = binary_data
    ours = _cohen_kappa(y_true, y_pred, weights="quadratic")
    theirs = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    assert abs(ours - theirs) < 1e-8


# ── Tolerant metrics (1e-6 — implementation details differ) ──────────────────


def test_roc_auc_matches_sklearn(binary_data):
    """ROC AUC — tie handling may differ, use 1e-6 tolerance."""
    y_true, _, y_prob = binary_data
    ours = _roc_auc(y_true, y_prob)
    theirs = roc_auc_score(y_true, y_prob)
    assert abs(ours - theirs) < 1e-6


def test_log_loss_matches_sklearn(binary_data):
    """Log loss — clip/renorm details differ, use 1e-6 tolerance."""
    y_true, _, y_prob = binary_data
    # Stack for 2-class format
    proba = np.column_stack([1 - y_prob, y_prob])
    ours = _log_loss(y_true, proba)
    theirs = log_loss(y_true, y_prob)
    assert abs(ours - theirs) < 1e-6


def test_mape_matches_sklearn(regression_data):
    """MAPE — zero handling differs, use 1e-6 on non-zero data."""
    y_true, y_pred = regression_data
    # Ensure no zeros to avoid division differences
    y_true = np.abs(y_true) + 1.0
    y_pred = np.abs(y_pred) + 1.0
    ours = _mape(y_true, y_pred)
    theirs = mean_absolute_percentage_error(y_true, y_pred)
    assert abs(ours - theirs) < 1e-6


# ── Multiple seeds for stochastic sensitivity ────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 42, 99, 123])
def test_accuracy_multi_seed(seed):
    """Accuracy matches sklearn across multiple random seeds."""
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, 2, 100)
    y_pred = rng.randint(0, 2, 100)
    assert abs(_acc(y_true, y_pred) - accuracy_score(y_true, y_pred)) < 1e-12


@pytest.mark.parametrize("seed", [0, 1, 42, 99, 123])
def test_roc_auc_multi_seed(seed):
    """ROC AUC matches sklearn across multiple random seeds."""
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, 2, 100)
    y_prob = rng.rand(100)
    if len(np.unique(y_true)) < 2:
        return  # skip degenerate case
    assert abs(_roc_auc(y_true, y_prob) - roc_auc_score(y_true, y_prob)) < 1e-6
