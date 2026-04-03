"""Tests for ml._scoring — unified scoring interface (A11)."""

import numpy as np
import pytest

from ml._scoring import METRIC_REGISTRY, Scorer, make_scorer
from ml._types import ConfigError


def test_custom_scorer_string():
    """make_scorer() returns the registered Scorer for known string names."""
    scorer = make_scorer("roc_auc")
    assert isinstance(scorer, Scorer)
    assert scorer.name == "roc_auc"
    assert scorer.greater_is_better is True
    assert scorer.needs_proba is True

    scorer_rmse = make_scorer("rmse")
    assert scorer_rmse.name == "rmse"
    assert scorer_rmse.greater_is_better is False
    assert scorer_rmse.needs_proba is False

    # Unknown string raises ConfigError
    with pytest.raises(ConfigError, match="not recognised"):
        make_scorer("invented_metric")


def test_custom_scorer_callable():
    """make_scorer() wraps a callable in a Scorer with correct metadata."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 50)
    y_prob = rng.rand(50)

    def my_gini(y_t, y_p):
        from sklearn.metrics import roc_auc_score
        return 2 * roc_auc_score(y_t, y_p) - 1

    my_gini.greater_is_better = True
    my_gini.needs_proba = True

    scorer = make_scorer(my_gini)
    assert isinstance(scorer, Scorer)
    assert scorer.name == "my_gini"
    assert scorer.greater_is_better is True
    assert scorer.needs_proba is True

    result = scorer(y_true, y_prob)
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0

    # Callable without attributes defaults to greater_is_better=True, needs_proba=False
    def simple(y_t, y_p):
        return float(np.mean(y_t == (y_p > 0.5)))

    scorer2 = make_scorer(simple)
    assert scorer2.greater_is_better is True
    assert scorer2.needs_proba is False
    assert scorer2.name == "simple"

    # Non-callable, non-string raises ConfigError
    with pytest.raises(ConfigError):
        make_scorer(42)


def test_scorer_greater_is_better():
    """METRIC_REGISTRY has correct greater_is_better flags for all built-ins."""
    higher_better = {"accuracy", "f1", "f1_weighted", "f1_macro", "precision",
                     "recall", "roc_auc", "roc_auc_ovr", "r2",
                     "qwk", "mcc", "gini"}
    lower_better = {"rmse", "mae", "log_loss", "mape", "smape", "rmsle", "log_cosh"}

    for name in higher_better:
        assert METRIC_REGISTRY[name].greater_is_better is True, (
            f"{name} should have greater_is_better=True"
        )

    for name in lower_better:
        assert METRIC_REGISTRY[name].greater_is_better is False, (
            f"{name} should have greater_is_better=False"
        )

    # All built-in scorers are callable and return float
    for _name, scorer in METRIC_REGISTRY.items():
        assert isinstance(scorer, Scorer)
        assert isinstance(scorer.name, str)
        assert isinstance(scorer.greater_is_better, bool)
        assert callable(scorer)


# ── Chain 2.1: Competition metrics ────────────────────────────────────────────


def test_scorer_qwk_ordinal():
    """QWK on 5-class ordinal data returns value in [-1, 1]."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 5, 100)
    y_pred = y_true + rng.choice([-1, 0, 1], 100).clip(0, 4)
    scorer = make_scorer("qwk")
    score = scorer(y_true, y_pred)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0
    # Perfect prediction should give 1.0
    assert make_scorer("qwk")(y_true, y_true) == 1.0


def test_scorer_gini_equals_2auc_minus_1():
    """Gini = 2*AUC - 1 (mathematical identity)."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 100)
    y_prob = rng.rand(100)
    scorer = make_scorer("gini")
    gini = scorer(y_true, y_prob)
    auc_scorer = make_scorer("roc_auc")
    expected = 2 * auc_scorer(y_true, y_prob) - 1
    assert abs(gini - expected) < 1e-9
    assert scorer.needs_proba is True


def test_scorer_mcc_imbalanced():
    """MCC handles 95/5 imbalanced data, range [-1, 1]."""
    n = 200
    y_true = np.concatenate([np.zeros(190, dtype=int), np.ones(10, dtype=int)])
    y_pred = np.zeros(n, dtype=int)  # always predict majority
    scorer = make_scorer("mcc")
    score = scorer(y_true, y_pred)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0
    # Always-majority prediction should be close to 0
    assert abs(score) < 0.2


def test_scorer_mape_zero_division():
    """MAPE returns inf when y_true has all zeros."""
    scorer = make_scorer("mape")
    y_true = np.zeros(10)
    y_pred = np.ones(10)
    score = scorer(y_true, y_pred)
    assert score == float("inf")


def test_scorer_smape_symmetric():
    """SMAPE(a, b) == SMAPE(b, a) — symmetric property."""
    scorer = make_scorer("smape")
    rng = np.random.RandomState(42)
    y_a = rng.rand(50) + 0.1
    y_b = rng.rand(50) + 0.1
    assert abs(scorer(y_a, y_b) - scorer(y_b, y_a)) < 1e-9


def test_scorer_rmsle_negative_clipping():
    """RMSLE clips negative predictions to 0 instead of crashing."""
    scorer = make_scorer("rmsle")
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([-1.0, 2.0, -0.5, 4.0])  # two negative predictions
    score = scorer(y_true, y_pred)
    assert isinstance(score, float)
    assert np.isfinite(score)


def test_scorer_log_cosh_smooth():
    """log_cosh is approximately 0 for perfect predictions."""
    scorer = make_scorer("log_cosh")
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Near-perfect predictions should give near-zero loss
    score = scorer(y, y + 1e-8)
    assert score < 1e-6


# ── Safety net: _confusion_matrix edge cases ─────────────────────────────────


def test_cm_label_in_pred_not_true():
    """CM handles label present in y_pred but absent from y_true."""
    from ml._scoring import _confusion_matrix
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 2, 1, 2])  # 2 never in y_true
    cm = _confusion_matrix(y_true, y_pred)
    # labels = [0, 1, 2] (union), shape 3x3
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1  # true=0, pred=0
    assert cm[0, 2] == 1  # true=0, pred=2
    assert cm[1, 1] == 1  # true=1, pred=1
    assert cm[1, 2] == 1  # true=1, pred=2
    assert cm[2, :].sum() == 0  # label 2 never in y_true


def test_cm_single_class():
    """CM with only one class produces 1x1 matrix."""
    from ml._scoring import _confusion_matrix
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])
    cm = _confusion_matrix(y_true, y_pred)
    assert cm.shape == (1, 1)
    assert cm[0, 0] == 3


def test_cm_empty_input():
    """CM with empty arrays returns empty matrix."""
    from ml._scoring import _confusion_matrix
    cm = _confusion_matrix(np.array([]), np.array([]))
    assert cm.shape == (0, 0)


def test_cm_string_labels():
    """CM works with string labels."""
    from ml._scoring import _confusion_matrix
    y_true = np.array(["cat", "dog", "cat", "bird"])
    y_pred = np.array(["cat", "cat", "dog", "bird"])
    cm = _confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert cm.sum() == 4


def test_cm_explicit_labels():
    """CM with explicit labels uses those labels (even if not all present)."""
    from ml._scoring import _confusion_matrix
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    cm = _confusion_matrix(y_true, y_pred, labels=np.array([0, 1, 2]))
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1
    assert cm[1, 1] == 1
    assert cm[2, :].sum() == 0


# ── Safety net: _infer_pos_label edge cases ──────────────────────────────────


def test_infer_pos_label_equal_count_strings():
    """Equal-count string labels: returns alphabetically first (min by count then name)."""
    from ml._scoring import _infer_pos_label
    # 5 of each — equal counts, so tie-break by key
    y = np.array(["no"] * 5 + ["yes"] * 5)
    result = _infer_pos_label(y)
    # min(counts, key=lambda k: (counts[k], k)) with equal counts
    # (5, "no") < (5, "yes"), so "no" is returned
    assert result == "no"


def test_infer_pos_label_no_one_in_ints():
    """Integer labels [0, 2] with no 1: returns max (2)."""
    from ml._scoring import _infer_pos_label
    y = np.array([0, 0, 2, 2, 0])
    result = _infer_pos_label(y)
    assert result == 2


def test_infer_pos_label_standard_binary():
    """Standard [0, 1] returns 1."""
    from ml._scoring import _infer_pos_label
    y = np.array([0, 0, 0, 1, 1])
    result = _infer_pos_label(y)
    assert result == 1


def test_infer_pos_label_multiclass_fallback():
    """Multiclass (>2 classes) returns default 1."""
    from ml._scoring import _infer_pos_label
    y = np.array([0, 1, 2, 0, 1, 2])
    result = _infer_pos_label(y)
    assert result == 1


def test_infer_pos_label_minority_string():
    """String labels with different counts: returns minority class."""
    from ml._scoring import _infer_pos_label
    y = np.array(["good"] * 8 + ["bad"] * 2)
    result = _infer_pos_label(y)
    assert result == "bad"


# ── Safety net: _precision_recall_f1 edge cases ────────────────────────────


def test_prf_all_wrong_predictions():
    """P/R/F1 when all predictions are wrong."""
    from ml._scoring import _precision_recall_f1
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    p, r, f1 = _precision_recall_f1(y_true, y_pred, average="macro")
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0


def test_prf_perfect_predictions():
    """P/R/F1 perfect predictions give 1.0."""
    from ml._scoring import _precision_recall_f1
    y = np.array([0, 1, 2, 0, 1, 2])
    p, r, f1 = _precision_recall_f1(y, y, average="macro")
    assert abs(p - 1.0) < 1e-12
    assert abs(r - 1.0) < 1e-12
    assert abs(f1 - 1.0) < 1e-12
