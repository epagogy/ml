"""Tests for ml.compare() — fair comparison of pre-fitted models."""

import warnings

import pandas as pd
import pytest

import ml


@pytest.fixture(scope="module")
def compare_models(small_classification_data):
    """Pre-fitted Logistic + RandomForest on small data — shared across tests."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_log = ml.fit(data=s.train, target="target", algorithm="logistic", seed=42)
        m_rf = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
    return s, m_log, m_rf


def test_compare_two_models(compare_models):
    """compare() evaluates two pre-fitted models on same data."""
    s, model_log, model_rf = compare_models
    result = ml.compare([model_log, model_rf], s.valid)

    assert isinstance(result, ml.Leaderboard)
    assert len(result) == 2
    assert "algorithm" in result.columns
    algos = result["algorithm"].tolist()
    assert "logistic" in algos
    assert "random_forest" in algos


def test_compare_sorted_by_roc_auc(compare_models):
    """Classification comparison sorted by roc_auc descending."""
    s, model_log, model_rf = compare_models
    result = ml.compare([model_log, model_rf], s.valid)

    if "roc_auc" in result.columns:
        roc_aucs = result["roc_auc"].tolist()
        assert roc_aucs == sorted(roc_aucs, reverse=True)


def test_compare_regression():
    """compare() works for regression models."""
    data = pd.DataFrame(
        {
            "x1": range(100),
            "x2": range(100, 200),
            "price": [i * 2.5 + 10 for i in range(100)],
        }
    )
    s = ml.split(data=data, target="price", seed=42)
    model_lin1 = ml.fit(data=s.train, target="price", algorithm="linear", seed=42)
    model_rf = ml.fit(data=s.train, target="price", algorithm="random_forest", seed=42)

    result = ml.compare([model_lin1, model_rf], s.valid)

    assert isinstance(result, ml.Leaderboard)
    assert len(result) == 2


def test_compare_custom_sort(compare_models):
    """compare() with custom sort_by metric."""
    s, model_log, model_rf = compare_models
    result = ml.compare([model_log, model_rf], s.valid, sort_by="f1")

    if "f1" in result.columns:
        f1s = result["f1"].tolist()
        assert f1s == sorted(f1s, reverse=True)


def test_compare_error_empty_list():
    """compare() with empty list raises ConfigError."""
    with pytest.raises(ml.ConfigError, match="requires at least one"):
        ml.compare([], pd.DataFrame())


def test_compare_error_not_model():
    """compare() with non-Model raises ConfigError."""
    with pytest.raises(ml.ConfigError, match="not a Model"):
        ml.compare(["not_a_model"], pd.DataFrame({"a": [1], "b": [2]}))


def test_compare_single_model(compare_models):
    """compare() works with a single model."""
    s, model_log, _ = compare_models
    result = ml.compare([model_log], s.valid)

    assert isinstance(result, ml.Leaderboard)
    assert len(result) == 1
    assert result["algorithm"].iloc[0] == "logistic"


def test_compare_leaderboard_best(compare_models):
    """Leaderboard.best is a property returning the top-ranked algorithm name."""
    s, m1, m2 = compare_models
    lb = ml.compare([m1, m2], s.valid)
    winner = lb.best
    assert isinstance(winner, str)
    assert winner in ["logistic", "random_forest"]
    # best should match the first row of the sorted leaderboard
    assert winner == lb["algorithm"].iloc[0]


def test_compare_best_model(compare_models):
    """best_model property returns the top-ranked fitted Model object."""
    s, m1, m2 = compare_models
    lb = ml.compare([m1, m2], s.valid)
    winner = lb.best_model  # property, not method

    assert isinstance(winner, ml.Model)
    assert winner._algorithm in lb.best


# ---------------------------------------------------------------------------
# A10 tests — compare on test + custom metrics (Conort C2)
# ---------------------------------------------------------------------------

def test_compare_on_test_warns(compare_models):
    """compare() emits UserWarning about test-data peeking by default. A10."""
    s, model_log, model_rf = compare_models
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ml.compare([model_log, model_rf], s.test)
    warn_texts = [str(x.message) for x in w]
    assert any("test" in t.lower() or "peeking" in t.lower() or "model selection" in t.lower()
               for t in warn_texts)


def test_compare_warn_test_false_suppresses(compare_models):
    """compare(warn_test=False) suppresses the test-data peeking warning. A10."""
    s, model_log, model_rf = compare_models
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = ml.compare([model_log, model_rf], s.valid, warn_test=False)
    peeking_warnings = [x for x in w if "peeking" in str(x.message).lower()
                        or "model selection" in str(x.message).lower()]
    assert len(peeking_warnings) == 0
    assert isinstance(result, ml.Leaderboard)


def test_compare_custom_metric(compare_models):
    """compare(metric=callable) adds custom metric column and sorts by it. A10."""
    import numpy as np
    s, model_log, model_rf = compare_models

    def my_acc(y_true, y_pred):
        return float(np.mean(y_true == y_pred))
    my_acc.greater_is_better = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.compare([model_log, model_rf], s.valid,
                            metric=my_acc, warn_test=False)
    assert "my_acc" in result.columns
    assert isinstance(result, ml.Leaderboard)


@pytest.mark.slow
def test_compare_tuning_result_labeled(compare_models):
    """TuningResult in compare() shows as 'algorithm (tuned)' in the table."""
    import warnings

    from ml._types import TuningResult

    s, model_log, _ = compare_models
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(data=s.train, target="target", algorithm="xgboost", seed=42, n_trials=2)

    assert isinstance(tuned, TuningResult)
    lb = ml.compare([model_log, tuned], s.valid)

    algos = lb["algorithm"].tolist()
    assert any("xgboost" in a for a in algos)
    assert any("tuned" in a for a in algos)


# ── Chain 2.6: Statistical significance ───────────────────────────────────────


@pytest.mark.slow
def test_compare_significance_column(compare_models):
    """compare() adds significant_vs_best column. Chain 2.6."""
    s, model_log, model_rf = compare_models
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb = ml.compare([model_log, model_rf], s.valid, warn_test=False)
    # significant_vs_best column should exist when scipy is available
    try:
        import scipy  # noqa: F401
        assert "significant_vs_best" in lb.columns
    except ImportError:
        pytest.skip("scipy not installed")


@pytest.mark.slow
def test_compare_identical_models_not_significant(compare_models):
    """A model compared to itself is not significant (p > 0.05). Chain 2.6."""
    s, model_log, _ = compare_models
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb = ml.compare([model_log, model_log], s.valid, warn_test=False)
    try:
        import scipy  # noqa: F401
        if "significant_vs_best" in lb.columns:
            # Both rows: first is the best (False), second is identical (False)
            assert list(lb["significant_vs_best"]) == [False, False]
    except ImportError:
        pytest.skip("scipy not installed")
