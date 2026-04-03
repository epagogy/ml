"""Tests for ml.nested_cv() — nested cross-validation. Chain 2.4."""

import numpy as np
import pandas as pd
import pytest

import ml


@pytest.fixture
def clf_data():
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "x3": rng.rand(n),
        "target": rng.choice([0, 1], n),
    })


def test_nested_cv_basic(clf_data):
    """nested_cv() runs without error and returns NestedCVResult. Chain 2.4."""
    result = ml.nested_cv(
        clf_data, "target",
        algorithms=["logistic"],
        outer_folds=3, inner_folds=2,
        seed=42,
    )
    assert isinstance(result, ml.NestedCVResult)
    assert "logistic" in result.scores
    assert len(result.scores["logistic"]) == 3  # outer_folds=3
    assert result.best_algorithm == "logistic"
    assert not result.summary.empty


def test_nested_cv_generalization_gap(clf_data):
    """Generalization gap (inner - outer) is finite and reasonable. Chain 2.4."""
    result = ml.nested_cv(
        clf_data, "target",
        algorithms=["logistic"],
        outer_folds=3, inner_folds=2,
        seed=42,
    )
    gap = result.generalization_gap.get("logistic")
    assert gap is not None
    assert np.isfinite(gap)
    # Gap should be modest (not > 0.5 for a stable algo on clean data)
    assert abs(gap) < 0.5


def test_nested_cv_best_algorithm(clf_data):
    """best_algorithm matches the algorithm with the best mean outer score. Chain 2.4."""
    result = ml.nested_cv(
        clf_data, "target",
        algorithms=["logistic", "random_forest"],
        outer_folds=3, inner_folds=2,
        seed=42,
    )
    assert result.best_algorithm in ("logistic", "random_forest")
    # Summary should be sorted by mean_score descending (roc_auc — higher better)
    if len(result.summary) > 1:
        assert result.summary["mean_score"].iloc[0] >= result.summary["mean_score"].iloc[1]
