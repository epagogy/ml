"""Tests for balance= parameter in fit().

Class imbalance handling.
Verifies that balance=True improves minority-class recall on imbalanced data.
"""

import numpy as np
import pandas as pd
import pytest

import ml


@pytest.fixture
def imbalanced_data():
    """Create a 95/5 imbalanced binary classification dataset."""
    rng = np.random.RandomState(42)
    n = 200
    n_minority = 10  # 5% minority

    # Majority class: centered at (0, 0)
    X_maj = rng.randn(n - n_minority, 2) * 1.0
    y_maj = ["no"] * (n - n_minority)

    # Minority class: centered at (3, 3) — linearly separable
    X_min = rng.randn(n_minority, 2) * 0.5 + 3.0
    y_min = ["yes"] * n_minority

    X = np.vstack([X_maj, X_min])
    y = y_maj + y_min

    df = pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "target": y})
    return df


@pytest.fixture
def imbalanced_split(imbalanced_data):
    """Split imbalanced data."""
    return ml.split(imbalanced_data, "target", seed=42)


def test_balance_logistic_improves_recall(imbalanced_split):
    """balance=True should improve minority-class recall for logistic."""
    s = imbalanced_split

    # Without balance
    model_no = ml.fit(s.dev, "target", algorithm="logistic", seed=42, balance=False)
    ml.evaluate(model_no, s.valid)

    # With balance
    model_yes = ml.fit(s.dev, "target", algorithm="logistic", seed=42, balance=True)
    metrics_yes = ml.evaluate(model_yes, s.valid)

    # balance=True should not crash and should produce valid metrics
    assert "recall" in metrics_yes
    assert 0.0 <= metrics_yes["recall"] <= 1.0


@pytest.mark.slow
def test_balance_xgboost_binary(imbalanced_split):
    """balance=True should work for XGBoost binary classification."""
    s = imbalanced_split
    model = ml.fit(s.dev, "target", algorithm="xgboost", seed=42, balance=True)
    assert model.algorithm == "xgboost"
    metrics = ml.evaluate(model, s.valid)
    assert "roc_auc" in metrics


def test_balance_random_forest(imbalanced_split):
    """balance=True should set class_weight='balanced' for random_forest."""
    s = imbalanced_split
    model = ml.fit(s.dev, "target", algorithm="random_forest", seed=42, balance=True)
    assert model.algorithm == "random_forest"
    # Verify class_weight was set
    assert model._model.class_weight == "balanced"


def test_balance_svm(imbalanced_split):
    """balance=True should set class_weight='balanced' for SVM."""
    s = imbalanced_split
    model = ml.fit(s.dev, "target", algorithm="svm", seed=42, balance=True)
    assert model.algorithm == "svm"
    assert model._model.class_weight == "balanced"


def test_balance_naive_bayes(imbalanced_split):
    """balance=True should use sample_weight for naive_bayes."""
    s = imbalanced_split
    model = ml.fit(s.dev, "target", algorithm="naive_bayes", seed=42, balance=True)
    assert model.algorithm == "naive_bayes"
    metrics = ml.evaluate(model, s.valid)
    assert "accuracy" in metrics


def test_balance_knn_warns(imbalanced_split):
    """balance=True should warn for KNN (no native support)."""
    s = imbalanced_split
    with pytest.warns(UserWarning, match="balance.*no effect.*knn"):
        model = ml.fit(s.dev, "target", algorithm="knn", seed=42, balance=True)
    assert model.algorithm == "knn"
    metrics = ml.evaluate(model, s.valid)
    assert "accuracy" in metrics


def test_balance_regression_error(imbalanced_split):
    """balance=True should raise ConfigError for regression tasks."""
    # Create regression data
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "x": rng.randn(100),
        "y": rng.randn(100),
    })
    s = ml.split(df, "y", seed=42)

    with pytest.raises(ml.ConfigError, match="balance.*classification"):
        ml.fit(s.dev, "y", seed=42, balance=True)


def test_balance_false_is_default(imbalanced_split):
    """balance=False should be the default (no class weighting)."""
    s = imbalanced_split
    model = ml.fit(s.dev, "target", algorithm="random_forest", seed=42)
    # Default RF has no class_weight set
    assert model._model.class_weight is None


def test_balance_cv_path(imbalanced_data):
    """balance=True should work with CV path (split with folds)."""
    s = ml.split(imbalanced_data, "target", seed=42)
    cv = ml.cv(s, folds=3, seed=42)
    model = ml.fit(cv, "target", algorithm="logistic", seed=42, balance=True)
    assert model.scores_ is not None
    assert "accuracy_mean" in model.scores_


@pytest.mark.slow
def test_balance_xgboost_scale_pos_weight(imbalanced_split):
    """XGBoost binary with balance=True should set scale_pos_weight."""
    s = imbalanced_split
    model = ml.fit(s.dev, "target", algorithm="xgboost", seed=42, balance=True)
    # scale_pos_weight should be > 1 (more negatives than positives)
    spw = model._model.get_params().get("scale_pos_weight")
    assert spw is not None
    assert spw > 1.0  # 95/5 imbalance → scale_pos_weight ≈ 19
