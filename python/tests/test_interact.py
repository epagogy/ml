"""Tests for ml.interact() — feature interaction detection."""

import warnings

import numpy as np
import pandas as pd
import pytest

import ml
from ml._types import ConfigError, DataError

# -- Fixtures ------------------------------------------------------------------


def _clf_data(n=300, seed=42):
    rng = np.random.RandomState(seed)
    # Create two interacting features: x1 * x2 determines target
    x1 = rng.rand(n)
    x2 = rng.rand(n)
    x3 = rng.rand(n)  # irrelevant
    target = ((x1 * x2) > 0.25).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": target})


def _reg_data(n=300, seed=42):
    rng = np.random.RandomState(seed)
    x1 = rng.rand(n)
    x2 = rng.rand(n)
    target = x1 * x2 + rng.rand(n) * 0.1
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


# -- Module fixture: fit once, test many things --------------------------------


@pytest.fixture(scope="module")
def clf_interact(request):
    """Pre-fitted classification model + interact result, shared across tests."""
    data = _clf_data(n=300, seed=42)
    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", seed=42)
        result = ml.interact(model, data=s.valid, seed=42)
    return data, s, model, result


# -- Core behavior -------------------------------------------------------------


def test_interact_returns_result(clf_interact):
    """interact() returns an InteractResult."""
    _, _, _, result = clf_interact
    assert isinstance(result, ml.InteractResult)


def test_interact_pairs_is_dataframe(clf_interact):
    """result.pairs is a DataFrame with required columns."""
    _, _, _, result = clf_interact
    assert isinstance(result.pairs, pd.DataFrame)
    for col in ["feature_a", "feature_b", "score"]:
        assert col in result.pairs.columns, f"Missing column: {col}"


def test_interact_pairs_sorted_descending(clf_interact):
    """Pairs are sorted by score descending."""
    _, _, _, result = clf_interact
    scores = result.pairs["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_interact_scores_nonnegative(clf_interact):
    """Interaction scores are >= 0."""
    _, _, _, result = clf_interact
    assert (result.pairs["score"] >= 0).all()


def test_interact_top_method(clf_interact):
    """result.top(n) returns the top N rows."""
    _, _, _, result = clf_interact
    top3 = result.top(3)
    assert len(top3) <= 3
    assert isinstance(top3, pd.DataFrame)


def test_interact_n_pairs_consistent(clf_interact):
    """n_pairs matches rows in pairs DataFrame."""
    _, _, _, result = clf_interact
    assert result.n_pairs == len(result.pairs)


def test_interact_repr(clf_interact):
    """InteractResult has a readable repr."""
    _, _, _, result = clf_interact
    r = repr(result)
    assert "InteractResult" in r
    assert "n_pairs" in r


def test_interact_summary_is_str(clf_interact):
    """result.summary is a non-empty string."""
    _, _, _, result = clf_interact
    assert isinstance(result.summary, str)
    assert len(result.summary) > 0


def test_interact_regression_model():
    """interact() works on regression models."""
    data = _reg_data()
    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", seed=42)
        result = ml.interact(model, data=s.valid, seed=42)
    assert isinstance(result, ml.InteractResult)
    assert len(result.pairs) > 0


def test_interact_n_top_limits_pairs():
    """n_top=2 limits to 1 pair (C(2,2)=1)."""
    data = _clf_data()
    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", seed=42)
        result = ml.interact(model, data=s.valid, n_top=2, seed=42)
    # With 2 features in n_top, exactly 1 pair
    assert result.n_features == 2
    assert result.n_pairs == 1


@pytest.mark.slow
def test_interact_with_tuning_result():
    """interact() accepts TuningResult (unwraps it)."""
    data = _clf_data()
    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train,
            target="target",
            algorithm="random_forest",
            n_trials=2,
            cv_folds=2,
            seed=42,
        )
        result = ml.interact(tuned, data=s.valid, seed=42)
    assert isinstance(result, ml.InteractResult)


def test_interact_target_column_excluded(clf_interact):
    """interact() works even if data contains the target column."""
    _, _, _, result = clf_interact
    assert "target" not in result.pairs["feature_a"].values
    assert "target" not in result.pairs["feature_b"].values


# -- Error handling ------------------------------------------------------------


def test_interact_non_dataframe_raises(clf_interact):
    """interact() raises DataError for non-DataFrame data."""
    _, _, model, _ = clf_interact
    with pytest.raises(DataError):
        ml.interact(model, data=[[1, 2, 3]], seed=42)


def test_interact_too_few_rows_raises(clf_interact):
    """interact() raises DataError for fewer than 20 rows."""
    _, s, model, _ = clf_interact
    tiny = s.valid.iloc[:5]
    with pytest.raises(DataError, match="20 rows"):
        ml.interact(model, data=tiny, seed=42)


def test_interact_n_top_too_small_raises(clf_interact):
    """interact() raises ConfigError for n_top < 2."""
    _, s, model, _ = clf_interact
    with pytest.raises(ConfigError):
        ml.interact(model, data=s.valid, n_top=1, seed=42)
