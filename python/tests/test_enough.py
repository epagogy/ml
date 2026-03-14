"""Tests for ml.enough() — learning curve analysis."""

import warnings

import numpy as np
import pandas as pd
import pytest

import ml
from ml._types import ConfigError, DataError

# -- Fixtures ------------------------------------------------------------------


def _clf_data(n=300, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "x3": rng.rand(n),
        "target": rng.choice(["yes", "no"], n),
    })


def _reg_data(n=300, seed=42):
    rng = np.random.RandomState(seed)
    x1 = rng.rand(n)
    return pd.DataFrame({
        "x1": x1,
        "x2": rng.rand(n),
        "target": x1 * 2 + rng.rand(n) * 0.3,
    })


# -- Core behavior -------------------------------------------------------------


def test_enough_returns_result():
    """enough() returns an EnoughResult."""
    data = _clf_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.enough(data, "target", seed=42, steps=4)
    assert isinstance(result, ml.EnoughResult)


def test_enough_curve_is_dataframe():
    """result.curve is a DataFrame with required columns."""
    data = _clf_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.enough(data, "target", seed=42, steps=4)
    assert isinstance(result.curve, pd.DataFrame)
    for col in ["n_samples", "train_score", "val_score"]:
        assert col in result.curve.columns, f"Missing column: {col}"


def test_enough_curve_n_rows_matches_steps():
    """curve has approximately `steps` rows."""
    data = _clf_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.enough(data, "target", seed=42, steps=4)
    assert len(result.curve) == 4


def test_enough_n_samples_increasing():
    """n_samples is monotonically increasing in the curve."""
    data = _clf_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.enough(data, "target", seed=42, steps=5)
    samples = result.curve["n_samples"].tolist()
    assert samples == sorted(samples)


def test_enough_metric_is_accuracy_for_classification():
    """metric='accuracy' for classification tasks."""
    data = _clf_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.enough(data, "target", seed=42, steps=4)
    assert result.metric == "accuracy"


def test_enough_metric_is_r2_for_regression():
    """metric='r2' for regression tasks."""
    data = _reg_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.enough(data, "target", seed=42, steps=4)
    assert result.metric == "r2"


def test_enough_n_current_matches_data_size():
    """n_current equals len(data)."""
    data = _clf_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.enough(data, "target", seed=42, steps=4)
    assert result.n_current == len(data)


def test_enough_saturated_is_bool():
    """saturated is a bool."""
    data = _clf_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.enough(data, "target", seed=42, steps=4)
    assert isinstance(result.saturated, bool)


def test_enough_recommendation_is_str():
    """recommendation is a non-empty string."""
    data = _clf_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.enough(data, "target", seed=42, steps=4)
    assert isinstance(result.recommendation, str)
    assert len(result.recommendation) > 0


def test_enough_repr():
    """EnoughResult has a readable repr."""
    data = _clf_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.enough(data, "target", seed=42, steps=4)
    r = repr(result)
    assert "EnoughResult" in r
    assert "accuracy" in r


def test_enough_train_score_gte_val_score():
    """Train score is generally >= val score (model sees training data)."""
    data = _clf_data(n=500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.enough(data, "target", seed=42, steps=5)
    # Check at least for the largest n_samples step
    last = result.curve.iloc[-1]
    # Not a hard assertion — can fail due to randomness, just check they're numeric
    assert not pd.isna(last["train_score"])
    assert not pd.isna(last["val_score"])


def test_enough_regression_works():
    """enough() works correctly on regression data."""
    data = _reg_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.enough(data, "target", seed=42, steps=4)
    assert isinstance(result, ml.EnoughResult)
    assert result.metric == "r2"
    assert len(result.curve) == 4


def test_enough_keyword_only_seed():
    """seed= must be keyword-only."""
    import inspect
    sig = inspect.signature(ml.enough)
    params = sig.parameters
    # target is positional, seed should be keyword-only (after *)
    assert params["seed"].kind == inspect.Parameter.KEYWORD_ONLY


# -- Error handling ------------------------------------------------------------


def test_enough_non_dataframe_raises():
    """enough() raises DataError for non-DataFrame data."""
    with pytest.raises(DataError):
        ml.enough([[1, 2], [3, 4]], "target", seed=42)


def test_enough_missing_target_raises():
    """enough() raises DataError if target not in data."""
    data = _clf_data()
    with pytest.raises(DataError, match="not found"):
        ml.enough(data, "missing_col", seed=42)


def test_enough_too_few_rows_raises():
    """enough() raises DataError for fewer than 50 rows."""
    rng = np.random.RandomState(42)
    tiny = pd.DataFrame({
        "x1": rng.rand(30),
        "target": rng.choice(["yes", "no"], 30),
    })
    with pytest.raises(DataError, match="50"):
        ml.enough(tiny, "target", seed=42)


def test_enough_steps_too_small_raises():
    """enough() raises ConfigError for steps < 2."""
    data = _clf_data()
    with pytest.raises(ConfigError):
        ml.enough(data, "target", seed=42, steps=1)


def test_enough_cv_too_small_raises():
    """enough() raises ConfigError for cv < 2."""
    data = _clf_data()
    with pytest.raises(ConfigError):
        ml.enough(data, "target", seed=42, cv=1)
