"""Tests for predict(intervals=True) —"""

import warnings

import numpy as np
import pandas as pd
import pytest

import ml


def test_predict_intervals_regression(small_regression_data):
    """predict(intervals=True) returns DataFrame for regression."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    result = ml.predict(model, s.valid, intervals=True, seed=42)
    assert isinstance(result, pd.DataFrame)


def test_predict_intervals_columns(small_regression_data):
    """intervals=True DataFrame has prediction/lower/upper columns."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    result = ml.predict(model, s.valid, intervals=True, seed=42)
    assert set(result.columns) == {"prediction", "lower", "upper"}


def test_predict_intervals_clf_error(small_classification_data):
    """predict(intervals=True) raises ConfigError for classification."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    with pytest.raises(ml.ConfigError, match="classification"):
        ml.predict(model, s.valid, intervals=True, seed=42)


def test_predict_intervals_seed_required(small_regression_data):
    """predict(intervals=True) raises ConfigError when seed is None."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    with pytest.raises(ml.ConfigError, match="seed"):
        ml.predict(model, s.valid, intervals=True)  # no seed


def test_predict_intervals_confidence_level(small_regression_data):
    """intervals=True respects confidence= parameter (wider = higher confidence)."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    result_90 = ml.predict(model, s.valid, intervals=True, confidence=0.90, seed=42)
    result_50 = ml.predict(model, s.valid, intervals=True, confidence=0.50, seed=42)
    width_90 = (result_90["upper"] - result_90["lower"]).mean()
    width_50 = (result_50["upper"] - result_50["lower"]).mean()
    assert width_90 >= width_50


def test_predict_intervals_lower_le_upper(small_regression_data):
    """intervals: lower <= upper for all samples."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    result = ml.predict(model, s.valid, intervals=True, confidence=0.90, seed=42)
    assert (result["lower"] <= result["upper"]).all()


def test_predict_intervals_index_matches(small_regression_data):
    """intervals DataFrame index matches input data index."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    result = ml.predict(model, s.valid, intervals=True, seed=42)
    assert list(result.index) == list(s.valid.index)


def test_predict_intervals_numeric_output(small_regression_data):
    """intervals DataFrame contains numeric float values."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    result = ml.predict(model, s.valid, intervals=True, seed=42)
    assert result["prediction"].dtype == np.float64
    assert result["lower"].dtype == np.float64
    assert result["upper"].dtype == np.float64


def test_predict_intervals_without_target_column(small_regression_data):
    """intervals=True works on data without target column."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    data_no_target = s.valid.drop(columns=["target"])
    result = ml.predict(model, data_no_target, intervals=True, seed=42)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"prediction", "lower", "upper"}
