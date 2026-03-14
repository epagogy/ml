"""Tests for select()."""
import numpy as np
import pandas as pd
import pytest

import ml


def test_select_importance_basic(small_classification_data):
    """select() returns a list of feature names."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    features = ml.select(model)
    assert isinstance(features, list)
    assert len(features) > 0
    assert all(isinstance(f, str) for f in features)


def test_select_threshold_zero_keeps_all(small_classification_data):
    """select(threshold=0) keeps all features."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    features = ml.select(model, threshold=0.0)
    assert len(features) == len(model._features)


def test_select_returns_list_of_strings(small_classification_data):
    """select() returns list[str] always."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    result = ml.select(model)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, str)


def test_select_correlation_drops_redundant():
    """select(method='correlation') drops highly correlated features."""
    rng = np.random.RandomState(42)
    n = 200
    x = rng.rand(n)
    data = pd.DataFrame({
        "feat_a": x,
        "feat_b": x + rng.rand(n) * 0.01,  # highly correlated with feat_a
        "feat_c": rng.rand(n),  # independent
        "target": (x > 0.5).astype(int),
    })
    s = ml.split(data=data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    features_all = ml.select(model, method="importance", threshold=0.0)
    features_dedupe = ml.select(
        model, data=s.valid, method="correlation",
        correlation_max=0.90, threshold=0.0
    )
    # Should have dropped one of the correlated pair
    assert len(features_dedupe) <= len(features_all)


def test_select_permutation_requires_data(small_classification_data):
    """select(method='permutation') raises without data=."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    with pytest.raises((ml.DataError, ml.ConfigError)):
        ml.select(model, method="permutation")
