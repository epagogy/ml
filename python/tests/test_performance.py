"""Performance benchmarks — Chain 13.

These tests use @pytest.mark.slow and are skipped in normal test runs.
Run with: pytest tests/test_performance.py -v -m slow
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

import ml

pytestmark = pytest.mark.slow


@pytest.fixture
def medium_data():
    """1000-row classification dataset."""
    rng = np.random.RandomState(42)
    n = 1000
    return pd.DataFrame(
        {f"f{i}": rng.randn(n) for i in range(20)} | {"target": rng.choice(["yes", "no"], n)}
    )


@pytest.fixture
def medium_regression():
    """1000-row regression dataset."""
    rng = np.random.RandomState(42)
    n = 1000
    return pd.DataFrame(
        {f"f{i}": rng.randn(n) for i in range(20)} | {"target": rng.randn(n)}
    )


def test_fit_performance(medium_data):
    """fit() completes within 30 seconds for 1000 rows."""
    s = ml.split(data=medium_data, target="target", seed=42)
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _model = ml.fit(data=s.train, target="target", algorithm="xgboost", seed=42)
    elapsed = time.time() - start
    assert elapsed < 30, f"fit() took {elapsed:.1f}s (limit: 30s)"


def test_predict_performance(medium_data):
    """predict() completes within 5 seconds for 1000 rows."""
    s = ml.split(data=medium_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="xgboost", seed=42)
    start = time.time()
    _preds = ml.predict(model, s.valid)
    elapsed = time.time() - start
    assert elapsed < 5, f"predict() took {elapsed:.1f}s (limit: 5s)"


def test_screen_performance(medium_data):
    """screen() completes within 120 seconds for 1000 rows."""
    s = ml.split(data=medium_data, target="target", seed=42)
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _lb = ml.screen(s, "target", seed=42)
    elapsed = time.time() - start
    assert elapsed < 120, f"screen() took {elapsed:.1f}s (limit: 120s)"


def test_split_performance(medium_data):
    """split() completes within 1 second for 1000 rows."""
    start = time.time()
    _s = ml.split(data=medium_data, target="target", seed=42)
    elapsed = time.time() - start
    assert elapsed < 1, f"split() took {elapsed:.1f}s (limit: 1s)"
