"""Tests for _compat.to_pandas() and normalize integration."""

import pandas as pd
import pytest

import ml


def test_normalize_to_pandas_passthrough():
    """to_pandas() with pandas DataFrame returns same object (no copy)."""
    from ml._compat import to_pandas

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = to_pandas(df)
    assert isinstance(result, pd.DataFrame)
    assert result is df  # exact same object — no copy


def test_normalize_non_dataframe_raises_data_error():
    """to_pandas() with a non-DataFrame type raises DataError."""
    from ml._compat import to_pandas

    with pytest.raises(ml.DataError, match="Expected pandas or Polars DataFrame"):
        to_pandas({"a": [1, 2, 3]})


def test_split_accepts_pandas_only():
    """ml.split() with a pandas DataFrame works fine (regression guard)."""
    import numpy as np

    rng = np.random.RandomState(42)
    df = pd.DataFrame({"x": rng.rand(100), "target": rng.choice([0, 1], 100)})
    s = ml.split(data=df, target="target", seed=42)
    assert isinstance(s, ml.SplitResult)
    assert len(s.train) > 0
    assert len(s.valid) > 0
    assert len(s.test) > 0
