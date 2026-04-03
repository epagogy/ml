"""Tests for ml.null_flags() — null indicator feature engineering. Chain 4.3."""

import numpy as np
import pandas as pd
import pytest

import ml
from ml.null_flags import null_flags


@pytest.fixture
def df_with_nulls():
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "age": rng.rand(n),
        "income": rng.rand(n),
        "score": rng.rand(n),
        "label": rng.choice([0, 1], n),
    })
    df.loc[::5, "age"] = np.nan      # ~20% NaN
    df.loc[::10, "income"] = np.nan  # ~10% NaN
    return df


def test_null_flags_adds_indicator_columns(df_with_nulls):
    """null_flags() adds {col}_is_null columns for nullable columns. Chain 4.3."""
    out = null_flags(df_with_nulls)
    assert "age_is_null" in out.columns
    assert "income_is_null" in out.columns


def test_null_flags_auto_detect_skips_complete_columns(df_with_nulls):
    """Auto-detect mode skips columns with no missing values. Chain 4.3."""
    out = null_flags(df_with_nulls)
    # 'score' has no NaN → no indicator added
    assert "score_is_null" not in out.columns
    assert "label_is_null" not in out.columns


def test_null_flags_indicator_values_binary(df_with_nulls):
    """Null indicator columns contain only 0 and 1. Chain 4.3."""
    out = null_flags(df_with_nulls)
    assert set(out["age_is_null"].unique()).issubset({0, 1})


def test_null_flags_specific_columns(df_with_nulls):
    """null_flags(columns=...) adds indicators only for specified columns. Chain 4.3."""
    out = null_flags(df_with_nulls, columns=["age"])
    assert "age_is_null" in out.columns
    assert "income_is_null" not in out.columns


def test_null_flags_preserves_originals(df_with_nulls):
    """null_flags() preserves original columns unchanged. Chain 4.3."""
    out = null_flags(df_with_nulls)
    # NaN values in original columns are still NaN after transformation
    assert out["age"].isna().sum() == df_with_nulls["age"].isna().sum()


def test_null_flags_via_ml_namespace(df_with_nulls):
    """ml.null_flags() accessible from public API. Chain 4.3."""
    out = ml.null_flags(df_with_nulls)
    assert "age_is_null" in out.columns
