"""Tests for ml.cluster_features() — correlated feature grouping. Chain 4.4."""

import numpy as np
import pandas as pd
import pytest

import ml
from ml.cluster import cluster_features


@pytest.fixture
def corr_df():
    """DataFrame with known correlation structure."""
    rng = np.random.RandomState(42)
    n = 300
    x1 = rng.rand(n)
    return pd.DataFrame({
        "x1": x1,
        "x1_copy": x1 + rng.rand(n) * 0.01,   # almost identical to x1 (r≈0.999)
        "x2": rng.rand(n),                       # independent
        "x3": rng.rand(n),                       # independent
    })


def test_cluster_features_returns_list_of_lists(corr_df):
    """cluster_features() returns list[list[str]]. Chain 4.4."""
    groups = cluster_features(corr_df)
    assert isinstance(groups, list)
    for g in groups:
        assert isinstance(g, list)
        assert all(isinstance(c, str) for c in g)


def test_cluster_features_covers_all_columns(corr_df):
    """All columns appear in exactly one group. Chain 4.4."""
    groups = cluster_features(corr_df)
    all_cols = [c for g in groups for c in g]
    assert sorted(all_cols) == sorted(corr_df.columns.tolist())
    # No duplicates
    assert len(all_cols) == len(set(all_cols))


def test_cluster_features_groups_correlated(corr_df):
    """Highly correlated features (x1, x1_copy) end up in the same group. Chain 4.4."""
    groups = cluster_features(corr_df, threshold=0.95)
    found = False
    for g in groups:
        if "x1" in g and "x1_copy" in g:
            found = True
            break
    assert found, f"x1 and x1_copy should be in the same group, got {groups}"


def test_cluster_features_independent_in_separate_groups(corr_df):
    """Independent features (x2, x3) end up in separate groups from x1. Chain 4.4."""
    groups = cluster_features(corr_df, threshold=0.95)
    # x2 and x3 should not be in the same group as x1/x1_copy
    x1_group = next((g for g in groups if "x1" in g), [])
    assert "x2" not in x1_group
    assert "x3" not in x1_group


def test_cluster_features_via_ml_namespace(corr_df):
    """ml.cluster_features() accessible from public API. Chain 4.4."""
    groups = ml.cluster_features(corr_df)
    assert isinstance(groups, list)
