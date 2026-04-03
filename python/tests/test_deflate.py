"""Tests for deflate() — multiple testing correction for model selection."""

import numpy as np
import pandas as pd
import pytest

import ml
from ml.deflate import _correct_pvalues


# ── Helpers ──────────────────────────────────────────────────────────────────


def _regression_leaderboard():
    """Two ridge + RF models on tips, regression path (paired t-test)."""
    data = ml.dataset("tips")
    s = ml.split(data, "tip", seed=42)
    m1 = ml.fit(s.train, "tip", algorithm="linear", seed=42)
    m2 = ml.fit(s.train, "tip", algorithm="random_forest", seed=42)
    lb = ml.compare([m1, m2], data=s.valid)
    return lb, s


def _classification_leaderboard():
    """Two models on iris, classification path (McNemar)."""
    data = ml.dataset("iris")
    s = ml.split(data, "species", seed=42)
    m1 = ml.fit(s.train, "species", algorithm="random_forest", seed=42)
    m2 = ml.fit(s.train, "species", algorithm="logistic", seed=42)
    lb = ml.compare([m1, m2], data=s.valid)
    return lb, s


# ── BH correction ───────────────────────────────────────────────────────────


def test_bh_regression():
    """BH correction on regression (paired t-test path) returns expected columns."""
    lb, s = _regression_leaderboard()
    result = ml.deflate(lb, data=s.valid, method="bh")
    assert isinstance(result, pd.DataFrame)
    assert "p_value" in result.columns
    assert "p_adjusted" in result.columns
    assert "significant" in result.columns
    # Baseline row gets NaN p-value
    assert result["p_value"].isna().sum() == 1


def test_bh_classification():
    """BH correction on classification (McNemar path) returns expected columns."""
    lb, s = _classification_leaderboard()
    result = ml.deflate(lb, data=s.valid, method="bh")
    assert isinstance(result, pd.DataFrame)
    assert "p_value" in result.columns
    assert "p_adjusted" in result.columns
    assert result["p_value"].isna().sum() == 1


# ── Bonferroni correction ───────────────────────────────────────────────────


def test_bonferroni():
    """Bonferroni correction produces adjusted p-values >= raw p-values."""
    lb, s = _regression_leaderboard()
    result = ml.deflate(lb, data=s.valid, method="bonferroni")
    valid = result.dropna(subset=["p_value"])
    for _, row in valid.iterrows():
        assert row["p_adjusted"] >= row["p_value"] - 1e-9


# ── Holm correction ─────────────────────────────────────────────────────────


def test_holm():
    """Holm correction runs without error and produces valid adjusted p-values."""
    lb, s = _regression_leaderboard()
    result = ml.deflate(lb, data=s.valid, method="holm")
    valid = result.dropna(subset=["p_value"])
    for _, row in valid.iterrows():
        assert 0.0 <= row["p_adjusted"] <= 1.0


# ── Error cases ──────────────────────────────────────────────────────────────


def test_no_models_raises():
    """deflate() raises ConfigError when leaderboard has no .models."""
    dummy_lb = pd.DataFrame({"algorithm": ["a"], "accuracy": [0.9]})
    with pytest.raises(ml.ConfigError, match="models"):
        ml.deflate(dummy_lb, data=pd.DataFrame({"x": [1]}))


def test_unknown_method_raises():
    """deflate() raises ConfigError for unrecognized method."""
    lb, s = _regression_leaderboard()
    with pytest.raises(ml.ConfigError, match="method="):
        ml.deflate(lb, data=s.valid, method="sidak")


def test_target_missing_raises():
    """deflate() raises DataError when target column is absent from data."""
    lb, s = _regression_leaderboard()
    data_no_target = s.valid.drop(columns=["tip"])
    with pytest.raises(ml.DataError, match="target="):
        ml.deflate(lb, data=data_no_target)


def test_single_model_raises():
    """deflate() raises ConfigError with fewer than 2 models."""
    data = ml.dataset("tips")
    s = ml.split(data, "tip", seed=42)
    m1 = ml.fit(s.train, "tip", algorithm="linear", seed=42)
    lb = ml.compare([m1], data=s.valid)
    with pytest.raises(ml.ConfigError, match="at least 2"):
        ml.deflate(lb, data=s.valid)


# ── Edge case: identical predictions ─────────────────────────────────────────


def test_identical_predictions_p_one():
    """Two identical models yield p=1.0 (no disagreement for McNemar)."""
    data = ml.dataset("iris")
    s = ml.split(data, "species", seed=42)
    m1 = ml.fit(s.train, "species", algorithm="random_forest", seed=42)
    # Same algorithm + seed = identical predictions
    m2 = ml.fit(s.train, "species", algorithm="random_forest", seed=42)
    lb = ml.compare([m1, m2], data=s.valid)
    result = ml.deflate(lb, data=s.valid)
    non_na = result.dropna(subset=["p_value"])
    assert (non_na["p_value"] == 1.0).all()


# ── _correct_pvalues internal function ───────────────────────────────────────


def test_correct_pvalues_bh_known():
    """BH correction on known values matches hand-computed result."""
    # 3 tests, raw p = [0.01, 0.04, 0.03], plus NaN for baseline
    raw = [np.nan, 0.01, 0.04, 0.03]
    adj = _correct_pvalues(raw, method="bh")
    # NaN passthrough
    assert np.isnan(adj[0])
    # m=3 valid p-values. Sorted: 0.01(rank1), 0.03(rank2), 0.04(rank3)
    # BH: p*m/rank → 0.01*3/1=0.03, 0.03*3/2=0.045, 0.04*3/3=0.04
    # Monotonicity (reverse cummin): 0.03, 0.04, 0.04
    # Unsorted back: p=0.01→0.03, p=0.04→0.04, p=0.03→0.04
    assert abs(adj[1] - 0.03) < 1e-9   # p=0.01 → adj=0.03
    assert abs(adj[2] - 0.04) < 1e-9   # p=0.04 → adj=0.04
    assert abs(adj[3] - 0.04) < 1e-9   # p=0.03 → adj=0.04


def test_correct_pvalues_bonferroni_known():
    """Bonferroni: p * m, capped at 1.0."""
    raw = [np.nan, 0.02, 0.6]
    adj = _correct_pvalues(raw, method="bonferroni")
    assert np.isnan(adj[0])
    assert abs(adj[1] - 0.04) < 1e-9   # 0.02 * 2
    assert abs(adj[2] - 1.0) < 1e-9    # 0.6 * 2 = 1.2 → capped at 1.0


def test_correct_pvalues_holm_known():
    """Holm step-down on known values."""
    raw = [np.nan, 0.01, 0.04, 0.03]
    adj = _correct_pvalues(raw, method="holm")
    assert np.isnan(adj[0])
    # m=3. Sorted: 0.01(rank1), 0.03(rank2), 0.04(rank3)
    # Holm: p*(m-i) → 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04
    # Monotonicity (cummax): 0.03, 0.06, 0.06
    # Unsorted: p=0.01→0.03, p=0.04→0.06, p=0.03→0.06
    assert abs(adj[1] - 0.03) < 1e-9
    assert abs(adj[2] - 0.06) < 1e-9
    assert abs(adj[3] - 0.06) < 1e-9


def test_correct_pvalues_all_nan():
    """All-NaN input returns same list unchanged."""
    raw = [np.nan, np.nan]
    adj = _correct_pvalues(raw, method="bh")
    assert all(np.isnan(p) for p in adj)
