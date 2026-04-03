"""Tests for ml.screen() — quick algorithm screening."""

import warnings

import pandas as pd
import pytest

import ml


@pytest.fixture(scope="module")
def churn_screen_result(churn_data):
    """Full 10-algorithm screen on churn — computed once, shared across tests."""
    s = ml.split(data=churn_data, target="churn", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ml.screen(data=s, target="churn", seed=42)


@pytest.mark.slow
def test_screen_basic_classification(churn_screen_result):
    """screen() returns Leaderboard with algorithms, metrics, and time columns."""
    result = churn_screen_result

    assert isinstance(result, ml.Leaderboard)
    assert len(result) > 0
    assert "algorithm" in result.columns
    assert "time_seconds" in result.columns
    metric_cols = [c for c in result.columns if c not in ["algorithm", "time_seconds"]
                   and not c.endswith(("_cv_std", "_cv_min", "_cv_max"))]
    assert len(metric_cols) > 0


def test_screen_basic_regression():
    """screen() works for regression tasks."""
    data = pd.DataFrame(
        {
            "x1": range(100),
            "x2": range(100, 200),
            "price": [i * 2.5 + 10 for i in range(100)],
        }
    )
    s = ml.split(data=data, target="price", seed=42)
    result = ml.screen(data=s, target="price", seed=42, algorithms=["linear", "random_forest"])

    assert isinstance(result, ml.Leaderboard)
    assert len(result) > 0
    assert "algorithm" in result.columns


def test_screen_subset_algorithms(small_classification_data):
    """screen() with subset of algorithms."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    result = ml.screen(
        data=s, target="target", seed=42, algorithms=["random_forest", "logistic"]
    )

    algos = result["algorithm"].tolist()
    assert "random_forest" in algos
    assert "logistic" in algos
    assert len(algos) == 2


def test_screen_seed_reproducible(small_classification_data):
    """Same seed produces same leaderboard."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    result1 = ml.screen(data=s, target="target", seed=42, algorithms=["logistic"])
    result2 = ml.screen(data=s, target="target", seed=42, algorithms=["logistic"])

    if "accuracy" in result1.columns:
        assert abs(result1["accuracy"].iloc[0] - result2["accuracy"].iloc[0]) < 0.001


# ── Gate 3 additions ──────────────────────────────────────────────────────────

def test_screen_seed_required_raises_typeerror(small_classification_data):
    """screen() requires seed= — omitting it raises TypeError."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with pytest.raises(TypeError):
        ml.screen(data=s, target="target")  # missing seed=


def test_screen_algorithms_string_raises_config_error(small_classification_data):
    """algorithms= must be a list, not a string."""
    from ml._types import ConfigError
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with pytest.raises(ConfigError, match="list"):
        ml.screen(data=s, target="target", seed=42, algorithms="xgboost")


def test_screen_multiclass_task(multiclass_data):
    """screen() works for multiclass classification."""
    s = ml.split(data=multiclass_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.screen(data=s, target="target", seed=42,
                           algorithms=["logistic", "random_forest"])
    assert isinstance(result, ml.Leaderboard)
    assert "algorithm" in result.columns
    assert len(result) > 0


@pytest.mark.slow
def test_screen_sort_by_auto_classification(churn_screen_result):
    """Default sort uses roc_auc (or roc_auc_ovr) for classification."""
    result = churn_screen_result
    # First row should have best roc_auc
    if "roc_auc" in result.columns:
        scores = result["roc_auc"].dropna()
        if len(scores) > 1:
            assert scores.iloc[0] >= scores.iloc[1]


@pytest.mark.slow
def test_screen_returns_leaderboard_with_best_property(churn_screen_result):
    """Leaderboard.best is a property returning the top algorithm name as a string."""
    assert hasattr(churn_screen_result, "best")
    winner = churn_screen_result.best
    assert isinstance(winner, str)
    assert len(winner) > 0
    # Should be one of the screened algorithms
    assert winner in churn_screen_result["algorithm"].tolist()


@pytest.mark.slow
def test_screen_best_matches_top_row(churn_screen_result):
    """Leaderboard.best matches the first row (top-ranked algorithm)."""
    winner = churn_screen_result.best
    assert winner == churn_screen_result["algorithm"].iloc[0]


def test_leaderboard_best_is_property():
    """Leaderboard.best is a @property, not a regular method."""
    from ml._types import Leaderboard
    assert isinstance(Leaderboard.best, property), (
        "Leaderboard.best must be a @property. "
        "Use lb.best (not lb.best()) to get the top algorithm name."
    )


def test_screen_uses_parallel(small_classification_data):
    """screen() uses _screen_n_jobs() for parallel-capable algorithms.

    With ml.config(n_jobs=1) set in conftest, screen respects the override.
    Without explicit config, screen defaults to n_jobs=-1 (all cores).
    """
    import warnings
    from unittest.mock import patch

    captured_kwargs = {}
    original_create = ml._engines.create

    def mock_create(algo, *, task, seed, **kwargs):
        captured_kwargs[algo] = dict(kwargs)
        return original_create(algo, task=task, seed=seed, **kwargs)

    s = ml.split(data=small_classification_data, target="target", seed=42)
    with patch("ml._engines.create", side_effect=mock_create), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.screen(s, "target", seed=42, algorithms=["random_forest"])

    # random_forest is a PARALLEL_ALGORITHM — screen passes n_jobs from _screen_n_jobs().
    # conftest sets ml.config(n_jobs=1) so _screen_n_jobs() returns 1 in test context.
    assert captured_kwargs.get("random_forest", {}).get("n_jobs") == 1


# ── A13: GPU detection ────────────────────────────────────────────────────────


# ── Chain 2: Validation Fortress ──────────────────────────────────────────────


def test_screen_timing_column(small_classification_data):
    """screen() includes time_seconds column > 0. Chain 2.7."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    result = ml.screen(data=s, target="target", seed=42,
                       algorithms=["logistic"])
    assert "time_seconds" in result.columns
    assert result["time_seconds"].iloc[0] >= 0


def test_screen_cv_std_column(small_classification_data):
    """screen() on CVResult includes cv_std columns. Chain 2.2."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    cv = ml.cv(s, folds=3, seed=42)
    result = ml.screen(data=cv, target="target", seed=42,
                       algorithms=["logistic"])
    # At least one _cv_std column should be present for CVResult
    std_cols = [c for c in result.columns if c.endswith("_cv_std")]
    assert len(std_cols) >= 1, "Expected at least one _cv_std column for CVResult"


def test_screen_cv_std_positive(small_classification_data):
    """cv_std values are non-negative. Chain 2.2."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    cv = ml.cv(s, folds=3, seed=42)
    result = ml.screen(data=cv, target="target", seed=42,
                       algorithms=["logistic"])
    std_cols = [c for c in result.columns if c.endswith("_cv_std")]
    for col in std_cols:
        assert result[col].iloc[0] >= 0


def test_screen_ranking_public_api(small_classification_data):
    """Leaderboard.ranking returns a DataFrame. Chain 2.2."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    result = ml.screen(data=s, target="target", seed=42,
                       algorithms=["logistic", "random_forest"])
    df = result.ranking
    import pandas as pd
    assert isinstance(df, pd.DataFrame)
    assert "algorithm" in df.columns
    assert len(df) == 2


def test_screen_multi_metric(small_classification_data):
    """screen(metrics=[...]) computes extra metrics per algorithm. Chain 2.3."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    result = ml.screen(data=s, target="target", seed=42,
                       algorithms=["logistic"],
                       metrics=["f1", "accuracy"])
    assert "f1" in result.columns
    assert "accuracy" in result.columns


def test_screen_multi_metric_sort_by(small_classification_data):
    """screen(metrics=[...], sort_by=...) sorts by requested metric. Chain 2.3."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    result = ml.screen(data=s, target="target", seed=42,
                       algorithms=["logistic", "random_forest"],
                       metrics=["f1"],
                       sort_by="f1")
    assert "f1" in result.columns
    # First row has highest f1
    f1_vals = result["f1"].dropna()
    if len(f1_vals) > 1:
        assert f1_vals.iloc[0] >= f1_vals.iloc[1]


# ── A13: GPU detection ────────────────────────────────────────────────────────


def test_gpu_detection():
    """_detect_gpu() returns a bool without crashing. A13."""
    from ml._engines import _detect_gpu
    result = _detect_gpu()
    assert isinstance(result, bool)


@pytest.mark.slow
def test_gpu_warning_emitted(monkeypatch):
    """When GPU is detected, xgboost engine creation emits UserWarning. A13."""
    import warnings

    from ml import _engines

    # Monkeypatch _detect_gpu to simulate GPU presence
    monkeypatch.setattr(_engines, "_detect_gpu", lambda: True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            engine = _engines.create("xgboost", task="classification", seed=42)
            # If xgboost is installed, check the warning was emitted
            gpu_warnings = [w for w in caught if "GPU" in str(w.message)]
            assert len(gpu_warnings) >= 1, "Expected GPU mode warning when GPU detected"
            # Engine should have device="cuda" set
            params = engine.get_params()
            assert params.get("device") == "cuda"
        except ml.ConfigError:
            # xgboost not installed — skip
            pytest.skip("xgboost not installed")


def test_screen_warns_small_dataset():
    """screen() warns when n_samples < n_algorithms * 20."""
    import numpy as np
    rng = np.random.RandomState(42)
    # 50 rows, screening default ~10 algorithms → 50 < 200
    small = pd.DataFrame({
        "x1": rng.randn(50),
        "x2": rng.randn(50),
        "target": rng.choice([0, 1], 50),
    })
    s = ml.split(data=small, target="target", seed=42)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ml.screen(data=s, target="target", seed=42)
        size_warns = [x for x in w if "noise" in str(x.message).lower()]
        assert len(size_warns) >= 1, f"Expected sample size warning, got: {[str(x.message) for x in w]}"
