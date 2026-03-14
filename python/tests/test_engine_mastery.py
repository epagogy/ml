"""Engine Mastery — tests for deep GBDT search spaces and new fit() params.

Tests the following improvements:
1.1 Deep LightGBM search space (num_leaves, path_smooth, min_split_gain, DART)
1.1b Conditional force_col_wise (moved to test_harvest.py — 2 tests there)
1.2 Deep CatBoost search space
1.3 Deep XGBoost search space (gamma, grow_policy)
1.4 balance= in tune()
1.5 Configurable early stopping (early_stopping=, eval_fraction=)
1.6 Monotonicity constraints (monotone=)
1.7 HistGradient search space
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import ml  # noqa: I001
from ml.tune import TUNE_DEFAULTS

pytestmark = pytest.mark.slow  # all engines + tune trials — 13s server, 424 MB peak

# ===== FIXTURES =====

@pytest.fixture
def classification_data():
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "age": rng.randint(18, 80, n).astype(float),
        "income": rng.rand(n) * 100_000,
        "score": rng.rand(n),
        "target": rng.choice([0, 1], n),
    })


@pytest.fixture
def imbalanced_data():
    """95/5 class split for balance= testing."""
    rng = np.random.RandomState(42)
    n = 300
    return pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "x3": rng.rand(n),
        "target": np.concatenate([np.zeros(285, dtype=int), np.ones(15, dtype=int)]),
    })


@pytest.fixture
def monotone_data():
    """Synthetic data where income always → higher risk (monotone=1)."""
    rng = np.random.RandomState(42)
    n = 300
    income = rng.rand(n) * 100_000
    risk = (income > 50_000).astype(int)
    return pd.DataFrame({"income": income, "noise": rng.rand(n), "risk": risk})


# ===== 1.1 DEEP LIGHTGBM SEARCH SPACE =====

def test_tune_lightgbm_has_num_leaves():
    """LightGBM TUNE_DEFAULTS now includes num_leaves — the critical tree param."""
    assert "num_leaves" in TUNE_DEFAULTS["lightgbm"], (
        "num_leaves missing from LightGBM TUNE_DEFAULTS — this is the single most "
        "important LightGBM parameter (leaf-wise tree growth)."
    )
    low, high = TUNE_DEFAULTS["lightgbm"]["num_leaves"]
    assert low >= 15, f"num_leaves lower bound too low: {low}"
    assert high >= 100, f"num_leaves upper bound too low: {high}"


def test_tune_lightgbm_has_path_smooth():
    """LightGBM search space includes path_smooth (reduces overfitting)."""
    assert "path_smooth" in TUNE_DEFAULTS["lightgbm"]


def test_tune_lightgbm_has_min_split_gain():
    """LightGBM search space includes min_split_gain."""
    assert "min_split_gain" in TUNE_DEFAULTS["lightgbm"]


def test_tune_lightgbm_dart_entry_exists():
    """lightgbm_dart entry exists in TUNE_DEFAULTS with DART-specific params."""
    assert "lightgbm_dart" in TUNE_DEFAULTS, "lightgbm_dart missing from TUNE_DEFAULTS"
    dart = TUNE_DEFAULTS["lightgbm_dart"]
    assert "drop_rate" in dart, "drop_rate missing from lightgbm_dart"
    assert "skip_drop" in dart, "skip_drop missing from lightgbm_dart"
    assert "num_leaves" in dart, "num_leaves should carry over to lightgbm_dart"


def test_tune_lightgbm_runs_with_num_leaves(classification_data):
    """tune() with lightgbm actually searches num_leaves."""
    result = ml.tune(
        classification_data, "target",
        algorithm="lightgbm", seed=42, n_trials=5, cv_folds=2,
    )
    assert "num_leaves" in result.best_params_, (
        "num_leaves not in tuned params — search space not applied"
    )
    assert isinstance(result.best_params_["num_leaves"], int)


# ===== 1.2 DEEP CATBOOST SEARCH SPACE =====

def test_tune_catboost_has_deep_search_space():
    """CatBoost TUNE_DEFAULTS now has 9 params (was 3)."""
    cb = TUNE_DEFAULTS["catboost"]
    required = ["depth", "learning_rate", "iterations", "l2_leaf_reg",
                "random_strength", "bagging_temperature", "border_count",
                "min_data_in_leaf"]
    missing = [p for p in required if p not in cb]
    assert not missing, f"CatBoost TUNE_DEFAULTS missing: {missing}"


def test_tune_catboost_grow_policy_is_categorical():
    """CatBoost grow_policy is a categorical list (SymmetricTree, Lossguide, Depthwise)."""
    cb = TUNE_DEFAULTS["catboost"]
    assert "grow_policy" in cb
    assert isinstance(cb["grow_policy"], list)
    assert "SymmetricTree" in cb["grow_policy"]


# ===== 1.3 DEEP XGBOOST SEARCH SPACE =====

def test_tune_xgboost_has_gamma():
    """XGBoost TUNE_DEFAULTS includes gamma (float range for min split reduction)."""
    xgb = TUNE_DEFAULTS["xgboost"]
    assert "gamma" in xgb, "gamma missing from XGBoost TUNE_DEFAULTS"
    low, high = xgb["gamma"]
    assert isinstance(low, float), "gamma should be float range"
    assert high >= 1.0


def test_tune_xgboost_has_grow_policy():
    """XGBoost TUNE_DEFAULTS includes grow_policy."""
    xgb = TUNE_DEFAULTS["xgboost"]
    assert "grow_policy" in xgb
    assert isinstance(xgb["grow_policy"], list)


def test_tune_xgboost_deeper_budget():
    """XGBoost n_estimators upper bound is now 1000 (was 300) for early stopping."""
    xgb = TUNE_DEFAULTS["xgboost"]
    low, high = xgb["n_estimators"]
    assert high >= 1000, f"XGBoost n_estimators upper bound too low: {high}"


# ===== 1.4 BALANCE IN TUNE =====

def test_tune_balance_runs_imbalanced(imbalanced_data):
    """tune(balance=True) runs without error on imbalanced data."""
    result = ml.tune(
        imbalanced_data, "target",
        algorithm="xgboost", seed=42, n_trials=5, cv_folds=2,
        balance=True,
    )
    assert result.best_model is not None


def test_tune_balance_false_default(classification_data):
    """tune() default balance=False is backward-compatible."""
    import inspect
    sig = inspect.signature(ml.tune)
    assert "balance" in sig.parameters
    assert sig.parameters["balance"].default is False


def test_tune_balance_weights_conflict(classification_data):
    """tune(balance=True, weights='w') raises ConfigError."""
    from ml._types import ConfigError
    df = classification_data.copy()
    df["w"] = 1.0
    with pytest.raises(ConfigError, match="Cannot use balance=True and weights="):
        ml.tune(df, "target", algorithm="xgboost", seed=42, balance=True, weights="w")


# ===== 1.5 CONFIGURABLE EARLY STOPPING =====

def test_fit_early_stopping_off(classification_data):
    """early_stopping=False uses all data (no 10% carve)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ml.fit(
            classification_data, "target",
            algorithm="lightgbm", seed=42,
            early_stopping=False,
        )
    # No early stopping warning should fire
    es_warns = [w for w in caught if "Early stopping" in str(w.message)]
    assert len(es_warns) == 0, "early_stopping=False should not carve eval set"
    # All rows used for training
    assert model._n_train_actual is None


def test_fit_early_stopping_custom_patience(classification_data):
    """early_stopping=50 sets patience=50 (not default 10)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ml.fit(
            classification_data, "target",
            algorithm="xgboost", seed=42,
            early_stopping=50,
        )
    # Early stopping should still fire (patience is set)
    es_warns = [w for w in caught if "Early stopping" in str(w.message)]
    assert len(es_warns) >= 1
    # Model should record actual training size
    assert model._n_train_actual is not None


def test_fit_early_stopping_config_stored(classification_data):
    """early_stopping=False → model._n_train_actual is None (no carve recorded)."""
    model = ml.fit(
        classification_data, "target",
        algorithm="xgboost", seed=42, early_stopping=False,
    )
    # With early_stopping=False, no carve → n_train_actual is None
    assert model._n_train_actual is None
    # n_train covers all rows (minus target)
    assert model._n_train == len(classification_data)


# ===== 1.6 MONOTONICITY CONSTRAINTS =====

def test_fit_monotone_xgboost_runs(monotone_data):
    """fit(monotone=...) runs for XGBoost without error."""
    model = ml.fit(
        monotone_data, "risk",
        algorithm="xgboost", seed=42,
        monotone={"income": 1},  # income should monotonically increase risk
        early_stopping=False,
    )
    assert model is not None
    assert model.algorithm == "xgboost"


def test_fit_monotone_lightgbm_runs(monotone_data):
    """fit(monotone=...) runs for LightGBM without error."""
    model = ml.fit(
        monotone_data, "risk",
        algorithm="lightgbm", seed=42,
        monotone={"income": 1},
        early_stopping=False,
    )
    assert model is not None


def test_fit_monotone_unsupported_raises(classification_data):
    """fit(monotone=...) raises ConfigError for algorithms that don't support it."""
    from ml._types import ConfigError
    with pytest.raises(ConfigError, match="monotone=.*not supported.*algorithm='svm'"):
        ml.fit(
            classification_data, "target",
            algorithm="svm", seed=42,
            monotone={"income": 1},
        )


# ===== 1.7 HISTGRADIENT SEARCH SPACE =====

def test_tune_histgradient_search_space_exists():
    """histgradient has a tuning search space (was missing entirely)."""
    assert "histgradient" in TUNE_DEFAULTS
    hg = TUNE_DEFAULTS["histgradient"]
    required = ["max_depth", "learning_rate", "max_iter", "min_samples_leaf",
                "max_leaf_nodes", "l2_regularization", "max_bins"]
    missing = [p for p in required if p not in hg]
    assert not missing, f"histgradient TUNE_DEFAULTS missing: {missing}"


def test_tune_histgradient_runs(classification_data):
    """tune() with histgradient runs end-to-end."""
    result = ml.tune(
        classification_data, "target",
        algorithm="histgradient", seed=42, n_trials=5, cv_folds=2,
    )
    assert result.best_model is not None
    assert "max_iter" in result.best_params_ or "learning_rate" in result.best_params_
