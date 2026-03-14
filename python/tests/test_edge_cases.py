"""Edge case guards and tests.

Tests for: empty DataFrames, target missing messages, constant features,
duplicate columns, infinite values, tune task mismatch, save/load extensions,
assess small test set, calibrate idempotency, fuzzy algorithm names, stack
single model.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import ml

# ---------------------------------------------------------------------------
# 10.2  Empty DataFrame guards
# ---------------------------------------------------------------------------


def test_fit_empty_dataframe_raises(small_classification_data):
    """fit() raises DataError on 0-row DataFrame."""
    empty = small_classification_data.iloc[0:0]
    with pytest.raises(ml.DataError, match="0 rows"):
        ml.fit(data=empty, target="target", seed=42)


def test_evaluate_empty_dataframe_raises(small_classification_data):
    """evaluate() raises DataError on 0-row DataFrame."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    empty = s.valid.iloc[0:0]
    with pytest.raises(ml.DataError):
        ml.evaluate(model, empty)


# ---------------------------------------------------------------------------
# 10.3  Target not found — improved error message
# ---------------------------------------------------------------------------


def test_fit_target_missing_lists_columns(small_classification_data):
    """fit() error message lists available columns."""
    with pytest.raises(ml.DataError, match="Available columns"):
        ml.fit(data=small_classification_data, target="nonexistent", seed=42)


def test_split_target_missing_lists_columns(small_classification_data):
    """split() error message lists available columns."""
    with pytest.raises(ml.DataError, match="Available columns"):
        ml.split(data=small_classification_data, target="nonexistent", seed=42)


# ---------------------------------------------------------------------------
# 10.4  Constant feature warning (already in fit.py as zero-variance guard)
# ---------------------------------------------------------------------------


def test_fit_constant_feature_warning(small_classification_data):
    """fit() warns on constant (zero-variance) feature."""
    data = small_classification_data.copy()
    data["constant"] = 42
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ml.fit(data=data, target="target", seed=42)
    assert any(
        "zero variance" in str(warning.message).lower()
        or "constant" in str(warning.message).lower()
        or "variance" in str(warning.message).lower()
        for warning in w
    )


def test_fit_constant_feature_still_fits(small_classification_data):
    """fit() succeeds despite constant features."""
    data = small_classification_data.copy()
    data["constant"] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=data, target="target", seed=42)
    assert model is not None


# ---------------------------------------------------------------------------
# 10.5  Duplicate column guard (already in fit.py)
# ---------------------------------------------------------------------------


def test_duplicate_columns_error(small_classification_data):
    """fit() raises DataError on duplicate column names."""
    data = small_classification_data.copy()
    first_col = [c for c in data.columns if c != "target"][0]
    data_dup = pd.concat([data, data[[first_col]]], axis=1)
    with pytest.raises(ml.DataError, match="[Dd]uplicate"):
        ml.fit(data=data_dup, target="target", seed=42)


def test_duplicate_columns_message(small_classification_data):
    """DataError message lists the duplicate column names."""
    data = small_classification_data.copy()
    first_col = [c for c in data.columns if c != "target"][0]
    data_dup = pd.concat([data, data[[first_col]]], axis=1)
    with pytest.raises(ml.DataError) as exc_info:
        ml.fit(data=data_dup, target="target", seed=42)
    assert first_col in str(exc_info.value)


# ---------------------------------------------------------------------------
# 10.6  Infinite values guard (already in fit.py)
# ---------------------------------------------------------------------------


def test_inf_values_error(small_regression_data):
    """fit() raises DataError when data contains inf."""
    data = small_regression_data.copy()
    numeric_col = [c for c in data.columns if c != "target"][0]
    data.loc[data.index[0], numeric_col] = np.inf
    with pytest.raises(ml.DataError, match="[Ii]nf"):
        ml.fit(data=data, target="target", seed=42)


def test_inf_columns_listed(small_regression_data):
    """DataError message lists columns with inf values."""
    data = small_regression_data.copy()
    numeric_col = [c for c in data.columns if c != "target"][0]
    data.loc[data.index[0], numeric_col] = np.inf
    with pytest.raises(ml.DataError) as exc_info:
        ml.fit(data=data, target="target", seed=42)
    assert numeric_col in str(exc_info.value)


# ---------------------------------------------------------------------------
# 10.7  tune() task mismatch (logistic on regression raises)
# ---------------------------------------------------------------------------


def test_tune_classification_algo_on_regression_raises(small_regression_data):
    """tune() raises when classification-only algo used on regression data."""
    data = small_regression_data.iloc[:50]
    with pytest.raises((ml.ConfigError, ml.DataError)):
        ml.tune(
            data=data,
            target="target",
            algorithm="logistic",
            seed=42,
            n_trials=2,
        )


def test_tune_task_mismatch_message(small_regression_data):
    """Error message mentions algorithm when all trials fail due to task mismatch."""
    data = small_regression_data.iloc[:50]
    with pytest.raises((ml.ConfigError, ml.DataError)) as exc_info:
        ml.tune(
            data=data,
            target="target",
            algorithm="logistic",
            seed=42,
            n_trials=2,
        )
    error_msg = str(exc_info.value).lower()
    # tune() reports "all N trials failed" with a hint about incompatible algorithm
    assert "trial" in error_msg or "algorithm" in error_msg or "failed" in error_msg


# ---------------------------------------------------------------------------
# 10.8  save/load: auto extension + fuzzy suggestion
# ---------------------------------------------------------------------------


def test_save_auto_extension_no_ext(small_classification_data, tmp_path):
    """save() auto-appends .pyml when no extension given."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    save_path = str(tmp_path / "mymodel")
    ml.save(model=model, path=save_path)
    import os
    assert os.path.exists(save_path + ".pyml")


def test_load_missing_file_raises(tmp_path):
    """load() raises FileNotFoundError for missing file."""
    with pytest.raises((FileNotFoundError, OSError)):
        ml.load(path=str(tmp_path / "nonexistent.ml"))


def test_load_fuzzy_suggestion(small_classification_data, tmp_path):
    """load() suggests close match when file not found."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    save_path = str(tmp_path / "mymodel.ml")
    ml.save(model=model, path=save_path)
    with pytest.raises((FileNotFoundError, OSError)) as exc_info:
        ml.load(path=str(tmp_path / "mymdel.ml"))  # typo
    # The error message should suggest the correct file
    err_str = str(exc_info.value)
    assert "mymodel.ml" in err_str or "Did you mean" in err_str


# ---------------------------------------------------------------------------
# 10.9  assess() small test set warning
# ---------------------------------------------------------------------------


def test_assess_small_test_warning(small_classification_data):
    """assess() warns when test set has <30 rows."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    tiny = s.test.iloc[:10]
    # Reset assess counter so we can call assess()
    model._assess_count = 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ml.assess(model, test=tiny)
    assert any(
        "30" in str(warning.message) or "small" in str(warning.message).lower()
        for warning in w
    )


def test_assess_adequate_size_no_small_warning(small_classification_data):
    """assess() emits no small-set warning for adequate test set size."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    # Need at least 30 rows in test — use test partition (tagged correctly for assess)
    test_data = s.test if len(s.test) >= 30 else pd.concat([s.test, s.test]).iloc[:40]
    test_data.attrs["_ml_partition"] = "test"
    model._assess_count = 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ml.assess(model, test=test_data)
    small_warnings = [
        x for x in w
        if "30" in str(x.message) and "rows" in str(x.message)
    ]
    assert len(small_warnings) == 0


# ---------------------------------------------------------------------------
# 10.10  calibrate() idempotency (already in calibrate.py using _calibrated flag)
# ---------------------------------------------------------------------------


def test_calibrate_already_calibrated_warns(small_classification_data):
    """calibrate() warns when called on an already-calibrated model."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    # Need enough data for calibration (>=100 rows)
    train_data = pd.concat([s.train] * 5).reset_index(drop=True)
    valid_data = pd.concat([s.valid] * 5).reset_index(drop=True)
    model = ml.fit(data=train_data, target="target", seed=42)
    calibrated = ml.calibrate(model, data=valid_data)
    assert calibrated._calibrated is True
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ml.calibrate(calibrated, data=valid_data)
    assert any(
        "calibrat" in str(warning.message).lower()
        for warning in w
    )


def test_calibrate_double_does_not_error(small_classification_data):
    """Calibrating an already-calibrated model returns a model (not an error)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    train_data = pd.concat([s.train] * 5).reset_index(drop=True)
    valid_data = pd.concat([s.valid] * 5).reset_index(drop=True)
    model = ml.fit(data=train_data, target="target", seed=42)
    calibrated = ml.calibrate(model, data=valid_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated2 = ml.calibrate(calibrated, data=valid_data)
    assert calibrated2 is not None


# ---------------------------------------------------------------------------
# 10.11  Fuzzy algorithm name suggestion
# ---------------------------------------------------------------------------


def test_fuzzy_algorithm_suggestion(small_classification_data):
    """ConfigError for unknown algorithm includes a close-match suggestion."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with pytest.raises(ml.ConfigError) as exc_info:
        ml.fit(data=s.train, target="target", algorithm="xgbost", seed=42)  # typo
    error_msg = str(exc_info.value)
    assert "xgboost" in error_msg.lower() or "Did you mean" in error_msg


def test_completely_unknown_algorithm_still_raises(small_classification_data):
    """ConfigError for unrecognized algorithm (no close match)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with pytest.raises(ml.ConfigError):
        ml.fit(data=s.train, target="target", algorithm="neural_net_transformer_xyz", seed=42)


def test_fuzzy_metric_suggestion(small_classification_data):
    """ConfigError for unknown metric includes a close-match suggestion."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with pytest.raises(ml.ConfigError) as exc_info:
        ml.tune(
            data=s.train,
            target="target",
            algorithm="random_forest",
            metric="accurcy",  # typo of "accuracy"
            seed=42,
            n_trials=1,
        )
    error_msg = str(exc_info.value)
    assert "accuracy" in error_msg or "Did you mean" in error_msg


# ---------------------------------------------------------------------------
# 10.12  stack() single model error
# ---------------------------------------------------------------------------


def test_stack_single_model_raises(small_classification_data):
    """stack() raises ConfigError with only 1 base algorithm."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with pytest.raises(ml.ConfigError, match="[aA]t least 2"):
        ml.stack(data=s.train, target="target", models=["random_forest"], seed=42)


def test_stack_two_models_succeeds(small_classification_data):
    """stack() succeeds with 2 base algorithms."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.stack(data=s.train, target="target",
                         models=["logistic", "random_forest"], seed=42)
    assert model is not None
