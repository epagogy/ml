"""Tests for evaluate()."""

import pytest

import ml


def test_evaluate_binary_classification(small_classification_data):
    """Test evaluate returns correct metrics for binary classification."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    metrics = ml.evaluate(model=model, data=s.valid)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "roc_auc" in metrics

    # Check values are in valid range
    for k, v in metrics.items():
        assert 0 <= v <= 1, f"{k}={v} out of range"


def test_evaluate_multiclass(multiclass_data):
    """Test evaluate returns weighted and macro metrics for multiclass."""
    s = ml.split(data=multiclass_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    metrics = ml.evaluate(model=model, data=s.valid)

    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert "f1_macro" in metrics
    assert "precision_weighted" in metrics
    assert "recall_weighted" in metrics
    assert "roc_auc_ovr" in metrics


def test_evaluate_regression(small_regression_data):
    """Test evaluate returns correct metrics for regression."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    metrics = ml.evaluate(model=model, data=s.valid)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics

    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0


def test_evaluate_target_not_found_error(small_classification_data):
    """Test evaluate raises when target not in data."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    no_target = s.valid.drop(columns=["target"])

    with pytest.raises(ml.DataError, match="target column 'target' not found"):
        ml.evaluate(model=model, data=no_target)


def test_evaluate_returns_dict(small_classification_data):
    """Test evaluate always returns dict, not DataFrame."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    metrics = ml.evaluate(model=model, data=s.valid)

    assert isinstance(metrics, dict)
    assert not isinstance(metrics, type(None))


def test_metrics_attribute_access(small_classification_data):
    """Metrics supports attribute-style access: met.accuracy."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    met = ml.evaluate(model=model, data=s.valid)

    # Attribute access works
    assert met.accuracy == met["accuracy"]

    # Missing attribute gives helpful error
    with pytest.raises(AttributeError, match="No metric 'nonexistent'"):
        _ = met.nonexistent


def test_metrics_fstring_compact(small_classification_data):
    """f'{metrics}' gives compact one-line repr, not full block."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    met = ml.evaluate(model=model, data=s.valid)

    inline = f"{met}"
    # Should be one line, compact
    assert "\n" not in inline
    assert inline.startswith("{")
    assert inline.endswith("}")
    assert "accuracy=" in inline


def test_evaluate_non_dataframe_error(small_classification_data):
    """evaluate() with non-DataFrame raises DataError."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    with pytest.raises(ml.DataError):
        ml.evaluate(model=model, data="not_a_dataframe")

    with pytest.raises(ml.DataError):
        ml.evaluate(model=model, data=[1, 2, 3])


def test_evaluate_intervals_default_false(small_classification_data):
    """intervals defaults to False — no _lower/_upper keys."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    metrics = ml.evaluate(model=model, data=s.valid)

    lower_keys = [k for k in metrics if k.endswith("_lower")]
    assert len(lower_keys) == 0


@pytest.mark.slow
def test_evaluate_intervals_true(small_classification_data):
    """intervals=True adds _lower/_upper bootstrap CI keys for every metric."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    metrics = ml.evaluate(model=model, data=s.valid, intervals=True)

    lower_keys = [k for k in metrics if k.endswith("_lower")]
    upper_keys = [k for k in metrics if k.endswith("_upper")]
    assert len(lower_keys) > 0
    assert len(upper_keys) > 0
    assert len(lower_keys) == len(upper_keys)
    # Bounds must be ordered: lower <= point estimate <= upper
    for k in lower_keys:
        base = k[: -len("_lower")]
        if base in metrics:
            assert metrics[k] <= metrics[base] <= metrics[base + "_upper"]


def test_evaluate_custom_metrics(small_classification_data):
    """evaluate() accepts custom callable metrics merged with built-ins."""
    import numpy as np

    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    def perfect_score(y_true, y_pred):
        return float(np.mean(y_true == y_pred))

    metrics = ml.evaluate(model=model, data=s.valid, metrics={"my_acc": perfect_score})
    assert "my_acc" in metrics
    assert "accuracy" in metrics  # built-ins still present
    assert 0.0 <= metrics["my_acc"] <= 1.0


@pytest.mark.slow
def test_evaluate_tuning_result(small_classification_data):
    """evaluate() accepts TuningResult and unwraps to best_model."""
    from ml._types import TuningResult

    s = ml.split(data=small_classification_data, target="target", seed=42)
    tuned = ml.tune(data=s.train, target="target", algorithm="xgboost", seed=42)
    assert isinstance(tuned, TuningResult)

    metrics = ml.evaluate(model=tuned, data=s.valid)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics


# ── Standard error ─────────────────────────────────────────────────


def test_evaluate_se_present(small_classification_data):
    """evaluate(se=True) adds _se keys for each metric."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="logistic", seed=42)
    metrics = ml.evaluate(model=model, data=s.valid, se=True)
    # At least one _se key should be present
    se_keys = [k for k in metrics if k.endswith("_se")]
    assert len(se_keys) >= 1, "Expected at least one metric_se key"


def test_evaluate_se_reasonable(small_classification_data):
    """SE values are smaller than the metric values themselves."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="logistic", seed=42)
    metrics = ml.evaluate(model=model, data=s.valid, se=True)
    if "accuracy" in metrics and "accuracy_se" in metrics:
        assert metrics["accuracy_se"] < metrics["accuracy"]
        assert metrics["accuracy_se"] >= 0


def test_evaluate_includes_brier_score(small_classification_data):
    """evaluate() returns brier_score for classification."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    metrics = ml.evaluate(model, s.valid)
    assert "brier_score" in metrics


def test_evaluate_brier_binary(small_classification_data):
    """brier_score is between 0 and 1 for binary classification."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    metrics = ml.evaluate(model, s.valid)
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_evaluate_brier_multiclass(multiclass_data):
    """evaluate() returns brier_score for multiclass."""
    s = ml.split(data=multiclass_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    metrics = ml.evaluate(model, s.valid)
    assert "brier_score" in metrics
    assert 0.0 <= metrics["brier_score"] <= 1.0


# ── Phase 2: Correctness hardening — seed reproducibility ────────────────────


def test_evaluate_se_reproducible(small_classification_data):
    """se=True with same seed gives identical SE values."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    m1 = ml.evaluate(model=model, data=s.valid, se=True)
    m2 = ml.evaluate(model=model, data=s.valid, se=True)
    se_keys = [k for k in m1 if k.endswith("_se")]
    assert len(se_keys) > 0
    for k in se_keys:
        assert m1[k] == m2[k], f"SE not reproducible: {k}: {m1[k]} != {m2[k]}"


def test_evaluate_regression_se(small_regression_data):
    """se=True works for regression tasks (previously only clf tested)."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    metrics = ml.evaluate(model=model, data=s.valid, se=True)
    se_keys = [k for k in metrics if k.endswith("_se")]
    assert len(se_keys) >= 1
    for k in se_keys:
        assert metrics[k] >= 0


@pytest.mark.slow
def test_evaluate_regression_intervals(small_regression_data):
    """intervals=True works for regression tasks."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    metrics = ml.evaluate(model=model, data=s.valid, intervals=True)
    lower_keys = [k for k in metrics if k.endswith("_lower")]
    assert len(lower_keys) > 0
    for k in lower_keys:
        base = k[: -len("_lower")]
        if base in metrics:
            assert metrics[k] <= metrics[base + "_upper"]


# ── Phase 2: Wire orphaned fixtures ──────────────────────────────────────────


def test_dirty_df_fit_predict(dirty_df):
    """fit/predict/evaluate on dirty data (NaN + mixed types + high cardinality)."""
    import warnings

    import numpy as np
    s = ml.split(data=dirty_df, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", seed=42)
        preds = ml.predict(model, s.valid)
        metrics = ml.evaluate(model, s.valid)
    assert len(preds) == len(s.valid)
    # Metrics should be finite (not NaN)
    for k, v in metrics.items():
        assert np.isfinite(v), f"Metric {k} is not finite: {v}"


def test_imbalanced_df_roc_auc(imbalanced_df):
    """roc_auc computes without crash on extreme imbalance (95/5 split)."""
    import warnings
    s = ml.split(data=imbalanced_df, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", seed=42)
        metrics = ml.evaluate(model, s.valid)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics


def test_wide_df_logistic_converges(wide_df):
    """Logistic regression converges on wide data (p=50, n=100)."""
    import warnings
    s = ml.split(data=wide_df, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="logistic", seed=42)
        metrics = ml.evaluate(model, s.valid)
    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0


# ── Partition guards ──


def test_evaluate_rejects_test_partition(small_classification_data):
    """evaluate() rejects test-tagged data — use assess() for test data."""
    ml.config(guards="strict")
    try:
        s = ml.split(data=small_classification_data, target="target", seed=42)
        model = ml.fit(data=s.train, target="target", seed=42)
        with pytest.raises(ml.PartitionError, match="test"):
            ml.evaluate(model=model, data=s.test)
    finally:
        ml.config(guards="off")


def test_evaluate_accepts_valid_partition(small_classification_data):
    """evaluate() succeeds on valid-tagged data."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    metrics = ml.evaluate(model=model, data=s.valid)
    assert isinstance(metrics, dict)


def test_evaluate_accepts_train_partition(small_classification_data):
    """evaluate() succeeds on train-tagged data (train-set evaluation)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    metrics = ml.evaluate(model=model, data=s.train)
    assert isinstance(metrics, dict)


def test_evaluate_silent_on_untagged_data(small_classification_data):
    """evaluate() does not warn/error when data has no partition tag."""
    import warnings

    import pandas as pd

    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    untagged = pd.DataFrame(s.valid.values, columns=s.valid.columns)
    assert "_ml_partition" not in untagged.attrs

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = ml.evaluate(model=model, data=untagged)
        partition_warns = [x for x in w if "partition" in str(x.message).lower()]
        assert len(partition_warns) == 0
    assert isinstance(metrics, dict)
