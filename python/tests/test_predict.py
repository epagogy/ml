"""Tests for predict()."""

import numpy as np
import pandas as pd
import pytest

import ml


def test_predict_basic(small_classification_data):
    """Test basic prediction."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    preds = model.predict(s.valid)

    assert isinstance(preds, pd.Series)
    assert len(preds) == len(s.valid)
    assert set(preds.unique()).issubset({"yes", "no"})


def test_predict_preserves_index(small_classification_data):
    """Test predict preserves DataFrame index."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    preds = model.predict(s.valid)

    assert preds.index.equals(s.valid.index)


def test_predict_single_row(small_classification_data):
    """Test predict works on single row."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    single_row = s.valid.iloc[[0]]
    pred = model.predict(single_row)

    assert len(pred) == 1


def test_predict_with_target_column(small_classification_data):
    """Test predict handles DataFrame with target column present."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    # Include target in prediction data (should be dropped)
    preds = model.predict(s.valid)

    assert len(preds) == len(s.valid)


def test_predict_not_dataframe_error(small_classification_data):
    """Test predict raises if not DataFrame."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    with pytest.raises(ml.DataError, match="expects DataFrame"):
        model.predict([1, 2, 3])


def test_predict_proba_binary(small_classification_data):
    """Test predict_proba for binary classification."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    probs = model.predict_proba(s.valid)

    assert isinstance(probs, pd.DataFrame)
    assert probs.shape == (len(s.valid), 2)
    assert set(probs.columns) == set(model.classes_)
    assert probs.dtypes.iloc[0] == np.float64
    # Probabilities sum to ~1.0
    assert all(np.isclose(probs.sum(axis=1), 1.0, atol=0.01))


def test_predict_proba_multiclass(multiclass_data):
    """Test predict_proba for multiclass."""
    s = ml.split(data=multiclass_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    probs = model.predict_proba(s.valid)

    assert isinstance(probs, pd.DataFrame)
    assert probs.shape == (len(s.valid), 3)
    assert set(probs.columns) == {"red", "green", "blue"}


def test_predict_proba_regression_error(small_regression_data):
    """Test predict_proba raises on regression."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)

    with pytest.raises(ml.ModelError, match="only works for classification"):
        model.predict_proba(s.valid)


def test_predict_regression(small_regression_data):
    """Test predict for regression."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    preds = model.predict(s.valid)

    assert isinstance(preds, pd.Series)
    assert len(preds) == len(s.valid)
    assert preds.dtype == float


def test_predict_with_categorical_features():
    """Test predict with categorical features."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "cat1": rng.choice(["a", "b", "c"], 100),
        "num": rng.rand(100),
        "target": rng.choice([0, 1], 100),
    })

    s = ml.split(data=data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    preds = model.predict(s.valid)

    assert len(preds) == len(s.valid)


def test_predict_proba_param_default_false(small_classification_data):
    """predict() without proba returns Series (backwards compat)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    preds = model.predict(s.valid)

    assert isinstance(preds, pd.Series)
    assert len(preds) == len(s.valid)


def test_predict_toplevel_callable(small_classification_data):
    """ml.predict() is a callable function, not a module reference."""
    assert callable(ml.predict)
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    preds = ml.predict(model, s.valid)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(s.valid)


def test_predict_classes_attribute(small_classification_data):
    """model.classes_ lists class labels matching predict() output values."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    assert hasattr(model, "classes_")
    assert set(model.classes_) == {"yes", "no"}
    preds = ml.predict(model, s.valid)
    assert set(preds.unique()).issubset(set(model.classes_))


# ---------------------------------------------------------------------------
# 5.2: Test-time augmentation (TTA)
# ---------------------------------------------------------------------------


def test_predict_augment_shape(small_regression_data):
    """TTA predictions have same shape as non-augmented."""
    import warnings
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    pred_normal = ml.predict(model, s.valid)
    pred_tta = ml.predict(model, s.valid, augment=5, seed=42)
    assert pred_tta.shape == pred_normal.shape
    assert pred_tta.index.equals(pred_normal.index)


def test_predict_augment_none_default(small_classification_data):
    """Default (augment=None) is single-pass — identical to normal predict."""
    import warnings
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
    pred1 = ml.predict(model, s.valid)
    pred2 = ml.predict(model, s.valid, augment=None)
    assert pred1.equals(pred2)


def test_predict_augment_requires_seed(small_regression_data):
    """predict() with augment= raises ConfigError when seed is missing."""
    import warnings
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    with pytest.raises(ml.ConfigError, match="seed"):
        ml.predict(model, s.valid, augment=5)


def test_predict_augment_reduces_variance(small_regression_data):
    """TTA with more passes has lower cross-run variance than single noisy pass."""
    import warnings
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)

    # Run augment=1 (single noisy pass) many times with different seeds
    single_preds = np.array([
        ml.predict(model, s.valid, augment=1, noise_scale=0.3, seed=i).values
        for i in range(10)
    ])
    var_single = float(np.mean(np.var(single_preds, axis=0)))

    # Run augment=10 (average of 10 noisy passes) many times with different seeds
    multi_preds = np.array([
        ml.predict(model, s.valid, augment=10, noise_scale=0.3, seed=i).values
        for i in range(10)
    ])
    var_multi = float(np.mean(np.var(multi_preds, axis=0)))

    # More passes → lower variance (central limit theorem)
    assert var_multi < var_single, (
        f"Expected var_multi ({var_multi:.6f}) < var_single ({var_single:.6f})"
    )


# ── Integration tests (TabPFN + TTA) ────────────────────────────


def test_tta_works_with_any_algorithm(small_classification_data):
    """TTA integrates with any fitted model (not just TabPFN)."""
    import warnings
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)

    preds = ml.predict(model, s.valid, augment=5, seed=42)
    assert len(preds) == len(s.valid)
    assert set(preds.unique()).issubset(set(model.classes_))

    proba = ml.predict(model, s.valid, proba=True, augment=5, seed=42)
    assert proba.shape == (len(s.valid), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_tta_proba_and_class_consistency(small_classification_data):
    """TTA class predictions are consistent with argmax of TTA proba."""
    import warnings
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)

    preds_class = ml.predict(model, s.valid, augment=10, seed=42)
    preds_proba = ml.predict(model, s.valid, proba=True, augment=10, seed=42)

    # Classes from argmax of averaged probabilities
    expected_class = preds_proba.values.argmax(axis=1)
    # Map back to class labels via model.classes_
    expected_labels = np.array(model.classes_)[expected_class]

    # TTA class uses majority vote, TTA proba uses mean probabilities → argmax.
    # These can legitimately disagree for borderline samples, so require ≥90% match.
    agreement = np.mean(np.array(list(preds_class.values)) == expected_labels)
    assert agreement >= 0.9, f"TTA class vs proba argmax agreement too low: {agreement:.2f}"


# ---------------------------------------------------------------------------
# Edge Cases(10.1: all-NaN column warnings)
# ---------------------------------------------------------------------------

def test_predict_all_nan_column_warns(small_classification_data):
    """predict() emits UserWarning when a feature column is entirely NaN."""
    import warnings
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    data_nan = s.valid.copy()
    first_feature = model._features[0]
    data_nan[first_feature] = float("nan")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.predict(data_nan)
    assert any(
        "NaN" in str(warning.message) or "nan" in str(warning.message).lower()
        for warning in w
    )


def test_predict_all_nan_returns_predictions(small_classification_data):
    """predict() returns predictions even with all-NaN column (with warning)."""
    import warnings
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    data_nan = s.valid.copy()
    first_feature = model._features[0]
    data_nan[first_feature] = float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = model.predict(data_nan)
    assert len(preds) == len(s.valid)


def test_predict_empty_dataframe_raises(small_classification_data):
    """predict() raises DataError on 0-row DataFrame."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    empty = s.valid.iloc[0:0]
    with pytest.raises(ml.DataError, match="0 rows"):
        model.predict(empty)


# ── ml.predict_proba() module-level ──────────────────────────────────────────


def test_ml_predict_proba_callable(small_classification_data):
    """ml.predict_proba() is importable and callable ."""
    assert callable(ml.predict_proba)
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    probs = ml.predict_proba(model, s.valid)
    assert isinstance(probs, pd.DataFrame)
    assert all(np.isclose(probs.sum(axis=1), 1.0, atol=0.01))


def test_ml_predict_proba_matches_model_method(small_classification_data):
    """ml.predict_proba() returns same result as model.predict_proba()."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    pd.testing.assert_frame_equal(
        ml.predict_proba(model, s.valid),
        model.predict_proba(s.valid),
    )


# ── NaN passthrough for tree-based non-forest algorithms ──────────────────────


@pytest.mark.parametrize("algorithm", ["extra_trees"])
def test_nan_passthrough_tree_algos(algorithm):
    """extra_trees passes NaN through without warning."""
    import warnings

    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "x1": rng.rand(200),
        "x2": rng.rand(200),
        "target": rng.choice([0, 1], 200),
    })
    # inject NaN into x1
    data.loc[data.index[:20], "x1"] = float("nan")

    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
        _ = ml.predict(model, s.valid)

    nan_warnings = [w for w in caught if "NaN" in str(w.message) and "passed through" not in str(w.message)]
    assert len(nan_warnings) == 0, f"Unexpected NaN warning for {algorithm}: {[str(w.message) for w in nan_warnings]}"


def test_nan_imputation_gradient_boosting():
    """gradient_boosting auto-imputes NaN (sklearn GBT doesn't handle NaN natively)."""
    import warnings

    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "x1": rng.rand(200),
        "x2": rng.rand(200),
        "target": rng.choice([0, 1], 200),
    })
    data.loc[data.index[:20], "x1"] = float("nan")

    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        model = ml.fit(data=s.train, target="target", algorithm="gradient_boosting", seed=42)
        preds = ml.predict(model, s.valid)
    assert len(preds) == len(s.valid)
