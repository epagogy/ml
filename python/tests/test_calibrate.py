"""Tests for ml.calibrate() — post-hoc probability calibration."""

import warnings

import numpy as np
import pandas as pd
import pytest

import ml

# -- Helpers --

def _binary_data(n=600, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice([0, 1], n),
    })


def _string_binary_data(n=600, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice(["yes", "no"], n),
    })


def _multiclass_data(n=600, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "x3": rng.rand(n),
        "target": rng.choice(["a", "b", "c"], n),
    })


def _regression_data(n=600, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.rand(n) * 100,
    })


def _fit_model(data, target="target", algorithm="logistic", seed=42):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ml.fit(data=data, target=target, algorithm=algorithm, seed=seed)


# -- Module fixture: fit once, test many calibration variants -----------------


@pytest.fixture(scope="module")
def binary_base():
    """Pre-fitted random_forest on binary data — shared across tests."""
    data = _binary_data(seed=42)
    s = ml.split(data=data, target="target", seed=42)
    model = _fit_model(s.train)
    return data, s, model


# -- Basic functionality --


def test_calibrate_basic(binary_base):
    """calibrate() returns a Model and predict works."""
    data, s, model = binary_base
    calibrated = ml.calibrate(model, data=s.valid)
    assert isinstance(calibrated, ml.Model)
    preds = ml.predict(calibrated, s.test)
    assert len(preds) == len(s.test)


def test_calibrate_auto_method_small(binary_base):
    """auto picks sigmoid for <1000 samples."""
    _, s, model = binary_base
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ml.calibrate(model, data=s.valid)
    isotonic_warns = [w for w in caught if "isotonic" in str(w.message).lower()]
    assert len(isotonic_warns) == 0


@pytest.mark.slow
def test_calibrate_auto_isotonic_large():
    """auto picks isotonic for >=1000 samples — needs large n, server only."""
    data = _binary_data(n=3000)
    s = ml.split(data=data, target="target", seed=42)
    model = _fit_model(s.train)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ml.calibrate(model, data=s.valid)

    isotonic_warns = [w for w in caught if "isotonic" in str(w.message).lower()]
    assert len(isotonic_warns) == 0


def test_calibrate_sigmoid(binary_base):
    """Explicit sigmoid method works."""
    _, s, model = binary_base
    calibrated = ml.calibrate(model, data=s.valid, method="sigmoid")
    preds = ml.predict(calibrated, s.test)
    assert len(preds) == len(s.test)


def test_calibrate_isotonic(binary_base):
    """Explicit isotonic method works (warns on small data)."""
    _, s, model = binary_base
    with pytest.warns(UserWarning, match="[Ii]sotonic"):
        calibrated = ml.calibrate(model, data=s.valid, method="isotonic")
    preds = ml.predict(calibrated, s.test)
    assert len(preds) == len(s.test)


def test_calibrate_proba_sum_to_one(binary_base):
    """Calibrated probabilities sum to ~1.0."""
    _, s, model = binary_base
    calibrated = ml.calibrate(model, data=s.valid)
    proba = calibrated.predict_proba(s.test)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_calibrate_preserves_predict():
    """Calibrated model predicts valid class labels."""
    data = _string_binary_data()
    s = ml.split(data=data, target="target", seed=42)
    model = _fit_model(s.train)
    calibrated = ml.calibrate(model, data=s.valid)
    preds = ml.predict(calibrated, s.test)
    assert set(preds.unique()).issubset({"yes", "no"})


def test_calibrate_preserves_attributes(binary_base):
    """Calibration preserves task/algorithm/features/target/seed."""
    _, s, model = binary_base
    calibrated = ml.calibrate(model, data=s.valid)
    assert calibrated.task == model.task
    assert calibrated.algorithm == model.algorithm
    assert calibrated.features == model.features
    assert calibrated.target == model.target
    assert calibrated.seed == model.seed


def test_calibrate_calibrated_flag(binary_base):
    """Calibrated model has _calibrated=True."""
    _, s, model = binary_base
    assert model._calibrated is False
    calibrated = ml.calibrate(model, data=s.valid)
    assert calibrated._calibrated is True


def test_calibrate_multiclass():
    """calibrate() works on 3+ classes."""
    data = _multiclass_data()
    s = ml.split(data=data, target="target", seed=42)
    model = _fit_model(s.train)

    calibrated = ml.calibrate(model, data=s.valid)
    preds = ml.predict(calibrated, s.test)
    assert set(preds.unique()).issubset({"a", "b", "c"})

    proba = calibrated.predict_proba(s.test)
    assert proba.shape[1] == 3
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# -- Error cases --


def test_calibrate_regression_raises():
    """calibrate() on regression model raises ModelError."""
    data = _regression_data()
    s = ml.split(data=data, target="target", seed=42)
    model = _fit_model(s.train, algorithm="linear")

    with pytest.raises(ml.ModelError, match="classification"):
        ml.calibrate(model, data=s.valid)


def test_calibrate_invalid_method_raises(binary_base):
    """Invalid method raises ConfigError. Note: 'platt' is now a valid alias for 'sigmoid'."""
    _, s, model = binary_base
    with pytest.raises(ml.ConfigError, match="method="):
        ml.calibrate(model, data=s.valid, method="unknown_method_xyz")


def test_calibrate_target_missing_raises(binary_base):
    """Missing target in data raises DataError."""
    _, s, model = binary_base
    bad_data = s.valid.drop(columns=["target"])
    with pytest.raises(ml.DataError, match="target"):
        ml.calibrate(model, data=bad_data)


def test_calibrate_too_few_samples_raises(binary_base):
    """<100 calibration samples raises DataError."""
    _, s, model = binary_base
    tiny = s.valid.head(50)
    with pytest.raises(ml.DataError, match="100"):
        ml.calibrate(model, data=tiny)


@pytest.mark.slow
def test_calibrate_tuning_result():
    """calibrate() unwraps TuningResult."""
    data = _binary_data(n=600)
    s = ml.split(data=data, target="target", seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = ml.tune(
            data=s.train,
            target="target",
            algorithm="random_forest",
            seed=42,
            n_trials=3,
        )

    calibrated = ml.calibrate(tuned, data=s.valid)
    assert isinstance(calibrated, ml.Model)
    assert calibrated._calibrated is True


def test_calibrate_string_target():
    """calibrate() works with string labels (LabelEncoder path)."""
    data = _string_binary_data()
    s = ml.split(data=data, target="target", seed=42)
    model = _fit_model(s.train)

    calibrated = ml.calibrate(model, data=s.valid)
    preds = ml.predict(calibrated, s.test)
    assert set(preds.unique()).issubset({"yes", "no"})


def test_calibrate_save_load_round_trip(binary_base, tmp_path):
    """Calibrated model survives save/load — _calibrated flag and predictions preserved."""
    _, s, model = binary_base
    calibrated = ml.calibrate(model, data=s.valid)

    path = tmp_path / "calibrated_model.ml"
    ml.save(calibrated, str(path))
    loaded = ml.load(str(path))

    assert isinstance(loaded, ml.Model)
    assert loaded._calibrated is True
    assert loaded.task == calibrated.task
    assert loaded.algorithm == calibrated.algorithm

    preds_before = ml.predict(calibrated, s.test)
    preds_after = ml.predict(loaded, s.test)
    import pandas as pd
    pd.testing.assert_series_equal(preds_before, preds_after)


def test_calibrate_already_calibrated_warns(binary_base):
    """Calling calibrate() on an already-calibrated model emits UserWarning."""
    _, s, model = binary_base
    calibrated = ml.calibrate(model, data=s.valid)
    assert calibrated._calibrated is True

    with pytest.warns(UserWarning, match="already calibrated"):
        ml.calibrate(calibrated, data=s.valid)
