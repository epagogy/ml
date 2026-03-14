"""Workflow transition tests — verifies function composition chains.

Tests the seams between verbs, not individual functions.
Each test exercises a realistic multi-step workflow that a user would follow.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import ml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_data():
    """Binary classification dataset (200 rows, 5 features)."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
    "f1": rng.randn(n),
    "f2": rng.randn(n),
    "f3": rng.randn(n),
    "f4": rng.randn(n),
    "f5": rng.randn(n),
    "target": rng.choice([0, 1], n),
    })


@pytest.fixture
def reg_data():
    """Regression dataset (200 rows, 5 features)."""
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 5)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + rng.randn(n) * 0.1
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return df


@pytest.fixture
def multiclass_data():
    """Multiclass classification dataset (300 rows, 4 features, 3 classes)."""
    rng = np.random.RandomState(42)
    n = 300
    return pd.DataFrame({
    "f1": rng.randn(n),
    "f2": rng.randn(n),
    "f3": rng.randn(n),
    "f4": rng.randn(n),
    "target": rng.choice(["cat", "dog", "fish"], n),
    })


# ---------------------------------------------------------------------------
# Workflow 1: split → fit → predict → evaluate → assess
# ---------------------------------------------------------------------------

class TestCoreChain:
    """The canonical workflow every user follows."""

    def test_full_chain_classification(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        preds = ml.predict(model, s.valid)
        metrics = ml.evaluate(model, data=s.valid)
        verdict = ml.assess(model, test=s.test)

        assert len(preds) == len(s.valid)
        assert "accuracy" in metrics
        assert "accuracy" in verdict # Evidence is a dict-like

    def test_full_chain_regression(self, reg_data):
        s = ml.split(reg_data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        preds = ml.predict(model, s.valid)
        metrics = ml.evaluate(model, data=s.valid)
        verdict = ml.assess(model, test=s.test)

        assert len(preds) == len(s.valid)
        assert "rmse" in metrics
        assert "r2" in metrics

    def test_full_chain_multiclass(self, multiclass_data):
        s = ml.split(multiclass_data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        preds = ml.predict(model, s.valid)
        metrics = ml.evaluate(model, data=s.valid)

        assert len(preds) == len(s.valid)
        assert set(preds.unique()).issubset({"cat", "dog", "fish"})
        assert "accuracy" in metrics

    def test_dev_partition_fit(self, clf_data):
        """fit on .dev (train+valid combined) for final model."""
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.dev, "target", seed=42)
        preds = ml.predict(model, s.test)
        assert len(preds) == len(s.test)

    def test_train_overfit_vs_valid(self, clf_data):
        """Training score should be >= validation score (overfitting signal)."""
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", algorithm="decision_tree", seed=42)
        train_metrics = ml.evaluate(model, data=s.train)
        valid_metrics = ml.evaluate(model, data=s.valid)
        assert train_metrics["accuracy"] >= valid_metrics["accuracy"] - 0.05


# ---------------------------------------------------------------------------
# Workflow 2: save → load → predict (roundtrip)
# ---------------------------------------------------------------------------

class TestSaveLoadChain:
    """Model serialization preserves predictions exactly."""

    def test_roundtrip_predictions_match(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        preds_before = ml.predict(model, s.valid)

        with tempfile.NamedTemporaryFile(suffix=".mlw", delete=False) as f:
            path = f.name
            try:
                ml.save(model, path)
                loaded = ml.load(path)
                preds_after = ml.predict(loaded, s.valid)
                pd.testing.assert_series_equal(preds_before, preds_after)
            finally:
                os.unlink(path)

    def test_roundtrip_evaluate_matches(self, reg_data):
        s = ml.split(reg_data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        metrics_before = ml.evaluate(model, data=s.valid)

        with tempfile.NamedTemporaryFile(suffix=".mlw", delete=False) as f:
            path = f.name
            try:
                ml.save(model, path)
                loaded = ml.load(path)
                metrics_after = ml.evaluate(loaded, data=s.valid)
                for k in metrics_before:
                    assert abs(metrics_before[k] - metrics_after[k]) < 1e-10, \
                        f"Metric {k} drifted after save/load"
            finally:
                os.unlink(path)


# ---------------------------------------------------------------------------
# Workflow 3: tune → fit → predict (hyperparameter transfer)
# ---------------------------------------------------------------------------

class TestTuneFitChain:
    """tune() best_params can be spread into fit() without error."""

    def test_tune_rf_then_fit(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        tuned = ml.tune(s.train, "target", algorithm="random_forest",
        seed=42, n_trials=3)
        model = ml.fit(s.train, "target", algorithm="random_forest",
        seed=42, **tuned.best_params)
        preds = ml.predict(model, s.valid)
        assert len(preds) == len(s.valid)

    def test_tune_knn_then_fit(self, clf_data):
        """Regression test: KNN weights param must not collide with fit(weights=)."""
        s = ml.split(clf_data, "target", seed=42)
        tuned = ml.tune(s.train, "target", algorithm="knn",
        seed=42, n_trials=3)
        # This used to fail with: weights='uniform' not found in data
        model = ml.fit(s.train, "target", algorithm="knn",
        seed=42, **tuned.best_params)
        preds = ml.predict(model, s.valid)
        assert len(preds) == len(s.valid)

    def test_tune_logistic_then_fit(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        tuned = ml.tune(s.train, "target", algorithm="logistic",
        seed=42, n_trials=3)
        model = ml.fit(s.train, "target", algorithm="logistic",
        seed=42, **tuned.best_params)
        preds = ml.predict(model, s.valid)
        assert len(preds) == len(s.valid)

    def test_tune_gradient_boosting_then_fit(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        tuned = ml.tune(s.train, "target", algorithm="gradient_boosting",
        seed=42, n_trials=3)
        model = ml.fit(s.train, "target", algorithm="gradient_boosting",
        seed=42, **tuned.best_params)
        preds = ml.predict(model, s.valid)
        assert len(preds) == len(s.valid)


# ---------------------------------------------------------------------------
# Workflow 4: screen → fit → compare
# ---------------------------------------------------------------------------

class TestScreenCompareChain:
    """screen() results inform fit(), compare() ranks them."""

    def test_screen_then_fit_top(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        lb = ml.screen(s, "target", seed=42)
        top_algo = lb.best # Leaderboard.best returns best algorithm name
        model = ml.fit(s.train, "target", algorithm=top_algo, seed=42)
        preds = ml.predict(model, s.valid)
        assert len(preds) == len(s.valid)

    def test_compare_two_algorithms(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        m1 = ml.fit(s.train, "target", algorithm="random_forest", seed=42)
        m2 = ml.fit(s.train, "target", algorithm="logistic", seed=42)
        lb = ml.compare([m1, m2], data=s.valid)
        assert len(lb.ranking) == 2


# ---------------------------------------------------------------------------
# Workflow 5: fit → explain → validate
# ---------------------------------------------------------------------------

class TestExplainValidateChain:
    """explain() output aligns with model features; validate() checks rules."""

    def test_explain_features_match_input(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", algorithm="random_forest", seed=42)
        imp = ml.explain(model)
        # Feature names in explanation should be from the input data
        feature_names = [c for c in s.train.columns if c != "target"]
        for feat in list(imp.keys())[:5]:
            assert feat in feature_names, f"Unknown feature {feat} in explanation"

    def test_validate_rules_pass(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        gate = ml.validate(model, test=s.test, rules={"accuracy": ">0.3"})
        assert gate.passed

    def test_validate_rules_fail(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        gate = ml.validate(model, test=s.test, rules={"accuracy": ">0.99"})
        assert not gate.passed


# ---------------------------------------------------------------------------
# Workflow 6: fit → calibrate → predict
# ---------------------------------------------------------------------------

class TestCalibrateChain:
    """calibrate() preserves the model interface."""

    def test_calibrate_then_predict(self):
        """calibrate needs >= 100 samples, so use bigger data."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
        "f1": rng.randn(500), "f2": rng.randn(500),
        "target": rng.choice([0, 1], 500),
        })
        s = ml.split(data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        cal = ml.calibrate(model, data=s.valid)
        preds = ml.predict(cal, s.test)
        assert len(preds) == len(s.test)

    def test_calibrate_proba_sums_to_one(self):
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
        "f1": rng.randn(500), "f2": rng.randn(500),
        "target": rng.choice([0, 1], 500),
        })
        s = ml.split(data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        cal = ml.calibrate(model, data=s.valid)
        proba = cal.predict_proba(s.test)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Workflow 7: fit → drift (monitoring)
# ---------------------------------------------------------------------------

class TestDriftChain:
    """drift() detects distribution shift between reference and new data."""

    def test_no_drift_same_data(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        report = ml.drift(reference=s.train, new=s.train)
        # Same data → no drift
        assert not report.drifted

    def test_drift_synthetic_shift(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        shifted = s.valid.copy()
        shifted["f1"] = shifted["f1"] + 10 # massive shift
        report = ml.drift(reference=s.train, new=shifted, exclude=["target"])
        assert report.drifted


# ---------------------------------------------------------------------------
# Workflow 8: stack → evaluate (ensemble)
# ---------------------------------------------------------------------------

class TestStackChain:
    """stack() builds ensemble → evaluate/predict work."""

    def test_stack_then_evaluate(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        stacked = ml.stack(s.train, "target", seed=42)
        metrics = ml.evaluate(stacked, data=s.valid)
        assert "accuracy" in metrics

    def test_stack_then_predict(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        stacked = ml.stack(s.train, "target", seed=42)
        preds = ml.predict(stacked, s.valid)
        assert len(preds) == len(s.valid)


# ---------------------------------------------------------------------------
# Workflow 9: cross-algorithm consistency
# ---------------------------------------------------------------------------

class TestCrossAlgorithm:
    """Same data, different algorithms — predictions must be consistent shape."""

    @pytest.mark.parametrize("algo", [
    "random_forest", "decision_tree", "logistic", "knn", "naive_bayes",
    ])
    def test_predict_shape_consistent(self, clf_data, algo):
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", algorithm=algo, seed=42)
        preds = ml.predict(model, s.valid)
        assert len(preds) == len(s.valid)
        assert preds.dtype in (np.int64, np.float64, object)

    def test_seed_determinism(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        m1 = ml.fit(s.train, "target", seed=42)
        m2 = ml.fit(s.train, "target", seed=42)
        p1 = ml.predict(m1, s.valid)
        p2 = ml.predict(m2, s.valid)
        pd.testing.assert_series_equal(p1, p2)


# ---------------------------------------------------------------------------
# Workflow 10: edge case transitions
# ---------------------------------------------------------------------------

class TestEdgeCaseTransitions:
    """Weird inputs that might corrupt state between functions."""

    def test_predict_with_extra_columns(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        extra = s.valid.copy()
        extra["bonus_col"] = 999
        preds = ml.predict(model, extra)
        assert len(preds) == len(s.valid)

    def test_predict_with_reordered_columns(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        reordered = s.valid[list(reversed(s.valid.columns))]
        preds = ml.predict(model, reordered)
        assert len(preds) == len(s.valid)

    def test_string_target_roundtrip(self):
        """String target labels survive fit→predict→evaluate."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
        "f1": rng.randn(100),
        "f2": rng.randn(100),
        "label": rng.choice(["yes", "no"], 100),
        })
        s = ml.split(data, "label", seed=42)
        model = ml.fit(s.train, "label", seed=42)
        preds = ml.predict(model, s.valid)
        assert set(preds.unique()).issubset({"yes", "no"})

    def test_nan_features_survive_chain(self):
        """NaN in features doesn't corrupt downstream predictions."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
        "f1": rng.randn(200),
        "f2": rng.randn(200),
        "target": rng.choice([0, 1], 200),
        })
        # Inject NaN
        data.loc[0:5, "f1"] = np.nan
        s = ml.split(data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        preds = ml.predict(model, s.valid)
        assert len(preds) == len(s.valid)

    def test_enough_guard_restore(self, clf_data):
        """enough() must restore partition guards even on success."""
        from ml._config import _CONFIG
        old = _CONFIG.get("guards", "strict")
        ml.enough(clf_data, "target", seed=42, steps=3)
        assert _CONFIG.get("guards", "strict") == old


# ---------------------------------------------------------------------------
# Workflow 11: cv → fit → evaluate
# ---------------------------------------------------------------------------

class TestCVChain:
    """Cross-validation path works end-to-end."""

    def test_cv_fit_evaluate(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        cv = ml.cv(s, seed=42, folds=3)
        model = ml.fit(cv, "target", seed=42)
        assert model is not None
        assert model.algorithm is not None


# ---------------------------------------------------------------------------
# Workflow 12: optimize → predict (threshold tuning)
# ---------------------------------------------------------------------------

class TestOptimizeChain:
    """optimize() finds best threshold, predict uses it."""

    def test_optimize_then_predict(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        opt = ml.optimize(model, data=s.valid, metric="f1")
        assert hasattr(opt, "threshold")
        assert 0.0 < opt.threshold < 1.0

    def test_optimize_returns_tuned_model(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        opt = ml.optimize(model, data=s.valid, metric="f1")
        # OptimizeResult has .threshold and .model, supports float()
        assert hasattr(opt, "model")
        assert float(opt) == opt.threshold
        # Predict through the optimized model
        preds = ml.predict(opt.model, s.test)
        assert len(preds) == len(s.test)


# ---------------------------------------------------------------------------
# Workflow 13: shelf (model staleness monitoring)
# ---------------------------------------------------------------------------

class TestShelfChain:
    """shelf() detects model degradation on new data."""

    def test_shelf_fresh_on_same_distribution(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        result = ml.shelf(model, new=s.valid, target="target")
        assert isinstance(result, ml.ShelfResult)
        assert result.fresh is not None

    def test_shelf_returns_metrics(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        result = ml.shelf(model, new=s.valid, target="target")
        assert isinstance(result.metrics_now, dict)


# ---------------------------------------------------------------------------
# Workflow 14: interact (feature interactions)
# ---------------------------------------------------------------------------

class TestInteractChain:
    """interact() detects feature interactions from a fitted model."""

    def test_interact_returns_pairs(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", algorithm="random_forest", seed=42)
        result = ml.interact(model, data=s.valid, seed=42)
        assert isinstance(result, ml.InteractResult)
        assert hasattr(result, "pairs")


# ---------------------------------------------------------------------------
# Workflow 15: blend (multi-model)
# ---------------------------------------------------------------------------

class TestBlendChain:
    """blend() combines predictions from multiple models."""

    def test_blend_two_models(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        m1 = ml.fit(s.train, "target", algorithm="random_forest", seed=42)
        m2 = ml.fit(s.train, "target", algorithm="logistic", seed=42)
        p1 = ml.predict(m1, s.valid, proba=True)
        p2 = ml.predict(m2, s.valid, proba=True)
        blended = ml.blend([p1, p2])
        assert len(blended) == len(s.valid)


# ---------------------------------------------------------------------------
# Workflow 16: nested_cv
# ---------------------------------------------------------------------------

class TestNestedCVChain:
    """nested_cv() runs full nested cross-validation pipeline."""

    def test_nested_cv_returns_result(self, clf_data):
        result = ml.nested_cv(
        clf_data, "target", seed=42,
        outer_folds=3, inner_folds=2,
        )
        assert isinstance(result, ml.NestedCVResult)
        assert result.best_algorithm != ""


# ---------------------------------------------------------------------------
# Workflow 17: encode → fit → predict (preprocessing chain)
# ---------------------------------------------------------------------------

class TestPreprocessingChain:
    """Encoding/imputing/scaling → fit → predict preserves alignment."""

    def test_encode_onehot_then_fit(self):
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
        "color": rng.choice(["red", "blue", "green"], 200),
        "size": rng.randn(200),
        "target": rng.choice([0, 1], 200),
        })
        enc = ml.encode(data.drop(columns=["target"]), columns=["color"],
        method="onehot")
        encoded_train = enc.transform(data.drop(columns=["target"]))
        encoded_train["target"] = data["target"].values
        s = ml.split(encoded_train, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        preds = ml.predict(model, s.valid)
        assert len(preds) == len(s.valid)

    def test_impute_then_fit(self):
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
        "f1": rng.randn(200),
        "f2": rng.randn(200),
        "target": rng.choice([0, 1], 200),
        })
        data.loc[0:10, "f1"] = np.nan
        data.loc[5:15, "f2"] = np.nan
        imp = ml.impute(data[["f1", "f2"]])
        clean = imp.transform(data[["f1", "f2"]])
        assert clean.isna().sum().sum() == 0
        clean["target"] = data["target"].values
        s = ml.split(clean, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        preds = ml.predict(model, s.valid)
        assert len(preds) == len(s.valid)

    def test_scale_then_fit(self):
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
        "f1": rng.randn(200) * 1000,
        "f2": rng.randn(200) * 0.001,
        "target": rng.choice([0, 1], 200),
        })
        scl = ml.scale(data[["f1", "f2"]])
        scaled = scl.transform(data[["f1", "f2"]])
        # Scaled features should have ~0 mean, ~1 std
        assert abs(scaled["f1"].mean()) < 0.2
        scaled["target"] = data["target"].values
        s = ml.split(scaled, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        preds = ml.predict(model, s.valid)
        assert len(preds) == len(s.valid)


# ---------------------------------------------------------------------------
# Workflow 18: temporal workflow
# ---------------------------------------------------------------------------

class TestTemporalChain:
    """split_temporal → cv_temporal → fit works end-to-end."""

    def test_temporal_split_then_fit(self):
        rng = np.random.RandomState(42)
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        data = pd.DataFrame({
        "f1": rng.randn(200),
        "f2": rng.randn(200),
        "date": dates,
        "target": rng.choice([0, 1], 200),
        })
        s = ml.split_temporal(data, "target", time="date")
        model = ml.fit(s.train, "target", seed=42)
        preds = ml.predict(model, s.valid)
        assert len(preds) == len(s.valid)
        # Temporal split should produce non-empty partitions
        assert len(s.train) > 0
        assert len(s.valid) > 0
        assert len(s.test) > 0

    def test_temporal_cv_then_fit(self):
        rng = np.random.RandomState(42)
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        data = pd.DataFrame({
        "f1": rng.randn(300),
        "f2": rng.randn(300),
        "date": dates,
        "target": rng.choice([0, 1], 300),
        })
        s = ml.split_temporal(data, "target", time="date")
        cv = ml.cv_temporal(s, folds=3)
        model = ml.fit(cv, "target", seed=42)
        assert model is not None


# ---------------------------------------------------------------------------
# Workflow 19: calibrate → save → load (calibrated roundtrip)
# ---------------------------------------------------------------------------

class TestCalibratedRoundtrip:
    """Calibrated model survives save/load cycle."""

    def test_calibrated_save_load_roundtrip(self):
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
        "f1": rng.randn(500), "f2": rng.randn(500),
        "target": rng.choice([0, 1], 500),
        })
        s = ml.split(data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        cal = ml.calibrate(model, data=s.valid)
        proba_before = cal.predict_proba(s.test)

        with tempfile.NamedTemporaryFile(suffix=".mlw", delete=False) as f:
            path = f.name
            try:
                ml.save(cal, path)
                loaded = ml.load(path)
                proba_after = loaded.predict_proba(s.test)
                pd.testing.assert_frame_equal(proba_before, proba_after)
            finally:
                os.unlink(path)


# ---------------------------------------------------------------------------
# Workflow 20: tune → stack (tuned models into ensemble)
# ---------------------------------------------------------------------------

class TestTuneStackChain:
    """Tuned models compose into stack ensemble."""

    def test_tune_then_stack(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        tuned = ml.tune(s.train, "target", algorithm="random_forest",
        seed=42, n_trials=3)
        # Stack uses its own models internally, but verify the chain doesn't crash
        stacked = ml.stack(s.train, "target", seed=42)
        metrics = ml.evaluate(stacked, data=s.valid)
        assert "accuracy" in metrics


# ---------------------------------------------------------------------------
# Workflow 21: quick (one-liner convenience)
# ---------------------------------------------------------------------------

class TestQuickChain:
    """quick() provides single-call workflow."""

    def test_quick_classification(self, clf_data):
        result = ml.quick(clf_data, "target", seed=42)
        assert result is not None

    def test_quick_regression(self, reg_data):
        result = ml.quick(reg_data, "target", seed=42)
        assert result is not None


# ---------------------------------------------------------------------------
# Workflow 22: select → fit (feature selection → modeling)
# ---------------------------------------------------------------------------

class TestSelectChain:
    """select() narrows features, then fit uses the reduced set."""

    def test_select_then_fit(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        selected = ml.select(model, data=s.train, threshold=0.01, seed=42)
        assert isinstance(selected, list)
        assert all(isinstance(f, str) for f in selected)
        # Fit on selected features only
        reduced_train = s.train[selected + ["target"]]
        reduced_model = ml.fit(reduced_train, "target", seed=42)
        reduced_preds = ml.predict(reduced_model, s.valid[selected])
        assert len(reduced_preds) == len(s.valid)


# ---------------------------------------------------------------------------
# Workflow 23: profile → check → leak (data quality trifecta)
# ---------------------------------------------------------------------------

class TestDataQualityTrifecta:
    """profile → check → leak form the data quality assessment chain."""

    def test_profile_check_leak_clean_data(self, clf_data):
        prof = ml.profile(clf_data, "target")
        assert prof is not None
        report = ml.check_data(clf_data, "target")
        assert isinstance(report, ml.CheckReport)
        leak_report = ml.leak(clf_data, "target")
        assert isinstance(leak_report, ml.LeakReport)

    def test_leak_detects_target_copy(self):
        """A perfect copy of the target should be flagged as leakage."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
        "f1": rng.randn(200),
        "leak": rng.choice([0, 1], 200),
        "target": None, # placeholder
        })
        data["target"] = data["leak"] # perfect leak
        report = ml.leak(data, "target")
        # The leaky feature should be detected
        assert not report.clean or len(report.suspects) > 0


# ---------------------------------------------------------------------------
# Workflow 24: enough → decision
# ---------------------------------------------------------------------------

class TestEnoughChain:
    """enough() provides learning curve to guide data collection."""

    def test_enough_returns_curve(self, clf_data):
        result = ml.enough(clf_data, "target", seed=42, steps=3)
        assert isinstance(result, ml.EnoughResult)
        assert len(result.curve) >= 2
        assert "n_samples" in result.curve.columns
        assert "val_score" in result.curve.columns

    def test_enough_saturation_flag(self, clf_data):
        result = ml.enough(clf_data, "target", seed=42, steps=4)
        assert isinstance(result.saturated, bool)
        assert isinstance(result.recommendation, str)


# ---------------------------------------------------------------------------
# Workflow 25: fit → update (model update / refit)
# ---------------------------------------------------------------------------

class TestUpdateChain:
    """update() refreshes a model on new data."""

    def test_update_preserves_algorithm(self, clf_data):
        s = ml.split(clf_data, "target", seed=42)
        model = ml.fit(s.train, "target", algorithm="random_forest", seed=42)
        try:
            updated = ml.update(model, data=s.dev, seed=42)
            assert updated.algorithm == model.algorithm
        except (ml.ConfigError, NotImplementedError, TypeError):
            pytest.skip("update() not supported for this workflow")


# ---------------------------------------------------------------------------
# Workflow 26: discretize → fit
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Workflow 27: split_group → cv_group → fit (group-aware workflow)
# ---------------------------------------------------------------------------

class TestGroupChain:
    """Group-aware splitting and CV — no group leaks across partitions."""

    @pytest.fixture
    def group_data(self):
        """Dataset with group structure (e.g. patients with repeated measures)."""
        rng = np.random.RandomState(42)
        n_groups = 40
        rows_per_group = 5
        n = n_groups * rows_per_group
        groups = np.repeat(np.arange(n_groups), rows_per_group)
        return pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "group_id": groups,
        "target": rng.choice([0, 1], n),
        })

    def test_split_group_then_fit(self, group_data):
        s = ml.split_group(group_data, "target", groups="group_id", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        preds = ml.predict(model, s.valid)
        assert len(preds) == len(s.valid)

    def test_split_group_no_group_leak(self, group_data):
        """No group appears in both train and test."""
        s = ml.split_group(group_data, "target", groups="group_id", seed=42)
        train_groups = set(s.train["group_id"].unique())
        test_groups = set(s.test["group_id"].unique())
        assert train_groups.isdisjoint(test_groups), \
        f"Group leak: {train_groups & test_groups}"

    def test_cv_group_then_fit(self, group_data):
        s = ml.split_group(group_data, "target", groups="group_id", seed=42)
        c = ml.cv_group(s, folds=3, groups="group_id", seed=42)
        model = ml.fit(c, "target", seed=42)
        assert model is not None

    def test_cv_group_no_group_leak_per_fold(self, group_data):
        """Within each fold, no group appears in both train and valid."""
        s = ml.split_group(group_data, "target", groups="group_id", seed=42)
        c = ml.cv_group(s, folds=3, groups="group_id", seed=42)
        # CVResult stores folds as (train_df, valid_df) DataFrame tuples
        for fold_train, fold_valid in c.folds:
            train_groups = set(fold_train["group_id"].unique())
            valid_groups = set(fold_valid["group_id"].unique())
            assert train_groups.isdisjoint(valid_groups), \
            f"Group leak in fold: {train_groups & valid_groups}"


class TestDiscretizeChain:
    """discretize() bins features before modeling."""

    def test_discretize_then_fit(self, clf_data):
        binner = ml.discretize(clf_data[["f1", "f2"]], columns=["f1", "f2"],
        n_bins=5)
        binned = binner.transform(clf_data[["f1", "f2"]])
        binned["f3"] = clf_data["f3"].values
        binned["target"] = clf_data["target"].values
        s = ml.split(binned, "target", seed=42)
        model = ml.fit(s.train, "target", seed=42)
        preds = ml.predict(model, s.valid)
        assert len(preds) == len(s.valid)
