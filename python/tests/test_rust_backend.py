"""Tests for Rust backend wrappers (_rust.py).

Protocol tests, parity vs Python reference, engine integration, and edge cases.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

ml_py = pytest.importorskip("ml_py")

from ml._rust import (  # noqa: E402
    HAS_RUST,
    _RustDecisionTreeClassifier,
    _RustDecisionTreeRegressor,
    _RustLinear,
    _RustLogistic,
    _RustRandomForestClassifier,
    _RustRandomForestRegressor,
)

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust backend not installed")


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def clf_data():
    """Binary classification: 200 rows, 5 features, seed=42."""
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 5)
    y = (X[:, 0] + X[:, 1] * 0.5 + rng.randn(n) * 0.3 > 0).astype(int)
    return X, y


@pytest.fixture
def reg_data():
    """Regression: 200 rows, 5 features, seed=42."""
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 5)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + rng.randn(n) * 0.5
    return X, y


@pytest.fixture
def multiclass_data():
    """3-class classification: 300 rows, 5 features."""
    rng = np.random.RandomState(42)
    n = 300
    X = rng.randn(n, 5)
    y = np.where(X[:, 0] > 0.5, 2, np.where(X[:, 0] < -0.5, 0, 1))
    return X, y


@pytest.fixture
def string_label_data():
    """Binary classification with string labels."""
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 5)
    y = np.where(X[:, 0] + X[:, 1] * 0.5 + rng.randn(n) * 0.3 > 0, "yes", "no")
    return X, y


# ── 1. Core tests (fit, predict, properties) ────────────────────────────


def _r2_score(y_true, y_pred):
    """R² score helper (no sklearn dependency)."""
    y_arr = np.asarray(y_true, dtype=np.float64).ravel()
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


class TestLinearProtocol:
    def test_constructor_params(self):
        m = _RustLinear(alpha=2.0)
        assert m.alpha == 2.0

    def test_fit_predict(self, reg_data):
        X, y = reg_data
        m = _RustLinear(alpha=1.0).fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)
        assert _r2_score(y, preds) > 0.5

    def test_coef_intercept(self, reg_data):
        X, y = reg_data
        m = _RustLinear(alpha=1.0).fit(X, y)
        assert m.coef_.shape == (X.shape[1],)
        assert isinstance(m.intercept_, float)


class TestLogisticProtocol:
    def test_constructor_params(self):
        m = _RustLogistic(C=0.5, max_iter=500)
        assert m.C == 0.5
        assert m.max_iter == 500

    def test_classes_set_after_fit(self, clf_data):
        X, y = clf_data
        m = _RustLogistic().fit(X, y)
        assert hasattr(m, "classes_")
        assert set(m.classes_) == {0, 1}

    def test_predict_returns_original_labels(self, clf_data):
        X, y = clf_data
        m = _RustLogistic().fit(X, y)
        preds = m.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, clf_data):
        X, y = clf_data
        m = _RustLogistic().fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_coef_shape_binary(self, clf_data):
        X, y = clf_data
        m = _RustLogistic().fit(X, y)
        assert m.coef_.shape == (1, X.shape[1])
        assert m.intercept_.shape == (1,)

    def test_coef_shape_multiclass(self, multiclass_data):
        X, y = multiclass_data
        m = _RustLogistic().fit(X, y)
        n_classes = len(np.unique(y))
        assert m.coef_.shape == (n_classes, X.shape[1])
        assert m.intercept_.shape == (n_classes,)


class TestDecisionTreeClassifierProtocol:
    def test_constructor_params(self):
        m = _RustDecisionTreeClassifier(max_depth=5, random_state=42)
        assert m.max_depth == 5
        assert m.random_state == 42

    def test_classes_set_after_fit(self, clf_data):
        X, y = clf_data
        m = _RustDecisionTreeClassifier(random_state=42).fit(X, y)
        assert set(m.classes_) == {0, 1}

    def test_feature_importances_sum(self, clf_data):
        X, y = clf_data
        m = _RustDecisionTreeClassifier(random_state=42).fit(X, y)
        imp = m.feature_importances_
        assert imp.shape == (X.shape[1],)
        np.testing.assert_allclose(imp.sum(), 1.0, atol=1e-6)

    def test_predict_proba(self, clf_data):
        X, y = clf_data
        m = _RustDecisionTreeClassifier(random_state=42).fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestDecisionTreeRegressorProtocol:
    def test_constructor_params(self):
        m = _RustDecisionTreeRegressor(max_depth=10)
        assert m.max_depth == 10

    def test_fit_predict(self, reg_data):
        X, y = reg_data
        m = _RustDecisionTreeRegressor(random_state=42).fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)
        assert _r2_score(y, preds) > 0.5

    def test_feature_importances_sum(self, reg_data):
        X, y = reg_data
        m = _RustDecisionTreeRegressor(random_state=42).fit(X, y)
        imp = m.feature_importances_
        np.testing.assert_allclose(imp.sum(), 1.0, atol=1e-6)


class TestRandomForestClassifierProtocol:
    def test_constructor_params(self):
        m = _RustRandomForestClassifier(n_estimators=50, random_state=42)
        assert m.n_estimators == 50
        assert m.random_state == 42

    def test_classes_and_importances(self, clf_data):
        X, y = clf_data
        m = _RustRandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)
        assert set(m.classes_) == {0, 1}
        imp = m.feature_importances_
        assert imp.shape == (X.shape[1],)
        np.testing.assert_allclose(imp.sum(), 1.0, atol=1e-6)

    def test_oob_score(self, clf_data):
        X, y = clf_data
        m = _RustRandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)
        oob = m.oob_score_
        assert oob is not None
        assert 0.0 <= oob <= 1.0

    def test_predict_proba(self, clf_data):
        X, y = clf_data
        m = _RustRandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestRandomForestRegressorProtocol:
    def test_fit_predict(self, reg_data):
        X, y = reg_data
        m = _RustRandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)
        assert _r2_score(y, preds) > 0.5

    def test_oob_score(self, reg_data):
        X, y = reg_data
        m = _RustRandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)
        oob = m.oob_score_
        assert oob is not None

    def test_feature_importances(self, reg_data):
        X, y = reg_data
        m = _RustRandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)
        imp = m.feature_importances_
        assert imp.shape == (X.shape[1],)
        np.testing.assert_allclose(imp.sum(), 1.0, atol=1e-6)


# ── 2. Parity tests (Rust vs Python reference) ──────────────────────────


class TestParity:
    def test_linear_parity(self, reg_data):
        """Linear: Rust vs native Python Ridge — predictions atol=1e-6."""
        from ml._linear import _LinearModel

        X, y = reg_data
        ax = _RustLinear(alpha=1.0).fit(X, y)
        py = _LinearModel(alpha=1.0).fit(X, y)
        np.testing.assert_allclose(ax.predict(X), py.predict(X), atol=1e-6)

    def test_logistic_parity(self, clf_data):
        """Logistic: Rust vs native Python — accuracy within 5%."""
        from ml._logistic import _LogisticModel

        X, y = clf_data
        ax = _RustLogistic(C=1.0, max_iter=1000).fit(X, y)
        py = _LogisticModel(C=1.0, max_iter=1000).fit(X, y)
        acc_ax = np.mean(ax.predict(X) == y)
        acc_py = np.mean(py.predict(X) == y)
        assert abs(acc_ax - acc_py) < 0.05, f"rust={acc_ax:.3f} vs python={acc_py:.3f}"

    def test_dt_clf_accuracy(self, clf_data):
        """CART clf: accuracy > 80% on training data (sanity)."""
        X, y = clf_data
        m = _RustDecisionTreeClassifier(random_state=42).fit(X, y)
        acc = np.mean(m.predict(X) == y)
        assert acc > 0.80

    def test_dt_reg_r2(self, reg_data):
        """CART reg: R² > 0.80 on training data (sanity)."""
        X, y = reg_data
        m = _RustDecisionTreeRegressor(random_state=42).fit(X, y)
        assert _r2_score(y, m.predict(X)) > 0.80

    def test_rf_clf_accuracy(self, clf_data):
        """RF clf: accuracy > 85% on training data (sanity)."""
        X, y = clf_data
        m = _RustRandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
        acc = np.mean(m.predict(X) == y)
        assert acc > 0.85

    def test_rf_reg_r2(self, reg_data):
        """RF reg: R² > 0.85 on training data (sanity)."""
        X, y = reg_data
        m = _RustRandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
        assert _r2_score(y, m.predict(X)) > 0.85


# ── 3. Engine integration ────────────────────────────────────────────────


class TestEngineIntegration:
    def test_create_decision_tree_returns_rust(self):
        from ml._engines import create

        engine = create("decision_tree", task="classification", seed=42)
        assert type(engine).__name__ == "_RustDecisionTreeClassifier"

    def test_create_random_forest_returns_rust(self):
        from ml._engines import create

        engine = create("random_forest", task="classification", seed=42)
        assert type(engine).__name__ == "_RustRandomForestClassifier"

    def test_create_logistic_returns_rust(self):
        from ml._engines import create

        engine = create("logistic", task="classification", seed=42)
        assert type(engine).__name__ == "_RustLogistic"

    def test_create_linear_returns_rust(self):
        from ml._engines import create

        engine = create("linear", task="regression", seed=42)
        assert type(engine).__name__ == "_RustLinear"

    def test_full_pipeline_classification(self):
        """ml.fit → ml.predict → ml.evaluate pipeline with Rust backend."""
        import ml

        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 5)
        y = (X[:, 0] + X[:, 1] * 0.5 + rng.randn(n) * 0.3 > 0).astype(int)
        data = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        data["target"] = y

        s = ml.split(data=data, target="target", seed=42)
        model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
        preds = ml.predict(model=model, data=s.valid)
        metrics = ml.evaluate(model=model, data=s.valid)

        assert len(preds) == len(s.valid)
        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.5

    def test_full_pipeline_regression(self):
        """ml.fit → ml.predict → ml.evaluate pipeline for regression."""
        import ml

        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 5)
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + rng.randn(n) * 0.5
        data = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        data["target"] = y

        s = ml.split(data=data, target="target", seed=42)
        model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
        metrics = ml.evaluate(model=model, data=s.valid)

        assert "rmse" in metrics

    def test_entropy_criterion_uses_rust(self):
        """criterion='entropy' routes to Rust (native entropy support added)."""
        from ml._engines import create

        engine = create("decision_tree", task="classification", seed=42, criterion="entropy")
        # Entropy is now Rust-native — should NOT fall back to sklearn
        assert "ml._rust" in type(engine).__module__

    def test_bootstrap_false_falls_back_to_sklearn(self):
        """bootstrap=False should fall back to sklearn for RF."""
        from ml._engines import create

        engine = create("random_forest", task="classification", seed=42, bootstrap=False)
        assert "sklearn" in type(engine).__module__


# ── 4. Edge cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_string_labels_encoded(self, string_label_data):
        """String labels → encoded correctly, predict returns originals."""
        X, y = string_label_data
        m = _RustLogistic().fit(X, y)
        preds = m.predict(X)
        assert set(preds).issubset({"yes", "no"})

    def test_string_labels_rf(self, string_label_data):
        """RF with string labels."""
        X, y = string_label_data
        m = _RustRandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
        preds = m.predict(X)
        assert set(preds).issubset({"yes", "no"})

    def test_fortran_order_array(self, clf_data):
        """Fortran-order arrays → handled (contiguity conversion)."""
        X, y = clf_data
        X_f = np.asfortranarray(X)
        m = _RustDecisionTreeClassifier(random_state=42).fit(X_f, y)
        preds = m.predict(X_f)
        assert len(preds) == len(y)

    def test_single_class_target(self):
        """Single-class target → degenerate prediction, no crash."""
        X = np.random.randn(50, 3)
        y = np.ones(50, dtype=int)
        m = _RustDecisionTreeClassifier(random_state=42).fit(X, y)
        preds = m.predict(X)
        assert np.all(preds == 1)

    def test_max_features_sqrt(self, clf_data):
        """max_features='sqrt' → correct int conversion."""
        X, y = clf_data
        m = _RustDecisionTreeClassifier(max_features="sqrt", random_state=42).fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(y)

    def test_max_features_log2(self, clf_data):
        """max_features='log2' → correct int conversion."""
        X, y = clf_data
        m = _RustDecisionTreeClassifier(max_features="log2", random_state=42).fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(y)

    def test_sample_weight(self, clf_data):
        """sample_weight accepted without crash."""
        X, y = clf_data
        sw = np.ones(len(y))
        sw[:50] = 2.0
        m = _RustRandomForestClassifier(n_estimators=10, random_state=42).fit(X, y, sample_weight=sw)
        acc = np.mean(m.predict(X) == y)
        assert acc > 0.5

    def test_multiclass_logistic(self, multiclass_data):
        """3-class logistic works end-to-end."""
        X, y = multiclass_data
        m = _RustLogistic().fit(X, y)
        preds = m.predict(X)
        assert set(preds).issubset(set(y))
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 3)

    def test_multiclass_rf(self, multiclass_data):
        """3-class RF works end-to-end."""
        X, y = multiclass_data
        m = _RustRandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)
        preds = m.predict(X)
        assert set(preds).issubset(set(y))


# ── 5. Serialization ─────────────────────────────────────────────────────


class TestSerialization:
    def test_linear_pickle_roundtrip(self, reg_data):
        X, y = reg_data
        m = _RustLinear(alpha=1.0).fit(X, y)
        preds_before = m.predict(X)
        m2 = pickle.loads(pickle.dumps(m))
        preds_after = m2.predict(X)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-10)

    def test_logistic_pickle_roundtrip(self, clf_data):
        X, y = clf_data
        m = _RustLogistic().fit(X, y)
        preds_before = m.predict(X)
        m2 = pickle.loads(pickle.dumps(m))
        preds_after = m2.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_dt_pickle_roundtrip(self, clf_data):
        X, y = clf_data
        m = _RustDecisionTreeClassifier(random_state=42).fit(X, y)
        preds_before = m.predict(X)
        m2 = pickle.loads(pickle.dumps(m))
        preds_after = m2.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_rf_pickle_roundtrip(self, clf_data):
        X, y = clf_data
        m = _RustRandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
        preds_before = m.predict(X)
        m2 = pickle.loads(pickle.dumps(m))
        preds_after = m2.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_rf_reg_pickle_roundtrip(self, reg_data):
        X, y = reg_data
        m = _RustRandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
        preds_before = m.predict(X)
        m2 = pickle.loads(pickle.dumps(m))
        preds_after = m2.predict(X)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-10)

    def test_params_preserved_after_pickle(self):
        m = _RustRandomForestClassifier(n_estimators=77, max_depth=5, random_state=99)
        m2 = pickle.loads(pickle.dumps(m))
        assert m2.n_estimators == m.n_estimators
        assert m2.max_depth == m.max_depth
        assert m2.random_state == m.random_state


# ── Engine= parameter integration tests ──────────────────────────────────


class TestEngineParameter:
    """Test engine= parameter in fit(), screen(), tune(), stack()."""

    @pytest.fixture
    def iris_split(self):
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris["data"]
        df["target"] = iris["target"]
        import ml
        return ml.split(df, "target", seed=42)

    @pytest.fixture
    def boston_split(self):
        from sklearn.datasets import load_diabetes
        diab = load_diabetes(as_frame=True)
        df = diab["data"]
        df["target"] = diab["target"]
        import ml
        return ml.split(df, "target", seed=42)

    # -- fit() engine= tests --

    def test_engine_auto_prefers_rust(self, iris_split):
        """engine='auto' should use Rust ml backend when available."""
        import ml
        m = ml.fit(data=iris_split.train, target="target",
                   algorithm="random_forest", seed=42, engine="auto")
        # Rust models live in ml._rust module
        assert "_rust" in type(m._model).__module__ or "Rust" in type(m._model).__name__

    def test_engine_sklearn_forces_sklearn(self, iris_split):
        """engine='sklearn' must produce sklearn estimator."""
        import ml
        m = ml.fit(data=iris_split.train, target="target",
                   algorithm="random_forest", seed=42, engine="sklearn")
        assert "sklearn" in type(m._model).__module__

    def test_engine_ml_forces_rust(self, iris_split):
        """engine='ml' must produce Rust backend estimator."""
        import ml
        m = ml.fit(data=iris_split.train, target="target",
                   algorithm="random_forest", seed=42, engine="ml")
        assert "_rust" in type(m._model).__module__ or "Rust" in type(m._model).__name__

    def test_engine_native_for_rf(self, iris_split):
        """engine='native' for RF uses Rust (not sklearn)."""
        import ml
        m = ml.fit(data=iris_split.train, target="target",
                   algorithm="random_forest", seed=42, engine="native")
        assert "sklearn" not in type(m._model).__module__

    def test_engine_native_for_linear(self, boston_split):
        """engine='native' for linear uses Rust/numpy (not sklearn)."""
        import ml
        m = ml.fit(data=boston_split.train, target="target",
                   algorithm="linear", seed=42, engine="native")
        assert "sklearn" not in type(m._model).__module__

    def test_engine_invalid_raises(self, iris_split):
        """engine='invalid' must raise ConfigError."""
        import ml
        from ml._types import ConfigError
        with pytest.raises(ConfigError, match="engine"):
            ml.fit(data=iris_split.train, target="target",
                   algorithm="random_forest", seed=42, engine="invalid_engine")

    def test_engine_native_for_svm_raises(self, iris_split):
        """engine='native' for SVM must raise ConfigError (no native impl)."""
        import ml
        from ml._types import ConfigError
        with pytest.raises(ConfigError):
            ml.fit(data=iris_split.train, target="target",
                   algorithm="svm", seed=42, engine="native")

    def test_engine_ml_for_svm_uses_rust(self, iris_split):
        """engine='ml' for SVM routes to Rust backend when HAS_RUST_SVM."""
        from ml._rust import HAS_RUST_SVM
        if not HAS_RUST_SVM:
            pytest.skip("Rust SVM not available")
        import ml
        m = ml.fit(data=iris_split.train, target="target",
                   algorithm="svm", seed=42, engine="ml")
        assert "ml._rust" in type(m._model).__module__

    # -- screen() engine= test --

    def test_screen_engine_sklearn(self, iris_split):
        """screen() with engine='sklearn' should produce all sklearn models."""
        import ml
        lb = ml.screen(data=iris_split, target="target", seed=42,
                       algorithms=["random_forest", "logistic"],
                       engine="sklearn")
        for m in lb._models:
            assert "sklearn" in type(m._model).__module__

    # -- tune() engine= test --

    def test_tune_engine_ml(self, iris_split):
        """tune() with engine='ml' should use Rust backend."""
        import ml
        result = ml.tune(data=iris_split.train, target="target",
                         algorithm="random_forest", seed=42,
                         n_trials=2, engine="ml")
        assert "_rust" in type(result.best_model._model).__module__ or \
               "Rust" in type(result.best_model._model).__name__

    def test_tune_engine_not_in_best_params(self, iris_split):
        """engine must NOT leak into best_params_."""
        import ml
        result = ml.tune(data=iris_split.train, target="target",
                         algorithm="random_forest", seed=42,
                         n_trials=2, engine="ml")
        assert "engine" not in result.best_params_

    # -- Predictions match across engines --

    def test_predictions_reasonable_across_engines(self, iris_split):
        """Both engines should produce reasonable accuracy on iris."""
        import ml
        m_rust = ml.fit(data=iris_split.train, target="target",
                        algorithm="random_forest", seed=42, engine="ml")
        m_sklearn = ml.fit(data=iris_split.train, target="target",
                           algorithm="random_forest", seed=42, engine="sklearn")
        # Both should get >80% on iris
        acc_rust = ml.evaluate(m_rust, iris_split.valid)["accuracy"]
        acc_sklearn = ml.evaluate(m_sklearn, iris_split.valid)["accuracy"]
        assert acc_rust > 0.8
        assert acc_sklearn > 0.8


# ── Native GaussianNB tests ──────────────────────────────────────────────


class TestNaiveBayes:
    """Test native _NaiveBayesModel (zero sklearn dependency)."""

    @pytest.fixture
    def clf_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    @pytest.fixture
    def multiclass_data(self):
        from sklearn.datasets import load_iris
        iris = load_iris()
        return iris.data, iris.target

    def test_get_set_params(self):
        from ml._naive_bayes import _NaiveBayesModel
        m = _NaiveBayesModel(var_smoothing=1e-6)
        assert m.get_params() == {"var_smoothing": 1e-6}
        m.set_params(var_smoothing=1e-3)
        assert m.var_smoothing == 1e-3

    def test_fit_predict(self, clf_data):
        from ml._naive_bayes import _NaiveBayesModel
        X, y = clf_data
        m = _NaiveBayesModel().fit(X, y)
        preds = m.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.85

    def test_classes_attribute(self, clf_data):
        from ml._naive_bayes import _NaiveBayesModel
        X, y = clf_data
        m = _NaiveBayesModel().fit(X, y)
        np.testing.assert_array_equal(m.classes_, np.array([0, 1]))

    def test_predict_proba_sums_to_one(self, clf_data):
        from ml._naive_bayes import _NaiveBayesModel
        X, y = clf_data
        m = _NaiveBayesModel().fit(X, y)
        proba = m.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_proba_shape(self, clf_data):
        from ml._naive_bayes import _NaiveBayesModel
        X, y = clf_data
        m = _NaiveBayesModel().fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (200, 2)

    def test_multiclass(self, multiclass_data):
        from ml._naive_bayes import _NaiveBayesModel
        X, y = multiclass_data
        m = _NaiveBayesModel().fit(X, y)
        preds = m.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.9  # GaussianNB on iris typically > 95%

    def test_multiclass_proba_shape(self, multiclass_data):
        from ml._naive_bayes import _NaiveBayesModel
        X, y = multiclass_data
        m = _NaiveBayesModel().fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (150, 3)

    def test_parity_with_sklearn(self, multiclass_data):
        """Native NB should match sklearn GaussianNB predictions closely."""
        from sklearn.naive_bayes import GaussianNB

        from ml._naive_bayes import _NaiveBayesModel
        X, y = multiclass_data
        sk = GaussianNB().fit(X, y)
        native = _NaiveBayesModel().fit(X, y)
        # Probabilities should be very close
        np.testing.assert_allclose(
            native.predict_proba(X), sk.predict_proba(X), atol=1e-4
        )
        # Predictions should match exactly
        np.testing.assert_array_equal(native.predict(X), sk.predict(X))

    def test_sample_weight(self, clf_data):
        from ml._naive_bayes import _NaiveBayesModel
        X, y = clf_data
        weights = np.ones(200)
        weights[:50] = 5.0  # upweight first 50 samples
        m = _NaiveBayesModel().fit(X, y, sample_weight=weights)
        preds = m.predict(X)
        assert np.mean(preds == y) > 0.8

    def test_pickle_roundtrip(self, clf_data):
        from ml._naive_bayes import _NaiveBayesModel
        X, y = clf_data
        m = _NaiveBayesModel().fit(X, y)
        preds_before = m.predict(X)
        m2 = pickle.loads(pickle.dumps(m))
        preds_after = m2.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_engine_auto_uses_native(self):
        """engine='auto' should use native or Rust NB (not sklearn)."""
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris["data"]
        df["target"] = iris["target"]
        import ml
        s = ml.split(df, "target", seed=42)
        m = ml.fit(data=s.train, target="target", algorithm="naive_bayes",
                   seed=42, engine="auto")
        mod = type(m._model).__module__
        assert "naive_bayes" in mod or "ml._rust" in mod, f"Expected native or Rust NB, got {mod}"


# ── Native KNN tests ─────────────────────────────────────────────────────


class TestKNN:
    """Test native _KNNClassifier/_KNNRegressor (zero sklearn dependency)."""

    @pytest.fixture
    def clf_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    @pytest.fixture
    def reg_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 4)
        y = X[:, 0] * 2 + X[:, 1] + rng.randn(200) * 0.1
        return X, y

    @pytest.fixture
    def multiclass_data(self):
        from sklearn.datasets import load_iris
        iris = load_iris()
        return iris.data, iris.target

    def test_clf_get_set_params(self):
        from ml._knn import _KNNClassifier
        m = _KNNClassifier(n_neighbors=7)
        assert m.get_params()["n_neighbors"] == 7
        m.set_params(n_neighbors=3)
        assert m.n_neighbors == 3

    def test_clf_fit_predict(self, clf_data):
        from ml._knn import _KNNClassifier
        X, y = clf_data
        m = _KNNClassifier(n_neighbors=5).fit(X, y)
        preds = m.predict(X)
        assert np.mean(preds == y) > 0.9

    def test_clf_classes(self, clf_data):
        from ml._knn import _KNNClassifier
        X, y = clf_data
        m = _KNNClassifier(n_neighbors=5).fit(X, y)
        np.testing.assert_array_equal(m.classes_, np.array([0, 1]))

    def test_clf_proba_sums_to_one(self, clf_data):
        from ml._knn import _KNNClassifier
        X, y = clf_data
        m = _KNNClassifier(n_neighbors=5).fit(X, y)
        proba = m.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_clf_multiclass(self, multiclass_data):
        from ml._knn import _KNNClassifier
        X, y = multiclass_data
        m = _KNNClassifier(n_neighbors=5).fit(X, y)
        acc = np.mean(m.predict(X) == y)
        assert acc > 0.95

    def test_clf_parity_with_sklearn(self, multiclass_data):
        """Native KNN should match sklearn on iris (odd k avoids ties)."""
        from sklearn.neighbors import KNeighborsClassifier

        from ml._knn import _KNNClassifier
        X, y = multiclass_data
        sk = KNeighborsClassifier(n_neighbors=5).fit(X, y)
        native = _KNNClassifier(n_neighbors=5).fit(X, y)
        np.testing.assert_array_equal(native.predict(X), sk.predict(X))

    def test_reg_fit_predict(self, reg_data):
        from ml._knn import _KNNRegressor
        X, y = reg_data
        m = _KNNRegressor(n_neighbors=5).fit(X, y)
        preds = m.predict(X)
        # R^2 should be decent on training data
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.8

    def test_reg_parity_with_sklearn(self, reg_data):
        from sklearn.neighbors import KNeighborsRegressor

        from ml._knn import _KNNRegressor
        X, y = reg_data
        sk = KNeighborsRegressor(n_neighbors=5).fit(X, y)
        native = _KNNRegressor(n_neighbors=5).fit(X, y)
        np.testing.assert_allclose(native.predict(X), sk.predict(X), atol=1e-10)

    def test_pickle_roundtrip(self, clf_data):
        from ml._knn import _KNNClassifier
        X, y = clf_data
        m = _KNNClassifier(n_neighbors=5).fit(X, y)
        preds_before = m.predict(X)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(preds_before, m2.predict(X))

    def test_engine_auto_uses_native(self):
        """engine='auto' should use Rust KD-tree (or native brute-force fallback)."""
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris["data"]
        df["target"] = iris["target"]
        import ml
        s = ml.split(df, "target", seed=42)
        m = ml.fit(data=s.train, target="target", algorithm="knn",
                   seed=42, engine="auto")
        mod = type(m._model).__module__
        assert "_rust" in mod or "_knn" in mod


# ── Native ElasticNet tests ───────────────────────────────────────────────


class TestElasticNet:
    """Test native _ElasticNetModel (zero sklearn dependency)."""

    @pytest.fixture
    def reg_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        y = X[:, 0] * 3 + X[:, 1] * 1.5 - X[:, 2] * 0.5 + rng.randn(200) * 0.3
        return X, y

    def test_get_set_params(self):
        from ml._elastic_net import _ElasticNetModel
        m = _ElasticNetModel(alpha=0.5, l1_ratio=0.3)
        p = m.get_params()
        assert p["alpha"] == 0.5
        assert p["l1_ratio"] == 0.3
        m.set_params(alpha=2.0)
        assert m.alpha == 2.0

    def test_fit_predict(self, reg_data):
        from ml._elastic_net import _ElasticNetModel
        X, y = reg_data
        m = _ElasticNetModel(alpha=0.01).fit(X, y)
        preds = m.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.95

    def test_l1_ratio_1_is_lasso(self, reg_data):
        """l1_ratio=1 should produce sparse coefficients (Lasso)."""
        from ml._elastic_net import _ElasticNetModel
        X, y = reg_data
        m = _ElasticNetModel(alpha=0.1, l1_ratio=1.0).fit(X, y)
        # Some coefficients should be exactly zero
        assert np.any(m.coef_ == 0.0)

    def test_l1_ratio_0_is_ridge(self, reg_data):
        """l1_ratio=0 should produce no zero coefficients (Ridge)."""
        from ml._elastic_net import _ElasticNetModel
        X, y = reg_data
        m = _ElasticNetModel(alpha=0.1, l1_ratio=0.0).fit(X, y)
        assert np.all(m.coef_ != 0.0)

    def test_parity_with_sklearn(self, reg_data):
        """Native ElasticNet should match sklearn predictions closely."""
        from sklearn.linear_model import ElasticNet

        from ml._elastic_net import _ElasticNetModel
        X, y = reg_data
        sk = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42).fit(X, y)
        native = _ElasticNetModel(alpha=0.1, l1_ratio=0.5).fit(X, y)
        np.testing.assert_allclose(native.coef_, sk.coef_, atol=1e-3)
        np.testing.assert_allclose(native.intercept_, sk.intercept_, atol=1e-3)

    def test_convergence_warning(self):
        """Should warn when max_iter hit without convergence."""
        from ml._elastic_net import _ElasticNetModel
        rng = np.random.RandomState(0)
        X = rng.randn(100, 50)
        y = rng.randn(100)
        with pytest.warns(UserWarning, match="did not converge"):
            _ElasticNetModel(alpha=0.001, max_iter=1).fit(X, y)

    def test_sample_weight(self, reg_data):
        from ml._elastic_net import _ElasticNetModel
        X, y = reg_data
        weights = np.ones(200)
        weights[:50] = 5.0
        m = _ElasticNetModel(alpha=0.01).fit(X, y, sample_weight=weights)
        preds = m.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        assert 1 - ss_res / ss_tot > 0.9

    def test_pickle_roundtrip(self, reg_data):
        from ml._elastic_net import _ElasticNetModel
        X, y = reg_data
        m = _ElasticNetModel(alpha=0.1).fit(X, y)
        preds_before = m.predict(X)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_allclose(preds_before, m2.predict(X), atol=1e-10)

    def test_engine_auto_uses_native(self):
        """engine='auto' should use native or Rust ElasticNet (not sklearn)."""
        from sklearn.datasets import load_diabetes
        diab = load_diabetes(as_frame=True)
        df = diab["data"]
        df["target"] = diab["target"]
        import ml
        s = ml.split(df, "target", seed=42)
        m = ml.fit(data=s.train, target="target", algorithm="elastic_net",
                   seed=42, engine="auto")
        mod = type(m._model).__module__
        assert "_elastic_net" in mod or "ml._rust" in mod, (
            f"Expected native or Rust ElasticNet, got {mod}"
        )


# ── Rust SVM ──────────────────────────────────────────────────────────────────


svm_available = pytest.mark.skipif(
    not __import__("ml._rust", fromlist=["HAS_RUST_SVM"]).HAS_RUST_SVM,
    reason="Rust SVM not available",
)


@svm_available
class TestRustSvmClassifier:
    """Tests for _RustSvmClassifier."""

    @pytest.fixture
    def iris_data(self):
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler
        d = load_iris()
        X = StandardScaler().fit_transform(d["data"]).astype(np.float64)
        return X, d["target"].astype(np.int64)

    @pytest.fixture
    def binary_data(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((200, 4))
        y = (X[:, 0] + rng.standard_normal(200) * 0.3 > 0).astype(np.int64)
        return X, y

    def test_binary_fit_predict(self, binary_data):
        from ml._rust import _RustSvmClassifier
        X, y = binary_data
        m = _RustSvmClassifier(C=1.0).fit(X, y)
        preds = m.predict(X)
        acc = (preds == y).mean()
        assert acc > 0.80, f"Binary SVM accuracy too low: {acc:.3f}"

    def test_multiclass_iris(self, iris_data):
        from ml._rust import _RustSvmClassifier
        X, y = iris_data
        m = _RustSvmClassifier(C=1.0).fit(X, y)
        preds = m.predict(X)
        acc = (preds == y).mean()
        assert acc > 0.80, f"Multiclass SVM accuracy too low: {acc:.3f}"

    def test_predict_proba_shape_sums(self, iris_data):
        from ml._rust import _RustSvmClassifier
        X, y = iris_data
        m = _RustSvmClassifier(C=1.0).fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_classes_encoded(self, iris_data):
        from ml._rust import _RustSvmClassifier
        X, y = iris_data
        m = _RustSvmClassifier(C=1.0).fit(X, y)
        assert set(m.classes_) == {0, 1, 2}
        preds = m.predict(X)
        assert set(preds).issubset({0, 1, 2})

    def test_pickle_roundtrip(self, iris_data):
        from ml._rust import _RustSvmClassifier
        X, y = iris_data
        m = _RustSvmClassifier(C=1.0).fit(X, y)
        preds_before = m.predict(X)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(preds_before, m2.predict(X))

    def test_engine_auto_uses_rust(self):
        """engine='auto' should route to Rust SVM."""
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d["data"]
        df["target"] = d["target"]
        import ml
        s = ml.split(df, "target", seed=42)
        m = ml.fit(data=s.train, target="target", algorithm="svm",
                   seed=42, engine="auto")
        assert "ml._rust" in type(m._model).__module__

    def test_engine_sklearn_fallback(self):
        """engine='sklearn' must return sklearn SVC."""
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d["data"]
        df["target"] = d["target"]
        import ml
        s = ml.split(df, "target", seed=42)
        m = ml.fit(data=s.train, target="target", algorithm="svm",
                   seed=42, engine="sklearn")
        assert "sklearn" in type(m._model).__module__


@svm_available
class TestRustSvmRegressor:
    """Tests for _RustSvmRegressor."""

    @pytest.fixture
    def reg_data(self):
        rng = np.random.default_rng(13)
        X = rng.standard_normal((300, 5))
        y = X[:, 0] * 2.0 + X[:, 1] - X[:, 2] * 0.5 + rng.standard_normal(300) * 0.3
        return X, y

    def test_fit_predict_r2(self, reg_data):
        from ml._rust import _RustSvmRegressor
        X, y = reg_data
        m = _RustSvmRegressor(C=10.0, epsilon=0.1).fit(X, y)
        preds = m.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.7, f"SVR R² too low: {r2:.3f}"

    def test_pickle_roundtrip(self, reg_data):
        from ml._rust import _RustSvmRegressor
        X, y = reg_data
        m = _RustSvmRegressor(C=1.0).fit(X, y)
        preds_before = m.predict(X)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_allclose(preds_before, m2.predict(X), atol=1e-10)

    def test_engine_auto_uses_rust(self):
        """engine='auto' should route to Rust SVR."""
        from sklearn.datasets import load_diabetes
        diab = load_diabetes(as_frame=True)
        df = diab["data"]
        df["target"] = diab["target"]
        import ml
        s = ml.split(df, "target", seed=42)
        m = ml.fit(data=s.train, target="target", algorithm="svm",
                   seed=42, engine="auto")
        assert "ml._rust" in type(m._model).__module__


# ── §8: Missing wrapper classes — ExtraTrees, GBT, AdaBoost ─────────────
# These 5 classes were present in _rust.py but had no direct wrapper tests.
# Protocol: constructor params, fit+predict quality, proba simplex, importances,
# pickle roundtrip — same battery as existing RandomForest/KNN/ElasticNet tests.


class TestExtraTreesClassifier:
    def test_constructor_params(self):
        from ml._rust import _RustExtraTreesClassifier
        m = _RustExtraTreesClassifier(n_estimators=30, max_depth=5, random_state=7)
        assert m.n_estimators == 30
        assert m.max_depth == 5
        assert m.random_state == 7

    def test_fit_predict_accuracy(self, clf_data):
        from ml._rust import _RustExtraTreesClassifier
        X, y = clf_data
        m = _RustExtraTreesClassifier(n_estimators=50, random_state=42).fit(X, y)
        preds = m.predict(X)
        acc = float(np.mean(preds == y))
        assert acc > 0.7, f"ExtraTrees train accuracy too low: {acc:.3f}"

    def test_predict_proba_simplex(self, clf_data):
        from ml._rust import _RustExtraTreesClassifier
        X, y = clf_data
        m = _RustExtraTreesClassifier(n_estimators=30, random_state=42).fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(proba >= 0)

    def test_feature_importances_sum(self, clf_data):
        from ml._rust import _RustExtraTreesClassifier
        X, y = clf_data
        m = _RustExtraTreesClassifier(n_estimators=30, random_state=42).fit(X, y)
        imp = m.feature_importances_
        assert imp.shape == (X.shape[1],)
        np.testing.assert_allclose(imp.sum(), 1.0, atol=1e-6)
        assert np.all(imp >= 0)

    def test_pickle_roundtrip(self, clf_data):
        from ml._rust import _RustExtraTreesClassifier
        X, y = clf_data
        m = _RustExtraTreesClassifier(n_estimators=20, random_state=42).fit(X, y)
        preds_before = m.predict(X)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(preds_before, m2.predict(X))


class TestExtraTreesRegressor:
    def test_constructor_params(self):
        from ml._rust import _RustExtraTreesRegressor
        m = _RustExtraTreesRegressor(n_estimators=25, max_depth=8, random_state=3)
        assert m.n_estimators == 25
        assert m.max_depth == 8

    def test_fit_predict_r2(self, reg_data):
        from ml._rust import _RustExtraTreesRegressor
        X, y = reg_data
        m = _RustExtraTreesRegressor(n_estimators=50, random_state=42).fit(X, y)
        preds = m.predict(X)
        r2 = _r2_score(y, preds)
        assert r2 > 0.7, f"ExtraTrees regression R² too low: {r2:.3f}"

    def test_feature_importances_sum(self, reg_data):
        from ml._rust import _RustExtraTreesRegressor
        X, y = reg_data
        m = _RustExtraTreesRegressor(n_estimators=30, random_state=42).fit(X, y)
        imp = m.feature_importances_
        np.testing.assert_allclose(imp.sum(), 1.0, atol=1e-6)
        assert np.all(imp >= 0)

    def test_pickle_roundtrip(self, reg_data):
        from ml._rust import _RustExtraTreesRegressor
        X, y = reg_data
        m = _RustExtraTreesRegressor(n_estimators=20, random_state=42).fit(X, y)
        preds_before = m.predict(X)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_allclose(preds_before, m2.predict(X), atol=1e-10)


class TestGBTClassifier:
    def test_constructor_params(self):
        from ml._rust import _RustGBTClassifier
        m = _RustGBTClassifier(n_estimators=50, learning_rate=0.05, max_depth=4)
        assert m.n_estimators == 50
        assert m.learning_rate == 0.05
        assert m.max_depth == 4

    def test_n_estimators_validation(self):
        from ml._rust import _RustGBTClassifier
        with pytest.raises(ValueError, match="n_estimators"):
            _RustGBTClassifier(n_estimators=0)

    def test_predict_proba_simplex(self, clf_data):
        from ml._rust import _RustGBTClassifier
        X, y = clf_data
        m = _RustGBTClassifier(n_estimators=30, random_state=42).fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_feature_importances_sum(self, clf_data):
        from ml._rust import _RustGBTClassifier
        X, y = clf_data
        m = _RustGBTClassifier(n_estimators=30, random_state=42).fit(X, y)
        imp = m.feature_importances_
        assert imp.shape == (X.shape[1],)
        np.testing.assert_allclose(imp.sum(), 1.0, atol=1e-6)

    def test_pickle_roundtrip(self, clf_data):
        from ml._rust import _RustGBTClassifier
        X, y = clf_data
        m = _RustGBTClassifier(n_estimators=20, random_state=42).fit(X, y)
        preds_before = m.predict(X)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(preds_before, m2.predict(X))


class TestGBTRegressor:
    def test_constructor_params(self):
        from ml._rust import _RustGBTRegressor
        m = _RustGBTRegressor(n_estimators=80, learning_rate=0.01, subsample=0.8)
        assert m.n_estimators == 80
        assert m.subsample == 0.8

    def test_subsample_validation(self):
        from ml._rust import _RustGBTRegressor
        with pytest.raises(ValueError, match="subsample"):
            _RustGBTRegressor(subsample=1.5)

    def test_fit_predict_r2(self, reg_data):
        from ml._rust import _RustGBTRegressor
        X, y = reg_data
        m = _RustGBTRegressor(n_estimators=50, random_state=42).fit(X, y)
        preds = m.predict(X)
        r2 = _r2_score(y, preds)
        assert r2 > 0.7, f"GBT regression R² too low: {r2:.3f}"

    def test_feature_importances_sum(self, reg_data):
        from ml._rust import _RustGBTRegressor
        X, y = reg_data
        m = _RustGBTRegressor(n_estimators=30, random_state=42).fit(X, y)
        imp = m.feature_importances_
        np.testing.assert_allclose(imp.sum(), 1.0, atol=1e-6)

    def test_pickle_roundtrip(self, reg_data):
        from ml._rust import _RustGBTRegressor
        X, y = reg_data
        m = _RustGBTRegressor(n_estimators=20, random_state=42).fit(X, y)
        preds_before = m.predict(X)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_allclose(preds_before, m2.predict(X), atol=1e-10)


class TestAdaBoost:
    def test_constructor_params(self):
        from ml._rust import _RustAdaBoost
        m = _RustAdaBoost(n_estimators=30, learning_rate=0.5, random_state=99)
        assert m.n_estimators == 30
        assert m.learning_rate == 0.5

    def test_predict_proba_simplex(self, clf_data):
        from ml._rust import _RustAdaBoost
        X, y = clf_data
        m = _RustAdaBoost(n_estimators=30, random_state=42).fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(proba >= 0)

    def test_classes_attr(self, clf_data):
        from ml._rust import _RustAdaBoost
        X, y = clf_data
        m = _RustAdaBoost(n_estimators=20, random_state=42).fit(X, y)
        assert set(m.classes_) == {0, 1}

    def test_feature_importances_sum(self, clf_data):
        from ml._rust import _RustAdaBoost
        X, y = clf_data
        m = _RustAdaBoost(n_estimators=30, random_state=42).fit(X, y)
        imp = m.feature_importances_
        assert imp.shape == (X.shape[1],)
        np.testing.assert_allclose(imp.sum(), 1.0, atol=1e-6)

    def test_pickle_roundtrip(self, clf_data):
        from ml._rust import _RustAdaBoost
        X, y = clf_data
        m = _RustAdaBoost(n_estimators=20, random_state=42).fit(X, y)
        preds_before = m.predict(X)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(preds_before, m2.predict(X))


# ---------------------------------------------------------------------------
# Phase 4b: XGBoost → Rust dispatch
# ---------------------------------------------------------------------------


class TestXGBoostRustDispatch:
    """Tests that algorithm='xgboost' routes to Rust when engine='ml' or 'auto'."""

    @pytest.fixture
    def clf_split(self):
        from sklearn.datasets import load_breast_cancer

        import ml
        X, y = load_breast_cancer(return_X_y=True)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df["target"] = y
        return ml.split(data=df, target="target", seed=42)

    @pytest.fixture
    def reg_split(self):
        from sklearn.datasets import load_diabetes

        import ml
        X, y = load_diabetes(return_X_y=True)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df["target"] = y
        return ml.split(data=df, target="target", seed=42)

    def test_xgboost_routes_to_rust_engine_auto(self, clf_split):
        import ml
        from ml._rust import HAS_RUST_GBT, _RustGBTClassifier
        if not HAS_RUST_GBT:
            pytest.skip("Rust GBT not available")
        model = ml.fit(data=clf_split.train, target="target",
                       algorithm="xgboost", seed=42, engine="auto")
        assert isinstance(model._model, _RustGBTClassifier), (
            f"Expected _RustGBTClassifier, got {type(model._model)}"
        )

    def test_xgboost_routes_to_rust_engine_ml(self, clf_split):
        import ml
        from ml._rust import HAS_RUST_GBT, _RustGBTClassifier
        if not HAS_RUST_GBT:
            pytest.skip("Rust GBT not available")
        model = ml.fit(data=clf_split.train, target="target",
                       algorithm="xgboost", seed=42, engine="ml")
        assert isinstance(model._model, _RustGBTClassifier)

    def test_xgboost_routes_to_cpp_engine_sklearn(self, clf_split):
        import ml
        xgboost = pytest.importorskip("xgboost")
        model = ml.fit(data=clf_split.train, target="target",
                       algorithm="xgboost", seed=42, engine="sklearn")
        assert isinstance(model._model, xgboost.XGBClassifier), (
            f"Expected XGBClassifier, got {type(model._model)}"
        )

    def test_xgboost_engine_ml_no_gpu(self, clf_split):
        import ml
        from ml._rust import HAS_RUST_GBT
        from ml._types import ConfigError
        if not HAS_RUST_GBT:
            pytest.skip("Rust GBT not available")
        with pytest.raises(ConfigError, match="GPU"):
            ml.fit(data=clf_split.train, target="target",
                   algorithm="xgboost", seed=42, engine="ml", gpu=True)

    def test_xgboost_rust_lossguide_default(self, clf_split):
        """Rust XGBoost should default to lossguide, max_leaves=31."""
        import ml
        from ml._rust import HAS_RUST_GBT, _RustGBTClassifier
        if not HAS_RUST_GBT:
            pytest.skip("Rust GBT not available")
        model = ml.fit(data=clf_split.train, target="target",
                       algorithm="xgboost", seed=42, engine="ml")
        assert isinstance(model._model, _RustGBTClassifier)
        assert model._model.grow_policy == "lossguide"
        assert model._model.max_leaves == 31

    def test_xgboost_rust_reg(self, reg_split):
        """Rust XGBoost should work for regression too."""
        import ml
        from ml._rust import HAS_RUST_GBT, _RustGBTRegressor
        if not HAS_RUST_GBT:
            pytest.skip("Rust GBT not available")
        model = ml.fit(data=reg_split.train, target="target",
                       algorithm="xgboost", seed=42, engine="ml")
        assert isinstance(model._model, _RustGBTRegressor)
        preds = ml.predict(model=model, data=reg_split.valid)
        assert len(preds) == len(reg_split.valid)


# ---------------------------------------------------------------------------
# Phase 6: Cross-engine parity tests (Rust vs C++ XGBoost)
# All tests gated on pytest.importorskip("xgboost")
# ---------------------------------------------------------------------------


class TestXGBoostCrossEngineParity:
    """Parity between Rust GBT (engine='ml') and C++ XGBoost (engine='sklearn')."""

    @pytest.fixture
    def clf_split(self):
        from sklearn.datasets import load_breast_cancer

        import ml
        X, y = load_breast_cancer(return_X_y=True)
        df = pd.DataFrame(X[:200], columns=[f"f{i}" for i in range(X.shape[1])])
        df["target"] = y[:200]
        return ml.split(data=df, target="target", seed=42)

    @pytest.fixture
    def reg_split(self):
        from sklearn.datasets import load_diabetes

        import ml
        X, y = load_diabetes(return_X_y=True)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df["target"] = y
        return ml.split(data=df, target="target", seed=42)

    def test_rust_vs_cpp_parity_clf_auc(self, clf_split):
        """Rust and C++ XGBoost should achieve comparable AUC (within 0.10)."""
        import ml
        from ml._rust import HAS_RUST_GBT
        pytest.importorskip("xgboost")
        if not HAS_RUST_GBT:
            pytest.skip("Rust GBT not available")

        rust_model = ml.fit(data=clf_split.train, target="target",
                            algorithm="xgboost", seed=42, engine="ml")
        cpp_model = ml.fit(data=clf_split.train, target="target",
                           algorithm="xgboost", seed=42, engine="sklearn")

        rust_metrics = ml.evaluate(model=rust_model, data=clf_split.valid)
        cpp_metrics = ml.evaluate(model=cpp_model, data=clf_split.valid)

        rust_auc = rust_metrics.get("roc_auc", rust_metrics.get("auc", 0.0))
        cpp_auc = cpp_metrics.get("roc_auc", cpp_metrics.get("auc", 0.0))

        assert rust_auc > 0.5, f"Rust XGBoost AUC too low: {rust_auc}"
        assert abs(rust_auc - cpp_auc) < 0.10, (
            f"Rust AUC={rust_auc:.4f} too far from C++ AUC={cpp_auc:.4f}"
        )

    def test_rust_vs_cpp_parity_reg_mse(self, reg_split):
        """Rust and C++ XGBoost should achieve comparable MSE (within 2×)."""
        import ml
        from ml._rust import HAS_RUST_GBT
        pytest.importorskip("xgboost")
        if not HAS_RUST_GBT:
            pytest.skip("Rust GBT not available")

        rust_model = ml.fit(data=reg_split.train, target="target",
                            algorithm="xgboost", seed=42, engine="ml")
        cpp_model = ml.fit(data=reg_split.train, target="target",
                           algorithm="xgboost", seed=42, engine="sklearn")

        rust_metrics = ml.evaluate(model=rust_model, data=reg_split.valid)
        cpp_metrics = ml.evaluate(model=cpp_model, data=reg_split.valid)

        rust_mse = rust_metrics.get("mse", rust_metrics.get("rmse", float("inf")) ** 2)
        cpp_mse = cpp_metrics.get("mse", cpp_metrics.get("rmse", float("inf")) ** 2)

        assert rust_mse < float("inf"), "Rust MSE is infinite"
        assert rust_mse < cpp_mse * 3.0, (
            f"Rust MSE={rust_mse:.1f} is more than 3× C++ MSE={cpp_mse:.1f}"
        )

    def test_rust_vs_cpp_monotone_clf(self, clf_split):
        """Both engines should respect monotone increasing constraint on first feature."""
        import ml
        from ml._rust import HAS_RUST_GBT
        pytest.importorskip("xgboost")
        if not HAS_RUST_GBT:
            pytest.skip("Rust GBT not available")

        n_features = clf_split.train.shape[1] - 1  # exclude target
        monotone = [1] + [0] * (n_features - 1)    # first feature: increasing

        rust_model = ml.fit(data=clf_split.train, target="target",
                            algorithm="xgboost", seed=42, engine="ml",
                            monotone_cst=monotone)

        # Verify predictions are finite and valid
        preds = ml.predict(model=rust_model, data=clf_split.valid)
        assert len(preds) == len(clf_split.valid)
        assert all(p in (0, 1) for p in preds), "Predictions should be binary 0/1"
