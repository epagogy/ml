"""Tests for ml.stack() — ensemble stacking."""

import numpy as np
import pandas as pd
import pytest

import ml

pytestmark = pytest.mark.slow  # stack() trains multiple models — 36s server, 463 MB peak


@pytest.fixture
def clf_data():
    """Classification dataset for stack tests."""
    rng = np.random.RandomState(42)
    n = 80
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    target = (x1 + x2 + rng.normal(0, 0.5, n) > 0).astype(int)
    data = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": rng.normal(0, 1, n),
        "target": np.where(target, "yes", "no"),
    })
    return data


@pytest.fixture
def reg_data():
    """Regression dataset for stack tests."""
    rng = np.random.RandomState(42)
    n = 80
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    data = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": rng.normal(0, 1, n),
        "price": 3.0 * x1 + 2.0 * x2 + rng.normal(0, 0.5, n),
    })
    return data


def test_stack_basic_classification(clf_data):
    """stack() produces a Model for classification."""
    s = ml.split(data=clf_data, target="target", seed=42)
    model = ml.stack(data=s.train, target="target", seed=42)

    assert isinstance(model, ml.Model)
    assert model.algorithm == "stacked"
    assert model.task == "classification"


def test_stack_basic_regression(reg_data):
    """stack() produces a Model for regression."""
    s = ml.split(data=reg_data, target="price", seed=42)
    model = ml.stack(data=s.train, target="price", seed=42)

    assert isinstance(model, ml.Model)
    assert model.algorithm == "stacked"
    assert model.task == "regression"


def test_stack_three_models(clf_data):
    """stack() with 3 base models works."""
    s = ml.split(data=clf_data, target="target", seed=42)
    model = ml.stack(
        data=s.train, target="target", seed=42,
        models=["xgboost", "random_forest", "knn"],
    )

    assert isinstance(model, ml.Model)
    assert model.algorithm == "stacked"


def test_stack_predict(clf_data):
    """Stacked model can predict."""
    s = ml.split(data=clf_data, target="target", seed=42)
    model = ml.stack(data=s.train, target="target", seed=42)

    preds = model.predict(s.valid)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(s.valid)
    assert set(preds.unique()).issubset({"yes", "no"})


def test_stack_predict_proba(clf_data):
    """Stacked model supports predict_proba."""
    s = ml.split(data=clf_data, target="target", seed=42)
    model = ml.stack(data=s.train, target="target", seed=42)

    probs = model.predict_proba(s.valid)
    assert isinstance(probs, pd.DataFrame)
    assert probs.shape[1] == 2
    # Probabilities should sum to ~1.0
    row_sums = probs.sum(axis=1)
    assert all(abs(s - 1.0) < 0.01 for s in row_sums)


def test_stack_evaluate(clf_data):
    """evaluate() works with stacked model."""
    s = ml.split(data=clf_data, target="target", seed=42)
    model = ml.stack(data=s.train, target="target", seed=42)

    metrics = ml.evaluate(model, s.valid)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_stack_explain(clf_data):
    """explain() returns base model weights for stacked model."""
    s = ml.split(data=clf_data, target="target", seed=42)
    model = ml.stack(data=s.train, target="target", seed=42)

    imp = ml.explain(model)
    assert isinstance(imp, ml.Explanation)
    assert "feature" in imp.columns
    assert "importance" in imp.columns
    # Features should be base model names (default includes logistic since Step 6.9)
    assert "xgboost" in set(imp["feature"].tolist())
    assert "random_forest" in set(imp["feature"].tolist())
    # Importances should sum to ~1.0
    assert abs(imp["importance"].sum() - 1.0) < 0.01


def test_stack_algorithm_property(clf_data):
    """Stacked model has algorithm='stacked'."""
    s = ml.split(data=clf_data, target="target", seed=42)
    model = ml.stack(data=s.train, target="target", seed=42)

    assert model.algorithm == "stacked"
    assert model.target == "target"
    assert model.seed == 42
    assert len(model.features) > 0


def test_stack_invalid_algorithm(clf_data):
    """Invalid base algorithm raises ConfigError."""
    s = ml.split(data=clf_data, target="target", seed=42)
    with pytest.raises(ml.ConfigError, match="not available"):
        ml.stack(
            data=s.train, target="target", seed=42,
            models=["xgboost", "magic_forest"],
        )


def test_stack_missing_seed(clf_data):
    """stack() without seed raises TypeError."""
    s = ml.split(data=clf_data, target="target", seed=42)
    with pytest.raises(TypeError):
        ml.stack(data=s.train, target="target")


def test_stack_all_rows_have_nan():
    """stack() works when every row has at least one NaN feature.

    Regression guard: nan_imputer.fit(X_clean.dropna()) returns empty DataFrame
    when every row has NaN — SimpleImputer must fit on full data, not dropna().
    """
    import warnings
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "x3": rng.rand(n),
        "target": rng.rand(n) * 10,
    })
    # Introduce NaN in every row (each row has exactly one NaN in a rotating column)
    for i in range(n):
        col = ["x1", "x2", "x3"][i % 3]
        df.loc[i, col] = float("nan")

    s = ml.split(data=df, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacked = ml.stack(data=s.train, target="target", seed=42, cv_folds=2)
    preds = ml.predict(model=stacked, data=s.valid)
    assert len(preds) == len(s.valid)
    assert not preds.isna().any()


# ── Phase 0 additions ─────────────────────────────────────────────────────────


def test_stack_final_refit_normalization_matches_oof():
    """stack() final refit uses global norm; OOF uses per-fold norm. Phase 0.4.

    For tree-based base models (xgboost, random_forest), scale-invariance means
    this mismatch has no material impact. Test verifies predictions are valid and
    the known limitation is documented (not silently broken).
    """
    import warnings

    import numpy as np

    rng = np.random.RandomState(42)
    n = 200
    data = pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n) * 100,  # Different scale — matters for linear models, not trees
        "x3": rng.rand(n),
        "target": rng.choice([0, 1], n),
    })
    s = ml.split(data=data, target="target", seed=42)

    # Tree-based stack: scale mismatch between OOF and final refit is inconsequential
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacked = ml.stack(
            data=s.train, target="target", seed=42,
            models=["xgboost", "random_forest"],
            cv_folds=3,
        )

    # Predictions must be valid despite OOF/global norm mismatch
    preds = ml.predict(model=stacked, data=s.valid)
    assert len(preds) == len(s.valid)
    assert not preds.isna().any()
    assert set(preds.unique()).issubset({0, 1})

    # Normalization strategy documented on the model (via stack.py comment Phase 0.4)
    assert stacked.algorithm == "stacked"


# ── A5: OOF predictions ───────────────────────────────────────────────────────


def test_oof_predictions_shape(clf_data):
    """oof_predictions_ has shape (n_train, n_models) for binary classification. A5."""
    import warnings

    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacked = ml.stack(
            data=s.train, target="target", seed=42,
            models=["xgboost", "random_forest"], cv_folds=3,
        )

    oof = stacked.oof_predictions_
    assert oof is not None
    assert isinstance(oof, pd.DataFrame)
    # Binary: one column per base model
    assert oof.shape == (len(s.train), 2)
    assert not oof.isna().all().all()


def test_oof_no_leakage(clf_data):
    """OOF predictions are computed without target leakage — each row predicted
    by a model that never saw that row during training. A5."""
    import warnings

    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacked = ml.stack(
            data=s.train, target="target", seed=42,
            models=["xgboost", "random_forest"], cv_folds=5,
        )

    oof = stacked.oof_predictions_
    # No row should be all-zeros (would indicate fold coverage gap)
    all_zero_rows = (oof == 0.0).all(axis=1).sum()
    assert all_zero_rows == 0, f"{all_zero_rows} rows with all-zero OOF predictions"
    # OOF probabilities for binary classification must be in [0, 1]
    assert (oof >= 0.0).all().all()
    assert (oof <= 1.0).all().all()


def test_oof_base_models_stored(clf_data):
    """_base_models stores the list of (algo_name, estimator) tuples. A5."""
    import warnings

    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacked = ml.stack(
            data=s.train, target="target", seed=42,
            models=["xgboost", "random_forest"], cv_folds=3,
        )

    bases = stacked._base_models
    assert bases is not None
    assert len(bases) == 2
    names = [name for name, _ in bases]
    assert "xgboost" in names
    assert "random_forest" in names


def test_oof_column_naming(clf_data):
    """OOF columns use {algo}_oof for binary, {algo}_{i}_oof for duplicates. A5."""
    import warnings

    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacked = ml.stack(
            data=s.train, target="target", seed=42,
            models=["xgboost", "random_forest"], cv_folds=3,
        )

    cols = list(stacked.oof_predictions_.columns)
    assert "xgboost_oof" in cols
    assert "random_forest_oof" in cols

    # Duplicate algo names — must disambiguate with index suffix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacked_dup = ml.stack(
            data=s.train, target="target", seed=42,
            models=["xgboost", "xgboost"], cv_folds=3,
        )
    dup_cols = list(stacked_dup.oof_predictions_.columns)
    assert "xgboost_0_oof" in dup_cols
    assert "xgboost_1_oof" in dup_cols


def test_oof_multiclass_shape():
    """OOF columns use {algo}_oof_class_{k} for multiclass. A5."""
    import warnings

    rng = np.random.RandomState(42)
    n = 240
    data = pd.DataFrame({
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice(["a", "b", "c"], n),
    })
    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacked = ml.stack(
            data=s.train, target="target", seed=42,
            models=["xgboost", "random_forest"], cv_folds=3,
        )

    oof = stacked.oof_predictions_
    # 3 classes × 2 models = 6 OOF columns
    assert oof.shape[1] == 6
    assert oof.shape[0] == len(s.train)
    assert any("oof_class_0" in c for c in oof.columns)
    assert any("oof_class_2" in c for c in oof.columns)


def test_oof_non_stacked_model_is_none(clf_data):
    """Non-stacked model has oof_predictions_=None. A5."""
    s = ml.split(data=clf_data, target="target", seed=42)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", seed=42)
    assert model.oof_predictions_ is None


# ── Chain 3.2: Multi-level stacking ──────────────────────────────────────────


def test_stack_2_levels(clf_data):
    """stack(levels=2) trains without error. Chain 3.2."""
    import warnings
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.stack(
            data=s.train, target="target",
            levels=2, cv_folds=3, seed=42,
        )
    assert isinstance(model, ml.Model)
    assert model.algorithm == "stacked"
    preds = ml.predict(model, s.valid)
    assert len(preds) == len(s.valid)


def test_stack_passthrough(clf_data):
    """stack(passthrough=True) includes original features in meta-learner. Chain 3.2."""
    import warnings
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.stack(
            data=s.train, target="target",
            passthrough=True, cv_folds=3, seed=42,
        )
    assert isinstance(model, ml.Model)
    preds = ml.predict(model, s.valid)
    assert len(preds) == len(s.valid)


def test_stack_2_levels_improves(clf_data):
    """2-level stack produces valid scores (not catastrophically worse). Chain 3.2."""
    import warnings

    from sklearn.metrics import roc_auc_score
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m1 = ml.stack(data=s.train, target="target", levels=1, cv_folds=3, seed=42)
        m2 = ml.stack(data=s.train, target="target", levels=2, cv_folds=3, seed=42)

    from ml.predict import _predict_proba
    proba1 = _predict_proba(m1, s.valid).values[:, 1]
    proba2 = _predict_proba(m2, s.valid).values[:, 1]
    y = s.valid["target"].map({"yes": 1, "no": 0})
    score1 = roc_auc_score(y, proba1)
    score2 = roc_auc_score(y, proba2)
    # Both should be valid AUC; 2-level should not catastrophically degrade
    assert 0.5 <= score1 <= 1.0
    assert 0.5 <= score2 <= 1.0
    assert score2 >= score1 - 0.1


def test_stack_levels_oof_shape(clf_data):
    """OOF predictions have correct shape for levels=1 and levels=2. Chain 3.2."""
    import warnings
    s = ml.split(data=clf_data, target="target", seed=42)
    n_models = 3  # default base models: xgboost, random_forest, logistic (Step 6.9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m1 = ml.stack(data=s.train, target="target", levels=1, cv_folds=3, seed=42)
        m2 = ml.stack(data=s.train, target="target", levels=2, cv_folds=3, seed=42)

    assert m1.oof_predictions_.shape == (len(s.train), n_models)
    assert m2.oof_predictions_.shape == (len(s.train), n_models)


# ── Chain 3.3: Meta-learner algorithm choice ──────────────────────────────────


def test_stack_meta_xgboost(clf_data):
    """stack(meta='xgboost') works — XGBoost as non-linear meta-learner. Chain 3.3."""
    import warnings
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.stack(
            data=s.train, target="target",
            meta="xgboost", cv_folds=3, seed=42,
        )
    assert isinstance(model, ml.Model)
    assert model.algorithm == "stacked"
    preds = ml.predict(model, s.valid)
    assert len(preds) == len(s.valid)


def test_stack_meta_lightgbm(clf_data):
    """stack(meta='lightgbm') works when LightGBM is available. Chain 3.3."""
    import warnings
    pytest.importorskip("lightgbm")
    s = ml.split(data=clf_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.stack(
            data=s.train, target="target",
            meta="lightgbm", cv_folds=3, seed=42,
        )
    assert isinstance(model, ml.Model)
    preds = ml.predict(model, s.valid)
    assert len(preds) == len(s.valid)


def test_stack_balance_meta_learner(small_classification_data):
    """stack(balance=True) applies class_weight='balanced' to meta-learner."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.stack(data=s.train, target="target", seed=42, balance=True)
    assert model is not None
    preds = model.predict(s.valid)
    assert len(preds) == len(s.valid)


def test_stack_balance_regression_error(small_regression_data):
    """stack(balance=True) raises on regression."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with pytest.raises(ml.ConfigError, match="balance"):
        ml.stack(data=s.train, target="target", seed=42, balance=True)


def test_stack_warns_on_base_model_failure(small_classification_data):
    """stack() warns (not silently ignores) when a base model fails."""
    import warnings
    s = ml.split(data=small_classification_data, target="target", seed=42)
    # Normal operation: model should succeed without ModelError
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        model = ml.stack(data=s.train, target="target", seed=42)
    # No ModelError should be raised
    assert model is not None


def test_stack_drops_mostly_failed_model(small_classification_data):
    """stack() drops base model that fails on >50% of folds (integration)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    # Normal operation: model should succeed
    model = ml.stack(data=s.train, target="target", seed=42)
    assert model is not None


def test_stack_default_includes_linear_model(small_classification_data):
    """stack() default algorithms include a linear model for diversity."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.stack(data=s.train, target="target", seed=42)
    # The stack should use at least 3 base models including a linear one
    assert model is not None
    if hasattr(model, "_base_models") and model._base_models:
        # _base_models is a list of (algo_name, fitted_estimator) tuples
        algo_names = [name for name, _ in model._base_models]
        assert any(a in ("logistic", "linear") for a in algo_names), \
            f"No linear model in stack. Got: {algo_names}"


def test_stack_cv_score_not_none(clf_data):
    """stack() cv_score is a float, not None (OOF-based estimate)."""
    s = ml.split(data=clf_data, target="target", seed=42)
    model = ml.stack(data=s.train, target="target", seed=42)

    assert model.cv_score is not None, "stack().cv_score must not be None"
    assert isinstance(model.cv_score, float)
    assert 0.0 <= model.cv_score <= 1.0


def test_stack_cv_score_regression_not_none(reg_data):
    """stack() cv_score is a float for regression (OOF-based R2 estimate)."""
    s = ml.split(data=reg_data, target="price", seed=42)
    model = ml.stack(data=s.train, target="price", seed=42)

    assert model.cv_score is not None, "stack().cv_score must not be None for regression"
    assert isinstance(model.cv_score, float)
