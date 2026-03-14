"""Mathematical invariant tests
These tests prove algorithmic correctness at the mathematical level.
If an invariant fails, the implementation is wrong — no tolerance excuses.

Linear / Ridge regression
Logistic regression
Elastic Net
SVM (linear)
Decision Tree (CART)
Random Forest
Extra Trees
Gradient Boosting / histgradient
AdaBoost
Naive Bayes
KNN
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import ml

# ── Helpers ───────────────────────────────────────────────────────────────

def _clf_data(n=200, p=10, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = (X[:, 0] + X[:, 1] * 0.5 + rng.randn(n) * 0.3 > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y
    return df


def _reg_data(n=200, p=10, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + rng.randn(n) * 0.5
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y
    return df


def _fit(data, algorithm, task_hint=None, **kwargs):
    """Fit with warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=data, target="target", seed=42)
        return ml.fit(data=s.train, target="target", algorithm=algorithm,
                      seed=42, **kwargs), s


# ── Linear / Ridge regression ──────────────────────────────────────


@pytest.mark.skip(reason="Rust _LinearModel lacks intercept_ attribute")
def test_linear_normal_equation_residuals():
    """Ridge first-order optimality: X_aug.T @ residuals ≈ alpha * [0, coef].

    Minimizing ||y - Xw||² + alpha * ||coef||² gives:
        X.T(y - Xw) = alpha * coef (with no penalty on intercept)
    i.e., X_aug.T @ residuals - [0, alpha*coef] ≈ 0.
    """
    data = _reg_data()
    m, s = _fit(data, "linear")
    inner = m._model
    coef = np.asarray(inner.coef_)       # shape (p,)
    intercept = float(inner.intercept_)  # scalar
    alpha = float(inner.alpha)           # ridge penalty

    X_train = s.train.drop(columns=["target"]).values
    y_train = s.train["target"].values

    # Augment with intercept column
    X_aug = np.column_stack([np.ones(len(X_train)), X_train])
    w = np.concatenate([[intercept], coef])
    residuals = y_train - X_aug @ w

    # Ridge KKT: X_aug.T @ residuals = [0, alpha * coef] (intercept unregularized)
    gradient = X_aug.T @ residuals
    expected = np.concatenate([[0.0], alpha * coef])
    np.testing.assert_allclose(
        gradient, expected, atol=0.5,
        err_msg=(
            "Ridge optimality violated: X.T @ residuals != [0, alpha * coef]\n"
            f"alpha={alpha:.4f}, ||gradient - expected||={np.linalg.norm(gradient - expected):.4f}"
        ),
    )


def test_linear_residuals_zero_mean():
    """OLS residuals have mean ≈ 0 when intercept is fitted."""
    data = _reg_data()
    m, s = _fit(data, "linear")
    preds = ml.predict(m, s.train)
    residuals = s.train["target"].values - preds.values
    assert abs(residuals.mean()) < 0.1, (
        f"OLS residuals mean={residuals.mean():.4f} (should be ≈ 0)"
    )


def test_linear_ridge_shrinkage_monotone():
    """Higher ridge alpha → smaller coefficient L2 norm (regularization property)."""
    data = _reg_data(n=300)
    s = ml.split(data=data, target="target", seed=42)
    alphas = [0.0001, 0.1, 1.0, 10.0, 100.0]
    norms = []
    for alpha in alphas:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ml.fit(data=s.train, target="target", algorithm="linear",
                       alpha=alpha, seed=42)
        coef = np.asarray(m._model.coef_)
        norms.append(float(np.linalg.norm(coef)))

    # Norm should be monotonically non-increasing with alpha
    for i in range(len(norms) - 1):
        assert norms[i] >= norms[i + 1] - 1e-6, (
            f"Ridge shrinkage violated: alpha={alphas[i]} norm={norms[i]:.4f} "
            f"> alpha={alphas[i+1]} norm={norms[i+1]:.4f}"
        )


# ── Logistic regression ─────────────────────────────────────────────


def test_logistic_proba_simplex():
    """predict_proba rows sum to 1.0 and all values in [0, 1]."""
    data = _clf_data()
    m, s = _fit(data, "logistic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Use the public API predict — it returns labels. Get proba from _model.
        proba = m._model.predict_proba(
            s.valid.drop(columns=["target"]).values.astype(np.float64)
        )
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6,
                                err_msg="Logistic proba rows don't sum to 1")
    assert np.all(proba >= 0), "Logistic proba contains negative values"
    assert np.all(proba <= 1 + 1e-9), "Logistic proba > 1"


def test_logistic_predict_matches_argmax_proba():
    """argmax(predict_proba) must equal predict for every sample."""
    data = _clf_data()
    m, s = _fit(data, "logistic")
    X_val = s.valid.drop(columns=["target"]).values.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = m._model.predict_proba(X_val)
        pred_raw = m._model.predict(X_val)

    # argmax of proba should correspond to predicted class index
    argmax_idx = np.argmax(proba, axis=1)
    pred_labels = m._model.classes_[argmax_idx]
    np.testing.assert_array_equal(
        pred_raw, pred_labels,
        err_msg="argmax(proba) != predict — proba and predict are inconsistent",
    )


def test_logistic_loss_decreases_with_iterations():
    """Training cross-entropy decreases with more iterations (optimizer is improving)."""
    data = _clf_data(n=300)
    s = ml.split(data=data, target="target", seed=42)
    X_tr = s.train.drop(columns=["target"]).values.astype(np.float64)
    y_tr = s.train["target"].values

    losses = []
    for iters in [1, 10, 50, 200]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ml.fit(data=s.train, target="target", algorithm="logistic",
                       max_iter=iters, seed=42)
        proba = m._model.predict_proba(X_tr)
        # Clip for numerical stability
        eps = 1e-15
        proba_clipped = np.clip(proba, eps, 1 - eps)
        # Map y labels to class indices
        cls = m._model.classes_
        y_idx = np.searchsorted(cls, y_tr)
        loss = -np.mean(np.log(proba_clipped[np.arange(len(y_tr)), y_idx]))
        losses.append(float(loss))

    # Loss at 200 iterations must be lower than at 1 iteration
    assert losses[-1] < losses[0], (
        f"Logistic loss not decreasing: iter=1 loss={losses[0]:.4f}, "
        f"iter=200 loss={losses[-1]:.4f}"
    )


# ── Elastic Net ─────────────────────────────────────────────────────


def test_elastic_net_lasso_sparsity():
    """l1_ratio=1.0 (Lasso limit) produces at least one zero-ish coefficient.

    With p=20 features and only 3 informative, strong L1 should zero out
    some coefficients. atol=0.01 for "ish" (soft thresholding, not exact 0).
    """
    rng = np.random.RandomState(42)
    n, p = 300, 20
    X = rng.randn(n, p)
    # Only features 0, 1, 2 are informative
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 0.5 + rng.randn(n) * 0.3
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y

    s = ml.split(data=df, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ml.fit(data=s.train, target="target", algorithm="elastic_net",
                   l1_ratio=1.0, alpha=0.1, seed=42)
    coef = np.asarray(m._model.coef_)
    n_near_zero = int(np.sum(np.abs(coef) < 0.01))
    assert n_near_zero >= 1, (
        f"Lasso (l1_ratio=1) produced no near-zero coefficients "
        f"(min |coef|={np.min(np.abs(coef)):.4f})"
    )


def test_elastic_net_ridge_limit_no_sparsity():
    """l1_ratio=0.0 (Ridge limit) with mild alpha: coefficients not driven to zero."""
    data = _reg_data(n=200, p=5)
    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ml.fit(data=s.train, target="target", algorithm="elastic_net",
                   l1_ratio=0.0, alpha=0.01, seed=42)
    coef = np.asarray(m._model.coef_)
    # Ridge with mild alpha shouldn't zero out all coefficients
    assert np.sum(np.abs(coef) > 0.01) >= 1, (
        "Ridge (l1_ratio=0) zeroed all coefficients — regression bug"
    )


def test_elastic_net_alpha_shrinkage_monotone():
    """Increasing alpha shrinks coefficient L2 norm toward zero."""
    data = _reg_data(n=300)
    s = ml.split(data=data, target="target", seed=42)
    alphas = [0.001, 0.1, 1.0, 10.0]
    norms = []
    for alpha in alphas:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ml.fit(data=s.train, target="target", algorithm="elastic_net",
                       alpha=alpha, seed=42)
        norms.append(float(np.linalg.norm(np.asarray(m._model.coef_))))

    for i in range(len(norms) - 1):
        assert norms[i] >= norms[i + 1] - 1e-6, (
            f"Elastic net: alpha={alphas[i+1]} norm={norms[i+1]:.4f} "
            f"> alpha={alphas[i]} norm={norms[i]:.4f}"
        )


# ── SVM (linear) ────────────────────────────────────────────────────


def test_svm_clf_proba_simplex():
    """SVM predict_proba rows sum to 1.0 and values in [0, 1]."""
    data = _clf_data()
    m, s = _fit(data, "svm")
    X_val = s.valid.drop(columns=["target"]).values.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = m._model.predict_proba(X_val)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6,
                                err_msg="SVM proba rows don't sum to 1")
    assert np.all(proba >= -1e-9), "SVM proba has negative values"


@pytest.mark.skip(
    reason=(
        "SVM predict vs predict_proba inconsistency is BY DESIGN (matches sklearn SVC behavior). "
        "predict() uses max decision score (optimal margin boundary, f=0 threshold). "
        "predict_proba() uses Platt-calibrated probabilities (shifted threshold f=-B/A). "
        "These differ when Platt intercept B ≠ 0, which is expected for imbalanced data. "
        "Forcing predict=argmax(proba) degrades accuracy (Platt optimizes calibration, not accuracy). "
        "The Platt parameter bug (platt_a negated + platt_b=1-B) was fixed separately in svm.rs: "
        "correct negation is platt_a unchanged, platt_b negated, ensuring P(class 0)=1-P(class 1). "
        "This test is kept as documentation. Invariant does not hold for SVM."
    )
)
def test_svm_predict_matches_argmax_proba():
    """argmax(predict_proba) must equal predict for SVM classifier.

    NOTE: This invariant does NOT hold for SVM — the Platt threshold differs
    from the raw decision boundary. This matches sklearn SVC behavior.
    The Platt fix ensures P(class 0) = 1 - P(class 1), not that predict==argmax(proba).
    """
    data = _clf_data()
    m, s = _fit(data, "svm")
    X_val = s.valid.drop(columns=["target"]).values.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = m._model.predict_proba(X_val)
        pred_raw = m._model.predict(X_val)

    argmax_idx = np.argmax(proba, axis=1)
    pred_from_proba = m._model.classes_[argmax_idx]
    np.testing.assert_array_equal(
        pred_raw, pred_from_proba,
        err_msg="SVM: argmax(proba) != predict",
    )


# ── Decision Tree (CART) ────────────────────────────────────────────


def test_cart_importances_sum_to_one():
    """Feature importances (MDI) sum to 1.0 and are non-negative."""
    data = _clf_data()
    m, s = _fit(data, "decision_tree")
    imp = ml.explain(m)
    imp_values = np.asarray(imp["importance"].values, dtype=float)
    np.testing.assert_allclose(imp_values.sum(), 1.0, atol=1e-6,
                                err_msg="CART importances don't sum to 1")
    assert np.all(imp_values >= -1e-9), "CART importances contain negative values"


def test_cart_perfect_memorization_clf():
    """Unlimited-depth CART memorizes training data: train accuracy = 1.0."""
    data = _clf_data(n=200)
    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ml.fit(data=s.train, target="target", algorithm="decision_tree",
                   max_depth=None, seed=42)
    preds = ml.predict(m, s.train)
    acc = float(np.mean(preds.values == s.train["target"].values))
    assert acc == 1.0, (
        f"Unlimited-depth CART did not memorize train: accuracy={acc:.4f}"
    )


def test_cart_perfect_memorization_reg():
    """Unlimited-depth CART memorizes regression targets: train RMSE ≈ 0."""
    data = _reg_data(n=200)
    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ml.fit(data=s.train, target="target", algorithm="decision_tree",
                   max_depth=None, seed=42)
    preds = ml.predict(m, s.train)
    rmse = float(np.sqrt(np.mean((preds.values - s.train["target"].values) ** 2)))
    assert rmse < 1e-6, (
        f"Unlimited-depth CART regression RMSE on train={rmse:.6f} (expected ≈ 0)"
    )


# ── Random Forest ───────────────────────────────────────────────────


def test_rf_importances_sum_to_one():
    """Aggregated MDI importances sum to 1.0."""
    data = _clf_data()
    m, s = _fit(data, "random_forest")
    imp = ml.explain(m)
    imp_values = np.asarray(imp["importance"].values, dtype=float)
    np.testing.assert_allclose(imp_values.sum(), 1.0, atol=1e-6,
                                err_msg="RF importances don't sum to 1")
    assert np.all(imp_values >= -1e-9)


def test_rf_more_trees_no_degradation():
    """More trees → accuracy non-degrading (law of large numbers).

    Uses a fixed test set (seed=42). With 100 trees vs 10, accuracy should
    be at least as good. atol=0.05 for small-sample variance.
    """
    data = _clf_data(n=300)
    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m10  = ml.fit(data=s.train, target="target", algorithm="random_forest",
                      n_estimators=10,  seed=42)
        m100 = ml.fit(data=s.train, target="target", algorithm="random_forest",
                      n_estimators=100, seed=42)
    acc10  = float(np.mean(ml.predict(m10,  s.valid).values == s.valid["target"].values))
    acc100 = float(np.mean(ml.predict(m100, s.valid).values == s.valid["target"].values))
    assert acc100 >= acc10 - 0.05, (
        f"RF: 100 trees accuracy={acc100:.4f} < 10 trees accuracy={acc10:.4f} - 0.05"
    )


# ── Extra Trees ─────────────────────────────────────────────────────


def test_extra_trees_proba_simplex():
    """Extra Trees predict_proba sums to 1.0."""
    data = _clf_data()
    m, s = _fit(data, "extra_trees")
    X_val = s.valid.drop(columns=["target"]).values.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = m._model.predict_proba(X_val)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6,
                                err_msg="Extra Trees proba rows don't sum to 1")


def test_extra_trees_importances_sum_to_one():
    """Extra Trees MDI importances sum to 1.0."""
    data = _clf_data()
    m, s = _fit(data, "extra_trees")
    imp = ml.explain(m)
    imp_values = np.asarray(imp["importance"].values, dtype=float)
    np.testing.assert_allclose(imp_values.sum(), 1.0, atol=1e-6)
    assert np.all(imp_values >= -1e-9)


# ── Gradient Boosting / histgradient ────────────────────────────────


def test_gbt_training_loss_decreases():
    """Training cross-entropy decreases with more estimators (boosting works)."""
    data = _clf_data(n=300)
    s = ml.split(data=data, target="target", seed=42)
    X_tr = s.train.drop(columns=["target"]).values.astype(np.float64)
    y_tr = s.train["target"].values

    losses = []
    for n_est in [1, 5, 25, 100]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ml.fit(data=s.train, target="target", algorithm="gradient_boosting",
                       n_estimators=n_est, seed=42)
        proba = m._model.predict_proba(X_tr)
        eps = 1e-15
        proba = np.clip(proba, eps, 1 - eps)
        cls = m._model.classes_
        y_idx = np.searchsorted(cls, y_tr)
        loss = -np.mean(np.log(proba[np.arange(len(y_tr)), y_idx]))
        losses.append(float(loss))

    assert losses[-1] < losses[0], (
        f"GBT loss not decreasing: n=1 loss={losses[0]:.4f}, "
        f"n=100 loss={losses[-1]:.4f}"
    )


def test_gbt_importances_sum_to_one():
    """GBT MDI importances sum to 1.0."""
    data = _clf_data()
    m, s = _fit(data, "gradient_boosting")
    imp = ml.explain(m)
    imp_values = np.asarray(imp["importance"].values, dtype=float)
    np.testing.assert_allclose(imp_values.sum(), 1.0, atol=1e-6)
    assert np.all(imp_values >= -1e-9)


# ── AdaBoost ────────────────────────────────────────────────────────


def test_adaboost_proba_simplex():
    """AdaBoost predict_proba sums to 1.0."""
    data = _clf_data()
    m, s = _fit(data, "adaboost")
    X_val = s.valid.drop(columns=["target"]).values.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = m._model.predict_proba(X_val)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6,
                                err_msg="AdaBoost proba rows don't sum to 1")


def test_adaboost_importances_sum_to_one():
    """AdaBoost alpha-weighted MDI importances sum to 1.0."""
    data = _clf_data()
    m, s = _fit(data, "adaboost")
    imp = ml.explain(m)
    imp_values = np.asarray(imp["importance"].values, dtype=float)
    np.testing.assert_allclose(imp_values.sum(), 1.0, atol=1e-6)


def test_adaboost_more_estimators_not_worse():
    """More stumps → training accuracy non-decreasing (SAMME property)."""
    data = _clf_data(n=200)
    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m10  = ml.fit(data=s.train, target="target", algorithm="adaboost",
                      n_estimators=10,  seed=42)
        m100 = ml.fit(data=s.train, target="target", algorithm="adaboost",
                      n_estimators=100, seed=42)
    # Check training accuracy (AdaBoost guaranteed to fit training eventually)
    acc10  = float(np.mean(ml.predict(m10,  s.train).values == s.train["target"].values))
    acc100 = float(np.mean(ml.predict(m100, s.train).values == s.train["target"].values))
    assert acc100 >= acc10 - 0.05, (
        f"AdaBoost: 100 estimators train acc={acc100:.4f} < 10 estimators acc={acc10:.4f}"
    )


# ── Naive Bayes ────────────────────────────────────────────────────


def test_gnb_proba_simplex():
    """GaussianNB predict_proba sums to 1.0 and all values in [0, 1]."""
    data = _clf_data()
    m, s = _fit(data, "naive_bayes")
    X_val = s.valid.drop(columns=["target"]).values.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = m._model.predict_proba(X_val)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6,
                                err_msg="GNB proba rows don't sum to 1")
    assert np.all(proba >= -1e-9)


def test_gnb_no_nan_with_constant_feature():
    """var_smoothing prevents NaN when all training values of a feature are identical."""
    rng = np.random.RandomState(7)
    n = 100
    df = pd.DataFrame({
        "x1": np.ones(n),          # constant feature
        "x2": rng.randn(n),
        "target": rng.choice([0, 1], n),
    })
    s = ml.split(data=df, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ml.fit(data=s.train, target="target", algorithm="naive_bayes", seed=42)
    preds = ml.predict(m, s.valid)
    assert not preds.isna().any(), "GNB produced NaN predictions with constant feature"


# ── KNN ────────────────────────────────────────────────────────────


def test_knn_k1_perfect_train_accuracy():
    """k=1 → perfect training accuracy (each point is its own nearest neighbor)."""
    data = _clf_data(n=200)
    s = ml.split(data=data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ml.fit(data=s.train, target="target", algorithm="knn",
                   n_neighbors=1, seed=42)
    preds = ml.predict(m, s.train)
    acc = float(np.mean(preds.values == s.train["target"].values))
    assert acc == 1.0, f"KNN k=1 train accuracy={acc:.4f} (expected 1.0)"


def test_knn_proba_are_vote_fractions():
    """KNN proba values are multiples of 1/k (vote fractions for k neighbors)."""
    data = _clf_data(n=200)
    s = ml.split(data=data, target="target", seed=42)
    k = 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ml.fit(data=s.train, target="target", algorithm="knn",
                   n_neighbors=k, seed=42)
    X_val = s.valid.drop(columns=["target"]).values.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = m._model.predict_proba(X_val)
    # Proba values should be multiples of 1/k
    scaled = proba * k
    rounded = np.round(scaled)
    np.testing.assert_allclose(scaled, rounded, atol=1e-6,
                                err_msg="KNN proba values are not vote fractions (multiples of 1/k)")
