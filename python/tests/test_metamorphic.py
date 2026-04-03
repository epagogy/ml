"""Metamorphic relation tests — Chain 18.

Based on Murphy et al. 2008. These tests catch bugs that unit tests structurally
cannot find, by verifying that algorithm outputs change in EXPECTED WAYS when the
input is transformed.

§18.1  MR-1.1  Binary label permutation (swap 0↔1 → predictions swap)
§18.2  MR-1.1b Multiclass accuracy invariance (relabeling preserves accuracy)
§18.3  MR-1.2  Feature permutation (reordering → same predictions)
§18.4  MR-2.1  Uninformative feature (adding noise column shouldn't improve accuracy)
§18.5  MR-2.2  Scale invariance for tree-based algorithms
§18.6  MR-3.1  Data subset consistency (more data ≥ less data ± epsilon)
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import ml
from tests.conftest import ALL_CLF, ALL_REG, ALL_TREE_BASED

# ── Derived lists ──────────────────────────────────────────────────────────────
_ALL_CLF = [a for a in ALL_CLF if a != "xgboost"]
_ALL_REG = [a for a in ALL_REG if a != "xgboost"]
_TREE_BASED = [a for a in ALL_TREE_BASED if a != "xgboost"]

# Feature permutation: only linear/distance-based algos are column-order-agnostic.
# Tree-based (even deterministic CART) and stochastic (RF/ET/GBT) use column
# INDICES during fit (for tie-breaking or random feature selection) — reordering
# columns changes these indices and changes predictions even with the same seed.
_PERMUTATION_INVARIANT_CLF = ["logistic", "naive_bayes", "knn", "svm"]
_PERMUTATION_INVARIANT_REG = ["linear", "elastic_net", "knn", "svm"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clf_data(n=200, p=8, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = (X[:, 0] * 1.5 + X[:, 1] * 0.8 + rng.randn(n) * 0.5 > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y
    return df


def _clf3_data(n=200, p=6, seed=42):
    """3-class classification dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    # 3 linearly-separable blobs (approximately)
    y = np.zeros(n, dtype=int)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 0] < -0.5] = 2
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y
    return df


def _reg_data(n=200, p=8, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = X[:, 0] * 3.0 + X[:, 1] * 1.5 + rng.randn(n) * 0.5
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y
    return df


def _fit_evaluate(train, valid, target, algorithm, seed=42, metric="accuracy"):
    """Fit + evaluate, suppressing warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ml.fit(data=train, target=target, algorithm=algorithm, seed=seed)
        metrics = ml.evaluate(model=m, data=valid)
    return m, metrics.get(metric, float("nan"))


def _fit_predict(train, test_features, target, algorithm, seed=42):
    """Fit + predict, suppressing warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ml.fit(data=train, target=target, algorithm=algorithm, seed=seed)
        preds = ml.predict(model=m, data=test_features)
    return preds


# ── §18.1  MR-1.1  Binary label permutation ───────────────────────────────────

@pytest.mark.parametrize("algorithm", _ALL_CLF)
def test_mr_label_permutation_binary(algorithm):
    """MR-1.1 (BINARY ONLY): Swapping 0↔1 should swap predictions.

    Train on (X, y) → preds1; train on (X, 1-y) → preds2.
    For deterministic algorithms: preds2 == 1 - preds1.
    For probabilistic: accuracy should be symmetric (≈ same absolute accuracy).
    """
    data = _clf_data(n=200, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=data, target="target", seed=42)

    # Original labels
    preds1 = _fit_predict(s.train, s.valid.drop(columns=["target"]), "target", algorithm)

    # Flipped labels
    flipped_train = s.train.copy()
    flipped_train["target"] = 1 - flipped_train["target"]
    preds2 = _fit_predict(flipped_train, s.valid.drop(columns=["target"]), "target", algorithm)

    # MR: preds should swap — fraction that agree should be near 0 or predictions should be inverted
    # Measure: accuracy on original valid vs accuracy on flipped valid (should be symmetric)
    original_valid_labels = s.valid["target"].values
    acc1 = float((preds1.values == original_valid_labels).mean())
    # preds2 should predict the FLIPPED labels, so accuracy on flipped labels
    acc2 = float((preds2.values == (1 - original_valid_labels)).mean())

    # Both accuracies should be similar (label identity doesn't matter, signal does)
    assert abs(acc1 - acc2) < 0.15, (
        f"{algorithm}: MR-1.1 symmetry broken: acc1={acc1:.3f}, acc2={acc2:.3f} "
        f"(diff={abs(acc1 - acc2):.3f} > 0.15)"
    )


# ── §18.2  MR-1.1b Multiclass accuracy invariance ─────────────────────────────

@pytest.mark.parametrize("algorithm", [a for a in _ALL_CLF if a not in ["naive_bayes"]])
def test_mr_multiclass_accuracy_invariance(algorithm):
    """MR-1.1b: Relabeling classes preserves accuracy (metric invariant to label identity).

    Permuting class labels (0→2, 1→0, 2→1) should not change accuracy because
    accuracy measures separation quality, not label assignment.
    """
    data = _clf3_data(n=300, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=data, target="target", seed=42)

    _, acc_original = _fit_evaluate(s.train, s.valid, "target", algorithm)

    # Permute labels: 0→2, 1→0, 2→1
    perm = {0: 2, 1: 0, 2: 1}
    perm_train = s.train.copy()
    perm_train["target"] = perm_train["target"].map(perm)
    perm_valid = s.valid.copy()
    perm_valid["target"] = perm_valid["target"].map(perm)

    _, acc_permuted = _fit_evaluate(perm_train, perm_valid, "target", algorithm)

    assert abs(acc_original - acc_permuted) < 0.05, (
        f"{algorithm}: MR-1.1b accuracy changed after label permutation: "
        f"original={acc_original:.3f}, permuted={acc_permuted:.3f}"
    )


# ── §18.3  MR-1.2  Feature permutation ────────────────────────────────────────

@pytest.mark.parametrize("algorithm", _PERMUTATION_INVARIANT_CLF)
def test_mr_feature_permutation_clf(algorithm):
    """MR-1.2: Reordering features (train+test same shuffle) preserves predictions.

    Only tested for linear/distance-based algorithms. Tree-based and stochastic
    algorithms use column indices in random feature selection or tie-breaking, so
    column reordering changes their internal state even with the same seed.
    """
    data = _clf_data(n=200, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=data, target="target", seed=42)

    feat_cols = [c for c in s.train.columns if c != "target"]
    shuffled_cols = list(reversed(feat_cols))  # simple reversal, both train and test

    # Original predictions
    preds_original = _fit_predict(s.train, s.valid[feat_cols], "target", algorithm)

    # Shuffled predictions (same shuffle on train and test)
    train_shuffled = s.train[shuffled_cols + ["target"]]
    test_shuffled = s.valid[shuffled_cols]
    preds_shuffled = _fit_predict(train_shuffled, test_shuffled, "target", algorithm)

    # Predictions should be identical (column reordering is semantically neutral)
    assert np.array_equal(preds_original.values, preds_shuffled.values), (
        f"{algorithm}: MR-1.2 feature permutation changed predictions"
    )


@pytest.mark.parametrize("algorithm", _PERMUTATION_INVARIANT_REG)
def test_mr_feature_permutation_reg(algorithm):
    """MR-1.2: Reordering features preserves regression predictions.

    Only for linear/distance-based algorithms (see CLF variant for rationale).
    """
    data = _reg_data(n=200, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=data, target="target", seed=42)

    feat_cols = [c for c in s.train.columns if c != "target"]
    shuffled_cols = list(reversed(feat_cols))

    preds_original = _fit_predict(s.train, s.valid[feat_cols], "target", algorithm)
    train_shuffled = s.train[shuffled_cols + ["target"]]
    test_shuffled = s.valid[shuffled_cols]
    preds_shuffled = _fit_predict(train_shuffled, test_shuffled, "target", algorithm)

    np.testing.assert_allclose(
        preds_original.values, preds_shuffled.values, atol=1e-10,
        err_msg=f"{algorithm}: MR-1.2 feature permutation changed regression predictions"
    )


# ── §18.4  MR-2.1  Uninformative feature ──────────────────────────────────────

@pytest.mark.parametrize("algorithm", _TREE_BASED)
def test_mr_uninformative_feature_tree(algorithm):
    """MR-2.1 (tree-based only): Adding a pure noise column shouldn't improve accuracy.

    Tree-based algorithms should prefer informative splits. Adding noise might slightly
    hurt due to increased feature space — it should not significantly help.
    Tolerance epsilon=0.05 (a random split could occasionally improve accuracy slightly).
    """
    data = _clf_data(n=200, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=data, target="target", seed=42)

    _, acc_original = _fit_evaluate(s.train, s.valid, "target", algorithm)

    # Add pure noise column to train AND test
    train_noisy = s.train.copy()
    test_noisy = s.valid.copy()
    rng = np.random.RandomState(99)
    train_noisy["noise_col"] = rng.randn(len(train_noisy))
    test_noisy["noise_col"] = rng.randn(len(test_noisy))

    _, acc_with_noise = _fit_evaluate(train_noisy, test_noisy, "target", algorithm)

    assert acc_with_noise <= acc_original + 0.05, (
        f"{algorithm}: MR-2.1 adding noise column improved accuracy by "
        f"{acc_with_noise - acc_original:.3f} > 0.05 "
        f"(original={acc_original:.3f}, with_noise={acc_with_noise:.3f})"
    )


# ── §18.5  MR-2.2  Scale invariance for tree-based ────────────────────────────

@pytest.mark.parametrize("algorithm", _TREE_BASED)
def test_mr_scale_invariance_tree(algorithm):
    """MR-2.2: Multiplying all features by 1000 preserves tree predictions.

    Trees split on thresholds, so predictions are invariant to monotone feature scaling.
    """
    data = _clf_data(n=200, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=data, target="target", seed=42)

    feat_cols = [c for c in s.train.columns if c != "target"]

    preds_original = _fit_predict(s.train, s.valid[feat_cols], "target", algorithm)

    # Scale all features by 1000
    train_scaled = s.train.copy()
    test_scaled = s.valid[feat_cols].copy()
    train_scaled[feat_cols] = train_scaled[feat_cols] * 1000
    test_scaled[feat_cols] = test_scaled[feat_cols] * 1000

    preds_scaled = _fit_predict(train_scaled, test_scaled, "target", algorithm)

    assert np.array_equal(preds_original.values, preds_scaled.values), (
        f"{algorithm}: MR-2.2 scale invariance broken — predictions changed after ×1000"
    )


# ── §18.6  MR-3.1  Data subset consistency ────────────────────────────────────

def _make_subset_pair_clf(n_small=80, n_large=160, n_test=80, seed=42):
    """Create (train_small, train_large, test) where train_large ⊃ train_small."""
    rng = np.random.RandomState(seed)
    n_total = n_large + n_test
    X = rng.randn(n_total, 6)
    y = (X[:, 0] * 1.5 + X[:, 1] * 0.8 + rng.randn(n_total) * 0.5 > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y

    # Fixed held-out test: last n_test rows (never used for training)
    test = df.iloc[n_large:].reset_index(drop=True)
    large = df.iloc[:n_large].reset_index(drop=True)
    # Small is first n_small rows of large (true subset)
    small = large.iloc[:n_small].reset_index(drop=True)
    return small, large, test


def _make_subset_pair_reg(n_small=80, n_large=160, n_test=80, seed=42):
    """Create (train_small, train_large, test) where train_large ⊃ train_small."""
    rng = np.random.RandomState(seed)
    n_total = n_large + n_test
    X = rng.randn(n_total, 6)
    y = X[:, 0] * 3.0 + X[:, 1] * 1.5 + rng.randn(n_total) * 0.5
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y

    test = df.iloc[n_large:].reset_index(drop=True)
    large = df.iloc[:n_large].reset_index(drop=True)
    small = large.iloc[:n_small].reset_index(drop=True)
    return small, large, test


@pytest.mark.parametrize("algorithm", _ALL_CLF)
def test_mr_subset_consistency_clf(algorithm):
    """MR-3.1: Training on a strict superset should not degrade quality vs subset.

    Uses a fixed held-out test set (never trained on) and a true subset/superset
    pair where train_large contains all rows from train_small plus more.
    Epsilon=0.10: stochastic algorithms and small data can show variability.
    """
    train_small, train_large, test = _make_subset_pair_clf(seed=42)
    test_features = test.drop(columns=["target"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_small = ml.fit(data=train_small, target="target", algorithm=algorithm, seed=42)
        m_large = ml.fit(data=train_large, target="target", algorithm=algorithm, seed=42)
        preds_small = ml.predict(model=m_small, data=test_features)
        preds_large = ml.predict(model=m_large, data=test_features)

    acc_small = float((preds_small.values == test["target"].values).mean())
    acc_large = float((preds_large.values == test["target"].values).mean())

    assert acc_large >= acc_small - 0.10, (
        f"{algorithm}: MR-3.1 more training data degraded accuracy: "
        f"n=80: {acc_small:.3f}, n=160: {acc_large:.3f} (drop > 0.10)"
    )


@pytest.mark.parametrize("algorithm", _ALL_REG)
def test_mr_subset_consistency_reg(algorithm):
    """MR-3.1 (regression): More training data should not degrade R²."""
    train_small, train_large, test = _make_subset_pair_reg(seed=42)
    test_features = test.drop(columns=["target"])
    y_test = test["target"].values

    def _r2(preds):
        ss_res = np.sum((y_test - preds.values) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_small = ml.fit(data=train_small, target="target", algorithm=algorithm, seed=42)
        m_large = ml.fit(data=train_large, target="target", algorithm=algorithm, seed=42)
        r2_small = _r2(ml.predict(model=m_small, data=test_features))
        r2_large = _r2(ml.predict(model=m_large, data=test_features))

    assert r2_large >= r2_small - 0.10, (
        f"{algorithm}: MR-3.1 more training data degraded R²: "
        f"n=80: {r2_small:.3f}, n=160: {r2_large:.3f} (drop > 0.10)"
    )
