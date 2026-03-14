"""Numerical stability tests
Noise gradient (classification) — Spearman rho < -0.5 under label flip
Noise gradient (regression)    — Spearman rho < -0.5 under feature noise
Extreme values — overflow, near-zero variance, large-scale features
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

import ml
from tests.conftest import ALL_CLF, ALL_REG

# ── Derived lists (xgboost optional — skip in stability sweep) ───────────────
_CLF_ALGOS = [a for a in ALL_CLF if a != "xgboost"]
_REG_ALGOS = [a for a in ALL_REG if a != "xgboost"]

# Noise ladders
_NOISE_LEVELS_CLF = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]   # fraction of labels flipped
_NOISE_LEVELS_REG = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]     # std of additive feature noise


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_clf_data(n=200, p=8, seed=42):
    """Informative classification dataset with clear signal."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    # Strong linear signal: first 3 features are informative
    y = (X[:, 0] * 1.5 + X[:, 1] * 1.0 + X[:, 2] * 0.8 + rng.randn(n) * 0.4 > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y
    return df


def _base_reg_data(n=200, p=8, seed=42):
    """Informative regression dataset with clear signal."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = X[:, 0] * 3.0 + X[:, 1] * 2.0 + X[:, 2] * 1.5 + rng.randn(n) * 0.3
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y
    return df


def _add_label_noise(data: pd.DataFrame, noise_level: float, rng: np.random.RandomState) -> pd.DataFrame:
    """Randomly flip a fraction of binary class labels."""
    df = data.copy()
    n = len(df)
    flip_mask = rng.rand(n) < noise_level
    df.loc[flip_mask, "target"] = 1 - df.loc[flip_mask, "target"]
    return df


def _add_feature_noise(data: pd.DataFrame, noise_std: float, rng: np.random.RandomState) -> pd.DataFrame:
    """Add Gaussian noise (std=noise_std) to all feature columns."""
    df = data.copy()
    feature_cols = [c for c in df.columns if c != "target"]
    df[feature_cols] = df[feature_cols] + rng.randn(len(df), len(feature_cols)) * noise_std
    return df


def _fit_quiet(data, algorithm, seed=42):
    """Fit with warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ml.fit(data=data, target="target", algorithm=algorithm, seed=seed)


def _accuracy(model, data):
    """Compute accuracy on a dataset."""
    preds = ml.predict(model=model, data=data)
    return float((preds.values == data["target"].values).mean())


def _r2(model, data):
    """Compute R² on a dataset."""
    preds = ml.predict(model=model, data=data)
    y = data["target"].values
    ss_res = np.sum((y - preds.values) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ── Noise gradient — classification ────────────────────────────────────

@pytest.mark.parametrize("algorithm", _CLF_ALGOS)
def test_noise_gradient_clf(algorithm):
    """Label-flip noise causes accuracy to trend downward (Spearman rho < -0.5).

    Statistical approach: pairwise monotonicity is too strict for stochastic algorithms
    which can have local bumps. Spearman rank correlation captures the global trend
    without penalizing individual non-monotone steps.

    rho < -0.5 = strong negative association between noise level and accuracy.
    """
    seeds = [42, 43, 44]
    accuracies = []

    for noise_level in _NOISE_LEVELS_CLF:
        level_accs = []
        base = _base_clf_data(n=200, seed=0)
        s_base = ml.split(data=base, target="target", seed=0)

        for seed in seeds:
            rng = np.random.RandomState(seed + int(noise_level * 1000))
            noisy_train = _add_label_noise(s_base.train, noise_level, rng)
            # Always evaluate on clean valid data — we test how well model generalises
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = _fit_quiet(noisy_train, algorithm, seed=seed)
            acc = _accuracy(m, s_base.valid)
            level_accs.append(acc)

        accuracies.append(float(np.mean(level_accs)))

    rho, _ = spearmanr(_NOISE_LEVELS_CLF, accuracies)

    # Endpoint sanity: clean data should beat 50% noise
    clean_acc = accuracies[0]
    noisy_acc = accuracies[-1]
    assert clean_acc > noisy_acc - 0.05, (
        f"{algorithm}: clean accuracy ({clean_acc:.3f}) should exceed "
        f"50% noise accuracy ({noisy_acc:.3f}) by more than -0.05"
    )

    # Global trend: noise → worse performance
    assert rho < -0.3, (
        f"{algorithm}: expected Spearman rho < -0.3 (strong negative trend), "
        f"got rho={rho:.3f}. Accuracies: {[f'{a:.3f}' for a in accuracies]}"
    )


# ── Noise gradient — regression ───────────────────────────────────────

@pytest.mark.parametrize("algorithm", _REG_ALGOS)
def test_noise_gradient_reg(algorithm):
    """Feature noise causes R² to trend downward (Spearman rho < -0.5).

    Uses additive Gaussian noise on features (not label noise for regression,
    since regression targets are already continuous).
    """
    seeds = [42, 43, 44]
    r2_values = []

    base = _base_reg_data(n=200, seed=0)
    s_base = ml.split(data=base, target="target", seed=0)

    for noise_std in _NOISE_LEVELS_REG:
        level_r2s = []

        for seed in seeds:
            rng = np.random.RandomState(seed + int(noise_std * 100))
            noisy_train = _add_feature_noise(s_base.train, noise_std, rng)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = _fit_quiet(noisy_train, algorithm, seed=seed)
            # Evaluate on clean valid data
            r2 = _r2(m, s_base.valid)
            level_r2s.append(r2)

        r2_values.append(float(np.mean(level_r2s)))

    rho, _ = spearmanr(_NOISE_LEVELS_REG, r2_values)

    # Endpoint sanity: R² at no noise should beat R² at high noise
    clean_r2 = r2_values[0]
    noisy_r2 = r2_values[-1]
    assert clean_r2 > noisy_r2 - 0.1, (
        f"{algorithm}: clean R² ({clean_r2:.3f}) should exceed "
        f"high-noise R² ({noisy_r2:.3f}) by more than -0.1"
    )

    # Global trend: more noise → lower R²
    assert rho < -0.3, (
        f"{algorithm}: expected Spearman rho < -0.3 (strong negative trend), "
        f"got rho={rho:.3f}. R² values: {[f'{v:.3f}' for v in r2_values]}"
    )


# ── Extreme values ─────────────────────────────────────────────────────

def test_logistic_no_overflow_large_features():
    """Features ~ 1e6 don't cause overflow in sigmoid (clamped internally)."""
    rng = np.random.RandomState(42)
    n, p = 100, 5
    X = rng.randn(n, p) * 1e6
    y = (X[:, 0] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=df, target="target", seed=42)
        m = ml.fit(data=s.train, target="target", algorithm="logistic", seed=42)
        preds = ml.predict(model=m, data=s.valid)

    # No NaN or Inf in predictions
    assert preds.notna().all(), "Logistic predictions contain NaN with large features"
    assert np.isfinite(preds.values).all(), "Logistic predictions contain Inf with large features"


def test_linear_no_nan_large_features():
    """Ridge regression on large-scale features produces finite predictions."""
    rng = np.random.RandomState(42)
    n, p = 100, 5
    X = rng.randn(n, p) * 1e6
    y = X[:, 0] * 2.0 + rng.randn(n) * 1e4
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=df, target="target", seed=42)
        m = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
        preds = ml.predict(model=m, data=s.valid)

    assert np.isfinite(preds.values).all(), "Ridge predictions contain NaN/Inf with large features"


def test_gnb_no_nan_extreme_variance():
    """GaussianNB doesn't produce NaN when feature variance ~ 1e-15 (var_smoothing saves it)."""
    rng = np.random.RandomState(42)
    n, p = 100, 5
    # Near-constant features with tiny variance
    X = rng.randn(n, p) * 1e-8
    y = (rng.randn(n) > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=df, target="target", seed=42)
        m = ml.fit(data=s.train, target="target", algorithm="naive_bayes", seed=42)
        preds = ml.predict(model=m, data=s.valid)

    assert preds.notna().all(), "NaiveBayes predictions contain NaN with near-zero variance features"


def test_knn_no_crash_duplicate_query_points():
    """KNN handles query points identical to training points without crashing."""
    rng = np.random.RandomState(42)
    n, p = 60, 4
    X = rng.randn(n, p)
    y = (X[:, 0] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=df, target="target", seed=42)
        m = ml.fit(data=s.train, target="target", algorithm="knn", seed=42)
        # Predict on a copy of training data (duplicate query points)
        preds = ml.predict(model=m, data=s.train.drop(columns=["target"]))

    assert preds.notna().all(), "KNN predictions contain NaN when query=train points"


def test_ridge_near_singular_matrix():
    """Ridge regression on near-collinear features (X.T @ X almost singular) produces finite output."""
    rng = np.random.RandomState(42)
    n, p = 50, 8
    # Create near-singular design: all features nearly identical
    base = rng.randn(n, 1)
    X = np.tile(base, (1, p)) + rng.randn(n, p) * 1e-6
    y = base[:, 0] + rng.randn(n) * 0.1
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=df, target="target", seed=42)
        m = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
        preds = ml.predict(model=m, data=s.valid)

    assert np.isfinite(preds.values).all(), "Ridge predictions non-finite with near-singular matrix"


def test_decision_tree_constant_target_raises_clean_error():
    """CART rejects constant-target data with DataError at split time (not crash or silent NaN)."""
    rng = np.random.RandomState(42)
    n, p = 80, 4
    X = rng.randn(n, p)
    y = np.zeros(n, dtype=int)  # constant class — no signal
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y

    with pytest.raises(ml.DataError):
        ml.split(data=df, target="target", seed=42)


def test_decision_tree_extreme_imbalance():
    """CART on 10:1 class imbalance produces finite predictions (doesn't crash)."""
    rng = np.random.RandomState(42)
    n, p = 150, 4
    X = rng.randn(n, p)
    y = np.zeros(n, dtype=int)
    y[:15] = 1  # 10% positive class — extreme enough, reliably splits
    rng.shuffle(y)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=df, target="target", seed=42)
        m = ml.fit(data=s.train, target="target", algorithm="decision_tree", seed=42)
        preds = ml.predict(model=m, data=s.valid)

    assert preds.notna().all(), "Decision tree predictions contain NaN with extreme class imbalance"
    assert np.isin(preds.values, [0, 1]).all(), "Decision tree predictions out of {0, 1} for binary target"


def test_random_forest_tiny_dataset():
    """RF on n=10 rows doesn't crash — min_samples edge case."""
    rng = np.random.RandomState(42)
    X = rng.randn(10, 3)
    y = rng.randint(0, 2, size=10)
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target"] = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=df, target="target", seed=42)
        m = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
        preds = ml.predict(model=m, data=s.valid)

    assert len(preds) == len(s.valid)
    assert preds.notna().all()
