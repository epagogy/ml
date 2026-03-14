"""Property-based tests
Uses Hypothesis to verify invariants hold across a wide range of inputs.
These catch edge cases that hand-crafted tests miss.

Predict output shape invariance (any valid n, p)
Probability simplex invariance (proba sums to 1 always)
Idempotent refit (same seed + data = same predictions)
Constant target baseline (y constant → regression predicts that constant)
"""

import warnings

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import ml

# ── Hypothesis settings ────────────────────────────────────────────────────────
# deadline: 10000ms per test (ml.fit can be slow for some combos)
# suppress_health_check: allow slow data generation and filter too much
_PBT_SETTINGS = settings(
    max_examples=30,
    deadline=15000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


# ── Predict output shape invariance ────────────────────────────────────

@given(
    n=st.integers(min_value=20, max_value=150),
    p=st.integers(min_value=2, max_value=15),
    seed=st.integers(min_value=0, max_value=999),
)
@_PBT_SETTINGS
def test_predict_shape_invariance(n, p, seed):
    """predict() output has same number of rows as input, for any valid (n, p, seed)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = (X[:, 0] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=df, target="target", seed=seed)
        m = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=seed)
        preds = ml.predict(model=m, data=s.valid)

    assert len(preds) == len(s.valid), (
        f"predict() returned {len(preds)} rows, expected {len(s.valid)} "
        f"(n={n}, p={p}, seed={seed})"
    )


# ── Probability simplex invariance ─────────────────────────────────────

@given(
    n=st.integers(min_value=30, max_value=200),
    seed=st.integers(min_value=0, max_value=999),
)
@_PBT_SETTINGS
def test_probability_simplex_logistic(n, seed):
    """logistic predict_proba rows sum to 1.0 for any valid (n, seed)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = (X[:, 0] + rng.randn(n) * 0.5 > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=df, target="target", seed=seed)
        m = ml.fit(data=s.train, target="target", algorithm="logistic", seed=seed)
        # Evaluate returns probabilities internally; check via predict with proba
        preds = ml.predict(model=m, data=s.valid, proba=True)

    row_sums = preds.sum(axis=1)
    np.testing.assert_allclose(
        row_sums.values, np.ones(len(row_sums)), atol=1e-6,
        err_msg=f"logistic predict_proba rows don't sum to 1 (n={n}, seed={seed})"
    )


# ── Idempotent refit ────────────────────────────────────────────────────

@given(
    n=st.integers(min_value=30, max_value=150),
    seed=st.integers(min_value=0, max_value=999),
)
@_PBT_SETTINGS
def test_idempotent_refit(n, seed):
    """Fitting twice with same data+seed gives identical predictions."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6)
    y = (X[:, 0] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=df, target="target", seed=seed)
        m1 = ml.fit(data=s.train, target="target", algorithm="decision_tree", seed=seed)
        m2 = ml.fit(data=s.train, target="target", algorithm="decision_tree", seed=seed)
        preds1 = ml.predict(model=m1, data=s.valid)
        preds2 = ml.predict(model=m2, data=s.valid)

    np.testing.assert_array_equal(
        preds1.values, preds2.values,
        err_msg=f"decision_tree refit not idempotent (n={n}, seed={seed})"
    )


# ── Constant regression target baseline ────────────────────────────────

@given(
    n=st.integers(min_value=30, max_value=150),
    const=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=999),
)
@_PBT_SETTINGS
def test_constant_target_baseline_reg(n, const, seed):
    """If y is constant, ridge regression prediction should approximate that constant.

    Ridge regression with intercept fits y_hat = const when all y = const.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = np.full(n, const)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["target"] = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s = ml.split(data=df, target="target", seed=seed)
        except ml.DataError:
            # If constant target fails validation — skip (expected behavior)
            return

        m = ml.fit(data=s.train, target="target", algorithm="linear", seed=seed)
        preds = ml.predict(model=m, data=s.valid)

    # Predictions should approximate the constant value
    np.testing.assert_allclose(
        preds.values, np.full(len(preds), const), atol=abs(const) * 0.1 + 1.0,
        err_msg=f"Ridge on constant target {const:.2f}: predictions don't approximate constant"
    )


# ── Evaluate returns floats ────────────────────────────────────────────

@given(
    n=st.integers(min_value=30, max_value=150),
    seed=st.integers(min_value=0, max_value=999),
)
@_PBT_SETTINGS
def test_evaluate_always_returns_floats(n, seed):
    """evaluate() always returns dict[str, float] regardless of data shape."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = rng.randint(0, 2, size=n)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["target"] = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=df, target="target", seed=seed)
        m = ml.fit(data=s.train, target="target", algorithm="decision_tree", seed=seed)
        metrics = ml.evaluate(model=m, data=s.valid)

    assert isinstance(metrics, dict), f"evaluate() returned {type(metrics)}, expected dict"
    for key, val in metrics.items():
        assert isinstance(val, (int, float)), (
            f"evaluate() value for '{key}' is {type(val)}, expected numeric"
        )
        assert val == val, f"evaluate() returned NaN for metric '{key}'"  # NaN check (NaN != NaN)
