"""Speed regression bounds
Absolute time limits per algorithm on 1K rows. Detects CATASTROPHIC regressions
(10× slowdown), not precision optimization. Passes on any reasonably fast machine.

Bounds calibrated on server (2026-03-04, n=1000 rows, 10 features):
    random_forest clf  0.006s → bound 0.20s
    random_forest reg  0.010s → bound 0.25s
    decision_tree clf  0.003s → bound 0.10s
    decision_tree reg  0.003s → bound 0.10s
    logistic      clf  0.003s → bound 0.10s
    linear        reg  0.002s → bound 0.10s
    knn           clf  0.003s → bound 0.50s
    knn           reg  0.002s → bound 0.50s
    naive_bayes   clf  0.002s → bound 0.10s
    elastic_net   reg  0.002s → bound 0.10s
    svm           clf  0.007s → bound 0.20s
    svm           reg  0.002s → bound 0.10s
    histgradient  clf  0.058s → bound 0.60s
    histgradient  reg  0.062s → bound 0.70s
    gradient_boosting clf 0.058s → bound 0.60s
    gradient_boosting reg 0.065s → bound 0.70s
    extra_trees   clf  0.004s → bound 0.20s
    extra_trees   reg  0.008s → bound 0.20s
    adaboost      clf  0.011s → bound 0.20s

Environment-adaptive: ML_SPEED_FACTOR (default 1.0).
server: 1.0 | macbook: 2.0 | slow CI: 3.0
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import pytest

import ml

# ── Speed factor for environment adaptation ───────────────────────────────────
SPEED_FACTOR = float(os.environ.get("ML_SPEED_FACTOR", "1.0"))

# ── Bounds (10× observed on server, 2026-03-04) ────────────────────────────────
# (algorithm, task, bound_seconds)
SPEED_BOUNDS = [
    ("random_forest",     "clf", 0.20),
    ("random_forest",     "reg", 0.25),
    ("decision_tree",     "clf", 0.10),
    ("decision_tree",     "reg", 0.10),
    ("logistic",          "clf", 0.10),
    ("linear",            "reg", 0.10),
    ("knn",               "clf", 0.50),
    ("knn",               "reg", 0.50),
    ("naive_bayes",       "clf", 0.10),
    ("elastic_net",       "reg", 0.10),
    ("svm",               "clf", 0.20),
    ("svm",               "reg", 0.10),
    ("histgradient",      "clf", 0.60),
    ("histgradient",      "reg", 0.70),
    ("gradient_boosting", "clf", 0.60),
    ("gradient_boosting", "reg", 0.70),
    ("extra_trees",       "clf", 0.20),
    ("extra_trees",       "reg", 0.20),
    ("adaboost",          "clf", 0.20),
]

_BOUND_IDS = [f"{algo}/{task}" for algo, task, _ in SPEED_BOUNDS]


# ── 1K dataset fixtures ───────────────────────────────────────────────────────

def _make_1k_clf():
    rng = np.random.RandomState(42)
    X = rng.randn(1000, 10)
    y = (X[:, 0] + X[:, 1] * 0.5 + rng.randn(1000) * 0.5 > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df["target"] = y
    return df


def _make_1k_reg():
    rng = np.random.RandomState(42)
    X = rng.randn(1000, 10)
    y = X[:, 0] * 2.0 + X[:, 1] * 1.5 + rng.randn(1000) * 0.5
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df["target"] = y
    return df


# Build splits once at module load (not re-split per test)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _S_CLF = ml.split(data=_make_1k_clf(), target="target", seed=42)
    _S_REG = ml.split(data=_make_1k_reg(), target="target", seed=42)


# ── Speed bound tests ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("algorithm,task,bound_s", SPEED_BOUNDS, ids=_BOUND_IDS)
def test_speed_bound(algorithm, task, bound_s):
    """ml.fit + ml.predict on 1K rows within bound. Detects catastrophic regressions.

    ML_SPEED_FACTOR scales all bounds for environment calibration:
    - server:   ML_SPEED_FACTOR=1.0  (default, calibration baseline)
    - macbook: ML_SPEED_FACTOR=2.0  (ARM, slower for Rust initialization)
    - slow CI: ML_SPEED_FACTOR=3.0  (shared runners, noisy neighbors)
    """
    s = _S_CLF if task == "clf" else _S_REG
    effective_bound = bound_s * SPEED_FACTOR

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        m = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
        ml.predict(model=m, data=s.valid)
        elapsed = time.perf_counter() - t0

    assert elapsed < effective_bound, (
        f"{algorithm}/{task}: {elapsed:.3f}s > {effective_bound:.3f}s "
        f"(base={bound_s}s × SPEED_FACTOR={SPEED_FACTOR}) — "
        f"CATASTROPHIC REGRESSION detected (>10× expected)"
    )
