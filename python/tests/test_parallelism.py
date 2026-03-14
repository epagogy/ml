"""Parallelism determinism tests
Rayon (Rust's parallel executor) must produce identical predictions regardless
of RAYON_NUM_THREADS. This is only guaranteed when seeding is per-tree-index
(not per-thread), which is how forest.rs implements it.

Tests run in subprocess isolation because RAYON_NUM_THREADS is read once at
rayon pool initialization (first use) and cannot be changed mid-process.

Random Forest determinism across thread counts
Extra Trees determinism across thread counts
GBT determinism across thread counts
"""

import json
import subprocess
import sys

import numpy as np
import pytest

# ── Subprocess prediction script ──────────────────────────────────────────────
# Runs in a fresh process with RAYON_NUM_THREADS set to a specific value.
# Outputs predictions as JSON to stdout.

_PREDICTION_SCRIPT = """
import sys, json, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, sys.argv[1])  # ml package path
import ml

rng = np.random.RandomState(42)
n, p = 200, 8
X = rng.randn(n, p)
y_clf = (X[:, 0] + rng.randn(n) * 0.5 > 0).astype(int)
y_reg = X[:, 0] * 2.0 + rng.randn(n) * 0.5

df_clf = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
df_clf["target"] = y_clf

df_reg = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
df_reg["target"] = y_reg

algorithm = sys.argv[2]
task = sys.argv[3]
df = df_clf if task == "clf" else df_reg

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    s = ml.split(data=df, target="target", seed=42)
    m = ml.fit(data=s.train, target="target", algorithm=algorithm, seed=42)
    preds = ml.predict(model=m, data=s.valid)

print(json.dumps(preds.tolist()))
"""


def _run_with_threads(n_threads, algorithm, task, ml_package_path):
    """Run prediction in subprocess with RAYON_NUM_THREADS set."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_PREDICTION_SCRIPT)
        script_path = f.name

    try:
        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = str(n_threads)

        result = subprocess.run(
            [sys.executable, script_path, ml_package_path, algorithm, task],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Subprocess failed: {result.stderr[:500]}")
        return json.loads(result.stdout.strip())
    finally:
        import os as _os
        _os.unlink(script_path)


def _get_ml_package_path():
    """Get the directory containing the ml package (i.e., the dir that contains ml/)."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.find_spec("ml")
    if spec is None:
        pytest.skip("ml package not found")
    # submodule_search_locations[0] is the ml/ directory itself
    # parent is the directory that contains ml/ (added to sys.path)
    return str(Path(spec.submodule_search_locations[0]).parent)


# ── Random Forest determinism ─────────────────────────────────────────

def test_forest_deterministic_across_threads():
    """Same seed → bitwise identical RF predictions at RAYON_NUM_THREADS=1 vs 4.

    Rayon seeding in forest.rs uses seed.wrapping_add(tree_index as u64),
    NOT per-thread. This ensures determinism regardless of thread count.
    """
    ml_path = _get_ml_package_path()
    preds_1 = _run_with_threads(1, "random_forest", "clf", ml_path)
    preds_4 = _run_with_threads(4, "random_forest", "clf", ml_path)

    assert preds_1 == preds_4, (
        f"random_forest predictions differ between RAYON_NUM_THREADS=1 and 4: "
        f"first diff at index {next(i for i, (a, b) in enumerate(zip(preds_1, preds_4)) if a != b)}"
    )


# ── Extra Trees determinism ────────────────────────────────────────────

def test_extra_trees_deterministic_across_threads():
    """ExtraTrees predictions are bitwise identical regardless of thread count."""
    ml_path = _get_ml_package_path()
    preds_1 = _run_with_threads(1, "extra_trees", "clf", ml_path)
    preds_4 = _run_with_threads(4, "extra_trees", "clf", ml_path)

    assert preds_1 == preds_4, (
        "extra_trees predictions differ between RAYON_NUM_THREADS=1 and 4"
    )


# ── GBT determinism ────────────────────────────────────────────────────

def test_gbt_deterministic_across_threads():
    """GBT predictions are identical regardless of RAYON_NUM_THREADS."""
    ml_path = _get_ml_package_path()
    preds_1 = _run_with_threads(1, "gradient_boosting", "reg", ml_path)
    preds_4 = _run_with_threads(4, "gradient_boosting", "reg", ml_path)

    # For regression, use allclose (f64 may have sub-ULP differences in summation order)
    arr1 = np.array(preds_1)
    arr4 = np.array(preds_4)
    np.testing.assert_allclose(
        arr1, arr4, atol=1e-10,
        err_msg="gradient_boosting predictions differ between RAYON_NUM_THREADS=1 and 4"
    )


# ── RF regression determinism ─────────────────────────────────────────

def test_forest_reg_deterministic_across_threads():
    """RF regression predictions are identical regardless of thread count."""
    ml_path = _get_ml_package_path()
    preds_1 = _run_with_threads(1, "random_forest", "reg", ml_path)
    preds_4 = _run_with_threads(4, "random_forest", "reg", ml_path)

    arr1 = np.array(preds_1)
    arr4 = np.array(preds_4)
    np.testing.assert_allclose(
        arr1, arr4, atol=1e-10,
        err_msg="random_forest/reg predictions differ between RAYON_NUM_THREADS=1 and 4"
    )
