"""Comprehensive benchmark: ml (Rust) vs sklearn.

Usage:
    python benchmarks/bench_comprehensive.py [--json] [--output PATH]

Benchmarks fit + predict speed, accuracy, and memory for all algorithms
on 6 datasets of varying size and shape.

Comprehensive benchmark suite.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import tracemalloc
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

# Ensure ml + benchmarks are importable when run as script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ml  # noqa: E402

# Suppress noisy warnings during benchmark runs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42

# ── Dataset Definitions ──────────────────────────────────────────────────────

DATASETS: dict[str, dict[str, Any]] = {
    "small_clf": {
        "n_rows": 500, "n_features": 10, "task": "classification",
        "n_classes": 2, "description": "500 rows, 10 features, binary",
    },
    "medium_clf": {
        "n_rows": 5_000, "n_features": 20, "task": "classification",
        "n_classes": 2, "description": "5K rows, 20 features, binary",
    },
    "large_clf": {
        "n_rows": 50_000, "n_features": 50, "task": "classification",
        "n_classes": 2, "description": "50K rows, 50 features, binary",
    },
    "small_reg": {
        "n_rows": 500, "n_features": 10, "task": "regression",
        "n_classes": 0, "description": "500 rows, 10 features, regression",
    },
    "medium_reg": {
        "n_rows": 5_000, "n_features": 20, "task": "regression",
        "n_classes": 0, "description": "5K rows, 20 features, regression",
    },
    "multiclass": {
        "n_rows": 2_000, "n_features": 15, "task": "classification",
        "n_classes": 5, "description": "2K rows, 15 features, 5-class",
    },
}

# ── Algorithm Definitions ────────────────────────────────────────────────────
# Each entry: tasks it supports, sklearn class for clf, sklearn class for reg

ALGORITHMS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "tasks": ("classification", "regression"),
        "sklearn_clf": "sklearn.ensemble.RandomForestClassifier",
        "sklearn_reg": "sklearn.ensemble.RandomForestRegressor",
        "params": {"n_estimators": 100, "random_state": SEED, "n_jobs": 1},
    },
    "gradient_boosting": {
        "tasks": ("classification", "regression"),
        "sklearn_clf": "sklearn.ensemble.GradientBoostingClassifier",
        "sklearn_reg": "sklearn.ensemble.GradientBoostingRegressor",
        "params": {"n_estimators": 100, "random_state": SEED},
    },
    "logistic": {
        "tasks": ("classification",),
        "sklearn_clf": "sklearn.linear_model.LogisticRegression",
        "sklearn_reg": None,
        "params": {"max_iter": 1000, "random_state": SEED},
    },
    "linear": {
        "tasks": ("regression",),
        "sklearn_clf": None,
        "sklearn_reg": "sklearn.linear_model.Ridge",
        "params": {"alpha": 1.0, "random_state": SEED},
    },
    "decision_tree": {
        "tasks": ("classification", "regression"),
        "sklearn_clf": "sklearn.tree.DecisionTreeClassifier",
        "sklearn_reg": "sklearn.tree.DecisionTreeRegressor",
        "params": {"random_state": SEED},
    },
    "knn": {
        "tasks": ("classification", "regression"),
        "sklearn_clf": "sklearn.neighbors.KNeighborsClassifier",
        "sklearn_reg": "sklearn.neighbors.KNeighborsRegressor",
        "params": {"n_neighbors": 5},
    },
    "svm": {
        "tasks": ("classification", "regression"),
        "sklearn_clf": "sklearn.svm.SVC",
        "sklearn_reg": "sklearn.svm.SVR",
        "params": {"kernel": "linear", "max_iter": 10000, "random_state": SEED},
        "max_rows": 10_000,  # SVM without shrinking is O(n²×iters); skip large datasets
    },
    "elastic_net": {
        "tasks": ("regression",),
        "sklearn_clf": None,
        "sklearn_reg": "sklearn.linear_model.ElasticNet",
        "params": {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 1000, "random_state": SEED},
    },
}

N_RUNS = 3  # median of 3 runs


# ── Data Generation ──────────────────────────────────────────────────────────

def _generate_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic dataset. Returns (df, X_train, X_test, y_train, y_test).

    df has a "target" column for the ml path.
    X/y arrays are for the sklearn path.
    """
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split

    spec = DATASETS[name]
    n_rows = spec["n_rows"]
    n_features = spec["n_features"]
    task = spec["task"]
    n_classes = spec["n_classes"]

    if task == "classification":
        n_informative = max(2, n_features // 2)
        n_redundant = min(2, n_features - n_informative)
        X, y = make_classification(
            n_samples=n_rows,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            random_state=SEED,
        )
    else:
        X, y = make_regression(
            n_samples=n_rows,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            random_state=SEED,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED,
    )

    # Build DataFrame for ml path
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y

    return df, X_train, X_test, y_train, y_test


# ── sklearn Helpers ──────────────────────────────────────────────────────────

def _import_sklearn_estimator(dotted_path: str) -> type:
    """Import a sklearn estimator class from its dotted path."""
    parts = dotted_path.rsplit(".", 1)
    module_path, class_name = parts[0], parts[1]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _sklearn_metric(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy (clf) or R-squared (reg) for sklearn predictions."""
    if task == "classification":
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)
    else:
        from sklearn.metrics import r2_score
        return r2_score(y_true, y_pred)


# ── Timing + Memory Measurement ─────────────────────────────────────────────

def _measure_fit_predict(
    fit_fn,
    predict_fn,
    n_runs: int = N_RUNS,
) -> dict[str, float]:
    """Measure fit and predict time (median of n_runs) + peak memory.

    Returns:
        fit_time_ms, predict_time_ms, peak_memory_mb
    """
    fit_times: list[float] = []
    predict_times: list[float] = []
    peak_mems: list[float] = []

    for _ in range(n_runs):
        gc.collect()

        # Fit
        tracemalloc.start()
        t0 = time.perf_counter()
        model_or_est = fit_fn()
        fit_elapsed = time.perf_counter() - t0
        _, fit_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        fit_times.append(fit_elapsed * 1000.0)  # ms

        # Predict
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        predict_fn(model_or_est)
        pred_elapsed = time.perf_counter() - t0
        _, pred_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        predict_times.append(pred_elapsed * 1000.0)  # ms
        peak_mems.append(max(fit_peak, pred_peak) / (1024 * 1024))  # MB

    return {
        "fit_time_ms": round(statistics.median(fit_times), 2),
        "predict_time_ms": round(statistics.median(predict_times), 2),
        "peak_memory_mb": round(statistics.median(peak_mems), 2),
    }


# ── ml (Rust engine) Benchmark ──────────────────────────────────────────────

def _bench_ml(
    algorithm: str,
    task: str,
    df: pd.DataFrame,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Benchmark ml with engine='ml' (Rust backend).

    Uses SAME 80/20 split as sklearn for fair comparison.
    Constructs DataFrames from X_train/X_test to use ml.fit() + ml.predict().
    Returns dict with fit_time_ms, predict_time_ms, metric, peak_memory_mb.
    """
    n_features = X_train.shape[1]
    cols = [f"f{i}" for i in range(n_features)]

    df_train = pd.DataFrame(X_train, columns=cols)
    df_train["target"] = y_train
    df_test = pd.DataFrame(X_test, columns=cols)
    df_test["target"] = y_test

    def fit_fn():
        return ml.fit(
            df_train, "target",
            algorithm=algorithm,
            seed=SEED,
            engine="ml",
        )

    def predict_fn(model):
        return ml.predict(model, df_test)

    timing = _measure_fit_predict(fit_fn, predict_fn)

    # Final metric from last run
    model = ml.fit(df_train, "target", algorithm=algorithm, seed=SEED, engine="ml")
    metrics = ml.evaluate(model, df_test)

    if task == "classification":
        metric_val = metrics.get("accuracy", float("nan"))
        metric_name = "accuracy"
    else:
        metric_val = metrics.get("r2", float("nan"))
        metric_name = "r2"

    return {
        **timing,
        "metric_name": metric_name,
        "metric_value": round(metric_val, 4),
    }


# ── sklearn Benchmark ────────────────────────────────────────────────────────

def _bench_sklearn(
    algorithm: str,
    task: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Benchmark sklearn directly (no ml wrapper).

    Returns dict with fit_time_ms, predict_time_ms, metric, peak_memory_mb.
    """
    algo_spec = ALGORITHMS[algorithm]
    params = dict(algo_spec["params"])

    if task == "classification":
        sklearn_path = algo_spec["sklearn_clf"]
    else:
        sklearn_path = algo_spec["sklearn_reg"]

    if sklearn_path is None:
        return {"error": f"{algorithm} does not support {task}"}

    EstClass = _import_sklearn_estimator(sklearn_path)

    # Filter params to only those the estimator accepts
    import inspect
    sig = inspect.signature(EstClass.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_params = {k: v for k, v in params.items() if k in valid_params}

    # ml auto-scales SVM/KNN/Logistic/ElasticNet — match that for fair comparison
    SCALE_SENSITIVE = {"svm", "knn", "logistic", "elastic_net"}
    use_pipeline = algorithm in SCALE_SENSITIVE

    def _make_estimator():
        raw = EstClass(**filtered_params)
        if use_pipeline:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            return Pipeline([("scaler", StandardScaler()), ("model", raw)])
        return raw

    def fit_fn():
        est = _make_estimator()
        est.fit(X_train, y_train)
        return est

    def predict_fn(est):
        return est.predict(X_test)

    timing = _measure_fit_predict(fit_fn, predict_fn)

    # Final metric from last run
    est = _make_estimator()
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    metric_val = _sklearn_metric(task, y_test, y_pred)

    if task == "classification":
        metric_name = "accuracy"
    else:
        metric_name = "r2"

    return {
        **timing,
        "metric_name": metric_name,
        "metric_value": round(metric_val, 4),
    }


# ── Main Benchmark Loop ─────────────────────────────────────────────────────

def run_benchmarks() -> list[dict[str, Any]]:
    """Run all (dataset, algorithm, engine) combinations. Returns list of result dicts."""
    results: list[dict[str, Any]] = []

    # Check sklearn availability
    try:
        import sklearn  # noqa: F401
        has_sklearn = True
    except ImportError:
        has_sklearn = False
        print("[WARN] sklearn not installed -- sklearn benchmarks will be skipped")

    # Check ml Rust backend
    try:
        from ml._rust import HAS_RUST
        has_rust = HAS_RUST
    except ImportError:
        has_rust = False
    if not has_rust:
        print("[WARN] ml Rust backend (ml-py) not available -- ml benchmarks will be skipped")

    # Pre-generate all datasets
    print("Generating datasets...")
    dataset_cache: dict[str, tuple] = {}
    for ds_name in DATASETS:
        print(f"  {ds_name}: {DATASETS[ds_name]['description']}")
        dataset_cache[ds_name] = _generate_dataset(ds_name)
    print()

    total_combos = 0
    for _algo_name, algo_spec in ALGORITHMS.items():
        for _ds_name, ds_spec in DATASETS.items():
            if ds_spec["task"] in algo_spec["tasks"]:
                total_combos += 1
    # x2 for ml + sklearn
    total_combos *= 2

    combo_idx = 0
    for ds_name, ds_spec in DATASETS.items():
        task = ds_spec["task"]
        df, X_train, X_test, y_train, y_test = dataset_cache[ds_name]

        for algo_name, algo_spec in ALGORITHMS.items():
            if task not in algo_spec["tasks"]:
                continue
            max_rows = algo_spec.get("max_rows")
            if max_rows and ds_spec["n_rows"] > max_rows:
                combo_idx += 2  # skip both ml and sklearn
                print(f"  [{combo_idx-1}/{total_combos}] {ds_name} x {algo_name} (ml)... SKIP (>{max_rows} rows)")
                print(f"  [{combo_idx}/{total_combos}] {ds_name} x {algo_name} (sklearn)... SKIP (>{max_rows} rows)")
                continue

            # ── ml (Rust) ──
            combo_idx += 1
            label = f"[{combo_idx}/{total_combos}] {ds_name} x {algo_name} (ml)"
            if has_rust:
                print(f"  {label}...", end="", flush=True)
                try:
                    r = _bench_ml(algo_name, task, df, X_train, X_test, y_train, y_test)
                    r.update({
                        "dataset": ds_name,
                        "algorithm": algo_name,
                        "engine": "ml",
                        "task": task,
                        "n_rows": ds_spec["n_rows"],
                        "n_features": ds_spec["n_features"],
                    })
                    results.append(r)
                    print(
                        f" fit={r['fit_time_ms']:.1f}ms"
                        f"  pred={r['predict_time_ms']:.1f}ms"
                        f"  {r['metric_name']}={r['metric_value']:.3f}"
                        f"  mem={r['peak_memory_mb']:.1f}MB"
                    )
                except Exception as e:
                    print(f" ERROR: {e}")
                    results.append({
                        "dataset": ds_name, "algorithm": algo_name, "engine": "ml",
                        "task": task, "n_rows": ds_spec["n_rows"],
                        "n_features": ds_spec["n_features"],
                        "error": str(e),
                    })
            else:
                print(f"  {label}... SKIP (no Rust)")
                combo_idx += 0  # don't double-count

            # ── sklearn ──
            combo_idx += 1
            label = f"[{combo_idx}/{total_combos}] {ds_name} x {algo_name} (sklearn)"
            if has_sklearn:
                print(f"  {label}...", end="", flush=True)
                try:
                    r = _bench_sklearn(algo_name, task, X_train, X_test, y_train, y_test)
                    if "error" in r:
                        print(f" SKIP: {r['error']}")
                        continue
                    r.update({
                        "dataset": ds_name,
                        "algorithm": algo_name,
                        "engine": "sklearn",
                        "task": task,
                        "n_rows": ds_spec["n_rows"],
                        "n_features": ds_spec["n_features"],
                    })
                    results.append(r)
                    print(
                        f" fit={r['fit_time_ms']:.1f}ms"
                        f"  pred={r['predict_time_ms']:.1f}ms"
                        f"  {r['metric_name']}={r['metric_value']:.3f}"
                        f"  mem={r['peak_memory_mb']:.1f}MB"
                    )
                except Exception as e:
                    print(f" ERROR: {e}")
                    results.append({
                        "dataset": ds_name, "algorithm": algo_name, "engine": "sklearn",
                        "task": task, "n_rows": ds_spec["n_rows"],
                        "n_features": ds_spec["n_features"],
                        "error": str(e),
                    })
            else:
                print(f"  {label}... SKIP (no sklearn)")

    return results


# ── Summary Tables ───────────────────────────────────────────────────────────

def _print_detailed_table(results: list[dict[str, Any]]) -> None:
    """Print a markdown-style table with all results."""
    # Filter out errors
    good = [r for r in results if "error" not in r]
    if not good:
        print("\nNo successful results to display.")
        return

    print("\n## Detailed Results\n")
    print(
        "| Dataset | Algorithm | Engine | Fit (ms) | Predict (ms) "
        "| Metric | Value | Memory (MB) |"
    )
    print(
        "|---------|-----------|--------|----------|-------------|"
        "--------|-------|-------------|"
    )

    for r in sorted(good, key=lambda x: (x["dataset"], x["algorithm"], x["engine"])):
        print(
            f"| {r['dataset']:12s} "
            f"| {r['algorithm']:18s} "
            f"| {r['engine']:7s} "
            f"| {r['fit_time_ms']:8.1f} "
            f"| {r['predict_time_ms']:11.1f} "
            f"| {r['metric_name']:6s} "
            f"| {r['metric_value']:5.3f} "
            f"| {r['peak_memory_mb']:11.1f} |"
        )


def _print_speedup_table(results: list[dict[str, Any]]) -> None:
    """Print speedup ratios: how many times faster ml is vs sklearn."""
    good = [r for r in results if "error" not in r]
    if not good:
        return

    # Group by (dataset, algorithm)
    pairs: dict[tuple[str, str], dict[str, dict]] = {}
    for r in good:
        key = (r["dataset"], r["algorithm"])
        if key not in pairs:
            pairs[key] = {}
        pairs[key][r["engine"]] = r

    # Only pairs that have both engines
    both = {k: v for k, v in pairs.items() if "ml" in v and "sklearn" in v}
    if not both:
        print("\nNo head-to-head comparisons available (need both ml and sklearn).")
        return

    print("\n## Speedup Summary (sklearn_time / ml_time)\n")
    print(
        "| Dataset | Algorithm | ml fit (ms) | sk fit (ms) | Fit speedup "
        "| ml pred (ms) | sk pred (ms) | Pred speedup | Accuracy delta |"
    )
    print(
        "|---------|-----------|-------------|-------------|-------------|"
        "--------------|--------------|--------------|----------------|"
    )

    total_fit_speedups: list[float] = []
    total_pred_speedups: list[float] = []

    for (ds, algo) in sorted(both.keys()):
        ml_r = both[(ds, algo)]["ml"]
        sk_r = both[(ds, algo)]["sklearn"]

        ml_fit = ml_r["fit_time_ms"]
        sk_fit = sk_r["fit_time_ms"]
        fit_speedup = sk_fit / ml_fit if ml_fit > 0 else float("inf")

        ml_pred = ml_r["predict_time_ms"]
        sk_pred = sk_r["predict_time_ms"]
        pred_speedup = sk_pred / ml_pred if ml_pred > 0 else float("inf")

        metric_delta = ml_r["metric_value"] - sk_r["metric_value"]

        total_fit_speedups.append(fit_speedup)
        total_pred_speedups.append(pred_speedup)

        # Format speedup with arrow
        fit_arrow = "faster" if fit_speedup > 1.0 else "slower"
        pred_arrow = "faster" if pred_speedup > 1.0 else "slower"

        print(
            f"| {ds:12s} "
            f"| {algo:18s} "
            f"| {ml_fit:11.1f} "
            f"| {sk_fit:11.1f} "
            f"| {fit_speedup:5.2f}x {fit_arrow:6s} "
            f"| {ml_pred:12.1f} "
            f"| {sk_pred:12.1f} "
            f"| {pred_speedup:5.2f}x {pred_arrow:6s} "
            f"| {metric_delta:+.4f}        |"
        )

    # Geometric mean of speedups
    if total_fit_speedups:
        geo_fit = np.exp(np.mean(np.log(np.clip(total_fit_speedups, 0.001, None))))
        geo_pred = np.exp(np.mean(np.log(np.clip(total_pred_speedups, 0.001, None))))
        n = len(total_fit_speedups)
        print(
            f"\n**Geometric mean** ({n} comparisons): "
            f"fit {geo_fit:.2f}x, predict {geo_pred:.2f}x"
        )

        # Count wins
        fit_wins = sum(1 for s in total_fit_speedups if s > 1.0)
        pred_wins = sum(1 for s in total_pred_speedups if s > 1.0)
        print(
            f"**Win rate**: fit {fit_wins}/{n} ({100*fit_wins/n:.0f}%), "
            f"predict {pred_wins}/{n} ({100*pred_wins/n:.0f}%)"
        )


def _print_errors(results: list[dict[str, Any]]) -> None:
    """Print any errors encountered."""
    errors = [r for r in results if "error" in r]
    if not errors:
        return
    print(f"\n## Errors ({len(errors)} total)\n")
    for r in errors:
        print(f"  - {r['dataset']} x {r['algorithm']} ({r['engine']}): {r['error']}")


# ── Metadata ─────────────────────────────────────────────────────────────────

def _capture_metadata() -> dict[str, Any]:
    """Capture environment metadata for reproducibility."""
    import platform

    meta: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "ml_version": getattr(ml, "__version__", "unknown"),
        "seed": SEED,
        "n_runs": N_RUNS,
    }

    try:
        import sklearn
        meta["sklearn_version"] = sklearn.__version__
    except ImportError:
        meta["sklearn_version"] = "not installed"

    try:
        import numpy
        meta["numpy_version"] = numpy.__version__
    except ImportError:
        pass

    try:
        import pandas
        meta["pandas_version"] = pandas.__version__
    except ImportError:
        pass

    try:
        from ml._rust import HAS_RUST
        meta["rust_backend"] = HAS_RUST
    except ImportError:
        meta["rust_backend"] = False

    try:
        import psutil
        meta["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        meta["ram_gb"] = -1.0

    return meta


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark: ml (Rust) vs sklearn",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Save full results as JSON",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="JSON output path (default: benchmarks/comprehensive_results.json)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Comprehensive Benchmark: ml (Rust) vs sklearn")
    print("=" * 70)

    meta = _capture_metadata()
    print(f"\n  Python {meta['python']} | {meta['platform']} {meta['machine']}")
    print(f"  ml {meta['ml_version']} | sklearn {meta.get('sklearn_version', 'N/A')}")
    print(f"  Rust backend: {meta.get('rust_backend', False)}")
    print(f"  CPUs: {meta['cpu_count']} | RAM: {meta['ram_gb']} GB")
    print(f"  Seed: {SEED} | Runs per measurement: {N_RUNS}")
    print(f"  Datasets: {len(DATASETS)} | Algorithms: {len(ALGORITHMS)}")
    print()

    t_start = time.perf_counter()
    results = run_benchmarks()
    wall_seconds = time.perf_counter() - t_start

    print(f"\nCompleted in {wall_seconds:.1f}s")

    # Print tables
    _print_detailed_table(results)
    _print_speedup_table(results)
    _print_errors(results)

    # JSON output
    if args.json or args.output:
        output_path = args.output or os.path.join(
            os.path.dirname(__file__), "comprehensive_results.json",
        )
        payload = {
            "metadata": meta,
            "wall_seconds": round(wall_seconds, 1),
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nJSON saved to: {output_path}")


if __name__ == "__main__":
    main()
