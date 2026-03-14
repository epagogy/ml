"""Per-function performance benchmarks for ml.

Measures wall time, peak memory, and throughput for each ml function
across 4 dataset sizes. Outputs JSON for regression tracking.

Usage:
    python benchmarks/bench_functions.py              # tiny + small (Mac default)
    python benchmarks/bench_functions.py --medium      # + medium
    python benchmarks/bench_functions.py --server       # + large (server only)
    python benchmarks/bench_functions.py --json        # JSON output only
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
import tracemalloc
import warnings
from datetime import datetime, timezone

import pandas as pd
from sklearn.datasets import make_classification, make_regression

# Ensure ml is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ml

# ── Hardware Detection ────────────────────────────────────────

def _detect_hardware() -> dict:
    """Auto-detect hardware profile."""
    cpu_count = os.cpu_count() or 1
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        ram_gb = -1  # unknown
    import numpy
    import sklearn
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "cpu_count": cpu_count,
        "ram_gb": round(ram_gb, 1),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": numpy.__version__,
        "is_server": cpu_count >= 16 or ram_gb > 30,
    }


# ── Dataset Generators ────────────────────────────────────────

SIZES = {
    "tiny": (1_000, 10),
    "small": (10_000, 20),
    "medium": (100_000, 50),
    "large": (1_000_000, 100),
}


def _make_clf(n_rows: int, n_features: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_rows,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_classes=2,
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y
    return df


def _make_reg(n_rows: int, n_features: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic regression dataset."""
    X, y = make_regression(
        n_samples=n_rows,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y
    return df


# ── Benchmark Runner ──────────────────────────────────────────

def _run_timed(fn, warmup: int = 3, measured: int = 5) -> dict:
    """Run function with warmup + measurement, return stats."""
    # Warmup
    for _ in range(warmup):
        gc.collect()
        fn()

    # Measured runs
    times = []
    peak_mems = []
    for _ in range(measured):
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(elapsed)
        peak_mems.append(peak)

    times.sort()
    peak_mems.sort()
    median_time = times[len(times) // 2]
    median_mem = peak_mems[len(peak_mems) // 2]
    iqr_time = times[3 * len(times) // 4] - times[len(times) // 4] if len(times) >= 4 else 0

    return {
        "median_seconds": round(median_time, 4),
        "iqr_seconds": round(iqr_time, 4),
        "peak_memory_mb": round(median_mem / (1024 * 1024), 2),
    }


def bench_split(df: pd.DataFrame) -> dict:
    """Benchmark ml.split()."""
    def fn():
        ml.split(data=df, target="target", seed=42)
    return _run_timed(fn)


def bench_fit(train: pd.DataFrame, algo: str = "random_forest") -> dict:
    """Benchmark ml.fit()."""
    def fn():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ml.fit(data=train, target="target", algorithm=algo, seed=42)
    return _run_timed(fn, warmup=1, measured=3)


def bench_predict(model, data: pd.DataFrame) -> dict:
    """Benchmark ml.predict()."""
    def fn():
        ml.predict(model, data)
    return _run_timed(fn)


def bench_evaluate(model, data: pd.DataFrame) -> dict:
    """Benchmark ml.evaluate()."""
    def fn():
        ml.evaluate(model, data)
    return _run_timed(fn)


def bench_profile(df: pd.DataFrame) -> dict:
    """Benchmark ml.profile()."""
    def fn():
        ml.profile(data=df, target="target")
    return _run_timed(fn)


def bench_screen(split_result, target: str = "target") -> dict:
    """Benchmark ml.screen() with 2 algorithms."""
    def fn():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ml.screen(
                split_result, target,
                algorithms=["random_forest", "logistic"],
                seed=42,
            )
    return _run_timed(fn, warmup=1, measured=3)


def bench_explain(model) -> dict:
    """Benchmark ml.explain()."""
    def fn():
        ml.explain(model)
    return _run_timed(fn)


# ── Main ──────────────────────────────────────────────────────

def run_benchmarks(sizes: list[str], json_only: bool = False) -> dict:
    """Run all benchmarks for given sizes."""
    hw = _detect_hardware()
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hw,
        "ml_version": getattr(ml, "__version__", "unknown"),
        "benchmarks": {},
    }

    for size_name in sizes:
        n_rows, n_features = SIZES[size_name]
        if not json_only:
            print(f"\n{'='*60}")
            print(f"  {size_name.upper()}: {n_rows:,} rows x {n_features} features")
            print(f"{'='*60}")

        df_clf = _make_clf(n_rows, n_features)
        df_reg = _make_reg(n_rows, n_features)

        size_results = {}

        # profile
        if not json_only:
            print("  profile ...", end=" ", flush=True)
        size_results["profile"] = bench_profile(df_clf)
        if not json_only:
            print(f"{size_results['profile']['median_seconds']:.3f}s")

        # split
        if not json_only:
            print("  split ...", end=" ", flush=True)
        size_results["split"] = bench_split(df_clf)
        if not json_only:
            print(f"{size_results['split']['median_seconds']:.3f}s")

        # fit (classification)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_clf = ml.split(data=df_clf, target="target", seed=42)

        if not json_only:
            print("  fit(rf, clf) ...", end=" ", flush=True)
        size_results["fit_rf_clf"] = bench_fit(s_clf.train, "random_forest")
        if not json_only:
            print(f"{size_results['fit_rf_clf']['median_seconds']:.3f}s")

        if not json_only:
            print("  fit(logistic, clf) ...", end=" ", flush=True)
        size_results["fit_logistic_clf"] = bench_fit(s_clf.train, "logistic")
        if not json_only:
            print(f"{size_results['fit_logistic_clf']['median_seconds']:.3f}s")

        # fit (regression)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_reg = ml.split(data=df_reg, target="target", seed=42)

        if not json_only:
            print("  fit(rf, reg) ...", end=" ", flush=True)
        size_results["fit_rf_reg"] = bench_fit(s_reg.train, "random_forest")
        if not json_only:
            print(f"{size_results['fit_rf_reg']['median_seconds']:.3f}s")

        if not json_only:
            print("  fit(linear, reg) ...", end=" ", flush=True)
        size_results["fit_linear_reg"] = bench_fit(s_reg.train, "linear")
        if not json_only:
            print(f"{size_results['fit_linear_reg']['median_seconds']:.3f}s")

        # predict
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_clf = ml.fit(data=s_clf.train, target="target", algorithm="random_forest", seed=42)

        if not json_only:
            print("  predict ...", end=" ", flush=True)
        size_results["predict"] = bench_predict(model_clf, s_clf.valid)
        if not json_only:
            r = size_results["predict"]
            rows = len(s_clf.valid)
            tput = rows / r["median_seconds"] if r["median_seconds"] > 0 else float("inf")
            print(f"{r['median_seconds']:.4f}s ({tput:,.0f} rows/s)")

        # evaluate
        if not json_only:
            print("  evaluate ...", end=" ", flush=True)
        size_results["evaluate"] = bench_evaluate(model_clf, s_clf.valid)
        if not json_only:
            print(f"{size_results['evaluate']['median_seconds']:.3f}s")

        # explain
        if not json_only:
            print("  explain ...", end=" ", flush=True)
        size_results["explain"] = bench_explain(model_clf)
        if not json_only:
            print(f"{size_results['explain']['median_seconds']:.4f}s")

        # screen (skip for large — too slow)
        if n_rows <= 100_000:
            if not json_only:
                print("  screen(rf+logistic) ...", end=" ", flush=True)
            size_results["screen"] = bench_screen(s_clf)
            if not json_only:
                print(f"{size_results['screen']['median_seconds']:.3f}s")

        # throughput calculations
        for _key, res in size_results.items():
            if res["median_seconds"] > 0:
                res["rows_per_second"] = round(n_rows / res["median_seconds"])

        results["benchmarks"][size_name] = size_results

        # Free memory
        del df_clf, df_reg
        gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="ml per-function benchmarks")
    parser.add_argument("--medium", action="store_true", help="Include medium (100K rows)")
    parser.add_argument("--server", action="store_true", help="Include large (1M rows, server only)")
    parser.add_argument("--json", action="store_true", help="JSON output only")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    sizes = ["tiny", "small"]
    if args.medium or args.server:
        sizes.append("medium")
    if args.server:
        sizes.append("large")

    results = run_benchmarks(sizes, json_only=args.json)

    if args.json or args.output:
        output = json.dumps(results, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            if not args.json:
                print(f"\nResults saved to {args.output}")
        else:
            print(output)


if __name__ == "__main__":
    main()
