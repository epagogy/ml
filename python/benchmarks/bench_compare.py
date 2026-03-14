"""Cross-library comparison benchmark: ml vs PyCaret vs FLAML.

Compares ml.screen() against equivalent functionality in other AutoML
libraries. Gracefully skips libraries not installed.

Usage:
    python benchmarks/bench_compare.py              # ml-only (Mac default)
    python benchmarks/bench_compare.py --all        # all installed libraries
    python benchmarks/bench_compare.py --json       # JSON output
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import tracemalloc
import warnings
from datetime import datetime, timezone

import pandas as pd
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ml

SIZES = {
    "small": (5_000, 20),
    "medium": (50_000, 30),
}


def _make_clf(n_rows: int, n_features: int, seed: int = 42) -> pd.DataFrame:
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


def _timed(fn) -> dict:
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "wall_seconds": round(elapsed, 3),
        "peak_memory_mb": round(peak / (1024 * 1024), 2),
        "result": result,
    }


# ── Library Runners ───────────────────────────────────────────

def bench_ml_screen(df: pd.DataFrame) -> dict:
    """ml.screen() with 2 algorithms."""
    def fn():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = ml.split(data=df, target="target", seed=42)
            lb = ml.screen(
                s, "target",
                algorithms=["random_forest", "logistic"],
                seed=42,
            )
        return {"n_models": len(lb)}
    return _timed(fn)


def bench_pycaret(df: pd.DataFrame) -> dict | None:
    """PyCaret compare_models() — skips if not installed."""
    try:
        from pycaret.classification import compare_models, setup
    except ImportError:
        return None

    def fn():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setup(df, target="target", session_id=42, verbose=False)
            best = compare_models(
                include=["rf", "lr"],
                sort="AUC",
                verbose=False,
            )
        return {"best_model": type(best).__name__}
    return _timed(fn)


def bench_flaml(df: pd.DataFrame) -> dict | None:
    """FLAML AutoML.fit() — skips if not installed."""
    try:
        from flaml import AutoML
    except ImportError:
        return None

    def fn():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            automl = AutoML()
            X = df.drop(columns=["target"])
            y = df["target"]
            automl.fit(
                X, y,
                task="classification",
                time_budget=30,
                estimator_list=["rf", "lgbm"],
                seed=42,
                verbose=0,
            )
        return {"best_estimator": automl.best_estimator}
    return _timed(fn)


# ── Main ──────────────────────────────────────────────────────

def run_benchmarks(
    sizes: list[str],
    include_all: bool = False,
    json_only: bool = False,
) -> dict:
    import numpy
    import sklearn
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "ml": getattr(ml, "__version__", "unknown"),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": numpy.__version__,
        },
        "note": "ml.screen() = holdout defaults. PyCaret = 10-fold CV defaults. "
                "FLAML = HPO with 30s budget. These are different tools.",
        "comparisons": {},
    }

    for size_name in sizes:
        n_rows, n_features = SIZES[size_name]
        if not json_only:
            print(f"\n{'='*60}")
            print(f"  {size_name.upper()}: {n_rows:,} rows x {n_features} features")
            print(f"{'='*60}")

        df = _make_clf(n_rows, n_features)
        size_results = {}

        # ml (always runs)
        if not json_only:
            print("  ml.screen() ...", end=" ", flush=True)
        res = bench_ml_screen(df)
        size_results["ml"] = {
            "wall_seconds": res["wall_seconds"],
            "peak_memory_mb": res["peak_memory_mb"],
        }
        if not json_only:
            print(f"{res['wall_seconds']:.2f}s / {res['peak_memory_mb']:.0f} MB")

        if include_all:
            # PyCaret
            if not json_only:
                print("  pycaret.compare_models() ...", end=" ", flush=True)
            res = bench_pycaret(df)
            if res is None:
                size_results["pycaret"] = {"skipped": True, "reason": "not installed"}
                if not json_only:
                    print("SKIPPED (not installed)")
            else:
                size_results["pycaret"] = {
                    "wall_seconds": res["wall_seconds"],
                    "peak_memory_mb": res["peak_memory_mb"],
                }
                if not json_only:
                    print(f"{res['wall_seconds']:.2f}s / {res['peak_memory_mb']:.0f} MB")

            # FLAML
            if not json_only:
                print("  flaml.AutoML() ...", end=" ", flush=True)
            res = bench_flaml(df)
            if res is None:
                size_results["flaml"] = {"skipped": True, "reason": "not installed"}
                if not json_only:
                    print("SKIPPED (not installed)")
            else:
                size_results["flaml"] = {
                    "wall_seconds": res["wall_seconds"],
                    "peak_memory_mb": res["peak_memory_mb"],
                }
                if not json_only:
                    print(f"{res['wall_seconds']:.2f}s / {res['peak_memory_mb']:.0f} MB")

        results["comparisons"][size_name] = size_results
        del df
        gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="ml cross-library comparison")
    parser.add_argument("--all", action="store_true", help="Compare with PyCaret + FLAML")
    parser.add_argument("--medium", action="store_true", help="Include medium size")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    sizes = ["small"]
    if args.medium:
        sizes.append("medium")

    results = run_benchmarks(sizes, include_all=args.all, json_only=args.json)

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
