"""End-to-end workflow benchmarks for ml.

Measures total wall time + peak memory for realistic user workflows
across 4 dataset sizes. Outputs JSON for regression tracking.

Usage:
    python benchmarks/bench_workflows.py              # tiny + small (Mac default)
    python benchmarks/bench_workflows.py --medium      # + medium
    python benchmarks/bench_workflows.py --large       # + large (server only)
    python benchmarks/bench_workflows.py --json        # JSON output only
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
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ml

SIZES = {
    "tiny": (1_000, 10),
    "small": (10_000, 20),
    "medium": (100_000, 50),
    "large": (1_000_000, 100),
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


def _detect_hardware() -> dict:
    cpu_count = os.cpu_count() or 1
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        ram_gb = -1
    return {
        "platform": platform.system(),
        "cpu_count": cpu_count,
        "ram_gb": round(ram_gb, 1),
        "python": platform.python_version(),
    }


def _timed(fn) -> dict:
    """Single timed run with memory tracking."""
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


# ── Workflows ──────────────────────────────────────────────────

def workflow_quick(df: pd.DataFrame) -> dict:
    """Quick workflow: split + fit + evaluate."""
    def fn():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = ml.split(data=df, target="target", seed=42)
            model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
            metrics = ml.evaluate(model, s.valid)
        return {"accuracy": metrics.get("accuracy")}
    return _timed(fn)


def workflow_standard(df: pd.DataFrame) -> dict:
    """Standard workflow: split + screen + fit best + evaluate."""
    def fn():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = ml.split(data=df, target="target", seed=42)
            lb = ml.screen(
                s, "target",
                algorithms=["random_forest", "logistic"],
                seed=42,
            )
            best_algo = lb["algorithm"].iloc[0]
            model = ml.fit(data=s.train, target="target", algorithm=best_algo, seed=42)
            metrics = ml.evaluate(model, s.valid)
        return {"best_algo": best_algo, "accuracy": metrics.get("accuracy")}
    return _timed(fn)


def workflow_power(df: pd.DataFrame) -> dict:
    """Power workflow: split + fit + compare + assess."""
    def fn():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = ml.split(data=df, target="target", seed=42)
            m_rf = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
            m_log = ml.fit(data=s.train, target="target", algorithm="logistic", seed=42)
            lb = ml.compare([m_rf, m_log], data=s.valid)
            verdict = ml.assess(m_rf, test=s.test)
        return {"leaderboard_rows": len(lb), "accuracy": verdict.get("accuracy")}
    return _timed(fn)


# ── Main ──────────────────────────────────────────────────────

def run_benchmarks(sizes: list[str], json_only: bool = False) -> dict:
    hw = _detect_hardware()
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hw,
        "ml_version": getattr(ml, "__version__", "unknown"),
        "workflows": {},
    }

    workflows = {
        "quick": workflow_quick,
        "standard": workflow_standard,
        "power": workflow_power,
    }

    for size_name in sizes:
        n_rows, n_features = SIZES[size_name]
        if not json_only:
            print(f"\n{'='*60}")
            print(f"  {size_name.upper()}: {n_rows:,} rows x {n_features} features")
            print(f"{'='*60}")

        df = _make_clf(n_rows, n_features)
        size_results = {}

        for wf_name, wf_fn in workflows.items():
            # Skip power workflow for large datasets (too slow)
            if size_name == "large" and wf_name == "power":
                continue

            if not json_only:
                print(f"  {wf_name} ...", end=" ", flush=True)

            res = wf_fn(df)
            size_results[wf_name] = {
                "wall_seconds": res["wall_seconds"],
                "peak_memory_mb": res["peak_memory_mb"],
            }

            if not json_only:
                print(f"{res['wall_seconds']:.2f}s / {res['peak_memory_mb']:.0f} MB")

        results["workflows"][size_name] = size_results
        del df
        gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="ml workflow benchmarks")
    parser.add_argument("--medium", action="store_true")
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    sizes = ["tiny", "small"]
    if args.medium or args.large:
        sizes.append("medium")
    if args.large:
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
