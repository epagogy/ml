"""Fast algorithm landscape diagnostic.

Benchmarks all 16 ml algorithm families across 2 data sizes with engine=auto.
Reports: fit time, wrapper overhead, predict time, quality, memory, scaling ratio.

Designed for development feedback, not paper publication. Runtime: ~5-6 min on server.

Usage:
    RAYON_NUM_THREADS=1 python benchmarks/bench_dev.py
    RAYON_NUM_THREADS=1 python benchmarks/bench_dev.py --algorithm logistic
    RAYON_NUM_THREADS=1 python benchmarks/bench_dev.py --json --output dev.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from _bench_utils import capture_versions, make_dataset, print_table, run_timed  # noqa: E402

import ml  # noqa: E402

# ── Algorithm registry ────────────────────────────────────────────────────
# (tasks, optional_dep_name, backend_label)
# Ordered: Rust → native → optional → sklearn

DEV_ALGOS: dict[str, tuple[list[str], str | None, str]] = {
    "random_forest":     (["clf", "reg"], None,        "Rust"),
    "decision_tree":     (["clf", "reg"], None,        "Rust"),
    "logistic":          (["clf"],        None,        "Rust"),
    "linear":            (["reg"],        None,        "Rust"),
    "knn":               (["clf", "reg"], None,        "Rust"),
    "naive_bayes":       (["clf"],        None,        "Rust"),
    "elastic_net":       (["reg"],        None,        "Rust"),
    "xgboost":           (["clf", "reg"], "xgboost",   "xgboost"),
    "lightgbm":          (["clf", "reg"], "lightgbm",  "lightgbm"),
    "catboost":          (["clf", "reg"], "catboost",  "catboost"),
    "tabpfn":            (["clf"],        "tabpfn",    "tabpfn"),
    "svm":               (["clf", "reg"], None,        "Rust"),
    "histgradient":      (["clf", "reg"], None,        "Rust"),
    "gradient_boosting": (["clf", "reg"], None,        "Rust"),
    "extra_trees":       (["clf", "reg"], None,        "Rust"),
    "adaboost":          (["clf"],        None,        "Rust"),
}

# Ensemble algorithms: pin n_estimators=100 to standardize and control runtime.
# Without this, XGBoost/LightGBM default to 500 trees → runtime explodes.
# adaboost excluded: _engines.py already defaults it to 100 internally.
ENSEMBLE_ALGOS = {"random_forest", "extra_trees", "gradient_boosting",
                  "xgboost", "lightgbm", "catboost"}

# Size-adaptive warmup/runs — matches bench_engines.py ladder for fast algos.
# Slow sklearn algos (histgradient, gradient_boosting, svm, extra_trees) use
# fewer iterations to keep total runtime under ~3 min.
SIZES: dict[str, tuple[int, int]] = {
    "tiny":  (1_000, 10),
    "small": (10_000, 20),
}
SIZE_RUNS: dict[str, tuple[int, int]] = {
    "tiny":  (3, 7),
    "small": (2, 5),
}
# Per-algo override: (warmup, runs) for algorithms that are slow per iteration.
ALGO_RUNS_OVERRIDE: dict[str, tuple[int, int]] = {
    "histgradient":      (1, 3),
    "svm":               (1, 3),
    "extra_trees":       (1, 3),
}

# Scaling ratio noise floor: don't report ratio when tiny fit < this (µs noise)
RATIO_MIN_S = 0.01


# ── Helpers ───────────────────────────────────────────────────────────────

def _optional_available(dep: str | None) -> bool:
    if dep is None:
        return True
    try:
        __import__(dep)
        return True
    except ImportError:
        return False


def _fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1000:.2f}ms"
    if seconds < 10:
        return f"{seconds:.3f}s"
    return f"{seconds:.1f}s"


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.0f}%"


def _fmt_scale(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.1f}x"


# ── Core cell ─────────────────────────────────────────────────────────────

def bench_dev_cell(
    algorithm: str,
    task: str,
    size_name: str,
    split,
    *,
    n_estimators_override: int | None = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmark one (algorithm, task, size) cell."""
    quality_key = "roc_auc" if task == "clf" else "r2"

    fit_kwargs: dict[str, Any] = {}
    if algorithm in ENSEMBLE_ALGOS and n_estimators_override is not None:
        fit_kwargs["n_estimators"] = n_estimators_override

    warmup, runs = ALGO_RUNS_OVERRIDE.get(algorithm, SIZE_RUNS[size_name])

    # ── 1. Total ml.fit() time ────────────────────────────────────────────
    model_ref: list = []

    def _do_fit():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ml.fit(
                data=split.train,
                target="target",
                algorithm=algorithm,
                seed=seed,
                **fit_kwargs,
            )
        model_ref.clear()
        model_ref.append(m)

    fit_stats = run_timed(_do_fit, warmup=warmup, runs=runs, measure_rss=True)
    fit_s = fit_stats["median_seconds"]
    rss_mb = fit_stats["rss_delta_mb"]

    model = model_ref[0]

    # ── 2. Raw estimator fit time (bypass ml wrapper) ─────────────────────
    raw_fit_s: float | None = None
    try:
        enc = model._feature_encoder
        inner = model._model
        X_raw = split.train.drop(columns=["target"])
        y_raw = split.train["target"]

        if enc is not None:
            X_enc = enc.transform(X_raw)
            y_enc = enc.encode_target(y_raw)
        else:
            X_enc = X_raw
            y_enc = y_raw

        # Convert to numpy if needed
        if hasattr(X_enc, "values"):
            X_np = X_enc.values.astype(np.float64)
        else:
            X_np = np.asarray(X_enc, dtype=np.float64)
        y_np = np.asarray(y_enc)

        def _do_raw_fit():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inner.fit(X_np, y_np)

        raw_stats = run_timed(_do_raw_fit, warmup=warmup, runs=runs, measure_rss=False)
        raw_fit_s = raw_stats["median_seconds"]
    except Exception:
        raw_fit_s = None

    # ── 3. Wrapper overhead % ─────────────────────────────────────────────
    overhead_pct: float | None = None
    if raw_fit_s is not None and fit_s > 0:
        overhead = (fit_s - raw_fit_s) / fit_s * 100.0
        overhead_pct = max(0.0, min(100.0, overhead))

    # ── 4. Predict time ───────────────────────────────────────────────────
    def _do_predict():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ml.predict(model=model, data=split.valid)

    pred_stats = run_timed(_do_predict, warmup=warmup, runs=runs, measure_rss=False)
    predict_s = pred_stats["median_seconds"]

    # ── 5. Quality ────────────────────────────────────────────────────────
    quality: float | None = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = ml.evaluate(model=model, data=split.valid)
        quality = metrics.get(quality_key)
    except Exception:
        quality = None

    return {
        "algorithm": algorithm,
        "backend": DEV_ALGOS[algorithm][2],
        "task": task,
        "size": size_name,
        "fit_s": fit_s,
        "raw_fit_s": raw_fit_s,
        "overhead_pct": overhead_pct,
        "predict_s": predict_s,
        "quality": quality,
        "rss_mb": rss_mb,
        "scaling": None,   # filled after collection
        "skipped": False,
        "skip_reason": None,
    }


def _skip_row(algorithm: str, task: str, size_name: str, reason: str) -> dict[str, Any]:
    return {
        "algorithm": algorithm,
        "backend": DEV_ALGOS[algorithm][2],
        "task": task,
        "size": size_name,
        "fit_s": None,
        "raw_fit_s": None,
        "overhead_pct": None,
        "predict_s": None,
        "quality": None,
        "rss_mb": None,
        "scaling": None,
        "skipped": True,
        "skip_reason": reason,
    }


# ── Scaling ratios ─────────────────────────────────────────────────────────

def compute_scaling_ratios(rows: list[dict]) -> list[dict]:
    """Fill scaling = fit_s[small] / fit_s[tiny] per (algo, task). Guard for noise floor."""
    tiny_times: dict[tuple[str, str], float] = {}
    for r in rows:
        if r["size"] == "tiny" and not r["skipped"] and r["fit_s"] is not None:
            tiny_times[(r["algorithm"], r["task"])] = r["fit_s"]

    for r in rows:
        if r["size"] == "small" and not r["skipped"] and r["fit_s"] is not None:
            tiny_t = tiny_times.get((r["algorithm"], r["task"]))
            if tiny_t is not None and tiny_t >= RATIO_MIN_S:
                r["scaling"] = r["fit_s"] / tiny_t
    return rows


# ── Table printing ─────────────────────────────────────────────────────────

def _build_display_rows(rows: list[dict], task: str) -> list[dict[str, str]]:
    """Convert raw rows for one task to display-ready dicts."""
    display = []
    for r in rows:
        if r["task"] != task:
            continue
        if r["skipped"]:
            display.append({
                "algorithm": r["algorithm"],
                "backend": r["backend"],
                "size": "—",
                "fit_s": f"SKIP ({r['skip_reason']})",
                "raw_s": "—",
                "overhead": "—",
                "pred_s": "—",
                "quality": "—",
                "scale": "—",
                "rss_mb": "—",
            })
        else:
            display.append({
                "algorithm": r["algorithm"],
                "backend": r["backend"],
                "size": r["size"],
                "fit_s": _fmt_time(r["fit_s"]) if r["fit_s"] is not None else "—",
                "raw_s": _fmt_time(r["raw_fit_s"]) if r["raw_fit_s"] is not None else "—",
                "overhead": _fmt_pct(r["overhead_pct"]),
                "pred_s": _fmt_time(r["predict_s"]) if r["predict_s"] is not None else "—",
                "quality": f"{r['quality']:.3f}" if r["quality"] is not None else "—",
                "scale": _fmt_scale(r["scaling"]) if r["size"] == "small" else "—",
                "rss_mb": f"{r['rss_mb']:.1f}" if r["rss_mb"] is not None and r["rss_mb"] >= 0 else "—",
            })
    return display


def print_tables(rows: list[dict], n_estimators: int = 100) -> None:
    clf_rows = _build_display_rows(rows, "clf")
    reg_rows = _build_display_rows(rows, "reg")

    cols = ["algorithm", "backend", "size", "fit_s", "raw_s", "overhead", "pred_s", "quality", "scale", "rss_mb"]

    if clf_rows:
        print_table(clf_rows, f"Algorithm Landscape — Classification  (engine=auto, n_estimators={n_estimators})", cols)
    if reg_rows:
        print_table(reg_rows, f"Algorithm Landscape — Regression  (engine=auto, n_estimators={n_estimators})", cols)

    # Skipped summary
    skipped = [r for r in rows if r["skipped"]]
    if skipped:
        seen: set[str] = set()
        print("Skipped:")
        for r in skipped:
            key = f"  {r['algorithm']}: {r['skip_reason']}"
            if key not in seen:
                print(key)
                seen.add(key)
    else:
        print("Skipped: (none)")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fast algorithm landscape benchmark")
    parser.add_argument("--algorithm", help="Run only this algorithm")
    parser.add_argument("--json", action="store_true", help="Emit JSON output instead of tables")
    parser.add_argument("--output", help="Write JSON to file (implies --json)")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="n_estimators for ensemble algorithms (default: 100)")
    args = parser.parse_args()

    emit_json = args.json or args.output is not None

    algo_filter: set[str] | None = None
    if args.algorithm:
        algo_filter = {args.algorithm}
        if args.algorithm not in DEV_ALGOS:
            print(f"Unknown algorithm: {args.algorithm}. Known: {list(DEV_ALGOS)}", file=sys.stderr)
            sys.exit(1)

    if not emit_json:
        print(f"\nml dev benchmark  —  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', 'unset')}")
        print(f"n_estimators={args.n_estimators} for ensembles")

    # Pre-generate datasets (shared across all algorithms)
    if not emit_json:
        print("\nGenerating datasets...", end=" ", flush=True)

    datasets: dict[tuple[str, str], Any] = {}
    for size_name, (n_rows, n_features) in SIZES.items():
        for task_full in ("classification", "regression"):
            task = "clf" if task_full == "classification" else "reg"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, s = make_dataset(task_full, n_rows, n_features, seed=42)
            datasets[(task, size_name)] = s

    if not emit_json:
        print("done.")

    all_rows: list[dict] = []

    for algo, (tasks, optional_dep, _backend) in DEV_ALGOS.items():
        if algo_filter and algo not in algo_filter:
            continue

        # Check optional dep
        if not _optional_available(optional_dep):
            for task in tasks:
                for size_name in SIZES:
                    all_rows.append(_skip_row(algo, task, size_name, f"{optional_dep} not installed"))
            continue

        for task in tasks:
            for size_name in SIZES:
                if not emit_json:
                    print(f"  {algo:20s} {task:3s} {size_name:6s} ...", end=" ", flush=True)
                t_wall = time.perf_counter()
                try:
                    split = datasets[(task, size_name)]
                    row = bench_dev_cell(
                        algo, task, size_name, split,
                        n_estimators_override=args.n_estimators,
                        seed=42,
                    )
                except Exception as exc:
                    row = _skip_row(algo, task, size_name, str(exc)[:60])

                all_rows.append(row)
                gc.collect()

                if not emit_json:
                    elapsed = time.perf_counter() - t_wall
                    status = "SKIP" if row["skipped"] else f"{elapsed:.1f}s"
                    print(status)

    # Compute scaling ratios
    all_rows = compute_scaling_ratios(all_rows)

    if emit_json:
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "meta": capture_versions(),
            "n_estimators": args.n_estimators,
            "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", "unset"),
            "cells": all_rows,
        }
        out = json.dumps(result, indent=2)
        if args.output:
            with open(args.output, "w") as fh:
                fh.write(out)
            print(f"Written to {args.output}")
        else:
            print(out)
    else:
        print_tables(all_rows, n_estimators=args.n_estimators)


if __name__ == "__main__":
    main()
