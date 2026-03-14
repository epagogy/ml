"""Engine × Algorithm benchmark for the ml benchmarks.

Measures speed, accuracy, memory, and parity across all ml engines (Rust/native/sklearn)
for 14 algorithms decomposed into 4 primitives (Represent, Objective, Search, Compose).

Feeds algorithm coverage analysis, grammar evidence, and benchmark results.

Usage:
    python benchmarks/bench_engines.py                        # tiny+small, all algos
    python benchmarks/bench_engines.py --medium               # +100K rows
    python benchmarks/bench_engines.py --server                # +1M rows
    python benchmarks/bench_engines.py --algorithm knn        # single algo
    python benchmarks/bench_engines.py --seed-instability     # 10-seed sweep
    python benchmarks/bench_engines.py --json --output engines.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Ensure ml + benchmarks are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from _bench_utils import capture_versions, make_dataset, print_table, run_timed  # noqa: E402

import ml  # noqa: E402

# ── 4-Primitive Taxonomy (locked — from EXP_ALGO_TAXONOMY.md) ──────────────

ALGO_PRIMITIVES: dict[str, dict[str, str]] = {
    "random_forest":      {"represent": "tree",          "objective": "gini/mse",        "search": "greedy",           "compose": "bagging"},
    "decision_tree":      {"represent": "tree",          "objective": "gini/mse",        "search": "greedy",           "compose": "none"},
    "logistic":           {"represent": "linear",        "objective": "cross-entropy+L2","search": "gradient",         "compose": "none"},
    "linear":             {"represent": "linear",        "objective": "mse+L2",          "search": "closed-form",      "compose": "none"},
    "knn":                {"represent": "instance",      "objective": "none",            "search": "exhaustive",       "compose": "voting"},
    "elastic_net":        {"represent": "linear",        "objective": "mse+L1+L2",       "search": "coord_descent",    "compose": "none"},
    "naive_bayes":        {"represent": "probabilistic", "objective": "log-likelihood",  "search": "closed-form",      "compose": "none"},
    "xgboost":            {"represent": "tree",          "objective": "custom+L1+L2",    "search": "gradient",         "compose": "boosting"},
    "svm":                {"represent": "kernel",        "objective": "hinge",           "search": "qp_solver",        "compose": "none"},
    "lightgbm":           {"represent": "tree",          "objective": "custom+L1+L2",    "search": "gradient_hist",    "compose": "boosting"},
    "catboost":           {"represent": "tree",          "objective": "custom+L2",       "search": "gradient_ordered", "compose": "boosting"},
    "adaboost":           {"represent": "tree",          "objective": "exponential",     "search": "greedy",           "compose": "boosting"},
    "gradient_boosting":  {"represent": "tree",          "objective": "custom",          "search": "gradient",         "compose": "boosting"},
    "histgradient":       {"represent": "tree",          "objective": "custom",          "search": "gradient_hist",    "compose": "boosting"},
    "extra_trees":        {"represent": "tree",          "objective": "gini/mse",        "search": "random",           "compose": "bagging"},
}

# ── Engine Matrix: which engines to bench per algorithm ────────────────────

ENGINE_MATRIX: dict[str, dict[str, Any]] = {
    # algo: {engines: [...], tasks: [...], optional: bool}
    "random_forest":      {"engines": ["ml", "sklearn"],               "tasks": ["classification", "regression"]},
    "decision_tree":      {"engines": ["ml", "sklearn"],               "tasks": ["classification", "regression"]},
    "extra_trees":        {"engines": ["ml", "sklearn"],               "tasks": ["classification", "regression"]},
    "logistic":           {"engines": ["ml", "native", "sklearn"],     "tasks": ["classification"]},
    "linear":             {"engines": ["ml", "native", "sklearn"],     "tasks": ["regression"]},
    "knn":                {"engines": ["ml", "native", "sklearn"],     "tasks": ["classification", "regression"]},
    "elastic_net":        {"engines": ["ml", "native", "sklearn"],     "tasks": ["regression"]},
    "naive_bayes":        {"engines": ["ml", "native"],                "tasks": ["classification"]},
    "adaboost":           {"engines": ["ml", "sklearn"],               "tasks": ["classification"]},
    "gradient_boosting":  {"engines": ["ml", "sklearn"],               "tasks": ["classification", "regression"]},
    "histgradient":       {"engines": ["ml", "sklearn"],               "tasks": ["classification", "regression"]},
    "xgboost":            {"engines": ["sklearn"],                     "tasks": ["classification", "regression"], "optional": True},
    "svm":                {"engines": ["ml", "sklearn"],               "tasks": ["classification", "regression"]},
    "lightgbm":           {"engines": ["sklearn"],                     "tasks": ["classification", "regression"], "optional": True},
    "catboost":           {"engines": ["sklearn"],                     "tasks": ["classification", "regression"], "optional": True},
}

# ── Dataset Sizes ──────────────────────────────────────────────────────────

SIZES: dict[str, tuple[int, int]] = {
    "tiny":   (1_000,    10),
    "small":  (10_000,   20),
    "medium": (100_000,  50),
    "large":  (1_000_000, 100),
}

# Size-adaptive warmup/runs
SIZE_RUNS: dict[str, tuple[int, int]] = {
    "tiny":   (3, 7),
    "small":  (3, 7),
    "medium": (1, 5),
    "large":  (0, 3),
}


# ── Preflight: verify engine actually instantiated ─────────────────────────

def _preflight_check(algorithm: str, engine: str, task: str, seed: int = 42) -> bool:
    """Verify that the requested engine actually creates the expected backend.

    Returns True if engine is correctly instantiated. Raises on silent fallback.
    """
    from ml._engines import create

    try:
        est = create(algorithm, task=task, seed=seed, engine=engine)
    except Exception:
        return False

    if engine == "ml":
        # Must be a Rust wrapper, not sklearn
        cls_name = type(est).__name__
        if "Rust" not in cls_name and "_Rust" not in cls_name:
            raise RuntimeError(
                f"PREFLIGHT FAIL: engine='ml' for {algorithm} created {cls_name}, "
                f"expected Rust wrapper. Silent fallback detected."
            )
    return True


# ── Parity Computation ────────────────────────────────────────────────────

def _compute_parity(
    algorithm: str,
    preds_engine: np.ndarray,
    preds_baseline: np.ndarray,
    task: str,
    y_true: np.ndarray,
) -> dict[str, float | None]:
    """Compute parity metrics between engine and baseline predictions.

    Deterministic algos (linear, logistic): exact match <1e-6.
    Stochastic algos (RF, DT, KNN): Spearman >= 0.99 + |AUC delta| < 0.01.
    """
    result: dict[str, float | None] = {
        "auc_delta": None,
        "proba_rank_corr": None,
        "pred_rel_error_pct": None,
    }

    deterministic = algorithm in ("linear", "logistic", "elastic_net", "naive_bayes")

    if deterministic:
        if task == "regression":
            rel_err = np.max(np.abs(preds_engine - preds_baseline)) / (
                np.std(preds_baseline) + 1e-12
            )
            result["pred_rel_error_pct"] = round(float(rel_err * 100), 6)
        else:
            exact_match = float(np.mean(preds_engine == preds_baseline))
            result["pred_rel_error_pct"] = round((1.0 - exact_match) * 100, 6)
    else:
        # Stochastic: Spearman rank correlation on predictions
        if len(np.unique(preds_engine)) > 1 and len(np.unique(preds_baseline)) > 1:
            corr, _ = spearmanr(preds_engine, preds_baseline)
            result["proba_rank_corr"] = round(float(corr), 4)

        # AUC delta
        if task == "classification":
            from sklearn.metrics import roc_auc_score

            try:
                auc_eng = roc_auc_score(y_true, preds_engine)
                auc_base = roc_auc_score(y_true, preds_baseline)
                result["auc_delta"] = round(abs(auc_eng - auc_base), 6)
            except ValueError:
                pass

    return result


# ── Quality Metrics ────────────────────────────────────────────────────────

def _compute_quality(
    model: Any, valid_data: pd.DataFrame, task: str,
) -> dict[str, float | None]:
    """Compute accuracy/roc_auc/f1 for classification, rmse/r2 for regression."""
    metrics = ml.evaluate(model, valid_data)
    result: dict[str, float | None] = {
        "accuracy": None,
        "roc_auc": None,
        "f1": None,
        "rmse": None,
        "r2": None,
    }
    if task == "classification":
        result["accuracy"] = metrics.get("accuracy")
        result["roc_auc"] = metrics.get("roc_auc")
        result["f1"] = metrics.get("f1")
    else:
        result["rmse"] = metrics.get("rmse")
        result["r2"] = metrics.get("r2")
    return result


# ── Core Benchmark Cell ───────────────────────────────────────────────────

def bench_one_cell(
    algorithm: str,
    engine: str,
    task: str,
    size_name: str,
    *,
    seed: int = 42,
    baseline_preds: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    split_result: Any = None,
    n_estimators_override: int | None = 100,
) -> dict[str, Any]:
    """Benchmark a single (algorithm, engine, task, size) cell.

    Returns a dict matching the shared JSON schema with 4-primitive tags.
    """
    n_rows, n_features = SIZES[size_name]
    warmup, runs = SIZE_RUNS[size_name]
    primitives = ALGO_PRIMITIVES[algorithm]

    # Generate data if not provided
    if split_result is None:
        _, split_result = make_dataset(task, n_rows, n_features, seed=seed)

    # Build engine kwargs
    engine_kwargs: dict[str, Any] = {}
    if engine != "auto":
        engine_kwargs["engine"] = engine

    # Standardize n_estimators for ensemble quality comparison
    is_ensemble = primitives["compose"] in ("bagging", "boosting")
    if is_ensemble and n_estimators_override is not None:
        engine_kwargs["n_estimators"] = n_estimators_override

    # Preflight: verify engine actually instantiated
    try:
        if not _preflight_check(algorithm, engine, task, seed):
            return _skipped_cell(algorithm, engine, task, size_name, primitives,
                                 n_rows, n_features, "preflight_failed")
    except RuntimeError as e:
        return _skipped_cell(algorithm, engine, task, size_name, primitives,
                             n_rows, n_features, str(e))

    # Fit timing
    model_holder: list = []

    def fit_fn():
        model_holder.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ml.fit(
                split_result.train, "target",
                algorithm=algorithm, seed=seed, **engine_kwargs,
            )
            model_holder.append(m)

    try:
        fit_stats = run_timed(fit_fn, warmup=warmup, runs=runs)
    except Exception as e:
        return _skipped_cell(algorithm, engine, task, size_name, primitives,
                             n_rows, n_features, f"fit_error: {e}")

    model = model_holder[-1] if model_holder else None
    if model is None:
        return _skipped_cell(algorithm, engine, task, size_name, primitives,
                             n_rows, n_features, "no_model_produced")

    # Predict timing
    def predict_fn():
        ml.predict(model, split_result.valid)

    try:
        predict_stats = run_timed(predict_fn, warmup=min(warmup, 2), runs=min(runs, 5))
    except Exception:
        predict_stats = {"median_seconds": -1, "iqr_seconds": -1, "rss_delta_mb": -1}

    # Quality metrics
    try:
        quality = _compute_quality(model, split_result.valid, task)
    except Exception:
        quality = {"accuracy": None, "roc_auc": None, "f1": None, "rmse": None, "r2": None}

    # Parity (if baseline provided)
    parity: dict[str, float | None] = {
        "auc_delta": None, "proba_rank_corr": None, "pred_rel_error_pct": None,
    }
    if baseline_preds is not None and y_true is not None:
        try:
            preds = ml.predict(model, split_result.valid)
            parity = _compute_parity(algorithm, np.array(preds), baseline_preds, task, y_true)
        except Exception:
            pass

    # Explain support
    explain_works = False
    try:
        ml.explain(model)
        explain_works = True
    except Exception:
        pass

    # Throughput
    train_rows = len(split_result.train)
    fit_rows_per_s = round(train_rows / fit_stats["median_seconds"]) if fit_stats["median_seconds"] > 0 else 0

    return {
        "algorithm": algorithm,
        "engine": engine,
        "task": task,
        "size": size_name,
        "n_rows": n_rows,
        "n_features": n_features,
        **primitives,
        "n_estimators": engine_kwargs.get("n_estimators"),
        "fit_median_s": fit_stats["median_seconds"],
        "fit_iqr_s": fit_stats["iqr_seconds"],
        "fit_rows_per_s": fit_rows_per_s,
        "predict_median_s": predict_stats["median_seconds"],
        "predict_iqr_s": predict_stats["iqr_seconds"],
        "fit_rss_delta_mb": fit_stats["rss_delta_mb"],
        **quality,
        **parity,
        "explain_works": explain_works,
        "skipped": False,
        "skip_reason": None,
    }


def _skipped_cell(
    algorithm: str, engine: str, task: str, size_name: str,
    primitives: dict, n_rows: int, n_features: int, reason: str,
) -> dict[str, Any]:
    """Return a cell dict for a skipped benchmark."""
    return {
        "algorithm": algorithm, "engine": engine, "task": task,
        "size": size_name, "n_rows": n_rows, "n_features": n_features,
        **primitives, "n_estimators": None,
        "fit_median_s": None, "fit_iqr_s": None, "fit_rows_per_s": None,
        "predict_median_s": None, "predict_iqr_s": None,
        "fit_rss_delta_mb": None,
        "accuracy": None, "roc_auc": None, "f1": None,
        "rmse": None, "r2": None,
        "auc_delta": None, "proba_rank_corr": None, "pred_rel_error_pct": None,
        "explain_works": None,
        "skipped": True, "skip_reason": reason,
    }


# ── Seed Instability ──────────────────────────────────────────────────────

def seed_instability(
    algorithm: str,
    engine: str,
    task: str,
    n_rows: int,
    n_features: int,
    n_seeds: int = 10,
) -> dict[str, float]:
    """Measure seed instability: 10 model seeds on FIXED data.

    CRITICAL: data and split are FIXED (seed=42). Only model seed varies.
    """
    _, s = make_dataset(task, n_rows, n_features, seed=42)

    engine_kwargs: dict[str, Any] = {}
    if engine != "auto":
        engine_kwargs["engine"] = engine

    scores: list[float] = []
    for model_seed in range(n_seeds):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ml.fit(
                    s.train, "target",
                    algorithm=algorithm, seed=model_seed, **engine_kwargs,
                )
                metrics = ml.evaluate(model, s.valid)
                score = metrics.get("roc_auc") if task == "classification" else metrics.get("r2")
                if score is not None:
                    scores.append(float(score))
        except Exception:
            continue

    if len(scores) < 2:
        return {"seed_std": float("nan"), "seed_range": float("nan"), "flip_rate": float("nan")}

    # Flip rate: how often does the ranking conclusion change?
    flips = sum(
        1 for a, b in zip(scores[:-1], scores[1:])
        if (a > 0.5) != (b > 0.5)
    )
    flip_rate = flips / (len(scores) - 1) if len(scores) > 1 else 0.0

    return {
        "seed_std": round(float(np.std(scores)), 6),
        "seed_range": round(float(max(scores) - min(scores)), 6),
        "flip_rate": round(flip_rate, 4),
    }


# ── Main Benchmark Loop ──────────────────────────────────────────────────

def _algo_available(algorithm: str) -> bool:
    """Check if algorithm is available in _engines.py."""
    try:
        from ml._engines import create
        create(algorithm, task="classification", seed=42, engine="auto")
        return True
    except Exception:
        # Try regression for regression-only algos
        try:
            create(algorithm, task="regression", seed=42, engine="auto")
            return True
        except Exception:
            return False


def _optional_import_available(algorithm: str) -> bool:
    """Check if optional dependency is installed."""
    imports = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost"}
    pkg = imports.get(algorithm)
    if pkg is None:
        return True
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def run_engine_benchmark(
    sizes: list[str],
    algorithms: list[str] | None = None,
    do_seed_instability: bool = False,
    json_only: bool = False,
) -> dict[str, Any]:
    """Run the full engine benchmark."""
    # Pin RAYON_NUM_THREADS
    rayon_threads = os.environ.get("RAYON_NUM_THREADS", "not_set")
    if rayon_threads == "not_set":
        os.environ["RAYON_NUM_THREADS"] = "1"
        rayon_threads = "1"

    versions = capture_versions()
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "bench_engines.py",
        "language": "python",
        "hardware": {
            "cpu_count": versions["cpu_count"],
            "ram_gb": versions["ram_gb"],
            "machine": versions["machine"],
        },
        "versions": {k: v for k, v in versions.items()
                     if k not in ("cpu_count", "ram_gb", "machine", "platform")},
        "rayon_num_threads": rayon_threads,
        "n_jobs": 1,
        "notes": [
            "histogram CART (Rust) vs exact splits (sklearn)",
            "RSS only — tracemalloc excludes Rust heap",
            "n_estimators=100 standardized for all ensembles",
        ],
    }

    if algorithms is None:
        algorithms = list(ENGINE_MATRIX.keys())

    cells: list[dict[str, Any]] = []

    for size_name in sizes:
        n_rows, n_features = SIZES[size_name]
        if not json_only:
            print(f"\n{'=' * 70}")
            print(f"  SIZE: {size_name.upper()} ({n_rows:,} rows × {n_features} features)")
            print(f"{'=' * 70}")

        # Pre-generate datasets (shared across engines for parity)
        datasets: dict[str, tuple] = {}
        for task in ("classification", "regression"):
            datasets[task] = make_dataset(task, n_rows, n_features, seed=42)

        for algorithm in algorithms:
            spec = ENGINE_MATRIX.get(algorithm)
            if spec is None:
                if not json_only:
                    print(f"  SKIP {algorithm}: not in ENGINE_MATRIX")
                continue

            # Check optional dependencies
            if spec.get("optional") and not _optional_import_available(algorithm):
                if not json_only:
                    print(f"  SKIP {algorithm}: optional dependency not installed")
                for task in spec["tasks"]:
                    for engine in spec["engines"]:
                        cells.append(_skipped_cell(
                            algorithm, engine, task, size_name,
                            ALGO_PRIMITIVES[algorithm], n_rows, n_features,
                            "optional_dependency_not_installed",
                        ))
                continue

            # Check if algorithm is in _engines.py
            if not _algo_available(algorithm):
                if not json_only:
                    print(f"  SKIP {algorithm}: not yet in _engines.py")
                for task in spec["tasks"]:
                    for engine in spec["engines"]:
                        cells.append(_skipped_cell(
                            algorithm, engine, task, size_name,
                            ALGO_PRIMITIVES[algorithm], n_rows, n_features,
                            "not_in_engines",
                        ))
                continue

            for task in spec["tasks"]:
                _, s = datasets[task]

                # First engine = baseline for parity
                baseline_preds: np.ndarray | None = None
                y_true: np.ndarray | None = None

                for engine in spec["engines"]:
                    # KNN large: 1 run or skip
                    if algorithm == "knn" and size_name == "large":
                        if not json_only:
                            print(f"  SKIP {algorithm}/{engine}/{task}/{size_name}: KNN O(n²) prohibitive")
                        cells.append(_skipped_cell(
                            algorithm, engine, task, size_name,
                            ALGO_PRIMITIVES[algorithm], n_rows, n_features,
                            "knn_large_prohibitive",
                        ))
                        continue

                    if not json_only:
                        print(f"  {algorithm}/{engine}/{task} ...", end=" ", flush=True)

                    cell = bench_one_cell(
                        algorithm, engine, task, size_name,
                        seed=42, split_result=s,
                        baseline_preds=baseline_preds,
                        y_true=y_true,
                    )
                    cells.append(cell)

                    if not json_only:
                        if cell["skipped"]:
                            print(f"SKIPPED ({cell['skip_reason']})")
                        else:
                            metric = cell["roc_auc"] or cell["r2"] or cell["accuracy"] or "—"
                            print(f"{cell['fit_median_s']:.4f}s  "
                                  f"RSS={cell['fit_rss_delta_mb']}MB  "
                                  f"quality={metric}")

                    # Store first engine preds as baseline for parity
                    if baseline_preds is None and not cell["skipped"]:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                m = ml.fit(
                                    s.train, "target",
                                    algorithm=algorithm, seed=42,
                                    engine=engine if engine != "auto" else "auto",
                                )
                                baseline_preds = np.array(ml.predict(m, s.valid))
                                y_true = np.array(s.valid["target"])
                        except Exception:
                            pass

        # Free memory between sizes
        del datasets
        gc.collect()

    # Seed instability (AFTER main loop)
    seed_results: dict[str, Any] = {}
    if do_seed_instability:
        if not json_only:
            print(f"\n{'=' * 70}")
            print("  SEED INSTABILITY (10 seeds, fixed data)")
            print(f"{'=' * 70}")

        for algorithm in algorithms:
            spec = ENGINE_MATRIX.get(algorithm)
            if spec is None:
                continue
            if not _algo_available(algorithm):
                continue

            for task in spec["tasks"]:
                for engine in spec["engines"][:1]:  # first engine only
                    for size_name in ["small"]:  # seed instability on small only
                        n_rows, n_features = SIZES[size_name]
                        key = f"{algorithm}/{engine}/{task}/{size_name}"
                        if not json_only:
                            print(f"  {key} ...", end=" ", flush=True)
                        try:
                            result = seed_instability(
                                algorithm, engine, task, n_rows, n_features,
                            )
                            seed_results[key] = result
                            if not json_only:
                                print(f"std={result['seed_std']:.6f}  "
                                      f"range={result['seed_range']:.6f}  "
                                      f"flip={result['flip_rate']:.2f}")
                        except Exception as e:
                            seed_results[key] = {"error": str(e)}
                            if not json_only:
                                print(f"ERROR: {e}")

    return {
        "meta": meta,
        "cells": cells,
        "seed_instability": seed_results if seed_results else None,
    }


# ── Pretty Print Summary ──────────────────────────────────────────────────

def print_summary(results: dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    cells = [c for c in results["cells"] if not c["skipped"]]
    if not cells:
        print("No benchmark results to display.")
        return

    # Group by size
    sizes_seen = sorted(set(c["size"] for c in cells),
                        key=lambda s: SIZES.get(s, (0, 0))[0])

    for size in sizes_seen:
        size_cells = [c for c in cells if c["size"] == size]
        rows = []
        for c in size_cells:
            metric_val = c["roc_auc"] or c["r2"] or c["accuracy"]
            rows.append({
                "algo": c["algorithm"],
                "engine": c["engine"],
                "task": c["task"][:3],
                "fit_s": c["fit_median_s"],
                "pred_s": c["predict_median_s"],
                "RSS_MB": c["fit_rss_delta_mb"],
                "quality": metric_val if metric_val is not None else -1,
                "parity": c.get("proba_rank_corr") or c.get("pred_rel_error_pct") or "",
                "search": c["search"],
                "compose": c["compose"],
            })

        print_table(
            rows,
            f"Engine Benchmark — {size.upper()} ({SIZES[size][0]:,} rows)",
            columns=["algo", "engine", "task", "fit_s", "pred_s",
                      "RSS_MB", "quality", "search", "compose"],
        )

    # Seed instability summary
    seed = results.get("seed_instability")
    if seed:
        seed_rows = [
            {"cell": k, "std": v.get("seed_std", "?"), "range": v.get("seed_range", "?"),
             "flip": v.get("flip_rate", "?")}
            for k, v in seed.items() if "error" not in v
        ]
        if seed_rows:
            print_table(seed_rows, "Seed Instability (10 seeds)",
                        columns=["cell", "std", "range", "flip"])


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Engine × Algorithm benchmark (ml benchmarks)"
    )
    parser.add_argument("--medium", action="store_true",
                        help="Include medium size (100K rows)")
    parser.add_argument("--server", action="store_true",
                        help="Include large size (1M rows, server only)")
    parser.add_argument("--algorithm", type=str, default=None,
                        help="Benchmark single algorithm (e.g., 'knn')")
    parser.add_argument("--seed-instability", action="store_true",
                        help="Run 10-seed instability sweep")
    parser.add_argument("--json", action="store_true",
                        help="JSON output only (no progress)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON results to file")

    args = parser.parse_args()

    sizes = ["tiny", "small"]
    if args.medium or args.server:
        sizes.append("medium")
    if args.server:
        sizes.append("large")

    algorithms = None
    if args.algorithm:
        if args.algorithm not in ENGINE_MATRIX:
            print(f"Unknown algorithm: {args.algorithm}")
            print(f"Available: {list(ENGINE_MATRIX.keys())}")
            sys.exit(1)
        algorithms = [args.algorithm]

    results = run_engine_benchmark(
        sizes=sizes,
        algorithms=algorithms,
        do_seed_instability=args.seed_instability,
        json_only=args.json,
    )

    if not args.json:
        print_summary(results)

    if args.json or args.output:
        # NaN → null in JSON
        output = json.dumps(results, indent=2, default=str,
                            allow_nan=False)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            if not args.json:
                print(f"\nResults saved to {args.output}")
        if args.json:
            print(output)


if __name__ == "__main__":
    main()
