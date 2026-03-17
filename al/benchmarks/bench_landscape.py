"""
Algo Landscape Benchmark — 50/50 holdout, all engines, all configs.

Inspired by Roth (2026) ML landscape methodology:
- 50/50 train/test split (no CV overhead → 10x more configs per wall-clock)
- Fixed seed for reproducibility
- Every (dataset × algorithm × config × engine) is one row
- Results stored in SQLite for analysis

Usage:
    python bench_landscape.py                    # full run
    python bench_landscape.py --quick            # smoke test (2 datasets, 2 algos)
    python bench_landscape.py --datasets iris cancer  # specific datasets
    python bench_landscape.py --report           # print summary from existing DB
"""

import argparse
import json
import os
import sqlite3
import sys
import time
import traceback
from itertools import product

import numpy as np
import pandas as pd

import ml

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(__file__), "landscape.db")


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            dataset TEXT NOT NULL,
            n_train INTEGER,
            n_test INTEGER,
            n_features INTEGER,
            n_classes INTEGER,
            task TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            engine TEXT NOT NULL,
            config TEXT NOT NULL,
            fit_time_ms REAL,
            predict_time_ms REAL,
            accuracy REAL,
            f1 REAL,
            rmse REAL,
            r2 REAL,
            mae REAL,
            error TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            mlw_version TEXT,
            hostname TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_results_dataset ON results(dataset)
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

BUNDLED_CLF = ["iris", "cancer", "wine", "churn", "fraud", "titanic"]
BUNDLED_REG = ["tips", "diabetes", "houses"]

# Target columns for each dataset
TARGETS = {
    "iris": "species",
    "cancer": "diagnosis",
    "wine": "cultivar",
    "churn": "churn",
    "fraud": "fraud",
    "titanic": "survived",
    "tips": "tip",
    "diabetes": "progression",
    "houses": "price",
}

# Task type
TASKS = {
    "iris": "clf", "cancer": "clf", "wine": "clf",
    "churn": "clf", "fraud": "clf", "titanic": "clf",
    "tips": "reg", "diabetes": "reg", "houses": "reg",
}


def load_dataset(name: str) -> pd.DataFrame:
    return ml.dataset(name)


# ---------------------------------------------------------------------------
# Algorithm configs — cartesian product style
# ---------------------------------------------------------------------------

def clf_configs():
    """Classification algorithm × hyperparameter grid."""
    configs = []

    # Random Forest
    for n_trees in [50, 100, 200]:
        for max_depth in [5, 10, 20]:
            configs.append(("random_forest", {
                "n_estimators": n_trees, "max_depth": max_depth,
            }))

    # Extra Trees
    for n_trees in [50, 100, 200]:
        for max_depth in [5, 10, 20]:
            configs.append(("extra_trees", {
                "n_estimators": n_trees, "max_depth": max_depth,
            }))

    # Gradient Boosting
    for n_est in [50, 100, 200]:
        for lr in [0.05, 0.1, 0.2]:
            for depth in [3, 5, 7]:
                configs.append(("gradient_boosting", {
                    "n_estimators": n_est, "learning_rate": lr,
                    "max_depth": depth,
                }))

    # Gradient Boosting + leaf_smooth (Bayesian shrinkage)
    for n_est in [100, 200]:
        for lr in [0.05, 0.1]:
            for depth in [5, 7]:
                for ls in [1.0, 5.0, 10.0]:
                    configs.append(("gradient_boosting", {
                        "n_estimators": n_est, "learning_rate": lr,
                        "max_depth": depth, "leaf_smooth": ls,
                    }))

    # Gradient Boosting + DART (dropout regularization)
    for n_est in [100, 200]:
        for lr in [0.1, 0.2]:
            for depth in [5, 7]:
                for dr in [0.05, 0.1, 0.2]:
                    configs.append(("gradient_boosting", {
                        "n_estimators": n_est, "learning_rate": lr,
                        "max_depth": depth, "dart_rate": dr,
                    }))

    # Gradient Boosting + subsample + colsample (stochastic)
    for n_est in [200, 300]:
        for lr in [0.05, 0.1]:
            for depth in [5, 7]:
                configs.append(("gradient_boosting", {
                    "n_estimators": n_est, "learning_rate": lr,
                    "max_depth": depth, "subsample": 0.8,
                    "colsample_bytree": 0.8, "reg_lambda": 1.0,
                }))

    # Gradient Boosting + combined (leaf_smooth + subsample + L2)
    for n_est in [200, 300]:
        for lr in [0.05, 0.1]:
            configs.append(("gradient_boosting", {
                "n_estimators": n_est, "learning_rate": lr,
                "max_depth": 6, "leaf_smooth": 5.0,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            }))

    # Decision Tree
    for max_depth in [3, 5, 10, 20]:
        configs.append(("decision_tree", {"max_depth": max_depth}))

    # Logistic
    for c in [0.01, 0.1, 1.0, 10.0]:
        configs.append(("logistic", {"C": c}))

    # KNN (knn_weights for Rust, weights for sklearn — handled in run_one)
    for k in [3, 5, 7, 11, 15]:
        for w in ["uniform", "distance"]:
            configs.append(("knn", {"n_neighbors": k, "_knn_weights": w}))

    # Naive Bayes
    configs.append(("naive_bayes", {}))

    # AdaBoost
    for n_est in [50, 100, 200]:
        for lr in [0.5, 1.0]:
            configs.append(("adaboost", {
                "n_estimators": n_est, "learning_rate": lr,
            }))

    # SVM
    for c in [0.1, 1.0, 10.0]:
        configs.append(("svm", {"C": c}))

    # XGBoost (uses its own engine — not ml or sklearn)
    for n_est in [50, 100, 200]:
        for lr in [0.05, 0.1, 0.2]:
            for depth in [3, 5, 7]:
                configs.append(("xgboost", {
                    "n_estimators": n_est, "learning_rate": lr,
                    "max_depth": depth,
                }))

    return configs


def reg_configs():
    """Regression algorithm × hyperparameter grid."""
    configs = []

    # Random Forest
    for n_trees in [50, 100, 200]:
        for max_depth in [5, 10, 20]:
            configs.append(("random_forest", {
                "n_estimators": n_trees, "max_depth": max_depth,
            }))

    # Extra Trees
    for n_trees in [50, 100, 200]:
        for max_depth in [5, 10, 20]:
            configs.append(("extra_trees", {
                "n_estimators": n_trees, "max_depth": max_depth,
            }))

    # Gradient Boosting
    for n_est in [50, 100, 200]:
        for lr in [0.05, 0.1, 0.2]:
            for depth in [3, 5, 7]:
                configs.append(("gradient_boosting", {
                    "n_estimators": n_est, "learning_rate": lr,
                    "max_depth": depth,
                }))

    # Gradient Boosting + leaf_smooth
    for n_est in [100, 200]:
        for lr in [0.05, 0.1]:
            for depth in [5, 7]:
                for ls in [1.0, 5.0, 10.0]:
                    configs.append(("gradient_boosting", {
                        "n_estimators": n_est, "learning_rate": lr,
                        "max_depth": depth, "leaf_smooth": ls,
                    }))

    # Gradient Boosting + DART
    for n_est in [100, 200]:
        for lr in [0.1, 0.2]:
            for depth in [5, 7]:
                for dr in [0.05, 0.1, 0.2]:
                    configs.append(("gradient_boosting", {
                        "n_estimators": n_est, "learning_rate": lr,
                        "max_depth": depth, "dart_rate": dr,
                    }))

    # Gradient Boosting + stochastic + regularized
    for n_est in [200, 300]:
        for lr in [0.05, 0.1]:
            for depth in [5, 7]:
                configs.append(("gradient_boosting", {
                    "n_estimators": n_est, "learning_rate": lr,
                    "max_depth": depth, "subsample": 0.8,
                    "colsample_bytree": 0.8, "reg_lambda": 1.0,
                }))

    # Gradient Boosting + combined
    for n_est in [200, 300]:
        for lr in [0.05, 0.1]:
            configs.append(("gradient_boosting", {
                "n_estimators": n_est, "learning_rate": lr,
                "max_depth": 6, "leaf_smooth": 5.0,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            }))

    # Decision Tree
    for max_depth in [3, 5, 10, 20]:
        configs.append(("decision_tree", {"max_depth": max_depth}))

    # Linear (Ridge)
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        configs.append(("linear", {"alpha": alpha}))

    # Elastic Net
    for alpha in [0.01, 0.1, 1.0]:
        for l1_ratio in [0.1, 0.5, 0.9]:
            configs.append(("elastic_net", {
                "alpha": alpha, "l1_ratio": l1_ratio,
            }))

    # KNN
    for k in [3, 5, 7, 11, 15]:
        for w in ["uniform", "distance"]:
            configs.append(("knn", {"n_neighbors": k, "_knn_weights": w}))

    # SVM
    for c in [0.1, 1.0, 10.0]:
        configs.append(("svm", {"C": c}))

    # XGBoost (uses its own engine — not ml or sklearn)
    for n_est in [50, 100, 200]:
        for lr in [0.05, 0.1, 0.2]:
            for depth in [3, 5, 7]:
                configs.append(("xgboost", {
                    "n_estimators": n_est, "learning_rate": lr,
                    "max_depth": depth,
                }))

    return configs


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

ENGINES = ["ml", "sklearn"]

SEED = 42


def run_one(dataset_name: str, target: str, task: str,
            algorithm: str, config: dict, engine: str,
            train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Run a single (dataset × algo × config × engine) trial."""
    n_train = len(train)
    n_test = len(test)
    n_features = train.shape[1] - 1
    n_classes = train[target].nunique() if task == "clf" else 0

    result = {
        "dataset": dataset_name,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "n_classes": n_classes,
        "task": task,
        "algorithm": algorithm,
        "engine": engine,
        "config": json.dumps(config, sort_keys=True),
        "fit_time_ms": None,
        "predict_time_ms": None,
        "accuracy": None,
        "f1": None,
        "rmse": None,
        "r2": None,
        "mae": None,
        "error": None,
    }

    try:
        # Remap _knn_weights to the right param name per engine
        fit_config = dict(config)
        if "_knn_weights" in fit_config:
            w_val = fit_config.pop("_knn_weights")
            if engine == "ml":
                fit_config["knn_weights"] = w_val
            # sklearn KNN uses weights= but ml.fit routes it differently;
            # just skip for sklearn — uniform is the default anyway

        # Fit on dev partition (train+valid with provenance)
        # XGBoost uses its own engine — don't pass engine= param
        t0 = time.perf_counter()
        fit_kwargs = dict(fit_config)
        if algorithm != "xgboost":
            fit_kwargs["engine"] = engine
        model = ml.fit(
            train, target, algorithm=algorithm,
            seed=SEED, **fit_kwargs,
        )
        fit_ms = (time.perf_counter() - t0) * 1000
        result["fit_time_ms"] = round(fit_ms, 2)

        # Predict on test
        test_x = test.drop(columns=[target])
        y_true = test[target]

        t0 = time.perf_counter()
        preds = ml.predict(model, test_x)
        pred_ms = (time.perf_counter() - t0) * 1000
        result["predict_time_ms"] = round(pred_ms, 2)

        # Compute metrics directly from predictions (bypass assess one-shot
        # guard — this is a benchmark, not a user workflow)
        if task == "clf":
            from sklearn.metrics import accuracy_score, f1_score
            result["accuracy"] = round(accuracy_score(y_true, preds), 6)
            # Always use weighted F1 — binary with pos_label=1 breaks on string labels
            result["f1"] = round(f1_score(y_true, preds, average="weighted", zero_division=0), 6)
        else:
            y_t = y_true.values.astype(float)
            p = preds.values.astype(float)
            ss_res = np.sum((y_t - p) ** 2)
            ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
            result["r2"] = round(1 - ss_res / max(ss_tot, 1e-15), 6)
            result["rmse"] = round(np.sqrt(np.mean((y_t - p) ** 2)), 6)
            result["mae"] = round(np.mean(np.abs(y_t - p)), 6)

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _existing_keys(db_path: str) -> set:
    """Return set of (dataset, algorithm, engine, config) already OK in DB."""
    if not os.path.exists(db_path):
        return set()
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT dataset, algorithm, engine, config FROM results WHERE error IS NULL"
    ).fetchall()
    conn.close()
    return {(r[0], r[1], r[2], r[3]) for r in rows}


def run_benchmark(datasets: list[str], quick: bool = False, db_path: str = DB_PATH) -> list[dict]:
    """Run full benchmark across datasets × algorithms × configs × engines."""
    import socket
    hostname = socket.gethostname()
    run_id = f"run_{int(time.time())}"
    results = []
    total = 0
    errors = 0
    skipped = 0
    existing = _existing_keys(db_path)

    for ds_name in datasets:
        task = TASKS[ds_name]
        target = TARGETS[ds_name]

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({task})")
        print(f"{'='*60}")

        df = load_dataset(ds_name)
        print(f"  Shape: {df.shape}")

        # ~50/50 split — use dev (train+valid) vs test
        s = ml.split(df, target, ratio=(0.49, 0.01, 0.5), seed=SEED)
        train = s.dev   # train+valid merged with provenance intact
        test = s.test
        print(f"  Train: {len(train)}, Test: {len(test)}")

        configs = clf_configs() if task == "clf" else reg_configs()
        if quick:
            # Smoke test: 3 configs per algo type
            seen_algos = {}
            filtered = []
            for algo, cfg in configs:
                if algo not in seen_algos:
                    seen_algos[algo] = 0
                if seen_algos[algo] < 2:
                    filtered.append((algo, cfg))
                    seen_algos[algo] += 1
            configs = filtered

        n_configs = len(configs) * len(ENGINES)
        print(f"  Configs: {len(configs)} × {len(ENGINES)} engines = {n_configs} trials")

        for algo, cfg in configs:
            # XGBoost has its own engine — only run once
            engines_for_algo = ["xgboost"] if algo == "xgboost" else ENGINES
            for engine in engines_for_algo:
                total += 1
                # Skip if already have a good result for this combo
                cfg_str = json.dumps(cfg, sort_keys=True)
                if (ds_name, algo, engine, cfg_str) in existing:
                    skipped += 1
                    continue
                r = run_one(ds_name, target, task, algo, cfg, engine, train, test)
                r["run_id"] = run_id
                r["mlw_version"] = ml.__version__
                r["hostname"] = hostname

                status = "OK" if r["error"] is None else f"ERR: {r['error'][:60]}"
                metric_val = r["accuracy"] or r["r2"] or "?"
                time_val = r["fit_time_ms"] or "?"

                if r["error"]:
                    errors += 1
                    print(f"  [{total:4d}] {algo:20s} {engine:7s} {json.dumps(cfg):40s} → {status}")
                else:
                    print(f"  [{total:4d}] {algo:20s} {engine:7s} {json.dumps(cfg):40s} → {metric_val:>8} {time_val:>8.1f}ms")

                results.append(r)

    print(f"\n{'='*60}")
    print(f"DONE: {total} trials, {errors} errors, {skipped} skipped (already in DB), run_id={run_id}")
    print(f"{'='*60}")

    return results


def save_results(results: list[dict], db_path: str):
    conn = init_db(db_path)
    for r in results:
        conn.execute("""
            INSERT INTO results (
                run_id, dataset, n_train, n_test, n_features, n_classes,
                task, algorithm, engine, config,
                fit_time_ms, predict_time_ms,
                accuracy, f1, rmse, r2, mae,
                error, mlw_version, hostname
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["run_id"], r["dataset"], r["n_train"], r["n_test"],
            r["n_features"], r["n_classes"], r["task"],
            r["algorithm"], r["engine"], r["config"],
            r["fit_time_ms"], r["predict_time_ms"],
            r["accuracy"], r["f1"], r["rmse"], r["r2"], r["mae"],
            r["error"], r["mlw_version"], r["hostname"],
        ))
    conn.commit()
    print(f"\nSaved {len(results)} results to {db_path}")
    conn.close()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(db_path: str):
    if not os.path.exists(db_path):
        print(f"No database at {db_path}")
        return

    conn = sqlite3.connect(db_path)

    # Overview
    total = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    errors = conn.execute("SELECT COUNT(*) FROM results WHERE error IS NOT NULL").fetchone()[0]
    runs = conn.execute("SELECT DISTINCT run_id FROM results").fetchall()
    print(f"\nLandscape DB: {total} trials, {errors} errors, {len(runs)} runs")

    # Engine speed comparison (Rust vs sklearn)
    print(f"\n{'='*70}")
    print("ENGINE SPEED COMPARISON (median fit_time_ms)")
    print(f"{'='*70}")

    from collections import defaultdict
    algo_times = defaultdict(lambda: {"rust": [], "sklearn": []})
    rows = conn.execute("""
        SELECT algorithm, engine, fit_time_ms
        FROM results
        WHERE error IS NULL AND fit_time_ms IS NOT NULL
    """).fetchall()
    for algo, engine, fit_ms in rows:
        key = "rust" if engine == "ml" else "sklearn"
        algo_times[algo][key].append(fit_ms)

    print(f"{'Algorithm':20s} {'Rust median':>12s} {'sklearn median':>14s} {'Speedup':>8s} {'n_rust':>6s} {'n_sk':>5s}")
    print("-" * 70)
    for algo in sorted(algo_times.keys()):
        rust = algo_times[algo]["rust"]
        sk = algo_times[algo]["sklearn"]
        if rust and sk:
            r_med = np.median(rust)
            s_med = np.median(sk)
            speedup = s_med / r_med if r_med > 0 else float("inf")
            print(f"{algo:20s} {r_med:10.1f}ms {s_med:12.1f}ms {speedup:7.1f}x {len(rust):>6d} {len(sk):>5d}")
        elif rust:
            r_med = np.median(rust)
            print(f"{algo:20s} {r_med:10.1f}ms {'—':>14s} {'—':>8s} {len(rust):>6d} {'0':>5s}")
        elif sk:
            s_med = np.median(sk)
            print(f"{algo:20s} {'—':>12s} {s_med:12.1f}ms {'—':>8s} {'0':>6s} {len(sk):>5d}")

    # Accuracy parity (Rust vs sklearn)
    print(f"\n{'='*70}")
    print("ACCURACY PARITY (Rust vs sklearn, same config)")
    print(f"{'='*70}")
    parity_rows = conn.execute("""
        SELECT r1.algorithm, r1.dataset, r1.config,
               r1.accuracy as rust_acc, r2.accuracy as sk_acc,
               r1.r2 as rust_r2, r2.r2 as sk_r2
        FROM results r1
        JOIN results r2 ON r1.dataset = r2.dataset
                        AND r1.algorithm = r2.algorithm
                        AND r1.config = r2.config
                        AND r1.run_id = r2.run_id
        WHERE r1.engine = 'ml' AND r2.engine = 'sklearn'
          AND r1.error IS NULL AND r2.error IS NULL
    """).fetchall()

    diffs_clf = []
    diffs_reg = []
    for algo, ds, cfg, r_acc, s_acc, r_r2, s_r2 in parity_rows:
        if r_acc is not None and s_acc is not None:
            diffs_clf.append(abs(r_acc - s_acc))
        if r_r2 is not None and s_r2 is not None:
            diffs_reg.append(abs(r_r2 - s_r2))

    if diffs_clf:
        print(f"Classification accuracy |diff|: "
              f"median={np.median(diffs_clf):.4f}, "
              f"max={np.max(diffs_clf):.4f}, "
              f"n={len(diffs_clf)}")
    if diffs_reg:
        print(f"Regression R2 |diff|: "
              f"median={np.median(diffs_reg):.4f}, "
              f"max={np.max(diffs_reg):.4f}, "
              f"n={len(diffs_reg)}")

    # Best configs per dataset
    print(f"\n{'='*70}")
    print("BEST CONFIG PER DATASET (Rust engine)")
    print(f"{'='*70}")
    for task_col, metric in [("clf", "accuracy"), ("clf", "f1"), ("reg", "r2")]:
        best = conn.execute(f"""
            SELECT dataset, algorithm, config, {metric}, fit_time_ms
            FROM results
            WHERE engine = 'ml' AND error IS NULL AND task = ?
              AND {metric} IS NOT NULL
            ORDER BY dataset, {metric} DESC
        """, (task_col,)).fetchall()

        seen = set()
        for ds, algo, cfg, val, fit_ms in best:
            if ds not in seen:
                seen.add(ds)
                print(f"  {ds:12s} {metric}={val:.4f}  {algo:20s} {fit_ms:>8.1f}ms  {cfg}")

    conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Algo Landscape Benchmark")
    parser.add_argument("--quick", action="store_true", help="Smoke test (fewer configs)")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to benchmark")
    parser.add_argument("--report", action="store_true", help="Print report from existing DB")
    parser.add_argument("--db", default=DB_PATH, help="Database path")
    args = parser.parse_args()

    if args.report:
        print_report(args.db)
        return

    datasets = args.datasets or (BUNDLED_CLF + BUNDLED_REG)

    # Validate dataset names
    for ds in datasets:
        if ds not in TARGETS:
            print(f"Unknown dataset: {ds}")
            print(f"Available: {', '.join(TARGETS.keys())}")
            sys.exit(1)

    print(f"mlw {ml.__version__}")
    print(f"Datasets: {datasets}")
    print(f"Quick mode: {args.quick}")
    print(f"DB: {args.db}")

    results = run_benchmark(datasets, quick=args.quick)
    save_results(results, args.db)
    print_report(args.db)


if __name__ == "__main__":
    main()
