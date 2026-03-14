"""Cross-language parity bridge: Python ml(Rust) vs R ml(Rust).

Both Python and R use the same Rust core (ml-py) for 5 algorithms:
  linear, logistic, random_forest, decision_tree, knn

This benchmark verifies that given IDENTICAL data (via CSV round-trip with %.17g),
both languages produce identical (deterministic) or near-identical (stochastic) results.

Parity thresholds by algo class:
  - Deterministic (linear, logistic): pred max_abs_diff < 1e-6
  - Stochastic (RF, DT, KNN): Spearman >= 0.99, |AUC delta| < 0.01

Usage:
    python benchmarks/bench_parity_bridge.py                   # all 5 algos
    python benchmarks/bench_parity_bridge.py --algorithm knn   # single algo
    python benchmarks/bench_parity_bridge.py --json --output parity_bridge.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from _bench_utils import capture_versions  # noqa: E402

import ml  # noqa: E402

# ── Rust-backed Algorithms ────────────────────────────────────────────────────

BRIDGE_ALGOS: dict[str, dict[str, Any]] = {
    "linear":        {"tasks": ["regression"],      "deterministic": True},
    "logistic":      {"tasks": ["classification"],  "deterministic": True},
    "random_forest": {"tasks": ["classification", "regression"], "deterministic": False},
    "decision_tree": {"tasks": ["classification", "regression"], "deterministic": False},
    "knn":           {"tasks": ["classification", "regression"], "deterministic": True},
}

# Parity thresholds
DETERMINISTIC_TOL = 1e-6
STOCHASTIC_SPEARMAN_MIN = 0.99
STOCHASTIC_AUC_DELTA_MAX = 0.01


def _csv_hash(path: str) -> str:
    """SHA256 hash of CSV file for integrity check."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _write_data_csv(df: pd.DataFrame, path: str) -> str:
    """Write DataFrame to CSV with maximum float precision."""
    df.to_csv(path, index=False, float_format="%.17g")
    return _csv_hash(path)


def _r_predict(
    algorithm: str,
    task: str,
    train_csv: str,
    valid_csv: str,
    output_csv: str,
    seed: int = 42,
) -> bool:
    """Call R to fit model on train_csv and predict on valid_csv.

    Returns True if R script succeeded.
    """
    r_script = f"""
suppressPackageStartupMessages(library(ml))

train <- read.csv("{train_csv}", check.names = FALSE)
valid  <- read.csv("{valid_csv}", check.names = FALSE)

# Ensure target types match
if ("{task}" == "classification") {{
  train$target <- factor(train$target)
  valid$target <- factor(valid$target)
}}

s_train <- train
model <- ml_fit(s_train, "target", algorithm = "{algorithm}",
                seed = {seed}L, engine = "ml")
preds <- predict(model, newdata = valid)

out <- data.frame(prediction = as.numeric(preds))
write.csv(out, "{output_csv}", row.names = FALSE)
"""
    result = subprocess.run(
        ["Rscript", "-e", r_script],
        capture_output=True, text=True, timeout=120,
    )
    return result.returncode == 0


def bridge_one_algo(
    algorithm: str,
    task: str,
    n_rows: int = 10_000,
    n_features: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Run parity bridge for one (algorithm, task) pair."""
    from sklearn.datasets import make_classification, make_regression

    # Generate data
    if task == "classification":
        X, y = make_classification(
            n_samples=n_rows, n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_classes=2, random_state=seed,
        )
    else:
        X, y = make_regression(
            n_samples=n_rows, n_features=n_features,
            n_informative=max(2, n_features // 2),
            random_state=seed,
        )

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y

    # Split (same seed → same split)
    s = ml.split(df, "target", seed=seed)

    # Python predictions (engine=ml → Rust)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        py_model = ml.fit(s.train, "target", algorithm=algorithm,
                          seed=seed, engine="ml")
        py_preds = np.array(ml.predict(py_model, s.valid), dtype=float)

    # Write CSVs for R
    with tempfile.TemporaryDirectory() as tmpdir:
        train_csv = os.path.join(tmpdir, "train.csv")
        valid_csv = os.path.join(tmpdir, "valid.csv")
        r_preds_csv = os.path.join(tmpdir, "r_preds.csv")

        train_hash = _write_data_csv(s.train, train_csv)
        valid_hash = _write_data_csv(s.valid, valid_csv)

        # Row count check
        train_rows = len(pd.read_csv(train_csv))
        valid_rows = len(pd.read_csv(valid_csv))
        assert train_rows == len(s.train), f"Train CSV row mismatch: {train_rows} vs {len(s.train)}"
        assert valid_rows == len(s.valid), f"Valid CSV row mismatch: {valid_rows} vs {len(s.valid)}"

        # R predictions
        r_ok = _r_predict(algorithm, task, train_csv, valid_csv, r_preds_csv, seed)

        if not r_ok or not os.path.exists(r_preds_csv):
            return {
                "algorithm": algorithm,
                "task": task,
                "n_rows": n_rows,
                "n_features": n_features,
                "python_ok": True,
                "r_ok": False,
                "parity": None,
                "pass": False,
                "error": "R script failed",
            }

        r_preds = pd.read_csv(r_preds_csv)["prediction"].values.astype(float)

    # Compute parity
    spec = BRIDGE_ALGOS[algorithm]
    is_det = spec["deterministic"]

    result: dict[str, Any] = {
        "algorithm": algorithm,
        "task": task,
        "n_rows": n_rows,
        "n_features": n_features,
        "python_ok": True,
        "r_ok": True,
        "train_csv_hash": train_hash,
        "valid_csv_hash": valid_hash,
    }

    if is_det:
        max_abs_diff = float(np.max(np.abs(py_preds - r_preds)))
        mean_abs_diff = float(np.mean(np.abs(py_preds - r_preds)))
        passed = max_abs_diff < DETERMINISTIC_TOL
        result["parity"] = {
            "type": "deterministic",
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "threshold": DETERMINISTIC_TOL,
        }
    else:
        from scipy.stats import spearmanr

        corr = float("nan")
        if len(np.unique(py_preds)) > 1 and len(np.unique(r_preds)) > 1:
            corr, _ = spearmanr(py_preds, r_preds)
            corr = float(corr)

        auc_delta = float("nan")
        if task == "classification":
            from sklearn.metrics import roc_auc_score

            try:
                y_true = np.array(s.valid["target"], dtype=float)
                auc_py = roc_auc_score(y_true, py_preds)
                auc_r = roc_auc_score(y_true, r_preds)
                auc_delta = abs(auc_py - auc_r)
            except ValueError:
                auc_delta = float("nan")

        passed = (not np.isnan(corr) and corr >= STOCHASTIC_SPEARMAN_MIN)
        if task == "classification" and not np.isnan(auc_delta):
            passed = passed and auc_delta < STOCHASTIC_AUC_DELTA_MAX

        result["parity"] = {
            "type": "stochastic",
            "spearman_corr": corr,
            "auc_delta": auc_delta if task == "classification" else None,
            "spearman_threshold": STOCHASTIC_SPEARMAN_MIN,
            "auc_delta_threshold": STOCHASTIC_AUC_DELTA_MAX,
        }

    result["pass"] = passed
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def run_parity_bridge(
    algorithms: list[str] | None = None,
    json_only: bool = False,
) -> dict[str, Any]:
    """Run cross-language parity bridge for all Rust-backed algorithms."""
    if algorithms is None:
        algorithms = list(BRIDGE_ALGOS.keys())

    versions = capture_versions()
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "bench_parity_bridge.py",
        "csv_precision": "%.17g",
        "versions": versions,
    }

    cells: list[dict] = []

    for algo in algorithms:
        spec = BRIDGE_ALGOS.get(algo)
        if spec is None:
            continue

        for task in spec["tasks"]:
            if not json_only:
                print(f"  {algo}/{task} ... ", end="", flush=True)

            try:
                result = bridge_one_algo(algo, task)
                cells.append(result)

                if not json_only:
                    if result["pass"]:
                        parity = result["parity"]
                        if parity["type"] == "deterministic":
                            print(f"PASS (max_abs_diff={parity['max_abs_diff']:.2e})")
                        else:
                            corr = parity.get("spearman_corr", "?")
                            print(f"PASS (spearman={corr:.4f})")
                    else:
                        print(f"FAIL: {result.get('error', result.get('parity'))}")
            except Exception as e:
                cells.append({
                    "algorithm": algo, "task": task,
                    "pass": False, "error": str(e),
                })
                if not json_only:
                    print(f"ERROR: {e}")

    return {"meta": meta, "cells": cells}


def main():
    parser = argparse.ArgumentParser(
        description="Cross-language parity bridge (Python↔R via shared Rust core)"
    )
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    algorithms = None
    if args.algorithm:
        if args.algorithm not in BRIDGE_ALGOS:
            print(f"Unknown: {args.algorithm}. Available: {list(BRIDGE_ALGOS.keys())}")
            sys.exit(1)
        algorithms = [args.algorithm]

    if not args.json:
        print("Cross-Language Parity Bridge: Python ml(Rust) vs R ml(Rust)")
        print("=" * 60)

    results = run_parity_bridge(algorithms=algorithms, json_only=args.json)

    if not args.json:
        passed = sum(1 for c in results["cells"] if c.get("pass"))
        total = len(results["cells"])
        print(f"\n{'=' * 60}")
        print(f"  {passed}/{total} parity checks passed")
        print(f"{'=' * 60}")

    if args.json or args.output:
        output = json.dumps(results, indent=2, default=str)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            if not args.json:
                print(f"Results saved to {args.output}")
        if args.json:
            print(output)


if __name__ == "__main__":
    main()
