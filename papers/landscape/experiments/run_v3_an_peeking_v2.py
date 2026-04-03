#!/usr/bin/env python3
"""V3 AN Peeking V2: Model Selection Bias × N-scaling.

Corrects the original AN "peeking" sub-experiment which measured FEATURE
SELECTION leakage (MI on all data vs train-only — Class I mechanism that
decays as O(p/n)).  This script measures the ACTUAL model selection peeking
from V1 Exp B: try K hyperparameter configs on test data, report best-of-K
minus honest baseline.  This is the Class II mechanism claimed in the paper.

Design:
  - Identical 19 configs to V1 Exp B:
      9 RF  (n_estimators × max_depth = {10,50,100} × {3,10,None})
      5 LR  (C = {0.01, 0.1, 1.0, 10.0, 100.0})
      5 DT  (max_depth = {2, 3, 5, 10, None})
  - Subsample at n = 50, 100, 200, 500, 1000, 2000
  - Peek counts K = 1, 2, 5, 10, 15, 19  (same as V1 Exp B)
  - 5 reps per subsample size (matching V1 N_REPS)
  - Honest baseline = random pick from aucs (same protocol as V1 line 778)
  - Filter: n_rows >= 2000 (the subsampling claim needs room to subsample)

Output:
  v3_an_peeking_v2.jsonl — one row per dataset, fields:
    an2_peeking_k{K}_n{N}: mean inflation across reps
    an2_peeking_k{K}_means: [6 floats] one per subsample size (for all K)

Usage:
    python3 run_v3_an_peeking_v2.py \
        --v1 leakage_landscape_v1.jsonl \
        --outdir results/v3/ \
        [--limit 10]

Usage:
    nohup python3 run_v3_an_peeking_v2.py \
        --v1 leakage_landscape_v1.jsonl --outdir results/v3/ \
        > v3_an2_log.txt 2>&1 &
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Config

SEED = 42
N_REPS = 5                                     # match V1
SUBSAMPLE_NS = [50, 100, 200, 500, 1000, 2000]
PEEK_COUNTS = [1, 2, 5, 10, 15, 19]            # match V1 Exp B

DATASET_CACHE_DIR = str(Path(os.environ.get(
    "DATASET_CACHE_DIR", str(Path.home() / ".dataset_cache"))))
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)


# Utilities

def safe_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_score)
    except Exception:
        return 0.5


def prepare_binary(X, y):
    from sklearn.preprocessing import LabelEncoder
    classes = np.unique(y)
    if len(classes) <= 1:
        return None, None
    if len(classes) > 2:
        counts = pd.Series(y).value_counts()
        top2 = counts.index[:2]
        mask = np.isin(y, top2)
        X, y = X[mask], y[mask]
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y


def build_configs():
    """Exact 19 configs from V1 Exp B (run_leakage_landscape.py lines 739-746)."""
    configs = []
    for n_est in [10, 50, 100]:
        for max_d in [3, 10, None]:
            configs.append(("rf", {"n_estimators": n_est, "max_depth": max_d}))
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        configs.append(("lr", {"C": C}))
    for max_d in [2, 3, 5, 10, None]:
        configs.append(("dt", {"max_depth": max_d}))
    assert len(configs) == 19, f"Expected 19 configs, got {len(configs)}"
    return configs


CONFIGS = build_configs()


# Dataset Loading (copied from run_v3_experiments.py)

def load_dataset(name, source):
    """Load dataset from V1 inventory. Returns (X, y) or (None, None)."""
    try:
        if source == "ml":
            import ml as mlpkg
            data = mlpkg.dataset(name)
            for target in ["target", "class", "label", "y"]:
                if target in data.columns:
                    break
            else:
                target = data.columns[-1]
            y = data[target].values
            X_df = data.drop(columns=[target])
            cat_cols = X_df.select_dtypes(include=["object", "category", "string"]).columns
            for col in cat_cols:
                X_df[col] = X_df[col].astype("category").cat.codes
            X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
            X = X_df.values.astype(float)

        elif source == "pmlb":
            import pmlb
            data = pmlb.fetch_data(name, local_cache_dir=str(Path.home() / ".pmlb_cache"))
            if "target" not in data.columns:
                return None, None
            y = data["target"].values
            X_df = data.drop(columns=["target"])
            cat_cols = X_df.select_dtypes(include=["object", "category", "string"]).columns
            for col in cat_cols:
                X_df[col] = X_df[col].astype("category").cat.codes
            X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
            X = X_df.values.astype(float)

        elif source == "openml":
            parts = name.split("_", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                did = int(parts[1])
            else:
                return None, None

            cache_path = os.path.join(DATASET_CACHE_DIR, f"openml_{did}.npz")
            if os.path.exists(cache_path):
                try:
                    d = np.load(cache_path, allow_pickle=False)
                    X, y = d["X"], d["y"]
                    X = np.nan_to_num(X.astype(float), nan=0.0)
                    y = prepare_binary(X, y)[1] if len(np.unique(y)) > 2 else y
                    if y is None:
                        return None, None
                    return X, y
                except Exception:
                    os.remove(cache_path)

            import openml
            ds = openml.datasets.get_dataset(did, download_data=True,
                                              download_qualities=False,
                                              download_features_meta_data=False)
            X_raw, y_raw, cat_mask, _ = ds.get_data(target=ds.default_target_attribute)
            if X_raw is None or y_raw is None:
                return None, None
            if isinstance(X_raw, pd.DataFrame):
                cat_cols = X_raw.select_dtypes(include=["object", "category", "string"]).columns
                for col in cat_cols:
                    X_raw[col] = X_raw[col].astype("category").cat.codes
                X_raw = X_raw.fillna(X_raw.median(numeric_only=True)).fillna(0)
                X = X_raw.values.astype(float)
            else:
                X = np.nan_to_num(np.array(X_raw, dtype=float), nan=0.0)
            if isinstance(y_raw, pd.Series):
                y = y_raw.values
            else:
                y = np.array(y_raw)

            try:
                np.savez_compressed(cache_path, X=X, y=y,
                                    n_cat=np.array(0), nan_pct=np.array(0.0))
            except Exception:
                pass
        else:
            return None, None

        X, y = prepare_binary(X, y)
        if X is None:
            return None, None

        X = np.array(X, dtype=float)
        mask = ~np.isnan(X).any(axis=1)
        X, y = X[mask], y[mask]

        if len(X) < 50:
            return None, None
        return X, y

    except Exception:
        return None, None


# V1 Inventory

def load_v1_inventory(path):
    """Load dataset inventory from V1 JSONL (only 'ok' status)."""
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if r.get("status") != "ok":
                    continue
                rows.append({
                    "name": r["name"],
                    "source": r["source"],
                    "n_rows": r.get("n_rows", 0),
                    "n_features": r.get("n_features", 0),
                })
            except Exception:
                pass
    return rows


# Experiment: Model Selection Peeking × N-scaling

def exp_an_peeking_v2(X, y, seed=SEED):
    """Model selection peeking at each subsample size.

    At each n in SUBSAMPLE_NS:
      1. Subsample to n rows (or use all if dataset smaller)
      2. 80/20 train/test split
      3. Fit ALL 19 configs on train, evaluate on test → 19 AUCs
      4. For each K in PEEK_COUNTS:
           inflation = max(K random AUCs) - honest_random_pick
      5. Average across N_REPS repetitions

    This is V1 Exp B's protocol applied at varying sample sizes.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    results = {"v3_status": "ok"}

    for target_n in SUBSAMPLE_NS:
        # Collect inflations across reps, keyed by K
        inflations = {k: [] for k in PEEK_COUNTS}

        for rep in range(N_REPS):
            rs = seed + rep
            rng = np.random.RandomState(rs)

            # Subsample
            if len(X) > target_n:
                idx = rng.choice(len(X), size=target_n, replace=False)
                Xs, ys = X[idx], y[idx]
            else:
                Xs, ys = X.copy(), y.copy()

            if len(Xs) < 30:
                continue

            # Train/test split
            try:
                X_tv, X_te, y_tv, y_te = train_test_split(
                    Xs, ys, test_size=0.2, random_state=rs, stratify=ys)
            except Exception:
                continue

            # Fit all 19 configs — exact V1 Exp B protocol
            aucs = []
            for atype, params in CONFIGS:
                try:
                    if atype == "rf":
                        clf = RandomForestClassifier(random_state=rs, **params)
                    elif atype == "lr":
                        clf = Pipeline([
                            ("s", StandardScaler()),
                            ("m", LogisticRegression(
                                max_iter=1000, random_state=rs, **params))
                        ])
                    else:
                        clf = DecisionTreeClassifier(random_state=rs, **params)
                    clf.fit(X_tv, y_tv)
                    aucs.append(safe_auc(y_te, clf.predict_proba(X_te)[:, 1]))
                except Exception:
                    pass

            if len(aucs) < 3:
                continue

            # Honest baseline: random pick (V1 protocol, line 778)
            honest = aucs[np.random.RandomState(rs * 1000).randint(len(aucs))]

            # Best-of-K inflation for each peek count
            for k in PEEK_COUNTS:
                ka = min(k, len(aucs))
                idx = np.random.RandomState(rs * 100 + k).choice(
                    len(aucs), size=ka, replace=False)
                inflations[k].append(max(aucs[i] for i in idx) - honest)

        # Store results for this subsample size
        for k in PEEK_COUNTS:
            vals = inflations[k]
            if vals:
                results[f"an2_peeking_k{k}_n{target_n}"] = round(
                    float(np.mean(vals)), 6)
            else:
                results[f"an2_peeking_k{k}_n{target_n}"] = None

    # Also store summary arrays (mean across n for each K)
    for k in PEEK_COUNTS:
        means = []
        for n in SUBSAMPLE_NS:
            v = results.get(f"an2_peeking_k{k}_n{n}")
            means.append(v)
        results[f"an2_peeking_k{k}_means"] = means

    return results


# Runner

def get_completed_keys(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    done.add(r["name"] + "|" + r["source"])
                except Exception:
                    pass
    return done


def main():
    parser = argparse.ArgumentParser(
        description="V3 AN Peeking V2: Model Selection Bias × N-scaling")
    parser.add_argument("--v1", required=True,
                        help="Path to V1 JSONL (dataset inventory)")
    parser.add_argument("--outdir", required=True,
                        help="Output directory")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only N datasets (0=all)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, "v3_an_peeking_v2.jsonl")

    datasets = load_v1_inventory(args.v1)
    print(f"Loaded {len(datasets)} ok datasets from V1")

    # Filter: n >= 2000 (consistent with n-invariance claim)
    completed = get_completed_keys(outpath)
    todo = []
    for ds in datasets:
        key = ds["name"] + "|" + ds["source"]
        if key in completed:
            continue
        if ds.get("n_rows", 0) < 2000:
            continue
        todo.append(ds)

    if args.limit > 0:
        todo = todo[:args.limit]

    total = len(todo)
    already = len(completed)
    print(f"\n{'='*60}")
    print(f"AN Peeking V2 (model selection): {total} to do, {already} done")
    print(f"19 configs × {len(SUBSAMPLE_NS)} subsample sizes × "
          f"{N_REPS} reps × {len(PEEK_COUNTS)} K levels")
    print(f"Output: {outpath}")
    print(f"{'='*60}")

    t0 = time.time()
    done = 0
    errors = 0

    for i, ds in enumerate(todo):
        name = ds["name"]
        source = ds["source"]
        elapsed = time.time() - t0
        rate = (done + errors) / max(elapsed, 1) * 3600

        if (i + 1) % 5 == 0 or i == 0:
            remaining = total - i
            eta_h = remaining / max(rate, 1)
            print(f"[{i+1}/{total}] {name[:40]:40s} "
                  f"(done={done} err={errors} {rate:.0f}/hr ETA={eta_h:.1f}h)",
                  flush=True)

        try:
            X, y = load_dataset(name, source)
            if X is None:
                errors += 1
                continue

            result = exp_an_peeking_v2(X, y)
            result["name"] = name
            result["source"] = source
            result["n_rows"] = len(y)
            result["n_features"] = X.shape[1]
            result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

            with open(outpath, "a") as f:
                f.write(json.dumps(result) + "\n")

            done += 1

        except Exception as e:
            errors += 1
            if (i + 1) % 5 == 0:
                print(f"  ERR: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nAN Peeking V2 complete: {done} ok, {errors} errors, "
          f"{elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Total in {outpath}: {already + done}")


if __name__ == "__main__":
    main()
