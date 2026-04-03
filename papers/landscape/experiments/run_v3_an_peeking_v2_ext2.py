#!/usr/bin/env python3
"""V3 AN Peeking V2 Extension 2: n=50000 and n=100000.

Extends V2 with two extreme sample sizes for the strongest possible floor claim.
Uses IDENTICAL protocol: 19 configs, 5 reps, honest random baseline.
Only processes datasets with n_rows >= 100000.

Output: v3_an_peeking_v2_ext2.jsonl (separate file).

Usage:
    python3 run_v3_an_peeking_v2_ext2.py \
        --v1 data/leakage_landscape_v1.jsonl \
        --outdir results/v3/ \
        [--limit 10]
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# Config

SEED = 42
N_REPS = 5
SUBSAMPLE_NS = [50000, 100000]
PEEK_COUNTS = [1, 2, 5, 10, 15, 19]

DATASET_CACHE_DIR = str(Path(os.environ.get(
    "DATASET_CACHE_DIR", str(Path.home() / ".dataset_cache"))))
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

# Import shared code from V2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_v3_an_peeking_v2 import (
    safe_auc, CONFIGS,
    load_dataset, load_v1_inventory, get_completed_keys
)


# Experiment (identical logic, extreme sample sizes)

def exp_an_peeking_ext2(X, y, seed=SEED):
    """Model selection peeking at n=50000, 100000.

    Identical protocol to V2 (19 configs, honest random baseline, 5 reps).
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    results = {"v3_status": "ok"}

    for target_n in SUBSAMPLE_NS:
        if len(X) < target_n:
            for k in PEEK_COUNTS:
                results[f"an2_peeking_k{k}_n{target_n}"] = None
            continue

        inflations = {k: [] for k in PEEK_COUNTS}

        for rep in range(N_REPS):
            rs = seed + rep
            rng = np.random.RandomState(rs)

            idx = rng.choice(len(X), size=target_n, replace=False)
            Xs, ys = X[idx], y[idx]

            try:
                X_tv, X_te, y_tv, y_te = train_test_split(
                    Xs, ys, test_size=0.2, random_state=rs, stratify=ys)
            except Exception:
                continue

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

            honest = aucs[np.random.RandomState(rs * 1000).randint(len(aucs))]

            for k in PEEK_COUNTS:
                ka = min(k, len(aucs))
                idx = np.random.RandomState(rs * 100 + k).choice(
                    len(aucs), size=ka, replace=False)
                inflations[k].append(max(aucs[i] for i in idx) - honest)

        for k in PEEK_COUNTS:
            vals = inflations[k]
            if vals:
                results[f"an2_peeking_k{k}_n{target_n}"] = round(
                    float(np.mean(vals)), 6)
            else:
                results[f"an2_peeking_k{k}_n{target_n}"] = None

    for k in PEEK_COUNTS:
        means = []
        for n in SUBSAMPLE_NS:
            v = results.get(f"an2_peeking_k{k}_n{n}")
            means.append(v)
        results[f"an2_peeking_k{k}_means_ext2"] = means

    return results


# Runner

def main():
    parser = argparse.ArgumentParser(
        description="V3 AN Peeking V2 Extension 2: n=50000, 100000")
    parser.add_argument("--v1", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, "v3_an_peeking_v2_ext2.jsonl")

    datasets = load_v1_inventory(args.v1)
    print(f"Loaded {len(datasets)} ok datasets from V1")

    # Filter: n >= 100000
    completed = get_completed_keys(outpath)
    todo = []
    for ds in datasets:
        key = ds["name"] + "|" + ds["source"]
        if key in completed:
            continue
        if ds.get("n_rows", 0) < 100000:
            continue
        todo.append(ds)

    if args.limit > 0:
        todo = todo[:args.limit]

    total = len(todo)
    already = len(completed)
    print(f"\n{'='*60}")
    print(f"AN Peeking V2 Extension 2: {total} to do, {already} done")
    print(f"19 configs × {len(SUBSAMPLE_NS)} subsample sizes × "
          f"{N_REPS} reps × {len(PEEK_COUNTS)} K levels")
    print("Filter: n_rows >= 100000")
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

            if len(X) < 100000:
                errors += 1
                continue

            result = exp_an_peeking_ext2(X, y)
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
    print(f"\nAN Peeking V2 Extension 2 complete: {done} ok, {errors} errors, "
          f"{elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Total in {outpath}: {already + done}")


if __name__ == "__main__":
    main()
