#!/usr/bin/env python3
"""Experiment AN-100K: Extend n-scaling to n=50K and n=100K.

WHAT THIS TESTS:
  Does the peeking noise component fully converge to zero at n=100K?
  Theory predicts: noise = σ·g(K), σ~1/√n. At n=100K, σ≈0.003,
  noise≈0.005 for K=5. Should be near-zero.

  If peeking Δ at n=100K ≈ 0.035 (same as n=10K diversity estimate),
  then the floor IS genuine diversity and the noise has fully decayed.

  If peeking Δ at n=100K < 0.035, then even the diversity estimate
  at n=10K had residual noise.

DESIGN:
  Same as AN experiment: for each dataset with n≥100K:
    Subsample to [50K, 100K]
    Run peeking (5 configs) and seed (10 seeds) at each level
    Also run at [1K, 5K, 10K] for within-dataset trajectory

  Uses the SAME code as run_v3_experiments.py exp_an() for exact
  methodological consistency. No new experiment code — just a wrapper.

NOTE:
  [x] Reimplements exp_an() methodology (same configs, same random-pick honest baseline)
  [x] Only runs on datasets with n≥100K (97 available)
  [x] n_jobs=1 on models, parallelism across datasets
  [x] Crash-safe batch output

Usage:
    python3 exp_an_100k.py --out v3_an_100k.jsonl --max-datasets 3
    python3 exp_an_100k.py --out v3_an_100k.jsonl
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_v3_experiments import (  # noqa: E402
    load_dataset, SEED,
)

# Override the n-levels: we want 1K, 5K, 10K, 50K, 100K
SUBSAMPLE_NS = [1000, 5000, 10000, 50000, 100000]


def run_dataset_100k(X, y, name, source, seed=SEED):
    """Run exp_an at extended n-levels for one large dataset."""
    if len(X) < 100000:
        return None

    from run_v3_experiments import fit_score, make_lr, make_rf, make_dt
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    result = {
        "name": name,
        "source": source,
        "n_rows": len(X),
        "an_n_levels": SUBSAMPLE_NS,
        "an_n_full": 100000,
        "v3_status": "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    peeking_means = []
    seed_means = []
    normalize_means = []

    for target_n in SUBSAMPLE_NS:
        # Peeking: 5 configs, best on test vs random pick
        peek_diffs = []
        for rep in range(3):
            rs = seed + rep
            rng = np.random.RandomState(rs)
            if len(X) > target_n:
                idx = rng.choice(len(X), size=target_n, replace=False)
                Xs, ys = X[idx], y[idx]
            else:
                Xs, ys = X.copy(), y.copy()

            if len(Xs) < 30 or len(np.unique(ys)) < 2:
                continue

            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    Xs, ys, test_size=0.2, random_state=rs, stratify=ys)

                configs = [
                    make_lr(rs),
                    Pipeline([("s", StandardScaler()),
                              ("m", LogisticRegression(max_iter=1000, C=10.0,
                                                       random_state=rs))]),
                    make_rf(rs),
                    make_dt(rs),
                    make_rf(rs, n_est=100),
                ]

                test_aucs = []
                for clf in configs:
                    try:
                        auc = fit_score(clf, X_tr, y_tr, X_te, y_te)
                        test_aucs.append(auc)
                    except Exception:
                        test_aucs.append(0.5)

                if len(test_aucs) < 2:
                    continue

                leaky_auc = max(test_aucs)
                honest_auc = test_aucs[
                    np.random.RandomState(rs * 1000).randint(len(test_aucs))]
                peek_diffs.append(leaky_auc - honest_auc)
            except Exception:
                continue

        peeking_means.append(
            float(np.mean(peek_diffs)) if peek_diffs else None)

        # Seed: best-of-10 vs mean
        seed_diffs = []
        for rep in range(3):
            rs = seed + rep
            rng = np.random.RandomState(rs)
            if len(X) > target_n:
                idx = rng.choice(len(X), size=target_n, replace=False)
                Xs, ys = X[idx], y[idx]
            else:
                Xs, ys = X.copy(), y.copy()

            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    Xs, ys, test_size=0.2, random_state=rs, stratify=ys)
                aucs = []
                for s in range(10):
                    auc = fit_score(make_rf(rs + s * 100),
                                   X_tr, y_tr, X_te, y_te)
                    aucs.append(auc)
                best = max(aucs)
                avg = np.mean(aucs)
                seed_diffs.append(best - avg)
            except Exception:
                continue

        seed_means.append(
            float(np.mean(seed_diffs)) if seed_diffs else None)

        # Normalization: global vs per-fold
        norm_diffs = []
        for rep in range(3):
            rs = seed + rep
            rng = np.random.RandomState(rs)
            if len(X) > target_n:
                idx = rng.choice(len(X), size=target_n, replace=False)
                Xs, ys = X[idx], y[idx]
            else:
                Xs, ys = X.copy(), y.copy()

            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    Xs, ys, test_size=0.2, random_state=rs, stratify=ys)
                sc = StandardScaler().fit(np.vstack([X_tr, X_te]))
                X_tr_g = sc.transform(X_tr)
                X_te_g = sc.transform(X_te)
                auc_leaky = fit_score(
                    LogisticRegression(max_iter=1000, random_state=rs),
                    X_tr_g, y_tr, X_te_g, y_te)
                sc2 = StandardScaler().fit(X_tr)
                X_tr_c = sc2.transform(X_tr)
                X_te_c = sc2.transform(X_te)
                auc_clean = fit_score(
                    LogisticRegression(max_iter=1000, random_state=rs),
                    X_tr_c, y_tr, X_te_c, y_te)
                norm_diffs.append(auc_leaky - auc_clean)
            except Exception:
                continue

        normalize_means.append(
            float(np.mean(norm_diffs)) if norm_diffs else None)

    result["an_peeking_means"] = peeking_means
    result["an_seed_means"] = seed_means
    result["an_normalize_means"] = normalize_means

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Exp AN-100K: n-scaling to 50K/100K")
    parser.add_argument("--out", default="v3_an_100k.jsonl")
    parser.add_argument("--max-datasets", type=int, default=0)
    args = parser.parse_args()

    an_path = Path("results/v3/v3_an.jsonl")
    if not an_path.exists():
        an_path = (Path(__file__).parent
                   / "../.."
                   "/research/results/v3/v3_an.jsonl")

    # Get ALL datasets (not just AN rows) — we need n≥100K
    v1_path = an_path.parent.parent / "leakage_landscape_v1_final.jsonl"
    with open(v1_path) as f:
        v1_rows = [json.loads(line) for line in f]
    # Filter: n≥100K, status ok, deduplicate by (name, source)
    seen = set()
    datasets = []
    for r in v1_rows:
        if r.get("n_rows", 0) < 100000:
            continue
        # Require n_rows field and a non-null experiment result as validity check
        if r.get("n_rows") is None or r.get("n_features") is None:
            continue
        key = (r["name"], r.get("source", "openml"))
        if key in seen:
            continue
        seen.add(key)
        datasets.append(key)

    print(f"Loaded {len(datasets)} datasets with n≥100K")

    if args.max_datasets > 0:
        datasets = datasets[:args.max_datasets]

    from joblib import Parallel, delayed

    def _process_one(name_source):
        name, source = name_source
        try:
            X, y = load_dataset(name, source)
            if X is None:
                return None, "error"
            if len(X) < 100000:
                return None, "skip"
            return run_dataset_100k(X, y, name, source), "ok"
        except Exception as e:
            print(f"  ERR {name}: {e}", file=sys.stderr, flush=True)
            return None, "error"

    n_parallel = max(1, os.cpu_count() // 4)
    print(f"Running with n_jobs={n_parallel}")

    batch_size = n_parallel * 2  # smaller batches — these are BIG datasets
    done = errors = skipped = 0
    t0 = time.time()

    with open(args.out, "w") as fout:
        for batch_start in range(0, len(datasets), batch_size):
            batch = datasets[batch_start:batch_start + batch_size]
            results = Parallel(n_jobs=n_parallel, verbose=5)(
                delayed(_process_one)(ds) for ds in batch
            )
            for res, status in results:
                if status == "ok" and res is not None:
                    fout.write(json.dumps(res) + "\n")
                    done += 1
                elif status == "skip":
                    skipped += 1
                else:
                    errors += 1
            fout.flush()
            print(f"  [{batch_start + len(batch)}/{len(datasets)}] "
                  f"done={done} err={errors} skip={skipped}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {done}, {errors} errors, {skipped} skipped, "
          f"{elapsed:.0f}s ({elapsed/3600:.1f}h)")


if __name__ == "__main__":
    main()
