#!/usr/bin/env python3
"""Experiment AT: Hyperparameter tuning inflation × sample size × algorithm.

Design:
  For each dataset × n-level × algorithm × K-budget:
    Split into train_inner (60%) / validation (20%) / test (20%)
    Draw K random HP configs, fit on train_inner, score on BOTH val and test
    Leaky: select best by TEST score, report TEST score
    Honest: select best by VALIDATION score, report TEST score
    Δ_leakage = leaky - honest = actual selection bias from test-set HP tuning

  This is the missing middle between:
    - Seeds (same true AUC, noise-only → vanishes at large n)
    - Algorithm screening (different true AUCs, K-invariant → persists)
    - Hyperparameter tuning (different true AUCs, K-DEPENDENT → persists?)

  The experiment tests whether hyperparameter tuning inflation has a non-zero
  floor at large n, and whether it is K-dependent (unlike screening).

Algorithms (4, ordered by expected hyperparameter sensitivity):
  LR:  LogisticRegression — narrow config space (C only)
  KNN: KNeighborsClassifier — moderate (n_neighbors, weights)
  RF:  RandomForestClassifier — moderate (max_depth, n_estimators, min_samples_leaf)
  XGB: XGBClassifier — high (learning_rate, max_depth, n_estimators, subsample, etc.)

N-levels: 50, 100, 200, 500, 1000, 2000, 5000, 10000
K-budgets: 10, 50, 100 (dose-response)
Repetitions: 5 per (dataset, n, algo, K) cell — different random seeds for
  subsampling and hyperparameter draws, same train/test split seed per rep.

Output: v3_at.jsonl — one row per dataset, all results nested.

Usage:
    python3 exp_at_tune_nscaling.py --data-dir ~/ml_paper/data --out v3_at.jsonl
    python3 exp_at_tune_nscaling.py --data-dir ~/ml_paper/data --out v3_at.jsonl --max-datasets 10  # smoke test
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

# Hyperparameter search spaces

def _sample_lr(rng):
    """Random LogisticRegression config."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    C = float(rng.choice([0.001, 0.01, 0.1, 1.0, 10.0, 100.0]))
    penalty = rng.choice(["l1", "l2"])
    solver = "saga" if penalty == "l1" else "lbfgs"
    return Pipeline([
        ("s", StandardScaler()),
        ("m", LogisticRegression(
            C=C, penalty=penalty, solver=solver,
            max_iter=2000, random_state=int(rng.randint(1e6))))
    ])


def _sample_knn(rng, n_train=None):
    """Random KNeighborsClassifier config. n_train caps n_neighbors."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    candidates = [1, 3, 5, 7, 11, 15, 21, 31]
    if n_train is not None:
        candidates = [k for k in candidates if k < n_train]
        if not candidates:
            candidates = [1]
    n_neighbors = int(rng.choice(candidates))
    weights = rng.choice(["uniform", "distance"])
    metric = rng.choice(["euclidean", "manhattan"])
    return Pipeline([
        ("s", StandardScaler()),
        ("m", KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, metric=metric))
    ])


def _sample_rf(rng):
    """Random RandomForestClassifier config."""
    from sklearn.ensemble import RandomForestClassifier
    n_estimators = int(rng.choice([10, 50, 100]))
    max_depth = rng.choice([None, 3, 5, 10, 20])
    if max_depth is not None:
        max_depth = int(max_depth)
    min_samples_leaf = int(rng.choice([1, 2, 5, 10, 20]))
    max_features = rng.choice(["sqrt", "log2", 0.5, 0.8, 1.0])
    if isinstance(max_features, float):
        max_features = float(max_features)
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features=max_features,
        n_jobs=1, random_state=int(rng.randint(1e6)))


def _sample_xgb(rng):
    """Random XGBClassifier config."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    learning_rate = float(rng.choice([0.01, 0.03, 0.1, 0.3]))
    max_depth = int(rng.choice([3, 4, 6, 8, 10, 12]))
    n_estimators = int(rng.choice([50, 100, 200]))
    subsample = float(rng.choice([0.6, 0.8, 1.0]))
    colsample_bytree = float(rng.choice([0.6, 0.8, 1.0]))
    reg_alpha = float(rng.choice([0, 0.01, 0.1, 1.0]))
    reg_lambda = float(rng.choice([0.1, 1.0, 5.0, 10.0]))
    min_child_weight = int(rng.choice([1, 3, 5, 10]))
    return XGBClassifier(
        learning_rate=learning_rate, max_depth=max_depth,
        n_estimators=n_estimators, subsample=subsample,
        colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
        reg_lambda=reg_lambda, min_child_weight=min_child_weight,
        eval_metric="logloss", verbosity=0, use_label_encoder=False,
        n_jobs=1, random_state=int(rng.randint(1e6)))


def _default_lr(seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("s", StandardScaler()),
        ("m", LogisticRegression(max_iter=1000, random_state=seed))
    ])


def _default_knn(seed):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("s", StandardScaler()),
        ("m", KNeighborsClassifier(n_neighbors=5))
    ])


def _default_rf(seed):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=seed)


def _default_xgb(seed):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(
        n_estimators=100, eval_metric="logloss", verbosity=0,
        use_label_encoder=False, n_jobs=1, random_state=seed)


ALGO_SPECS = {
    "lr":  {"sampler": _sample_lr,  "default": _default_lr,  "label": "LogisticRegression"},
    "knn": {"sampler": _sample_knn, "default": _default_knn, "label": "KNN"},
    "rf":  {"sampler": _sample_rf,  "default": _default_rf,  "label": "RandomForest"},
    "xgb": {"sampler": _sample_xgb, "default": _default_xgb, "label": "XGBoost"},
}

K_BUDGETS = [10, 50, 100]
N_LEVELS_MAIN = [50, 100, 200, 500, 1000, 2000]
N_LEVELS_EXT = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
N_REPS = 3  # match AN experiment


def run_one_cell(X, y, target_n, algo_key, K, seed=SEED):
    """Run one (n, algo, K) cell. Returns list of Δ values across reps."""
    from sklearn.model_selection import train_test_split

    from run_v3_experiments import safe_auc  # noqa: E402

    spec = ALGO_SPECS[algo_key]
    deltas = []

    for rep in range(N_REPS):
        rs = seed + rep * 1000
        rng = np.random.RandomState(rs)

        # Subsample to target_n
        if len(X) > target_n:
            idx = rng.choice(len(X), size=target_n, replace=False)
            Xs, ys = X[idx], y[idx]
        else:
            Xs, ys = X.copy(), y.copy()

        if len(Xs) < 50:  # need 60% for train_inner = 30 rows minimum
            continue

        # Ensure both classes present
        if len(np.unique(ys)) < 2:
            continue

        # Split into train_inner (60%), validation (20%), test (20%)
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                Xs, ys, test_size=0.2, random_state=rs, stratify=ys)
            X_tri, X_val, y_tri, y_val = train_test_split(
                X_tr, y_tr, test_size=0.25, random_state=rs + 999,
                stratify=y_tr)  # 0.25 of 0.8 = 0.2 overall
        except ValueError:
            continue

        # Ensure both classes in all splits
        if (len(np.unique(y_tri)) < 2 or len(np.unique(y_val)) < 2
                or len(np.unique(y_te)) < 2):
            continue

        n_train = len(X_tri)

        # Fit K random HP configs on train_inner, score on BOTH val and test
        # IMPORTANT: fit ONCE, score TWICE. fit_score() refits internally,
        # so we fit manually and score with safe_auc.
        val_aucs = []
        test_aucs = []
        for k in range(K):
            try:
                hp_rng = np.random.RandomState(rs + k + 1)
                if algo_key == "knn":
                    clf = spec["sampler"](hp_rng, n_train=n_train)
                else:
                    clf = spec["sampler"](hp_rng)
                if clf is None:
                    continue
                # Fit once on train_inner
                clf.fit(X_tri, y_tri)
                # Score on both val and test (same fitted model)
                proba_val = clf.predict_proba(X_val)[:, 1]
                proba_test = clf.predict_proba(X_te)[:, 1]
                auc_val = safe_auc(y_val, proba_val)
                auc_test = safe_auc(y_te, proba_test)
                # Skip if either returned the 0.5 default (degenerate)
                if auc_val == 0.5 or auc_test == 0.5:
                    continue
                val_aucs.append(auc_val)
                test_aucs.append(auc_test)
            except Exception:
                continue

        if len(test_aucs) < 3:
            continue

        # Leaky: select best config by TEST score, report TEST score
        best_test_idx = int(np.argmax(test_aucs))
        leaky_auc = test_aucs[best_test_idx]

        # Honest: select best config by VALIDATION score, report TEST score
        best_val_idx = int(np.argmax(val_aucs))
        honest_auc = test_aucs[best_val_idx]

        # Δ_leakage = test-selected test AUC - validation-selected test AUC
        # This is the actual selection bias from using test for HP tuning.
        delta_leakage = leaky_auc - honest_auc
        # Δ_spread = max(test) - mean(test) — HP diversity, not leakage
        delta_spread = max(test_aucs) - float(np.mean(test_aucs))

        deltas.append({"leakage": delta_leakage, "spread": delta_spread})

    return deltas


N_LEVELS_FAST = [500, 1000, 2000, 5000, 10000]


def run_dataset(X, y, name, source, algo_filter=None, k_filter=None,
                fast_levels=False, seed=SEED):
    """Run all cells for one dataset. Returns result dict."""
    n_full = len(X)
    if fast_levels:
        n_levels = [n for n in N_LEVELS_FAST if n <= n_full]
        n_bucket = 10000 if n_full >= 10000 else 2000
    elif n_full >= 10000:
        n_levels = N_LEVELS_EXT
        n_bucket = 10000
    else:
        n_levels = N_LEVELS_MAIN
        n_bucket = 2000

    algos = [a for a in ALGO_SPECS if algo_filter is None or a in algo_filter]
    ks = k_filter or K_BUDGETS

    result = {
        "name": name,
        "source": source,
        "n_rows": n_full,
        "at_n_levels": n_levels,
        "at_n_full": n_bucket,
        "v3_status": "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for algo_key in algos:
        for K in ks:
            leakage_key = f"at_{algo_key}_k{K}_leakage"
            spread_key = f"at_{algo_key}_k{K}_spread"
            leakage_means = []
            spread_means = []
            for target_n in n_levels:
                if target_n > n_full:
                    leakage_means.append(None)
                    spread_means.append(None)
                    continue
                deltas = run_one_cell(X, y, target_n, algo_key, K, seed)
                if deltas:
                    leakage_means.append(
                        float(np.mean([d["leakage"] for d in deltas])))
                    spread_means.append(
                        float(np.mean([d["spread"] for d in deltas])))
                else:
                    leakage_means.append(None)
                    spread_means.append(None)
            result[leakage_key] = leakage_means
            result[spread_key] = spread_means

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Exp AT: Hyperparameter tuning inflation × n × algorithm")
    parser.add_argument("--out", default="v3_at.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--max-datasets", type=int, default=0,
                        help="Limit datasets (0 = all)")
    parser.add_argument("--min-rows", type=int, default=200,
                        help="Minimum dataset rows")
    parser.add_argument("--algos", default="lr,knn,rf,xgb",
                        help="Comma-separated algorithm keys (default: all)")
    parser.add_argument("--k-budgets", default="10,50,100",
                        help="Comma-separated K budgets (default: 10,50,100)")
    parser.add_argument("--ext-only", action="store_true",
                        help="Only run extension datasets (n>=10000)")
    parser.add_argument("--fast-levels", action="store_true",
                        help="Only test n=500,1K,2K,5K,10K (skip small n)")
    args = parser.parse_args()

    # Load dataset list from existing v3_an.jsonl
    an_path = Path(__file__).parent / "../data/v3/v3_an.jsonl"
    if not an_path.exists():
        # Try relative to cwd
        for candidate in [
            Path("results/v3/v3_an.jsonl"),
            Path("v3_an.jsonl"),
        ]:
            if candidate.exists():
                an_path = candidate
                break

    # Parse algo and K-budget filters
    algo_filter = set(args.algos.split(","))
    k_filter = [int(k) for k in args.k_budgets.split(",")]

    if an_path.exists():
        with open(an_path) as f:
            an_rows = [json.loads(line) for line in f]
        if args.ext_only:
            datasets = [(r["name"], r.get("source", "openml")) for r in an_rows
                         if r.get("v3_status") == "ok" and r.get("an_n_full") == 10000]
        else:
            datasets = [(r["name"], r.get("source", "openml")) for r in an_rows
                         if r.get("v3_status") == "ok"]
    else:
        print(f"WARNING: {an_path} not found, need dataset list")
        return

    print(f"Loaded {len(datasets)} datasets from {an_path}")
    print(f"Algos: {algo_filter}, K-budgets: {k_filter}")

    if args.max_datasets > 0:
        datasets = datasets[:args.max_datasets]
        print(f"Limited to {len(datasets)} datasets")

    out_path = Path(args.out)
    t0 = time.time()

    def _process_one(name_source):
        """Process one dataset. Returns (result_dict, status)."""
        name, source = name_source
        try:
            X, y = load_dataset(name, source)
            if X is None:
                return None, "error"
            if len(X) < args.min_rows:
                return None, "skip"
            result = run_dataset(X, y, name, source,
                                 algo_filter=algo_filter,
                                 k_filter=k_filter,
                                 fast_levels=args.fast_levels)
            return result, "ok"
        except Exception as e:
            print(f"  ERR {name}: {e}", file=sys.stderr, flush=True)
            return None, "error"

    # Each model uses n_jobs=1 internally. Parallelize across datasets.
    n_parallel = max(1, os.cpu_count() // 4)
    print(f"Running {len(datasets)} datasets with n_jobs={n_parallel}",
          flush=True)

    # Process in batches for crash safety — flush after each batch
    from joblib import Parallel, delayed
    batch_size = n_parallel * 3
    done = 0
    errors = 0
    skipped = 0

    with open(out_path, "w") as fout:
        for batch_start in range(0, len(datasets), batch_size):
            batch = datasets[batch_start:batch_start + batch_size]
            results = Parallel(n_jobs=n_parallel, verbose=5)(
                delayed(_process_one)(ds) for ds in batch
            )
            for result, status in results:
                if status == "ok" and result is not None:
                    fout.write(json.dumps(result) + "\n")
                    done += 1
                elif status == "skip":
                    skipped += 1
                else:
                    errors += 1
            fout.flush()
            print(f"  [{batch_start + len(batch)}/{len(datasets)}] "
                  f"done={done} err={errors} skip={skipped}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {done} datasets, {errors} errors, {skipped} skipped, "
          f"{elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
