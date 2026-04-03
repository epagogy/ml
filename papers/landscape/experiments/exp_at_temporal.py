#!/usr/bin/env python3
"""Experiment AT-T: Temporal HP tuning leakage with nested walk-forward CV.

Design:
  Outer loop: walk-forward temporal splits (5 expanding windows)
    train_outer = all data before cutpoint (past)
    test_outer  = data after cutpoint (future)

    Inner loop: 3-fold CV on train_outer for HP selection (honest)
      For each of K HP configs:
        inner_cv_score = mean OOF AUC across inner folds

    honest_config = argmax(inner_cv_scores)  — selected by past-only CV
    leaky_config  = argmax(test_scores)      — selected by peeking at future

    Δ_leakage = test_AUC[leaky_config] - test_AUC[honest_config]

  Two conditions per dataset:
    TEMPORAL: sort by first PC, walk-forward splits
    IID:      random 5-fold CV (control)

  Prediction:
    - IID Δ → 0 at large n (val and test are exchangeable)
    - Temporal Δ persists (past ≠ future, distribution shift)

Usage:
    python3 exp_at_temporal.py --out v3_att.jsonl --max-datasets 5  # smoke
    python3 exp_at_temporal.py --out v3_att.jsonl                   # full
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_v3_experiments import (  # noqa: E402
    load_dataset, safe_auc, SEED,
)

# HP search spaces (same as exp_at, n_jobs=1 for parallelism)

def _sample_rf(rng):
    from sklearn.ensemble import RandomForestClassifier
    n_estimators = int(rng.choice([10, 50, 100]))
    max_depth = rng.choice([None, 3, 5, 10, 20])
    if max_depth is not None:
        max_depth = int(max_depth)
    min_samples_leaf = int(rng.choice([1, 2, 5, 10, 20]))
    max_features = rng.choice(["sqrt", "log2", 0.5, 0.8])
    if isinstance(max_features, (float, np.floating)):
        max_features = float(max_features)
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features=max_features,
        n_jobs=1, random_state=int(rng.randint(1e6)))


def _sample_xgb(rng):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    learning_rate = float(rng.choice([0.01, 0.03, 0.1, 0.3]))
    max_depth = int(rng.choice([3, 4, 6, 8, 10]))
    n_estimators = int(rng.choice([50, 100, 200]))
    subsample = float(rng.choice([0.6, 0.8, 1.0]))
    colsample_bytree = float(rng.choice([0.6, 0.8, 1.0]))
    reg_lambda = float(rng.choice([0.1, 1.0, 5.0]))
    return XGBClassifier(
        learning_rate=learning_rate, max_depth=max_depth,
        n_estimators=n_estimators, subsample=subsample,
        colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
        eval_metric="logloss", verbosity=0,
        n_jobs=1, random_state=int(rng.randint(1e6)))


ALGO_SPECS = {
    "rf": _sample_rf,
    "xgb": _sample_xgb,
}

K_DEFAULT = 50
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3


def _make_temporal_order(X, seed=42):
    """Sort rows by first PC to create synthetic temporal ordering."""
    from sklearn.preprocessing import StandardScaler
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = StandardScaler().fit_transform(X_clean)
    if X_scaled.shape[1] >= 2:
        pc1 = PCA(n_components=1, random_state=seed).fit_transform(X_scaled)[:, 0]
    else:
        pc1 = X_scaled[:, 0]
    return np.argsort(pc1)


def _walk_forward_splits(n_total, n_folds=N_OUTER_FOLDS, min_test=30):
    """Generate expanding-window walk-forward splits.

    Returns list of (train_idx, test_idx) tuples.
    Each fold: train = [0, cutpoint), test = [cutpoint, cutpoint + test_size).
    Test size is fixed at n_total / (n_folds + 1).
    """
    test_size = n_total // (n_folds + 1)
    if test_size < min_test:
        return []
    folds = []
    for i in range(n_folds):
        train_end = test_size * (i + 1)
        test_end = train_end + test_size
        if test_end > n_total:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, min(test_end, n_total))
        if len(train_idx) >= 50 and len(test_idx) >= min_test:
            folds.append((train_idx, test_idx))
    return folds


def run_one_fold(X_train, y_train, X_test, y_test, algo_key, K, seed,
                 temporal=False):
    """Run one outer fold: inner CV (honest) vs test-peek (leaky).

    Args:
        temporal: if True, inner CV uses walk-forward splits (respects order).
                  if False, inner CV uses shuffled StratifiedKFold (i.i.d.).

    Returns dict with leakage, spread, or None if failed.
    """
    from sklearn.base import clone

    sampler = ALGO_SPECS[algo_key]

    # Ensure both classes in train and test
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    if len(X_train) < 100:  # need enough for inner CV + outer eval
        return None

    # Ensure enough minority class for inner CV
    if min(np.bincount(y_train.astype(int))) < N_INNER_FOLDS:
        return None

    # Generate K HP configs
    configs = []
    for k in range(K):
        hp_rng = np.random.RandomState(seed + k + 1)
        clf = sampler(hp_rng)
        if clf is None:
            return None
        configs.append(clf)

    # Build inner CV splits — temporal or i.i.d.
    if temporal:
        # Walk-forward inner splits (respects temporal order of train_outer)
        inner_splits = _walk_forward_splits(
            len(X_train), n_folds=N_INNER_FOLDS, min_test=20)
        if len(inner_splits) < 2:
            return None
    else:
        # Shuffled stratified (i.i.d. control)
        cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                             random_state=seed)
        inner_splits = list(cv.split(X_train, y_train))

    # Score each config: inner CV on train (honest) + test set (leaky)
    inner_cv_scores = []
    test_scores = []

    for clf in configs:
        try:
            # Inner CV score (honest HP selection)
            fold_aucs = []
            for tr_idx, va_idx in inner_splits:
                clf_inner = clone(clf)
                clf_inner.fit(X_train[tr_idx], y_train[tr_idx])
                proba_va = clf_inner.predict_proba(X_train[va_idx])[:, 1]
                auc_va = safe_auc(y_train[va_idx], proba_va)
                if auc_va != 0.5:
                    fold_aucs.append(auc_va)
            if not fold_aucs:
                continue
            inner_score = float(np.mean(fold_aucs))

            # Test score (leaky HP selection) — refit on full train, score test
            clf_fresh = clone(clf)
            clf_fresh.fit(X_train, y_train)
            proba = clf_fresh.predict_proba(X_test)[:, 1]
            test_score = safe_auc(y_test, proba)

            if test_score == 0.5:
                continue

            inner_cv_scores.append(inner_score)
            test_scores.append(test_score)
        except Exception:
            continue

    if len(test_scores) < 3:
        return None

    # Leaky: select by test score, report test score
    best_test_idx = int(np.argmax(test_scores))
    leaky_auc = test_scores[best_test_idx]

    # Honest: select by inner CV score, report test score
    best_cv_idx = int(np.argmax(inner_cv_scores))
    honest_auc = test_scores[best_cv_idx]

    return {
        "leakage": leaky_auc - honest_auc,
        "spread": max(test_scores) - float(np.mean(test_scores)),
        "honest_auc": honest_auc,
        "leaky_auc": leaky_auc,
        "k_effective": len(test_scores),
    }


def run_dataset(X, y, name, source, K=K_DEFAULT, seed=SEED):
    """Run temporal and i.i.d. conditions for one dataset."""
    result = {
        "name": name,
        "source": source,
        "n_rows": len(X),
        "n_features": X.shape[1],
        "v3_status": "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # --- TEMPORAL condition: sort by first PC, walk-forward ---
    temporal_order = _make_temporal_order(X, seed)
    X_temp = X[temporal_order]
    y_temp = y[temporal_order]

    temporal_folds = _walk_forward_splits(len(X_temp))
    if not temporal_folds:
        result["v3_status"] = "skip_too_small"
        return result

    for algo_key in ALGO_SPECS:
        fold_results = []
        for train_idx, test_idx in temporal_folds:
            fold_out = run_one_fold(
                X_temp[train_idx], y_temp[train_idx],
                X_temp[test_idx], y_temp[test_idx],
                algo_key, K, seed, temporal=True)
            if fold_out is not None:
                fold_results.append(fold_out)

        if fold_results:
            result[f"temporal_{algo_key}_leakage"] = float(
                np.mean([f["leakage"] for f in fold_results]))
            result[f"temporal_{algo_key}_spread"] = float(
                np.mean([f["spread"] for f in fold_results]))
            result[f"temporal_{algo_key}_n_folds"] = len(fold_results)
        else:
            result[f"temporal_{algo_key}_leakage"] = None
            result[f"temporal_{algo_key}_spread"] = None

    # --- IID condition: random stratified folds ---
    iid_folds = []
    skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                          random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        iid_folds.append((train_idx, test_idx))

    for algo_key in ALGO_SPECS:
        fold_results = []
        for train_idx, test_idx in iid_folds:
            fold_out = run_one_fold(
                X[train_idx], y[train_idx],
                X[test_idx], y[test_idx],
                algo_key, K, seed)
            if fold_out is not None:
                fold_results.append(fold_out)

        if fold_results:
            result[f"iid_{algo_key}_leakage"] = float(
                np.mean([f["leakage"] for f in fold_results]))
            result[f"iid_{algo_key}_spread"] = float(
                np.mean([f["spread"] for f in fold_results]))
            result[f"iid_{algo_key}_n_folds"] = len(fold_results)
        else:
            result[f"iid_{algo_key}_leakage"] = None
            result[f"iid_{algo_key}_spread"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Exp AT-T: Temporal HP tuning leakage")
    parser.add_argument("--out", default="v3_att.jsonl")
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--K", type=int, default=K_DEFAULT)
    args = parser.parse_args()

    an_path = Path("results/v3/v3_an.jsonl")
    if not an_path.exists():
        an_path = Path(__file__).parent / "../data/v3/v3_an.jsonl"

    with open(an_path) as f:
        an_rows = [json.loads(line) for line in f]
    datasets = [(r["name"], r.get("source", "openml")) for r in an_rows
                if r.get("v3_status") == "ok"
                and r.get("an_n_full") == 10000]  # ext datasets only

    print(f"Loaded {len(datasets)} ext datasets")

    if args.max_datasets > 0:
        datasets = datasets[:args.max_datasets]

    from joblib import Parallel, delayed

    def _process_one(name_source):
        name, source = name_source
        try:
            X, y = load_dataset(name, source)
            if X is None:
                return None, "error"
            if len(X) < args.min_rows:
                return None, "skip"
            return run_dataset(X, y, name, source, K=args.K), "ok"
        except Exception as e:
            print(f"  ERR {name}: {e}", file=sys.stderr, flush=True)
            return None, "error"

    n_parallel = max(1, os.cpu_count() // 4)
    print(f"Running with n_jobs={n_parallel}, K={args.K}")

    batch_size = n_parallel * 3
    done = errors = skipped = 0
    t0 = time.time()

    with open(args.out, "w") as fout:
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
    print(f"\nDone: {done}, {errors} errors, {skipped} skipped, "
          f"{elapsed:.0f}s ({elapsed/3600:.1f}h)")


if __name__ == "__main__":
    main()
