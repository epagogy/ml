#!/usr/bin/env python3
"""Experiment AT-T2: Temporal HP tuning leakage — the quant practitioner's failure mode.

WHAT THIS MEASURES:
  A practitioner runs K backtests on historical data, picks the best config,
  deploys it. The backtest said AUC=0.85, reality is AUC=0.72. The leak is
  multiple comparisons on the SAME historical period — not peeking at the future.

DESIGN (nested walk-forward):
  Outer loop: expanding-window walk-forward (5 folds)
    train_outer = all data before cutpoint t
    test_outer  = data in window [t, t+Δ) — the unseen future

    Inner loop (HP selection — what the practitioner actually does):
      TEMPORAL condition: walk-forward CV on train_outer
        Sub-split train_outer into expanding sub-windows
        For each of K HP configs: mean AUC across inner walk-forward folds
        Select config with best inner CV score (honest — no future leakage)

      IID condition: shuffled stratified CV on train_outer (control)
        For each of K HP configs: mean AUC across shuffled folds
        Select config with best inner CV score (also honest within i.i.d. frame)

    OPTIMISM = inner_cv_score[selected_config] - test_score[selected_config]
      This is what the practitioner THINKS minus REALITY.
      Positive = overconfident. This is the number that matters.

    Also measure: SELECTION BIAS = test[oracle_best] - test[cv_selected]
      How much better could you have done if you could see the future.

  KEY DIFFERENCE FROM exp_at_temporal.py (V1):
    V1 had a "leaky" arm that selected by TEST score — an impossible oracle.
    Nobody does that in practice. The real failure mode is:
      1. Run K backtests (all on historical data, no future peeking)
      2. Pick the best one
      3. Deploy it
      4. Reality disappoints because past CV is biased for the future

  PREDICTION:
    - IID: optimism ≈ σ√(2lnK)/√n → shrinks with n, well-understood
    - Temporal: optimism has ADDITIONAL component from distribution shift
      between train and test periods. Does NOT vanish with n because the
      bias is systematic (past ≠ future), not just noise.

  DESIGN NOTES (from 3 failed prior versions):
    [x] No impossible oracle — practitioner never sees test scores for selection
    [x] Inner CV is walk-forward in temporal condition (not shuffled!)
    [x] Outer splits are walk-forward (expanding window, never random)
    [x] IID control uses same data but random shuffle + stratified CV
    [x] Optimism = inner_best_score - test_score (not oracle comparison)
    [x] n_jobs=1 on models, parallelism across datasets only
    [x] Handles constant-variance columns (nan_to_num before scaling)
    [x] Handles class imbalance in small folds (min class count check)
    [x] XGBoost 3.2.0 compatible (eval_metric in constructor, no use_label_encoder)
    [x] Crash-safe: flush after each dataset batch

Usage:
    python3 exp_at_temporal_v2.py --out v3_att2.jsonl --max-datasets 3   # smoke
    python3 exp_at_temporal_v2.py --out v3_att2.jsonl                    # full run
    python3 exp_at_temporal_v2.py --out v3_att2.jsonl --genuine-only     # real temporal only
"""

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_v3_experiments import (  # noqa: E402
    load_dataset, safe_auc, SEED,
)


# GENUINE TEMPORAL DATASETS
# These have a real time column or are inherently ordered by time.
# PC1-sorted datasets are synthetic temporal (acknowledged proxy).

# NOTE: We maintain two tiers of temporal data:
#   Tier 1: Genuine temporal (has a date/time column or is intrinsically ordered)
#   Tier 2: PC1-sorted (covariate shift proxy, NOT true temporal)
# Both are valuable but results should be reported separately.

GENUINE_TEMPORAL = {
    # OpenML dataset ID: (name, time_column_or_None, description)
    # electricity: 45312 rows, NSW electricity market, class=UP/DOWN
    44120: ("electricity", None, "NSW electricity prices, inherently temporal"),
    # NOTE: electricity has no explicit timestamp column but rows ARE time-ordered.
    # The original dataset (Harries 1999) is a time series of 45K half-hourly readings.
}

# HP SEARCH SPACES — identical to exp_at_tune_nscaling.py for comparability


def _sample_lr(rng):
    """LogisticRegression with StandardScaler pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
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
    """KNN with StandardScaler pipeline. n_train caps n_neighbors."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
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
    """RandomForest — no scaler needed (tree-based)."""
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
    """XGBoost — eval_metric in constructor (3.2.0 compat), no use_label_encoder."""
    # NOTE: XGBoost 3.2.0 removed use_label_encoder param.
    # eval_metric MUST be in constructor, not fit(). Verified against CLAUDE.md hard-won lessons.
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
    "lr": _sample_lr,
    "knn": _sample_knn,
    "rf": _sample_rf,
    "xgb": _sample_xgb,
}

# EXPERIMENT PARAMETERS

K_BUDGETS = [10, 50, 100]
N_OUTER_FOLDS = 5       # walk-forward outer folds
N_INNER_FOLDS = 3       # inner CV folds (walk-forward or shuffled)
MIN_TRAIN_ROWS = 80     # minimum train rows for inner CV to be meaningful
MIN_TEST_ROWS = 30      # minimum test rows for AUC to be meaningful
MIN_MINORITY_CLASS = 5  # minimum minority class count per fold


# TEMPORAL ORDERING

def make_temporal_order_pc1(X, seed=42):
    """Sort rows by first principal component to create synthetic temporal ordering.

    NOTE: This is a covariate shift proxy, NOT real temporal structure.
    PC1 sorting creates a smooth gradient in feature space, which mimics
    distribution drift but does NOT capture concept drift (P(y|x) change).
    Results on PC1-sorted data should be labeled 'synthetic temporal'.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # NOTE: nan_to_num BEFORE scaling. StandardScaler on NaN = NaN propagation.
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # NOTE: Check for constant columns. StandardScaler sets them to 0
    # (mean-centered), but PCA on all-zero columns is fine (zero variance = zero loading).
    X_scaled = StandardScaler().fit_transform(X_clean)

    if X_scaled.shape[1] >= 2:
        pc1 = PCA(n_components=1, random_state=seed).fit_transform(X_scaled)[:, 0]
    else:
        pc1 = X_scaled[:, 0]
    return np.argsort(pc1)


# WALK-FORWARD SPLITS

def walk_forward_outer_splits(n_total, n_folds=N_OUTER_FOLDS,
                              min_test=MIN_TEST_ROWS):
    """Expanding-window walk-forward splits for outer evaluation.

    Layout for n_folds=5:
      test_size = n / (n_folds + 1)
      Fold 0: train=[0, 1*ts), test=[1*ts, 2*ts)
      Fold 1: train=[0, 2*ts), test=[2*ts, 3*ts)
      ...
      Fold 4: train=[0, 5*ts), test=[5*ts, 6*ts)

    NOTE: This is an expanding window (train always starts at 0).
    Rolling window would discard early data. Expanding is more realistic for
    the practitioner use case (they use ALL available history for backtesting).
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
        if len(train_idx) >= MIN_TRAIN_ROWS and len(test_idx) >= min_test:
            folds.append((train_idx, test_idx))
    return folds


def walk_forward_inner_splits(n_train, n_folds=N_INNER_FOLDS, min_test=20):
    """Walk-forward splits for inner HP evaluation on train_outer.

    NOTE: This is the CRITICAL correctness requirement.
    In the temporal condition, inner CV MUST also be walk-forward.
    If inner CV shuffles, we destroy the temporal structure and the
    inner CV score becomes a biased estimate of future performance
    in the WRONG direction — it would be LESS biased than reality,
    hiding the problem we're trying to measure.

    Layout (same expanding window as outer, but on train_outer only):
      Fold 0: train_inner=[0, 1*ts), val_inner=[1*ts, 2*ts)
      Fold 1: train_inner=[0, 2*ts), val_inner=[2*ts, 3*ts)
      ...
    """
    test_size = n_train // (n_folds + 1)
    if test_size < min_test:
        return []

    folds = []
    for i in range(n_folds):
        train_end = test_size * (i + 1)
        test_end = train_end + test_size
        if test_end > n_train:
            break
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, min(test_end, n_train))
        # Need enough minority class for fitting
        if len(train_idx) >= 30 and len(val_idx) >= min_test:
            folds.append((train_idx, val_idx))
    return folds


# CORE EXPERIMENT: ONE OUTER FOLD

def run_one_outer_fold(X_train, y_train, X_test, y_test,
                       algo_key, K, seed, temporal=False):
    """Run one outer fold: inner CV for HP selection, then evaluate on test.

    This is the practitioner's workflow:
      1. Generate K HP configs
      2. For each config, run inner CV on train_outer → get inner_cv_score
      3. Pick config with best inner_cv_score (honest selection)
      4. Refit that config on ALL of train_outer
      5. Evaluate on test_outer → get test_score
      6. OPTIMISM = inner_cv_score - test_score

    Args:
        X_train, y_train: outer training data (the practitioner's historical data)
        X_test, y_test: outer test data (the unseen future)
        algo_key: which algorithm family
        K: number of HP configs to try
        seed: random seed
        temporal: if True, inner CV uses walk-forward; if False, shuffled stratified

    Returns:
        dict with optimism, selection_bias, scores, or None if failed
    """
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold

    sampler = ALGO_SPECS[algo_key]

    # Guard: both classes must be present
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    if len(X_train) < MIN_TRAIN_ROWS:
        return None

    # NOTE: Check minority class count in train. If too few, stratified
    # CV will fail or produce degenerate folds.
    train_counts = np.bincount(y_train.astype(int))
    if len(train_counts) < 2 or min(train_counts) < MIN_MINORITY_CLASS:
        return None

    # Build inner CV splits
    if temporal:
        # NOTE: Walk-forward inner splits. The train data is already
        # temporally ordered (it's a prefix of the temporally-sorted full dataset).
        # So indices [0..n_train) preserve temporal order.
        inner_splits = walk_forward_inner_splits(len(X_train))
        if len(inner_splits) < 2:
            return None
        # NOTE: Verify inner splits respect temporal order.
        # Each split: train_inner is a prefix, val_inner is the next segment.
        # This means we never train on "future" data within inner CV. Correct.
    else:
        # NOTE: Shuffled stratified folds for i.i.d. control.
        # This is the standard practice for cross-sectional data.
        cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                             random_state=seed)
        try:
            inner_splits = list(cv.split(X_train, y_train))
        except ValueError:
            return None

    # Generate K HP configs
    configs = []
    for k in range(K):
        hp_rng = np.random.RandomState(seed + k + 1)
        if algo_key == "knn":
            clf = sampler(hp_rng, n_train=len(X_train) // 2)
        else:
            clf = sampler(hp_rng)
        if clf is None:
            return None
        configs.append(clf)

    # Score each config on inner CV
    # NOTE: For each config, we compute:
    #   inner_cv_score: mean AUC across inner CV folds (what practitioner sees)
    #   test_score: AUC on outer test set (what reality delivers)
    # The practitioner picks the config with best inner_cv_score.
    # They NEVER see test_score until deployment.

    inner_cv_scores = []   # what the practitioner thinks each config scores
    test_scores = []       # what each config actually scores on future data

    for clf in configs:
        try:
            # Inner CV score (the practitioner's backtest)
            fold_aucs = []
            for tr_idx, va_idx in inner_splits:
                clf_inner = clone(clf)

                # NOTE: Check both classes in this inner fold.
                # Walk-forward on small data can produce single-class folds.
                y_tr_inner = y_train[tr_idx]
                y_va_inner = y_train[va_idx]
                if len(np.unique(y_tr_inner)) < 2:
                    continue
                if len(np.unique(y_va_inner)) < 2:
                    continue

                clf_inner.fit(X_train[tr_idx], y_tr_inner)
                proba_va = clf_inner.predict_proba(X_train[va_idx])[:, 1]
                auc_va = safe_auc(y_va_inner, proba_va)
                if auc_va != 0.5:  # 0.5 = degenerate (safe_auc default)
                    fold_aucs.append(auc_va)

            if not fold_aucs:
                continue
            inner_score = float(np.mean(fold_aucs))

            # Test score (reality) — refit on ALL train_outer, score test
            # NOTE: We refit on the full train_outer because that's what
            # the practitioner does before deployment. They don't deploy one of
            # the inner CV models — they retrain on all available data.
            clf_fresh = clone(clf)
            clf_fresh.fit(X_train, y_train)
            proba_test = clf_fresh.predict_proba(X_test)[:, 1]
            test_score = safe_auc(y_test, proba_test)

            if test_score == 0.5:
                continue

            inner_cv_scores.append(inner_score)
            test_scores.append(test_score)

        except Exception:
            continue

    if len(test_scores) < 3:
        return None

    # Compute metrics

    # The practitioner selects by inner CV score (they can't see test scores)
    best_cv_idx = int(np.argmax(inner_cv_scores))

    # OPTIMISM: what practitioner thinks (inner CV) minus reality (test)
    # NOTE: This is the KEY metric. Positive = overconfident.
    # This is NOT an impossible oracle — the practitioner genuinely selects
    # by inner CV and then inner_cv_scores[best_cv_idx] is what they report
    # to their boss / paper / production system.
    optimism = inner_cv_scores[best_cv_idx] - test_scores[best_cv_idx]

    # SELECTION BIAS: how much better oracle selection would have been
    # This is informational — nobody can actually do this.
    best_test_idx = int(np.argmax(test_scores))
    selection_bias = test_scores[best_test_idx] - test_scores[best_cv_idx]

    # SPREAD: variance across HP configs on test set
    test_arr = np.array(test_scores)
    inner_arr = np.array(inner_cv_scores)

    return {
        "optimism": optimism,
        "selection_bias": selection_bias,
        "inner_cv_selected": inner_cv_scores[best_cv_idx],
        "test_selected": test_scores[best_cv_idx],
        "test_oracle": test_scores[best_test_idx],
        "test_mean": float(np.mean(test_arr)),
        "test_std": float(np.std(test_arr)),
        "inner_cv_mean": float(np.mean(inner_arr)),
        "inner_cv_std": float(np.std(inner_arr)),
        "inner_test_corr": float(np.corrcoef(inner_arr, test_arr)[0, 1])
            if len(test_arr) >= 3 else None,
        "k_effective": len(test_scores),
    }


# DATASET-LEVEL RUNNER

def run_dataset(X, y, name, source, K_budgets=None, seed=SEED,
                is_genuine_temporal=False):
    """Run both temporal and i.i.d. conditions for one dataset.

    For each K budget, for each outer fold:
      - Temporal condition: PC1-sorted (or genuine temporal), walk-forward outer + inner
      - IID condition: random shuffle, stratified outer CV, shuffled inner CV

    Returns a single result dict with all measurements.
    """
    if K_budgets is None:
        K_budgets = K_BUDGETS

    result = {
        "name": name,
        "source": source,
        "n_rows": len(X),
        "n_features": X.shape[1],
        "is_genuine_temporal": is_genuine_temporal,
        "K_budgets": K_budgets,
        "v3_status": "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # TEMPORAL CONDITION

    if is_genuine_temporal:
        # Data is already in temporal order (e.g., electricity dataset)
        X_ordered = X
        y_ordered = y
        result["temporal_ordering"] = "genuine"
    else:
        # Sort by PC1 as synthetic temporal proxy
        temporal_order = make_temporal_order_pc1(X, seed)
        X_ordered = X[temporal_order]
        y_ordered = y[temporal_order]
        result["temporal_ordering"] = "pc1"

    outer_folds_temporal = walk_forward_outer_splits(len(X_ordered))
    if not outer_folds_temporal:
        result["v3_status"] = "skip_too_small_temporal"
        return result

    result["n_outer_folds_temporal"] = len(outer_folds_temporal)

    for K in K_budgets:
        for algo_key in ALGO_SPECS:
            prefix = f"temporal_{algo_key}_k{K}"
            fold_results = []

            for fold_i, (train_idx, test_idx) in enumerate(outer_folds_temporal):
                # NOTE: train_idx is always [0, cutpoint) and test_idx
                # is [cutpoint, cutpoint+Δ). Because X_ordered is temporally
                # sorted, this is a genuine walk-forward split. train never
                # contains future data.
                fold_out = run_one_outer_fold(
                    X_ordered[train_idx], y_ordered[train_idx],
                    X_ordered[test_idx], y_ordered[test_idx],
                    algo_key, K, seed + fold_i * 100,
                    temporal=True,  # walk-forward inner CV
                )
                if fold_out is not None:
                    fold_results.append(fold_out)

            if fold_results:
                result[f"{prefix}_optimism_mean"] = float(
                    np.mean([f["optimism"] for f in fold_results]))
                result[f"{prefix}_optimism_std"] = float(
                    np.std([f["optimism"] for f in fold_results]))
                result[f"{prefix}_selection_bias_mean"] = float(
                    np.mean([f["selection_bias"] for f in fold_results]))
                result[f"{prefix}_inner_test_corr_mean"] = float(np.mean(
                    [f["inner_test_corr"] for f in fold_results
                     if f["inner_test_corr"] is not None]))
                result[f"{prefix}_test_mean"] = float(
                    np.mean([f["test_selected"] for f in fold_results]))
                result[f"{prefix}_inner_cv_mean"] = float(
                    np.mean([f["inner_cv_selected"] for f in fold_results]))
                result[f"{prefix}_n_folds"] = len(fold_results)
                # Store per-fold detail for later analysis
                result[f"{prefix}_folds"] = fold_results
            else:
                result[f"{prefix}_optimism_mean"] = None
                result[f"{prefix}_n_folds"] = 0

    # IID CONDITION (control)

    # NOTE: Use the ORIGINAL data (not PC1-sorted) with random
    # stratified outer folds. This is the i.i.d. baseline where we expect
    # optimism ≈ σ√(2lnK)/√n → vanishes at large n.
    from sklearn.model_selection import StratifiedKFold

    # NOTE: Why StratifiedKFold for outer? Because in the i.i.d. setting,
    # there's no temporal structure. Stratification ensures class balance across
    # folds. This is standard practice for cross-sectional data.
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                               random_state=seed)
    try:
        outer_folds_iid = list(outer_cv.split(X, y))
    except ValueError:
        result["v3_status"] = "skip_iid_cv_failed"
        return result

    result["n_outer_folds_iid"] = len(outer_folds_iid)

    for K in K_budgets:
        for algo_key in ALGO_SPECS:
            prefix = f"iid_{algo_key}_k{K}"
            fold_results = []

            for fold_i, (train_idx, test_idx) in enumerate(outer_folds_iid):
                fold_out = run_one_outer_fold(
                    X[train_idx], y[train_idx],
                    X[test_idx], y[test_idx],
                    algo_key, K, seed + fold_i * 100,
                    temporal=False,  # shuffled inner CV
                )
                if fold_out is not None:
                    fold_results.append(fold_out)

            if fold_results:
                result[f"{prefix}_optimism_mean"] = float(
                    np.mean([f["optimism"] for f in fold_results]))
                result[f"{prefix}_optimism_std"] = float(
                    np.std([f["optimism"] for f in fold_results]))
                result[f"{prefix}_selection_bias_mean"] = float(
                    np.mean([f["selection_bias"] for f in fold_results]))
                result[f"{prefix}_inner_test_corr_mean"] = float(np.mean(
                    [f["inner_test_corr"] for f in fold_results
                     if f["inner_test_corr"] is not None]))
                result[f"{prefix}_test_mean"] = float(
                    np.mean([f["test_selected"] for f in fold_results]))
                result[f"{prefix}_inner_cv_mean"] = float(
                    np.mean([f["inner_cv_selected"] for f in fold_results]))
                result[f"{prefix}_n_folds"] = len(fold_results)
                result[f"{prefix}_folds"] = fold_results
            else:
                result[f"{prefix}_optimism_mean"] = None
                result[f"{prefix}_n_folds"] = 0

    return result


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description="Exp AT-T2: Temporal HP tuning leakage (practitioner failure mode)")
    parser.add_argument("--out", default="v3_att2.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--max-datasets", type=int, default=0,
                        help="Limit datasets (0 = all)")
    parser.add_argument("--min-rows", type=int, default=500,
                        help="Minimum dataset rows (need enough for 5 walk-forward folds)")
    parser.add_argument("--K", type=int, default=0,
                        help="Single K budget (0 = all three: 10,50,100)")
    parser.add_argument("--genuine-only", action="store_true",
                        help="Only run genuine temporal datasets (skip PC1-sorted)")
    parser.add_argument("--ext-only", action="store_true",
                        help="Only run extension datasets (n>=10000)")
    args = parser.parse_args()

    K_budgets = [args.K] if args.K > 0 else K_BUDGETS

    # Load dataset list
    an_path = None
    for candidate in [
        Path("results/v3/v3_an.jsonl"),
        Path(__file__).parent / "results/v3/v3_an.jsonl",
        Path(__file__).parent / "../data/v3/v3_an.jsonl",
    ]:
        if candidate.exists():
            an_path = candidate
            break

    if an_path is None:
        print("ERROR: Cannot find v3_an.jsonl. Provide dataset list.", flush=True)
        sys.exit(1)

    with open(an_path) as f:
        an_rows = [json.loads(line) for line in f]

    # Build dataset list
    if args.genuine_only:
        # Only genuine temporal datasets
        datasets = []
        for did, (dname, _, desc) in GENUINE_TEMPORAL.items():
            datasets.append((f"openml_{did}", "openml"))
        print(f"Genuine temporal only: {len(datasets)} datasets")
    elif args.ext_only:
        datasets = [(r["name"], r.get("source", "openml")) for r in an_rows
                     if r.get("v3_status") == "ok"
                     and r.get("an_n_full") == 10000]
    else:
        datasets = [(r["name"], r.get("source", "openml")) for r in an_rows
                     if r.get("v3_status") == "ok"]

    # Also ensure genuine temporal datasets are included
    if not args.genuine_only:
        existing_names = {d[0] for d in datasets}
        for did, (dname, _, desc) in GENUINE_TEMPORAL.items():
            oname = f"openml_{did}"
            if oname not in existing_names:
                datasets.append((oname, "openml"))

    print(f"Loaded {len(datasets)} datasets from {an_path}")

    if args.max_datasets > 0:
        datasets = datasets[:args.max_datasets]
        print(f"Limited to {len(datasets)} datasets")

    print(f"K budgets: {K_budgets}")
    print(f"Outer folds: {N_OUTER_FOLDS}, Inner folds: {N_INNER_FOLDS}")

    # Identify genuine temporal datasets by OpenML ID
    genuine_ids = set()
    for did in GENUINE_TEMPORAL:
        genuine_ids.add(f"openml_{did}")

    # Process datasets with joblib

    def _process_one(name_source):
        name, source = name_source
        try:
            X, y = load_dataset(name, source)
            if X is None:
                return None, "error", name
            if len(X) < args.min_rows:
                return None, "skip", name

            is_genuine = name in genuine_ids
            result = run_dataset(
                X, y, name, source,
                K_budgets=K_budgets,
                seed=SEED,
                is_genuine_temporal=is_genuine,
            )
            return result, "ok", name
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  ERR {name}: {e}\n{tb}", file=sys.stderr, flush=True)
            return None, "error", name

    # Sequential per-dataset processing with immediate flush.
    # No joblib — each dataset writes to disk as soon as it completes.
    # Slower than parallel but zero data loss on crash or slow datasets.
    done = errors = skipped = 0
    t0 = time.time()
    out_path = Path(args.out)

    with open(out_path, "w") as fout:
        for i, ds in enumerate(datasets):
            result, status, dname = _process_one(ds)
            if status == "ok" and result is not None:
                fout.write(json.dumps(result, default=str) + "\n")
                fout.flush()
                done += 1
            elif status == "skip":
                skipped += 1
            else:
                errors += 1

            elapsed = time.time() - t0
            rate = done / max(elapsed, 1) * 3600
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  [{i+1}/{len(datasets)}] {dname[:40]:40s} "
                      f"done={done} err={errors} skip={skipped} "
                      f"elapsed={elapsed:.0f}s rate={rate:.0f}/h",
                      flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE: {done} datasets, {errors} errors, {skipped} skipped")
    print(f"Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Output: {out_path}")

    # Quick summary stats if we have results
    if out_path.exists() and done > 0:
        print(f"\n{'='*60}")
        print("QUICK SUMMARY (across all datasets):")
        rows = []
        with open(out_path) as f:
            for line in f:
                rows.append(json.loads(line))

        for condition in ["temporal", "iid"]:
            for algo in ALGO_SPECS:
                for K in K_budgets:
                    key = f"{condition}_{algo}_k{K}_optimism_mean"
                    vals = [r[key] for r in rows
                            if key in r and r[key] is not None]
                    if vals:
                        arr = np.array(vals)
                        print(f"  {condition:8s} {algo:3s} K={K:3d}: "
                              f"optimism={np.mean(arr):+.4f} "
                              f"(sd={np.std(arr):.4f}, n={len(arr)})")

        # The key comparison
        print("\n  PREDICTION: temporal optimism > iid optimism")
        print("  If temporal optimism is larger AND persists at large n,")
        print("  then distribution shift creates irreducible HP tuning leakage.")


if __name__ == "__main__":
    main()
