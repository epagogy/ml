#!/usr/bin/env python3
"""V3 Experiments for the Leakage Landscape paper.

Three experiments that complete the paper's evidence base:

  AN: N-scaling — how leakage effects change with sample size
      Subsample each dataset at n=50,100,200,500,1000,2000.
      For each size: peeking, seed inflation, normalization, oversampling.
      Target: datasets with n >= 200 (expected ~866).

  AP: Seed dose-response — best-of-K seed inflation at K=5,10,25,50,100
      LR (deterministic baseline) + RF (stochastic).
      Target: all 2,047 datasets (expected ~1,363 with clean results).

  AO: CV coverage gap — actual vs nominal 95% CI coverage
      20 reps of 5-fold CV. Measure fraction of datasets where true test AUC
      falls within the CV-derived 95% CI (z-based and t-based).
      Target: all 2,047 datasets.

Reads V1 JSONL for dataset inventory. Writes separate V3 JSONL files.

Usage:
    python3 run_v3_experiments.py --v1 leakage_landscape_v1.jsonl --outdir results/v3/
    python3 run_v3_experiments.py --v1 ... --outdir ... --only an     # just AN
    python3 run_v3_experiments.py --v1 ... --outdir ... --only ap     # just AP
    python3 run_v3_experiments.py --v1 ... --outdir ... --only ao     # just AO
    python3 run_v3_experiments.py --v1 ... --outdir ... --limit 10    # first 10

Usage:
    nohup python3 run_v3_experiments.py \
        --v1 leakage_landscape_v1.jsonl --outdir results/v3/ \
        > v3_log.txt 2>&1 &
"""

import argparse
import json
import math
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
CV_FOLDS = 5
N_REPS_AO = 20        # repeated CV reps for coverage experiment
SUBSAMPLE_NS_MAIN = [50, 100, 200, 500, 1000, 2000]
SUBSAMPLE_NS_EXT  = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
SEED_KS = [5, 10, 25, 50, 100]
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


def fit_score(clf, X_tr, y_tr, X_te, y_te):
    try:
        clf.fit(X_tr, y_tr)
        return safe_auc(y_te, clf.predict_proba(X_te)[:, 1])
    except Exception:
        return 0.5


def make_lr(seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    return Pipeline([("s", StandardScaler()),
                     ("m", LogisticRegression(max_iter=1000, random_state=seed))])


def make_rf(seed, n_est=50):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=n_est, random_state=seed)


def make_dt(seed):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(random_state=seed)


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


# Dataset Loading

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
            # Try numpy cache first
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

            # Cache for future runs
            try:
                np.savez_compressed(cache_path, X=X, y=y,
                                    n_cat=np.array(0), nan_pct=np.array(0.0))
            except Exception:
                pass
        else:
            return None, None

        # Ensure binary
        X, y = prepare_binary(X, y)
        if X is None:
            return None, None

        # Drop NaN rows
        X = np.array(X, dtype=float)
        mask = ~np.isnan(X).any(axis=1)
        X, y = X[mask], y[mask]

        if len(X) < 50:
            return None, None
        return X, y

    except Exception:
        return None, None


# Experiment AN: N-Scaling

def exp_an(X, y, seed=SEED):
    """N-scaling experiment.

    Auto-selects n-levels based on dataset size:
      ≥10,000 rows → [50, 100, 200, 500, 1000, 2000, 5000, 10000]
      ≥200 rows    → [50, 100, 200, 500, 1000, 2000]

    Runs 4 leakage types (peeking, seed inflation, normalization, oversampling)
    and measures ΔAUC at each n-level.

    Returns dict with:
      an_n_levels:          list of n-levels tested
      an_n_full:            max n-level (dataset size bucket)
      an_peeking_means:     [N floats] mean ΔAUC at each n
      an_seed_means:        [N floats]
      an_normalize_means:   [N floats]
      an_oversample_means:  [N floats]
    """
    from sklearn.preprocessing import StandardScaler

    n_rows = len(y)
    if n_rows >= 10000:
        subsample_ns = SUBSAMPLE_NS_EXT
        n_full = 10000
    else:
        subsample_ns = [n for n in SUBSAMPLE_NS_MAIN if n <= n_rows]
        n_full = 2000

    peeking_means = []
    seed_means = []
    normalize_means = []
    oversample_means = []

    for target_n in subsample_ns:
        # --- Peeking (Class II): model selection, best-of-K on test ---
        # V1 Exp B protocol: try K configs, report max - random honest pick.
        # NOT MI feature selection (that's Exp C / Class I).
        # Baseline = random honest pick from same pool (not fixed LR).
        peek_diffs = []
        for rep in range(3):
            rs = seed + rep
            rng = np.random.RandomState(rs)
            if len(X) > target_n:
                idx = rng.choice(len(X), size=target_n, replace=False)
                Xs, ys = X[idx], y[idx]
            else:
                Xs, ys = X.copy(), y.copy()

            if len(Xs) < 30:
                continue

            try:
                from sklearn.model_selection import train_test_split
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import Pipeline
                X_tr, X_te, y_tr, y_te = train_test_split(
                    Xs, ys, test_size=0.2, random_state=rs, stratify=ys)

                # 5 configs: LR(C=1), LR(C=10), RF(50), DT, RF(100)
                configs = [
                    make_lr(rs),
                    Pipeline([("s", StandardScaler()),
                              ("m", LogisticRegression(max_iter=1000, C=10.0,
                                                       random_state=rs))]),
                    make_rf(rs),
                    make_dt(rs),
                    make_rf(rs, n_est=100),
                ]

                # Fit all configs, collect test AUCs
                test_aucs = []
                for clf in configs:
                    try:
                        auc = fit_score(clf, X_tr, y_tr, X_te, y_te)
                        test_aucs.append(auc)
                    except Exception:
                        test_aucs.append(0.5)

                if len(test_aucs) < 2:
                    continue

                # Leaky: pick best on test set
                leaky_auc = max(test_aucs)
                # Honest: random pick from same pool (V1 Exp B protocol)
                honest_auc = test_aucs[
                    np.random.RandomState(rs * 1000).randint(len(test_aucs))]
                peek_diffs.append(leaky_auc - honest_auc)
            except Exception:
                continue

        peeking_means.append(float(np.mean(peek_diffs)) if peek_diffs else None)

        # --- Seed inflation (Class II): best-of-10 vs single seed ---
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
                from sklearn.model_selection import train_test_split
                X_tr, X_te, y_tr, y_te = train_test_split(
                    Xs, ys, test_size=0.2, random_state=rs, stratify=ys)

                aucs = []
                for s in range(10):
                    auc = fit_score(make_rf(rs + s * 100), X_tr, y_tr, X_te, y_te)
                    aucs.append(auc)
                best = max(aucs)
                avg = np.mean(aucs)
                seed_diffs.append(best - avg)
            except Exception:
                continue

        seed_means.append(float(np.mean(seed_diffs)) if seed_diffs else None)

        # --- Normalization (Class I): global vs per-fold ---
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
                from sklearn.model_selection import train_test_split
                from sklearn.linear_model import LogisticRegression
                X_tr, X_te, y_tr, y_te = train_test_split(
                    Xs, ys, test_size=0.2, random_state=rs, stratify=ys)

                # Leaky: scale on all data
                sc = StandardScaler().fit(np.vstack([X_tr, X_te]))
                X_tr_g = sc.transform(X_tr)
                X_te_g = sc.transform(X_te)
                auc_leaky = fit_score(
                    LogisticRegression(max_iter=1000, random_state=rs),
                    X_tr_g, y_tr, X_te_g, y_te)

                # Clean: scale on train only
                sc2 = StandardScaler().fit(X_tr)
                X_tr_c = sc2.transform(X_tr)
                X_te_c = sc2.transform(X_te)
                auc_clean = fit_score(
                    LogisticRegression(max_iter=1000, random_state=rs),
                    X_tr_c, y_tr, X_te_c, y_te)

                norm_diffs.append(auc_leaky - auc_clean)
            except Exception:
                continue

        normalize_means.append(float(np.mean(norm_diffs)) if norm_diffs else None)

        # --- Oversampling (Class III): before vs after split ---
        over_diffs = []
        min_class_frac = min(np.mean(y == 0), np.mean(y == 1))
        if min_class_frac < 0.3:  # only meaningful for imbalanced
            for rep in range(3):
                rs = seed + rep
                rng = np.random.RandomState(rs)
                if len(X) > target_n:
                    idx = rng.choice(len(X), size=target_n, replace=False)
                    Xs, ys = X[idx], y[idx]
                else:
                    Xs, ys = X.copy(), y.copy()

                try:
                    from sklearn.model_selection import train_test_split
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        Xs, ys, test_size=0.2, random_state=rs, stratify=ys)

                    # Leaky: oversample before split (on all data)
                    minority = ys == np.argmin(np.bincount(ys.astype(int)))
                    n_dup = minority.sum()
                    dup_idx = rng.choice(np.where(minority)[0], size=n_dup, replace=True)
                    Xs_dup = np.vstack([Xs, Xs[dup_idx]])
                    ys_dup = np.concatenate([ys, ys[dup_idx]])
                    X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(
                        Xs_dup, ys_dup, test_size=0.2, random_state=rs, stratify=ys_dup)
                    auc_leaky = fit_score(make_rf(rs), X_tr_l, y_tr_l, X_te, y_te)

                    # Clean: oversample after split (train only)
                    min_tr = y_tr == np.argmin(np.bincount(y_tr.astype(int)))
                    n_dup_tr = min_tr.sum()
                    dup_tr = rng.choice(np.where(min_tr)[0], size=n_dup_tr, replace=True)
                    X_tr_c = np.vstack([X_tr, X_tr[dup_tr]])
                    y_tr_c = np.concatenate([y_tr, y_tr[dup_tr]])
                    auc_clean = fit_score(make_rf(rs), X_tr_c, y_tr_c, X_te, y_te)

                    over_diffs.append(auc_leaky - auc_clean)
                except Exception:
                    continue

        oversample_means.append(float(np.mean(over_diffs)) if over_diffs else None)

    return {
        "v3_status": "ok",
        "an_n_levels": subsample_ns,
        "an_n_full": n_full,
        "an_peeking_means": peeking_means,
        "an_seed_means": seed_means,
        "an_normalize_means": normalize_means,
        "an_oversample_means": oversample_means,
    }


# Experiment AP: Seed Dose-Response

def exp_ap(X, y, seed=SEED):
    """Seed inflation dose-response.

    For K in [5, 10, 25, 50, 100], compute:
      best-of-K AUC - mean-of-K AUC
    for both LR (should be ~0 since deterministic) and RF (stochastic).

    Returns dict with:
      ap_lr_inflation_k{K}: float for each K
      ap_rf_inflation_k{K}: float for each K
    """
    from sklearn.model_selection import train_test_split

    # Subsample large datasets to keep runtime bounded (same cap as V1)
    if len(X) > 10000:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=10000, replace=False)
        X, y = X[idx], y[idx]

    results = {"v3_status": "ok"}

    # Single split, fit ALL 100 seeds once, then compute best-of-K for each K
    # This is ~5x faster than fitting K seeds per K level
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
    except Exception:
        results["v3_status"] = "skip"
        return results

    max_k = max(SEED_KS)  # 100
    lr_aucs = []
    rf_aucs = []
    for s in range(max_k):
        rs = seed + s * 100
        lr_aucs.append(fit_score(make_lr(rs), X_tr, y_tr, X_te, y_te))
        rf_aucs.append(fit_score(make_rf(rs, n_est=30), X_tr, y_tr, X_te, y_te))

    lr_aucs = np.array(lr_aucs)
    rf_aucs = np.array(rf_aucs)

    # For each K, use 10 random draws of K seeds from the pool of 100
    rng = np.random.RandomState(seed + 999)
    for K in SEED_KS:
        lr_infl = []
        rf_infl = []
        n_draws = min(10, math.comb(max_k, K))  # 10 bootstrap samples
        for _ in range(n_draws):
            idx = rng.choice(max_k, size=K, replace=False)
            lr_infl.append(float(lr_aucs[idx].max() - lr_aucs[idx].mean()))
            rf_infl.append(float(rf_aucs[idx].max() - rf_aucs[idx].mean()))

        results[f"ap_lr_inflation_k{K}"] = round(float(np.mean(lr_infl)), 6)
        results[f"ap_rf_inflation_k{K}"] = round(float(np.mean(rf_infl)), 6)

    return results


# Experiment AO: CV Coverage Gap

def exp_ao(X, y, seed=SEED):
    """CV coverage experiment.

    Run 20 independent reps of 5-fold CV. Each rep uses a different random seed
    for the fold assignment. Compute mean + SD across reps.
    Then check: does the true test AUC fall within the 95% CI?

    Two CI types:
      z-based: mean +/- 1.96 * SD/sqrt(K)
      t-based: mean +/- t_{K-1, 0.025} * SD/sqrt(K)

    Returns dict with:
      ao_{algo}_coverage_z: fraction of reps where true AUC is in z-CI
      ao_{algo}_coverage_t: fraction of reps where true AUC is in t-CI
      ao_{algo}_n_valid: number of valid reps
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from scipy.stats import t as t_dist

    # Subsample large datasets (same cap as V1)
    if len(X) > 10000:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=10000, replace=False)
        X, y = X[idx], y[idx]

    results = {"v3_status": "ok"}

    try:
        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
    except Exception:
        results["v3_status"] = "skip"
        return results

    t_crit = t_dist.ppf(0.975, df=CV_FOLDS - 1)  # ~2.776 for K=5

    for algo_name, maker in [("lr", make_lr), ("rf", make_rf), ("dt", make_dt)]:
        # True test AUC (gold standard)
        try:
            if algo_name == "dt":
                clf = maker(seed)
            else:
                clf = maker(seed)
            clf.fit(X_dev, y_dev)
            true_auc = safe_auc(y_test, clf.predict_proba(X_test)[:, 1])
        except Exception:
            results[f"ao_{algo_name}_coverage_z"] = None
            results[f"ao_{algo_name}_coverage_t"] = None
            results[f"ao_{algo_name}_n_valid"] = 0
            continue

        z_hits = 0
        t_hits = 0
        n_valid = 0

        for rep in range(N_REPS_AO):
            rs = seed + rep * 7  # different fold splits
            skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=rs)

            fold_aucs = []
            for train_idx, val_idx in skf.split(X_dev, y_dev):
                X_tr, X_val = X_dev[train_idx], X_dev[val_idx]
                y_tr, y_val = y_dev[train_idx], y_dev[val_idx]

                try:
                    if algo_name == "dt":
                        clf_fold = maker(rs)
                    else:
                        clf_fold = maker(rs)
                    auc = fit_score(clf_fold, X_tr, y_tr, X_val, y_val)
                    fold_aucs.append(auc)
                except Exception:
                    fold_aucs.append(0.5)

            if len(fold_aucs) < CV_FOLDS:
                continue

            cv_mean = np.mean(fold_aucs)
            cv_sd = np.std(fold_aucs, ddof=1)
            se = cv_sd / math.sqrt(CV_FOLDS)

            # z-based CI
            z_lo = cv_mean - 1.96 * se
            z_hi = cv_mean + 1.96 * se
            if z_lo <= true_auc <= z_hi:
                z_hits += 1

            # t-based CI
            t_lo = cv_mean - t_crit * se
            t_hi = cv_mean + t_crit * se
            if t_lo <= true_auc <= t_hi:
                t_hits += 1

            n_valid += 1

        if n_valid > 0:
            results[f"ao_{algo_name}_coverage_z"] = round(z_hits / n_valid, 4)
            results[f"ao_{algo_name}_coverage_t"] = round(t_hits / n_valid, 4)
            results[f"ao_{algo_name}_n_valid"] = n_valid
        else:
            results[f"ao_{algo_name}_coverage_z"] = None
            results[f"ao_{algo_name}_coverage_t"] = None
            results[f"ao_{algo_name}_n_valid"] = 0

    return results


# Main

def load_v1_inventory(path):
    """Load V1 JSONL and return list of (name, source, n_rows) for ok datasets."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") == "ok":
                rows.append({
                    "name": r["name"],
                    "source": r["source"],
                    "n_rows": r.get("n_rows", 0),
                })
    return rows


def get_completed_keys(path):
    """Load completed dataset keys from output JSONL."""
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


def run_experiment(exp_name, datasets, exp_fn, outpath, filter_fn=None, limit=0):
    """Run one V3 experiment across datasets with resume support."""
    completed = get_completed_keys(outpath)
    todo = []
    for ds in datasets:
        key = ds["name"] + "|" + ds["source"]
        if key in completed:
            continue
        if filter_fn and not filter_fn(ds):
            continue
        todo.append(ds)

    if limit > 0:
        todo = todo[:limit]

    total = len(todo)
    already = len(completed)
    print(f"\n{'='*60}")
    print(f"Experiment {exp_name}: {total} to do, {already} already done")
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

        if (i + 1) % 10 == 0 or i == 0:
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

            result = exp_fn(X, y)
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
            if (i + 1) % 10 == 0:
                print(f"  ERR: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\n{exp_name} complete: {done} ok, {errors} errors, "
          f"{elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Total in {outpath}: {already + done}")
    return done, errors


def main():
    parser = argparse.ArgumentParser(description="V3 Experiments: AN + AP + AO")
    parser.add_argument("--v1", required=True, help="Path to V1 JSONL (dataset inventory)")
    parser.add_argument("--outdir", required=True, help="Output directory for V3 results")
    parser.add_argument("--only", choices=["an", "ap", "ao"], default=None,
                        help="Run only one experiment")
    parser.add_argument("--limit", type=int, default=0, help="Process only N datasets (0=all)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    datasets = load_v1_inventory(args.v1)
    print(f"Loaded {len(datasets)} ok datasets from V1")

    an_path = os.path.join(args.outdir, "v3_an.jsonl")
    ap_path = os.path.join(args.outdir, "v3_ap.jsonl")
    ao_path = os.path.join(args.outdir, "v3_ao.jsonl")

    # AN: datasets with n >= 200 (need enough rows for subsampling)
    # Extension datasets (n >= 10000) automatically get levels [50..10000]
    if args.only is None or args.only == "an":
        run_experiment(
            "AN (n-scaling)", datasets, exp_an, an_path,
            filter_fn=lambda ds: ds.get("n_rows", 0) >= 200,
            limit=args.limit,
        )

    # AP: all datasets (seed inflation)
    if args.only is None or args.only == "ap":
        run_experiment(
            "AP (seed dose-response)", datasets, exp_ap, ap_path,
            limit=args.limit,
        )

    # AO: all datasets (CV coverage)
    if args.only is None or args.only == "ao":
        run_experiment(
            "AO (CV coverage)", datasets, exp_ao, ao_path,
            limit=args.limit,
        )

    print(f"\nAll V3 experiments complete. Results in {args.outdir}/")


if __name__ == "__main__":
    main()
