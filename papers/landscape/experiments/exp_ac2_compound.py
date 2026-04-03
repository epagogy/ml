#!/usr/bin/env python3
"""Experiment AC2: Compound Class II selection (screen → tune → seed).

WHAT THIS MEASURES:
  The practitioner's full pipeline: screen algorithms, tune the winner's HPs,
  try seeds. All on the same data. How much total selection noise accumulates?

THEORY PREDICTS:
  Compound noise = σ · √(2·ln(K₁·K₂·K₃))  [sub-additive]
  For K₁=6 algos, K₂=50 HPs, K₃=10 seeds: effective K=3000
  Individual sum: σ·(g(6)+g(50)+g(10)) = 5.62σ
  Compound: σ·g(3000) = 4.01σ  [71% of sum, 29% sub-additive]

DESIGN:
  Two arms per dataset:
    LEAKY: screen→tune→seed, all selected by test-set score
    HONEST: screen→tune→seed, all selected by inner CV score

  Also measure each step individually for decomposition.

  Δ_compound = test_AUC[leaky_pipeline] - test_AUC[honest_pipeline]

DESIGN NOTES:
  [x] No impossible oracle — both arms are operationally feasible
  [x] Sequential pipeline: tune operates on screen winner, seed on tune winner
  [x] Inner CV uses StratifiedKFold (i.i.d. experiment)
  [x] n_jobs=1 on models, parallelism across datasets
  [x] Crash-safe batch output
  [x] XGBoost 3.2.0 compatible
  [x] safe_auc filter uses 'or' (skip if either is 0.5)

Usage:
    python3 exp_ac2_compound.py --out v3_ac2.jsonl --max-datasets 3  # smoke
    python3 exp_ac2_compound.py --out v3_ac2.jsonl                   # full
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_v3_experiments import (  # noqa: E402
    load_dataset, fit_score, SEED,
)


# Algorithm pool for screening

def _make_algo_pool(seed):
    """6 algorithms for screening (same as Exp B pool, simplified)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=100, eval_metric="logloss",
                            verbosity=0, n_jobs=1, random_state=seed)
    except ImportError:
        xgb = None

    pool = {
        "lr_c1": Pipeline([("s", StandardScaler()),
                           ("m", LogisticRegression(max_iter=1000, C=1.0,
                                                    random_state=seed))]),
        "lr_c10": Pipeline([("s", StandardScaler()),
                            ("m", LogisticRegression(max_iter=1000, C=10.0,
                                                     random_state=seed))]),
        "rf_50": RandomForestClassifier(n_estimators=50, n_jobs=1,
                                        random_state=seed),
        "rf_100": RandomForestClassifier(n_estimators=100, n_jobs=1,
                                         random_state=seed),
        "dt": DecisionTreeClassifier(random_state=seed),
    }
    if xgb is not None:
        pool["xgb"] = xgb
    return pool


# HP search space for tuning the screen winner

def _hp_configs_for_algo(algo_name, K, seed):
    """Generate K random HP configs for the given algorithm family."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    configs = []
    for k in range(K):
        rng = np.random.RandomState(seed + k + 1)

        if algo_name.startswith("lr"):
            C = float(rng.choice([0.001, 0.01, 0.1, 1.0, 10.0, 100.0]))
            penalty = rng.choice(["l1", "l2"])
            solver = "saga" if penalty == "l1" else "lbfgs"
            configs.append(Pipeline([
                ("s", StandardScaler()),
                ("m", LogisticRegression(C=C, penalty=penalty, solver=solver,
                                         max_iter=2000,
                                         random_state=int(rng.randint(1e6))))
            ]))
        elif algo_name.startswith("rf"):
            n_est = int(rng.choice([10, 50, 100]))
            max_depth = rng.choice([None, 3, 5, 10, 20])
            if max_depth is not None:
                max_depth = int(max_depth)
            min_leaf = int(rng.choice([1, 2, 5, 10, 20]))
            configs.append(RandomForestClassifier(
                n_estimators=n_est, max_depth=max_depth,
                min_samples_leaf=min_leaf, n_jobs=1,
                random_state=int(rng.randint(1e6))))
        elif algo_name == "dt":
            max_depth = rng.choice([None, 3, 5, 10, 20])
            if max_depth is not None:
                max_depth = int(max_depth)
            min_leaf = int(rng.choice([1, 2, 5, 10, 20]))
            configs.append(DecisionTreeClassifier(
                max_depth=max_depth, min_samples_leaf=min_leaf,
                random_state=int(rng.randint(1e6))))
        elif algo_name == "xgb":
            try:
                from xgboost import XGBClassifier
                lr = float(rng.choice([0.01, 0.03, 0.1, 0.3]))
                md = int(rng.choice([3, 4, 6, 8, 10]))
                n_est = int(rng.choice([50, 100, 200]))
                sub = float(rng.choice([0.6, 0.8, 1.0]))
                configs.append(XGBClassifier(
                    learning_rate=lr, max_depth=md, n_estimators=n_est,
                    subsample=sub, eval_metric="logloss", verbosity=0,
                    n_jobs=1, random_state=int(rng.randint(1e6))))
            except ImportError:
                pass

    return configs


# Core experiment

N_INNER_FOLDS = 3
K_SCREEN = 6   # number of algorithms to screen
K_TUNE = 50    # HP configs to try for winner
K_SEED = 10    # seeds to try for final config


def run_dataset(X, y, name, source, seed=SEED):
    """Run compound Class II experiment on one dataset.

    Returns result dict with compound, individual, and predicted values.
    """
    from sklearn.model_selection import train_test_split

    result = {
        "name": name,
        "source": source,
        "n_rows": len(X),
        "n_features": X.shape[1],
        "v3_status": "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if len(X) < 200:
        result["v3_status"] = "skip_too_small"
        return result

    # 80/20 train/test split (matching original experiments)
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
    except ValueError:
        result["v3_status"] = "skip_split_failed"
        return result

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        result["v3_status"] = "skip_single_class"
        return result

    # Inner CV setup
    min_class = min(np.bincount(y_tr.astype(int)))
    n_folds = min(N_INNER_FOLDS, min_class)
    if n_folds < 2:
        result["v3_status"] = "skip_low_minority"
        return result
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # STEP 1: SCREEN — evaluate algorithm pool
    algo_pool = _make_algo_pool(seed)

    screen_test = {}   # algo_name → test AUC
    screen_cv = {}     # algo_name → inner CV AUC

    for aname, clf in algo_pool.items():
        try:
            # Inner CV score
            cv_scores = cross_val_score(clf, X_tr, y_tr, cv=cv,
                                        scoring="roc_auc", error_score=np.nan)
            cv_mean = float(np.mean(cv_scores))

            # Test score
            clf_fresh = clone(clf)
            test_auc = fit_score(clf_fresh, X_tr, y_tr, X_te, y_te)

            if np.any(np.isnan(cv_scores)) or np.isnan(cv_mean) or test_auc == 0.5:
                continue

            screen_cv[aname] = cv_mean
            screen_test[aname] = test_auc
        except Exception:
            continue

    if len(screen_test) < 2:
        result["v3_status"] = "skip_screen_failed"
        return result

    # Leaky screen: pick best by test
    leaky_screen_winner = max(screen_test, key=screen_test.get)
    # Honest screen: pick best by CV
    honest_screen_winner = max(screen_cv, key=screen_cv.get)

    result["screen_leaky_winner"] = leaky_screen_winner
    result["screen_honest_winner"] = honest_screen_winner
    result["screen_leaky_auc"] = screen_test[leaky_screen_winner]
    result["screen_honest_auc"] = screen_test[honest_screen_winner]
    result["screen_delta"] = (screen_test[leaky_screen_winner]
                              - screen_test[honest_screen_winner])
    result["screen_k"] = len(screen_test)

    # STEP 2: TUNE — HP search for screen winner

    # NOTE: Both arms tune the SAME algorithm with the SAME
    # HP pool. Only the SELECTION CRITERION differs (CV vs test).
    # Using honest_screen_winner as the algorithm — conservative choice.
    # If we used leaky_screen_winner, the honest arm would be disadvantaged
    # by tuning an algorithm chosen by the leaky criterion.
    tune_algo = honest_screen_winner
    tune_configs = _hp_configs_for_algo(tune_algo, K_TUNE, seed)
    result["tune_algo"] = tune_algo

    def _score_configs(configs, label):
        """Score configs, return (cv_scores, test_scores, original_indices)."""
        cv_scores_list = []
        test_scores_list = []
        orig_indices = []
        for i, clf in enumerate(configs):
            try:
                cv_s = cross_val_score(clf, X_tr, y_tr, cv=cv,
                                       scoring="roc_auc", error_score=np.nan)
                cv_mean = float(np.mean(cv_s))
                clf_f = clone(clf)
                test_s = fit_score(clf_f, X_tr, y_tr, X_te, y_te)
                if np.any(np.isnan(cv_s)) or np.isnan(cv_mean) or test_s == 0.5:
                    continue
                cv_scores_list.append(cv_mean)
                test_scores_list.append(test_s)
                orig_indices.append(i)
            except Exception:
                continue
        return cv_scores_list, test_scores_list, orig_indices

    # NOTE: SAME configs scored once, selected by different criteria
    tune_cv, tune_test, tune_orig_idx = _score_configs(tune_configs, "tune")

    if len(tune_test) < 3:
        result["v3_status"] = "skip_tune_failed"
        return result

    # NOTE: argmax into filtered list, then map back to
    # original config list via orig_indices. Without this mapping, the wrong
    # classifier is retrieved for the seed step.
    leaky_tune_pos = int(np.argmax(tune_test))
    leaky_tune_auc = tune_test[leaky_tune_pos]
    leaky_tune_clf = tune_configs[tune_orig_idx[leaky_tune_pos]]

    honest_tune_pos = int(np.argmax(tune_cv))
    honest_tune_auc = tune_test[honest_tune_pos]
    honest_tune_clf = tune_configs[tune_orig_idx[honest_tune_pos]]

    result["tune_leaky_auc"] = leaky_tune_auc
    result["tune_honest_auc"] = honest_tune_auc
    result["tune_delta"] = leaky_tune_auc - honest_tune_auc
    result["tune_k"] = len(tune_test)

    # STEP 3: SEED — try K_SEED seeds for the tuned config

    def _seed_search(base_clf, K_seeds, seed_offset):
        cv_scores_list = []
        test_scores_list = []
        for s in range(K_seeds):
            try:
                clf_s = clone(base_clf)
                # Set random_state on the estimator (or pipeline's final step)
                rs = seed_offset + s * 100
                # NOTE: use set_params for reliable seed injection
                try:
                    if hasattr(clf_s, 'steps'):
                        clf_s.steps[-1][1].set_params(random_state=rs)
                    else:
                        clf_s.set_params(random_state=rs)
                except (ValueError, TypeError):
                    pass  # algorithm doesn't support random_state

                cv_s = cross_val_score(clf_s, X_tr, y_tr, cv=cv,
                                       scoring="roc_auc", error_score=np.nan)
                cv_mean = float(np.mean(cv_s))
                clf_f = clone(clf_s)
                test_s = fit_score(clf_f, X_tr, y_tr, X_te, y_te)
                if np.any(np.isnan(cv_s)) or np.isnan(cv_mean) or test_s == 0.5:
                    continue
                cv_scores_list.append(cv_mean)
                test_scores_list.append(test_s)
            except Exception:
                continue
        return cv_scores_list, test_scores_list

    # NOTE: same seed offset for both arms — differ only
    # in which config they got from the tune step, not in seed draws.
    leaky_seed_cv, leaky_seed_test = _seed_search(leaky_tune_clf, K_SEED, seed)
    honest_seed_cv, honest_seed_test = _seed_search(honest_tune_clf, K_SEED, seed)

    if len(leaky_seed_test) < 2 or len(honest_seed_test) < 2:
        result["v3_status"] = "skip_seed_failed"
        return result

    # Leaky seed: pick best by test
    leaky_seed_idx = int(np.argmax(leaky_seed_test))
    leaky_final_auc = leaky_seed_test[leaky_seed_idx]

    # Honest seed: pick best by CV
    honest_seed_idx = int(np.argmax(honest_seed_cv))
    honest_final_auc = honest_seed_test[honest_seed_idx]

    result["seed_leaky_auc"] = leaky_final_auc
    result["seed_honest_auc"] = honest_final_auc
    result["seed_delta"] = leaky_final_auc - honest_final_auc
    result["seed_k_leaky"] = len(leaky_seed_test)
    result["seed_k_honest"] = len(honest_seed_test)

    # COMPOUND METRICS

    result["compound_leaky_auc"] = leaky_final_auc
    result["compound_honest_auc"] = honest_final_auc
    result["compound_delta"] = leaky_final_auc - honest_final_auc
    result["sum_individual_deltas"] = (result["screen_delta"]
                                       + result["tune_delta"]
                                       + result["seed_delta"])

    # Theoretical prediction: σ · g(K_total) where K_total = K1·K2·K3
    # σ estimated from the seed step (pure noise)
    if len(honest_seed_test) >= 2:
        sigma_est = np.std(honest_seed_test, ddof=1)
        g_total = np.sqrt(2 * np.log(K_SCREEN * K_TUNE * K_SEED))
        result["predicted_compound_noise"] = float(sigma_est * g_total)
        result["sigma_est"] = float(sigma_est)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Exp AC2: Compound Class II (screen→tune→seed)")
    parser.add_argument("--out", default="v3_ac2.jsonl")
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--min-rows", type=int, default=500)
    args = parser.parse_args()

    an_path = Path("results/v3/v3_an.jsonl")
    if not an_path.exists():
        an_path = (Path(__file__).parent
                   / "../.."
                   "/research/results/v3/v3_an.jsonl")

    with open(an_path) as f:
        an_rows = [json.loads(line) for line in f]
    datasets = [(r["name"], r.get("source", "openml")) for r in an_rows
                if r.get("v3_status") == "ok"]

    print(f"Loaded {len(datasets)} datasets")

    if args.max_datasets > 0:
        datasets = datasets[:args.max_datasets]


    def _process_one(name_source):
        name, source = name_source
        try:
            X, y = load_dataset(name, source)
            if X is None:
                return None, "error"
            if len(X) < args.min_rows:
                return None, "skip"
            return run_dataset(X, y, name, source), "ok"
        except Exception as e:
            print(f"  ERR {name}: {e}", file=sys.stderr, flush=True)
            return None, "error"

    # Sequential per-dataset with immediate flush. Zero data loss.
    done = errors = skipped = 0
    t0 = time.time()

    with open(args.out, "w") as fout:
        for i, ds in enumerate(datasets):
            res, status = _process_one(ds)
            if status == "ok" and res is not None:
                fout.write(json.dumps(res) + "\n")
                fout.flush()
                done += 1
            elif status == "skip":
                skipped += 1
            else:
                errors += 1
            if (i + 1) % 5 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1) * 3600
                print(f"  [{i+1}/{len(datasets)}] "
                      f"done={done} err={errors} skip={skipped} "
                      f"{rate:.0f}/h", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {done}, {errors} errors, {skipped} skipped, "
          f"{elapsed:.0f}s ({elapsed/3600:.1f}h)")


if __name__ == "__main__":
    main()
