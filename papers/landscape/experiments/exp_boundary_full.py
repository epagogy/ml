#!/usr/bin/env python3
"""Experiment BOUNDARY-FULL: Temporal + group leakage on ALL eligible datasets.

No hand-picking. Discovers ALL cached OpenML datasets with temporal or group
columns, runs random CV vs boundary-aware CV on each, reports distribution.

Usage:
    python3 exp_boundary_full.py --out v3_boundary_full.jsonl
    python3 exp_boundary_full.py --out v3_boundary_full.jsonl --max-datasets 10
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


TEMPORAL_HINTS = ["date", "time", "timestamp", "year", "month", "hour",
                  "minute", "week", "day"]
GROUP_HINTS = ["id", "patient", "user", "subject", "account", "customer",
               "client", "household", "store", "region", "district",
               "session", "advertiser"]


def discover_datasets(cached_dir, min_rows=500):
    """Scan cached OpenML datasets for temporal and group columns."""
    import openml

    cached_ids = set()
    for f in os.listdir(cached_dir):
        if f.startswith("openml_") and f.endswith(".npz"):
            try:
                cached_ids.add(int(f.split("_")[1].split(".")[0]))
            except ValueError:
                pass

    datasets_meta = openml.datasets.list_datasets(output_format="dataframe")
    relevant = datasets_meta[
        datasets_meta.index.isin(cached_ids) &
        (datasets_meta["NumberOfInstances"] > min_rows)
    ]

    temporal = []
    group = []

    for did in relevant.nlargest(300, "NumberOfInstances").index:
        try:
            d = openml.datasets.get_dataset(did, download_data=False,
                                            download_qualities=False)
            feat_names = {f.name.lower(): f.name for f in d.features.values()}
            n = int(relevant.loc[did, "NumberOfInstances"])
            dname = relevant.loc[did, "name"]

            tcols = [v for k, v in feat_names.items()
                     if any(h in k for h in TEMPORAL_HINTS)]
            gcols = [v for k, v in feat_names.items()
                     if any(h in k for h in GROUP_HINTS)]

            if tcols:
                temporal.append((did, dname, n, tcols[0]))
            if gcols:
                # Filter out false positives: Bid_Open, Bid_High etc
                real_gcols = [c for c in gcols
                              if not any(x in c.lower()
                                        for x in ["bid", "ask", "open",
                                                   "high", "low", "close"])]
                if real_gcols:
                    group.append((did, dname, n, real_gcols[0]))
        except Exception:
            continue

    return temporal, group


def load_openml_df(did):
    """Load OpenML dataset as DataFrame with column names."""
    import openml
    try:
        ds = openml.datasets.get_dataset(did)
        df, y, _, _ = ds.get_data(target=ds.default_target_attribute,
                                  dataset_format="dataframe")
        return df, y
    except Exception:
        return None, None


def prepare_temporal(df, y, time_col, max_rows=20000):
    """Sort by time, convert to numeric X/y, subsample if needed."""
    y = np.array(y)
    if y.dtype == object or y.dtype == bool:
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y.astype(str))
    y = y.astype(float)
    valid = ~np.isnan(y)
    y, df = y[valid], df.loc[valid].reset_index(drop=True)

    unique_y = np.unique(y)
    if len(unique_y) > 2:
        y = (y > np.median(y)).astype(int)
    elif len(unique_y) == 2:
        y = (y == unique_y[1]).astype(int)
    else:
        return None, None

    # Parse time column
    if time_col in df.columns:
        if df[time_col].dtype == object:
            try:
                df[time_col] = pd.to_datetime(df[time_col],
                                               infer_datetime_format=True)
            except Exception:
                df[time_col] = pd.to_numeric(df[time_col], errors="coerce")

        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df = df.sort_values(time_col).reset_index(drop=True)
            y = y[df.index] if hasattr(y, 'iloc') else y
        elif pd.api.types.is_numeric_dtype(df[time_col]):
            df = df.sort_values(time_col).reset_index(drop=True)
            y = y[df.index] if hasattr(y, 'iloc') else y

        df = df.drop(columns=[time_col])

    # Numeric only
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > len(df) * 0.5:
                df[col] = converted

    X = df.select_dtypes(include=[np.number]).values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if X.shape[1] < 2 or X.shape[0] < 500:
        return None, None

    # Subsample (keep last max_rows, preserving temporal order)
    if len(X) > max_rows:
        X, y = X[-max_rows:], y[-max_rows:]

    return X, y.astype(int)


def run_boundary(X, y, n_folds=5):
    """Random CV vs walk-forward. Returns dict with deltas."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    from sklearn.base import clone

    n = len(X)
    step = n // (n_folds + 1)
    clf = Pipeline([("s", StandardScaler()),
                    ("m", LogisticRegression(max_iter=1000, random_state=42))])

    # Random CV
    try:
        rcv = cross_val_score(clf, X, y, scoring="roc_auc",
                              cv=StratifiedKFold(n_folds, shuffle=True,
                                                  random_state=42),
                              error_score=np.nan)
        rcv = rcv[~np.isnan(rcv)]
        auc_random = float(np.mean(rcv)) if len(rcv) > 0 else None
    except Exception:
        auc_random = None

    # Walk-forward
    wf = []
    for i in range(n_folds):
        tr_end = step * (i + 1)
        te_end = tr_end + step
        if te_end > n:
            break
        y_tr, y_te = y[:tr_end], y[tr_end:te_end]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        try:
            c = clone(clf)
            c.fit(X[:tr_end], y_tr)
            wf.append(roc_auc_score(y_te, c.predict_proba(X[tr_end:te_end])[:, 1]))
        except Exception:
            continue

    auc_wf = float(np.mean(wf)) if wf else None

    # Size-matched random (same fold sizes as walk-forward, shuffled rows)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    Xs, ys = X[idx], y[idx]
    smr = []
    for i in range(n_folds):
        tr_end = step * (i + 1)
        te_end = tr_end + step
        if te_end > n:
            break
        y_tr, y_te = ys[:tr_end], ys[tr_end:te_end]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        try:
            c = clone(clf)
            c.fit(Xs[:tr_end], y_tr)
            smr.append(roc_auc_score(y_te, c.predict_proba(Xs[tr_end:te_end])[:, 1]))
        except Exception:
            continue

    auc_smr = float(np.mean(smr)) if smr else None

    result = {}
    if auc_random is not None and auc_wf is not None:
        result["random_auc"] = auc_random
        result["walkfwd_auc"] = auc_wf
        result["total_delta"] = auc_random - auc_wf
    if auc_smr is not None and auc_wf is not None:
        result["size_matched_auc"] = auc_smr
        result["pure_temporal"] = auc_smr - auc_wf
        result["training_size"] = auc_random - auc_smr if auc_random else None
    result["n_wf_folds"] = len(wf)
    result["n_smr_folds"] = len(smr)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="v3_boundary_full.jsonl")
    parser.add_argument("--max-datasets", type=int, default=0)
    args = parser.parse_args()

    cache_dir = os.path.expanduser("~/.dataset_cache")
    print("Discovering datasets...", flush=True)
    temporal, group = discover_datasets(cache_dir)
    print(f"Found {len(temporal)} temporal, {len(group)} group", flush=True)

    if args.max_datasets > 0:
        temporal = temporal[:args.max_datasets]

    t0 = time.time()
    done = errors = 0

    with open(args.out, "w") as fout:
        for i, (did, name, n, tcol) in enumerate(temporal):
            print(f"  [{i+1}/{len(temporal)}] {name[:40]:40s}...",
                  end=" ", flush=True)
            try:
                df, y = load_openml_df(did)
                if df is None:
                    print("LOAD FAIL")
                    errors += 1
                    continue

                X, y_clean = prepare_temporal(df, y, tcol)
                if X is None:
                    print("PREP FAIL")
                    errors += 1
                    continue

                result = run_boundary(X, y_clean)
                if "total_delta" not in result:
                    print("NO RESULT")
                    errors += 1
                    continue

                result["name"] = name
                result["did"] = did
                result["type"] = "temporal"
                result["time_col"] = tcol
                result["n"] = len(X)
                result["n_features"] = X.shape[1]

                fout.write(json.dumps(result) + "\n")
                fout.flush()
                done += 1

                td = result.get("total_delta", 0)
                pt = result.get("pure_temporal", 0)
                print(f"n={len(X)} total={td:+.4f} temporal={pt:+.4f}"
                      if pt else f"n={len(X)} total={td:+.4f}")
            except Exception as e:
                print(f"ERR: {e}")
                errors += 1

    elapsed = time.time() - t0
    print(f"\nDone: {done} datasets, {errors} errors, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
