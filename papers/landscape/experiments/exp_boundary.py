#!/usr/bin/env python3
"""Experiment BOUNDARY: Real temporal + group leakage on real data.

NO SYNTHETIC DATA. Real timestamps, real group IDs.

Measures: optimism = random_cv_AUC - boundary_aware_cv_AUC
  This is the structural leakage that random CV HIDES.

TEMPORAL datasets (30+ FOREX minute bars, electricity, credit):
  Random CV vs walk-forward CV on the ACTUAL Timestamp column.

GROUP datasets (Click_prediction with ad_id/advertiser_id):
  Random CV vs GroupKFold on the ACTUAL group column.

Design per dataset:
  1. Load with column names from OpenML
  2. Identify time/group column
  3. Run RF + XGB with random 5-fold CV → AUC_random
  4. Run RF + XGB with boundary-aware CV → AUC_boundary
  5. Δ_structural = AUC_random - AUC_boundary (positive = random is optimistic)
  6. Flush per dataset

Usage:
    python3 exp_boundary.py --out v3_boundary.jsonl --max-datasets 5
    python3 exp_boundary.py --out v3_boundary.jsonl
"""

import argparse
import json
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Dataset registry: real temporal + group datasets

TEMPORAL_DATASETS = [
    # FOREX — 3 representative pairs (efficient market baseline)
    (41711, "FOREX_eurusd", "Timestamp"),
    (41717, "FOREX_usdjpy", "Timestamp"),
    (41795, "FOREX_gbpusd", "Timestamp"),
    # Real concept drift / non-stationary
    (44120, "electricity", None),  # NSW electricity, inherently time-ordered
    (1597, "creditcard", "Time"),  # fraud detection with Time column
    # OpenML datasets with real temporal structure
    (42933, "Beijing_Air_Quality", "hour"),     # 420K, PM2.5, 12 stations
    (1476, "Gas_Sensor_Drift", None),           # 14K, sensor drift 36 months (batch-ordered)
    ("uci:222", "Bank_Marketing", "month"),     # 45K, campaign temporal
    ("uci:357", "Occupancy_Detection", "date"), # 20K, room occupancy
    ("uci:560", "Seoul_Bike_Sharing", "Date"),  # 8.7K, hourly temporal
]

GROUP_DATASETS = [
    # Click prediction — real ad/advertiser groups
    (1219, "Click_prediction_small", "ad_id"),
    # UCI / OpenML datasets with real group structure
    ("uci:296", "Diabetes_130_Hospitals", "patient_nbr"),  # 101K, patient readmission
    (42933, "Beijing_Air_Quality_GROUP", "station"),        # 420K, 12 stations
]


def load_openml_with_columns(dataset_id):
    """Load dataset preserving column names. Handles OpenML IDs and UCI IDs."""
    import pandas as pd

    if isinstance(dataset_id, str) and dataset_id.startswith("uci:"):
        # UCI dataset via ucimlrepo
        uci_id = int(dataset_id.split(":")[1])
        try:
            from ucimlrepo import fetch_ucirepo
            ds = fetch_ucirepo(id=uci_id)
            df = ds.data.features
            y = ds.data.targets
            if y is not None and isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            return df, y
        except Exception as e:
            print(f"  Failed to load UCI {uci_id}: {e}", flush=True)
            return None, None
    else:
        # OpenML dataset
        import openml
        try:
            ds = openml.datasets.get_dataset(dataset_id)
            df, y_arr, _, _ = ds.get_data(
                target=ds.default_target_attribute,
                dataset_format="dataframe")
            return df, y_arr
        except Exception as e:
            print(f"  Failed to load OpenML {dataset_id}: {e}", flush=True)
            return None, None


def prepare_Xy(df, y, time_col=None, group_col=None):
    """Prepare X, y, and sort by time if temporal. Returns X, y, groups."""

    if y is None:
        return None, None, None

    # Convert y to binary int
    y = np.array(y)
    if y.dtype == object or y.dtype == bool:
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y.astype(str))
    y = y.astype(float)

    # Drop NaN targets BEFORE any processing
    valid_mask = ~np.isnan(y)
    y = y[valid_mask]
    df = df.loc[valid_mask].reset_index(drop=True)

    # Binarize if multiclass
    unique_y = np.unique(y)
    if len(unique_y) > 2:
        # Threshold at median for continuous or majority-class for categorical
        y = (y > np.median(y)).astype(int)
    elif len(unique_y) == 2:
        y = (y == unique_y[1]).astype(int)
    else:
        return None, None, None
    y = y.astype(int)

    groups = None
    if group_col and group_col in df.columns:
        groups = df[group_col].values
        df = df.drop(columns=[group_col])

    # Sort by time column if temporal
    if time_col and time_col in df.columns:
        import pandas as pd
        # Parse timestamps — handle all formats
        month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                      "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
        if df[time_col].dtype == object:
            first_val = str(df[time_col].iloc[0]).lower().strip()
            if first_val in month_map:
                df[time_col] = df[time_col].str.lower().str.strip().map(month_map)
            else:
                # Try datetime first (ISO, DD/MM/YYYY, etc.)
                try:
                    df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True)
                except Exception:
                    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        elif not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        y = y[df.index]
        df = df.sort_values(time_col).reset_index(drop=True)
        y = np.array(y)

        # Resample minute/sub-hourly data to hourly — real aggregation
        if len(df) > 20000:
            if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df["_hour"] = df[time_col].dt.floor("h")
            else:
                ts = df[time_col].values.astype(float)
                df["_hour"] = (ts - ts[0]) // 3600
            # For features: take mean per hour. For target: take mode per hour.
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols
                          if c != time_col and c != "_hour"]
            agg_dict = {c: "mean" for c in numeric_cols}
            df_hourly = df.groupby("_hour").agg(agg_dict).reset_index(drop=True)

            # Target: majority vote per hour
            df["_y"] = y
            y_hourly = df.groupby("_hour")["_y"].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            ).values

            df = df_hourly
            y = y_hourly
            if groups is not None:
                groups = None  # groups don't survive resampling

        df = df.drop(columns=[time_col], errors="ignore")
        df = df.drop(columns=["_hour", "_y"], errors="ignore")

    elif time_col is None and group_col is None:
        # Inherently ordered (electricity) — don't shuffle
        pass

    # Convert string columns: try numeric first, leave datetime/categorical alone
    import pandas as pd
    for col in df.columns:
        if df[col].dtype == object and col != time_col and col != group_col:
            numeric_converted = pd.to_numeric(df[col], errors="coerce")
            if numeric_converted.notna().sum() > len(df) * 0.5:
                df[col] = numeric_converted
            # else: leave as object, will be dropped by select_dtypes

    # Convert to numeric, drop non-numeric
    X = df.select_dtypes(include=[np.number]).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if X.shape[0] < 500 or X.shape[1] < 2:
        return None, None, None

    # Subsample large datasets to 20K for speed — preserving order
    MAX_ROWS = 20000
    if X.shape[0] > MAX_ROWS:
        # Take last MAX_ROWS rows (preserves temporal recency)
        X = X[-MAX_ROWS:]
        y = y[-MAX_ROWS:]
        if groups is not None:
            groups = groups[-MAX_ROWS:]

    return X, y, groups


def run_temporal(X, y, dataset_name, n_folds=5):
    """Compare random CV vs walk-forward CV on temporally-ordered data."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    n = len(X)
    result = {"name": dataset_name, "type": "temporal", "n": n,
              "n_features": X.shape[1]}

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    algos = {
        "lr": Pipeline([("s", StandardScaler()),
                        ("m", LogisticRegression(max_iter=1000,
                                                  random_state=42))]),
        "rf": RandomForestClassifier(n_estimators=50, n_jobs=1,
                                      random_state=42),
    }

    for algo_name, clf in algos.items():
        # Random CV (naive — ignores temporal order)
        random_cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                     random_state=42)
        try:
            random_scores = cross_val_score(clf, X, y, cv=random_cv,
                                            scoring="roc_auc",
                                            error_score=np.nan)
            random_scores = random_scores[~np.isnan(random_scores)]
            auc_random = float(np.mean(random_scores)) if len(random_scores) > 0 else None
        except Exception:
            auc_random = None

        # Walk-forward CV (respects temporal order)
        # Expanding window: train on [0, i*step), test on [i*step, (i+1)*step)
        step = n // (n_folds + 1)
        wf_scores = []
        for i in range(n_folds):
            train_end = step * (i + 1)
            test_end = train_end + step
            if test_end > n:
                break
            X_tr, y_tr = X[:train_end], y[:train_end]
            X_te, y_te = X[train_end:test_end], y[train_end:test_end]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            if len(X_tr) < 50:
                continue

            try:
                from sklearn.base import clone
                clf_fold = clone(clf)
                clf_fold.fit(X_tr, y_tr)
                proba = clf_fold.predict_proba(X_te)[:, 1]
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_te, proba)
                wf_scores.append(auc)
            except Exception:
                continue

        auc_temporal = float(np.mean(wf_scores)) if wf_scores else None

        if auc_random is not None and auc_temporal is not None:
            delta = auc_random - auc_temporal
            result[f"{algo_name}_random_auc"] = auc_random
            result[f"{algo_name}_temporal_auc"] = auc_temporal
            result[f"{algo_name}_delta"] = delta
            result[f"{algo_name}_n_wf_folds"] = len(wf_scores)

    return result


def run_group(X, y, groups, dataset_name, group_col_name, n_folds=5):
    """Compare random CV vs group CV on data with real group structure."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n = len(X)
    n_groups = len(np.unique(groups))
    actual_folds = min(n_folds, n_groups)

    if actual_folds < 2:
        return None

    result = {"name": dataset_name, "type": "group",
              "group_col": group_col_name,
              "n": n, "n_features": X.shape[1], "n_groups": n_groups}

    algos = {
        "lr": Pipeline([("s", StandardScaler()),
                        ("m", LogisticRegression(max_iter=1000,
                                                  random_state=42))]),
        "rf": RandomForestClassifier(n_estimators=50, n_jobs=1,
                                      random_state=42),
    }

    for algo_name, clf in algos.items():
        # Random CV (naive — scatters group members across folds)
        random_cv = StratifiedKFold(n_splits=actual_folds, shuffle=True,
                                     random_state=42)
        try:
            random_scores = cross_val_score(clf, X, y, cv=random_cv,
                                            scoring="roc_auc",
                                            error_score=np.nan)
            random_scores = random_scores[~np.isnan(random_scores)]
            auc_random = float(np.mean(random_scores)) if len(random_scores) > 0 else None
        except Exception:
            auc_random = None

        # Group CV (respects group boundaries)
        group_cv = GroupKFold(n_splits=actual_folds)
        try:
            group_scores = cross_val_score(clf, X, y, cv=group_cv,
                                           groups=groups,
                                           scoring="roc_auc",
                                           error_score=np.nan)
            group_scores = group_scores[~np.isnan(group_scores)]
            auc_group = float(np.mean(group_scores)) if len(group_scores) > 0 else None
        except Exception:
            auc_group = None

        if auc_random is not None and auc_group is not None:
            delta = auc_random - auc_group
            result[f"{algo_name}_random_auc"] = auc_random
            result[f"{algo_name}_group_auc"] = auc_group
            result[f"{algo_name}_delta"] = delta

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Exp BOUNDARY: Real temporal + group leakage")
    parser.add_argument("--out", default="v3_boundary.jsonl")
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--temporal-only", action="store_true")
    parser.add_argument("--group-only", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    done = errors = 0

    with open(args.out, "w") as fout:
        # TEMPORAL
        if not args.group_only:
            datasets = TEMPORAL_DATASETS
            if args.max_datasets > 0:
                datasets = datasets[:args.max_datasets]

            print(f"=== TEMPORAL: {len(datasets)} datasets ===", flush=True)
            for i, (did, name, time_col) in enumerate(datasets):
                print(f"  [{i+1}/{len(datasets)}] {name}...",
                      end=" ", flush=True)
                try:
                    df, y = load_openml_with_columns(did)
                    if df is None:
                        print("LOAD FAILED")
                        errors += 1
                        continue

                    X, y_clean, _ = prepare_Xy(df, y, time_col=time_col)
                    if X is None:
                        print("PREP FAILED")
                        errors += 1
                        continue

                    result = run_temporal(X, y_clean, name)
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                    done += 1

                    # Print result inline
                    deltas = [f"{k}={v:+.4f}" for k, v in result.items()
                              if k.endswith("_delta")]
                    print(f"n={X.shape[0]} {' '.join(deltas)}")
                except Exception as e:
                    print(f"ERR: {e}")
                    errors += 1

        # GROUP
        if not args.temporal_only:
            datasets = GROUP_DATASETS
            if args.max_datasets > 0:
                datasets = datasets[:args.max_datasets]

            print(f"\n=== GROUP: {len(datasets)} datasets ===", flush=True)
            for i, (did, name, group_col) in enumerate(datasets):
                print(f"  [{i+1}/{len(datasets)}] {name} (group={group_col})...",
                      end=" ", flush=True)
                try:
                    df, y = load_openml_with_columns(did)
                    if df is None:
                        print("LOAD FAILED")
                        errors += 1
                        continue

                    X, y_clean, groups = prepare_Xy(df, y,
                                                     group_col=group_col)
                    if X is None or groups is None:
                        print("PREP FAILED")
                        errors += 1
                        continue

                    result = run_group(X, y_clean, groups, name, group_col)
                    if result is not None:
                        fout.write(json.dumps(result) + "\n")
                        fout.flush()
                        done += 1
                        deltas = [f"{k}={v:+.4f}" for k, v in result.items()
                                  if k.endswith("_delta")]
                        print(f"n={X.shape[0]} groups={len(np.unique(groups))} "
                              f"{' '.join(deltas)}")
                    else:
                        print("TOO FEW GROUPS")
                        errors += 1
                except Exception as e:
                    print(f"ERR: {e}")
                    errors += 1

    elapsed = time.time() - t0
    print(f"\nDone: {done} datasets, {errors} errors, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
