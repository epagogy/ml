"""
Algorithm Landscape Experiment — Eval/Assess Gap Measurement.

Measures CV optimism (eval-assess gap) across algorithm families,
engines (Rust/sklearn/XGBoost), and dataset properties on OpenML
CC18 (72 clf) + CTR23 (35 reg) benchmarks.

Usage:
    python bench_openml.py --suite cc18 --seeds 0-9 --workers 4
    python bench_openml.py --suite cc18 --seeds 0-2 --smoke   # 3 datasets, quick test
    python bench_openml.py --report                            # print summary
"""

import argparse
import hashlib
import json
import logging
import platform
import sqlite3
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

import ml

log = logging.getLogger("bench_openml")

BENCH_DIR = Path(__file__).parent
CACHE_DIR = BENCH_DIR / "cache"
DB_PATH = BENCH_DIR / "landscape_openml.db"

CELL_TIMEOUT = 300  # seconds per (algo, engine, dataset, seed)

# ---------------------------------------------------------------------------
# Dataclasses (from spec §2.4)
# ---------------------------------------------------------------------------

@dataclass
class PreparedFold:
    """Named output of prepare_fold() — typed intermediate state.
    Carries provenance: which algorithm's encoding rules produced this data."""
    X_train: np.ndarray
    X_valid: np.ndarray
    encoding: str              # 'onehot' or 'ordinal'
    feature_names: list[str]   # post-encoding column names
    prepare_time_ms: float = 0.0


@dataclass
class BenchmarkResult:
    """Typed sink-state for one (algorithm, engine, dataset, seed) cell."""
    eval_scores: list[float]   # per-fold primary metric
    eval_mean: float = 0.0
    eval_std: float = 0.0
    assess_score: float = 0.0
    gap: float = 0.0           # sign-normalized: positive = overfit
    algorithm: str = ""
    engine: str = ""
    dataset_id: int = 0
    dataset_name: str = ""
    seed: int = 0
    config_label: str = "default"
    config_json: str = "{}"
    # Per-fold detail metrics
    eval_metrics: dict = field(default_factory=dict)
    assess_metrics: dict = field(default_factory=dict)
    # Per-fold detail for eval_folds table
    fold_details: list[dict] = field(default_factory=list)
    # Timing
    prepare_time_ms: float = 0.0
    fit_time_ms: float = 0.0
    predict_time_ms: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Algorithm × Engine Grid (spec §3.1)
# ---------------------------------------------------------------------------

# Linear-family algorithms needing one-hot encoding + scaling
LINEAR_ALGOS = {"linear", "logistic", "elastic_net", "knn", "svm"}

# ROSC classification for engine comparison validity
# Algorithms with mathematically identical implementations across engines.
# Excluded: gradient_boosting (Newton-leaf vs MSE-residual, C1),
#           svm (L2-squared-hinge vs L1-hinge, different loss function).
SAME_ROSC_CELL = {
    "decision_tree", "random_forest", "extra_trees",
    "knn", "linear", "logistic", "elastic_net",
    "naive_bayes", "adaboost",
}

# Same algorithm family but different implementation.
# These get their own analysis track — not included in parity convergence.
SAME_FAMILY_DIFFERENT_IMPL = {
    "gradient_boosting",  # Newton-leaf (Rust) vs MSE-residual (sklearn HistGBT)
    "svm",                # L2-squared-hinge dual CD (Rust) vs L1-hinge C-SVM (sklearn SVC)
}

# Algorithm grid: (algorithm, engine, clf_only, reg_only)
ALGO_GRID = [
    ("decision_tree",      "ml",      False, False),
    ("decision_tree",      "sklearn", False, False),
    ("random_forest",      "ml",      False, False),
    ("random_forest",      "sklearn", False, False),
    ("extra_trees",        "ml",      False, False),
    ("extra_trees",        "sklearn", False, False),
    ("gradient_boosting",  "ml",      False, False),
    ("gradient_boosting",  "sklearn", False, False),
    ("xgboost",            "xgboost", False, False),
    ("adaboost",           "ml",      True,  False),
    ("adaboost",           "sklearn", True,  False),
    ("linear",             "ml",      False, True),
    ("linear",             "sklearn", False, True),
    ("logistic",           "ml",      True,  False),
    ("logistic",           "sklearn", True,  False),
    ("knn",                "ml",      False, False),
    ("knn",                "sklearn", False, False),
    ("elastic_net",        "ml",      False, True),
    ("elastic_net",        "sklearn", False, True),
    ("naive_bayes",        "ml",      True,  False),
    ("naive_bayes",        "sklearn", True,  False),
    ("svm",                "ml",      False, False),
    ("svm",                "sklearn", False, False),
]


def get_configs(algorithm: str) -> list[tuple[str, dict]]:
    """Return (config_label, hyperparams) pairs for an algorithm.
    Default config uses ml defaults. Tuned config for CART/Forest/GBT/Ridge only (C9 fix)."""

    # SVM: Rust is linear-only, so force kernel="linear" for fair engine comparison.
    # Without this, sklearn defaults to RBF — a different algorithm entirely (C1 fix).
    if algorithm == "svm":
        configs = [("default", {"kernel": "linear"})]
    else:
        configs = [("default", {})]

    # Tuned configs restricted to 4 representative families
    tuned = {
        "decision_tree": {"max_depth": 10, "min_samples_leaf": 5},
        "random_forest": {"n_estimators": 500, "max_depth": 15},
        "gradient_boosting": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6},
        "linear": {"alpha": 0.1},
    }
    if algorithm in tuned:
        configs.append(("tuned", tuned[algorithm]))

    return configs


def get_grid_for_task(task: str) -> list[tuple[str, str]]:
    """Return (algorithm, engine) pairs valid for given task."""
    result = []
    for algo, engine, clf_only, reg_only in ALGO_GRID:
        if task == "clf" and reg_only:
            continue
        if task == "reg" and clf_only:
            continue
        result.append((algo, engine))
    return result


# ---------------------------------------------------------------------------
# Dataset loading + caching (spec §2)
# ---------------------------------------------------------------------------

def fetch_suite(suite_id: int) -> list[int]:
    """Fetch OpenML suite dataset IDs."""
    import openml
    suite = openml.study.get_suite(suite_id)
    return list(suite.data)


def download_dataset(dataset_id: int) -> tuple[pd.DataFrame, str, str]:
    """Download OpenML dataset, return (df, target_name, task_type).
    Caches as parquet in CACHE_DIR."""
    import openml

    dataset_id = int(dataset_id)  # OpenML returns numpy.int64
    cache_path = CACHE_DIR / f"{dataset_id}.parquet"
    meta_path = CACHE_DIR / f"{dataset_id}.meta.json"

    if cache_path.exists() and meta_path.exists():
        df = pd.read_parquet(cache_path)
        meta = json.loads(meta_path.read_text())
        return df, meta["target"], meta["task"]

    ds = openml.datasets.get_dataset(
        dataset_id, download_data=True,
        download_qualities=False, download_features_meta_data=True,
    )
    X, y, categorical_indicator, attribute_names = ds.get_data(
        target=ds.default_target_attribute,
    )

    df = pd.DataFrame(X, columns=attribute_names)
    target = ds.default_target_attribute
    df[target] = y

    # Determine task type — category/object/bool dtype is always classification
    if y.dtype == object or y.dtype.name == "category" or y.dtype == bool:
        task = "clf"
    elif y.nunique() <= 20:
        task = "clf"
    else:
        task = "reg"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    meta_path.write_text(json.dumps({
        "target": target, "task": task,
        "dataset_id": dataset_id, "name": ds.name,
        "openml_version": ds.version,
    }))

    return df, target, task


def should_drop(df: pd.DataFrame, target: str, task: str) -> str | None:
    """Check if dataset should be dropped. Returns reason or None."""
    y = df[target].dropna()
    if len(y) < 100:
        return f"n={len(y)} < 100 after dropping missing target"
    if task == "clf" and y.nunique() < 2:
        return "single-class target"
    if task == "reg":
        try:
            if float(y.std()) < 1e-10:
                return "zero-variance target"
        except (TypeError, ValueError):
            return "non-numeric regression target"
    return None


def compute_sha256(path: Path) -> str:
    """SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Preprocessing — K_scope boundary (spec §2.4)
# ---------------------------------------------------------------------------

def global_preprocess(df: pd.DataFrame, target: str, task: str) -> tuple[np.ndarray, np.ndarray, list[str], list[bool]]:
    """Global structural preprocessing (before any split).
    Returns (X_raw, y_encoded, feature_names, is_categorical_mask)."""

    X = df.drop(columns=[target])
    y = df[target].values

    # Drop columns with >50% missing
    missing_pct = X.isnull().mean()
    keep = missing_pct[missing_pct <= 0.5].index.tolist()
    X = X[keep]

    # Encode target
    if task == "clf":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Identify categorical columns
    is_cat = []
    feature_names = list(X.columns)
    for col in X.columns:
        is_cat.append(X[col].dtype == object or X[col].dtype.name == "category")

    return X.values, y.astype(float), feature_names, is_cat


def prepare_fold(
    X_train_raw: np.ndarray,
    X_valid_raw: np.ndarray,
    feature_names: list[str],
    is_cat: list[bool],
    algorithm: str,
    task: str,
) -> PreparedFold:
    """Per-fold preprocessing. Fits on X_train_raw ONLY, transforms both.
    This is the K_scope boundary — NEVER sees X_valid during fit."""

    t0 = time.perf_counter()
    use_onehot = algorithm in LINEAR_ALGOS
    use_scaling = algorithm in LINEAR_ALGOS

    n_features = X_train_raw.shape[1]
    cat_idx = [i for i in range(n_features) if is_cat[i]]
    num_idx = [i for i in range(n_features) if not is_cat[i]]

    # --- Numeric: extract + cast to float + median imputation ---
    result_names = [feature_names[i] for i in num_idx]
    enc_label = "onehot" if use_onehot else "ordinal"

    if num_idx:
        num_tr = X_train_raw[:, num_idx].astype(float)
        num_va = X_valid_raw[:, num_idx].astype(float)

        for j in range(num_tr.shape[1]):
            col_train = num_tr[:, j]
            mask_tr = np.isnan(col_train)
            if mask_tr.any():
                median_val = np.nanmedian(col_train)
                if np.isnan(median_val):
                    median_val = 0.0
                col_train[mask_tr] = median_val
                col_valid = num_va[:, j]
                col_valid[np.isnan(col_valid)] = median_val
    else:
        num_tr = np.empty((X_train_raw.shape[0], 0))
        num_va = np.empty((X_valid_raw.shape[0], 0))

    # --- Categorical: one-hot (linear) or ordinal (trees) ---
    if cat_idx:
        cat_train = X_train_raw[:, cat_idx]
        cat_valid = X_valid_raw[:, cat_idx]

        # Handle NaN in categoricals — fill with "__missing__"
        cat_train = np.where(
            pd.isna(cat_train), "__missing__",
            cat_train.astype(str),
        )
        cat_valid = np.where(
            pd.isna(cat_valid), "__missing__",
            cat_valid.astype(str),
        )

        if use_onehot:
            enc = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                max_categories=20,
            )
            enc.fit(cat_train)
            cat_tr_encoded = enc.transform(cat_train)
            cat_va_encoded = enc.transform(cat_valid)
            cat_names = list(enc.get_feature_names_out(
                [feature_names[i] for i in cat_idx]
            ))
        else:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            enc.fit(cat_train)
            cat_tr_encoded = enc.transform(cat_train)
            cat_va_encoded = enc.transform(cat_valid)
            cat_names = [feature_names[i] for i in cat_idx]

        # Combine numeric + categorical
        X_tr_combined = np.hstack([num_tr, cat_tr_encoded])
        X_va_combined = np.hstack([num_va, cat_va_encoded])
        result_names = result_names + cat_names
    else:
        X_tr_combined = num_tr
        X_va_combined = num_va

    # --- Scaling for distance/gradient-based algorithms ---
    if use_scaling:
        scaler = StandardScaler()
        X_tr_combined = scaler.fit_transform(X_tr_combined)
        X_va_combined = scaler.transform(X_va_combined)

    # Final NaN cleanup (safety net)
    X_tr_combined = np.nan_to_num(X_tr_combined, nan=0.0)
    X_va_combined = np.nan_to_num(X_va_combined, nan=0.0)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PreparedFold(
        X_train=X_tr_combined,
        X_valid=X_va_combined,
        encoding=enc_label,
        feature_names=result_names,
        prepare_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Sign normalization (spec §4.3)
# ---------------------------------------------------------------------------

def sign_normalize(eval_mean: float, assess_score: float, metric: str) -> float:
    """Returns gap where positive = overfit for ALL metrics."""
    if metric in ("rmse", "mae"):
        # Lower-is-better: assess - eval (overfit = assess worse = higher)
        return assess_score - eval_mean
    else:
        # Higher-is-better: eval - assess (overfit = eval inflated)
        return eval_mean - assess_score


def primary_metric(task: str, n_classes: int) -> str:
    """Primary metric per spec §4.2."""
    if task == "clf":
        return "bal_accuracy" if n_classes > 2 else "auc"
    return "rmse"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    task: str,
    n_classes: int,
) -> dict[str, float]:
    """Compute all metrics for a prediction."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    metrics: dict[str, float] = {}

    if task == "clf":
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["bal_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        metrics["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # AUC — needs predict_proba
        if y_prob is not None:
            try:
                if n_classes == 2:
                    # Binary: use probability of positive class
                    prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                    metrics["auc"] = roc_auc_score(y_true, prob_pos)
                else:
                    metrics["auc"] = roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="weighted",
                    )
            except (ValueError, IndexError):
                metrics["auc"] = np.nan
        else:
            metrics["auc"] = np.nan
    else:
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["r2"] = r2_score(y_true, y_pred)
        metrics["mae"] = mean_absolute_error(y_true, y_pred)

    return metrics


# ---------------------------------------------------------------------------
# Core benchmark loop (spec §4.1)
# ---------------------------------------------------------------------------

def run_cell(
    dataset_id: int,
    algorithm: str,
    engine: str,
    config_label: str,
    config: dict,
    seed: int,
    suite: str,
) -> BenchmarkResult:
    """Run one (algorithm, engine, dataset, seed, config) cell.
    Returns a BenchmarkResult with eval + assess metrics."""

    result = BenchmarkResult(
        eval_scores=[],
        algorithm=algorithm,
        engine=engine,
        dataset_id=dataset_id,
        seed=seed,
        config_label=config_label,
        config_json=json.dumps(config, sort_keys=True),
    )

    try:
        # Disable provenance guards — benchmark manages its own splits
        ml.config(guards="off")

        # Load cached dataset
        df, target, task = download_dataset(dataset_id)
        result.dataset_name = df.attrs.get("name", str(dataset_id))

        drop_reason = should_drop(df, target, task)
        if drop_reason:
            result.error = f"dropped: {drop_reason}"
            return result

        # Global preprocessing
        X_raw, y, feature_names, is_cat = global_preprocess(df, target, task)
        n_classes = len(np.unique(y)) if task == "clf" else 0
        pm = primary_metric(task, n_classes)

        # Outer split: 80/20 (stratified for clf)
        stratify = y if task == "clf" else None
        inner_X, outer_X, inner_y, outer_y = train_test_split(
            X_raw, y, test_size=0.2, random_state=seed, stratify=stratify,
        )

        # CV splitter
        if task == "clf":
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        else:
            cv = KFold(n_splits=10, shuffle=True, random_state=seed)

        # --- Stage 1: 10-fold CV on inner (EVALUATE) ---
        fold_metrics_list = []
        total_prep_ms = 0.0
        total_fit_ms = 0.0
        total_pred_ms = 0.0

        for fold_i, (train_idx, valid_idx) in enumerate(cv.split(inner_X, inner_y)):
            # K_SCOPE BOUNDARY: prepare_fold fits ONLY on fold_train
            fold = prepare_fold(
                inner_X[train_idx], inner_X[valid_idx],
                feature_names, is_cat, algorithm, task,
            )
            fold_prep_ms = fold.prepare_time_ms
            total_prep_ms += fold_prep_ms

            # Build training DataFrame for ml.fit
            fold_train_df = pd.DataFrame(fold.X_train, columns=fold.feature_names)
            fold_train_df["__target__"] = inner_y[train_idx]

            # Fit — pass task= to prevent auto-detection on encoded targets
            t0 = time.perf_counter()
            fit_kwargs = dict(config)
            if engine != "xgboost":
                fit_kwargs["engine"] = engine
            ml_task = "classification" if task == "clf" else "regression"
            model = ml.fit(
                fold_train_df, "__target__",
                algorithm=algorithm, seed=seed,
                task=ml_task,
                **fit_kwargs,
            )
            fold_fit_ms = (time.perf_counter() - t0) * 1000
            total_fit_ms += fold_fit_ms

            # Predict
            fold_valid_df = pd.DataFrame(fold.X_valid, columns=fold.feature_names)
            t0 = time.perf_counter()
            preds = ml.predict(model, fold_valid_df)
            fold_pred_ms = (time.perf_counter() - t0) * 1000
            total_pred_ms += fold_pred_ms

            # Probabilities for AUC
            y_prob = None
            try:
                proba = model.predict_proba(fold_valid_df)
                y_prob = proba.values if hasattr(proba, "values") else np.asarray(proba)
            except Exception:
                pass

            fold_m = score_predictions(
                inner_y[valid_idx], preds.values, y_prob, task, n_classes,
            )
            fold_metrics_list.append(fold_m)

            # Store per-fold detail for eval_folds table
            result.fold_details.append({
                "fold": fold_i,
                "prepare_time_ms": fold_prep_ms,
                "fit_time_ms": fold_fit_ms,
                "predict_time_ms": fold_pred_ms,
                **fold_m,
            })

        # Aggregate eval metrics
        eval_metrics_agg = {}
        for key in fold_metrics_list[0]:
            vals = [fm[key] for fm in fold_metrics_list if not np.isnan(fm.get(key, np.nan))]
            if vals:
                eval_metrics_agg[f"eval_{key}_mean"] = float(np.mean(vals))
                eval_metrics_agg[f"eval_{key}_std"] = float(np.std(vals))
            else:
                eval_metrics_agg[f"eval_{key}_mean"] = np.nan
                eval_metrics_agg[f"eval_{key}_std"] = np.nan

        result.eval_metrics = eval_metrics_agg
        result.eval_scores = [
            fm.get(pm, np.nan) for fm in fold_metrics_list
        ]
        result.eval_mean = eval_metrics_agg.get(f"eval_{pm}_mean", np.nan)
        result.eval_std = eval_metrics_agg.get(f"eval_{pm}_std", np.nan)

        # --- Stage 2: Refit on full inner, predict outer (ASSESS) ---
        assess_fold = prepare_fold(
            inner_X, outer_X, feature_names, is_cat, algorithm, task,
        )

        assess_train_df = pd.DataFrame(assess_fold.X_train, columns=assess_fold.feature_names)
        assess_train_df["__target__"] = inner_y

        t0 = time.perf_counter()
        fit_kwargs = dict(config)
        if engine != "xgboost":
            fit_kwargs["engine"] = engine
        ml_task = "classification" if task == "clf" else "regression"
        final_model = ml.fit(
            assess_train_df, "__target__",
            algorithm=algorithm, seed=seed,
            task=ml_task,
            **fit_kwargs,
        )
        assess_fit_ms = (time.perf_counter() - t0) * 1000

        assess_valid_df = pd.DataFrame(assess_fold.X_valid, columns=assess_fold.feature_names)
        t0 = time.perf_counter()
        assess_preds = ml.predict(final_model, assess_valid_df)
        assess_pred_ms = (time.perf_counter() - t0) * 1000

        y_prob = None
        try:
            proba = final_model.predict_proba(assess_valid_df)
            y_prob = proba.values if hasattr(proba, "values") else np.asarray(proba)
        except Exception:
            pass

        assess_m = score_predictions(
            outer_y, assess_preds.values, y_prob, task, n_classes,
        )
        result.assess_metrics = {f"assess_{k}": v for k, v in assess_m.items()}
        result.assess_score = assess_m.get(pm, np.nan)

        # Gap (sign-normalized)
        result.gap = sign_normalize(result.eval_mean, result.assess_score, pm)

        # Timing
        result.prepare_time_ms = total_prep_ms + assess_fold.prepare_time_ms
        result.fit_time_ms = total_fit_ms + assess_fit_ms
        result.predict_time_ms = total_pred_ms + assess_pred_ms

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        log.error("Cell error: ds=%s algo=%s engine=%s seed=%d: %s",
                  dataset_id, algorithm, engine, seed, traceback.format_exc())

    return result


# ---------------------------------------------------------------------------
# Database (spec §6.2)
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_folds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    suite TEXT NOT NULL,
    dataset_id INTEGER NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_parquet_sha256 TEXT,
    n_samples INTEGER,
    n_features INTEGER,
    n_classes INTEGER,
    task TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    engine TEXT NOT NULL,
    config TEXT NOT NULL,
    config_label TEXT NOT NULL,
    seed INTEGER NOT NULL,
    fold INTEGER NOT NULL,
    accuracy REAL,
    bal_accuracy REAL,
    f1 REAL,
    auc REAL,
    rmse REAL,
    r2 REAL,
    mae REAL,
    prepare_time_ms REAL,
    fit_time_ms REAL,
    predict_time_ms REAL,
    error TEXT,
    mlw_version TEXT,
    hostname TEXT,
    timestamp TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS assess_holdout (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    suite TEXT NOT NULL,
    dataset_id INTEGER NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_parquet_sha256 TEXT,
    n_samples INTEGER,
    n_features INTEGER,
    n_classes INTEGER,
    task TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    engine TEXT NOT NULL,
    config TEXT NOT NULL,
    config_label TEXT NOT NULL,
    seed INTEGER NOT NULL,
    -- Eval summary (from CV)
    eval_accuracy_mean REAL,
    eval_accuracy_std REAL,
    eval_bal_accuracy_mean REAL,
    eval_auc_mean REAL,
    eval_auc_std REAL,
    eval_f1_mean REAL,
    eval_rmse_mean REAL,
    eval_r2_mean REAL,
    -- Assess metrics (held-out 20%)
    assess_accuracy REAL,
    assess_bal_accuracy REAL,
    assess_f1 REAL,
    assess_auc REAL,
    assess_rmse REAL,
    assess_r2 REAL,
    assess_mae REAL,
    -- The gap (positive = overfit for all metrics)
    gap_auc REAL,
    gap_bal_accuracy REAL,
    gap_accuracy REAL,
    gap_rmse REAL,
    gap_r2 REAL,
    -- Timing
    cv_total_time_ms REAL,
    cv_prepare_time_ms REAL,
    assess_fit_time_ms REAL,
    assess_predict_time_ms REAL,
    error TEXT,
    mlw_version TEXT,
    hostname TEXT,
    timestamp TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id INTEGER PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    suite TEXT NOT NULL,
    task TEXT NOT NULL,
    n_samples INTEGER,
    n_features INTEGER,
    n_classes INTEGER,
    n_numeric INTEGER,
    n_categorical INTEGER,
    n_missing_pct REAL,
    target_imbalance REAL,
    max_assess_primary REAL,
    learnable INTEGER,
    openml_version INTEGER,
    parquet_sha256 TEXT,
    source_url TEXT
);

CREATE TABLE IF NOT EXISTS source_manifests (
    manifest_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    track TEXT NOT NULL CHECK(track IN ('paper', 'product')),
    frozen_at TEXT,
    manifest_json TEXT NOT NULL,
    timestamp TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS convergence_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manifest_id TEXT NOT NULL,
    win_rate REAL NOT NULL,
    max_slowdown REAL NOT NULL,
    max_gap_ratio REAL NOT NULL,
    converged INTEGER NOT NULL,
    timestamp TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_eval_algo ON eval_folds(algorithm, engine, dataset_id);
CREATE INDEX IF NOT EXISTS idx_eval_ds ON eval_folds(dataset_id, algorithm);
CREATE INDEX IF NOT EXISTS idx_assess_algo ON assess_holdout(algorithm, engine, dataset_id);
CREATE INDEX IF NOT EXISTS idx_assess_gap ON assess_holdout(algorithm, gap_accuracy);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    return conn


def save_result(conn: sqlite3.Connection, result: BenchmarkResult,
                run_id: str, suite: str, hostname: str):
    """Save a BenchmarkResult to both eval_folds and assess_holdout."""

    cache_path = CACHE_DIR / f"{result.dataset_id}.parquet"
    parquet_sha = compute_sha256(cache_path) if cache_path.exists() else None

    # Get dataset info
    try:
        df, target, task = download_dataset(result.dataset_id)
        n_samples = len(df)
        n_features = df.shape[1] - 1
        y = df[target]
        n_classes = y.nunique() if task == "clf" else 0
    except Exception:
        n_samples = n_features = n_classes = 0
        task = "clf"

    # Save assess_holdout row
    em = result.eval_metrics
    am = result.assess_metrics

    conn.execute("""
        INSERT INTO assess_holdout (
            run_id, suite, dataset_id, dataset_name, dataset_parquet_sha256,
            n_samples, n_features, n_classes, task,
            algorithm, engine, config, config_label, seed,
            eval_accuracy_mean, eval_accuracy_std, eval_bal_accuracy_mean,
            eval_auc_mean, eval_auc_std, eval_f1_mean,
            eval_rmse_mean, eval_r2_mean,
            assess_accuracy, assess_bal_accuracy, assess_f1, assess_auc,
            assess_rmse, assess_r2, assess_mae,
            gap_auc, gap_bal_accuracy, gap_accuracy, gap_rmse, gap_r2,
            cv_total_time_ms, cv_prepare_time_ms,
            assess_fit_time_ms, assess_predict_time_ms,
            error, mlw_version, hostname
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?, ?
        )
    """, (
        run_id, suite, result.dataset_id, result.dataset_name, parquet_sha,
        n_samples, n_features, n_classes, task,
        result.algorithm, result.engine, result.config_json, result.config_label, result.seed,
        em.get("eval_accuracy_mean"), em.get("eval_accuracy_std"), em.get("eval_bal_accuracy_mean"),
        em.get("eval_auc_mean"), em.get("eval_auc_std"), em.get("eval_f1_mean"),
        em.get("eval_rmse_mean"), em.get("eval_r2_mean"),
        am.get("assess_accuracy"), am.get("assess_bal_accuracy"), am.get("assess_f1"), am.get("assess_auc"),
        am.get("assess_rmse"), am.get("assess_r2"), am.get("assess_mae"),
        _gap(em.get("eval_auc_mean"), am.get("assess_auc"), "auc"),
        _gap(em.get("eval_bal_accuracy_mean"), am.get("assess_bal_accuracy"), "bal_accuracy"),
        _gap(em.get("eval_accuracy_mean"), am.get("assess_accuracy"), "accuracy"),
        _gap(em.get("eval_rmse_mean"), am.get("assess_rmse"), "rmse"),
        _gap(em.get("eval_r2_mean"), am.get("assess_r2"), "r2"),
        result.fit_time_ms, result.prepare_time_ms,
        result.fit_time_ms, result.predict_time_ms,
        result.error, ml.__version__, hostname,
    ))
    # Save per-fold eval_folds rows
    for fd in result.fold_details:
        conn.execute("""
            INSERT INTO eval_folds (
                run_id, suite, dataset_id, dataset_name, dataset_parquet_sha256,
                n_samples, n_features, n_classes, task,
                algorithm, engine, config, config_label, seed, fold,
                accuracy, bal_accuracy, f1, auc, rmse, r2, mae,
                prepare_time_ms, fit_time_ms, predict_time_ms,
                error, mlw_version, hostname
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?
            )
        """, (
            run_id, suite, result.dataset_id, result.dataset_name, parquet_sha,
            n_samples, n_features, n_classes, task,
            result.algorithm, result.engine, result.config_json, result.config_label,
            result.seed, fd["fold"],
            fd.get("accuracy"), fd.get("bal_accuracy"), fd.get("f1"), fd.get("auc"),
            fd.get("rmse"), fd.get("r2"), fd.get("mae"),
            fd.get("prepare_time_ms"), fd.get("fit_time_ms"), fd.get("predict_time_ms"),
            result.error, ml.__version__, hostname,
        ))
    conn.commit()


def _gap(eval_val, assess_val, metric):
    """Compute sign-normalized gap, handling None."""
    if eval_val is None or assess_val is None:
        return None
    if np.isnan(eval_val) or np.isnan(assess_val):
        return None
    return sign_normalize(eval_val, assess_val, metric)


# ---------------------------------------------------------------------------
# Environment snapshot (spec §6.1)
# ---------------------------------------------------------------------------

def save_environment(bench_dir: Path):
    """Record environment for reproducibility."""
    env = {
        "uname": platform.uname()._asdict(),
        "python_version": sys.version,
        "mlw_version": ml.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # pip freeze
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30,
        )
        env["pip_freeze"] = result.stdout.strip().split("\n")
    except Exception:
        env["pip_freeze"] = []

    # git hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(bench_dir),
        )
        env["git_hash"] = result.stdout.strip()
    except Exception:
        env["git_hash"] = "unknown"

    out_path = bench_dir / "environment.json"
    out_path.write_text(json.dumps(env, indent=2, default=str))
    return env


# ---------------------------------------------------------------------------
# K_scope canary test (spec §6.3)
# ---------------------------------------------------------------------------

def canary_test(dataset_ids: list[int], n_check: int = 3) -> bool:
    """Leakage detection canary. Inject outlier into X_valid, verify
    prepare_fold() imputations differ between global-fit and fold-fit paths."""

    rng = np.random.RandomState(999)
    check_ids = [int(x) for x in rng.choice(dataset_ids, size=min(n_check, len(dataset_ids)), replace=False)]

    for did in check_ids:
        try:
            df, target, task = download_dataset(did)
            X_raw, y, feature_names, is_cat = global_preprocess(df, target, task)

            # Find a numeric column
            num_idx = [i for i in range(X_raw.shape[1]) if not is_cat[i]]
            if not num_idx:
                continue

            col = num_idx[0]
            n = X_raw.shape[0]
            split_at = int(n * 0.8)

            X_train = X_raw[:split_at].copy()
            X_valid = X_raw[split_at:].copy()

            # Extract numeric values for median computation (handles object arrays)
            train_col_vals = X_train[:, col].astype(float)
            fold_median = float(np.nanmedian(train_col_vals))

            # Inject NaN + outlier into valid
            X_valid_with_outlier = X_valid.copy()
            X_valid_with_outlier[0, col] = np.nan
            X_valid_with_outlier[-1, col] = 1e9

            valid_col_vals = X_valid_with_outlier[:, col].astype(float)
            all_vals = np.concatenate([train_col_vals, valid_col_vals])
            global_median = float(np.nanmedian(all_vals))

            # Run prepare_fold — should use fold_median, not global_median
            prepared = prepare_fold(
                X_train, X_valid_with_outlier,
                feature_names, is_cat, "decision_tree", task,
            )

            # First output column corresponds to first numeric column
            imputed_val = float(prepared.X_valid[0, 0])

            if abs(global_median - fold_median) > 1e-10:
                if abs(imputed_val - fold_median) < abs(imputed_val - global_median):
                    log.info("Canary OK for dataset %d: K_scope boundary intact", did)
                else:
                    log.error("CANARY FAIL dataset %d: prepare_fold may leak!", did)
                    return False

        except Exception as e:
            log.warning("Canary check skipped for dataset %d: %s", did, e)

    return True


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def parse_seeds(s: str) -> list[int]:
    """Parse '0-9' or '0,1,2' into list of ints."""
    if "-" in s:
        start, end = s.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in s.split(",")]


def run_benchmark(
    suite: str,
    seeds: list[int],
    workers: int = 4,
    smoke: bool = False,
    db_path: Path = DB_PATH,
):
    """Run the full benchmark."""
    import socket
    hostname = socket.gethostname()

    # Suite mapping
    suite_map = {"cc18": 99, "ctr23": 353}
    suite_id = suite_map.get(suite)
    if suite_id is None:
        print(f"Unknown suite: {suite}. Use cc18 or ctr23.")
        return

    task_type = "clf" if suite == "cc18" else "reg"

    # Fetch dataset IDs
    print(f"Fetching OpenML suite {suite} (ID={suite_id})...")
    dataset_ids = fetch_suite(suite_id)
    print(f"  {len(dataset_ids)} datasets in suite")

    if smoke:
        dataset_ids = dataset_ids[:3]
        seeds = seeds[:2]
        print(f"  SMOKE MODE: {len(dataset_ids)} datasets, {len(seeds)} seeds")

    # Environment snapshot
    env = save_environment(BENCH_DIR)
    git_hash = env.get("git_hash", "unknown")
    run_id = f"{git_hash}-{time.strftime('%Y%m%dT%H%M%S')}"
    print(f"  run_id: {run_id}")

    # Canary test
    print("Running K_scope canary test...")
    canary_test(dataset_ids)
    print("  Canary passed.")

    # Build work items
    grid = get_grid_for_task(task_type)
    work = []
    for did in dataset_ids:
        for algo, engine in grid:
            for config_label, config in get_configs(algo):
                for seed in seeds:
                    work.append((did, algo, engine, config_label, config, seed, suite))

    print(f"\nTotal cells: {len(work)} ({len(dataset_ids)} datasets × "
          f"{len(grid)} algo-engine × {len(seeds)} seeds)")

    # Init DB
    conn = init_db(db_path)

    # Run cells
    completed = 0
    errors = 0
    t_start = time.perf_counter()

    # Sequential for now — multiprocessing needs pickleable functions
    for did, algo, engine, config_label, config, seed, suite_name in work:
        completed += 1
        if completed % 50 == 0 or completed == 1:
            elapsed = time.perf_counter() - t_start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(work) - completed) / rate if rate > 0 else 0
            print(f"  [{completed}/{len(work)}] {rate:.1f} cells/s, "
                  f"ETA {eta/60:.0f}min, errors={errors}")

        result = run_cell(did, algo, engine, config_label, config, seed, suite_name)
        if result.error:
            errors += 1
            if completed <= 10 or errors <= 5:
                log.warning("  ERROR: ds=%d %s/%s seed=%d: %s",
                           did, algo, engine, seed, result.error[:80])

        save_result(conn, result, run_id, suite_name, hostname)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {completed} cells in {elapsed/60:.1f}min, {errors} errors")
    print(f"Results in: {db_path}")

    conn.close()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(db_path: Path):
    """Print gap analysis summary from existing DB."""
    if not db_path.exists():
        print(f"No database at {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) FROM assess_holdout").fetchone()[0]
    errors = conn.execute(
        "SELECT COUNT(*) FROM assess_holdout WHERE error IS NOT NULL"
    ).fetchone()[0]
    print(f"\nLandscape DB: {total} cells, {errors} errors")

    # Gap summary by algorithm
    print(f"\n{'='*80}")
    print("EVAL-ASSESS GAP BY ALGORITHM (mean ± std across datasets, per-dataset averaged over seeds)")
    print(f"{'='*80}")

    rows = conn.execute("""
        SELECT algorithm, engine, task,
               AVG(gap_auc) as mean_gap_auc,
               AVG(gap_bal_accuracy) as mean_gap_bal,
               AVG(gap_accuracy) as mean_gap_acc,
               AVG(gap_rmse) as mean_gap_rmse,
               COUNT(*) as n
        FROM assess_holdout
        WHERE error IS NULL
        GROUP BY algorithm, engine, task
        ORDER BY algorithm, engine
    """).fetchall()

    print(f"{'Algorithm':20s} {'Engine':8s} {'Task':4s} {'Gap AUC':>10s} {'Gap BalAcc':>10s} "
          f"{'Gap Acc':>10s} {'Gap RMSE':>10s} {'N':>5s}")
    print("-" * 80)
    for r in rows:
        gap_auc = f"{r['mean_gap_auc']:.4f}" if r['mean_gap_auc'] is not None else "—"
        gap_bal = f"{r['mean_gap_bal']:.4f}" if r['mean_gap_bal'] is not None else "—"
        gap_acc = f"{r['mean_gap_acc']:.4f}" if r['mean_gap_acc'] is not None else "—"
        gap_rmse = f"{r['mean_gap_rmse']:.4f}" if r['mean_gap_rmse'] is not None else "—"
        print(f"{r['algorithm']:20s} {r['engine']:8s} {r['task']:4s} "
              f"{gap_auc:>10s} {gap_bal:>10s} {gap_acc:>10s} {gap_rmse:>10s} {r['n']:>5d}")

    conn.close()


# ---------------------------------------------------------------------------
# Source Hash Registry (spec §6.4)
# ---------------------------------------------------------------------------

# Paths relative to the al/ directory
RUST_SOURCE_FILES = {
    "rust_cart": "core/src/cart.rs",
    "rust_forest": "core/src/forest.rs",
    "rust_gbt": "core/src/gbt.rs",
    "rust_linear": "core/src/linear.rs",
    "rust_logistic": "core/src/logistic.rs",
    "rust_knn": "core/src/knn.rs",
    "rust_elastic_net": "core/src/elastic_net.rs",
    "rust_naive_bayes": "core/src/naive_bayes.rs",
    "rust_adaboost": "core/src/adaboost.rs",
    "rust_svm": "core/src/svm.rs",
}

# Which Rust source files map to which algorithm
ALGO_TO_RUST_KEY = {
    "decision_tree": "rust_cart",
    "random_forest": "rust_forest",
    "extra_trees": "rust_forest",  # same source
    "gradient_boosting": "rust_gbt",
    "linear": "rust_linear",
    "logistic": "rust_logistic",
    "knn": "rust_knn",
    "elastic_net": "rust_elastic_net",
    "naive_bayes": "rust_naive_bayes",
    "adaboost": "rust_adaboost",
    "svm": "rust_svm",
}


def compute_source_manifest(al_dir: Path | None = None) -> dict:
    """Compute SHA-256 hashes of all source files that affect results."""
    if al_dir is None:
        al_dir = BENCH_DIR.parent  # benchmarks/../ = al/

    manifest = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # Rust algorithm sources
    for key, rel_path in RUST_SOURCE_FILES.items():
        src_path = al_dir / rel_path
        if src_path.exists():
            manifest[key] = compute_sha256(src_path)
        else:
            manifest[key] = "NOT_FOUND"

    # External engine versions
    try:
        import sklearn
        manifest["sklearn_version"] = sklearn.__version__
    except ImportError:
        manifest["sklearn_version"] = "NOT_INSTALLED"

    try:
        import xgboost
        manifest["xgboost_version"] = xgboost.__version__
    except ImportError:
        manifest["xgboost_version"] = "NOT_INSTALLED"

    manifest["numpy_version"] = np.__version__
    manifest["mlw_version"] = ml.__version__

    # Benchmark infrastructure
    bench_file = BENCH_DIR / "bench_openml.py"
    if bench_file.exists():
        manifest["bench_openml"] = compute_sha256(bench_file)

    # Git hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(al_dir),
        )
        manifest["git_hash"] = result.stdout.strip()
    except Exception:
        manifest["git_hash"] = "unknown"

    return manifest


def manifest_id(manifest: dict) -> str:
    """Deterministic ID from manifest content (hash of sorted JSON)."""
    content = json.dumps(manifest, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def save_manifest(conn: sqlite3.Connection, manifest: dict,
                  run_id: str, track: str = "product"):
    """Save source manifest to DB."""
    mid = manifest_id(manifest)
    conn.execute("""
        INSERT OR IGNORE INTO source_manifests
            (manifest_id, run_id, track, manifest_json)
        VALUES (?, ?, ?, ?)
    """, (mid, run_id, track, json.dumps(manifest, sort_keys=True)))
    conn.commit()
    return mid


def detect_stale_results(conn: sqlite3.Connection, manifest: dict) -> list[tuple[str, str]]:
    """Find (algorithm, engine) pairs whose source hash changed.
    Returns list of (algorithm, engine) that need re-running."""
    stale = []

    # Get the latest manifest from DB
    row = conn.execute("""
        SELECT manifest_json FROM source_manifests
        WHERE track = 'product'
        ORDER BY timestamp DESC LIMIT 1
    """).fetchone()

    if row is None:
        return []  # First run — nothing is stale

    old_manifest = json.loads(row[0])

    # Check each Rust algorithm
    for algo, rust_key in ALGO_TO_RUST_KEY.items():
        old_hash = old_manifest.get(rust_key, "")
        new_hash = manifest.get(rust_key, "")
        if old_hash and new_hash and old_hash != new_hash:
            stale.append((algo, "ml"))

    # Check sklearn version
    if old_manifest.get("sklearn_version") != manifest.get("sklearn_version"):
        for algo, engine, _, _ in ALGO_GRID:
            if engine == "sklearn":
                stale.append((algo, engine))

    # Check xgboost version
    if old_manifest.get("xgboost_version") != manifest.get("xgboost_version"):
        stale.append(("xgboost", "xgboost"))

    return list(set(stale))


# ---------------------------------------------------------------------------
# State Machine (spec §6.5) — result lifecycle
# ---------------------------------------------------------------------------

VALID_STATES = {"running", "current", "stale", "frozen", "error"}


def mark_stale(conn: sqlite3.Connection, algorithm: str, engine: str):
    """Mark all PRODUCT-track CURRENT results for (algo, engine) as STALE."""
    for table in ("eval_folds", "assess_holdout"):
        conn.execute(f"""
            UPDATE {table}
            SET status = 'stale'
            WHERE algorithm = ? AND engine = ?
              AND status = 'current'
        """, (algorithm, engine))
    conn.commit()


def freeze_manifest(conn: sqlite3.Connection, mid: str):
    """Freeze a manifest — all its results become paper-track FROZEN."""
    conn.execute("""
        UPDATE source_manifests SET frozen_at = datetime('now')
        WHERE manifest_id = ?
    """, (mid,))
    # Note: eval_folds/assess_holdout don't have manifest_id/track/status columns
    # in Phase 1 schema. These will be added when living benchmark is active.
    conn.commit()


# ---------------------------------------------------------------------------
# Statistical Analysis (spec §5)
# ---------------------------------------------------------------------------

def run_statistical_analysis(db_path: Path):
    """Full statistical analysis pipeline per spec §5."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    results = {}

    # --- §5.2: Friedman + Nemenyi (algorithm families) ---
    for task in ("clf", "reg"):
        family_gaps = _get_family_gap_matrix(conn, task)
        if family_gaps is not None and len(family_gaps) > 2:
            friedman = _friedman_nemenyi(family_gaps, task)
            results[f"friedman_{task}"] = friedman

    # --- §5.3: Bayesian signed-rank (within-algorithm engine comparison) ---
    engine_results = _bayesian_engine_comparison(conn)
    results["engine_comparison"] = engine_results

    # --- §5.3b: Wilcoxon signed-rank (secondary) ---
    wilcoxon_results = _wilcoxon_engine_comparison(conn)
    results["wilcoxon"] = wilcoxon_results

    # Save
    out_path = BENCH_DIR / "stats_analysis.json"
    out_path.write_text(json.dumps(results, indent=2, default=_json_default))
    print(f"\nStatistical analysis saved to {out_path}")

    # Print summary
    _print_stats_summary(results)

    conn.close()
    return results


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _get_family_gap_matrix(conn: sqlite3.Connection, task: str) -> pd.DataFrame | None:
    """Get per-dataset gap matrix: rows=datasets, columns=algorithm families.
    Uses per-dataset averages across seeds (§5.1)."""

    metric_col = "gap_auc" if task == "clf" else "gap_rmse"

    rows = conn.execute(f"""
        SELECT dataset_id, algorithm, AVG({metric_col}) as mean_gap
        FROM assess_holdout
        WHERE error IS NULL AND task = ? AND {metric_col} IS NOT NULL
        GROUP BY dataset_id, algorithm
    """, (task,)).fetchall()

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["dataset_id", "algorithm", "mean_gap"])
    pivot = df.pivot(index="dataset_id", columns="algorithm", values="mean_gap")
    pivot = pivot.dropna(axis=1, how="all").dropna(axis=0)
    return pivot


def _friedman_nemenyi(gap_matrix: pd.DataFrame, task: str) -> dict:
    """Friedman test + Nemenyi post-hoc on algorithm family rankings."""
    from scipy.stats import friedmanchisquare

    algorithms = list(gap_matrix.columns)
    k = len(algorithms)
    n = len(gap_matrix)

    # Rank within each dataset (lower gap = better = lower rank)
    ranks = gap_matrix.rank(axis=1, method="average")
    mean_ranks = ranks.mean()

    # Friedman test
    groups = [gap_matrix[col].values for col in algorithms]
    try:
        stat, p_value = friedmanchisquare(*groups)
    except Exception:
        return {"error": "Friedman test failed", "k": k, "n": n}

    result = {
        "task": task,
        "k": k,
        "n": n,
        "friedman_stat": float(stat),
        "friedman_p": float(p_value),
        "mean_ranks": {algo: float(mr) for algo, mr in mean_ranks.items()},
        "rank_order": list(mean_ranks.sort_values().index),
    }

    # Nemenyi CD if significant
    if p_value < 0.05:
        try:
            import scikit_posthocs as sp
            nemenyi = sp.posthoc_nemenyi_friedman(gap_matrix.values)
            nemenyi.index = algorithms
            nemenyi.columns = algorithms
            # CD approximation: q_α × sqrt(k(k+1) / 6N)
            from scipy.stats import studentized_range
            q_alpha = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
            cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
            result["nemenyi_cd"] = float(cd)
            result["nemenyi_pvalues"] = {
                f"{algorithms[i]}_vs_{algorithms[j]}": float(nemenyi.iloc[i, j])
                for i in range(k) for j in range(i + 1, k)
            }
        except ImportError:
            result["nemenyi_error"] = "scikit-posthocs not installed"
        except Exception as e:
            result["nemenyi_error"] = str(e)

    return result


def _bayesian_engine_comparison(conn: sqlite3.Connection) -> dict:
    """Bayesian signed-rank test for within-ROSC-cell engine comparison (§5.3a)."""
    results = {}

    for algo in SAME_ROSC_CELL:
        for task in ("clf", "reg"):
            metric_col = "gap_auc" if task == "clf" else "gap_rmse"

            # Get per-dataset averages for ml vs sklearn
            rows = conn.execute(f"""
                SELECT a.dataset_id,
                       AVG(CASE WHEN a.engine='ml' THEN a.{metric_col} END) as ml_gap,
                       AVG(CASE WHEN a.engine='sklearn' THEN a.{metric_col} END) as sk_gap
                FROM assess_holdout a
                WHERE a.algorithm = ? AND a.task = ? AND a.error IS NULL
                  AND a.{metric_col} IS NOT NULL
                GROUP BY a.dataset_id
                HAVING ml_gap IS NOT NULL AND sk_gap IS NOT NULL
            """, (algo, task)).fetchall()

            if len(rows) < 5:
                continue

            ml_gaps = np.array([r[1] for r in rows])
            sk_gaps = np.array([r[2] for r in rows])

            # ROPE: ±0.005 AUC (clf) or ±0.01 normalized RMSE (reg)
            rope = 0.005 if task == "clf" else 0.01

            try:
                import baycomp
                p_left, p_rope, p_right = baycomp.two_on_single(
                    ml_gaps, sk_gaps, rope=rope,
                )
                results[f"{algo}_{task}"] = {
                    "algorithm": algo,
                    "task": task,
                    "n_datasets": len(rows),
                    "p_ml_wins": float(p_left),
                    "p_rope": float(p_rope),
                    "p_sklearn_wins": float(p_right),
                    "rope": rope,
                    "interpretation": (
                        "ml_superior" if p_left > 0.95
                        else "equivalent" if p_rope > 0.95
                        else "sklearn_superior" if p_right > 0.95
                        else "inconclusive"
                    ),
                    "ml_gap_mean": float(np.mean(ml_gaps)),
                    "sk_gap_mean": float(np.mean(sk_gaps)),
                }
            except ImportError:
                results[f"{algo}_{task}"] = {
                    "error": "baycomp not installed",
                    "algorithm": algo,
                    "task": task,
                }
            except Exception as e:
                results[f"{algo}_{task}"] = {
                    "error": str(e),
                    "algorithm": algo,
                    "task": task,
                }

    return results


def _wilcoxon_engine_comparison(conn: sqlite3.Connection) -> dict:
    """Wilcoxon signed-rank tests for same-ROSC-cell engine pairs (§5.3b)."""
    from scipy.stats import wilcoxon

    results = {}

    for algo in SAME_ROSC_CELL:
        for task in ("clf", "reg"):
            metric_col = "gap_auc" if task == "clf" else "gap_rmse"

            rows = conn.execute(f"""
                SELECT a.dataset_id,
                       AVG(CASE WHEN a.engine='ml' THEN a.{metric_col} END) as ml_gap,
                       AVG(CASE WHEN a.engine='sklearn' THEN a.{metric_col} END) as sk_gap
                FROM assess_holdout a
                WHERE a.algorithm = ? AND a.task = ? AND a.error IS NULL
                  AND a.{metric_col} IS NOT NULL
                GROUP BY a.dataset_id
                HAVING ml_gap IS NOT NULL AND sk_gap IS NOT NULL
            """, (algo, task)).fetchall()

            if len(rows) < 5:
                continue

            ml_gaps = np.array([r[1] for r in rows])
            sk_gaps = np.array([r[2] for r in rows])
            diffs = ml_gaps - sk_gaps

            wins = int(np.sum(diffs < 0))  # ml gap smaller = ml wins
            losses = int(np.sum(diffs > 0))
            ties = int(np.sum(diffs == 0))

            try:
                stat, p_value = wilcoxon(ml_gaps, sk_gaps, alternative="two-sided")
                # Rank-biserial correlation
                n = len(diffs)
                r_rb = 1 - (2 * stat) / (n * (n + 1) / 2) if n > 0 else 0

                results[f"{algo}_{task}"] = {
                    "algorithm": algo,
                    "task": task,
                    "n_datasets": len(rows),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p_value),
                    "rank_biserial_r": float(r_rb),
                    "wins_ml": wins,
                    "losses_ml": losses,
                    "ties": ties,
                }
            except Exception as e:
                results[f"{algo}_{task}"] = {
                    "error": str(e),
                    "algorithm": algo,
                    "task": task,
                }

    return results


def _print_stats_summary(results: dict):
    """Print human-readable statistical analysis summary."""

    # Friedman
    for key in ("friedman_clf", "friedman_reg"):
        if key not in results:
            continue
        r = results[key]
        task = r.get("task", key.split("_")[1])
        print(f"\n{'='*70}")
        print(f"FRIEDMAN TEST — {task.upper()} (k={r['k']}, N={r['n']})")
        print(f"{'='*70}")
        print(f"  χ² = {r['friedman_stat']:.2f}, p = {r['friedman_p']:.6f}")
        if r["friedman_p"] < 0.05:
            print("  → SIGNIFICANT: algorithm families differ")
            if "nemenyi_cd" in r:
                print(f"  Nemenyi CD = {r['nemenyi_cd']:.3f}")
        else:
            print("  → NOT significant: cannot distinguish algorithm families")
        print("  Mean ranks (lower gap = better):")
        for algo in r.get("rank_order", []):
            print(f"    {algo:20s} {r['mean_ranks'][algo]:.2f}")

    # Bayesian engine comparison
    ec = results.get("engine_comparison", {})
    if ec:
        print(f"\n{'='*70}")
        print("BAYESIAN ENGINE COMPARISON (P(ml wins) / P(ROPE) / P(sklearn wins))")
        print(f"{'='*70}")
        for key, r in sorted(ec.items()):
            if "error" in r:
                print(f"  {r['algorithm']:20s} {r['task']:4s} ERROR: {r['error']}")
                continue
            interp = r["interpretation"].upper()
            print(f"  {r['algorithm']:20s} {r['task']:4s} "
                  f"P(ml)={r['p_ml_wins']:.3f} P(ROPE)={r['p_rope']:.3f} "
                  f"P(sk)={r['p_sklearn_wins']:.3f} → {interp} "
                  f"[N={r['n_datasets']}]")

    # Wilcoxon
    wc = results.get("wilcoxon", {})
    if wc:
        print(f"\n{'='*70}")
        print("WILCOXON SIGNED-RANK — ENGINE PARITY")
        print(f"{'='*70}")
        for key, r in sorted(wc.items()):
            if "error" in r:
                continue
            sig = "*" if r["p_value"] < 0.05 else " "
            print(f"  {r['algorithm']:20s} {r['task']:4s} "
                  f"W={r['wilcoxon_stat']:>8.1f} p={r['p_value']:.4f}{sig} "
                  f"r={r['rank_biserial_r']:>6.3f} "
                  f"W/L/T={r['wins_ml']}/{r['losses_ml']}/{r['ties']}")


# ---------------------------------------------------------------------------
# Living Benchmark — Online Learning Loop (spec §7)
# ---------------------------------------------------------------------------

def observe_weaknesses(db_path: Path) -> dict:
    """Query landscape DB for ml-Rust weaknesses using EVAL scores ONLY.
    NEVER queries assess_* columns — K_scope rule for the meta-experiment."""

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Find algorithms where ml loses to sklearn (by eval scores only)
    losses = conn.execute("""
        SELECT ml.algorithm, ml.dataset_id, ml.dataset_name,
               ml.eval_auc_mean as ml_eval, sk.eval_auc_mean as sk_eval,
               ml.cv_total_time_ms as ml_time, sk.cv_total_time_ms as sk_time
        FROM assess_holdout ml
        JOIN assess_holdout sk
          ON ml.dataset_id = sk.dataset_id
          AND ml.algorithm = sk.algorithm
          AND ml.seed = sk.seed
          AND ml.config_label = sk.config_label
        WHERE ml.engine = 'ml' AND sk.engine = 'sklearn'
          AND ml.error IS NULL AND sk.error IS NULL
          AND ml.eval_auc_mean IS NOT NULL AND sk.eval_auc_mean IS NOT NULL
          AND ml.eval_auc_mean < sk.eval_auc_mean
        ORDER BY (sk.eval_auc_mean - ml.eval_auc_mean) DESC
        LIMIT 20
    """).fetchall()

    # Find slowest algorithms
    slow = conn.execute("""
        SELECT ml.algorithm, ml.dataset_id,
               ml.cv_total_time_ms as ml_time, sk.cv_total_time_ms as sk_time,
               ml.cv_total_time_ms / NULLIF(sk.cv_total_time_ms, 0) as slowdown
        FROM assess_holdout ml
        JOIN assess_holdout sk
          ON ml.dataset_id = sk.dataset_id
          AND ml.algorithm = sk.algorithm
          AND ml.seed = sk.seed
          AND ml.config_label = sk.config_label
        WHERE ml.engine = 'ml' AND sk.engine = 'sklearn'
          AND ml.error IS NULL AND sk.error IS NULL
          AND ml.cv_total_time_ms > 2 * sk.cv_total_time_ms
        ORDER BY slowdown DESC
        LIMIT 10
    """).fetchall()

    # Win rate by algorithm
    winrates = conn.execute("""
        SELECT ml.algorithm,
               COUNT(*) as n,
               SUM(CASE WHEN ml.eval_auc_mean >= sk.eval_auc_mean THEN 1 ELSE 0 END) as wins,
               AVG(ml.eval_auc_mean - sk.eval_auc_mean) as mean_diff
        FROM assess_holdout ml
        JOIN assess_holdout sk
          ON ml.dataset_id = sk.dataset_id
          AND ml.algorithm = sk.algorithm
          AND ml.seed = sk.seed
          AND ml.config_label = sk.config_label
        WHERE ml.engine = 'ml' AND sk.engine = 'sklearn'
          AND ml.error IS NULL AND sk.error IS NULL
          AND ml.eval_auc_mean IS NOT NULL
        GROUP BY ml.algorithm
    """).fetchall()

    conn.close()

    report = {
        "worst_losses": [dict(r) for r in losses[:10]],
        "slowest": [dict(r) for r in slow],
        "winrates": {
            r["algorithm"]: {
                "n": r["n"],
                "wins": r["wins"],
                "rate": r["wins"] / r["n"] if r["n"] > 0 else 0,
                "mean_diff": r["mean_diff"],
            }
            for r in winrates
        },
    }

    # Print report
    print(f"\n{'='*70}")
    print("OBSERVE — ml-Rust weaknesses (eval scores ONLY, never assess)")
    print(f"{'='*70}")

    print("\nWin rates by algorithm:")
    for algo, wr in sorted(report["winrates"].items(), key=lambda x: x[1]["rate"]):
        print(f"  {algo:20s} {wr['wins']}/{wr['n']} = {wr['rate']:.1%} "
              f"(mean Δeval = {wr['mean_diff']:+.4f})")

    if report["worst_losses"]:
        print("\nBiggest eval losses (top 10):")
        for r in report["worst_losses"]:
            diff = r["sk_eval"] - r["ml_eval"]
            print(f"  {r['algorithm']:20s} ds={r['dataset_id']:>5d} "
                  f"ml={r['ml_eval']:.4f} sk={r['sk_eval']:.4f} Δ={diff:.4f}")

    if report["slowest"]:
        print("\nSlowest vs sklearn (>2× slower):")
        for r in report["slowest"]:
            print(f"  {r['algorithm']:20s} ds={r['dataset_id']:>5d} "
                  f"{r['slowdown']:.1f}× slower")

    return report


def check_convergence(db_path: Path) -> dict:
    """Check product-track convergence criteria (§7.4).
    All checks use eval_* scores only — never assess_*."""

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Overall win rate — SAME_ROSC_CELL only (identical math, fair comparison)
    algo_placeholders = ",".join(["?" for _ in SAME_ROSC_CELL])
    row = conn.execute(f"""
        SELECT COUNT(*) as n,
               SUM(CASE WHEN ml.eval_auc_mean >= sk.eval_auc_mean THEN 1 ELSE 0 END) as wins
        FROM assess_holdout ml
        JOIN assess_holdout sk
          ON ml.dataset_id = sk.dataset_id
          AND ml.algorithm = sk.algorithm
          AND ml.seed = sk.seed
          AND ml.config_label = sk.config_label
        WHERE ml.engine = 'ml' AND sk.engine = 'sklearn'
          AND ml.error IS NULL AND sk.error IS NULL
          AND ml.eval_auc_mean IS NOT NULL
          AND ml.algorithm IN ({algo_placeholders})
    """, list(SAME_ROSC_CELL)).fetchone()

    n = row["n"] if row else 0
    wins = row["wins"] if row else 0
    win_rate = wins / n if n > 0 else 0

    # Max slowdown — SAME_ROSC_CELL only
    slow_row = conn.execute(f"""
        SELECT MAX(ml.cv_total_time_ms / NULLIF(sk.cv_total_time_ms, 0)) as max_slow
        FROM assess_holdout ml
        JOIN assess_holdout sk
          ON ml.dataset_id = sk.dataset_id
          AND ml.algorithm = sk.algorithm
          AND ml.seed = sk.seed
          AND ml.config_label = sk.config_label
        WHERE ml.engine = 'ml' AND sk.engine = 'sklearn'
          AND ml.error IS NULL AND sk.error IS NULL
          AND ml.cv_total_time_ms IS NOT NULL
          AND ml.algorithm IN ({algo_placeholders})
    """, list(SAME_ROSC_CELL)).fetchone()
    max_slowdown = slow_row["max_slow"] if slow_row and slow_row["max_slow"] else 1.0

    # Max gap ratio — SAME_ROSC_CELL only, per-dataset averages (§5.1)
    # Uses per-dataset mean gap to smooth seed variance, excludes near-zero denominators
    gap_row = conn.execute(f"""
        SELECT MAX(ABS(ml_avg) / ABS(sk_avg)) as max_ratio
        FROM (
            SELECT ml.algorithm, ml.dataset_id,
                   AVG(ml.gap_auc) as ml_avg, AVG(sk.gap_auc) as sk_avg
            FROM assess_holdout ml
            JOIN assess_holdout sk
              ON ml.dataset_id = sk.dataset_id
              AND ml.algorithm = sk.algorithm
              AND ml.seed = sk.seed
              AND ml.config_label = sk.config_label
            WHERE ml.engine = 'ml' AND sk.engine = 'sklearn'
              AND ml.error IS NULL AND sk.error IS NULL
              AND ml.gap_auc IS NOT NULL AND sk.gap_auc IS NOT NULL
              AND ml.algorithm IN ({algo_placeholders})
            GROUP BY ml.algorithm, ml.dataset_id
            HAVING ABS(sk_avg) > 0.001
        )
    """, list(SAME_ROSC_CELL)).fetchone()
    max_gap_ratio = gap_row["max_ratio"] if gap_row and gap_row["max_ratio"] else 1.0

    converged = (
        win_rate >= 0.45 and      # ≥45% eval win rate
        max_slowdown <= 10.0 and  # No >10× slower
        max_gap_ratio <= 2.0      # No >2× gap ratio
    )

    result = {
        "win_rate": win_rate,
        "wins": wins,
        "n": n,
        "max_slowdown": max_slowdown,
        "max_gap_ratio": max_gap_ratio,
        "converged": converged,
    }

    print(f"\n{'='*70}")
    print("CONVERGENCE CHECK (eval scores only)")
    print(f"{'='*70}")
    print(f"  Win rate: {wins}/{n} = {win_rate:.1%} (threshold: ≥45%) "
          f"{'✓' if win_rate >= 0.45 else '✗'}")
    print(f"  Max slowdown: {max_slowdown:.1f}× (threshold: ≤10×) "
          f"{'✓' if max_slowdown <= 10.0 else '✗'}")
    print(f"  Max gap ratio: {max_gap_ratio:.1f}× (threshold: ≤2×) "
          f"{'✓' if max_gap_ratio <= 2.0 else '✗'}")
    print(f"  → {'CONVERGED' if converged else 'NOT CONVERGED'}")

    # Supplementary: same-family-different-impl (informational, not gating)
    sf_placeholders = ",".join(["?" for _ in SAME_FAMILY_DIFFERENT_IMPL])
    sf_row = conn.execute(f"""
        SELECT COUNT(*) as n,
               SUM(CASE WHEN ml.eval_auc_mean >= sk.eval_auc_mean THEN 1 ELSE 0 END) as wins,
               ml.algorithm
        FROM assess_holdout ml
        JOIN assess_holdout sk
          ON ml.dataset_id = sk.dataset_id
          AND ml.algorithm = sk.algorithm
          AND ml.seed = sk.seed
          AND ml.config_label = sk.config_label
        WHERE ml.engine = 'ml' AND sk.engine = 'sklearn'
          AND ml.error IS NULL AND sk.error IS NULL
          AND ml.eval_auc_mean IS NOT NULL
          AND ml.algorithm IN ({sf_placeholders})
        GROUP BY ml.algorithm
    """, list(SAME_FAMILY_DIFFERENT_IMPL)).fetchall()

    if sf_row:
        print(f"\n  Same-family, different implementation (informational):")
        for r in sf_row:
            sf_n, sf_wins, sf_algo = r
            sf_rate = sf_wins / sf_n if sf_n > 0 else 0
            print(f"    {sf_algo:20s} {sf_wins}/{sf_n} = {sf_rate:.1%}")

    conn.close()
    return result


# ---------------------------------------------------------------------------
# Multiprocessing workers (spec §4.7)
# ---------------------------------------------------------------------------

def _worker_run_cell(args: tuple) -> BenchmarkResult:
    """Pickleable wrapper for run_cell (multiprocessing requires top-level function)."""
    did, algo, engine, config_label, config, seed, suite = args
    return run_cell(did, algo, engine, config_label, config, seed, suite)


def run_benchmark_parallel(
    suite: str,
    seeds: list[int],
    workers: int = 4,
    smoke: bool = False,
    db_path: Path = DB_PATH,
):
    """Run benchmark with multiprocessing.
    Each worker writes to its own DB, merged at end (§4.7 contention fix)."""
    import socket
    hostname = socket.gethostname()

    suite_map = {"cc18": 99, "ctr23": 353}
    suite_id = suite_map.get(suite)
    if suite_id is None:
        print(f"Unknown suite: {suite}. Use cc18 or ctr23.")
        return

    task_type = "clf" if suite == "cc18" else "reg"

    print(f"Fetching OpenML suite {suite} (ID={suite_id})...")
    dataset_ids = fetch_suite(suite_id)
    print(f"  {len(dataset_ids)} datasets in suite")

    if smoke:
        dataset_ids = dataset_ids[:3]
        seeds = seeds[:2]
        print(f"  SMOKE MODE: {len(dataset_ids)} datasets, {len(seeds)} seeds")

    # Environment + manifest
    env = save_environment(BENCH_DIR)
    manifest = compute_source_manifest()
    git_hash = env.get("git_hash", "unknown")
    run_id = f"{git_hash}-{time.strftime('%Y%m%dT%H%M%S')}"
    print(f"  run_id: {run_id}")
    print(f"  manifest: {manifest_id(manifest)}")

    # Canary
    print("Running K_scope canary test...")
    canary_test(dataset_ids)
    print("  Canary passed.")

    # Build work items
    grid = get_grid_for_task(task_type)
    work = []
    for did in dataset_ids:
        for algo, engine in grid:
            for config_label, config in get_configs(algo):
                for seed in seeds:
                    work.append((did, algo, engine, config_label, config, seed, suite))

    print(f"\nTotal cells: {len(work)} ({len(dataset_ids)} datasets × "
          f"{len(grid)} algo-engine × {len(seeds)} seeds)")
    print(f"Workers: {workers}")

    # Init main DB + save manifest
    conn = init_db(db_path)
    save_manifest(conn, manifest, run_id, "product")

    completed = 0
    errors = 0
    t_start = time.perf_counter()

    if workers <= 1:
        # Sequential
        for item in work:
            completed += 1
            if completed % 50 == 0 or completed == 1:
                elapsed = time.perf_counter() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(work) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(work)}] {rate:.1f} cells/s, "
                      f"ETA {eta/60:.0f}min, errors={errors}")
            result = _worker_run_cell(item)
            if result.error:
                errors += 1
            save_result(conn, result, run_id, suite, hostname)
    else:
        # Parallel via ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker_run_cell, item): item for item in work}
            for future in as_completed(futures):
                completed += 1
                if completed % 50 == 0 or completed == 1:
                    elapsed = time.perf_counter() - t_start
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(work) - completed) / rate if rate > 0 else 0
                    print(f"  [{completed}/{len(work)}] {rate:.1f} cells/s, "
                          f"ETA {eta/60:.0f}min, errors={errors}")
                try:
                    result = future.result(timeout=CELL_TIMEOUT)
                    if result.error:
                        errors += 1
                    save_result(conn, result, run_id, suite, hostname)
                except Exception as e:
                    errors += 1
                    item = futures[future]
                    log.error("Worker error: %s: %s", item[:3], e)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {completed} cells in {elapsed/60:.1f}min, {errors} errors")
    print(f"Results in: {db_path}")
    conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Algorithm Landscape Experiment — Eval/Assess Gap Measurement",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    run_p = sub.add_parser("run", help="Run benchmark")
    run_p.add_argument("--suite", choices=["cc18", "ctr23"], default="cc18")
    run_p.add_argument("--seeds", default="0-9")
    run_p.add_argument("--workers", type=int, default=1)
    run_p.add_argument("--smoke", action="store_true")
    run_p.add_argument("--db", type=Path, default=DB_PATH)

    # --- report ---
    rep_p = sub.add_parser("report", help="Print summary from existing DB")
    rep_p.add_argument("--db", type=Path, default=DB_PATH)

    # --- stats ---
    stats_p = sub.add_parser("stats", help="Run statistical analysis (§5)")
    stats_p.add_argument("--db", type=Path, default=DB_PATH)

    # --- observe ---
    obs_p = sub.add_parser("observe", help="Observe ml-Rust weaknesses (§7)")
    obs_p.add_argument("--db", type=Path, default=DB_PATH)

    # --- converge ---
    conv_p = sub.add_parser("converge", help="Check convergence criteria (§7.4)")
    conv_p.add_argument("--db", type=Path, default=DB_PATH)

    # --- manifest ---
    sub.add_parser("manifest", help="Compute + print source hash manifest")

    # Backwards compat: no subcommand = run
    # Also support old-style --report, --smoke etc.
    parser.add_argument("--suite", choices=["cc18", "ctr23"], default="cc18")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--db", type=Path, default=DB_PATH)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "report" or getattr(args, "report", False):
        print_report(args.db)
    elif args.command == "stats":
        run_statistical_analysis(args.db)
    elif args.command == "observe":
        observe_weaknesses(args.db)
    elif args.command == "converge":
        check_convergence(args.db)
    elif args.command == "manifest":
        m = compute_source_manifest()
        print(json.dumps(m, indent=2))
        print(f"\nManifest ID: {manifest_id(m)}")
    else:
        # Default: run benchmark
        seeds = parse_seeds(args.seeds)
        runner = run_benchmark_parallel if args.workers > 1 else run_benchmark
        runner(
            suite=args.suite,
            seeds=seeds,
            workers=args.workers,
            smoke=args.smoke,
            db_path=args.db,
        )


if __name__ == "__main__":
    main()
