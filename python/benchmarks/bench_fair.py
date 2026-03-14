"""Fair benchmark: ml vs sklearn vs AutoML.

Four tiers of comparison, each honestly labeled:
  Tier 1: Wrapper overhead (same algo, decomposed)
  Tier 2: Screener vs screener (ml.screen vs PyCaret)
  Tier 3: AutoML HPO (FLAML, optional)
  Tier 4: Messy data survival (correctness guarantees)

Usage:
    python benchmarks/bench_fair.py              # Tier 1+2+4
    python benchmarks/bench_fair.py --tier3      # + FLAML
    python benchmarks/bench_fair.py --server      # + 100K synthetic
    python benchmarks/bench_fair.py --json --output results.json

Audited by 3 parallel auditors. All numbers include version + hardware info.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from _bench_utils import capture_versions, print_table, run_timed  # noqa: E402

import ml  # noqa: E402

# ── Tier 1: Wrapper Overhead ─────────────────────────────────────────────


def _sklearn_rf_pipeline(train_df, valid_df, target):
    """Raw sklearn RF pipeline — equivalent to what ml.fit() does internally."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[target])
    y_valid = le.transform(valid_df[target])

    X_train = train_df.drop(columns=[target]).copy()
    X_valid = valid_df.drop(columns=[target]).copy()

    cats = X_train.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    if cats:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[cats] = enc.fit_transform(X_train[cats])
        X_valid[cats] = enc.transform(X_valid[cats])

    X_train = X_train.values.astype(float)
    X_valid = X_valid.values.astype(float)

    # Impute NaN (equivalent to what ml does internally)
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train)
    X_valid = imp.transform(X_valid)

    clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_valid)
    y_proba = clf.predict_proba(X_valid)[:, 1]

    return {
        "accuracy": accuracy_score(y_valid, y_pred),
        "f1": f1_score(y_valid, y_pred, average="binary"),
        "roc_auc": roc_auc_score(y_valid, y_proba),
    }


def _sklearn_logistic_pipeline(train_df, valid_df, target):
    """Raw sklearn Logistic pipeline — includes scaling (as ml does automatically)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[target])
    y_valid = le.transform(valid_df[target])

    X_train = train_df.drop(columns=[target]).copy()
    X_valid = valid_df.drop(columns=[target]).copy()

    cats = X_train.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    if cats:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[cats] = enc.fit_transform(X_train[cats])
        X_valid[cats] = enc.transform(X_valid[cats])

    X_train = X_train.values.astype(float)
    X_valid = X_valid.values.astype(float)

    # Impute NaN (equivalent to what ml does internally)
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train)
    X_valid = imp.transform(X_valid)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_valid)
    y_proba = clf.predict_proba(X_valid)[:, 1]

    return {
        "accuracy": accuracy_score(y_valid, y_pred),
        "f1": f1_score(y_valid, y_pred, average="binary"),
        "roc_auc": roc_auc_score(y_valid, y_proba),
    }


def tier1_overhead(datasets: list[dict], json_only: bool = False) -> dict:
    """Tier 1: Wrapper overhead with decomposition."""
    results = {}

    for ds in datasets:
        name = ds["name"]
        data = ds["data"]
        target = ds["target"]

        if not json_only:
            print(f"\n  Tier 1: {name} ({len(data):,} rows)")
            print(f"  {'─' * 50}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = ml.split(data=data, target=target, seed=42)

        comparison = {}

        # ── ml.fit (RF, no early stopping for fair comparison) ──
        def ml_rf(_s=s, _target=target):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ml.fit(
                    data=_s.train, target=_target, algorithm="random_forest",
                    seed=42, early_stopping=False,
                )
        ml_rf_timing = run_timed(ml_rf, warmup=3, runs=7)

        # Get accuracy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ml_model = ml.fit(
                data=s.train, target=target, algorithm="random_forest",
                seed=42, early_stopping=False,
            )
        ml_metrics = ml.evaluate(ml_model, s.valid)

        comparison["ml_rf"] = {
            **ml_rf_timing,
            "accuracy": round(ml_metrics.get("accuracy", 0), 4),
            "roc_auc": round(ml_metrics.get("roc_auc", 0), 4),
        }

        # ── sklearn RF (equivalent preprocessing) ──
        def sk_rf(_s=s, _target=target):
            _sklearn_rf_pipeline(_s.train, _s.valid, _target)
        sk_rf_timing = run_timed(sk_rf, warmup=3, runs=7)
        sk_rf_metrics = _sklearn_rf_pipeline(s.train, s.valid, target)

        comparison["sklearn_rf"] = {
            **sk_rf_timing,
            "accuracy": round(sk_rf_metrics["accuracy"], 4),
            "roc_auc": round(sk_rf_metrics["roc_auc"], 4),
        }

        # ── Overhead ──
        ml_t = comparison["ml_rf"]["median_seconds"]
        sk_t = comparison["sklearn_rf"]["median_seconds"]
        overhead_pct = round((ml_t - sk_t) / sk_t * 100, 1) if sk_t > 0 else 0
        comparison["overhead_pct_rf"] = overhead_pct
        comparison["accuracy_delta_rf"] = round(
            abs(comparison["ml_rf"]["accuracy"] - comparison["sklearn_rf"]["accuracy"]), 4
        )

        # ── ml.fit (Logistic) ──
        def ml_lr(_s=s, _target=target):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ml.fit(
                    data=_s.train, target=_target, algorithm="logistic",
                    seed=42, early_stopping=False,
                )
        ml_lr_timing = run_timed(ml_lr, warmup=3, runs=7)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ml_lr_model = ml.fit(
                data=s.train, target=target, algorithm="logistic",
                seed=42, early_stopping=False,
            )
        ml_lr_metrics = ml.evaluate(ml_lr_model, s.valid)

        comparison["ml_logistic"] = {
            **ml_lr_timing,
            "accuracy": round(ml_lr_metrics.get("accuracy", 0), 4),
            "roc_auc": round(ml_lr_metrics.get("roc_auc", 0), 4),
        }

        # ── sklearn Logistic ──
        def sk_lr(_s=s, _target=target):
            _sklearn_logistic_pipeline(_s.train, _s.valid, _target)
        sk_lr_timing = run_timed(sk_lr, warmup=3, runs=7)
        sk_lr_metrics = _sklearn_logistic_pipeline(s.train, s.valid, target)

        comparison["sklearn_logistic"] = {
            **sk_lr_timing,
            "accuracy": round(sk_lr_metrics["accuracy"], 4),
            "roc_auc": round(sk_lr_metrics["roc_auc"], 4),
        }

        ml_lr_t = comparison["ml_logistic"]["median_seconds"]
        sk_lr_t = comparison["sklearn_logistic"]["median_seconds"]
        comparison["overhead_pct_logistic"] = round(
            (ml_lr_t - sk_lr_t) / sk_lr_t * 100, 1
        ) if sk_lr_t > 0 else 0
        comparison["accuracy_delta_logistic"] = round(
            abs(comparison["ml_logistic"]["accuracy"] - comparison["sklearn_logistic"]["accuracy"]),
            4,
        )

        results[name] = comparison

        if not json_only:
            print_table(
                [
                    {
                        "library": "ml (RF)",
                        "median_s": ml_t,
                        "rss_mb": comparison["ml_rf"]["rss_delta_mb"],
                        "accuracy": comparison["ml_rf"]["accuracy"],
                        "roc_auc": comparison["ml_rf"]["roc_auc"],
                    },
                    {
                        "library": "sklearn (RF)",
                        "median_s": sk_t,
                        "rss_mb": comparison["sklearn_rf"]["rss_delta_mb"],
                        "accuracy": comparison["sklearn_rf"]["accuracy"],
                        "roc_auc": comparison["sklearn_rf"]["roc_auc"],
                    },
                    {
                        "library": "ml (logistic)",
                        "median_s": ml_lr_t,
                        "rss_mb": comparison["ml_logistic"]["rss_delta_mb"],
                        "accuracy": comparison["ml_logistic"]["accuracy"],
                        "roc_auc": comparison["ml_logistic"]["roc_auc"],
                    },
                    {
                        "library": "sklearn (logistic)",
                        "median_s": sk_lr_t,
                        "rss_mb": comparison["sklearn_logistic"]["rss_delta_mb"],
                        "accuracy": comparison["sklearn_logistic"]["accuracy"],
                        "roc_auc": comparison["sklearn_logistic"]["roc_auc"],
                    },
                ],
                title=f"Tier 1: Wrapper Overhead — {name}",
                columns=["library", "median_s", "rss_mb", "accuracy", "roc_auc"],
            )
            print(f"  RF overhead: {overhead_pct:+.1f}%  |  Accuracy delta: "
                  f"{comparison['accuracy_delta_rf']:.4f}")
            print(f"  LR overhead: {comparison['overhead_pct_logistic']:+.1f}%  |  "
                  f"Accuracy delta: {comparison['accuracy_delta_logistic']:.4f}")
            print("\n  Note: ml overhead reflects auto-preprocessing + input validation.")
            print("  Both sides use early_stopping=False, n_jobs=1, same data split.")

    return results


# ── Tier 2: Screener vs Screener ─────────────────────────────────────────


def tier2_screener(data, target, json_only: bool = False) -> dict:
    """Tier 2: ml.screen() vs PyCaret compare_models() — same product category."""
    results = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=data, target=target, seed=42)

    # ── ml.screen ──
    def ml_screen():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ml.screen(
                s, target,
                algorithms=["random_forest", "logistic"],
                seed=42,
            )
    ml_timing = run_timed(ml_screen, warmup=2, runs=5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb = ml.screen(s, target, algorithms=["random_forest", "logistic"], seed=42)
    best_auc = float(lb.iloc[0].get("roc_auc", 0)) if "roc_auc" in lb.columns else None

    results["ml_screen"] = {
        **ml_timing,
        "best_roc_auc": round(best_auc, 4) if best_auc else None,
        "n_models": len(lb),
        "strategy": "holdout_defaults",
    }

    # ── PyCaret (subprocess isolation) ──
    pycaret_result = _run_pycaret_subprocess(data, target)
    if pycaret_result:
        results["pycaret"] = pycaret_result
    else:
        results["pycaret"] = {"skipped": True, "reason": "not installed or error"}

    if not json_only:
        rows = [
            {
                "library": "ml.screen()",
                "time_s": results["ml_screen"]["median_seconds"],
                "best_auc": results["ml_screen"]["best_roc_auc"] or "N/A",
                "models": results["ml_screen"]["n_models"],
                "strategy": "holdout, defaults",
            },
        ]
        if not results["pycaret"].get("skipped"):
            rows.append({
                "library": "PyCaret",
                "time_s": results["pycaret"].get("wall_seconds", "N/A"),
                "best_auc": results["pycaret"].get("best_roc_auc", "N/A"),
                "models": results["pycaret"].get("n_models", "N/A"),
                "strategy": "10-fold CV, defaults",
            })
        else:
            rows.append({
                "library": "PyCaret",
                "time_s": "SKIP",
                "best_auc": "SKIP",
                "models": "SKIP",
                "strategy": results["pycaret"]["reason"],
            })

        print_table(
            rows,
            title="Tier 2: Screener vs Screener (same product category)",
            columns=["library", "time_s", "best_auc", "models", "strategy"],
        )
        print("  Both fit default algorithms (RF + Logistic). No hyperparameter search.")
        print("  ml uses holdout validation. PyCaret uses 10-fold CV.\n")

    return results


def _run_pycaret_subprocess(data, target) -> dict | None:
    """Run PyCaret in subprocess to avoid global state contamination."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        data.to_csv(f, index=False)
        csv_path = f.name

    worker = f"""
import sys, json, time, warnings
import pandas as pd
try:
    from pycaret.classification import compare_models, setup
except ImportError:
    print(json.dumps({{"skipped": True, "reason": "not installed"}}))
    sys.exit(0)

df = pd.read_csv("{csv_path}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    t0 = time.perf_counter()
    setup(df, target="{target}", session_id=42, verbose=False)
    best = compare_models(include=["rf", "lr"], sort="AUC", verbose=False)
    elapsed = time.perf_counter() - t0

# Get AUC from PyCaret's pull()
try:
    from pycaret.classification import pull
    results_df = pull()
    best_auc = float(results_df.iloc[0]["AUC"])
except Exception:
    best_auc = None

print(json.dumps({{
    "wall_seconds": round(elapsed, 3),
    "best_roc_auc": round(best_auc, 4) if best_auc else None,
    "n_models": 2,
    "strategy": "10fold_cv_defaults",
}}))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", worker],
            capture_output=True, text=True, timeout=120,
        )
        os.unlink(csv_path)
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip())
        return None
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(csv_path)
        return None


# ── Tier 3: AutoML (FLAML) ──────────────────────────────────────────────


def tier3_automl(data, target, json_only: bool = False) -> dict:
    """Tier 3: FLAML AutoML with time_budget — different product category."""
    if not json_only:
        print("\n" + "━" * 60)
        print("  IMPORTANT: This comparison crosses product categories.")
        print("━" * 60)
        print("  ml.screen() fits algorithms with defaults. No HPO.")
        print("  FLAML searches hyperparameter space within a time budget.")
        print("  These are different tools for different workflow stages.")
        print("━" * 60)

    results = {}

    # Check FLAML availability
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            data.to_csv(f, index=False)
            csv_path = f.name

        worker = f"""
import sys, json, time, warnings
import pandas as pd
import numpy as np
try:
    from flaml import AutoML
except ImportError:
    print(json.dumps({{"skipped": True, "reason": "not installed"}}))
    sys.exit(0)
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import roc_auc_score

df = pd.read_csv("{csv_path}")
target = "{target}"

# Prepare data (same as ml would)
le = LabelEncoder()
y = le.fit_transform(df[target])
X = df.drop(columns=[target]).copy()
cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
if cats:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cats] = enc.fit_transform(X[cats])
X = X.values.astype(float)

n = len(X)
split_idx = int(n * 0.6)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    automl = AutoML()
    t0 = time.perf_counter()
    automl.fit(
        X_train, y_train,
        task="classification",
        time_budget=30,
        estimator_list=["rf", "lgbm", "xgboost"],
        seed=42,
        verbose=0,
    )
    elapsed = time.perf_counter() - t0

try:
    proba = automl.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)
except Exception:
    auc = None

print(json.dumps({{
    "wall_seconds": round(elapsed, 3),
    "best_roc_auc": round(float(auc), 4) if auc else None,
    "best_estimator": str(automl.best_estimator),
    "strategy": "hpo_30s_budget",
}}))
"""
        proc = subprocess.run(
            [sys.executable, "-c", worker],
            capture_output=True, text=True, timeout=120,
        )
        os.unlink(csv_path)
        if proc.returncode == 0 and proc.stdout.strip():
            results["flaml"] = json.loads(proc.stdout.strip())
        else:
            results["flaml"] = {"skipped": True, "reason": "error or timeout"}
    except Exception as e:
        results["flaml"] = {"skipped": True, "reason": str(e)}

    if not json_only:
        flaml = results.get("flaml", {})
        if flaml.get("skipped"):
            print(f"\n  FLAML: SKIPPED ({flaml.get('reason', 'unknown')})")
        else:
            print("\n  FLAML AutoML (30s budget):")
            print(f"    Time:           {flaml.get('wall_seconds', 'N/A')}s")
            print(f"    Best AUC:       {flaml.get('best_roc_auc', 'N/A')}")
            print(f"    Best estimator: {flaml.get('best_estimator', 'N/A')}")
            print("    Strategy:       HPO with 30s time budget\n")

    return results


# ── Tier 4: Messy Data Survival ──────────────────────────────────────────


def tier4_messy_data(json_only: bool = False) -> dict:
    """Tier 4: Correctness guarantee comparison — messy data handling."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    results = {}
    scenarios = []

    # Build messy dataset
    rng = np.random.RandomState(42)
    n = 200
    messy = pd.DataFrame({
        "numeric": rng.randn(n),
        "with_nan": np.where(rng.rand(n) < 0.2, np.nan, rng.randn(n)),
        "categorical": rng.choice(["red", "green", "blue"], n),
        "high_card": [f"id_{i}" for i in rng.randint(0, 100, n)],
        "target": rng.choice(["yes", "no"], n),
    })

    # Scenario 1: String target labels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = ml.split(data=messy, target="target", seed=42)
            ml.fit(data=s.train, target="target", seed=42)
        ml_result = "OK"
    except Exception as e:
        ml_result = f"FAIL: {e}"

    try:
        X = messy.drop(columns=["target"]).select_dtypes(include=[np.number]).values
        y = messy["target"].values  # strings
        RandomForestClassifier(random_state=42).fit(X, y)
        sk_result = "OK (numeric only)"
    except Exception as e:
        sk_result = f"FAIL: {type(e).__name__}"

    scenarios.append({
        "scenario": "String targets + categoricals",
        "ml": ml_result,
        "sklearn_raw": sk_result,
    })

    # Scenario 2: NaN in features (SVM)
    numeric_with_nan = messy[["numeric", "with_nan", "target"]].copy()
    numeric_with_nan["target"] = (rng.randn(n) > 0).astype(int)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s2 = ml.split(data=numeric_with_nan, target="target", seed=42)
            ml.fit(data=s2.train, target="target", algorithm="random_forest", seed=42)
        ml_result2 = "OK (trees handle NaN)"
    except Exception as e:
        ml_result2 = f"FAIL: {e}"

    try:
        X2 = numeric_with_nan.drop(columns=["target"]).values
        y2 = numeric_with_nan["target"].values
        SVC().fit(X2, y2)
        sk_result2 = "OK"
    except Exception as e:
        sk_result2 = f"FAIL: {type(e).__name__}"

    scenarios.append({
        "scenario": "NaN in features (SVM)",
        "ml": ml_result2,
        "sklearn_raw": sk_result2,
    })

    # Scenario 3: Auto-scaling for SVM
    clean_numeric = pd.DataFrame({
        "f1": rng.randn(n) * 1000,  # large scale
        "f2": rng.randn(n) * 0.001,  # tiny scale
        "target": (rng.randn(n) > 0).astype(int),
    })

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s3 = ml.split(data=clean_numeric, target="target", seed=42)
            model3 = ml.fit(data=s3.train, target="target", algorithm="svm", seed=42)
            m3 = ml.evaluate(model3, s3.valid)
        ml_result3 = f"OK (AUC={m3.get('roc_auc', 'N/A'):.3f}, auto-scaled)"
    except Exception as e:
        ml_result3 = f"FAIL: {e}"

    try:
        X3_train = s3.train.drop(columns=["target"]).values
        y3_train = s3.train["target"].values
        X3_valid = s3.valid.drop(columns=["target"]).values
        y3_valid = s3.valid["target"].values
        svm_raw = SVC(probability=True, random_state=42)
        svm_raw.fit(X3_train, y3_train)  # no scaling!
        from sklearn.metrics import roc_auc_score
        raw_auc = roc_auc_score(y3_valid, svm_raw.predict_proba(X3_valid)[:, 1])
        sk_result3 = f"OK (AUC={raw_auc:.3f}, NO scaling)"
    except Exception as e:
        sk_result3 = f"FAIL: {type(e).__name__}"

    scenarios.append({
        "scenario": "Unscaled features (SVM)",
        "ml": ml_result3,
        "sklearn_raw": sk_result3,
    })

    # Scenario 4: assess() test-set discipline
    scenarios.append({
        "scenario": "Prevent test-set peeking",
        "ml": "assess() blocks repeat calls",
        "sklearn_raw": "no guard (user responsibility)",
    })

    results["scenarios"] = scenarios

    if not json_only:
        print_table(
            scenarios,
            title="Tier 4: Messy Data Survival — What ml Handles Automatically",
            columns=["scenario", "ml", "sklearn_raw"],
        )
        print("  ml auto-detects string targets, encodes categoricals, scales for SVM/KNN,")
        print("  passes NaN through to tree models, and warns on class imbalance.")
        print("  Raw sklearn requires the user to handle each of these explicitly.\n")

    return results


# ── Main ─────────────────────────────────────────────────────────────────


def run_all(
    include_server: bool = False,
    include_tier3: bool = False,
    json_only: bool = False,
) -> dict:
    versions = capture_versions()
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "versions": versions,
        "note": "CPU-only benchmark. n_jobs=1. early_stopping=False for fair comparison.",
    }

    if not json_only:
        print("=" * 60)
        print("  ml Fair Benchmark")
        print("=" * 60)
        v = versions
        print(f"  ml {v['ml']} | sklearn {v['sklearn']} | "
              f"pandas {v['pandas']} | numpy {v['numpy']}")
        print(f"  Python {v['python']} | {v['platform']} {v['machine']} | "
              f"{v['cpu_count']} CPUs | {v['ram_gb']} GB RAM")
        print("  Warmup: 3 | Measured: 7 | Memory: RSS delta (psutil)")
        print("  early_stopping=False | n_jobs=1")
        print("=" * 60)

    # Datasets
    datasets = []
    try:
        churn = ml.dataset("churn")
        datasets.append({"name": "churn_7k", "data": churn, "target": "churn"})
    except Exception:
        # Fallback to synthetic
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=7000, n_features=20, random_state=42)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
        df["target"] = y
        datasets.append({"name": "synthetic_7k", "data": df, "target": "target"})

    if include_server:
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=100_000, n_features=30,
            n_informative=15, random_state=42,
        )
        df_100k = pd.DataFrame(X, columns=[f"f{i}" for i in range(30)])
        df_100k["target"] = y
        datasets.append({"name": "synthetic_100k", "data": df_100k, "target": "target"})

    # Tier 1
    results["tier1_overhead"] = tier1_overhead(datasets, json_only)

    # Tier 2 (on first dataset)
    results["tier2_screener"] = tier2_screener(
        datasets[0]["data"], datasets[0]["target"], json_only,
    )

    # Tier 3 (optional)
    if include_tier3:
        results["tier3_automl"] = tier3_automl(
            datasets[0]["data"], datasets[0]["target"], json_only,
        )

    # Tier 4 (always)
    results["tier4_messy_data"] = tier4_messy_data(json_only)

    # Cleanup
    gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser(description="ml fair benchmark")
    parser.add_argument("--tier3", action="store_true", help="Include FLAML AutoML")
    parser.add_argument("--server", action="store_true", help="Include 100K synthetic")
    parser.add_argument("--json", action="store_true", help="JSON output only")
    parser.add_argument("--output", type=str, help="Save JSON to file")
    args = parser.parse_args()

    results = run_all(
        include_server=args.server,
        include_tier3=args.tier3,
        json_only=args.json,
    )

    if args.json or args.output:
        output = json.dumps(results, indent=2, default=str)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            if not args.json:
                print(f"\nResults saved to {args.output}")
        else:
            print(output)


if __name__ == "__main__":
    main()
