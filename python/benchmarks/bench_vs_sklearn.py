"""Side-by-side comparison: ml vs sklearn — code, timing, metrics.

Validates the "4 lines vs 12 lines" claim honestly.

Run: python benchmarks/bench_vs_sklearn.py [--json]
"""

import json
import platform
import sys
import time

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import ml

# ---------------------------------------------------------------------------
# Code snippets (the actual claim)
# ---------------------------------------------------------------------------

ML_CODE_CLF = """\
import ml
s     = ml.split(data, "churn", seed=42)
model = ml.fit(s.train, "churn", seed=42)
print(ml.evaluate(model, s.test))"""

SKLEARN_CODE_CLF = """\
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X = data.drop(columns=["churn"])
y = le.fit_transform(data["churn"])
for col in X.select_dtypes(include=["object", "category"]):
    X[col] = X[col].astype("category").cat.codes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))"""

ML_CODE_REG = """\
import ml
s     = ml.split(data, "tip", seed=42)
model = ml.fit(s.train, "tip", seed=42)
print(ml.evaluate(model, s.test))"""

SKLEARN_CODE_REG = """\
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = data.drop(columns=["tip"])
y = data["tip"]
for col in X.select_dtypes(include=["object", "category"]):
    X[col] = X[col].astype("category").cat.codes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
print(mean_squared_error(y_test, preds) ** 0.5, mean_absolute_error(y_test, preds), r2_score(y_test, preds))"""

ML_CODE_SCALED = """\
import ml
s     = ml.split(data, "diagnosis", seed=42)
model = ml.fit(s.train, "diagnosis", algorithm="logistic", seed=42)
print(ml.evaluate(model, s.test))"""

SKLEARN_CODE_SCALED = """\
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()
X = data.drop(columns=["diagnosis"]).values.astype(float)
y = le.fit_transform(data["diagnosis"])
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))"""


def count_lines(code):
    return len([line for line in code.strip().split("\n") if line.strip()])


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_fn(fn, warmup=1, runs=5):
    """Time a function with warmup. Returns median seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Test 1: Code complexity comparison
# ---------------------------------------------------------------------------

def test_code_complexity():
    """Count lines and unique imports for each workflow."""
    cases = [
        ("Classification (RF, mixed types)", ML_CODE_CLF, SKLEARN_CODE_CLF),
        ("Regression (RF, mixed types)", ML_CODE_REG, SKLEARN_CODE_REG),
        ("Classification (Logistic, scaling)", ML_CODE_SCALED, SKLEARN_CODE_SCALED),
    ]

    print("\n" + "=" * 70)
    print("  TEST 1: Code Complexity")
    print("=" * 70)
    print(f"  {'task':<38} {'ml lines':>10} {'sklearn lines':>14} {'ratio':>8}")
    print(f"  {'-' * 62}")

    results = []
    for name, ml_code, sk_code in cases:
        ml_lines = count_lines(ml_code)
        sk_lines = count_lines(sk_code)
        ratio = f"{sk_lines / ml_lines:.1f}x"
        print(f"  {name:<38} {ml_lines:>10} {sk_lines:>14} {ratio:>8}")
        results.append({
            "task": name,
            "ml_lines": ml_lines,
            "sklearn_lines": sk_lines,
            "ratio": round(sk_lines / ml_lines, 1),
        })

    print()
    print("  ml code (classification, mixed types):")
    for line in ML_CODE_CLF.strip().split("\n"):
        print(f"    {line}")
    print()
    print("  sklearn equivalent:")
    for line in SKLEARN_CODE_CLF.strip().split("\n"):
        print(f"    {line}")

    print()
    print("  Notes:")
    print("  - ml auto-encodes categoricals, auto-splits 3-way, returns all metrics")
    print("  - sklearn requires manual encoding, 2-way split, one metric per call")
    print("  - ml additionally creates a validation set (3-way split vs 2-way)")
    return results


# ---------------------------------------------------------------------------
# Test 2: Wrapper overhead timing
# ---------------------------------------------------------------------------

def _sklearn_rf_churn(data, target):
    """Reproduce ml.split + ml.fit + ml.evaluate in raw sklearn."""
    le = LabelEncoder()
    le.fit(data[target])

    X = data.drop(columns=[target]).copy()
    y = le.transform(data[target])

    for col in X.select_dtypes(include=["object", "category"]).columns:
        cats = X[col].astype("category").cat.categories
        mapping = {c: i for i, c in enumerate(cats)}
        X[col] = X[col].map(mapping).fillna(-1).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average="binary", zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba[:, 1]),
    }


def _sklearn_logistic_cancer(data, target):
    """Reproduce ml workflow for logistic regression (needs scaling)."""
    le = LabelEncoder()
    le.fit(data[target])

    X = data.drop(columns=[target]).values.astype(float)
    y = le.transform(data[target])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average="binary", zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba[:, 1]),
    }


def _sklearn_rf_tips(data, target):
    """Reproduce ml workflow for RF regression."""
    X = data.drop(columns=[target]).copy()
    y = data[target].values

    for col in X.select_dtypes(include=["object", "category"]).columns:
        cats = X[col].astype("category").cat.categories
        mapping = {c: i for i, c in enumerate(cats)}
        X[col] = X[col].map(mapping).fillna(-1).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }


def test_timing():
    """Compare wall time: ml wrapper vs raw sklearn."""
    churn = ml.dataset("churn")
    cancer = ml.dataset("cancer")
    tips = ml.dataset("tips")

    cases = [
        (
            "RF classification (churn)",
            lambda: (lambda s: ml.evaluate(ml.fit(s.train, "churn", algorithm="random_forest", seed=42), s.valid))(
                ml.split(churn, "churn", seed=42)
            ),
            lambda: _sklearn_rf_churn(churn, "churn"),
        ),
        (
            "Logistic classification (cancer)",
            lambda: (lambda s: ml.evaluate(ml.fit(s.train, "diagnosis", algorithm="logistic", seed=42), s.valid))(
                ml.split(cancer, "diagnosis", seed=42)
            ),
            lambda: _sklearn_logistic_cancer(cancer, "diagnosis"),
        ),
        (
            "RF regression (tips)",
            lambda: (lambda s: ml.evaluate(ml.fit(s.train, "tip", algorithm="random_forest", seed=42), s.valid))(
                ml.split(tips, "tip", seed=42)
            ),
            lambda: _sklearn_rf_tips(tips, "tip"),
        ),
    ]

    print("\n" + "=" * 70)
    print("  TEST 2: Wrapper Overhead (median of 5 runs, 1 warmup)")
    print("=" * 70)
    print(f"  {'task':<34} {'ml (s)':>8} {'sklearn (s)':>12} {'overhead':>10}")
    print(f"  {'-' * 58}")

    results = []
    for name, ml_fn, sk_fn in cases:
        ml_time = time_fn(ml_fn, warmup=1, runs=5)
        sk_time = time_fn(sk_fn, warmup=1, runs=5)
        overhead = (ml_time / sk_time - 1) * 100
        sign = "+" if overhead >= 0 else ""
        print(f"  {name:<34} {ml_time:>8.3f} {sk_time:>12.3f} {sign}{overhead:>8.1f}%")
        results.append({
            "task": name,
            "ml_seconds": round(ml_time, 4),
            "sklearn_seconds": round(sk_time, 4),
            "overhead_pct": round(overhead, 1),
        })

    print()
    print("  Note: ml includes auto-encoding, auto-scaling, 3-way split, all metrics.")
    print("  Overhead is the cost of these conveniences vs manual sklearn.")
    return results


# ---------------------------------------------------------------------------
# Test 3: Metric parity
# ---------------------------------------------------------------------------

def test_parity():
    """Prove ml produces same metrics as sklearn on same data and split."""
    print("\n" + "=" * 70)
    print("  TEST 3: Metric Parity (ml vs sklearn, same data + split)")
    print("=" * 70)

    cases = [
        ("RF clf (churn)", "churn", "churn", "random_forest", "clf"),
        ("Logistic clf (cancer)", "cancer", "diagnosis", "logistic", "clf"),
        ("RF reg (tips)", "tips", "tip", "random_forest", "reg"),
    ]

    results = []
    all_pass = True

    for name, dataset_name, target, algo, task in cases:
        data = ml.dataset(dataset_name)
        s = ml.split(data, target, seed=42)
        ml_model = ml.fit(s.train, target, algorithm=algo, seed=42)
        ml_metrics = ml.evaluate(ml_model, s.valid)

        # Replicate in sklearn using same split
        train_df = s.train.copy()
        valid_df = s.valid.copy()

        if task == "clf":
            le = LabelEncoder()
            le.fit(train_df[target])
            y_train = le.transform(train_df[target])
            y_valid = le.transform(valid_df[target])

            X_train = train_df.drop(columns=[target]).copy()
            X_valid = valid_df.drop(columns=[target]).copy()

            if algo == "logistic":
                X_train = X_train.values.astype(float)
                X_valid = X_valid.values.astype(float)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_valid = scaler.transform(X_valid)
                clf = LogisticRegression(random_state=42, max_iter=1000)
            else:
                for col in X_train.select_dtypes(include=["object", "category"]).columns:
                    cats = X_train[col].astype("category").cat.categories
                    mapping = {c: i for i, c in enumerate(cats)}
                    X_train[col] = X_train[col].map(mapping).fillna(-1).astype(float)
                    X_valid[col] = X_valid[col].map(mapping).fillna(-1).astype(float)
                clf = RandomForestClassifier(n_estimators=100, random_state=42)

            clf.fit(X_train, y_train)
            sk_preds = clf.predict(X_valid)
            sk_proba = clf.predict_proba(X_valid)
            sk_metrics = {
                "accuracy": accuracy_score(y_valid, sk_preds),
                "f1": f1_score(y_valid, sk_preds, average="binary", zero_division=0),
                "roc_auc": roc_auc_score(y_valid, sk_proba[:, 1]),
            }
        else:
            y_train = train_df[target].values
            y_valid = valid_df[target].values
            X_train = train_df.drop(columns=[target]).copy()
            X_valid = valid_df.drop(columns=[target]).copy()
            for col in X_train.select_dtypes(include=["object", "category"]).columns:
                cats = X_train[col].astype("category").cat.categories
                mapping = {c: i for i, c in enumerate(cats)}
                X_train[col] = X_train[col].map(mapping).fillna(-1).astype(float)
                X_valid[col] = X_valid[col].map(mapping).fillna(-1).astype(float)
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            reg.fit(X_train, y_train)
            sk_preds = reg.predict(X_valid)
            sk_metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_valid, sk_preds))),
                "mae": mean_absolute_error(y_valid, sk_preds),
                "r2": r2_score(y_valid, sk_preds),
            }

        print(f"\n  {name}")
        print(f"  {'metric':<12} {'ml':>10} {'sklearn':>10} {'delta':>10} {'ok':>4}")
        print(f"  {'-' * 48}")

        case_result = {"name": name, "metrics": []}
        case_pass = True
        for key in ml_metrics:
            if key not in sk_metrics:
                continue
            delta = abs(ml_metrics[key] - sk_metrics[key])
            ok = delta < 0.01
            marker = "Y" if ok else "N"
            if not ok:
                case_pass = False
                all_pass = False
            print(f"  {key:<12} {ml_metrics[key]:>10.4f} {sk_metrics[key]:>10.4f} {delta:>10.4f} {marker:>4}")
            case_result["metrics"].append({
                "metric": key,
                "ml": round(ml_metrics[key], 6),
                "sklearn": round(sk_metrics[key], 6),
                "delta": round(delta, 6),
                "pass": ok,
            })

        case_result["pass"] = case_pass
        results.append(case_result)

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")
    if all_pass:
        print("  ml wraps sklearn faithfully — same results within tolerance.")
    else:
        print("  Note: roc_auc deltas are expected when ordinal encoding differs")
        print("  (ml uses its own category mapper). Use parity_check.py for exact parity.")
    return results


# ---------------------------------------------------------------------------
# Test 4: Algorithm coverage
# ---------------------------------------------------------------------------

def test_coverage():
    """Verify all algorithms can fit and evaluate."""
    print("\n" + "=" * 70)
    print("  TEST 4: Algorithm Coverage (all families)")
    print("=" * 70)

    # Use cancer (no NaN, no categoricals) for NaN-sensitive algorithms
    cancer = ml.dataset("cancer")
    s_cancer = ml.split(cancer, "diagnosis", seed=42)

    # Use tips for regression-only algorithms
    tips = ml.dataset("tips")
    s_tips = ml.split(tips, "tip", seed=42)

    algos = [
        # (algo, task, split, target)
        ("random_forest", "clf", s_cancer, "diagnosis"),
        ("decision_tree", "clf", s_cancer, "diagnosis"),
        ("logistic", "clf", s_cancer, "diagnosis"),
        ("knn", "clf", s_cancer, "diagnosis"),
        ("naive_bayes", "clf", s_cancer, "diagnosis"),
        ("adaboost", "clf", s_cancer, "diagnosis"),
        ("svm", "clf", s_cancer, "diagnosis"),
        ("gradient_boosting", "clf", s_cancer, "diagnosis"),
        ("extra_trees", "clf", s_cancer, "diagnosis"),
        ("linear", "reg", s_tips, "tip"),
        ("elastic_net", "reg", s_tips, "tip"),
    ]

    # Try xgboost if available
    try:
        import xgboost  # noqa: F401
        algos.append(("xgboost", "clf", s_cancer, "diagnosis"))
    except ImportError:
        pass

    print(f"  {'algorithm':<22} {'task':>5} {'fit':>5} {'eval':>6} {'explain':>8} {'time (s)':>10}")
    print(f"  {'-' * 60}")

    results = []
    for algo, task, split, target in algos:
        t0 = time.perf_counter()
        can_fit = True
        can_eval = True
        can_explain = True
        try:
            model = ml.fit(split.train, target, algorithm=algo, seed=42)
        except Exception as exc:
            can_fit = False
            can_eval = False
            can_explain = False
            elapsed = time.perf_counter() - t0
            print(f"  {algo:<22} {task:>5} {'N':>5} {'—':>6} {'—':>8} {elapsed:>10.3f}  ({exc})")
            results.append({"algorithm": algo, "task": task, "fit": False, "eval": False, "explain": False})
            continue

        try:
            ml.evaluate(model, split.valid)
        except Exception:
            can_eval = False

        try:
            ml.explain(model)
        except Exception:
            can_explain = False

        elapsed = time.perf_counter() - t0
        fit_s = "Y" if can_fit else "N"
        eval_s = "Y" if can_eval else "N"
        expl_s = "Y" if can_explain else "N"
        print(f"  {algo:<22} {task:>5} {fit_s:>5} {eval_s:>6} {expl_s:>8} {elapsed:>10.3f}")
        results.append({
            "algorithm": algo,
            "task": task,
            "fit": can_fit,
            "eval": can_eval,
            "explain": can_explain,
            "seconds": round(elapsed, 3),
        })

    working = sum(1 for r in results if r["fit"])
    total = len(results)
    print(f"\n  {working}/{total} algorithms fit + evaluate successfully")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    use_json = "--json" in sys.argv

    import sklearn

    hw = {
        "python": sys.version.split()[0],
        "ml_version": ml.__version__,
        "sklearn_version": sklearn.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    print("ml vs sklearn — side-by-side comparison")
    print(f"ml {ml.__version__} | sklearn {sklearn.__version__} | Python {hw['python']}")
    print(f"{hw['platform']}")

    r1 = test_code_complexity()
    r2 = test_timing()
    r3 = test_parity()
    r4 = test_coverage()

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    avg_ratio = np.mean([r["ratio"] for r in r1])
    avg_overhead = np.mean([r["overhead_pct"] for r in r2])
    parity_pass = all(r["pass"] for r in r3)
    algo_count = sum(1 for r in r4 if r["fit"])

    print(f"  Code reduction:    {avg_ratio:.1f}x fewer lines on average")
    print(f"  Wrapper overhead:  {avg_overhead:+.1f}% average vs raw sklearn")
    print(f"  Metric parity:     {'PASS' if parity_pass else 'FAIL'}")
    print(f"  Algorithm coverage: {algo_count}/{len(r4)} families")
    print("=" * 70)

    if use_json:
        out = {
            "hardware": hw,
            "code_complexity": r1,
            "timing": r2,
            "parity": r3,
            "coverage": r4,
            "summary": {
                "avg_code_ratio": round(avg_ratio, 1),
                "avg_overhead_pct": round(avg_overhead, 1),
                "parity_pass": parity_pass,
                "algo_coverage": f"{algo_count}/{len(r4)}",
            },
        }
        outpath = "benchmarks/bench_vs_sklearn.json"
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  JSON written to {outpath}")


if __name__ == "__main__":
    main()
