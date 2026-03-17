"""Benchmark all ml Rust algorithms vs sklearn equivalents.

Reproduces the homepage speed claims. Run on Linux x86_64.

    pip install ml-py scikit-learn numpy
    python3 benchmarks/bench_vs_sklearn.py

Environment: Python 3.12, sklearn 1.8, numpy 2.x, ml-py 1.0.0
Hardware:    AMD Ryzen 9 9900X (12c/24t), 32GB DDR5, RTX 5060 Ti
Protocol:   median of 5 runs after 1 warmup, N=10K, p=20
"""
import json
import platform
import sys
import time

import numpy as np
import sklearn
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet as SkElasticNet
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ml_py import (
    AdaBoost,
    DecisionTree,
    ElasticNet as RustEN,
    ExtraTrees,
    GradientBoosting,
    KNN,
    Linear as RustRidge,
    Logistic as RustLogistic,
    NaiveBayes,
    RandomForest,
    SvmClassifier,
)

# ── Data ──

np.random.seed(42)
N = 10_000
P = 20
X = np.random.randn(N, P).astype(np.float64)
signal = X[:, 0] + 0.8 * X[:, 1] - 0.6 * X[:, 2] + 0.4 * X[:, 3]
y_clf = (signal > 0).astype(np.int64)
y_reg = (signal + 0.5 * np.random.randn(N)).astype(np.float64)
X_tr = np.ascontiguousarray(X[:8000])
X_te = np.ascontiguousarray(X[8000:])
y_clf_tr, y_clf_te = y_clf[:8000], y_clf[8000:]
y_reg_tr, y_reg_te = y_reg[:8000], y_reg[8000:]

scaler = StandardScaler().fit(X_tr)
X_tr_s = np.ascontiguousarray(scaler.transform(X_tr))
X_te_s = np.ascontiguousarray(scaler.transform(X_te))

RUNS = 5


# ── Metrics ──

def acc(p, y):
    return float(np.mean(p == y))


def r2(p, y):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot)


# ── Harness ──

results = []


def bench(name, task, ml_fn, sk_fn):
    ml_fn()  # warmup
    sk_fn()
    ml_times, sk_times = [], []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        ml_met = ml_fn()
        ml_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        sk_met = sk_fn()
        sk_times.append(time.perf_counter() - t0)
    ml_ms = np.median(ml_times) * 1000
    sk_ms = np.median(sk_times) * 1000
    speedup = sk_ms / ml_ms if ml_ms > 0.001 else float("inf")
    results.append({
        "algorithm": name, "task": task,
        "ml_ms": round(ml_ms, 2), "sk_ms": round(sk_ms, 2),
        "speedup": round(speedup, 2),
        "ml_metric": round(ml_met, 4), "sk_metric": round(sk_met, 4),
    })


# ── Classification ──

def _ml_dt_clf():
    m = DecisionTree(seed=42)
    m.fit_clf(X_tr, y_clf_tr)
    return acc(np.array(m.predict_clf(X_te)), y_clf_te)

def _sk_dt_clf():
    m = DecisionTreeClassifier(random_state=42)
    m.fit(X_tr, y_clf_tr)
    return acc(m.predict(X_te), y_clf_te)

bench("Decision Tree", "clf", _ml_dt_clf, _sk_dt_clf)


def _ml_rf_clf():
    m = RandomForest(n_trees=100, seed=42)
    m.fit_clf(X_tr, y_clf_tr)
    return acc(np.array(m.predict_clf(X_te)), y_clf_te)

def _sk_rf_clf():
    m = RandomForestClassifier(100, random_state=42, n_jobs=-1)
    m.fit(X_tr, y_clf_tr)
    return acc(m.predict(X_te), y_clf_te)

bench("Random Forest", "clf", _ml_rf_clf, _sk_rf_clf)


def _ml_et_clf():
    m = ExtraTrees(n_trees=100, seed=42)
    m.fit_clf(X_tr, y_clf_tr)
    return acc(np.array(m.predict_clf(X_te)), y_clf_te)

def _sk_et_clf():
    m = ExtraTreesClassifier(100, random_state=42, n_jobs=-1)
    m.fit(X_tr, y_clf_tr)
    return acc(m.predict(X_te), y_clf_te)

bench("Extra Trees", "clf", _ml_et_clf, _sk_et_clf)


def _ml_gbt_clf():
    m = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3, seed=42)
    m.fit_clf(X_tr, y_clf_tr)
    return acc(np.array(m.predict_clf(X_te)), y_clf_te)

def _sk_gbt_clf():
    m = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    m.fit(X_tr, y_clf_tr)
    return acc(m.predict(X_te), y_clf_te)

bench("Gradient Boosting", "clf", _ml_gbt_clf, _sk_gbt_clf)


def _ml_ada_clf():
    m = AdaBoost(n_estimators=50, seed=42)
    m.fit_clf(X_tr, y_clf_tr)
    return acc(np.array(m.predict_clf(X_te)), y_clf_te)

def _sk_ada_clf():
    m = AdaBoostClassifier(n_estimators=50, random_state=42)
    m.fit(X_tr, y_clf_tr)
    return acc(m.predict(X_te), y_clf_te)

bench("AdaBoost", "clf", _ml_ada_clf, _sk_ada_clf)


def _ml_log_clf():
    m = RustLogistic(max_iter=200)
    m.fit(X_tr_s, y_clf_tr)
    return acc(np.array(m.predict(X_te_s)), y_clf_te)

def _sk_log_clf():
    m = LogisticRegression(max_iter=200)
    m.fit(X_tr_s, y_clf_tr)
    return acc(m.predict(X_te_s), y_clf_te)

bench("Logistic", "clf", _ml_log_clf, _sk_log_clf)


def _ml_knn_clf():
    m = KNN(k=5)
    m.fit_clf(X_tr_s, y_clf_tr)
    return acc(np.array(m.predict_clf(X_te_s)), y_clf_te)

def _sk_knn_clf():
    m = KNeighborsClassifier(5)
    m.fit(X_tr_s, y_clf_tr)
    return acc(m.predict(X_te_s), y_clf_te)

bench("KNN", "clf", _ml_knn_clf, _sk_knn_clf)


def _ml_nb_clf():
    m = NaiveBayes()
    m.fit_clf(X_tr, y_clf_tr)
    return acc(np.array(m.predict_clf(X_te)), y_clf_te)

def _sk_nb_clf():
    m = GaussianNB()
    m.fit(X_tr, y_clf_tr)
    return acc(m.predict(X_te), y_clf_te)

bench("Naive Bayes", "clf", _ml_nb_clf, _sk_nb_clf)


def _ml_svm_clf():
    m = SvmClassifier(c=1.0, max_iter=1000)
    m.fit(X_tr_s, y_clf_tr)
    return acc(np.array(m.predict_clf(X_te_s)), y_clf_te)

def _sk_svm_clf():
    m = LinearSVC(C=1.0, max_iter=1000, dual=True)
    m.fit(X_tr_s, y_clf_tr)
    return acc(m.predict(X_te_s), y_clf_te)

bench("SVM (linear)", "clf", _ml_svm_clf, _sk_svm_clf)

# ── Regression ──


def _ml_dt_reg():
    m = DecisionTree(seed=42)
    m.fit_reg(X_tr, y_reg_tr)
    return r2(np.array(m.predict_reg(X_te)), y_reg_te)

def _sk_dt_reg():
    m = DecisionTreeRegressor(random_state=42)
    m.fit(X_tr, y_reg_tr)
    return r2(m.predict(X_te), y_reg_te)

bench("Decision Tree", "reg", _ml_dt_reg, _sk_dt_reg)


def _ml_rf_reg():
    m = RandomForest(n_trees=100, seed=42)
    m.fit_reg(X_tr, y_reg_tr)
    return r2(np.array(m.predict_reg(X_te)), y_reg_te)

def _sk_rf_reg():
    m = RandomForestRegressor(100, random_state=42, n_jobs=-1)
    m.fit(X_tr, y_reg_tr)
    return r2(m.predict(X_te), y_reg_te)

bench("Random Forest", "reg", _ml_rf_reg, _sk_rf_reg)


def _ml_et_reg():
    m = ExtraTrees(n_trees=100, seed=42)
    m.fit_reg(X_tr, y_reg_tr)
    return r2(np.array(m.predict_reg(X_te)), y_reg_te)

def _sk_et_reg():
    m = ExtraTreesRegressor(100, random_state=42, n_jobs=-1)
    m.fit(X_tr, y_reg_tr)
    return r2(m.predict(X_te), y_reg_te)

bench("Extra Trees", "reg", _ml_et_reg, _sk_et_reg)


def _ml_gbt_reg():
    m = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3, seed=42)
    m.fit_reg(X_tr, y_reg_tr)
    return r2(np.array(m.predict_reg(X_te)), y_reg_te)

def _sk_gbt_reg():
    m = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    m.fit(X_tr, y_reg_tr)
    return r2(m.predict(X_te), y_reg_te)

bench("Gradient Boosting", "reg", _ml_gbt_reg, _sk_gbt_reg)


def _ml_ridge_reg():
    m = RustRidge(alpha=1.0)
    m.fit(X_tr_s, y_reg_tr)
    return r2(np.array(m.predict(X_te_s)), y_reg_te)

def _sk_ridge_reg():
    m = Ridge(alpha=1.0)
    m.fit(X_tr_s, y_reg_tr)
    return r2(m.predict(X_te_s), y_reg_te)

bench("Ridge", "reg", _ml_ridge_reg, _sk_ridge_reg)


def _ml_en_reg():
    m = RustEN(alpha=1.0, l1_ratio=0.5, max_iter=1000)
    m.fit(X_tr_s, y_reg_tr)
    return r2(np.array(m.predict(X_te_s)), y_reg_te)

def _sk_en_reg():
    m = SkElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000)
    m.fit(X_tr_s, y_reg_tr)
    return r2(m.predict(X_te_s), y_reg_te)

bench("Elastic Net", "reg", _ml_en_reg, _sk_en_reg)


def _ml_knn_reg():
    m = KNN(k=5)
    m.fit_reg(X_tr_s, y_reg_tr)
    return r2(np.array(m.predict_reg(X_te_s)), y_reg_te)

def _sk_knn_reg():
    m = KNeighborsRegressor(5)
    m.fit(X_tr_s, y_reg_tr)
    return r2(m.predict(X_te_s), y_reg_te)

bench("KNN", "reg", _ml_knn_reg, _sk_knn_reg)

# ── Output ──

meta = {
    "n": N, "p": P, "runs": RUNS,
    "python": sys.version.split()[0],
    "sklearn": sklearn.__version__,
    "numpy": np.__version__,
    "platform": platform.platform(),
    "cpu": platform.processor() or "unknown",
}

print()
print(f"Benchmark: N={N:,}, p={P}, median of {RUNS} runs")
print(f"Python {meta['python']}, sklearn {meta['sklearn']}, {meta['platform']}")
print()
hdr = f"| {'Algorithm':22s} | {'Task':4s} | {'ml':>10s} | {'sklearn':>10s} | {'Speedup':>9s} | {'ml':>6s} | {'sk':>6s} |"
sep = f"|{'-' * 24}|{'-' * 6}|{'-' * 12}|{'-' * 12}|{'-' * 11}|{'-' * 8}|{'-' * 8}|"
print(hdr)
print(sep)
for r in sorted(results, key=lambda x: -x["speedup"]):
    sp = f"{r['speedup']:.1f}x"
    print(
        f"| {r['algorithm']:22s} | {r['task']:4s} "
        f"| {r['ml_ms']:7.1f} ms | {r['sk_ms']:7.1f} ms "
        f"| {sp:>9s} | {r['ml_metric']:.4f} | {r['sk_metric']:.4f} |"
    )

# Save JSON for homepage consumption
output = {"meta": meta, "results": results}
with open("benchmarks/results_vs_sklearn.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved to benchmarks/results_vs_sklearn.json")
