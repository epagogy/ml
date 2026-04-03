"""The Leakage Landscape — Full Experimental Suite V4.

Maps where ML workflow errors matter across 2000+ classification datasets.
See paper for experimental design.

21 experiments (S+U removed as redundant Class I):
  A:  Normalization leakage — StandardScaler (global vs per-fold, 4 algos)
  A2: Normalization leakage — MinMaxScaler (global vs per-fold, 4 algos)
  A3: Normalization under outliers (inject 10% outliers into test, both scalers)
  B:  Peeking / model selection bias (max of K configs)
  C:  Feature selection leakage (global vs per-fold MI selection)
  D:  Split strategy (2-way CV vs 3-way holdout)
  E:  Imputation leakage (global vs per-fold mean impute)
  F:  Target encoding leakage (global vs per-fold)
  G:  Oversampling leakage (before vs after split, random duplication)
  H:  Duplicate leakage (injected at 0/5/10/30%, 4 algorithms: LR/RF/DT/KNN)
  J:  Compound effect (all applicable leakages stacked)
  K:  Tuning baseline (default vs tuned, all algorithms)
  L:  Model variance (repeated CV, overfit gap — mediating factor)
  P:  Grouped split leakage (KFold vs GroupKFold on k-means clusters)
  Q:  High-cardinality vocabulary leakage (global vs per-fold CountVectorizer)
  R:  Target proxy calibration (inject sigmoid(y*strength)+noise, dose-response)
  T:  Binning/discretization leakage (global KBinsDiscretizer vs per-fold)
  V:  Nested vs flat CV (GridSearchCV.best_score_ vs outer-fold, Cawley & Talbot)
  W:  Row-order detection (LR on row index, flags temporally-ordered datasets)

Sources: ml (116) + PMLB (162) + OpenML (2000+)
Resume: appends to JSONL, skips completed datasets
Backfill: P-W run on datasets completed before these experiments were added

Usage:
  nohup python3 run_leakage_landscape.py > landscape_log.txt 2>&1 &
"""

import warnings
import os
import time
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtri  # inverse normal CDF

warnings.filterwarnings("ignore")

# === CONFIG ===
# All output paths configurable via RESULTS_DIR env var (default: script directory)
_RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(Path(__file__).parent)))
RESULTS_FILE = str(_RESULTS_DIR / "leakage_landscape.jsonl")
REPORT_FILE = str(_RESULTS_DIR / "leakage_landscape_report.md")
CALIBRATION_FILE = str(_RESULTS_DIR / "leakage_calibration.json")
DATASET_CACHE_DIR = str(Path(os.environ.get("DATASET_CACHE_DIR", str(Path.home() / ".dataset_cache"))))
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
MAX_ROWS = 10000
MAX_FEATURES = 2000
MIN_ROWS = 100
CV_FOLDS = 5
N_REPS = 5
SEED = 42

RUN_CALIBRATION = True
RUN_A = True   # Normalization (StandardScaler)
RUN_A2 = True  # Normalization (MinMaxScaler)
RUN_A3 = True  # Normalization under outliers (both scalers)
RUN_B = True   # Peeking
RUN_C = True   # Feature selection (p > 20 only)
RUN_D = True   # Split strategy
RUN_E = True   # Imputation
RUN_F = True   # Target encoding (categorical only)
RUN_G = True   # Oversampling (imbalanced only)
RUN_H = True   # Duplicates
RUN_J = True   # Compound
RUN_K = True   # Tuning baseline
RUN_L = True   # Model variance
RUN_P = True   # Grouped split leakage
RUN_Q = True   # High-cardinality vocabulary leakage
RUN_R = True   # Target proxy calibration (sensitivity analysis)
RUN_T = True   # Binning/discretization leakage
RUN_V = True   # Nested vs flat CV (Cawley & Talbot 2010)
RUN_W = True   # Row-order detection


# ============================================================
# Utilities
# ============================================================

def get_completed():
    """Load already-completed dataset keys from JSONL."""
    done = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    done.add(obj.get("name", "") + "|" + obj.get("source", ""))
                except Exception:
                    pass
    return done


def get_needs_backfill():
    """Find datasets completed but missing A2/A3 columns."""
    needs = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    key = obj.get("name", "") + "|" + obj.get("source", "")
                    if obj.get("status") == "ok" and "a2_lr_gap_diff" not in obj:
                        needs.add(key)
                except Exception:
                    pass
    return needs


def get_needs_backfill_new():
    """Find datasets completed but missing P-W experiment columns."""
    needs = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    key = obj.get("name", "") + "|" + obj.get("source", "")
                    if obj.get("status") == "ok" and "p_gap_lr" not in obj:
                        needs.add(key)
                except Exception:
                    pass
    # Exclude datasets already backfilled
    if os.path.exists(BACKFILL_NEW_FILE):
        with open(BACKFILL_NEW_FILE) as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    key = obj.get("name", "") + "|" + obj.get("source", "")
                    needs.discard(key)
                except Exception:
                    pass
    return needs


BACKFILL_FILE = str(_RESULTS_DIR / "leakage_landscape_a2a3_backfill.jsonl")
BACKFILL_NEW_FILE = str(_RESULTS_DIR / "leakage_landscape_pqrtvw_backfill.jsonl")


def save_backfill(result):
    """Append one backfill result to supplemental JSONL."""
    with open(BACKFILL_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


def save_backfill_new(result):
    """Append one P-W backfill result to supplemental JSONL."""
    with open(BACKFILL_NEW_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


def save_result(result):
    """Append one result to JSONL."""
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


def assign_split(name, source):
    """Deterministic 50/50 discovery/confirmation split via hash (for A-L experiments)."""
    h = hashlib.md5(f"{name}|{source}".encode()).hexdigest()
    return "discovery" if int(h[:8], 16) % 2 == 0 else "confirmation"


def assign_split_pw(name, source):
    """Separate 50/50 split for P-W hypotheses (independent of A-L split).
    Uses different hash seed to ensure independence from A-L split assignment."""
    h = hashlib.md5(f"{name}|{source}|pw".encode()).hexdigest()
    return "discovery" if int(h[:8], 16) % 2 == 0 else "confirmation"


def prepare_binary(X, y):
    """Ensure binary classification. Top-2 classes if multiclass."""
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


def safe_subsample(X, y, max_rows=MAX_ROWS):
    """Stratified subsample if too large."""
    if len(y) > max_rows:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(y), size=max_rows, replace=False)
        return X[idx], y[idx]
    return X, y


def safe_auc(y_true, y_score):
    """AUC that returns 0.5 on failure."""
    from sklearn.metrics import roc_auc_score
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_score)
    except Exception:
        return 0.5


def fit_score(clf, X_tr, y_tr, X_te, y_te):
    """Fit model, return test AUC. Returns 0.5 on failure."""
    try:
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        return safe_auc(y_te, proba)
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


def make_knn():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    return Pipeline([("s", StandardScaler()),
                     ("m", KNeighborsClassifier(n_neighbors=5))])


def find_categorical_indices(X, min_card=5, max_card=50):
    """Find columns that look categorical (few unique integer-like values)."""
    cat_idx = []
    for j in range(X.shape[1]):
        unique = np.unique(X[:, j][~np.isnan(X[:, j])])
        if min_card <= len(unique) <= max_card:
            if np.allclose(unique, unique.astype(int)):
                cat_idx.append(j)
    return cat_idx


# ============================================================
# Theoretical Bounds
# ============================================================

def compute_theory(meta):
    """Compute theoretical leakage predictions for a dataset.

    Returns dict of theoretical deltas keyed by experiment.
    """
    n = meta["n_rows"]
    p = meta["n_features"]
    K = CV_FOLDS
    imbal = meta["imbalance"]
    n_test = int(n * 0.2)

    theory = {}

    # A: Normalization — p/n × K/(K-1)
    theory["a_theory"] = round(p / n * K / (K - 1), 6)

    # B: Peeking — σ√(2 ln K) where σ ≈ √(0.25 / n_test)
    sigma = np.sqrt(0.25 / max(n_test, 1))
    for k in [1, 2, 5, 10, 15, 25, 50]:
        theory[f"b_theory_k{k}"] = round(sigma * np.sqrt(2 * np.log(max(k, 1))), 6)

    # C: Feature selection — k × [Φ⁻¹(1 - k/(2p))]² / n
    k_sel = min(10, max(1, p // 2))
    if p > 2 * k_sel:
        alpha = k_sel / (2 * p)
        z = ndtri(1 - alpha)
        theory["c_theory"] = round(k_sel * z**2 / n, 6)
    else:
        theory["c_theory"] = 0.0

    # D: No clean formula (selection bias, similar to B)
    theory["d_theory"] = round(sigma * np.sqrt(2 * np.log(3)), 6)  # 3 algorithms

    # E: Imputation — m × p/n × K/(K-1)
    for m in [0.1, 0.3]:
        theory[f"e_theory_{int(m*100)}"] = round(m * p / n * K / (K - 1), 6)

    # F: Target encoding — C/(n + C×m) in R², approximate as C/n
    # C = max cardinality (unknown here, estimated from n_cat)
    theory["f_theory"] = round(20 / n, 6) if meta["n_cat"] > 0 else 0.0

    # G: Random Oversampling — (1-(1-α)²) × n_min/n × (1-AUC_true)
    # Approximate: α=0.5, AUC_true=0.7
    n_min_frac = imbal
    theory["g_theory"] = round(0.75 * n_min_frac * 0.3, 6) if imbal < 0.3 else 0.0

    # H: Duplicates — d × (a_train - a_true) ≈ d × 0.15
    for d in [0.05, 0.10, 0.30]:
        theory[f"h_theory_{int(d*100)}"] = round(d * 0.15, 6)

    # V: Nested vs flat CV — Cawley & Talbot: σ√(2 ln K) where K=5 C values
    # Same formula as peeking (B) but applied to inner CV inflation
    theory["v_theory"] = round(sigma * np.sqrt(2 * np.log(5)), 6)

    return theory


# ============================================================
# Dataset Collection
# ============================================================

def collect_ml_datasets():
    """All classification datasets from ml.datasets()."""
    import ml
    catalog = ml.datasets()
    clf = catalog[catalog["task"] == "classification"]
    datasets = []
    for _, row in clf.iterrows():
        name = str(row["name"])
        target = str(row["target"])
        try:
            data = ml.dataset(name)
            if target not in data.columns:
                continue
            datasets.append({
                "name": name,
                "source": "ml",
                "loader": lambda n=name, t=target: _load_ml(n, t),
            })
        except Exception:
            pass
    return datasets


def _load_ml(name, target):
    import ml
    data = ml.dataset(name)
    y = data[target].values
    X_df = data.drop(columns=[target])
    cat_cols = X_df.select_dtypes(include=["object", "category", "string"]).columns
    n_cat = len(cat_cols)
    nan_pct = X_df.isna().mean().mean()
    for col in cat_cols:
        X_df[col] = X_df[col].astype("category").cat.codes
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
    X = X_df.values.astype(float)
    return X, y, n_cat, nan_pct


def collect_pmlb_datasets():
    """All classification datasets from PMLB."""
    import pmlb
    datasets = []
    for name in pmlb.classification_dataset_names:
        datasets.append({
            "name": name,
            "source": "pmlb",
            "loader": lambda n=name: _load_pmlb(n),
        })
    return datasets


def _load_pmlb(name):
    import pmlb
    data = pmlb.fetch_data(name, local_cache_dir=str(Path.home() / ".pmlb_cache"))
    if "target" not in data.columns:
        return None, None, 0, 0.0
    y = data["target"].values
    X_df = data.drop(columns=["target"])
    cat_cols = X_df.select_dtypes(include=["object", "category", "string"]).columns
    n_cat = len(cat_cols)
    nan_pct = X_df.isna().mean().mean()
    for col in cat_cols:
        X_df[col] = X_df[col].astype("category").cat.codes
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
    X = X_df.values.astype(float)
    return X, y, n_cat, nan_pct


def collect_openml_datasets():
    """Classification datasets from OpenML (100-500K rows, 2-2000 features)."""
    import openml
    catalog = openml.datasets.list_datasets(output_format="dataframe")
    clf = catalog[
        (catalog["NumberOfClasses"] >= 2) &
        (catalog["NumberOfInstances"] >= MIN_ROWS) &
        (catalog["NumberOfInstances"] <= 500000) &
        (catalog["NumberOfFeatures"] >= 2) &
        (catalog["NumberOfFeatures"] <= MAX_FEATURES) &
        (catalog["status"] == "active")
    ].copy()
    datasets = []
    for _, row in clf.iterrows():
        did = int(row["did"])
        name = str(row["name"])
        datasets.append({
            "name": f"openml_{did}_{name}",
            "source": "openml",
            "openml_id": did,
            "loader": lambda d=did: _load_openml(d),
        })
    return datasets


def _load_openml(did):
    """Load OpenML dataset, using local numpy cache to skip re-download/re-parse."""
    cache_path = os.path.join(DATASET_CACHE_DIR, f"openml_{did}.npz")

    # Cache hit — pure disk read, no network
    if os.path.exists(cache_path):
        try:
            d = np.load(cache_path, allow_pickle=False)
            X = d["X"]
            y = d["y"]
            n_cat = int(d["n_cat"])
            nan_pct = float(d["nan_pct"])
            return X, y, n_cat, nan_pct
        except Exception:
            os.remove(cache_path)  # corrupt cache — fall through to re-download

    # Cache miss — download, parse, save
    import openml
    ds = openml.datasets.get_dataset(did, download_data=True,
                                      download_qualities=False,
                                      download_features_meta_data=False)
    X, y, cat_mask, attr_names = ds.get_data(target=ds.default_target_attribute)
    if X is None or y is None:
        return None, None, 0, 0.0
    if isinstance(X, pd.DataFrame):
        cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns
        n_cat = len(cat_cols)
        nan_pct = X.isna().mean().mean()
        for col in cat_cols:
            X[col] = X[col].astype("category").cat.codes
        X = X.fillna(X.median(numeric_only=True)).fillna(0)
        X = X.values.astype(float)
    else:
        n_cat = sum(cat_mask) if cat_mask is not None else 0
        nan_pct = np.isnan(X.astype(float)).mean() if X.dtype != object else 0.0
        X = np.nan_to_num(X.astype(float), nan=0.0)
    if isinstance(y, pd.Series):
        y = y.values

    # Save to cache (suppress errors — cache failure is non-fatal)
    try:
        np.savez_compressed(cache_path, X=X, y=y,
                            n_cat=np.array(n_cat),
                            nan_pct=np.array(nan_pct))
    except Exception:
        pass

    return X, y, n_cat, nan_pct


# ============================================================
# Simulation Calibration
# ============================================================

def run_calibration():
    """Verify measurement protocol on synthetic data with known leakage."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score

    configs = [
        {"n": 200, "p": 50},   # high p/n
        {"n": 1000, "p": 50},  # moderate p/n
        {"n": 5000, "p": 50},  # low p/n
        {"n": 200, "p": 200},  # very high p/n
        {"n": 5000, "p": 5},   # very low p/n
    ]

    results = []
    for cfg in configs:
        n, p = cfg["n"], cfg["p"]
        rng = np.random.RandomState(SEED)
        X = rng.randn(n, p)
        beta = np.zeros(p)
        beta[:5] = rng.randn(5) * 0.5
        prob = 1 / (1 + np.exp(-X @ beta))
        y = (rng.rand(n) < prob).astype(int)

        X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                    random_state=SEED, stratify=y)
        # Global normalization (leaky)
        scaler = StandardScaler().fit(np.vstack([X_tv, X_te]))
        X_tv_g = scaler.transform(X_tv)
        X_te_g = scaler.transform(X_te)
        cv_g = cross_val_score(LogisticRegression(max_iter=1000, random_state=SEED),
                               X_tv_g, y_tv, cv=CV_FOLDS, scoring="roc_auc")
        lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_tv_g, y_tv)
        test_g = roc_auc_score(y_te, lr.predict_proba(X_te_g)[:, 1])
        gap_g = float(np.mean(cv_g) - test_g)

        # Per-fold normalization (clean)
        pipe = Pipeline([("s", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=1000, random_state=SEED))])
        cv_pf = cross_val_score(pipe, X_tv, y_tv, cv=CV_FOLDS, scoring="roc_auc")
        pipe.fit(X_tv, y_tv)
        test_pf = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        gap_pf = float(np.mean(cv_pf) - test_pf)

        measured = gap_g - gap_pf
        theoretical = p / n * CV_FOLDS / (CV_FOLDS - 1)

        results.append({
            "n": n, "p": p, "p_over_n": round(p / n, 4),
            "measured": round(measured, 6),
            "theoretical": round(theoretical, 6),
            "ratio": round(measured / theoretical, 3) if theoretical > 0.001 else None,
        })

    with open(CALIBRATION_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SIMULATION CALIBRATION ===")
    print(f"{'n':>6} {'p':>4} {'p/n':>6} {'measured':>10} {'theory':>10} {'ratio':>7}")
    for r in results:
        ratio_str = f"{r['ratio']:.2f}" if r["ratio"] is not None else "N/A"
        print(f"{r['n']:>6} {r['p']:>4} {r['p_over_n']:>6.3f} "
              f"{r['measured']:>+10.4f} {r['theoretical']:>10.4f} {ratio_str:>7}")

    return results


# ============================================================
# Experiment A: Normalization Leakage
# ============================================================

def exp_a(X, y, meta, seed=SEED):
    """Global vs per-fold normalization. 4 algorithms: LR, RF, DT, KNN.
    Also captures fold-level variance for mediation analysis."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score

    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=seed, stratify=y)
    results = {}

    algos = [
        ("lr", lambda: LogisticRegression(max_iter=1000, random_state=seed)),
        ("rf", lambda: RandomForestClassifier(n_estimators=50, random_state=seed)),
        ("dt", lambda: DecisionTreeClassifier(random_state=seed)),
        ("knn", lambda: KNeighborsClassifier(n_neighbors=5)),
    ]

    for algo_name, make_model in algos:
        # Global scaling (leaky)
        scaler = StandardScaler().fit(np.vstack([X_tv, X_te]))
        X_tv_s = scaler.transform(X_tv)
        X_te_s = scaler.transform(X_te)
        try:
            cv_g = cross_val_score(make_model(), X_tv_s, y_tv, cv=CV_FOLDS, scoring="roc_auc")
            m = make_model().fit(X_tv_s, y_tv)
            test_g = roc_auc_score(y_te, m.predict_proba(X_te_s)[:, 1])
            gap_g = float(np.mean(cv_g) - test_g)
        except Exception:
            gap_g = None

        # Per-fold scaling (clean) — model inside pipeline
        try:
            pipe = Pipeline([("s", StandardScaler()), ("m", make_model())])
            cv_pf = cross_val_score(pipe, X_tv, y_tv, cv=CV_FOLDS, scoring="roc_auc")
            pipe.fit(X_tv, y_tv)
            test_pf = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
            gap_pf = float(np.mean(cv_pf) - test_pf)
        except Exception:
            gap_pf = None
            cv_pf = None

        if gap_g is not None and gap_pf is not None:
            results[f"a_{algo_name}_gap_global"] = round(gap_g, 6)
            results[f"a_{algo_name}_gap_perfold"] = round(gap_pf, 6)
            results[f"a_{algo_name}_gap_diff"] = round(gap_g - gap_pf, 6)
            # Model variance = std of CV fold scores (mediator signal)
            results[f"a_{algo_name}_cv_std"] = round(float(np.std(cv_pf)), 6)

    return results if results else None


# ============================================================
# Experiment A2: MinMaxScaler Normalization Leakage
# ============================================================

def exp_a2(X, y, meta, seed=SEED):
    """Global vs per-fold MinMaxScaler. 4 algorithms: LR, RF, DT, KNN.
    MinMaxScaler is more sensitive to outliers than StandardScaler because
    a single test-set outlier can compress the entire feature range."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score

    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=seed, stratify=y)
    results = {}

    algos = [
        ("lr", lambda: LogisticRegression(max_iter=1000, random_state=seed)),
        ("rf", lambda: RandomForestClassifier(n_estimators=50, random_state=seed)),
        ("dt", lambda: DecisionTreeClassifier(random_state=seed)),
        ("knn", lambda: KNeighborsClassifier(n_neighbors=5)),
    ]

    for algo_name, make_model in algos:
        # Global scaling (leaky)
        scaler = MinMaxScaler().fit(np.vstack([X_tv, X_te]))
        X_tv_s = scaler.transform(X_tv)
        X_te_s = scaler.transform(X_te)
        try:
            cv_g = cross_val_score(make_model(), X_tv_s, y_tv, cv=CV_FOLDS, scoring="roc_auc")
            m = make_model().fit(X_tv_s, y_tv)
            test_g = roc_auc_score(y_te, m.predict_proba(X_te_s)[:, 1])
            gap_g = float(np.mean(cv_g) - test_g)
        except Exception:
            gap_g = None

        # Per-fold scaling (clean)
        try:
            pipe = Pipeline([("s", MinMaxScaler()), ("m", make_model())])
            cv_pf = cross_val_score(pipe, X_tv, y_tv, cv=CV_FOLDS, scoring="roc_auc")
            pipe.fit(X_tv, y_tv)
            test_pf = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
            gap_pf = float(np.mean(cv_pf) - test_pf)
        except Exception:
            gap_pf = None

        if gap_g is not None and gap_pf is not None:
            results[f"a2_{algo_name}_gap_global"] = round(gap_g, 6)
            results[f"a2_{algo_name}_gap_perfold"] = round(gap_pf, 6)
            results[f"a2_{algo_name}_gap_diff"] = round(gap_g - gap_pf, 6)

    return results if results else None


# ============================================================
# Experiment A3: Normalization Under Outliers
# ============================================================

def exp_a3(X, y, meta, seed=SEED):
    """Test normalization leakage when test set contains outliers.
    Injects 10% outlier samples into test features (multiply by 10).
    Tests both StandardScaler and MinMaxScaler with LR and KNN."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score

    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=seed, stratify=y)

    # Inject outliers: multiply 10% of test rows by 10
    rng = np.random.RandomState(seed)
    n_outliers = max(1, int(0.1 * len(X_te)))
    outlier_idx = rng.choice(len(X_te), size=n_outliers, replace=False)
    X_te_out = X_te.copy()
    X_te_out[outlier_idx] *= 10.0

    results = {}

    scalers = [("std", StandardScaler), ("mm", MinMaxScaler)]
    algos = [
        ("lr", lambda: LogisticRegression(max_iter=1000, random_state=seed)),
        ("knn", lambda: KNeighborsClassifier(n_neighbors=5)),
    ]

    for sc_name, ScalerCls in scalers:
        for algo_name, make_model in algos:
            # Global scaling on data WITH outliers (leaky)
            scaler = ScalerCls().fit(np.vstack([X_tv, X_te_out]))
            X_tv_s = scaler.transform(X_tv)
            X_te_s = scaler.transform(X_te_out)
            try:
                m = make_model().fit(X_tv_s, y_tv)
                auc_g = roc_auc_score(y_te, m.predict_proba(X_te_s)[:, 1])
            except Exception:
                auc_g = None

            # Per-fold scaling (correct — scaler fit on train only)
            try:
                scaler_pf = ScalerCls().fit(X_tv)
                X_tv_pf = scaler_pf.transform(X_tv)
                X_te_pf = scaler_pf.transform(X_te_out)
                m2 = make_model().fit(X_tv_pf, y_tv)
                auc_pf = roc_auc_score(y_te, m2.predict_proba(X_te_pf)[:, 1])
            except Exception:
                auc_pf = None

            if auc_g is not None and auc_pf is not None:
                results[f"a3_{sc_name}_{algo_name}_global"] = round(auc_g, 6)
                results[f"a3_{sc_name}_{algo_name}_perfold"] = round(auc_pf, 6)
                results[f"a3_{sc_name}_{algo_name}_diff"] = round(auc_g - auc_pf, 6)

    return results if results else None


# ============================================================
# Experiment B: Peeking / Model Selection Bias
# ============================================================

def exp_b(X, y, meta, seed=SEED):
    """Peeking inflation: try K configs, report max - honest."""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    configs = []
    for n_est in [10, 50, 100]:
        for max_d in [3, 10, None]:
            configs.append(("rf", {"n_estimators": n_est, "max_depth": max_d}))
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        configs.append(("lr", {"C": C}))
    for max_d in [2, 3, 5, 10, None]:
        configs.append(("dt", {"max_depth": max_d}))
    # Total: 9 RF + 5 LR + 5 DT = 19 configs

    peek_counts = [1, 2, 5, 10, 15, 19]
    inflations = {k: [] for k in peek_counts}

    for rep in range(N_REPS):
        rs = seed + rep
        try:
            X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                        random_state=rs, stratify=y)
        except Exception:
            continue

        aucs = []
        for atype, params in configs:
            try:
                if atype == "rf":
                    clf = RandomForestClassifier(random_state=rs, **params)
                elif atype == "lr":
                    clf = Pipeline([("s", StandardScaler()),
                                    ("m", LogisticRegression(max_iter=1000, random_state=rs, **params))])
                else:
                    clf = DecisionTreeClassifier(random_state=rs, **params)
                clf.fit(X_tv, y_tv)
                aucs.append(safe_auc(y_te, clf.predict_proba(X_te)[:, 1]))
            except Exception:
                pass

        if len(aucs) < 3:
            continue

        honest = aucs[np.random.RandomState(rs * 1000).randint(len(aucs))]
        for k in peek_counts:
            ka = min(k, len(aucs))
            idx = np.random.RandomState(rs * 100 + k).choice(len(aucs), size=ka, replace=False)
            inflations[k].append(max(aucs[i] for i in idx) - honest)

    results = {}
    for k, v in inflations.items():
        if v:
            results[f"b_infl_k{k}"] = round(np.mean(v), 6)
    return results if results else None


# ============================================================
# Experiment C: Feature Selection Leakage
# ============================================================

def exp_c(X, y, meta, seed=SEED):
    """Global vs per-fold MI feature selection. Only for p > 20."""
    if meta["n_features"] <= 20:
        return None

    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score

    k_features = min(10, meta["n_features"] // 2)
    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=seed, stratify=y)
    results = {}

    for algo_name, make_pipe in [
        ("lr", lambda: Pipeline([("s", StandardScaler()),
                                  ("m", LogisticRegression(max_iter=1000, random_state=seed))])),
        ("rf", lambda: Pipeline([("m", RandomForestClassifier(n_estimators=50, random_state=seed))]))
    ]:
        # Global MI selection (leaky): compute MI on ALL data including test
        try:
            X_all = np.vstack([X_tv, X_te])
            y_all = np.concatenate([y_tv, y_te])
            selector = SelectKBest(mutual_info_classif, k=k_features).fit(X_all, y_all)
            X_tv_sel = selector.transform(X_tv)
            X_te_sel = selector.transform(X_te)

            cv_g = cross_val_score(make_pipe(), X_tv_sel, y_tv, cv=CV_FOLDS, scoring="roc_auc")
            m = make_pipe().fit(X_tv_sel, y_tv)
            test_g = roc_auc_score(y_te, m.predict_proba(X_te_sel)[:, 1])
            gap_g = float(np.mean(cv_g) - test_g)
        except Exception:
            gap_g = None

        # Per-fold MI selection (clean): MI + model in pipeline
        try:
            pipe = Pipeline([
                ("sel", SelectKBest(mutual_info_classif, k=k_features)),
                *make_pipe().steps
            ])
            cv_pf = cross_val_score(pipe, X_tv, y_tv, cv=CV_FOLDS, scoring="roc_auc")
            pipe.fit(X_tv, y_tv)
            test_pf = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
            gap_pf = float(np.mean(cv_pf) - test_pf)
        except Exception:
            gap_pf = None

        if gap_g is not None and gap_pf is not None:
            results[f"c_{algo_name}_gap_global"] = round(gap_g, 6)
            results[f"c_{algo_name}_gap_perfold"] = round(gap_pf, 6)
            results[f"c_{algo_name}_gap_diff"] = round(gap_g - gap_pf, 6)

    return results if results else None


# ============================================================
# Experiment D: Split Strategy (2-way vs 3-way)
# ============================================================

def exp_d(X, y, meta, seed=SEED):
    """2-way CV vs 3-way holdout for model selection."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score

    def algos(rs):
        return [
            ("lr", Pipeline([("s", StandardScaler()),
                              ("m", LogisticRegression(max_iter=1000, random_state=rs))])),
            ("dt", DecisionTreeClassifier(random_state=rs)),
            ("rf", RandomForestClassifier(n_estimators=50, random_state=rs)),
        ]

    paired = []
    for rep in range(N_REPS):
        rs = seed + rep
        try:
            X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                        random_state=rs, stratify=y)
            X_tr, X_val, y_tr, y_val = train_test_split(X_tv, y_tv, test_size=0.25,
                                                          random_state=rs, stratify=y_tv)
            X_dev = np.vstack([X_tr, X_val])
            y_dev = np.concatenate([y_tr, y_val])

            # 2-way: CV on full trainval, pick best
            best_cv, best2 = -1, None
            for nm, clf in algos(rs):
                try:
                    sc = np.mean(cross_val_score(clf, X_tv, y_tv, cv=CV_FOLDS, scoring="roc_auc"))
                    if sc > best_cv:
                        best_cv, best2 = sc, nm
                except Exception:
                    pass

            t2 = None
            if best2:
                for nm, clf in algos(rs):
                    if nm == best2:
                        t2 = fit_score(clf, X_tv, y_tv, X_te, y_te)
                        break

            # 3-way: select on validation set
            best_v, best3 = -1, None
            for nm, clf in algos(rs):
                try:
                    clf.fit(X_tr, y_tr)
                    va = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
                    if va > best_v:
                        best_v, best3 = va, nm
                except Exception:
                    pass

            t3 = None
            if best3:
                for nm, clf in algos(rs):
                    if nm == best3:
                        t3 = fit_score(clf, X_dev, y_dev, X_te, y_te)
                        break

            if t2 is not None and t3 is not None:
                paired.append({"g2": best_cv - t2, "g3": best_v - t3, "t2": t2, "t3": t3})
        except Exception:
            pass

    if not paired:
        return None
    df = pd.DataFrame(paired)
    return {
        "d_gap_2way": round(df["g2"].mean(), 6),
        "d_gap_3way": round(df["g3"].mean(), 6),
        "d_gap_diff": round((df["g2"] - df["g3"]).mean(), 6),
        "d_test_2way": round(df["t2"].mean(), 6),
        "d_test_3way": round(df["t3"].mean(), 6),
        "d_n_reps": len(df),
    }


# ============================================================
# Experiment E: Imputation Leakage
# ============================================================

def exp_e(X, y, meta, seed=SEED):
    """Inject MCAR missingness, global vs per-fold mean impute."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score

    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=seed, stratify=y)
    results = {}

    for miss_rate in [0.10, 0.30]:
        # Inject MCAR missingness
        rng = np.random.RandomState(seed + int(miss_rate * 100))
        mask_tv = rng.rand(*X_tv.shape) < miss_rate
        mask_te = rng.rand(*X_te.shape) < miss_rate
        X_tv_miss = X_tv.copy().astype(float)
        X_te_miss = X_te.copy().astype(float)
        X_tv_miss[mask_tv] = np.nan
        X_te_miss[mask_te] = np.nan

        mr_key = int(miss_rate * 100)

        for algo_name, make_model in [
            ("lr", lambda: Pipeline([("s", StandardScaler()),
                                      ("m", LogisticRegression(max_iter=1000, random_state=seed))])),
            ("rf", lambda: Pipeline([("m", RandomForestClassifier(n_estimators=50, random_state=seed))]))
        ]:
            # Global imputation (leaky)
            try:
                X_all_miss = np.vstack([X_tv_miss, X_te_miss])
                imp = SimpleImputer(strategy="mean").fit(X_all_miss)
                X_tv_imp = imp.transform(X_tv_miss)
                X_te_imp = imp.transform(X_te_miss)

                cv_g = cross_val_score(make_model(), X_tv_imp, y_tv, cv=CV_FOLDS, scoring="roc_auc")
                m = make_model().fit(X_tv_imp, y_tv)
                test_g = roc_auc_score(y_te, m.predict_proba(X_te_imp)[:, 1])
                gap_g = float(np.mean(cv_g) - test_g)
            except Exception:
                gap_g = None

            # Per-fold imputation (clean)
            try:
                pipe = Pipeline([
                    ("imp", SimpleImputer(strategy="mean")),
                    *make_model().steps
                ])
                cv_pf = cross_val_score(pipe, X_tv_miss, y_tv, cv=CV_FOLDS, scoring="roc_auc")
                pipe.fit(X_tv_miss, y_tv)
                test_pf = roc_auc_score(y_te, pipe.predict_proba(X_te_miss)[:, 1])
                gap_pf = float(np.mean(cv_pf) - test_pf)
            except Exception:
                gap_pf = None

            if gap_g is not None and gap_pf is not None:
                results[f"e_{algo_name}_gap_diff_{mr_key}"] = round(gap_g - gap_pf, 6)

    return results if results else None


# ============================================================
# Experiment F: Target Encoding Leakage
# ============================================================

def exp_f(X, y, meta, seed=SEED):
    """Global vs per-fold target encoding on categorical features."""
    cat_idx = find_categorical_indices(X)
    if len(cat_idx) == 0:
        return None

    try:
        from sklearn.preprocessing import TargetEncoder
    except ImportError:
        return None  # sklearn < 1.3

    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import roc_auc_score

    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=seed, stratify=y)
    num_idx = [i for i in range(X.shape[1]) if i not in cat_idx]
    results = {}

    max_card = max(len(np.unique(X[:, j])) for j in cat_idx)
    results["f_max_cardinality"] = int(max_card)

    for algo_name, make_model in [
        ("lr", lambda: Pipeline([("s", StandardScaler()),
                                  ("m", LogisticRegression(max_iter=1000, random_state=seed))])),
        ("rf", lambda: Pipeline([("m", RandomForestClassifier(n_estimators=50, random_state=seed))]))
    ]:
        # Global target encoding (leaky)
        try:
            X_all = np.vstack([X_tv, X_te])
            y_all = np.concatenate([y_tv, y_te])
            te = TargetEncoder(categories="auto", smooth="auto")
            X_all_cat = X_all[:, cat_idx]
            te.fit(X_all_cat, y_all)
            X_tv_enc = np.hstack([X_tv[:, num_idx], te.transform(X_tv[:, cat_idx])])
            X_te_enc = np.hstack([X_te[:, num_idx], te.transform(X_te[:, cat_idx])])

            cv_g = cross_val_score(make_model(), X_tv_enc, y_tv, cv=CV_FOLDS, scoring="roc_auc")
            m = make_model().fit(X_tv_enc, y_tv)
            test_g = roc_auc_score(y_te, m.predict_proba(X_te_enc)[:, 1])
            gap_g = float(np.mean(cv_g) - test_g)
        except Exception:
            gap_g = None

        # Per-fold target encoding (clean) — TargetEncoder in pipeline
        try:
            ct = ColumnTransformer([
                ("te", TargetEncoder(categories="auto", smooth="auto"), cat_idx),
                ("num", "passthrough", num_idx),
            ])
            pipe = Pipeline([("ct", ct), *make_model().steps])
            cv_pf = cross_val_score(pipe, X_tv, y_tv, cv=CV_FOLDS, scoring="roc_auc")
            pipe.fit(X_tv, y_tv)
            test_pf = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
            gap_pf = float(np.mean(cv_pf) - test_pf)
        except Exception:
            gap_pf = None

        if gap_g is not None and gap_pf is not None:
            results[f"f_{algo_name}_gap_diff"] = round(gap_g - gap_pf, 6)

    return results if results else None


# ============================================================
# Experiment G: Oversampling Leakage
# ============================================================

def exp_g(X, y, meta, seed=SEED):
    """Oversample minority before vs after split. Only for imbalanced data."""
    if meta["imbalance"] >= 0.30:
        return None

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    results = {}

    for rep in range(N_REPS):
        rs = seed + rep
        rng = np.random.RandomState(rs)

        # Method: random oversampling (duplicate minority to match majority)
        minority_class = 1 if np.mean(y == 1) < 0.5 else 0
        majority_class = 1 - minority_class

        # LEAKY: oversample BEFORE split
        try:
            min_idx = np.where(y == minority_class)[0]
            maj_count = np.sum(y == majority_class)
            n_oversample = maj_count - len(min_idx)
            if n_oversample <= 0:
                continue
            dup_idx = rng.choice(min_idx, size=n_oversample, replace=True)
            X_over = np.vstack([X, X[dup_idx]])
            y_over = np.concatenate([y, y[dup_idx]])

            X_tv_l, X_te_l, y_tv_l, y_te_l = train_test_split(
                X_over, y_over, test_size=0.2, random_state=rs, stratify=y_over)

            clf = Pipeline([("s", StandardScaler()),
                            ("m", LogisticRegression(max_iter=1000, random_state=rs))])
            auc_leaky = fit_score(clf, X_tv_l, y_tv_l, X_te_l, y_te_l)
        except Exception:
            auc_leaky = None

        # CLEAN: oversample AFTER split (only training set)
        try:
            X_tv_c, X_te_c, y_tv_c, y_te_c = train_test_split(
                X, y, test_size=0.2, random_state=rs, stratify=y)

            min_idx_tr = np.where(y_tv_c == minority_class)[0]
            maj_count_tr = np.sum(y_tv_c == majority_class)
            n_over_tr = maj_count_tr - len(min_idx_tr)
            if n_over_tr > 0:
                dup_tr = rng.choice(min_idx_tr, size=n_over_tr, replace=True)
                X_tv_aug = np.vstack([X_tv_c, X_tv_c[dup_tr]])
                y_tv_aug = np.concatenate([y_tv_c, y_tv_c[dup_tr]])
            else:
                X_tv_aug, y_tv_aug = X_tv_c, y_tv_c

            clf = Pipeline([("s", StandardScaler()),
                            ("m", LogisticRegression(max_iter=1000, random_state=rs))])
            auc_clean = fit_score(clf, X_tv_aug, y_tv_aug, X_te_c, y_te_c)
        except Exception:
            auc_clean = None

        if auc_leaky is not None and auc_clean is not None:
            key = f"g_lr_rep{rep}"
            results[key] = round(auc_leaky - auc_clean, 6)

    # Aggregate reps
    rep_vals = [v for k, v in results.items() if k.startswith("g_lr_rep")]
    if rep_vals:
        return {
            "g_lr_mean_diff": round(np.mean(rep_vals), 6),
            "g_lr_n_reps": len(rep_vals),
        }
    return None


# ============================================================
# Experiment H: Duplicate Leakage
# ============================================================

def exp_h(X, y, meta, seed=SEED):
    """Inject duplicates from test into train at 0/5/10/30%.
    4 algorithms: LR, RF, DT, KNN. Captures overfit gap per algo."""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    dup_rates = [0.0, 0.05, 0.10, 0.30]
    algo_makers = [
        ("rf", lambda rs: RandomForestClassifier(n_estimators=50, random_state=rs)),
        ("lr", lambda rs: Pipeline([("s", StandardScaler()),
                                     ("m", LogisticRegression(max_iter=1000, random_state=rs))])),
        ("dt", lambda rs: DecisionTreeClassifier(random_state=rs)),
        ("knn", lambda rs: Pipeline([("s", StandardScaler()),
                                      ("m", KNeighborsClassifier(n_neighbors=5))])),
    ]

    aucs = {(a, d): [] for a, _ in algo_makers for d in dup_rates}

    for rep in range(N_REPS):
        rs = seed + rep
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                        random_state=rs, stratify=y)
        except Exception:
            continue

        for d in dup_rates:
            n_dup = max(1, int(len(X_te) * d)) if d > 0 else 0
            if n_dup > 0:
                di = np.random.RandomState(rs + int(d * 100)).choice(
                    len(X_te), size=n_dup, replace=False)
                Xd = np.vstack([X_tr, X_te[di]])
                yd = np.concatenate([y_tr, y_te[di]])
            else:
                Xd, yd = X_tr, y_tr

            for algo_name, maker in algo_makers:
                aucs[(algo_name, d)].append(fit_score(maker(rs), Xd, yd, X_te, y_te))

    results = {}
    for algo_name, _ in algo_makers:
        base_vals = aucs[(algo_name, 0.0)]
        if not base_vals:
            continue
        base = np.mean(base_vals)
        for d in [0.05, 0.10, 0.30]:
            dk = int(d * 100)
            vals = aucs[(algo_name, d)]
            if vals:
                results[f"h_{algo_name}_{dk:02d}"] = round(np.mean(vals) - base, 6)

    # Overfit gap per algorithm (mediator for duplicate theory: Δ = d × overfit)
    try:
        rs = seed
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                    random_state=rs, stratify=y)
        for algo_name, maker in algo_makers:
            try:
                clf = maker(rs).fit(X_tr, y_tr)
                train_auc = safe_auc(y_tr, clf.predict_proba(X_tr)[:, 1])
                test_auc = safe_auc(y_te, clf.predict_proba(X_te)[:, 1])
                results[f"h_{algo_name}_overfit"] = round(train_auc - test_auc, 6)
            except Exception:
                pass
    except Exception:
        pass

    return results if results else None


# ============================================================
# Experiment J: Compound Effect
# ============================================================

def exp_j(X, y, meta, seed=SEED):
    """Stack all applicable leakages. Compare to sum of individual."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score

    n, p = X.shape
    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=seed, stratify=y)

    # CLEAN pipeline: all preprocessing per-fold, no contamination
    try:
        steps = [("s", StandardScaler())]
        if p > 20:
            k_feat = min(10, p // 2)
            steps.insert(0, ("sel", SelectKBest(mutual_info_classif, k=k_feat)))
        steps.append(("m", LogisticRegression(max_iter=1000, random_state=seed)))
        clean_pipe = Pipeline(steps)

        cv_clean = cross_val_score(clean_pipe, X_tv, y_tv, cv=CV_FOLDS, scoring="roc_auc")
        clean_pipe.fit(X_tv, y_tv)
        clean_test = roc_auc_score(y_te, clean_pipe.predict_proba(X_te)[:, 1])
        clean_gap = float(np.mean(cv_clean) - clean_test)
    except Exception:
        return None

    # LEAKY pipeline: global scaling + global feature sel + 10% dups + peeking
    try:
        X_all = np.vstack([X_tv, X_te])
        y_all = np.concatenate([y_tv, y_te])

        # 1. Global scaling
        scaler = StandardScaler().fit(X_all)
        X_tv_s = scaler.transform(X_tv)
        X_te_s = scaler.transform(X_te)

        # 2. Global feature selection (if applicable)
        if p > 20:
            k_feat = min(10, p // 2)
            sel = SelectKBest(mutual_info_classif, k=k_feat).fit(
                scaler.transform(X_all), y_all)
            X_tv_s = sel.transform(X_tv_s)
            X_te_s = sel.transform(X_te_s)

        # 3. Inject 10% duplicates from test into train
        n_dup = max(1, int(len(X_te_s) * 0.10))
        rng = np.random.RandomState(seed)
        di = rng.choice(len(X_te_s), size=n_dup, replace=False)
        X_tv_leak = np.vstack([X_tv_s, X_te_s[di]])
        y_tv_leak = np.concatenate([y_tv, y_te[di]])

        # 4. Peeking: try 5 configs, pick best on test
        cfgs = [
            LogisticRegression(max_iter=1000, C=c, random_state=seed)
            for c in [0.01, 0.1, 1.0, 10.0, 100.0]
        ]
        best_auc = -1
        for clf in cfgs:
            try:
                clf.fit(X_tv_leak, y_tv_leak)
                auc = roc_auc_score(y_te, clf.predict_proba(X_te_s)[:, 1])
                if auc > best_auc:
                    best_auc = auc
            except Exception:
                pass

        if best_auc < 0:
            return None

        # The compound delta: leaky best AUC vs clean single-config test AUC
        compound_delta = best_auc - clean_test
        leaky_gap = best_auc - float(np.mean(cv_clean))  # meaningless but stored

    except Exception:
        return None

    return {
        "j_clean_test": round(clean_test, 6),
        "j_leaky_best": round(best_auc, 6),
        "j_compound_delta": round(compound_delta, 6),
        "j_clean_gap": round(clean_gap, 6),
    }


# ============================================================
# Experiment K: Tuning Baseline
# ============================================================

def exp_k(X, y, meta, seed=SEED):
    """Default vs tuned hyperparameters. How much does tuning gain?
    Anchors leakage magnitudes: if tuning_gain >> leakage, leakage is noise."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=seed, stratify=y)
    results = {}

    # (name, default_maker, tuned_maker)
    configs = [
        ("lr",
         lambda: Pipeline([("s", StandardScaler()),
                           ("m", LogisticRegression(max_iter=1000, random_state=seed))]),
         lambda: Pipeline([("s", StandardScaler()),
                           ("m", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))])),
        ("rf",
         lambda: RandomForestClassifier(n_estimators=50, random_state=seed),
         lambda: RandomForestClassifier(n_estimators=100, max_depth=10,
                                         min_samples_leaf=5, random_state=seed)),
        ("dt",
         lambda: DecisionTreeClassifier(random_state=seed),
         lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=10,
                                         random_state=seed)),
        ("knn",
         lambda: Pipeline([("s", StandardScaler()),
                           ("m", KNeighborsClassifier(n_neighbors=5))]),
         lambda: Pipeline([("s", StandardScaler()),
                           ("m", KNeighborsClassifier(n_neighbors=10, weights="distance"))])),
    ]

    for algo_name, make_default, make_tuned in configs:
        try:
            clf_d = make_default()
            cv_d = cross_val_score(clf_d, X_tv, y_tv, cv=CV_FOLDS, scoring="roc_auc")
            clf_d = make_default().fit(X_tv, y_tv)
            test_d = safe_auc(y_te, clf_d.predict_proba(X_te)[:, 1])

            clf_t = make_tuned()
            cv_t = cross_val_score(clf_t, X_tv, y_tv, cv=CV_FOLDS, scoring="roc_auc")
            clf_t = make_tuned().fit(X_tv, y_tv)
            test_t = safe_auc(y_te, clf_t.predict_proba(X_te)[:, 1])

            results[f"k_{algo_name}_default"] = round(test_d, 6)
            results[f"k_{algo_name}_tuned"] = round(test_t, 6)
            results[f"k_{algo_name}_gain"] = round(test_t - test_d, 6)
        except Exception:
            pass

    return results if results else None


# ============================================================
# Experiment L: Model Variance (Mediating Factor)
# ============================================================

def exp_l(X, y, meta, seed=SEED):
    """Model variance per algorithm via repeated CV.
    Tests whether model variance mediates leakage susceptibility."""
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=2, random_state=seed)
    results = {}

    algos = [
        ("lr", lambda: Pipeline([("s", StandardScaler()),
                                  ("m", LogisticRegression(max_iter=1000, random_state=seed))])),
        ("rf", lambda: RandomForestClassifier(n_estimators=50, random_state=seed)),
        ("dt", lambda: DecisionTreeClassifier(random_state=seed)),
        ("knn", lambda: Pipeline([("s", StandardScaler()),
                                   ("m", KNeighborsClassifier(n_neighbors=5))])),
    ]

    for algo_name, make_model in algos:
        try:
            scores = cross_val_score(make_model(), X, y, cv=cv, scoring="roc_auc")
            results[f"l_{algo_name}_mean"] = round(np.mean(scores), 6)
            results[f"l_{algo_name}_std"] = round(np.std(scores), 6)
            results[f"l_{algo_name}_iqr"] = round(
                float(np.percentile(scores, 75) - np.percentile(scores, 25)), 6)
        except Exception:
            pass

        # Overfit gap (bias proxy)
        try:
            X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                        random_state=seed, stratify=y)
            clf = make_model().fit(X_tv, y_tv)
            train_auc = safe_auc(y_tv, clf.predict_proba(X_tv)[:, 1])
            test_auc = safe_auc(y_te, clf.predict_proba(X_te)[:, 1])
            results[f"l_{algo_name}_overfit"] = round(train_auc - test_auc, 6)
        except Exception:
            pass

    return results if results else None


# ============================================================
# Experiment P: Grouped Split Leakage
# ============================================================

def exp_p(X, y, meta, seed=SEED):
    """Grouped split leakage: KFold vs GroupKFold using k-means clusters.
    Groups are fitted on X_train ONLY (avoids test-set contamination).
    Shuffled-group control verifies near-zero effect under random grouping.
    Natural group detection (integer column with 10 < cardinality < n/5) adds
    a primary arm where real group structure is present."""
    if meta["n_rows"] < 200:
        return None

    from sklearn.model_selection import cross_val_score, GroupKFold, KFold, train_test_split
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    n = meta["n_rows"]
    n_groups = min(20, max(CV_FOLDS + 1, int(np.sqrt(n))))

    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=seed, stratify=y)

    # Fit KMeans on training data ONLY (no test contamination — A1 fix)
    scaler = StandardScaler().fit(X_tv)
    X_tv_s = scaler.transform(X_tv)
    try:
        km = KMeans(n_clusters=n_groups, n_init=3, random_state=seed)
        km.fit(X_tv_s)
        groups = km.labels_
        # Ensure each group has >= CV_FOLDS samples
        unique, counts = np.unique(groups, return_counts=True)
        while np.any(counts < CV_FOLDS) and n_groups > CV_FOLDS + 1:
            n_groups = max(CV_FOLDS + 1, n_groups - 3)
            km = KMeans(n_clusters=n_groups, n_init=3, random_state=seed)
            km.fit(X_tv_s)
            groups = km.labels_
            unique, counts = np.unique(groups, return_counts=True)
        if np.any(counts < CV_FOLDS):
            return None
    except Exception:
        return None

    # Shuffled groups (negative control: should produce ~0 gap vs KFold)
    rng = np.random.RandomState(seed + 1)
    groups_shuffled = rng.permutation(groups)

    # Natural group detection: integer column with 10 < cardinality < n/5
    natural_group_found = False
    natural_col = None
    for j in range(X_tv.shape[1]):
        uniq = np.unique(X_tv[:, j])
        if 10 < len(uniq) < n / 5 and np.allclose(uniq, uniq.astype(int)):
            # Verify enough groups for GroupKFold
            if len(np.unique(X_tv[:, j].astype(int))) >= CV_FOLDS:
                natural_group_found = True
                natural_col = j
                break

    results = {
        "p_n_groups": int(n_groups),
        "p_natural_group_found": natural_group_found,
    }

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    gkf = GroupKFold(n_splits=CV_FOLDS)

    algos = [
        ("lr", Pipeline([("s", StandardScaler()),
                         ("m", LogisticRegression(max_iter=1000, random_state=seed))])),
        ("rf", RandomForestClassifier(n_estimators=50, random_state=seed)),
    ]

    for algo_name, clf in algos:
        try:
            kf_scores = cross_val_score(clf, X_tv, y_tv, cv=kf, scoring="roc_auc")
            gkf_scores = cross_val_score(clf, X_tv, y_tv, cv=gkf, groups=groups,
                                          scoring="roc_auc")
            shf_scores = cross_val_score(clf, X_tv, y_tv, cv=gkf, groups=groups_shuffled,
                                          scoring="roc_auc")
            results[f"p_gap_{algo_name}"] = round(float(np.mean(kf_scores) -
                                                         np.mean(gkf_scores)), 6)
            results[f"p_shuffled_gap_{algo_name}"] = round(float(np.mean(kf_scores) -
                                                                   np.mean(shf_scores)), 6)
        except Exception:
            pass

        # Natural group arm (if available)
        if natural_group_found and natural_col is not None:
            try:
                nat_groups = X_tv[:, natural_col].astype(int)
                kf_scores_nat = cross_val_score(clf, X_tv, y_tv, cv=kf, scoring="roc_auc")
                nat_scores = cross_val_score(clf, X_tv, y_tv, cv=gkf, groups=nat_groups,
                                              scoring="roc_auc")
                results[f"p_natural_gap_{algo_name}"] = round(
                    float(np.mean(kf_scores_nat) - np.mean(nat_scores)), 6)
            except Exception:
                pass

    return results if len(results) > 2 else None


# ============================================================
# Experiment Q: High-Cardinality Vocabulary Leakage
# ============================================================

def exp_q(X, y, meta, seed=SEED):
    """High-cardinality encoding leakage: global vs per-fold CountVectorizer.
    Rows are converted to pseudo-documents from high-cardinality categorical columns
    (cardinality > 20). Global vocabulary includes test-set tokens; per-fold vocabulary
    is fitted only on each fold's training data.
    Note: this is 'high-cardinality encoding leakage', not TF-IDF text leakage."""
    if meta["n_cat"] == 0:
        return None

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, StratifiedKFold

    # Find high-cardinality columns (cardinality > 20)
    hcard_cols = []
    for j in range(X.shape[1]):
        uniq = np.unique(X[:, j][~np.isnan(X[:, j])])
        if len(uniq) > 20:
            hcard_cols.append(j)
    if not hcard_cols:
        return None

    def to_docs(X_sub):
        docs = []
        for row in X_sub:
            tokens = [f"c{j}_{int(row[j])}" for j in hcard_cols if not np.isnan(row[j])]
            docs.append(" ".join(tokens) if tokens else "__empty__")
        return docs

    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=seed, stratify=y)
    docs_tv = to_docs(X_tv)
    docs_te = to_docs(X_te)
    docs_all = docs_tv + docs_te

    # Global vocabulary (leaky): fit CountVectorizer on train+test
    try:
        cv_global = CountVectorizer(max_features=500).fit(docs_all)
        vocab_size = len(cv_global.vocabulary_)
        if vocab_size < 3:
            return None
        X_tv_g = cv_global.transform(docs_tv).toarray()
        X_te_g = cv_global.transform(docs_te).toarray()

        # CV on globally-vectorized data
        from sklearn.model_selection import cross_val_score
        lr_g = LogisticRegression(max_iter=1000, random_state=seed)
        cv_g_scores = cross_val_score(lr_g, X_tv_g, y_tv, cv=CV_FOLDS, scoring="roc_auc")
        lr_g2 = LogisticRegression(max_iter=1000, random_state=seed).fit(X_tv_g, y_tv)
        test_g = safe_auc(y_te, lr_g2.predict_proba(X_te_g)[:, 1])
        gap_g = float(np.mean(cv_g_scores) - test_g)
    except Exception:
        return None

    # Per-fold vocabulary (clean): manual StratifiedKFold
    try:
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        fold_cv_aucs = []
        for train_idx, val_idx in skf.split(range(len(docs_tv)), y_tv):
            d_tr = [docs_tv[i] for i in train_idx]
            d_val = [docs_tv[i] for i in val_idx]
            y_tr_f, y_val_f = y_tv[train_idx], y_tv[val_idx]
            cv_pf_fold = CountVectorizer(max_features=500).fit(d_tr)
            X_tr_f = cv_pf_fold.transform(d_tr).toarray()
            X_val_f = cv_pf_fold.transform(d_val).toarray()
            lr_f = LogisticRegression(max_iter=1000, random_state=seed)
            try:
                lr_f.fit(X_tr_f, y_tr_f)
                fold_cv_aucs.append(safe_auc(y_val_f, lr_f.predict_proba(X_val_f)[:, 1]))
            except Exception:
                pass
        if not fold_cv_aucs:
            return None
        # Test AUC using vocabulary fitted on X_tv only
        cv_pf_final = CountVectorizer(max_features=500).fit(docs_tv)
        X_tv_pf = cv_pf_final.transform(docs_tv).toarray()
        X_te_pf = cv_pf_final.transform(docs_te).toarray()
        lr_pf = LogisticRegression(max_iter=1000, random_state=seed).fit(X_tv_pf, y_tv)
        test_pf = safe_auc(y_te, lr_pf.predict_proba(X_te_pf)[:, 1])
        gap_pf = float(np.mean(fold_cv_aucs) - test_pf)
    except Exception:
        return None

    return {
        "q_gap": round(gap_g - gap_pf, 6),
        "q_vocab_size": int(vocab_size),
        "q_n_vocab_cols": len(hcard_cols),
    }


# ============================================================
# Experiment R: Target Proxy Calibration
# ============================================================

def exp_r(X, y, meta, seed=SEED):
    """Target proxy injection — CALIBRATION/SENSITIVITY ANALYSIS.
    NOT a leakage discovery: injects sigmoid(y*strength+noise) as a feature to
    measure AUC inflation across controlled dose levels. strength=0 is the null
    control (should show ~0 delta). Covers Kapoor L2.1 (target proxy features).
    y must be binary (runs post-prepare_binary)."""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=seed, stratify=y)
    rng = np.random.RandomState(seed)
    n_total = len(y_tv) + len(y_te)
    y_all = np.concatenate([y_tv, y_te])

    results = {}

    # Baseline: original features only
    for algo_name, make_model in [
        ("lr", lambda: Pipeline([("s", StandardScaler()),
                                  ("m", LogisticRegression(max_iter=1000,
                                                            random_state=seed))])),
        ("rf", lambda: RandomForestClassifier(n_estimators=50, random_state=seed)),
    ]:
        try:
            m = make_model().fit(X_tv, y_tv)
            results[f"r_{algo_name}_baseline"] = round(
                safe_auc(y_te, m.predict_proba(X_te)[:, 1]), 6)
        except Exception:
            pass

    # Inject proxy at each strength level
    strengths = [(0.0, "null"), (0.5, "weak"), (2.0, "med"), (5.0, "strong")]
    noise = rng.randn(n_total) * 0.5  # shared noise across strengths for comparability

    for strength, label in strengths:
        proxy_all = 1.0 / (1.0 + np.exp(-(y_all.astype(float) * strength + noise)))
        proxy_tv = proxy_all[:len(y_tv)].reshape(-1, 1)
        proxy_te = proxy_all[len(y_tv):].reshape(-1, 1)

        X_tv_inj = np.hstack([X_tv, proxy_tv])
        X_te_inj = np.hstack([X_te, proxy_te])

        for algo_name, make_model in [
            ("lr", lambda: Pipeline([("s", StandardScaler()),
                                      ("m", LogisticRegression(max_iter=1000,
                                                                random_state=seed))])),
            ("rf", lambda: RandomForestClassifier(n_estimators=50, random_state=seed)),
        ]:
            baseline_key = f"r_{algo_name}_baseline"
            if baseline_key not in results:
                continue
            try:
                m = make_model().fit(X_tv_inj, y_tv)
                auc_inj = safe_auc(y_te, m.predict_proba(X_te_inj)[:, 1])
                results[f"r_{algo_name}_delta_{label}"] = round(
                    auc_inj - results[baseline_key], 6)
            except Exception:
                pass

    return results if results else None


# ============================================================
# Experiment T: Binning / Discretization Leakage
# ============================================================

def exp_t(X, y, meta, seed=SEED):
    """Binning leakage: global KBinsDiscretizer(quantile) vs per-fold.
    Hypothesis H_T: binning leakage has a different scaling law from normalization
    because quantile estimation error is nonlinear (binomial variance at quantile
    boundaries), not Gaussian (mean/std estimation). This is tested by comparing
    the log-log slope of binning leakage vs n with the Class I slope (-1)."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.base import clone

    # Guard: skip if data is near-categorical (few unique values per feature)
    mean_unique = float(np.mean([len(np.unique(X[:, j])) for j in range(X.shape[1])]))
    if mean_unique < 5:
        return None

    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=seed, stratify=y)
    results = {"t_mean_unique_values": round(mean_unique, 2)}

    algos = [
        ("lr", Pipeline([("disc", KBinsDiscretizer(n_bins=10, strategy="quantile",
                                                    encode="ordinal",
                                                    subsample=None)),
                          ("s", StandardScaler()),
                          ("m", LogisticRegression(max_iter=1000, random_state=seed))])),
        ("rf", Pipeline([("disc", KBinsDiscretizer(n_bins=10, strategy="quantile",
                                                    encode="ordinal",
                                                    subsample=None)),
                          ("m", RandomForestClassifier(n_estimators=50,
                                                       random_state=seed))])),
    ]

    for algo_name, pipe_template in algos:
        # Global discretization (leaky): fit KBinsDiscretizer on train+test
        try:
            X_all = np.vstack([X_tv, X_te])
            disc_g = KBinsDiscretizer(n_bins=10, strategy="quantile", encode="ordinal",
                                      subsample=None)
            disc_g.fit(X_all)
            X_tv_d = disc_g.transform(X_tv)
            X_te_d = disc_g.transform(X_te)

            # Remove constant features after binning
            variances = np.var(X_tv_d, axis=0)
            nonconst = variances > 0
            if nonconst.sum() == 0:
                continue
            X_tv_d = X_tv_d[:, nonconst]
            X_te_d = X_te_d[:, nonconst]

            # Downstream model (same as per-fold but without disc step)
            if algo_name == "lr":
                sub_pipe = Pipeline([
                    ("s", StandardScaler()),
                    ("m", LogisticRegression(max_iter=1000, random_state=seed))
                ])
            else:
                sub_pipe = RandomForestClassifier(n_estimators=50, random_state=seed)

            cv_g = cross_val_score(clone(sub_pipe), X_tv_d, y_tv, cv=CV_FOLDS,
                                   scoring="roc_auc")
            m_g = clone(sub_pipe).fit(X_tv_d, y_tv)
            test_g = safe_auc(y_te, m_g.predict_proba(X_te_d)[:, 1])
            gap_g = float(np.mean(cv_g) - test_g)
        except Exception:
            gap_g = None

        # Per-fold discretization (clean): KBinsDiscretizer inside Pipeline
        try:
            cv_pf = cross_val_score(clone(pipe_template), X_tv, y_tv, cv=CV_FOLDS,
                                    scoring="roc_auc")
            pipe_final = clone(pipe_template).fit(X_tv, y_tv)
            test_pf = safe_auc(y_te, pipe_final.predict_proba(X_te)[:, 1])
            gap_pf = float(np.mean(cv_pf) - test_pf)
        except Exception:
            gap_pf = None

        if gap_g is not None and gap_pf is not None:
            results[f"t_{algo_name}_gap_diff"] = round(gap_g - gap_pf, 6)

    return results if len(results) > 1 else None


# ============================================================
# Experiment V: Nested vs Flat CV (Cawley & Talbot 2010)
# ============================================================

def exp_v(X, y, meta, seed=SEED):
    """Nested vs flat CV: the Cawley & Talbot (2010) HPO inflation test.
    Flat (leaky): GridSearchCV.best_score_ — this is what practitioners report.
    Nested (clean): 5-fold outer × 3-fold inner, evaluate on held-out outer fold.
    Gap = flat_best_score - nested_test_auc = HPO inflation."""
    if meta["n_rows"] < 200:
        return None

    from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                          GridSearchCV)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=seed, stratify=y)
    param_grid = {"m__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
    base_pipe = Pipeline([("s", StandardScaler()),
                          ("m", LogisticRegression(max_iter=1000, random_state=seed))])

    # FLAT (leaky): GridSearchCV with 5-fold, report best_score_
    try:
        gscv = GridSearchCV(base_pipe, param_grid, cv=CV_FOLDS, scoring="roc_auc",
                            refit=True, n_jobs=1)
        gscv.fit(X_tv, y_tv)
        v_flat_auc = float(gscv.best_score_)
    except Exception:
        return None

    # NESTED (clean): 5-fold outer × 3-fold inner for C selection
    try:
        outer_cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        outer_test_aucs = []
        for train_idx, test_idx in outer_cv.split(X_tv, y_tv):
            X_outer_tr = X_tv[train_idx]
            X_outer_te = X_tv[test_idx]
            y_outer_tr = y_tv[train_idx]
            y_outer_te = y_tv[test_idx]
            inner_pipe = Pipeline([("s", StandardScaler()),
                                   ("m", LogisticRegression(max_iter=1000,
                                                             random_state=seed))])
            inner_gscv = GridSearchCV(inner_pipe, param_grid, cv=3,
                                      scoring="roc_auc", refit=True, n_jobs=1)
            inner_gscv.fit(X_outer_tr, y_outer_tr)
            auc = safe_auc(y_outer_te,
                           inner_gscv.predict_proba(X_outer_te)[:, 1])
            outer_test_aucs.append(auc)
        v_nested_auc = float(np.mean(outer_test_aucs))
    except Exception:
        return None

    return {
        "v_flat_auc": round(v_flat_auc, 6),
        "v_nested_auc": round(v_nested_auc, 6),
        "v_inflation": round(v_flat_auc - v_nested_auc, 6),
    }


# ============================================================
# Experiment W: Row-Order Detection
# ============================================================

def exp_w(X, y, meta, seed=SEED):
    """Row-order detection: LR trained on row index only.
    AUC > n-adaptive threshold flags datasets likely sorted by outcome or time.
    Threshold = 0.50 + 1.96*sqrt(0.25/n_test), giving ~2.5% false positive rate
    across all dataset sizes (not a fixed 0.60).
    Output is a dataset diagnostic, NOT a leakage magnitude — use w_ordered as a
    covariate in meta-regression to test whether ordering moderates other leakages."""
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    n = meta["n_rows"]
    n_test = max(1, int(n * 0.2))
    threshold = 0.50 + 1.96 * np.sqrt(0.25 / n_test)

    # Row index as the only feature
    X_idx = np.arange(len(y)).reshape(-1, 1).astype(float)
    try:
        pipe = Pipeline([("s", StandardScaler()),
                         ("m", LogisticRegression(max_iter=1000, random_state=seed))])
        scores = cross_val_score(pipe, X_idx, y, cv=CV_FOLDS, scoring="roc_auc")
        w_auc = float(np.mean(scores))
    except Exception:
        return None

    return {
        "w_rowindex_auc": round(w_auc, 6),
        "w_threshold": round(threshold, 6),
        "w_ordered": bool(w_auc > threshold),
    }


# ============================================================
# Main Loop
# ============================================================

def main():
    print("=" * 70)
    print("THE LEAKAGE LANDSCAPE — V3 Experiment Suite")
    print("=" * 70)

    # Simulation calibration
    if RUN_CALIBRATION:
        try:
            run_calibration()
        except Exception as e:
            print(f"Calibration failed: {e}")

    # Collect all datasets
    print("\n[1/4] Collecting ml datasets...")
    ml_ds = collect_ml_datasets()
    print(f"  {len(ml_ds)} ml datasets")

    print("[2/4] Collecting PMLB datasets...")
    pmlb_ds = collect_pmlb_datasets()
    print(f"  {len(pmlb_ds)} PMLB datasets")

    print("[3/4] Collecting OpenML datasets...")
    openml_ds = collect_openml_datasets()
    print(f"  {len(openml_ds)} OpenML datasets")

    # Deduplicate: ml > PMLB > OpenML priority
    all_ds = ml_ds + pmlb_ds + openml_ds
    print(f"\nTotal before dedup: {len(all_ds)}")

    seen_names = set()
    unique_ds = []
    for ds in all_ds:
        base = ds["name"].lower().replace("_", "").replace("-", "")
        if base not in seen_names:
            seen_names.add(base)
            unique_ds.append(ds)
    print(f"After dedup: {len(unique_ds)}")

    # Assign discovery/confirmation split
    disc_count = sum(1 for ds in unique_ds if assign_split(ds["name"], ds["source"]) == "discovery")
    print(f"Discovery: {disc_count}, Confirmation: {len(unique_ds) - disc_count}")

    # Check what's already done
    completed = get_completed()
    remaining = [ds for ds in unique_ds if (ds["name"] + "|" + ds["source"]) not in completed]
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    # Run experiments
    total = len(remaining)
    times = []
    for i, ds in enumerate(remaining):
        name = ds["name"]
        source = ds["source"]
        t0 = time.time()

        print(f"\n[{i+1}/{total}] {name} ({source})", end="", flush=True)

        result = {
            "name": name,
            "source": source,
            "split": assign_split(name, source),
            "split_pw": assign_split_pw(name, source),
            "completed_with_new_exps": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            # Load
            load_out = ds["loader"]()
            if load_out is None or load_out[0] is None:
                print(" SKIP(load)")
                result["status"] = "skip_load"
                save_result(result)
                continue

            X_raw, y_raw, n_cat, nan_pct = load_out

            # Prepare binary
            X, y = prepare_binary(X_raw, y_raw)
            if X is None or len(y) < MIN_ROWS:
                print(f" SKIP(small:{len(y) if y is not None else 0})")
                result["status"] = "skip_small"
                save_result(result)
                continue

            if len(np.unique(y)) < 2:
                print(" SKIP(1class)")
                result["status"] = "skip_single_class"
                save_result(result)
                continue

            n, p = X.shape
            if p > MAX_FEATURES:
                print(f" SKIP(p={p})")
                result["status"] = "skip_high_dim"
                save_result(result)
                continue

            # Clean data
            X = np.nan_to_num(X.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

            # Metadata
            imbalance = min(np.mean(y == 0), np.mean(y == 1))
            meta = {
                "n_rows": int(n),
                "n_features": int(p),
                "p_over_n": p / n,
                "n_cat": int(n_cat),
                "imbalance": float(imbalance),
                "nan_pct": float(nan_pct),
            }
            result.update({
                "n_rows": int(n),
                "n_features": int(p),
                "p_over_n": round(p / n, 6),
                "n_cat": int(n_cat),
                "imbalance": round(float(imbalance), 4),
                "nan_pct": round(float(nan_pct), 4),
                "status": "ok",
            })

            # Theoretical bounds
            theory = compute_theory(meta)
            result.update(theory)

            # Subsample
            Xs, ys = safe_subsample(X, y)
            meta_s = {**meta, "n_rows": len(ys)}

            print(f" N={n} p={p} p/n={p/n:.3f}", end="", flush=True)

            # Experiment A: StandardScaler normalization
            if RUN_A:
                try:
                    r = exp_a(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        d = r.get("a_lr_gap_diff", 0)
                        print(f" A:{d:+.3f}", end="")
                except Exception:
                    print(" A:ERR", end="")

            # Experiment A2: MinMaxScaler normalization
            if RUN_A2:
                try:
                    r = exp_a2(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        d = r.get("a2_knn_gap_diff", r.get("a2_lr_gap_diff", 0))
                        print(f" A2:{d:+.3f}", end="")
                except Exception:
                    print(" A2:ERR", end="")

            # Experiment A3: Normalization under outliers
            if RUN_A3:
                try:
                    r = exp_a3(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        d = r.get("a3_mm_knn_diff", r.get("a3_mm_lr_diff", 0))
                        print(f" A3:{d:+.3f}", end="")
                except Exception:
                    print(" A3:ERR", end="")

            # Experiment B
            if RUN_B:
                try:
                    r = exp_b(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        print(f" B:{r.get('b_infl_k10', 0):+.3f}", end="")
                except Exception:
                    print(" B:ERR", end="")

            # Experiment C
            if RUN_C:
                try:
                    r = exp_c(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        print(f" C:{r.get('c_lr_gap_diff', 0):+.3f}", end="")
                except Exception:
                    print(" C:ERR", end="")

            # Experiment D
            if RUN_D:
                try:
                    r = exp_d(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        print(f" D:{r.get('d_gap_diff', 0):+.3f}", end="")
                except Exception:
                    print(" D:ERR", end="")

            # Experiment E
            if RUN_E:
                try:
                    r = exp_e(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        print(f" E:{r.get('e_lr_gap_diff_10', 0):+.3f}", end="")
                except Exception:
                    print(" E:ERR", end="")

            # Experiment F
            if RUN_F:
                try:
                    r = exp_f(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        print(f" F:{r.get('f_lr_gap_diff', 0):+.3f}", end="")
                except Exception:
                    print(" F:ERR", end="")

            # Experiment G
            if RUN_G:
                try:
                    r = exp_g(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        print(f" G:{r.get('g_lr_mean_diff', 0):+.3f}", end="")
                except Exception:
                    print(" G:ERR", end="")

            # Experiment H
            if RUN_H:
                try:
                    r = exp_h(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        print(f" H:{r.get('h_rf_10', 0):+.3f}", end="")
                except Exception:
                    print(" H:ERR", end="")

            # Experiment J
            if RUN_J:
                try:
                    r = exp_j(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        print(f" J:{r.get('j_compound_delta', 0):+.3f}", end="")
                except Exception:
                    print(" J:ERR", end="")

            # Experiment K: Tuning baseline
            if RUN_K:
                try:
                    r = exp_k(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        gains = [v for k, v in r.items() if k.endswith("_gain")]
                        if gains:
                            print(f" K:{max(gains):+.3f}", end="")
                except Exception:
                    print(" K:ERR", end="")

            # Experiment L: Model variance
            if RUN_L:
                try:
                    r = exp_l(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        stds = [v for k, v in r.items() if k.endswith("_std")]
                        if stds:
                            print(f" L:{max(stds):.3f}", end="")
                except Exception:
                    print(" L:ERR", end="")

            # Experiment P: Grouped split leakage
            if RUN_P:
                try:
                    r = exp_p(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        print(f" P:{r.get('p_gap_lr', 0):+.3f}", end="")
                except Exception:
                    print(" P:ERR", end="")

            # Experiment Q: Vocabulary leakage
            if RUN_Q:
                try:
                    r = exp_q(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        print(f" Q:{r.get('q_gap', 0):+.3f}", end="")
                except Exception:
                    print(" Q:ERR", end="")

            # Experiment R: Target proxy calibration
            if RUN_R:
                try:
                    r = exp_r(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        print(f" R:{r.get('r_lr_delta_strong', 0):+.3f}", end="")
                except Exception:
                    print(" R:ERR", end="")

            # Experiment T: Binning leakage
            if RUN_T:
                try:
                    r = exp_t(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        print(f" T:{r.get('t_lr_gap_diff', 0):+.3f}", end="")
                except Exception:
                    print(" T:ERR", end="")

            # Experiment V: Nested vs flat CV
            if RUN_V:
                try:
                    r = exp_v(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        print(f" V:{r.get('v_inflation', 0):+.3f}", end="")
                except Exception:
                    print(" V:ERR", end="")

            # Experiment W: Row-order detection
            if RUN_W:
                try:
                    r = exp_w(Xs, ys, meta_s)
                    if r:
                        result.update(r)
                        flag = "*" if r.get("w_ordered") else ""
                        print(f" W:{r.get('w_rowindex_auc', 0):.3f}{flag}", end="")
                except Exception:
                    print(" W:ERR", end="")

            elapsed = time.time() - t0
            result["elapsed_s"] = round(elapsed, 1)
            times.append(elapsed)
            avg = np.mean(times[-50:])
            eta_h = avg * (total - i - 1) / 3600
            print(f" [{elapsed:.0f}s] ETA:{eta_h:.1f}h")

        except Exception as e:
            result["status"] = f"error: {str(e)[:100]}"
            print(f" DATASET_ERR: {str(e)[:50]}")

        save_result(result)

    # === Backfill A2/A3 for datasets completed before these experiments existed ===
    backfill_keys = get_needs_backfill()
    if backfill_keys:
        print(f"\n\n{'='*70}")
        print(f"BACKFILLING A2/A3 for {len(backfill_keys)} previously completed datasets")
        print(f"{'='*70}")

        # Rebuild loader map
        loader_map = {ds["name"] + "|" + ds["source"]: ds for ds in unique_ds}
        bf_list = [loader_map[k] for k in backfill_keys if k in loader_map]
        bf_times = []

        for i, ds in enumerate(bf_list):
            name = ds["name"]
            source = ds["source"]
            t0 = time.time()
            print(f"\n[BF {i+1}/{len(bf_list)}] {name} ({source})", end="", flush=True)

            bf_result = {"name": name, "source": source}

            try:
                load_out = ds["loader"]()
                if load_out is None or load_out[0] is None:
                    print(" SKIP(load)")
                    continue

                X_raw, y_raw, n_cat, nan_pct = load_out
                X, y = prepare_binary(X_raw, y_raw)
                if X is None or len(y) < MIN_ROWS:
                    print(" SKIP(small)")
                    continue

                n, p = X.shape
                Xs, ys = safe_subsample(X, y)

                imbalance = min(np.bincount(ys)) / len(ys)
                meta = {"n_rows": len(ys), "n_features": p, "p_over_n": p / len(ys),
                        "n_cat": n_cat, "imbalance": float(imbalance), "nan_pct": float(nan_pct)}

                # A2: MinMaxScaler
                if RUN_A2:
                    try:
                        r = exp_a2(Xs, ys, meta)
                        if r:
                            bf_result.update(r)
                            d = r.get("a2_knn_gap_diff", r.get("a2_lr_gap_diff", 0))
                            print(f" A2:{d:+.3f}", end="")
                    except Exception:
                        print(" A2:ERR", end="")

                # A3: Outlier test
                if RUN_A3:
                    try:
                        r = exp_a3(Xs, ys, meta)
                        if r:
                            bf_result.update(r)
                            d = r.get("a3_mm_knn_diff", r.get("a3_mm_lr_diff", 0))
                            print(f" A3:{d:+.3f}", end="")
                    except Exception:
                        print(" A3:ERR", end="")

                elapsed = time.time() - t0
                bf_times.append(elapsed)
                avg = np.mean(bf_times[-50:])
                eta_h = avg * (len(bf_list) - i - 1) / 3600
                print(f" [{elapsed:.0f}s] ETA:{eta_h:.1f}h")

            except Exception as e:
                print(f" BF_ERR: {str(e)[:50]}")

            save_backfill(bf_result)

    # === Backfill P-W for datasets completed before these experiments ===
    backfill_new_keys = get_needs_backfill_new()
    if backfill_new_keys:
        print(f"\n\n{'='*70}")
        print(f"BACKFILLING P-W for {len(backfill_new_keys)} previously completed datasets")
        print(f"{'='*70}")

        # Build n_rows lookup for size-ascending sort (small first → faster P-W stats)
        _nrows_map = {}
        with open(RESULTS_FILE) as _f:
            for _line in _f:
                try:
                    _obj = json.loads(_line.strip())
                    _k = _obj.get("name", "") + "|" + _obj.get("source", "")
                    if _k in backfill_new_keys:
                        _nrows_map[_k] = _obj.get("n_rows", 999_999)
                except Exception:
                    pass
        loader_map = {ds["name"] + "|" + ds["source"]: ds for ds in unique_ds}
        bf_new_list = sorted(
            [loader_map[k] for k in backfill_new_keys if k in loader_map],
            key=lambda ds: _nrows_map.get(ds["name"] + "|" + ds["source"], 999_999),
        )
        bf_new_times = []

        for i, ds in enumerate(bf_new_list):
            name = ds["name"]
            source = ds["source"]
            t0 = time.time()
            print(f"\n[BF-NEW {i+1}/{len(bf_new_list)}] {name} ({source})",
                  end="", flush=True)

            bf_result = {
                "name": name,
                "source": source,
                "split_pw": assign_split_pw(name, source),
                "completed_with_new_exps": False,
            }

            try:
                load_out = ds["loader"]()
                if load_out is None or load_out[0] is None:
                    print(" SKIP(load)")
                    continue

                X_raw, y_raw, n_cat, nan_pct = load_out
                X, y = prepare_binary(X_raw, y_raw)
                if X is None or len(y) < MIN_ROWS:
                    print(" SKIP(small)")
                    continue

                n, p = X.shape
                X = np.nan_to_num(X.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
                Xs, ys = safe_subsample(X, y)

                imbalance = min(np.bincount(ys)) / len(ys)
                meta = {"n_rows": len(ys), "n_features": p, "p_over_n": p / len(ys),
                        "n_cat": n_cat, "imbalance": float(imbalance),
                        "nan_pct": float(nan_pct)}

                if RUN_P:
                    try:
                        r = exp_p(Xs, ys, meta)
                        if r:
                            bf_result.update(r)
                            print(f" P:{r.get('p_gap_lr', 0):+.3f}", end="")
                    except Exception:
                        print(" P:ERR", end="")

                if RUN_Q:
                    try:
                        r = exp_q(Xs, ys, meta)
                        if r:
                            bf_result.update(r)
                            print(f" Q:{r.get('q_gap', 0):+.3f}", end="")
                    except Exception:
                        print(" Q:ERR", end="")

                if RUN_R:
                    try:
                        r = exp_r(Xs, ys, meta)
                        if r:
                            bf_result.update(r)
                            print(f" R:{r.get('r_lr_delta_strong', 0):+.3f}", end="")
                    except Exception:
                        print(" R:ERR", end="")

                if RUN_T:
                    try:
                        r = exp_t(Xs, ys, meta)
                        if r:
                            bf_result.update(r)
                            print(f" T:{r.get('t_lr_gap_diff', 0):+.3f}", end="")
                    except Exception:
                        print(" T:ERR", end="")

                if RUN_V:
                    try:
                        r = exp_v(Xs, ys, meta)
                        if r:
                            bf_result.update(r)
                            print(f" V:{r.get('v_inflation', 0):+.3f}", end="")
                    except Exception:
                        print(" V:ERR", end="")

                if RUN_W:
                    try:
                        r = exp_w(Xs, ys, meta)
                        if r:
                            bf_result.update(r)
                            flag = "*" if r.get("w_ordered") else ""
                            print(f" W:{r.get('w_rowindex_auc', 0):.3f}{flag}", end="")
                    except Exception:
                        print(" W:ERR", end="")

                elapsed = time.time() - t0
                bf_new_times.append(elapsed)
                avg = np.mean(bf_new_times[-50:])
                eta_h = avg * (len(bf_new_list) - i - 1) / 3600
                print(f" [{elapsed:.0f}s] ETA:{eta_h:.1f}h")

            except Exception as e:
                print(f" BF-NEW_ERR: {str(e)[:50]}")

            save_backfill_new(bf_result)

    # Generate report
    print("\n\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)
    generate_report()


# ============================================================
# Report Generation
# ============================================================

def generate_report():
    """Generate analysis report from JSONL results.
    Merges main results with A2/A3 backfill and P-W backfill files."""
    results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except Exception:
                pass

    # Helper: merge a backfill file into results
    def merge_backfill(bf_path, label):
        if not os.path.exists(bf_path):
            return
        backfill = {}
        with open(bf_path) as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    key = obj.get("name", "") + "|" + obj.get("source", "")
                    backfill[key] = obj
                except Exception:
                    pass
        for r in results:
            key = r.get("name", "") + "|" + r.get("source", "")
            if key in backfill:
                for k, v in backfill[key].items():
                    if k not in ("name", "source") and k not in r:
                        r[k] = v
        print(f"Merged {len(backfill)} {label} backfill records")

    merge_backfill(BACKFILL_FILE, "A2/A3")
    merge_backfill(BACKFILL_NEW_FILE, "P-W")

    df = pd.DataFrame(results)
    ok = df[df["status"] == "ok"].copy()
    print(f"Total: {len(df)}, OK: {len(ok)}, Skip/Error: {len(df) - len(ok)}")

    disc = ok[ok["split"] == "discovery"]
    conf = ok[ok["split"] == "confirmation"]

    lines = []
    lines.append("# The Leakage Landscape — Results")
    lines.append("")
    lines.append(f"*{len(ok)} datasets: ml ({len(ok[ok['source']=='ml'])}), "
                 f"PMLB ({len(ok[ok['source']=='pmlb'])}), "
                 f"OpenML ({len(ok[ok['source']=='openml'])})*")
    lines.append(f"*Discovery: {len(disc)}, Confirmation: {len(conf)}*")
    lines.append("")
    lines.append(f"**N range**: [{int(ok['n_rows'].min())}, {int(ok['n_rows'].max())}]")
    lines.append(f"**p range**: [{int(ok['n_features'].min())}, {int(ok['n_features'].max())}]")
    lines.append(f"**p/n range**: [{ok['p_over_n'].min():.4f}, {ok['p_over_n'].max():.2f}]")
    lines.append("")

    # === Grand Summary ===
    lines.append("## 1. Grand Summary")
    lines.append("")
    lines.append("| Experiment | N | Mean | Median | Std | t | p | d |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    summary_cols = [
        ("A: Std LR gap", "a_lr_gap_diff"),
        ("A: Std RF gap", "a_rf_gap_diff"),
        ("A: Std DT gap", "a_dt_gap_diff"),
        ("A: Std KNN gap", "a_knn_gap_diff"),
        ("A2: MM LR gap", "a2_lr_gap_diff"),
        ("A2: MM RF gap", "a2_rf_gap_diff"),
        ("A2: MM DT gap", "a2_dt_gap_diff"),
        ("A2: MM KNN gap", "a2_knn_gap_diff"),
        ("A3: Out Std LR", "a3_std_lr_diff"),
        ("A3: Out Std KNN", "a3_std_knn_diff"),
        ("A3: Out MM LR", "a3_mm_lr_diff"),
        ("A3: Out MM KNN", "a3_mm_knn_diff"),
        ("B: Peek K=10", "b_infl_k10"),
        ("B: Peek K=19", "b_infl_k19"),
        ("C: FSel LR gap", "c_lr_gap_diff"),
        ("C: FSel RF gap", "c_rf_gap_diff"),
        ("D: Split diff", "d_gap_diff"),
        ("E: Imp LR 10%", "e_lr_gap_diff_10"),
        ("E: Imp LR 30%", "e_lr_gap_diff_30"),
        ("F: TEnc LR gap", "f_lr_gap_diff"),
        ("G: Oversamp LR", "g_lr_mean_diff"),
        ("H: Dup RF 10%", "h_rf_10"),
        ("H: Dup LR 10%", "h_lr_10"),
        ("H: Dup DT 10%", "h_dt_10"),
        ("H: Dup KNN 10%", "h_knn_10"),
        ("J: Compound", "j_compound_delta"),
        ("K: Tune LR gain", "k_lr_gain"),
        ("K: Tune RF gain", "k_rf_gain"),
        ("K: Tune DT gain", "k_dt_gain"),
        ("K: Tune KNN gain", "k_knn_gain"),
        ("P: Group LR gap", "p_gap_lr"),
        ("P: Group RF gap", "p_gap_rf"),
        ("P: Shuffled LR gap", "p_shuffled_gap_lr"),
        ("Q: Vocab gap", "q_gap"),
        ("R: Proxy LR null", "r_lr_delta_null"),
        ("R: Proxy LR weak", "r_lr_delta_weak"),
        ("R: Proxy LR med", "r_lr_delta_med"),
        ("R: Proxy LR strong", "r_lr_delta_strong"),
        ("T: Bin LR gap", "t_lr_gap_diff"),
        ("T: Bin RF gap", "t_rf_gap_diff"),
        ("V: CV inflation", "v_inflation"),
    ]

    for label, col in summary_cols:
        if col not in ok.columns:
            continue
        vals = ok[col].dropna()
        if len(vals) < 3:
            continue
        t_s, p_v = stats.ttest_1samp(vals, 0)
        d = vals.mean() / vals.std() if vals.std() > 0 else 0
        lines.append(f"| {label} | {len(vals)} | {vals.mean():+.4f} | {vals.median():+.4f} | "
                     f"{vals.std():.4f} | {t_s:.2f} | {p_v:.2e} | {d:.3f} |")
    lines.append("")

    # === Scaling Class Verification (CORE FIGURE) ===
    lines.append("## 2. Scaling Class Verification (Core Figure)")
    lines.append("")
    lines.append("Log-log slopes: expected -1 (Class I), -0.5 (Class II), 0 (Class III)")
    lines.append("")

    scaling_tests = [
        ("Class I: Norm Std", "a_lr_gap_diff", -1.0),
        ("Class I: Norm MinMax", "a2_lr_gap_diff", -1.0),
        ("Class I: Norm MM KNN", "a2_knn_gap_diff", -1.0),
        ("Class I: Outlier MM KNN", "a3_mm_knn_diff", -1.0),
        ("Class I: Feature Sel", "c_lr_gap_diff", -1.0),
        ("Class I: Imputation 10%", "e_lr_gap_diff_10", -1.0),
        ("Class II: Peeking K=10", "b_infl_k10", -0.5),
        ("Class II: TEnc", "f_lr_gap_diff", -0.5),
        ("Class III: Dup RF 10%", "h_rf_10", 0.0),
        ("Class III: Oversamp", "g_lr_mean_diff", 0.0),
        ("Class I: Binning LR", "t_lr_gap_diff", -1.0),
        ("Class II: Nested CV", "v_inflation", -0.5),
        ("Class II: Group split LR", "p_gap_lr", -0.5),
    ]

    lines.append("| Type | N datasets | Slope | 95% CI | Expected | Match? |")
    lines.append("|---|---:|---:|---|---:|---|")

    for label, col, expected in scaling_tests:
        if col not in ok.columns:
            continue
        sub = ok[["n_rows", col]].dropna()
        if len(sub) < 20:
            continue

        # Bin by N (4 bins)
        sub["log_n"] = np.log10(sub["n_rows"])
        sub["n_bin"] = pd.qcut(sub["log_n"], 4, duplicates="drop")
        binned = sub.groupby("n_bin", observed=True).agg(
            log_n_mean=("log_n", "mean"),
            effect_median=(col, "median"),
            count=(col, "count"),
        ).reset_index()
        binned = binned[binned["effect_median"].abs() > 1e-8]

        if len(binned) >= 3:
            # Only use positive effects for log-log
            pos = binned[binned["effect_median"] > 0]
            if len(pos) >= 3:
                log_effect = np.log10(pos["effect_median"].values)
                log_n = pos["log_n_mean"].values
                slope, intercept, r_val, p_val, std_err = stats.linregress(log_n, log_effect)
                ci_lo = slope - 1.96 * std_err
                ci_hi = slope + 1.96 * std_err
                match = "YES" if ci_lo <= expected <= ci_hi else "~" if abs(slope - expected) < 0.5 else "NO"
                lines.append(f"| {label} | {len(sub)} | {slope:.2f} | [{ci_lo:.2f}, {ci_hi:.2f}] | "
                             f"{expected:.1f} | {match} |")
            else:
                lines.append(f"| {label} | {len(sub)} | — | insufficient positive effects | "
                             f"{expected:.1f} | — |")
        else:
            lines.append(f"| {label} | {len(sub)} | — | insufficient bins | "
                         f"{expected:.1f} | — |")
    lines.append("")

    # === Theory vs Empirical ===
    lines.append("## 3. Theory vs Empirical")
    lines.append("")

    theory_pairs = [
        ("A Normalization", "a_lr_gap_diff", "a_theory"),
        ("C Feature Sel", "c_lr_gap_diff", "c_theory"),
        ("E Imputation 10%", "e_lr_gap_diff_10", "e_theory_10"),
    ]

    lines.append("| Type | N | r(theory,empirical) | MAE | Theory explains |")
    lines.append("|---|---:|---:|---:|---|")
    for label, emp_col, thy_col in theory_pairs:
        if emp_col not in ok.columns or thy_col not in ok.columns:
            continue
        sub = ok[[emp_col, thy_col]].dropna()
        if len(sub) < 10:
            continue
        r, p_val = stats.pearsonr(sub[thy_col], sub[emp_col])
        mae = np.mean(np.abs(sub[thy_col] - sub[emp_col]))
        expl = "WELL" if r > 0.5 else "PARTIAL" if r > 0.2 else "POORLY"
        lines.append(f"| {label} | {len(sub)} | {r:.3f} (p={p_val:.2e}) | {mae:.4f} | {expl} |")
    lines.append("")

    # === By Dataset Size ===
    lines.append("## 4. By Dataset Size")
    lines.append("")
    ok["size_bin"] = pd.cut(ok["n_rows"], bins=[0, 500, 2000, 10000, 500000],
                            labels=["<500", "500-2K", "2K-10K", ">10K"])

    for label, col in [("A: Norm LR", "a_lr_gap_diff"), ("B: Peek K=10", "b_infl_k10"),
                       ("H: Dup RF 10%", "h_rf_10"), ("J: Compound", "j_compound_delta")]:
        if col not in ok.columns:
            continue
        lines.append(f"### {label}")
        lines.append("| Size | N | Mean | Median | t | p |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for b in ["<500", "500-2K", "2K-10K", ">10K"]:
            s = ok[ok["size_bin"] == b][col].dropna()
            if len(s) < 3:
                lines.append(f"| {b} | {len(s)} | — | — | — | — |")
            else:
                t_s, p_v = stats.ttest_1samp(s, 0)
                lines.append(f"| {b} | {len(s)} | {s.mean():+.4f} | {s.median():+.4f} | "
                             f"{t_s:.2f} | {p_v:.4f} |")
        lines.append("")

    # === By p/n Ratio ===
    lines.append("## 5. By p/n Ratio")
    lines.append("")
    ok["pn_bin"] = pd.cut(ok["p_over_n"], bins=[0, 0.01, 0.05, 0.2, 1.0, 100],
                          labels=["<0.01", "0.01-0.05", "0.05-0.2", "0.2-1.0", ">1.0"])

    for label, col in [("A: Norm LR", "a_lr_gap_diff"), ("C: FSel LR", "c_lr_gap_diff"),
                       ("E: Imp LR 10%", "e_lr_gap_diff_10")]:
        if col not in ok.columns:
            continue
        lines.append(f"### {label}")
        lines.append("| p/n | N | Mean | Median | t | p |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for b in ["<0.01", "0.01-0.05", "0.05-0.2", "0.2-1.0", ">1.0"]:
            s = ok[ok["pn_bin"] == b][col].dropna()
            if len(s) < 3:
                lines.append(f"| {b} | {len(s)} | — | — | — | — |")
            else:
                t_s, p_v = stats.ttest_1samp(s, 0)
                lines.append(f"| {b} | {len(s)} | {s.mean():+.4f} | {s.median():+.4f} | "
                             f"{t_s:.2f} | {p_v:.4f} |")
        lines.append("")

    # === Regression: What Predicts Each Leakage Type? ===
    lines.append("## 6. Inductive Regression")
    lines.append("")
    lines.append("`effect ~ log(N) + log(p) + p/n + n_cat + imbalance + nan_pct`")
    lines.append("")

    from sklearn.linear_model import LinearRegression
    predictors = ["n_rows", "n_features", "p_over_n", "n_cat", "imbalance", "nan_pct"]
    pred_labels = ["log(N)", "log(p)", "p/n", "n_cat", "imbalance", "nan_pct"]

    for label, col in summary_cols[:10]:  # Top 10 experiments
        if col not in ok.columns:
            continue
        sub = ok[predictors + [col]].dropna()
        if len(sub) < 30:
            continue

        X_reg = np.column_stack([
            np.log1p(sub["n_rows"].values),
            np.log1p(sub["n_features"].values),
            sub["p_over_n"].values,
            sub["n_cat"].values,
            sub["imbalance"].values,
            sub["nan_pct"].values,
        ])
        y_reg = sub[col].values
        reg = LinearRegression().fit(X_reg, y_reg)
        r2 = reg.score(X_reg, y_reg)

        lines.append(f"### {label} (R²={r2:.3f})")
        lines.append("| Predictor | Coefficient |")
        lines.append("|---|---:|")
        for fn, c in zip(pred_labels, reg.coef_):
            lines.append(f"| {fn} | {c:+.6f} |")
        lines.append(f"| intercept | {reg.intercept_:+.6f} |")
        lines.append("")

    # === Discovery vs Confirmation ===
    lines.append("## 7. Discovery vs Confirmation")
    lines.append("")
    lines.append("| Experiment | Discovery mean | Confirmation mean | Confirmed? |")
    lines.append("|---|---:|---:|---|")

    for label, col in summary_cols:
        if col not in disc.columns or col not in conf.columns:
            continue
        d_vals = disc[col].dropna()
        c_vals = conf[col].dropna()
        if len(d_vals) < 5 or len(c_vals) < 5:
            continue
        d_t, d_p = stats.ttest_1samp(d_vals, 0)
        c_t, c_p = stats.ttest_1samp(c_vals, 0)
        confirmed = "YES" if (d_p < 0.05 and c_p < 0.05 and np.sign(d_vals.mean()) == np.sign(c_vals.mean())) else "NO"
        lines.append(f"| {label} | {d_vals.mean():+.4f} (p={d_p:.3e}) | "
                     f"{c_vals.mean():+.4f} (p={c_p:.3e}) | {confirmed} |")
    lines.append("")

    # === Compound Additivity ===
    lines.append("## 8. Compound Additivity")
    lines.append("")

    if "j_compound_delta" in ok.columns:
        compound = ok[["j_compound_delta", "a_lr_gap_diff", "b_infl_k10",
                        "c_lr_gap_diff", "h_rf_10"]].dropna()
        if len(compound) > 10:
            compound["sum_individual"] = (
                compound["a_lr_gap_diff"].fillna(0) +
                compound["b_infl_k10"].fillna(0) +
                compound["c_lr_gap_diff"].fillna(0) +
                compound["h_rf_10"].fillna(0)
            )
            compound["ratio"] = compound["j_compound_delta"] / compound["sum_individual"].replace(0, np.nan)
            med_ratio = compound["ratio"].dropna().median()
            lines.append(f"N datasets with all experiments: {len(compound)}")
            lines.append(f"Median compound / sum(individual): {med_ratio:.2f}")
            lines.append("  > 1.0 = superadditive, < 1.0 = subadditive")
            lines.append(f"  Mean compound: {compound['j_compound_delta'].mean():+.4f}")
            lines.append(f"  Mean sum individual: {compound['sum_individual'].mean():+.4f}")
    lines.append("")

    # === Extreme Datasets ===
    lines.append("## 9. Extreme Datasets (Top 10)")
    lines.append("")
    for label, col in [("A: Norm LR", "a_lr_gap_diff"), ("B: Peek K=10", "b_infl_k10"),
                       ("H: Dup RF 10%", "h_rf_10"), ("J: Compound", "j_compound_delta")]:
        if col not in ok.columns:
            continue
        valid = ok[["name", "source", "n_rows", "n_features", "p_over_n", col]].dropna()
        valid = valid.sort_values(col, ascending=False)
        if len(valid) < 5:
            continue
        lines.append(f"### {label}")
        lines.append("| Dataset | Source | N | p | p/n | Effect |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, r in valid.head(10).iterrows():
            lines.append(f"| {r['name'][:35]} | {r['source']} | {int(r['n_rows'])} | "
                         f"{int(r['n_features'])} | {r['p_over_n']:.3f} | {r[col]:+.4f} |")
        lines.append("")

    # === Tuning vs Leakage ===
    lines.append("## 10. Tuning Gain vs Leakage Magnitude")
    lines.append("")
    lines.append("If tuning gain >> leakage, then leakage is noise in practice.")
    lines.append("")
    lines.append("| Algorithm | Median tuning gain | Median norm leak | Median dup leak | Ratio (tune/leak) |")
    lines.append("|---|---:|---:|---:|---:|")
    for algo in ["lr", "rf", "dt", "knn"]:
        tune_col = f"k_{algo}_gain"
        norm_col = f"a_{algo}_gap_diff"
        dup_col = f"h_{algo}_10"
        if tune_col in ok.columns:
            tune_med = ok[tune_col].dropna().median()
            norm_med = ok[norm_col].dropna().abs().median() if norm_col in ok.columns else 0
            dup_med = ok[dup_col].dropna().abs().median() if dup_col in ok.columns else 0
            max_leak = max(norm_med, dup_med, 1e-6)
            ratio = abs(tune_med) / max_leak
            lines.append(f"| {algo.upper()} | {tune_med:+.4f} | {norm_med:.4f} | {dup_med:.4f} | {ratio:.1f}x |")
    lines.append("")

    # === Model Variance as Mediator ===
    lines.append("## 11. Model Variance as Mediator")
    lines.append("")
    lines.append("Does model variance predict leakage susceptibility?")
    lines.append("")

    # Cross-algorithm: variance vs normalization leakage
    lines.append("### Variance (L) × Normalization Leakage (A)")
    lines.append("| Algorithm | r(variance, leak) | p-value | Overfit × dup leak r | p |")
    lines.append("|---|---:|---:|---:|---:|")
    for algo in ["lr", "rf", "dt", "knn"]:
        var_col = f"l_{algo}_std"
        norm_col = f"a_{algo}_gap_diff"
        overfit_col = f"l_{algo}_overfit"
        dup_col = f"h_{algo}_10"

        r_norm, p_norm = None, None
        r_dup, p_dup = None, None

        if var_col in ok.columns and norm_col in ok.columns:
            sub = ok[[var_col, norm_col]].dropna()
            if len(sub) > 10:
                r_norm, p_norm = stats.pearsonr(sub[var_col], sub[norm_col])

        if overfit_col in ok.columns and dup_col in ok.columns:
            sub = ok[[overfit_col, dup_col]].dropna()
            if len(sub) > 10:
                r_dup, p_dup = stats.pearsonr(sub[overfit_col], sub[dup_col])

        r_n_str = f"{r_norm:.3f}" if r_norm is not None else "—"
        p_n_str = f"{p_norm:.2e}" if p_norm is not None else "—"
        r_d_str = f"{r_dup:.3f}" if r_dup is not None else "—"
        p_d_str = f"{p_dup:.2e}" if p_dup is not None else "—"
        lines.append(f"| {algo.upper()} | {r_n_str} | {p_n_str} | {r_d_str} | {p_d_str} |")
    lines.append("")

    # Algorithm × Leakage Heatmap (Figure 4)
    lines.append("### Algorithm × Leakage Heatmap (Figure 4)")
    lines.append("")
    lines.append("Median leakage effect per algorithm:")
    lines.append("")
    lines.append("| Leakage Type | LR | RF | DT | KNN |")
    lines.append("|---|---:|---:|---:|---:|")
    for leak_label, prefix, suffix in [
        ("Normalization", "a_", "_gap_diff"),
        ("Duplicates 10%", "h_", "_10"),
    ]:
        row = f"| {leak_label} |"
        for algo in ["lr", "rf", "dt", "knn"]:
            col = f"{prefix}{algo}{suffix}"
            if col in ok.columns:
                med = ok[col].dropna().median()
                row += f" {med:+.4f} |"
            else:
                row += " — |"
        lines.append(row)
    lines.append("")

    # === Hypothesis Test Summary with BH FDR ===
    lines.append("## 12. Hypothesis Tests (BH FDR corrected)")
    lines.append("")

    pvals = []
    test_labels = []
    for label, col in summary_cols:
        if col not in ok.columns:
            continue
        vals = ok[col].dropna()
        if len(vals) < 5:
            continue
        _, p_v = stats.ttest_1samp(vals, 0)
        pvals.append(p_v)
        test_labels.append(label)

    if pvals:
        # BH FDR correction
        m = len(pvals)
        sorted_idx = np.argsort(pvals)
        sorted_p = np.array(pvals)[sorted_idx]
        bh_threshold = np.array([(i + 1) / m * 0.05 for i in range(m)])
        reject = sorted_p <= bh_threshold
        # Find largest k where p(k) <= k/m * alpha
        max_reject = 0
        for k in range(m):
            if reject[k]:
                max_reject = k + 1
        rejected = set(sorted_idx[:max_reject])

        lines.append("| # | Test | p-value | BH reject? |")
        lines.append("|---:|---|---:|---|")
        for j, (lab, pv) in enumerate(zip(test_labels, pvals)):
            rej = "YES ***" if j in rejected else "no"
            lines.append(f"| {j+1} | {lab} | {pv:.2e} | {rej} |")
        lines.append(f"\nFDR level: 0.05, {len(rejected)}/{m} hypotheses rejected")
    lines.append("")

    # === Dataset Diagnostics (W: Row-Order Detection) ===
    # W is a diagnostic, not a leakage magnitude — separate section
    lines.append("## 13. Dataset Diagnostics (Row-Order Detection)")
    lines.append("")
    lines.append("W measures whether datasets are sorted by outcome/time.")
    lines.append("Use `w_ordered` as a meta-regression covariate, not a leakage effect.")
    lines.append("")
    if "w_rowindex_auc" in ok.columns:
        w_sub = ok[["w_rowindex_auc", "w_ordered", "n_rows", "source"]].dropna()
        n_ordered = w_sub["w_ordered"].sum() if "w_ordered" in w_sub.columns else "—"
        lines.append(f"**Datasets tested**: {len(w_sub)}")
        lines.append(f"**Likely ordered (w_ordered=True)**: {n_ordered} "
                     f"({100*n_ordered/max(len(w_sub), 1):.1f}%)")
        lines.append(f"**Median row-index AUC**: {w_sub['w_rowindex_auc'].median():.3f}")
        lines.append("")
        # By source
        lines.append("| Source | N | Pct ordered | Median AUC |")
        lines.append("|---|---:|---:|---:|")
        for src in ["ml", "pmlb", "openml"]:
            s = w_sub[w_sub["source"] == src]
            if len(s) < 3:
                continue
            pct = 100 * s["w_ordered"].sum() / len(s) if "w_ordered" in s else 0
            lines.append(f"| {src} | {len(s)} | {pct:.1f}% | "
                         f"{s['w_rowindex_auc'].median():.3f} |")
        lines.append("")

        # Does ordering correlate with other leakages?
        if "d_gap_diff" in ok.columns:
            w_merge = ok[["w_ordered", "d_gap_diff"]].dropna()
            if len(w_merge) > 10:
                ordered = w_merge[w_merge["w_ordered"] == True]["d_gap_diff"]
                unordered = w_merge[w_merge["w_ordered"] == False]["d_gap_diff"]
                if len(ordered) > 3 and len(unordered) > 3:
                    t_stat, p_val = stats.ttest_ind(ordered, unordered)
                    lines.append("**Split strategy leakage (D) by row-order status:**")
                    lines.append(f"  Ordered datasets: mean={ordered.mean():+.4f} (N={len(ordered)})")
                    lines.append(f"  Unordered datasets: mean={unordered.mean():+.4f} (N={len(unordered)})")
                    lines.append(f"  t={t_stat:.2f}, p={p_val:.3e}")
                    lines.append("")
    lines.append("")

    # === Target Proxy Dose-Response (R) ===
    lines.append("## 14. Target Proxy Dose-Response (Calibration)")
    lines.append("")
    lines.append("R is a CALIBRATION experiment: controlled injection, not discovery.")
    lines.append("strength=0 is null control; should show ~0 delta.")
    lines.append("")
    if "r_lr_delta_null" in ok.columns:
        r_cols = [("null (strength=0)", "r_lr_delta_null"),
                  ("weak (strength=0.5)", "r_lr_delta_weak"),
                  ("medium (strength=2)", "r_lr_delta_med"),
                  ("strong (strength=5)", "r_lr_delta_strong")]
        lines.append("| Strength | N | Mean delta | Median delta | Std |")
        lines.append("|---|---:|---:|---:|---:|")
        for label, col in r_cols:
            if col in ok.columns:
                vals = ok[col].dropna()
                if len(vals) > 3:
                    lines.append(f"| {label} | {len(vals)} | {vals.mean():+.4f} | "
                                 f"{vals.median():+.4f} | {vals.std():.4f} |")
        lines.append("")
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated by run_leakage_landscape.py. {len(ok)} datasets. "
                 f"{time.strftime('%Y-%m-%d %H:%M:%S')}*")

    output = "\n".join(lines)
    with open(REPORT_FILE, "w") as f:
        f.write(output)
    print(f"\nReport saved to {REPORT_FILE}")
    print(output[:5000])


if __name__ == "__main__":
    main()
