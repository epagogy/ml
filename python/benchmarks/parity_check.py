"""Parity check: ml vs raw sklearn on the same data, split, and algorithm.

Proves that ml wraps sklearn faithfully — same predictions, same metrics.
Run: python benchmarks/parity_check.py
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

import ml


def compare(name, ml_metrics, sk_metrics):
    """Print side-by-side comparison."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  {'metric':<12} {'ml':>10} {'sklearn':>10} {'delta':>10}")
    print(f"  {'-' * 44}")
    all_close = True
    for key in ml_metrics:
        ml_val = ml_metrics[key]
        sk_val = sk_metrics.get(key)
        if sk_val is None:
            continue
        delta = abs(ml_val - sk_val)
        marker = "✓" if delta < 0.0001 else "~" if delta < 0.01 else "✗"
        if delta >= 0.01:
            all_close = False
        print(f"  {key:<12} {ml_val:>10.4f} {sk_val:>10.4f} {delta:>9.4f} {marker}")
    return all_close


def test_random_forest_classification():
    """Random Forest on churn — binary classification."""
    data = ml.dataset("churn")
    target = "churn"

    # ml workflow
    s = ml.split(data, target, seed=42)
    ml_model = ml.fit(s.train, target, algorithm="random_forest", seed=42)
    ml_metrics = ml.evaluate(ml_model, s.valid)
    # sklearn workflow — use exact same split
    train_df = s.train.copy()
    valid_df = s.valid.copy()

    le = LabelEncoder()
    le.fit(train_df[target])

    X_train = train_df.drop(columns=[target])
    y_train = le.transform(train_df[target])
    X_valid = valid_df.drop(columns=[target])
    y_valid = le.transform(valid_df[target])

    # Ordinal-encode categoricals (same as ml default)
    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        cats = X_train[col].astype("category").cat.categories
        mapping = {c: i for i, c in enumerate(cats)}
        X_train[col] = X_train[col].map(mapping).fillna(-1).astype(float)
        X_valid[col] = X_valid[col].map(mapping).fillna(-1).astype(float)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    sk_preds = clf.predict(X_valid)
    sk_proba = clf.predict_proba(X_valid)

    # ml uses average="binary" for binary classification
    sk_metrics = {
        "accuracy": accuracy_score(y_valid, sk_preds),
        "f1": f1_score(y_valid, sk_preds, average="binary", zero_division=0),
        "roc_auc": roc_auc_score(y_valid, sk_proba[:, 1]),
    }

    return compare("Random Forest — churn (binary clf)", ml_metrics, sk_metrics)


def test_random_forest_regression():
    """Random Forest on tips — regression."""
    data = ml.dataset("tips")
    target = "tip"

    # ml workflow
    s = ml.split(data, target, seed=42)
    ml_model = ml.fit(s.train, target, algorithm="random_forest", seed=42)
    ml_metrics = ml.evaluate(ml_model, s.valid)

    # sklearn workflow — use exact same split
    train_df = s.train.copy()
    valid_df = s.valid.copy()

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_valid = valid_df.drop(columns=[target])
    y_valid = valid_df[target]

    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        cats = X_train[col].astype("category").cat.categories
        mapping = {c: i for i, c in enumerate(cats)}
        X_train[col] = X_train[col].map(mapping).fillna(-1).astype(float)
        X_valid[col] = X_valid[col].map(mapping).fillna(-1).astype(float)

    from sklearn.ensemble import RandomForestRegressor

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    sk_preds = reg.predict(X_valid)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    sk_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_valid, sk_preds))),
        "mae": mean_absolute_error(y_valid, sk_preds),
        "r2": r2_score(y_valid, sk_preds),
    }

    return compare("Random Forest — tips (regression)", ml_metrics, sk_metrics)


def test_logistic_classification():
    """Logistic regression on cancer — needs scaling."""
    data = ml.dataset("cancer")
    target = "diagnosis"

    # ml workflow (auto-scales for logistic)
    s = ml.split(data, target, seed=42)
    ml_model = ml.fit(s.train, target, algorithm="logistic", seed=42)
    ml_metrics = ml.evaluate(ml_model, s.valid)

    # sklearn workflow — must manually scale + encode string labels
    train_df = s.train.copy()
    valid_df = s.valid.copy()

    le = LabelEncoder()
    le.fit(train_df[target])
    y_train = le.transform(train_df[target])
    y_valid = le.transform(valid_df[target])

    X_train = train_df.drop(columns=[target]).values.astype(float)
    X_valid = valid_df.drop(columns=[target]).values.astype(float)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    sk_preds = clf.predict(X_valid)
    sk_proba = clf.predict_proba(X_valid)

    sk_metrics = {
        "accuracy": accuracy_score(y_valid, sk_preds),
        "f1": f1_score(y_valid, sk_preds, average="binary", zero_division=0),
        "roc_auc": roc_auc_score(y_valid, sk_proba[:, 1]),
    }

    return compare("Logistic Regression — cancer (binary clf, scaled)", ml_metrics, sk_metrics)


def test_cross_library_churn():
    """Compare ml vs FLAML vs PyCaret on churn — same data, each library's best."""
    data = ml.dataset("churn")
    target = "churn"

    # ml: explicit algorithm choice
    s = ml.split(data, target, seed=42)
    ml_model = ml.fit(s.train, target, algorithm="random_forest", seed=42)
    ml_acc = ml.evaluate(ml_model, s.valid)["accuracy"]

    # Prepare sklearn-style arrays for FLAML
    le = LabelEncoder()
    le.fit(s.train[target])

    X_train = s.train.drop(columns=[target]).copy()
    y_train = le.transform(s.train[target])
    X_valid = s.valid.drop(columns=[target]).copy()
    y_valid = le.transform(s.valid[target])

    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        cats = X_train[col].astype("category").cat.categories
        mapping = {c: i for i, c in enumerate(cats)}
        X_train[col] = X_train[col].map(mapping).fillna(-1).astype(float)
        X_valid[col] = X_valid[col].map(mapping).fillna(-1).astype(float)

    print(f"\n{'=' * 60}")
    print("  Cross-library comparison — churn (binary clf)")
    print(f"{'=' * 60}")
    print(f"  {'library':<16} {'accuracy':>10} {'notes'}")
    print(f"  {'-' * 50}")
    print(f"  {'ml (RF)':<16} {ml_acc:>10.4f}   explicit algorithm, 3-way split")

    # sklearn baseline
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    sk_acc = accuracy_score(y_valid, clf.predict(X_valid))
    print(f"  {'sklearn (RF)':<16} {sk_acc:>10.4f}   manual preprocessing")

    # FLAML (AutoML — searches algorithms automatically)
    try:
        from flaml import AutoML
        automl = AutoML()
        automl.fit(
            X_train=X_train, y_train=y_train,
            task="classification", time_budget=30, seed=42,
            verbose=0,
        )
        flaml_acc = accuracy_score(y_valid, automl.predict(X_valid))
        print(f"  {'FLAML (auto)':<16} {flaml_acc:>10.4f}   30s search, best: {automl.best_estimator}")
    except ImportError:
        print(f"  {'FLAML':<16} {'(not installed)':>10}")

    # PyCaret (AutoML)
    try:
        import warnings

        from pycaret.classification import compare_models, predict_model, setup
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_df = s.train.copy()
            valid_df = s.valid.copy()
            setup(train_df, target=target, session_id=42, verbose=False)
            best = compare_models(sort="Accuracy", verbose=False, n_select=1)
            preds = predict_model(best, data=valid_df, verbose=False)
            pc_acc = accuracy_score(
                le.transform(valid_df[target]),
                le.transform(preds["prediction_label"]),
            )
            print(f"  {'PyCaret (auto)':<16} {pc_acc:>10.4f}   auto search")
    except ImportError:
        print(f"  {'PyCaret':<16} {'(not installed)':>10}")
    except Exception as e:
        print(f"  {'PyCaret':<16} {'(error)':<10}   {e}")

    print("\n  All libraries use the same train/valid data.")
    print("  AutoML libraries search many algorithms; ml uses one you choose.")


if __name__ == "__main__":
    import sys

    import sklearn

    cross = "--cross-library" in sys.argv

    print("ml parity check — same data, same split, same algorithm")
    print(f"ml {ml.__version__} | sklearn {sklearn.__version__}")
    print()

    results = []
    results.append(test_random_forest_classification())
    results.append(test_random_forest_regression())
    results.append(test_logistic_classification())

    print(f"\n{'=' * 60}")
    passed = sum(results)
    total = len(results)
    print(f"  {passed}/{total} parity checks passed (delta < 0.01)")
    if passed == total:
        print("  ml wraps sklearn faithfully — same results.")
    else:
        print("  Some deltas > 0.01 — investigate preprocessing differences.")
    print(f"{'=' * 60}")

    if cross:
        test_cross_library_churn()
    else:
        print("\n  Cross-library comparison skipped. Run with --cross-library to enable.")
