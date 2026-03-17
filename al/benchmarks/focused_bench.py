"""Focused benchmark on datasets where Rust GBT trails XGBoost."""
import sqlite3
import json
import time
import socket

import numpy as np
import pandas as pd
import ml

DB_PATH = "benchmarks/landscape.db"
SEED = 42


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
        dataset TEXT NOT NULL, n_train INTEGER, n_test INTEGER,
        n_features INTEGER, n_classes INTEGER, task TEXT NOT NULL,
        algorithm TEXT NOT NULL, engine TEXT NOT NULL, config TEXT NOT NULL,
        fit_time_ms REAL, predict_time_ms REAL, accuracy REAL, f1 REAL,
        rmse REAL, r2 REAL, mae REAL, error TEXT,
        timestamp TEXT DEFAULT (datetime('now')),
        mlw_version TEXT, hostname TEXT
    )""")
    conn.commit()

    rows = conn.execute(
        "SELECT dataset, algorithm, engine, config FROM results WHERE error IS NULL"
    ).fetchall()
    existing = {(r[0], r[1], r[2], r[3]) for r in rows}

    run_id = f"focused_{int(time.time())}"
    hostname_str = socket.gethostname()

    # Build aggressive config grid
    configs = []

    # Stochastic + regularized + leaf_smooth sweep
    for n_est in [300, 400, 500]:
        for lr in [0.01, 0.03, 0.05, 0.1]:
            for depth in [4, 6, 8]:
                for ls in [0.0, 2.0, 5.0]:
                    cfg = {
                        "n_estimators": n_est,
                        "learning_rate": lr,
                        "max_depth": depth,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "reg_lambda": 1.0,
                    }
                    if ls > 0:
                        cfg["leaf_smooth"] = ls
                    configs.append(cfg)

    # Early stopping with many rounds
    for n_est in [500, 1000]:
        for lr in [0.01, 0.05]:
            for depth in [4, 6]:
                cfg = {
                    "n_estimators": n_est,
                    "learning_rate": lr,
                    "max_depth": depth,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                    "n_iter_no_change": 20,
                    "validation_fraction": 0.1,
                }
                configs.append(cfg)

    # DART + leaf_smooth combos
    for n_est in [300, 500]:
        for lr in [0.05, 0.1]:
            for depth in [5, 7]:
                for dr in [0.05, 0.1]:
                    for ls in [0.0, 3.0]:
                        cfg = {
                            "n_estimators": n_est,
                            "learning_rate": lr,
                            "max_depth": depth,
                            "dart_rate": dr,
                        }
                        if ls > 0:
                            cfg["leaf_smooth"] = ls
                        configs.append(cfg)

    TARGETS = {
        "cancer": "diagnosis",
        "churn": "churn",
        "fraud": "fraud",
        "diabetes": "progression",
        "tips": "tip",
    }
    TASKS = {
        "cancer": "clf",
        "churn": "clf",
        "fraud": "clf",
        "diabetes": "reg",
        "tips": "reg",
    }

    # Load datasets
    datasets = {}
    for name in TARGETS:
        df = ml.dataset(name)
        target = TARGETS[name]
        s = ml.split(df, target, ratio=(0.49, 0.01, 0.5), seed=SEED)
        datasets[name] = (s.dev, s.test, target, TASKS[name])

    total = len(configs) * len(datasets)
    done = 0
    skipped = 0
    new_results = 0

    from sklearn.metrics import accuracy_score, f1_score

    for ds_name, (train, test, target, task) in datasets.items():
        n_classes = train[target].nunique() if task == "clf" else 0
        for cfg in configs:
            config_str = json.dumps(cfg, sort_keys=True)
            key = (ds_name, "gradient_boosting", "ml", config_str)
            done += 1
            if key in existing:
                skipped += 1
                continue

            result = {
                "dataset": ds_name,
                "n_train": len(train),
                "n_test": len(test),
                "n_features": train.shape[1] - 1,
                "n_classes": n_classes,
                "task": task,
                "algorithm": "gradient_boosting",
                "engine": "ml",
                "config": config_str,
            }
            try:
                t0 = time.perf_counter()
                model = ml.fit(
                    train,
                    target,
                    algorithm="gradient_boosting",
                    seed=SEED,
                    engine="ml",
                    **cfg,
                )
                fit_ms = (time.perf_counter() - t0) * 1000

                preds = ml.predict(model, test.drop(columns=[target]))
                y_true = test[target]

                if task == "clf":
                    result["accuracy"] = round(accuracy_score(y_true, preds), 6)
                    result["f1"] = round(
                        f1_score(y_true, preds, average="weighted", zero_division=0), 6
                    )
                else:
                    y_t = y_true.values.astype(float)
                    p = preds.values.astype(float)
                    ss_res = np.sum((y_t - p) ** 2)
                    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
                    result["r2"] = round(1 - ss_res / max(ss_tot, 1e-15), 6)
                    result["rmse"] = round(np.sqrt(np.mean((y_t - p) ** 2)), 6)
                    result["mae"] = round(np.mean(np.abs(y_t - p)), 6)

                result["fit_time_ms"] = round(fit_ms, 2)
            except Exception as e:
                result["error"] = f"{type(e).__name__}: {e}"

            conn.execute(
                """INSERT INTO results (run_id, dataset, n_train, n_test, n_features,
                   n_classes, task, algorithm, engine, config, fit_time_ms, predict_time_ms,
                   accuracy, f1, rmse, r2, mae, error, mlw_version, hostname)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    result["dataset"],
                    result.get("n_train"),
                    result.get("n_test"),
                    result.get("n_features"),
                    result.get("n_classes"),
                    result["task"],
                    result["algorithm"],
                    result["engine"],
                    result["config"],
                    result.get("fit_time_ms"),
                    result.get("predict_time_ms"),
                    result.get("accuracy"),
                    result.get("f1"),
                    result.get("rmse"),
                    result.get("r2"),
                    result.get("mae"),
                    result.get("error"),
                    ml.__version__,
                    hostname_str,
                ),
            )
            conn.commit()
            new_results += 1
            existing.add(key)

            if done % 100 == 0:
                print(f"Progress: {done}/{total} ({skipped} skipped, {new_results} new)")

    print(f"Done: {total} total, {skipped} skipped, {new_results} new")
    conn.close()


if __name__ == "__main__":
    main()
