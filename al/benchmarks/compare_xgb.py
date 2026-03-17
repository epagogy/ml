"""Compare Rust GBT vs XGBoost head-to-head from landscape.db."""
import sqlite3
import os

DB = os.path.join(os.path.dirname(__file__), "landscape.db")
conn = sqlite3.connect(DB)

datasets = ["iris", "cancer", "wine", "churn", "fraud", "titanic",
            "diabetes", "houses", "tips"]

print("=" * 70)
print("RUST GBT vs XGBOOST — HEAD-TO-HEAD")
print("=" * 70)
header = f"{'Dataset':12s} {'Task':5s} {'RustGBT':>10s} {'XGBoost':>10s} {'Diff':>8s} {'Winner':>8s}"
print(header)
print("-" * len(header))

wins = {"RUST": 0, "XGB": 0, "TIE": 0}

for ds in datasets:
    row = conn.execute("SELECT task FROM results WHERE dataset=? LIMIT 1", (ds,)).fetchone()
    if not row:
        continue
    task = row[0]
    metric = "accuracy" if task == "clf" else "r2"

    r = conn.execute(
        f"SELECT MAX({metric}) FROM results "
        "WHERE dataset=? AND algorithm='gradient_boosting' AND engine='ml' AND error IS NULL",
        (ds,),
    ).fetchone()
    rust_best = r[0] if r and r[0] else None

    r = conn.execute(
        f"SELECT MAX({metric}) FROM results "
        "WHERE dataset=? AND algorithm='xgboost' AND error IS NULL",
        (ds,),
    ).fetchone()
    xgb_best = r[0] if r and r[0] else None

    if rust_best is not None and xgb_best is not None:
        diff = rust_best - xgb_best
        winner = "RUST" if diff > 0.001 else ("XGB" if diff < -0.001 else "TIE")
        wins[winner] += 1
        print(f"{ds:12s} {task:5s} {rust_best:10.4f} {xgb_best:10.4f} {diff:+8.4f} {winner:>8s}")

print()
print(f"Score: RUST {wins['RUST']} — XGB {wins['XGB']} — TIE {wins['TIE']}")

print()
print("=" * 70)
print("OVERALL BEST PER DATASET (any algorithm, any engine)")
print("=" * 70)

for ds in datasets:
    row = conn.execute("SELECT task FROM results WHERE dataset=? LIMIT 1", (ds,)).fetchone()
    if not row:
        continue
    task = row[0]
    metric = "accuracy" if task == "clf" else "r2"

    r = conn.execute(
        f"SELECT algorithm, engine, {metric}, config FROM results "
        "WHERE dataset=? AND error IS NULL "
        f"ORDER BY {metric} DESC LIMIT 1",
        (ds,),
    ).fetchone()
    if r:
        print(f"  {ds:12s} {r[0]:20s} ({r[1]:7s}) {metric}={r[2]:.4f}  {r[3][:60]}")

print()
print("=" * 70)
print("RUST GBT BEST CONFIG PER DATASET")
print("=" * 70)

for ds in datasets:
    row = conn.execute("SELECT task FROM results WHERE dataset=? LIMIT 1", (ds,)).fetchone()
    if not row:
        continue
    task = row[0]
    metric = "accuracy" if task == "clf" else "r2"

    r = conn.execute(
        f"SELECT {metric}, config, fit_time_ms FROM results "
        "WHERE dataset=? AND algorithm='gradient_boosting' AND engine='ml' AND error IS NULL "
        f"ORDER BY {metric} DESC LIMIT 1",
        (ds,),
    ).fetchone()
    if r:
        print(f"  {ds:12s} {metric}={r[0]:.4f}  {r[2]:8.1f}ms  {r[1]}")

# Count total trials
total = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
ok = conn.execute("SELECT COUNT(*) FROM results WHERE error IS NULL").fetchone()[0]
err = conn.execute("SELECT COUNT(*) FROM results WHERE error IS NOT NULL").fetchone()[0]
print(f"\nTotal trials: {total} ({ok} ok, {err} errors)")

conn.close()
