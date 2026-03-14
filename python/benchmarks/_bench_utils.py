"""Shared benchmark utilities.

Used by bench_fair.py, bench_functions.py, bench_compare.py, bench_workflows.py.
"""

from __future__ import annotations

import gc
import os
import platform
import time
import tracemalloc
from typing import Any


def make_dataset(
    task: str, n_rows: int, n_features: int, seed: int = 42,
) -> tuple:
    """Generate synthetic dataset for benchmarking.

    Returns (df_with_target, split_result). Shared across bench_engines and bench_dev.
    """
    import pandas as pd
    from sklearn.datasets import make_classification, make_regression

    import ml

    if task == "classification":
        X, y = make_classification(
            n_samples=n_rows,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_classes=2,
            random_state=seed,
        )
    else:
        X, y = make_regression(
            n_samples=n_rows,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            random_state=seed,
        )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y
    s = ml.split(df, "target", seed=seed)
    return df, s


def capture_versions() -> dict[str, Any]:
    """Capture library versions + hardware info for reproducible benchmarks."""
    import numpy
    import pandas

    try:
        import sklearn
        sk_ver = sklearn.__version__
    except ImportError:
        sk_ver = "not installed"

    try:
        import ml as _ml
        ml_ver = getattr(_ml, "__version__", "unknown")
    except ImportError:
        ml_ver = "not installed"

    cpu_count = os.cpu_count() or 1
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        ram_gb = -1.0

    return {
        "ml": ml_ver,
        "sklearn": sk_ver,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "cpu_count": cpu_count,
        "ram_gb": ram_gb,
    }


def run_timed(
    fn,
    *,
    warmup: int = 5,
    runs: int = 10,
    measure_rss: bool = True,
) -> dict[str, float]:
    """Run function with warmup + measurement. Returns timing + memory stats.

    Returns dict with:
        min_seconds, median_seconds, p95_seconds, iqr_seconds,
        rss_delta_mb (psutil, -1 if unavailable),
        tracemalloc_peak_mb (Python-heap only, supplementary).
    """
    # Warmup — absorbs import JIT, Cython compilation, numpy pool init
    for _ in range(warmup):
        gc.collect()
        fn()

    # Get psutil process handle (if available)
    proc = None
    if measure_rss:
        try:
            import psutil
            proc = psutil.Process()
        except ImportError:
            pass

    # Measured runs
    times: list[float] = []
    rss_deltas: list[int] = []
    tm_peaks: list[int] = []

    for _ in range(runs):
        gc.collect()

        rss_before = proc.memory_info().rss if proc else None

        tracemalloc.start()
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        _, tm_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(elapsed)
        tm_peaks.append(tm_peak)
        if rss_before is not None:
            rss_deltas.append(proc.memory_info().rss - rss_before)

    times.sort()
    n = len(times)

    # Correct IQR: interpolated quartiles
    q1_idx = (n - 1) * 0.25
    q3_idx = (n - 1) * 0.75
    q1 = times[int(q1_idx)] + (q1_idx % 1) * (
        times[min(int(q1_idx) + 1, n - 1)] - times[int(q1_idx)]
    )
    q3 = times[int(q3_idx)] + (q3_idx % 1) * (
        times[min(int(q3_idx) + 1, n - 1)] - times[int(q3_idx)]
    )

    # RSS delta: median (can be negative due to GC — clamp to 0)
    if rss_deltas:
        rss_sorted = sorted(rss_deltas)
        rss_med = rss_sorted[len(rss_sorted) // 2]
        rss_mb = round(max(0, rss_med) / (1024 * 1024), 2)
    else:
        rss_mb = -1.0

    # tracemalloc peak: median
    tm_sorted = sorted(tm_peaks)
    tm_med = tm_sorted[len(tm_sorted) // 2]

    return {
        "min_seconds": round(times[0], 5),
        "median_seconds": round(times[n // 2], 5),
        "p95_seconds": round(times[int(n * 0.95)], 5),
        "iqr_seconds": round(q3 - q1, 5),
        "rss_delta_mb": rss_mb,
        "tracemalloc_peak_mb": round(tm_med / (1024 * 1024), 2),
    }


def print_table(
    rows: list[dict[str, Any]],
    title: str,
    columns: list[str] | None = None,
) -> None:
    """Print a columnar table to stdout."""
    if not rows:
        return

    if columns is None:
        columns = list(rows[0].keys())

    # Compute column widths
    widths = {col: len(col) for col in columns}
    formatted: list[list[str]] = []
    for row in rows:
        cells = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                cell = f"{val:.4f}" if abs(val) < 100 else f"{val:.1f}"
            else:
                cell = str(val)
            widths[col] = max(widths[col], len(cell))
            cells.append(cell)
        formatted.append(cells)

    # Print
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

    # Header
    header = "  ".join(col.rjust(widths[col]) for col in columns)
    print(f"  {header}")
    separator = "  ".join("─" * widths[col] for col in columns)
    print(f"  {separator}")

    # Rows
    for cells in formatted:
        line = "  ".join(cell.rjust(widths[columns[i]]) for i, cell in enumerate(cells))
        print(f"  {line}")

    print()
