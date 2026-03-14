#!/usr/bin/env python3
"""
Safe test runner — works on MacBook Air 8GB, GitHub Actions, any machine.

Each test file runs in its own subprocess (full RAM release between files).
Peak memory per subprocess: ~510MB max. Total suite: ~10 min on MacBook Air.

Profiled 2026-02-26 on server: 917 tests, ALL PASS, max 509MB/file.

Usage:
    python3 safe_test_runner.py              # full suite (all non-@slow tests)
    python3 safe_test_runner.py --quick      # fast files only (<5s each)
    python3 safe_test_runner.py --file tests/test_stack.py
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

PYTHON = sys.executable
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
TIMEOUT_SECS = 120  # per file — 2.7x server max (45s). Kills hangs, allows M1/M2 variance.

# Quick files: ≤183 MB peak on server (with @slow excluded).
# Excluded from quick (>300 MB peak): test_predict, test_validate, test_optimize
# test_help.py safe now — quick() tests marked @slow, only lightweight help() tests remain
QUICK_FILES = [
    "test_version.py",
    "test_config.py",
    "test_repr.py",
    "test_blend.py",
    "test_null_flags.py",
    "test_cluster.py",
    "test_profile.py",
    "test_scoring.py",
    "test_select.py",
    "test_utils.py",
    "test_leak.py",
    "test_interact.py",
    "test_normalize.py",
    "test_nested.py",
    "test_plot.py",
    "test_hardening.py",
    "test_bin.py",
    "test_report.py",
    "test_assess.py",
    "test_check_data.py",
    "test_shelf.py",
    "test_io.py",
    "test_predict_intervals.py",
    "test_evaluate.py",
    "test_split.py",
    "test_scale.py",
    "test_impute.py",
    "test_tokenize.py",
    "test_encode.py",
    "test_encode_datetime.py",
    "test_enough.py",
    "test_help.py",
    "test_pipeline.py",
    "test_calibrate.py",
    "test_preprocessing.py",
    "test_explain.py",
]


def get_all_files():
    """All test files: quick ones first (fail fast), slow ones after."""
    quick_set = set(QUICK_FILES)
    all_files = sorted(
        os.path.basename(f)
        for f in glob.glob(os.path.join(TEST_DIR, "test_*.py"))
    )
    rest = [f for f in all_files if f not in quick_set]
    return QUICK_FILES + rest


def run_file(test_file):
    filepath = os.path.join(TEST_DIR, test_file)
    if not os.path.exists(filepath):
        return "SKIP", 0, 0, "not found"

    cmd = [
        PYTHON, "-m", "pytest", filepath,
        "-x", "-q", "--tb=short", "--no-header",
        "-m", "not slow",
        "-p", "no:cacheprovider",
    ]

    # Thread caps guaranteed BEFORE any import in subprocess.
    # conftest.py also sets these, but subprocess env is the only race-free path.
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "2",
    }

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECS,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
        )
        elapsed = time.time() - start
        output = result.stdout + result.stderr

        passed = 0
        m = re.search(r"(\d+) passed", output)
        if m:
            passed = int(m.group(1))
        mf = re.search(r"(\d+) failed", output)
        if mf:
            passed -= int(mf.group(1))

        if result.returncode == 0:
            return "PASS", elapsed, passed, ""
        if result.returncode == 5:
            return "SKIP", elapsed, 0, "all @slow"
        if "MemoryError" in output or "Cannot allocate" in output:
            return "OOM", elapsed, passed, output[-400:]

        fail_lines = [ln for ln in output.splitlines() if "FAILED" in ln or "ERROR" in ln]
        snippet = "\n".join(fail_lines[:5]) if fail_lines else output[-400:]
        return "FAIL", elapsed, passed, snippet

    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - start, 0, f"killed after {TIMEOUT_SECS}s"


def main():
    parser = argparse.ArgumentParser(
        description="Safe subprocess test runner — MacBook Air / CI compatible"
    )
    parser.add_argument("--file", help="Run one file only")
    parser.add_argument("--quick", action="store_true", help="Fast files only (<5s each)")
    args = parser.parse_args()

    if args.file:
        files = [os.path.basename(args.file)]
    elif args.quick:
        files = [f for f in QUICK_FILES if os.path.exists(os.path.join(TEST_DIR, f))]
    else:
        files = get_all_files()

    print(f"=== ml test suite === ({len(files)} files, {TIMEOUT_SECS}s limit/file)")
    print("    Peak RAM/file ≤510MB  |  Total ≈10min on MacBook Air M1")
    print()

    total_passed = 0
    problems = []
    results = []

    for i, f in enumerate(files, 1):
        print(f"[{i:2d}/{len(files)}] {f:45s}", end="", flush=True)
        status, elapsed, passed, detail = run_file(f)
        total_passed += passed

        if status == "PASS":
            print(f"  PASS  {passed:3d}  {elapsed:5.1f}s")
        elif status == "SKIP":
            print(f"  skip  ({detail})")
        elif status == "FAIL":
            problems.append((f, status, elapsed, detail))
            print(f"  FAIL  {passed:3d} ok  {elapsed:5.1f}s")
            for line in detail.split("\n")[:2]:
                if line.strip():
                    print(f"         {line[:120]}")
        elif status == "TIMEOUT":
            problems.append((f, status, elapsed, detail))
            print(f"  TIMEOUT  {elapsed:5.1f}s  ← investigate")
        elif status == "OOM":
            problems.append((f, status, elapsed, detail))
            print(f"  OOM      {elapsed:5.1f}s  ← investigate")

        results.append((f, status, elapsed, passed))

    print()
    print("=" * 65)
    print(f"  {total_passed} passed   {len(problems)} problems")
    print()

    if problems:
        print("PROBLEMS:")
        for f, s, e, _d in problems:
            print(f"  {s:8s}  {f}  ({e:.1f}s)")
    else:
        print("  ALL CLEAN — ready to push!")


if __name__ == "__main__":
    main()
