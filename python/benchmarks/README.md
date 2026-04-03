# Benchmarks

## Per-Function Benchmarks

Measures wall time, peak memory, and throughput for each `ml` function across dataset sizes.

```bash
python benchmarks/bench_functions.py              # tiny + small (Mac)
python benchmarks/bench_functions.py --medium      # + 100K rows
python benchmarks/bench_functions.py --large       # + 1M rows (server only)
python benchmarks/bench_functions.py --json --output results.json
```

Functions benchmarked: `profile`, `split`, `fit`, `predict`, `evaluate`, `explain`, `screen`.

## Workflow Benchmarks

End-to-end timing for realistic user workflows.

```bash
python benchmarks/bench_workflows.py              # tiny + small
python benchmarks/bench_workflows.py --medium      # + 100K rows
```

Workflows:
- **quick**: split + fit + evaluate
- **standard**: split + screen + fit best + evaluate
- **power**: split + fit 2 models + compare + assess

## Cross-Library Comparison

Compare `ml.screen()` vs PyCaret `compare_models()` vs FLAML `AutoML.fit()`.

```bash
python benchmarks/bench_compare.py              # ml-only (Mac default)
python benchmarks/bench_compare.py --all        # all installed libraries
```

Gracefully skips libraries not installed. Install for full comparison:
```bash
pip install pycaret flaml
```

## Baseline Results (ml 1.0.0, MacBook Air M2 16GB)

**Function benchmarks** (median of 5 runs, 3 warmup):

| Function | Tiny (1K×10) | Small (10K×20) | Memory (small) |
|----------|-------------|---------------|---------------|
| profile | 3.9 ms | 14.9 ms | 3.1 MB |
| split | 4.8 ms | 13.2 ms | 5.6 MB |
| fit (RF clf) | 423 ms | 2.17 s | 5.8 MB |
| fit (logistic) | 14 ms | 30 ms | 8.8 MB |
| fit (RF reg) | 572 ms | 8.07 s | 5.8 MB |
| fit (linear) | 9.1 ms | 22.5 ms | 8.8 MB |
| predict | 8.6 ms | 21 ms | 1.0 MB |
| evaluate | 24.3 ms | 48.9 ms | 1.1 MB |
| explain | 13.2 ms | 14.3 ms | 0.2 MB |
| screen | 1.53 s | 0.75 s | 9.0 MB |

**Workflow benchmarks** (end-to-end):

| Workflow | Tiny | Small | Peak Memory |
|----------|------|-------|-------------|
| quick (split+fit+eval) | 0.53 s | 2.36 s | 7.4 MB |
| standard (split+screen+fit+eval) | 2.16 s | 3.07 s | 10.6 MB |
| power (split+fit×2+compare+assess) | 0.52 s | 2.41 s | 10.4 MB |

## Parity Check

Verifies that ml produces bit-identical results to raw sklearn on the same data, split, and algorithm.

```bash
python benchmarks/parity_check.py
```

**Results (ml 1.0.0 / sklearn 1.6.1):**

```
Random Forest — churn (binary clf)
  accuracy     0.7977     0.7977    0.0000 ✓
  f1           0.5622     0.5622    0.0000 ✓
  roc_auc      0.8323     0.8323    0.0000 ✓

Random Forest — tips (regression)
  rmse         1.2106     1.2106    0.0000 ✓
  mae          0.9413     0.9413    0.0000 ✓
  r2           0.0206     0.0206    0.0000 ✓

Logistic Regression — cancer (binary clf, scaled)
  accuracy     0.9737     0.9737    0.0000 ✓
  f1           0.9639     0.9639    0.0000 ✓
  roc_auc      1.0000     1.0000    0.0000 ✓

3/3 passed — ml wraps sklearn faithfully.
```
