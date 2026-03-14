# Benchmarks

## parity_check.R

Verifies that ml (R) produces identical results to raw ranger/nnet on the same data, split, and algorithm.

```r
Rscript benchmarks/parity_check.R
```

**Results (ml 1.0.0 / ranger 0.18.0):**

```
Random Forest — diabetes (regression)
  rmse       74.9298    74.9298    0.0000 v
  mae        60.5788    60.5788    0.0000 v
  r2          -0.0414    -0.0414    0.0000 v

Random Forest — cancer (binary clf)
  accuracy    0.5965     0.5965    0.0000 v

Logistic Regression — iris binary (scaled)
  accuracy    1.0000     1.0000    0.0000 v

3/3 passed — ml wraps ranger/nnet faithfully.
```

ml handles encoding, scaling, and splits automatically. Raw R requires manual preprocessing. Same algorithm + same data = same numbers.
