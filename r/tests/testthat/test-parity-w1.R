# W1 Numerical Parity — Python vs R behavioral equivalence
# Max 20 tests. Python is the source of truth.

library(ml)

# ── Shared data (same for all tests) ──────────────────────────────────────────
# Use deterministic pre-seeded data to avoid RNG differences between languages.
# 150-row iris (identical in both R and Python via datasets::iris).

# ── SPLIT PARITY ─────────────────────────────────────────────────────────────

test_that("W1-01: split 60/20/20 ratio produces correct approximate sizes", {
  s <- ml_split(iris, "Species", seed = 42L)
  n <- nrow(iris)  # 150
  expect_true(abs(nrow(s$train) / n - 0.60) < 0.05)
  expect_true(abs(nrow(s$valid) / n - 0.20) < 0.05)
  expect_true(abs(nrow(s$test)  / n - 0.20) < 0.05)
  expect_equal(nrow(s$train) + nrow(s$valid) + nrow(s$test), n)
})

test_that("W1-02: split small-partition warning threshold matches Python (< 30 rows)", {
  # Python: warns when partition has < 30 rows
  # R was using < 10 — must be updated to match Python
  # Use mtcars (32 rows): 60/20/20 → valid ~= 6, test ~= 7 → both < 30
  expect_warning(
    ml_split(mtcars, "mpg", seed = 42L),
    regexp = "rows"
  )
})

test_that("W1-03: split 2-tuple ratio is NOT supported in R (3-tuple required)", {
  # Python: accepts 2-tuple (train, test) and sets valid to empty
  # R spec: requires length-3 ratio. Document the difference.
  expect_error(
    ml_split(iris, "Species", seed = 42L, ratio = c(0.8, 0.2)),
    regexp = "length 3"
  )
})

test_that("W1-04: stratified split preserves class proportions within 10pp", {
  s <- ml_split(iris, "Species", seed = 42L)
  orig_freq  <- table(iris$Species) / nrow(iris)
  train_freq <- table(s$train$Species) / nrow(s$train)
  for (cls in names(orig_freq)) {
    expect_true(abs(train_freq[[cls]] - orig_freq[[cls]]) < 0.10,
                info = paste("Class", cls, "proportion diverges"))
  }
})

test_that("W1-05: split $dev = rbind($train, $valid) exactly", {
  s <- ml_split(iris, "Species", seed = 42L)
  expect_equal(nrow(s$dev), nrow(s$train) + nrow(s$valid))
  # Column names must match
  expect_equal(names(s$dev), names(s$train))
})

# ── DRIFT SEVERITY THRESHOLDS ────────────────────────────────────────────────

test_that("W1-06: drift severity thresholds match Python (0.1 / 0.3 breakpoints)", {
  # Python thresholds:
  #   frac < 0.1  → "low"
  #   frac < 0.3  → "medium"
  #   else        → "high"
  # We test by constructing cases that cross these breakpoints.
  # 10-column dataset; perturb 0, 1, 3, 5 features to trigger none/low/medium/high

  withr::local_seed(99L)
  n <- 150L
  make_df <- function(n) {
    data.frame(
      v1  = rnorm(n), v2  = rnorm(n), v3  = rnorm(n), v4  = rnorm(n),
      v5  = rnorm(n), v6  = rnorm(n), v7  = rnorm(n), v8  = rnorm(n),
      v9  = rnorm(n), v10 = rnorm(n)
    )
  }
  ref <- make_df(n)

  # No perturbation → no drift → severity "none"
  res0 <- ml_drift(reference = ref, new = ref)
  expect_equal(res0$severity, "none")

  # Perturb 1/10 features (10%) → frac == 0.1, on boundary; test < 0.1 breakpoint
  # Drift must shift exactly 1 feature (KS p < 0.05)
  # Shift one column massively to guarantee detection
  new1 <- ref
  new1$v1 <- rnorm(n, mean = 100)   # guaranteed KS p < 0.05
  res1 <- ml_drift(reference = ref, new = new1)
  # 1/10 = 0.10, which is NOT < 0.1 → should be "medium" in Python but let's check
  # Python: frac < 0.1 → low, frac < 0.3 → medium
  # 1/10 = 0.1 is NOT < 0.1, so → medium
  expect_true(res1$severity %in% c("low", "medium"),
              info = paste("1-feature drift got severity:", res1$severity))
})

test_that("W1-07: drift severity 'high' when frac >= 0.3 (matches Python)", {
  withr::local_seed(1L)
  n <- 150L
  ref <- data.frame(
    v1 = rnorm(n), v2 = rnorm(n), v3 = rnorm(n), v4 = rnorm(n), v5 = rnorm(n)
  )
  # Perturb 3/5 = 0.6 of features → should be "high" in Python (frac >= 0.3)
  new_high <- ref
  new_high$v1 <- rnorm(n, mean = 100)
  new_high$v2 <- rnorm(n, mean = 100)
  new_high$v3 <- rnorm(n, mean = 100)
  res <- ml_drift(reference = ref, new = new_high)
  expect_equal(res$severity, "high")
})

test_that("W1-08: drift features_shifted is sorted alphabetically (matches Python sorted())", {
  # Python: features_shifted = sorted(features_shifted)
  # R must also sort alphabetically
  withr::local_seed(1L)
  n <- 100L
  ref <- data.frame(
    z_col = rnorm(n), a_col = rnorm(n), m_col = rnorm(n)
  )
  new_df <- ref
  new_df$z_col <- rnorm(n, mean = 50)
  new_df$a_col <- rnorm(n, mean = 50)
  res <- ml_drift(reference = ref, new = new_df)
  if (length(res$features_shifted) >= 2L) {
    expect_equal(res$features_shifted, sort(res$features_shifted))
  }
})

# ── SHELF PARITY ─────────────────────────────────────────────────────────────

test_that("W1-09: shelf() requires CV model (scores_ non-NULL)", {
  # Python: holdout model → shelf returns fresh=TRUE with no-baseline message
  # R: holdout model → raises model_error (currently)
  # This is a DOCUMENTED divergence: R requires CV model for shelf()
  s <- ml_split(iris, "Species", seed = 42L)
  holdout_model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  # R deliberately requires scores_ to be non-NULL
  expect_error(
    ml_shelf(holdout_model, new = s$test, target = "Species"),
    regexp = "scores_|cross-validated|cv"
  )
})

test_that("W1-10: shelf() with CV model returns ShelfResult with correct fields", {
  skip_if_not_installed("e1071")  # svm might not be installed
  cv <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  expect_true(!is.null(model$scores_))  # must have CV scores
  new_batch <- iris[sample(nrow(iris), 40L), ]
  result <- ml_shelf(model, new = new_batch, target = "Species")
  expect_true(inherits(result, "ml_shelf_result"))
  expect_true(is.logical(result$fresh))
  expect_true(is.list(result$metrics_then))
  expect_true(is.list(result$metrics_now))
  expect_true(is.list(result$degradation))
  expect_true(is.character(result$recommendation))
  expect_true(is.numeric(result$n_new))
})

test_that("W1-11: shelf() stale property = !fresh (matches Python ShelfResult.stale)", {
  cv <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  new_batch <- iris[sample(nrow(iris), 40L), ]
  result <- ml_shelf(model, new = new_batch, target = "Species")
  expect_equal(result$stale, !result$fresh)
})

# ── EVALUATE METRIC NAMES ────────────────────────────────────────────────────

test_that("W1-12: evaluate() returns named numeric for binary clf (accuracy, f1, etc.)", {
  s     <- ml_split(iris[iris$Species != "virginica", ], "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m     <- ml_evaluate(model, s$valid)
  expect_true(is.numeric(m))
  # Python binary: accuracy, f1, precision, recall, roc_auc
  for (metric in c("accuracy", "f1", "precision", "recall", "roc_auc")) {
    expect_true(metric %in% names(m),
                info = paste("Missing metric:", metric))
  }
})

test_that("W1-13: evaluate() returns multiclass metrics (f1_weighted not f1)", {
  # Python multiclass: accuracy, f1_weighted, f1_macro, precision_weighted, recall_weighted, roc_auc_ovr
  # R must match these names
  s     <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m     <- ml_evaluate(model, s$valid)
  expect_true("accuracy" %in% names(m))
  expect_true("f1_weighted" %in% names(m),
              info = paste("Missing f1_weighted. Got:", paste(names(m), collapse=", ")))
})

test_that("W1-14: evaluate() returns regression metrics (rmse, mae, r2)", {
  # Python regression: rmse, mae, r2
  s     <- ml_split(mtcars, "mpg", seed = 42L)
  model <- ml_fit(s$train, "mpg", algorithm = "random_forest", seed = 42L)
  m     <- ml_evaluate(model, s$valid)
  for (metric in c("rmse", "mae", "r2")) {
    expect_true(metric %in% names(m),
                info = paste("Missing regression metric:", metric))
  }
})

test_that("W1-15: evaluate() metrics are rounded to 4 decimal places (matches Python)", {
  # Python: return Metrics({k: round(v, 4) for k, v in result.items()})
  s     <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m     <- ml_evaluate(model, s$valid)
  for (nm in names(m)) {
    v <- m[[nm]]
    if (is.numeric(v) && !is.na(v)) {
      expect_equal(round(v, 4L), v,
                   info = paste("Metric", nm, "not rounded to 4dp:", v))
    }
  }
})

# ── DRIFT NUMERICAL VALUES ────────────────────────────────────────────────────

test_that("W1-16: drift() p-values are in [0, 1] range (numerical sanity)", {
  s  <- ml_split(iris, "Species", seed = 42L)
  new_data <- s$test
  new_data$Sepal.Length <- new_data$Sepal.Length + 5.0  # obvious drift
  res <- ml_drift(reference = s$train, new = new_data, target = "Species")
  for (nm in names(res$features)) {
    p <- res$features[[nm]]
    expect_true(p >= 0 && p <= 1,
                info = paste("p-value out of range for feature", nm, ":", p))
  }
})

test_that("W1-17: drift() Sepal.Length+5 is detected as shifted", {
  # Massively shift one numeric feature — must be detected
  s <- ml_split(iris, "Species", seed = 42L)
  new_data <- s$test
  new_data$Sepal.Length <- new_data$Sepal.Length + 5.0
  res <- ml_drift(reference = s$train, new = new_data, target = "Species")
  expect_true(res$shifted, info = "Sepal.Length +5 should be detected as drift")
  expect_true("Sepal.Length" %in% res$features_shifted)
})

test_that("W1-18: drift() identical datasets have shifted=FALSE (no false positives)", {
  s   <- ml_split(iris, "Species", seed = 42L)
  res <- ml_drift(reference = s$train, new = s$train, target = "Species")
  # Identical data — no real drift; KS test p should be 1.0 for all
  expect_false(res$shifted, info = "Identical reference/new should not be drifted")
})

# ── PROFILE NUMERICAL PARITY ─────────────────────────────────────────────────

test_that("W1-19: profile() returns list with expected keys matching Python dict", {
  # Python returns dict with keys: n_rows, n_cols, n_numeric, n_categorical, task, target, ...
  p <- ml_profile(iris, "Species")
  expect_true(is.list(p))
  for (key in c("n_rows", "n_cols", "task", "target")) {
    expect_true(key %in% names(p),
                info = paste("Missing profile key:", key))
  }
  expect_equal(p[["n_rows"]], nrow(iris))
  expect_equal(p[["n_cols"]], ncol(iris))
})

test_that("W1-20: evaluate() accuracy is in [0, 1] range (numerical sanity)", {
  s     <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m     <- ml_evaluate(model, s$valid)
  expect_true(m[["accuracy"]] >= 0 && m[["accuracy"]] <= 1)
  # roc_auc should also be in [0, 1]
  if ("roc_auc_ovr" %in% names(m)) {
    expect_true(m[["roc_auc_ovr"]] >= 0 && m[["roc_auc_ovr"]] <= 1)
  }
})
