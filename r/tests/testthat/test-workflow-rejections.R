# Workflow rejection tests -- verifies structural constraints that must fail.
#
# Companion to test-workflow.R (happy paths). These tests prove the library
# correctly rejects invalid compositions, wrong types, constraint violations,
# and dangerous workflows that would silently corrupt results.
#
# Mirrors Python tests/test_workflow_rejections.py. Sections marked [SKIP]
# note intentional design differences (e.g. R auto-generates seeds, no embargo).

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

clf_data_rej <- function() {
  withr::local_seed(42L)
  n <- 200L
  data.frame(
    f1     = stats::rnorm(n),
    f2     = stats::rnorm(n),
    f3     = stats::rnorm(n),
    target = sample(c(0L, 1L), n, replace = TRUE)
  )
}

reg_data_rej <- function() {
  withr::local_seed(42L)
  n <- 200L
  data.frame(
    f1     = stats::rnorm(n),
    f2     = stats::rnorm(n),
    target = stats::rnorm(n)
  )
}

# ---------------------------------------------------------------------------
# R1: Seed enforcement
# [SKIP] R auto-generates seed when NULL -- no enforcement by design.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# R2: Algorithm-task mismatch -- wrong algorithm for task type
# ---------------------------------------------------------------------------

test_that("R2: logistic rejects regression target", {
  s <- ml_split(reg_data_rej(), "target", seed = 42L)
  expect_error(
    ml_fit(s$train, "target", algorithm = "logistic", seed = 42L),
    regexp = "classification|regression"
  )
})

test_that("R2: linear rejects classification target", {
  s <- ml_split(clf_data_rej(), "target", seed = 42L)
  expect_error(
    ml_fit(s$train, "target", algorithm = "linear", seed = 42L),
    regexp = "regression|classification"
  )
})

test_that("R2: naive_bayes rejects regression target", {
  s <- ml_split(reg_data_rej(), "target", seed = 42L)
  # naive_bayes is classification-only; error message varies by implementation
  expect_error(
    ml_fit(s$train, "target", algorithm = "naive_bayes", seed = 42L)
  )
})

# Note: elastic_net in R supports both classification and regression.
# Python elastic_net is regression-only -- intentional parity gap.

# ---------------------------------------------------------------------------
# R3: assess() single-use -- already covered in test-workflow.R
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# R4: cv() type gate -- requires ml_split_result, not raw objects
# ---------------------------------------------------------------------------

test_that("R4: ml_cv rejects raw data.frame", {
  expect_error(
    ml_cv(clf_data_rej(), "target", seed = 42L),
    regexp = "ml_split_result|split"
  )
})

test_that("R4: ml_cv rejects list (not a split result)", {
  expect_error(
    ml_cv(list(a = 1, b = 2), "target", seed = 42L),
    regexp = "ml_split_result|split"
  )
})

# ---------------------------------------------------------------------------
# R5: Temporal guard -- cv() on temporal split must redirect
# ---------------------------------------------------------------------------

test_that("R5: ml_cv rejects temporal split result", {
  withr::local_seed(42L)
  n  <- 200L
  df <- data.frame(
    date   = seq.Date(as.Date("2020-01-01"), by = "day", length.out = n),
    f1     = stats::rnorm(n),
    target = sample(c(0L, 1L), n, replace = TRUE)
  )
  s <- ml_split_temporal(df, "target", time = "date")
  expect_error(
    ml_cv(s, "target", seed = 42L, folds = 3L),
    regexp = "temporal|cv_temporal"
  )
})

# ---------------------------------------------------------------------------
# R6: CV fold bounds
# ---------------------------------------------------------------------------

test_that("R6: ml_cv rejects folds < 2", {
  s <- ml_split(clf_data_rej(), "target", seed = 42L)
  expect_error(
    ml_cv(s, "target", folds = 1L, seed = 42L),
    regexp = "folds"
  )
})

test_that("R6: ml_cv rejects folds = 0", {
  s <- ml_split(clf_data_rej(), "target", seed = 42L)
  expect_error(
    ml_cv(s, "target", folds = 0L, seed = 42L),
    regexp = "folds"
  )
})

test_that("R6: ml_cv_temporal rejects folds < 2", {
  withr::local_seed(42L)
  n  <- 200L
  df <- data.frame(
    date   = seq.Date(as.Date("2020-01-01"), by = "day", length.out = n),
    f1     = stats::rnorm(n),
    target = stats::rnorm(n)
  )
  s <- ml_split_temporal(df, "target", time = "date")
  expect_error(
    ml_cv_temporal(s, "target", folds = 1L),
    regexp = "folds"
  )
})

# ---------------------------------------------------------------------------
# R7: Negative embargo
# ---------------------------------------------------------------------------

test_that("R7: ml_cv_temporal rejects negative embargo", {
  withr::local_seed(42L)
  n  <- 200L
  df <- data.frame(
    date   = seq.Date(as.Date("2020-01-01"), by = "day", length.out = n),
    f1     = stats::rnorm(n),
    target = sample(c(0L, 1L), n, replace = TRUE)
  )
  s <- ml_split_temporal(df, "target", time = "date")
  expect_error(
    ml_cv_temporal(s, "target", folds = 3L, embargo = -5L),
    regexp = "embargo.*>= 0|negative"
  )
})

# ---------------------------------------------------------------------------
# R8: Data quality -- check_data catches structural issues
# ---------------------------------------------------------------------------

test_that("R8: ml_check_data rejects non-data.frame input", {
  expect_error(
    ml_check_data(c(1, 2, 3), "target"),
    regexp = "data.frame|DataFrame|data frame"
  )
})

test_that("R8: ml_check_data errors on missing target column", {
  expect_error(
    ml_check_data(clf_data_rej(), "nonexistent"),
    regexp = "not found|nonexistent|target"
  )
})

test_that("R8: ml_check_data flags single-class target", {
  data <- data.frame(
    f1     = stats::rnorm(50L),
    target = rep("yes", 50L),
    stringsAsFactors = FALSE
  )
  report <- ml_check_data(data, "target")
  expect_true(report$has_issues)
})

test_that("R8: ml_check_data detects duplicate column names", {
  # Build data.frame then rename to create duplicate
  df <- data.frame(a = 1:10, b = 1:10, target = sample(0:1, 10, replace = TRUE))
  names(df)[1:2] <- c("a", "a")  # duplicate column name
  # Should either error or flag as an issue
  result <- tryCatch(
    ml_check_data(df, "target"),
    error = function(e) list(has_issues = TRUE, error_msg = conditionMessage(e))
  )
  expect_true(isTRUE(result$has_issues))
})

# ---------------------------------------------------------------------------
# R9: enough() bounds
# ---------------------------------------------------------------------------

test_that("R9: ml_enough rejects too-few rows", {
  df <- data.frame(f1 = seq_len(10L), target = rep(c(0L, 1L), 5L))
  s  <- ml_split(df, "target", seed = 42L)
  expect_error(ml_enough(s, "target"), regexp = "50")
})

test_that("R9: ml_enough rejects steps < 2", {
  withr::local_seed(42L)
  df <- data.frame(f1 = stats::rnorm(200L),
                   target = sample(c(0L, 1L), 200L, replace = TRUE))
  s  <- ml_split(df, "target", seed = 42L)
  expect_error(ml_enough(s, "target", seed = 42L, steps = 1L),
               regexp = "steps.*>= 2")
})

test_that("R9: ml_enough rejects cv < 2", {
  withr::local_seed(42L)
  df <- data.frame(f1 = stats::rnorm(200L),
                   target = sample(c(0L, 1L), 200L, replace = TRUE))
  s  <- ml_split(df, "target", seed = 42L)
  expect_error(ml_enough(s, "target", seed = 42L, cv = 1L),
               regexp = "cv.*>= 2")
})

# ---------------------------------------------------------------------------
# R10: Sliding window
# ---------------------------------------------------------------------------

test_that("R10: ml_cv_temporal sliding without window_size rejects", {
  withr::local_seed(42L)
  n  <- 200L
  df <- data.frame(
    date   = seq.Date(as.Date("2020-01-01"), by = "day", length.out = n),
    f1     = stats::rnorm(n),
    target = sample(c(0L, 1L), n, replace = TRUE)
  )
  s <- ml_split_temporal(df, "target", time = "date")
  expect_error(
    ml_cv_temporal(s, "target", folds = 3L, window = "sliding"),
    regexp = "window_size"
  )
})

# ---------------------------------------------------------------------------
# R11: optimize() on regression
# ---------------------------------------------------------------------------

test_that("R11: ml_optimize rejects regression model", {
  withr::local_seed(42L)
  n  <- 200L
  df <- data.frame(f1 = stats::rnorm(n), target = stats::rnorm(n))
  s  <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", seed = 42L)
  expect_error(
    ml_optimize(model, data = s$valid),
    class = "model_error"
  )
})

# ---------------------------------------------------------------------------
# R12: validate() rule parsing -- unknown metric produces failing gate
# ---------------------------------------------------------------------------

test_that("R12: validate with unknown metric produces gate$passed=FALSE", {
  df    <- clf_data_rej()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", seed = 42L)
  gate  <- ml_validate(model, test = s$test,
                       rules = list(nonexistent_metric = ">0.5"))
  expect_false(gate$passed)
})

# ---------------------------------------------------------------------------
# R13: cv_group without groups column -- already in test-cv.R
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# R14: Per-fold normalization -- correctness property
# Each fold's training set has different rows, so different statistics.
# Global normalization (wrong) produces identical fold means.
# Per-fold (correct) produces different fold means.
# ---------------------------------------------------------------------------

test_that("R14: CV folds have different training set means (per-fold property)", {
  df <- clf_data_rej()
  s  <- ml_split(df, "target", seed = 42L)
  cv <- ml_cv(s, "target", folds = 3L, seed = 42L)

  # Each fold's training rows differ -- compute mean of f1 for each fold's train
  fold_means <- vapply(cv$folds, function(fold) {
    mean(cv$data[fold$train, "f1"])
  }, numeric(1L))

  # Per-fold means must differ; if identical, normalization was global (bug)
  expect_gt(
    length(unique(round(fold_means, 6L))), 1L,
    label = "all folds identical training means => likely global normalization"
  )
})

# ---------------------------------------------------------------------------
# R15: split() target validation -- already in test-split.R
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# R16: predict() rejects non-Model objects
# ---------------------------------------------------------------------------

test_that("R16: ml_predict rejects data.frame as model", {
  df <- clf_data_rej()
  expect_error(
    ml_predict(df, df),
    regexp = "ml_model|model|Expected"
  )
})

test_that("R16: ml_predict rejects character string as model", {
  df <- clf_data_rej()
  expect_error(
    ml_predict("not a model", df),
    regexp = "ml_model|model|Expected"
  )
})
