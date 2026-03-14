# W2 API Contract Parity — error classes, return types, field names
# Max 20 tests. Python is the source of truth.

library(ml)

# ── ERROR CLASS NAMES ────────────────────────────────────────────────────────

test_that("W2-01: config_error class matches Python ConfigError (config_error + ml_error)", {
  err <- tryCatch(
    ml_split(iris, "Species", seed = 42L, ratio = c(0.9, 0.1)),  # wrong length → config_error
    error = function(e) e
  )
  expect_true(inherits(err, "config_error"),
              info = paste("Expected config_error class, got:", paste(class(err), collapse=", ")))
  expect_true(inherits(err, "ml_error"))
})

test_that("W2-02: data_error class raised for missing target column", {
  # Python: DataError("target='xxx' not found in data...")
  err <- tryCatch(
    ml_split(iris, "nonexistent_col", seed = 42L),
    error = function(e) e
  )
  expect_true(inherits(err, "data_error"),
              info = paste("Expected data_error class, got:", paste(class(err), collapse=", ")))
  expect_true(inherits(err, "ml_error"))
})

test_that("W2-03: model_error class raised for missing scores_ in shelf()", {
  s     <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  err   <- tryCatch(
    ml_shelf(model, new = s$test, target = "Species"),
    error = function(e) e
  )
  expect_true(inherits(err, "model_error"),
              info = paste("Expected model_error class, got:", paste(class(err), collapse=", ")))
})

test_that("W2-04: config_error message includes helpful guidance (not just 'Error')", {
  err <- tryCatch(
    ml_split(iris, "Species", seed = 42L, ratio = c(0.5, 0.5)),  # sums to 1 but wrong length
    error = function(e) e
  )
  expect_true(inherits(err, "config_error"))
  expect_true(nchar(conditionMessage(err)) > 10,
              info = "Error message should be non-trivial")
})

# ── SPLIT RESULT CONTRACT ────────────────────────────────────────────────────

test_that("W2-05: split result has correct S3 class (ml_split_result)", {
  s <- ml_split(iris, "Species", seed = 42L)
  expect_true(inherits(s, "ml_split_result"))
})

test_that("W2-06: split result supports $ accessor for train/valid/test/dev", {
  s <- ml_split(iris, "Species", seed = 42L)
  expect_true(is.data.frame(s$train))
  expect_true(is.data.frame(s$valid))
  expect_true(is.data.frame(s$test))
  expect_true(is.data.frame(s$dev))  # train + valid combined
})

test_that("W2-07: CV split result has correct S3 class (ml_split_result)", {
  cv <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  expect_true(inherits(cv, "ml_split_result"),
              info = paste("Expected ml_split_result, got:", paste(class(cv), collapse=", ")))
})

test_that("W2-08: CV split result has $folds list and $k integer", {
  cv <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  expect_true(is.list(cv$folds))
  expect_equal(length(cv$folds), 3L)
  expect_equal(cv$k, 3L)
})

# ── MODEL RESULT CONTRACT ────────────────────────────────────────────────────

test_that("W2-09: fit result has correct S3 class (ml_model)", {
  s <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_true(inherits(model, "ml_model"))
})

test_that("W2-10: model has required fields (algorithm, task, target, features)", {
  # Python Model has: algorithm, task, _target, _features, cv_score, scores_
  # R must expose: algorithm, task, target, features
  s     <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_equal(model$algorithm, "logistic")
  expect_equal(model$task, "classification")
  expect_equal(model$target, "Species")
  expect_true(is.character(model$features))
  expect_true(length(model$features) > 0L)
})

# ── METRICS CONTRACT ─────────────────────────────────────────────────────────

test_that("W2-11: evaluate result has correct S3 class (ml_metrics)", {
  s     <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m     <- ml_evaluate(model, s$valid)
  expect_true(inherits(m, "ml_metrics"))
})

test_that("W2-12: metrics support [[]] access by name", {
  s     <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m     <- ml_evaluate(model, s$valid)
  acc   <- m[["accuracy"]]
  expect_true(is.numeric(acc))
  expect_true(acc >= 0 && acc <= 1)
})

# ── COMPARE RESULT CONTRACT ───────────────────────────────────────────────────

test_that("W2-13: compare returns ml_leaderboard class", {
  s  <- ml_split(iris, "Species", seed = 42L)
  m1 <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m2 <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42L)
  lb <- ml_compare(list(m1, m2), s$valid)
  expect_true(inherits(lb, "ml_leaderboard"),
              info = paste("Expected ml_leaderboard, got:", paste(class(lb), collapse=", ")))
})

test_that("W2-14: compare leaderboard has 'algorithm' and 'time' columns", {
  # Python Leaderboard df has: algorithm, {metric}, cv_std, time_seconds
  s  <- ml_split(iris, "Species", seed = 42L)
  m1 <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m2 <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42L)
  lb <- ml_compare(list(m1, m2), s$valid)
  df <- as.data.frame(lb)
  expect_true("algorithm" %in% names(df),
              info = paste("Missing 'algorithm' col. Got:", paste(names(df), collapse=", ")))
  expect_true("time" %in% names(df),
              info = paste("Missing 'time' col. Got:", paste(names(df), collapse=", ")))
})

# ── VALIDATE RESULT CONTRACT ──────────────────────────────────────────────────

test_that("W2-15: validate result has passed, metrics, failures fields", {
  # Python ValidateResult: .passed, .metrics, .failures
  s     <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  gate  <- ml_validate(model, test = s$test, rules = list(accuracy = "> 0.5"))
  expect_true(is.logical(gate$passed))
  expect_true(is.list(gate$metrics) || is.numeric(gate$metrics))
  expect_true(is.character(gate$failures))
})

# ── EXPLAIN CONTRACT ──────────────────────────────────────────────────────────

test_that("W2-16: explain returns ml_explanation with feature/importance columns", {
  # Python: Explanation wraps DataFrame with feature/importance columns
  # R: ml_explanation with $feature and $importance
  s     <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42L)
  exp   <- ml_explain(model)
  expect_true(inherits(exp, "ml_explanation"),
              info = paste("Expected ml_explanation, got:", paste(class(exp), collapse=", ")))
  df <- as.data.frame(exp)
  expect_true("feature" %in% names(df))
  expect_true("importance" %in% names(df))
})

# ── PROFILE RESULT CONTRACT ───────────────────────────────────────────────────

test_that("W2-17: profile result $task matches detected task", {
  p <- ml_profile(iris, "Species")
  expect_equal(p$task, "classification")
  p2 <- ml_profile(mtcars, "mpg")
  expect_equal(p2$task, "regression")
})

test_that("W2-18: profile result $warnings is character vector", {
  # Python: warnings is a list of strings
  p <- ml_profile(iris, "Species")
  expect_true(is.character(p$warnings),
              info = paste("Expected character, got:", class(p$warnings)))
})

# ── DRIFT RESULT CONTRACT ─────────────────────────────────────────────────────

test_that("W2-19: drift result class is ml_drift_result with correct fields", {
  s   <- ml_split(iris, "Species", seed = 42L)
  res <- ml_drift(reference = s$train, new = s$test, target = "Species")
  expect_true(inherits(res, "ml_drift_result"))
  # All required Python DriftResult fields
  for (field_name in c("shifted", "features", "features_shifted", "severity",
                        "n_reference", "n_new", "threshold", "distinguishable")) {
    expect_true(field_name %in% names(res),
                info = paste("Missing drift field:", field_name))
  }
})

test_that("W2-20: drift result $auc is NULL for statistical method (matches Python None)", {
  # Python: DriftResult.auc = None for statistical mode
  s   <- ml_split(iris, "Species", seed = 42L)
  res <- ml_drift(reference = s$train, new = s$test, target = "Species",
                   method = "statistical")
  expect_null(res$auc, label = "auc should be NULL for statistical method")
  expect_null(res$train_scores, label = "train_scores should be NULL for statistical method")
})
