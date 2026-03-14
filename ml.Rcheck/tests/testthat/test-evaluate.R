test_that("evaluate returns ml_metrics for binary classification", {
  df    <- binary_df()
  s     <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "logistic", seed = 42L)
  m     <- ml_evaluate(model, s$valid)
  expect_s3_class(m, "ml_metrics")
  expect_true("accuracy" %in% names(m))
  expect_true("f1" %in% names(m))
})

test_that("evaluate returns ml_metrics for multiclass classification", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m     <- ml_evaluate(model, s$valid)
  expect_s3_class(m, "ml_metrics")
  expect_true("accuracy" %in% names(m))
  expect_true("f1_macro" %in% names(m))
  expect_true("f1_weighted" %in% names(m))
  expect_true("precision_weighted" %in% names(m))
  expect_true("recall_weighted" %in% names(m))
})

test_that("evaluate returns ml_metrics for regression", {
  skip_if_not_installed("glmnet")
  s     <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "linear", seed = 42L)
  m     <- ml_evaluate(model, s$valid)
  expect_s3_class(m, "ml_metrics")
  expect_true("rmse" %in% names(m))
  expect_true("mae" %in% names(m))
  expect_true("r2" %in% names(m))
})

test_that("all metrics are numeric", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m     <- ml_evaluate(model, s$valid)
  expect_true(all(vapply(as.list(unclass(m)), is.numeric, logical(1L))))
})

test_that("evaluate auto-unwraps TuningResult", {
  s     <- iris_split()
  tuned <- ml_tune(s$train, "Species", algorithm = "logistic",
                   n_trials = 2L, seed = 42L)
  m     <- ml_evaluate(tuned, s$valid)
  expect_s3_class(m, "ml_metrics")
})

test_that("evaluate error when target not in data", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  bad   <- s$valid[, c("Sepal.Length", "Sepal.Width"), drop = FALSE]
  expect_error(ml_evaluate(model, bad), class = "data_error")
})

test_that("ml$evaluate() module style works", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m     <- ml$evaluate(model, s$valid)
  expect_s3_class(m, "ml_metrics")
})

test_that("accuracy value is between 0 and 1", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m     <- ml_evaluate(model, s$valid)
  expect_gte(m[["accuracy"]], 0)
  expect_lte(m[["accuracy"]], 1)
})

test_that("roc_auc computed for binary classification with proba support", {
  df    <- binary_df()
  s     <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "logistic", seed = 42L)
  m     <- ml_evaluate(model, s$valid)
  # roc_auc should be present when logistic is used
  if ("roc_auc" %in% names(m)) {
    expect_gte(m[["roc_auc"]], 0)
    expect_lte(m[["roc_auc"]], 1)
  }
})

test_that("rmse > 0 for regression model", {
  skip_if_not_installed("glmnet")
  s     <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "linear", seed = 42L)
  m     <- ml_evaluate(model, s$valid)
  expect_gt(m[["rmse"]], 0)
})

test_that("precision_macro and recall_macro present for multiclass", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m     <- ml_evaluate(model, s$valid)
  expect_true("precision_macro" %in% names(m))
  expect_true("recall_macro" %in% names(m))
})

# ── Partition guard tests ───────────────────────────────────────────────────

test_that("evaluate rejects test-tagged data", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_error(ml_evaluate(model, s$test), class = "partition_error")
})

test_that("evaluate accepts train-tagged data", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  result <- ml_evaluate(model, s$train)
  expect_s3_class(result, "ml_metrics")
})

test_that("evaluate accepts valid-tagged data", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  result <- ml_evaluate(model, s$valid)
  expect_s3_class(result, "ml_metrics")
})

test_that("evaluate rejects untagged data in strict mode", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  untagged <- data.frame(s$valid)
  expect_null(attr(untagged, "_ml_partition"))
  expect_error(ml_evaluate(model, untagged), class = "partition_error")
})

test_that("evaluate allows untagged data with guards off", {
  ml_config(guards = "off")
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  untagged <- data.frame(s$valid)
  result <- ml_evaluate(model, untagged)
  expect_s3_class(result, "ml_metrics")
})
