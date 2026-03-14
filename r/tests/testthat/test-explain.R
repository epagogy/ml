test_that("explain returns ml_explanation with feature/importance columns", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  exp   <- ml_explain(model)
  expect_s3_class(exp, "ml_explanation")
  expect_s3_class(exp, "data.frame")
  expect_true("feature" %in% names(exp))
  expect_true("importance" %in% names(exp))
})

test_that("explain importance sums to ~1.0", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  exp   <- ml_explain(model)
  expect_true(abs(sum(exp$importance) - 1.0) < 0.01)
})

test_that("explain sorted descending by importance", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  exp   <- ml_explain(model)
  expect_true(all(diff(exp$importance) <= 0))
})

test_that("xgboost explain works", {
  skip_if_not_installed("xgboost")
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "xgboost", seed = 42L)
  exp   <- ml_explain(model)
  expect_s3_class(exp, "ml_explanation")
})

test_that("random_forest explain works", {
  skip_if_not_installed("ranger")
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42L)
  exp   <- ml_explain(model)
  expect_s3_class(exp, "ml_explanation")
})

test_that("explain errors for knn (not supported)", {
  skip_if_not_installed("kknn")
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "knn", seed = 42L)
  expect_error(ml_explain(model), class = "model_error")
})

test_that("explain auto-unwraps TuningResult", {
  s     <- iris_split()
  tuned <- ml_tune(s$train, "Species", algorithm = "logistic",
                   n_trials = 2L, seed = 42L)
  exp   <- ml_explain(tuned)
  expect_s3_class(exp, "ml_explanation")
})

test_that("ml$explain() module style works", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  exp   <- ml$explain(model)
  expect_s3_class(exp, "ml_explanation")
})
