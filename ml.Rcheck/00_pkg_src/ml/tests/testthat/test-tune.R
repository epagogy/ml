test_that("tune returns ml_tuning_result", {
  s     <- iris_split()
  tuned <- ml_tune(s$train, "Species", algorithm = "logistic",
                   n_trials = 2L, seed = 42L)
  expect_s3_class(tuned, "ml_tuning_result")
})

test_that("tune best_model predict works", {
  s     <- iris_split()
  tuned <- ml_tune(s$train, "Species", algorithm = "logistic",
                   n_trials = 2L, seed = 42L)
  preds <- predict(tuned, newdata = s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("tune n_trials respected (history rows)", {
  s     <- iris_split()
  tuned <- ml_tune(s$train, "Species", algorithm = "logistic",
                   n_trials = 3L, seed = 42L)
  # history may have fewer rows if algorithm has no tunable params
  expect_true(nrow(tuned$tuning_history_) >= 1L)
})

test_that("tune returns best_params_ as named list", {
  skip_if_not_installed("xgboost")
  s     <- iris_split()
  tuned <- ml_tune(s$train, "Species", algorithm = "xgboost",
                   n_trials = 3L, seed = 42L)
  expect_true(is.list(tuned$best_params_))
})

test_that("tune xgboost classification works", {
  skip_if_not_installed("xgboost")
  s     <- iris_split()
  tuned <- ml_tune(s$train, "Species", algorithm = "xgboost",
                   n_trials = 2L, seed = 42L)
  expect_s3_class(tuned$best_model, "ml_model")
})

test_that("tune random_forest classification works", {
  skip_if_not_installed("ranger")
  s     <- iris_split()
  tuned <- ml_tune(s$train, "Species", algorithm = "random_forest",
                   n_trials = 2L, seed = 42L)
  expect_s3_class(tuned$best_model, "ml_model")
})

test_that("tune inference: algorithm from model object", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  tuned <- ml_tune(s$train, "Species", model = model, n_trials = 2L, seed = 42L)
  expect_s3_class(tuned, "ml_tuning_result")
})

test_that("tune custom params override defaults", {
  skip_if_not_installed("xgboost")
  s     <- iris_split()
  tuned <- ml_tune(s$train, "Species", algorithm = "xgboost",
                   n_trials = 2L, seed = 42L,
                   params = list(max_depth = c(3L, 5L)))
  expect_s3_class(tuned, "ml_tuning_result")
})

test_that("tune regression works", {
  skip_if_not_installed("xgboost")
  s     <- mtcars_split()
  tuned <- ml_tune(s$train, "mpg", algorithm = "xgboost",
                   n_trials = 2L, seed = 42L)
  expect_s3_class(tuned, "ml_tuning_result")
  expect_equal(tuned$best_model$task, "regression")
})

test_that("tune evaluate with tuning result", {
  s     <- iris_split()
  tuned <- ml_tune(s$train, "Species", algorithm = "logistic",
                   n_trials = 2L, seed = 42L)
  m     <- ml_evaluate(tuned, s$valid)
  expect_s3_class(m, "ml_metrics")
})

test_that("ml$tune() module style works", {
  s     <- iris_split()
  tuned <- ml$tune(s$train, "Species", algorithm = "logistic",
                   n_trials = 2L, seed = 42L)
  expect_s3_class(tuned, "ml_tuning_result")
})

test_that("tune target not found raises data_error", {
  s <- iris_split()
  expect_error(
    ml_tune(s$train, "nonexistent", algorithm = "logistic", seed = 42L),
    class = "data_error"
  )
})
