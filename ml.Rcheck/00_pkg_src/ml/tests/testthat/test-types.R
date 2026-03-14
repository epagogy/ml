test_that("ml_split_result: $train, $valid, $test accessible", {
  s <- iris_split()
  expect_s3_class(s, "ml_split_result")
  expect_true(is.data.frame(s$train))
  expect_true(is.data.frame(s$valid))
  expect_true(is.data.frame(s$test))
})

test_that("ml_split_result: $dev = rbind(train, valid)", {
  s <- iris_split()
  expect_equal(nrow(s$dev), nrow(s$train) + nrow(s$valid))
})

test_that("ml_split_result: print works", {
  s <- iris_split()
  expect_output(print(s), "Split")
})

test_that("ml_cv_result: accessing $train raises config_error", {
  cv <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  expect_s3_class(cv, "ml_cv_result")
  expect_error(cv$train, class = "config_error")
  expect_error(cv$valid, class = "config_error")
  expect_error(cv$test,  class = "config_error")
})

test_that("ml_cv_result: $folds accessible", {
  cv <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  expect_equal(length(cv$folds), 3L)
})

test_that("ml_model construction and attribute access", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_s3_class(model, "ml_model")
  expect_equal(model$algorithm, "logistic")
  expect_equal(model$task, "classification")
  expect_equal(model$target, "Species")
  expect_true(is.numeric(model$seed))
  expect_true(model$n_train > 0L)
  expect_true(model$time >= 0)
})

test_that("ml_model print works", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_output(print(model), "Model")
})

test_that("predict.ml_model dispatches correctly", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  preds <- predict(model, newdata = s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("ml_tuning_result: best_model delegates predict", {
  s      <- iris_split()
  tuned  <- ml_tune(s$train, "Species", algorithm = "logistic",
                    n_trials = 2L, seed = 42L)
  expect_s3_class(tuned, "ml_tuning_result")
  preds  <- predict(tuned, newdata = s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("ml_tuning_result print works", {
  s     <- iris_split()
  tuned <- ml_tune(s$train, "Species", algorithm = "logistic",
                   n_trials = 2L, seed = 42L)
  expect_output(print(tuned), "Tuned")
})

test_that("ml_metrics print works", {
  s       <- iris_split()
  model   <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  metrics <- ml_evaluate(model, s$valid)
  expect_s3_class(metrics, "ml_metrics")
  expect_output(print(metrics), "Metrics")
})

test_that("ml_explanation is a data.frame with feature/importance columns", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  exp   <- ml_explain(model)
  expect_s3_class(exp, "ml_explanation")
  expect_true("feature" %in% names(exp))
  expect_true("importance" %in% names(exp))
})

test_that("ml_validate_result print works", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  gate  <- ml_validate(model, test = s$test, rules = list(accuracy = ">0.1"))
  expect_s3_class(gate, "ml_validate_result")
  expect_output(print(gate), "Validate")
})

test_that("ml_profile_result print works", {
  prof <- ml_profile(iris, "Species")
  expect_s3_class(prof, "ml_profile_result")
  expect_output(print(prof), "Profile")
})

test_that("ml_leaderboard print works", {
  s  <- iris_split()
  lb <- ml_screen(s, "Species", seed = 42L,
                  algorithms = c("logistic"))
  expect_s3_class(lb, "ml_leaderboard")
  expect_output(print(lb), "Leaderboard")
})
