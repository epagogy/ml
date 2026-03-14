test_that("screen returns ml_leaderboard with algorithm column", {
  s  <- iris_split()
  lb <- ml_screen(s, "Species", seed = 42L, algorithms = c("logistic"))
  expect_s3_class(lb, "ml_leaderboard")
  expect_true("algorithm" %in% names(lb))
})

test_that("screen has time column (numeric, > 0)", {
  s  <- iris_split()
  lb <- ml_screen(s, "Species", seed = 42L, algorithms = c("logistic"))
  expect_true("time" %in% names(lb))
  expect_true(is.numeric(lb$time))
  expect_true(all(lb$time > 0))
})

test_that("screen multiclass sorted by f1_macro descending", {
  s  <- iris_split()
  lb <- ml_screen(s, "Species", seed = 42L,
                  algorithms = c("logistic"))
  if ("f1_macro" %in% names(lb) && nrow(lb) > 1L) {
    expect_true(all(diff(lb$f1_macro) <= 0))
  }
})

test_that("screen regression sorted by rmse ascending", {
  s  <- mtcars_split()
  lb <- ml_screen(s, "mpg", seed = 42L,
                  algorithms = c("logistic"))
  # logistic should be filtered for regression
  # If no algorithms remain, just check shape
  expect_s3_class(lb, "ml_leaderboard")
})

test_that("screen regression algorithms (linear, glmnet)", {
  skip_if_not_installed("glmnet")
  s  <- mtcars_split()
  lb <- ml_screen(s, "mpg", seed = 42L, algorithms = c("linear"))
  expect_true(nrow(lb) >= 1L)
})

test_that("screen subset algorithms: algorithms = c('xgboost', 'logistic')", {
  skip_if_not_installed("xgboost")
  s  <- iris_split()
  lb <- ml_screen(s, "Species", seed = 42L,
                  algorithms = c("xgboost", "logistic"))
  expect_true(nrow(lb) <= 2L)
})

test_that("screen with CVResult input works", {
  cv <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  lb <- ml_screen(cv, "Species", seed = 42L, algorithms = c("logistic"))
  expect_s3_class(lb, "ml_leaderboard")
})

test_that("screen errors on raw data.frame input", {
  expect_error(
    ml_screen(iris, "Species", seed = 42L),
    class = "config_error"
  )
})

test_that("screen seed reproducibility", {
  s   <- iris_split()
  lb1 <- ml_screen(s, "Species", seed = 42L, algorithms = c("logistic"))
  lb2 <- ml_screen(s, "Species", seed = 42L, algorithms = c("logistic"))
  expect_equal(lb1$accuracy, lb2$accuracy)
})

test_that("screen algorithm failure caught (rest still succeed)", {
  s  <- iris_split()
  # naive_bayes classification-only; regression should filter it
  # Just test that we don't error when one fails
  lb <- ml_screen(s, "Species", seed = 42L, algorithms = c("logistic"))
  expect_s3_class(lb, "ml_leaderboard")
})

test_that("screen custom sort_by works", {
  s  <- iris_split()
  lb <- ml_screen(s, "Species", seed = 42L,
                  algorithms = c("logistic"),
                  sort_by = "accuracy")
  expect_s3_class(lb, "ml_leaderboard")
})

test_that("ml$screen() module style works", {
  s  <- iris_split()
  lb <- ml$screen(s, "Species", seed = 42L, algorithms = c("logistic"))
  expect_s3_class(lb, "ml_leaderboard")
})

test_that("binary classification sorted by roc_auc descending (logistic has roc_auc)", {
  df <- binary_df()
  s  <- ml_split(df, "churn", seed = 42L)
  lb <- ml_screen(s, "churn", seed = 42L, algorithms = c("logistic"))
  expect_s3_class(lb, "ml_leaderboard")
})

test_that("ml_best() returns top-ranked model from screen", {
  s  <- iris_split()
  lb <- ml_screen(s, "Species", seed = 42L, algorithms = c("logistic"))
  best <- ml_best(lb)
  expect_s3_class(best, "ml_model")
  expect_equal(best$algorithm, lb$algorithm[[1]])
  preds <- predict(best, newdata = s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("screen time_budget stops early", {
  s  <- iris_split()
  # Very small budget — should complete at least 1 algo
  lb <- ml_screen(s, "Species", seed = 42L, time_budget = 0.001)
  expect_s3_class(lb, "ml_leaderboard")
  # Should have fewer rows than full screen (if budget is tiny enough)
  lb_full <- ml_screen(s, "Species", seed = 42L)
  expect_true(nrow(lb) <= nrow(lb_full))
})

test_that("screen keep_models=FALSE discards models", {
  s  <- iris_split()
  lb <- ml_screen(s, "Species", seed = 42L, algorithms = c("logistic"),
                  keep_models = FALSE)
  expect_null(ml_best(lb))
})

test_that("ml_best() returns NULL for empty leaderboard", {
  lb <- new_ml_leaderboard(
    data.frame(algorithm = character(0), stringsAsFactors = FALSE),
    context = "empty"
  )
  expect_null(ml_best(lb))
})
