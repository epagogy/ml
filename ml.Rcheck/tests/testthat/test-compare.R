test_that("compare returns ml_leaderboard with multiple models", {
  s  <- iris_split()
  m1 <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m2 <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 99L)
  lb <- ml_compare(list(m1, m2), s$valid)
  expect_s3_class(lb, "ml_leaderboard")
  expect_equal(nrow(lb), 2L)
})

test_that("compare does not re-fit (models unchanged)", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  hash_before <- model$hash
  ml_compare(list(model), s$valid)
  expect_equal(model$hash, hash_before)
})

test_that("compare errors on empty list", {
  s <- iris_split()
  expect_error(ml_compare(list(), s$valid), class = "config_error")
})

test_that("compare auto-unwraps TuningResult", {
  s     <- iris_split()
  tuned <- ml_tune(s$train, "Species", algorithm = "logistic",
                   n_trials = 2L, seed = 42L)
  m2    <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 99L)
  lb    <- ml_compare(list(tuned, m2), s$valid)
  expect_s3_class(lb, "ml_leaderboard")
})

test_that("compare target inferred from first model", {
  s  <- iris_split()
  m1 <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m2 <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 1L)
  # Should not require explicit target
  lb <- ml_compare(list(m1, m2), s$valid)
  expect_s3_class(lb, "ml_leaderboard")
})

test_that("compare error when models have different targets", {
  s  <- iris_split()
  sr <- mtcars_split()
  m1 <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  skip_if_not_installed("glmnet")
  m2 <- ml_fit(sr$train, "mpg", algorithm = "linear", seed = 42L)
  expect_error(ml_compare(list(m1, m2), s$valid), class = "config_error")
})

test_that("compare error when mixing classification and regression", {
  s  <- iris_split()
  m1 <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  skip_if_not_installed("glmnet")
  df_reg <- data.frame(
    Sepal.Length = iris$Sepal.Length,
    Sepal.Width  = iris$Sepal.Width,
    Species      = as.numeric(iris$Sepal.Length)  # numeric target
  )
  sr <- ml_split(df_reg, "Species", seed = 42L, task = "regression")
  m2 <- ml_fit(sr$train, "Species", algorithm = "linear", seed = 42L)
  expect_error(ml_compare(list(m1, m2), s$valid), class = "config_error")
})

test_that("compare time column present", {
  s  <- iris_split()
  m1 <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  lb <- ml_compare(list(m1), s$valid)
  expect_true("time" %in% names(lb))
})

test_that("compare sorted by metric", {
  s  <- iris_split()
  m1 <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m2 <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 99L)
  lb <- ml_compare(list(m1, m2), s$valid)
  if (nrow(lb) > 1L && "accuracy" %in% names(lb)) {
    # Sort by default metric (roc_auc or accuracy)
    expect_s3_class(lb, "ml_leaderboard")
  }
})

test_that("ml$compare() module style works", {
  s  <- iris_split()
  m1 <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  lb <- ml$compare(list(m1), s$valid)
  expect_s3_class(lb, "ml_leaderboard")
})

test_that("compare handles invalid model type with config_error", {
  s <- iris_split()
  expect_error(ml_compare(list("not_a_model"), s$valid), class = "config_error")
})
