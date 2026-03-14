test_that("ml_enough returns ml_enough_result for well-populated classification", {
  s      <- iris_split()
  result <- ml_enough(s, "Species", seed = 42L)
  expect_s3_class(result, "ml_enough_result")
  expect_true(is.logical(result$saturated))
  expect_equal(result$metric, "accuracy")
  expect_type(result$recommendation, "character")
  expect_true(nchar(result$recommendation) > 0L)
})

test_that("ml_enough curve is a data.frame with correct columns", {
  s      <- iris_split()
  result <- ml_enough(s, "Species", seed = 42L, steps = 4L)
  expect_s3_class(result$curve, "data.frame")
  expect_true(all(c("n_samples", "train_score", "val_score") %in% names(result$curve)))
  expect_gte(nrow(result$curve), 1L)
})

test_that("ml_enough returns ml_enough_result for well-populated regression", {
  withr::local_seed(42L)
  df     <- data.frame(x = stats::rnorm(200L), y = stats::rnorm(200L))
  s      <- ml_split(df, "y", seed = 42L)
  result <- ml_enough(s, "y", seed = 42L)
  expect_s3_class(result, "ml_enough_result")
  expect_equal(result$metric, "r2")
  expect_type(result$recommendation, "character")
})

test_that("ml_enough raises data_error for tiny dataset (< 50 rows)", {
  df <- tiny_clf()
  s  <- ml_split(df, "target", seed = 42L)
  expect_error(ml_enough(s, "target"), class = "data_error")
})

test_that("ml_enough raises data_error for tiny regression dataset", {
  df <- tiny_reg()
  s  <- ml_split(df, "y", seed = 42L)
  expect_error(ml_enough(s, "y"), class = "data_error")
})

test_that("ml_enough raises config_error for steps < 2", {
  withr::local_seed(42L)
  df <- data.frame(x = stats::rnorm(200L), y = sample(c(0L, 1L), 200L, replace = TRUE))
  s  <- ml_split(df, "y", seed = 42L)
  expect_error(ml_enough(s, "y", seed = 42L, steps = 1L), class = "config_error")
})

test_that("ml_enough raises config_error for cv < 2", {
  withr::local_seed(42L)
  df <- data.frame(x = stats::rnorm(200L), y = sample(c(0L, 1L), 200L, replace = TRUE))
  s  <- ml_split(df, "y", seed = 42L)
  expect_error(ml_enough(s, "y", seed = 42L, cv = 1L), class = "config_error")
})

test_that("ml_enough n_current equals training rows", {
  s      <- iris_split()
  result <- ml_enough(s, "Species", seed = 42L)
  expect_equal(result$n_current, nrow(s$train))
})
