test_that("task detection: factor target → classification", {
  expect_equal(.detect_task(factor(c("a", "b", "a"))), "classification")
})

test_that("task detection: character target → classification", {
  expect_equal(.detect_task(c("yes", "no", "yes")), "classification")
})

test_that("task detection: many unique numerics → regression", {
  expect_equal(.detect_task(stats::rnorm(200)), "regression")
})

test_that("task detection: few unique numerics → classification", {
  expect_equal(.detect_task(rep(1:3, 100)), "classification")
})

test_that("task detection: override 'regression' works", {
  expect_equal(.detect_task(rep(1:3, 100), task = "regression"), "regression")
})

test_that("task detection: invalid task raises config_error", {
  expect_error(.detect_task(c("a","b"), task = "magic"), class = "config_error")
})

test_that("coerce tibble to data.frame", {
  skip_if_not_installed("tibble")
  tbl  <- tibble::as_tibble(iris)
  df   <- .coerce_data(tbl)
  expect_true(is.data.frame(df))
  expect_false(inherits(df, "tbl_df"))
})

test_that("prepare: ordinal encoding for tree algorithm", {
  X    <- data.frame(color = c("red", "blue", "red"), stringsAsFactors = FALSE)
  y    <- c(1L, 0L, 1L)
  norm <- .prepare(X, y, algorithm = "xgboost", task = "classification")
  expect_true(length(norm$category_maps) > 0L)
  expect_false(norm$use_onehot)
})

test_that("prepare: one-hot encoding for linear algorithm", {
  X    <- data.frame(color = c("red", "blue", "red"), stringsAsFactors = FALSE)
  y    <- c(1L, 0L, 1L)
  norm <- .prepare(X, y, algorithm = "logistic", task = "classification")
  expect_true(norm$use_onehot)
  expect_true("color" %in% norm$onehot_cols)
})

test_that("transform_fit: output is numeric data.frame", {
  X    <- data.frame(color = c("red", "blue", "red"), x = c(1, 2, 3),
                     stringsAsFactors = FALSE)
  y    <- c(1L, 0L, 1L)
  norm <- .prepare(X, y, algorithm = "logistic", task = "classification")
  res  <- .transform_fit(X, norm)
  expect_true(all(vapply(res$X, is.numeric, logical(1L))))
})

test_that("encode_target: string labels → 0-based integers", {
  y    <- c("yes", "no", "yes")
  norm <- .prepare(data.frame(x = 1:3), y, algorithm = "xgboost", task = "classification")
  enc  <- .encode_target(y, norm)
  expect_true(all(enc %in% 0:1))
})

test_that("decode: 0-based integers → original labels", {
  y    <- c("yes", "no", "yes")
  norm <- .prepare(data.frame(x = 1:3), y, algorithm = "xgboost", task = "classification")
  enc  <- .encode_target(y, norm)
  dec  <- .decode(enc, norm)
  expect_equal(as.character(dec), y)
})

test_that("non-tree algorithm: NA imputed with median", {
  X <- data.frame(x = c(1, 2, NA, 4, 5))
  y <- c(1L, 0L, 1L, 0L, 1L)
  norm <- .prepare(X, y, algorithm = "logistic", task = "classification")
  # Expects a warning about NA imputation
  suppressWarnings(res <- .transform_fit(X, norm))
  # NA should be gone after imputation
  expect_false(any(is.na(res$X)))
})

test_that("check_duplicate_cols: raises data_error on duplicates", {
  df <- data.frame(a = 1:3, b = 1:3)
  names(df) <- c("a", "a")
  expect_error(.check_duplicate_cols(df), class = "data_error")
})
