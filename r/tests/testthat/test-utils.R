test_that("ml_algorithms returns a data.frame", {
  df <- ml_algorithms()
  expect_true(is.data.frame(df))
  expect_true("algorithm" %in% names(df))
  expect_true("classification" %in% names(df))
  expect_true("regression" %in% names(df))
})

test_that("ml_algorithms classification filter works", {
  df <- ml_algorithms(task = "classification")
  expect_true(all(df$classification == TRUE))
})

test_that("ml_algorithms regression filter works", {
  df <- ml_algorithms(task = "regression")
  expect_true(all(df$regression == TRUE))
})

test_that("ml_dataset iris returns iris-like data.frame", {
  d <- ml_dataset("iris")
  expect_true(is.data.frame(d))
  expect_true(nrow(d) == 150L)
})

test_that("ml_dataset churn loads", {
  d <- ml_dataset("churn")
  expect_true(is.data.frame(d))
  expect_true(nrow(d) > 0L)
  expect_true("churn" %in% names(d))
})

test_that("ml_dataset fraud loads with ~2% fraud rate", {
  d <- ml_dataset("fraud")
  expect_true(is.data.frame(d))
  expect_equal(nrow(d), 10000L)
  fraud_rate <- mean(d$fraud == 1L)
  expect_true(fraud_rate > 0.005 && fraud_rate < 0.1)
})

test_that("ml_dataset unknown raises data_error", {
  expect_error(ml_dataset("nonexistent_dataset_xyz"), class = "data_error")
})
