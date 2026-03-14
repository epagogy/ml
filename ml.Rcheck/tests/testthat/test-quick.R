test_that("ml_quick returns model, metrics, split", {
  result <- ml_quick(iris, "Species", seed = 42L)
  expect_type(result, "list")
  expect_s3_class(result$model, "ml_model")
  expect_s3_class(result$metrics, "ml_metrics")
  expect_s3_class(result$split, "ml_split_result")
})

test_that("ml_quick model can predict", {
  result <- ml_quick(iris, "Species", seed = 42L)
  preds <- predict(result$model, newdata = result$split$valid)
  expect_equal(length(preds), nrow(result$split$valid))
})

test_that("ml_quick works for regression", {
  result <- ml_quick(mtcars, "mpg", seed = 42L)
  expect_s3_class(result$model, "ml_model")
  expect_true("rmse" %in% names(result$metrics))
})
