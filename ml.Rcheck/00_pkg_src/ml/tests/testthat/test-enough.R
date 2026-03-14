test_that("ml_enough returns sufficient=TRUE for well-populated classification", {
  # iris: 50 samples per class, well above 30 threshold
  s      <- iris_split()
  result <- ml_enough(s, "Species")
  expect_s3_class(result, "ml_enough_result")
  expect_true(result$sufficient)
  expect_equal(result$task, "classification")
})

test_that("ml_enough returns sufficient=FALSE for tiny classification", {
  # tiny_clf: 30 rows total, ~15 per class — below 30 threshold
  df     <- tiny_clf()
  s      <- ml_split(df, "target", seed = 42L)
  result <- ml_enough(s, "target")
  expect_s3_class(result, "ml_enough_result")
  expect_false(result$sufficient)
})

test_that("ml_enough returns sufficient=TRUE for well-populated regression", {
  # mtcars: 32 rows, split gives ~24 train — just passes 50? Actually 24 < 50
  # Use a larger dataset for this test
  set.seed(42L)
  df     <- data.frame(x = stats::rnorm(200L), y = stats::rnorm(200L))
  s      <- ml_split(df, "y", seed = 42L)
  result <- ml_enough(s, "y")
  expect_s3_class(result, "ml_enough_result")
  expect_true(result$sufficient)
  expect_equal(result$task, "regression")
})

test_that("ml_enough returns sufficient=FALSE for tiny regression", {
  df     <- tiny_reg()
  s      <- ml_split(df, "y", seed = 42L)
  result <- ml_enough(s, "y")
  expect_s3_class(result, "ml_enough_result")
  expect_false(result$sufficient)
})
