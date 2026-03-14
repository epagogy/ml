test_that("ml_check passes for deterministic algorithm", {
  result <- ml_check(iris, "Species", algorithm = "logistic", seed = 42L)
  expect_true(result$passed)
  expect_equal(result$algorithm, "logistic")
  expect_equal(result$seed, 42L)
})

test_that("ml_check passes for random_forest (default)", {
  s <- iris_split()
  result <- ml_check(iris, "Species", seed = 42L)
  expect_true(result$passed)
  expect_equal(result$algorithm, "random_forest")
})

test_that("ml_check_data returns clean report for iris", {
  report <- ml_check_data(iris, "Species")
  expect_s3_class(report, "ml_check_report")
  expect_true(report$passed)
  expect_equal(length(report$errors), 0L)
})

test_that("ml_check_data detects NA target", {
  df <- iris
  df$Species[1:5] <- NA
  report <- ml_check_data(df, "Species")
  expect_true(report$has_issues)
  expect_true(any(grepl("NA value", report$warnings)))
})

test_that("ml_check_data detects zero-variance column", {
  df <- iris
  df$const <- 1
  report <- ml_check_data(df, "Species")
  expect_true(any(grepl("zero variance", report$warnings)))
})

test_that("ml_check_data detects ID column", {
  df <- iris
  df$row_id <- seq_len(nrow(df))
  report <- ml_check_data(df, "Species")
  expect_true(any(grepl("ID column", report$warnings)))
})

test_that("ml_check_data detects high-null column", {
  df <- iris
  df$mostly_na <- NA_real_
  df$mostly_na[1:20] <- 1.0
  report <- ml_check_data(df, "Species")
  expect_true(any(grepl("missing values", report$warnings)))
})

test_that("ml_check_data detects Inf in features", {
  df <- iris
  df$Sepal.Length[1] <- Inf
  report <- ml_check_data(df, "Species")
  expect_true(any(grepl("infinite", report$warnings)))
})

test_that("ml_check_data detects class imbalance", {
  n <- 200L
  set.seed(42)
  df <- data.frame(
    x = rnorm(n),
    y = c(rep("rare", 5), rep("common", n - 5L))
  )
  report <- ml_check_data(df, "y")
  expect_true(any(grepl("imbalance", report$warnings)))
})

test_that("ml_check_data detects highly correlated features", {
  set.seed(42)
  x1 <- rnorm(100)
  df <- data.frame(
    x1 = x1,
    x2 = x1 + rnorm(100, sd = 0.01),
    y = sample(c("a", "b"), 100, replace = TRUE)
  )
  report <- ml_check_data(df, "y")
  expect_true(any(grepl("correlated", report$warnings)))
})

test_that("ml_check_data severity='error' raises on issues", {
  df <- iris
  df$const <- 1
  expect_error(
    ml_check_data(df, "Species", severity = "error"),
    class = "data_error"
  )
})

test_that("ml_check_data errors on missing target", {
  expect_error(
    ml_check_data(iris, "nonexistent"),
    class = "data_error"
  )
})

test_that("ml_check_data detects duplicate rows", {
  df <- iris[rep(1:10, each = 20), ]
  report <- ml_check_data(df, "Species")
  expect_true(any(grepl("duplicate", report$warnings)))
})

test_that("ml_check_data print method works", {
  report <- ml_check_data(iris, "Species")
  expect_output(print(report), "Data Check Report")
})
