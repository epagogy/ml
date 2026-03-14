test_that("ml_leak returns clean for iris", {
  report <- ml_leak(iris, "Species")
  expect_s3_class(report, "ml_leak_report")
  expect_true(is.logical(report$clean))
  expect_true(length(report$checks) >= 3L)
})

test_that("ml_leak works with SplitResult", {
  s <- iris_split()
  report <- ml_leak(s, "Species")
  expect_s3_class(report, "ml_leak_report")
  # Should have duplicate check since we passed a split
  check_names <- vapply(report$checks, `[[`, character(1), "name")
  expect_true("Duplicate rows (train/test)" %in% check_names)
})

test_that("ml_leak detects high-correlation feature", {
  set.seed(42)
  n <- 100
  y <- rnorm(n)
  df <- data.frame(
    leaked = y + rnorm(n, sd = 0.01),  # near-perfect correlation
    noise = rnorm(n),
    target = y
  )
  report <- ml_leak(df, "target")
  expect_false(report$clean)
  expect_true(length(report$suspects) > 0L)
})

test_that("ml_leak detects ID column", {
  df <- iris
  df$row_id <- seq_len(nrow(df))
  report <- ml_leak(df, "Species")
  suspect_features <- vapply(report$suspects, `[[`, character(1), "feature")
  expect_true("row_id" %in% suspect_features)
})

test_that("ml_leak detects target name in feature", {
  df <- data.frame(
    churn_count = rnorm(50),
    age = rnorm(50),
    churn = sample(c("yes", "no"), 50, replace = TRUE)
  )
  report <- ml_leak(df, "churn")
  suspect_features <- vapply(report$suspects, `[[`, character(1), "feature")
  expect_true("churn_count" %in% suspect_features)
})

test_that("ml_leak detects leakage name pattern", {
  df <- data.frame(
    future_sales = rnorm(50),
    age = rnorm(50),
    revenue = rnorm(50)
  )
  report <- ml_leak(df, "revenue")
  suspect_features <- vapply(report$suspects, `[[`, character(1), "feature")
  expect_true("future_sales" %in% suspect_features)
})

test_that("ml_leak errors on missing target", {
  expect_error(
    ml_leak(iris, "nonexistent"),
    class = "data_error"
  )
})

test_that("ml_leak print method works", {
  report <- ml_leak(iris, "Species")
  expect_output(print(report), "Leak Report")
})

test_that("ml_leak skips correlation for multiclass", {
  report <- ml_leak(iris, "Species")
  corr_check <- report$checks[[1]]
  expect_true(grepl("skipped", corr_check$detail))
})

test_that("ml_leak binary classification runs correlation", {
  df <- binary_df()
  report <- ml_leak(df, "churn")
  corr_check <- report$checks[[1]]
  expect_true(grepl("max |r|", corr_check$detail) || corr_check$detail == "no numeric features")
})
