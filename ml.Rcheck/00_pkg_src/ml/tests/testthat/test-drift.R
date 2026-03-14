test_that("drift returns ml_drift_result (statistical)", {
  s      <- iris_split()
  result <- ml_drift(reference = s$train, new = s$test, target = "Species")
  expect_s3_class(result, "ml_drift_result")
})

test_that("drift detects obvious numeric shift", {
  s      <- iris_split()
  new    <- s$test
  new$Sepal.Length <- new$Sepal.Length + 5  # large shift
  result <- ml_drift(reference = s$train, new = new, target = "Species")
  expect_true("Sepal.Length" %in% result$features_shifted)
})

test_that("drift is stable on same distribution (approx)", {
  set.seed(42L)
  n  <- 200L
  df <- data.frame(x = stats::rnorm(n), y = stats::rnorm(n))
  s1 <- df[1:100, ]
  s2 <- df[101:200, ]
  result <- ml_drift(reference = s1, new = s2)
  # Same distribution — expect low severity
  expect_true(result$severity %in% c("none", "low"))
})

test_that("drift severity levels: none/low/medium/high", {
  s <- iris_split()
  result <- ml_drift(reference = s$train, new = s$test, target = "Species")
  expect_true(result$severity %in% c("none", "low", "medium", "high"))
})

test_that("drift $shifted is logical", {
  s      <- iris_split()
  result <- ml_drift(reference = s$train, new = s$test, target = "Species")
  expect_true(is.logical(result$shifted))
})

test_that("drift $features is a named numeric vector", {
  s      <- iris_split()
  result <- ml_drift(reference = s$train, new = s$test, target = "Species")
  expect_true(is.numeric(result$features))
  expect_true(!is.null(names(result$features)))
})

test_that("drift excludes target column from features", {
  s      <- iris_split()
  result <- ml_drift(reference = s$train, new = s$test, target = "Species")
  expect_false("Species" %in% names(result$features))
})

test_that("drift exclude= parameter works", {
  s      <- iris_split()
  result <- ml_drift(reference = s$train, new = s$test,
                     target = "Species", exclude = "Sepal.Length")
  expect_false("Sepal.Length" %in% names(result$features))
})

test_that("drift adversarial mode returns auc", {
  skip_if_not_installed("ranger")
  s      <- iris_split()
  result <- ml_drift(reference = s$train, new = s$test,
                     target = "Species", method = "adversarial", seed = 42L)
  expect_s3_class(result, "ml_drift_result")
  expect_false(is.null(result$auc))
  expect_true(result$auc >= 0 && result$auc <= 1)
})

test_that("drift adversarial mode returns train_scores for reference rows", {
  skip_if_not_installed("ranger")
  s      <- iris_split()
  result <- ml_drift(reference = s$train, new = s$test,
                     method = "adversarial", seed = 42L, target = "Species")
  expect_equal(length(result$train_scores), nrow(s$train))
})

test_that("drift error on unknown method", {
  s <- iris_split()
  expect_error(
    ml_drift(reference = s$train, new = s$test, method = "magic"),
    class = "config_error"
  )
})

test_that("drift error on too-small reference (<5 rows)", {
  tiny <- iris[1:3, ]
  s    <- iris_split()
  expect_error(
    ml_drift(reference = tiny, new = s$test),
    class = "data_error"
  )
})

test_that("drift print works", {
  s      <- iris_split()
  result <- ml_drift(reference = s$train, new = s$test, target = "Species")
  expect_output(print(result), "Drift")
})

test_that("ml$drift() module style works", {
  s      <- iris_split()
  result <- ml$drift(reference = s$train, new = s$test, target = "Species")
  expect_s3_class(result, "ml_drift_result")
})
