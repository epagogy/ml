test_that("linear regression predicts with low error", {
  set.seed(42)
  n  <- 200
  X  <- matrix(rnorm(n * 5), n, 5)
  y  <- X %*% c(2, -1, 0.5, 0, 0) + rnorm(n) * 0.3
  fit  <- .lm_fit(X[1:150, ], y[1:150])
  pred <- .lm_predict(fit, X[151:200, ])
  rmse <- sqrt(mean((pred - y[151:200])^2))
  expect_lt(rmse, 0.6)
})

test_that("coef has length == n_features", {
  set.seed(1)
  X   <- matrix(rnorm(100 * 4), 100, 4)
  y   <- X[, 1] + rnorm(100) * 0.1
  fit <- .lm_fit(X, y)
  expect_equal(length(fit$coef), 4L)
})

test_that("higher alpha shrinks coefficients", {
  set.seed(2)
  X     <- matrix(rnorm(100 * 3), 100, 3)
  y     <- X %*% c(2, -1, 0.5) + rnorm(100) * 0.5
  fit_lo <- .lm_fit(X, y, alpha = 0.001)
  fit_hi <- .lm_fit(X, y, alpha = 100)
  expect_lt(sum(fit_hi$coef^2), sum(fit_lo$coef^2))
})

test_that("alpha=0 recovers OLS coefficients closely", {
  set.seed(3)
  X    <- matrix(rnorm(200 * 3), 200, 3)
  w    <- c(2.0, -1.0, 0.5)
  y    <- X %*% w + rnorm(200) * 0.05
  fit  <- .lm_fit(X, y, alpha = 0)
  expect_equal(fit$coef, w, tolerance = 0.05)
})

test_that("ml_fit integration — algorithm='linear' returns ml_model", {
  skip_if_not_installed("ml")
  df <- data.frame(
    x1 = rnorm(100), x2 = rnorm(100),
    y  = rnorm(100)
  )
  s <- ml_split(df, "y", seed = 42L)
  m <- ml_fit(s$train, "y", algorithm = "linear", seed = 42L)
  expect_s3_class(m, "ml_model")
  preds <- ml_predict(m, s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("classification target raises config error", {
  expect_error(
    {
      s <- ml_split(iris, "Species", seed = 42L)
      ml_fit(s$train, "Species", algorithm = "linear", seed = 42L)
    },
    regexp = "regression"
  )
})

test_that("glmnet not required for linear fit", {
  # linear should work even if glmnet is not loaded
  df <- data.frame(x = rnorm(100), y = rnorm(100))
  s  <- ml_split(df, "y", seed = 42L)
  # If this errors with a 'glmnet required' message, the test fails
  expect_no_error(ml_fit(s$train, "y", algorithm = "linear", seed = 42L))
})
