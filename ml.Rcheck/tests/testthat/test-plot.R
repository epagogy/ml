test_that("ml_plot importance does not error", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42L)
  tmp   <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp)
  on.exit({ grDevices::dev.off(); unlink(tmp) })
  expect_no_error(ml_plot(model, kind = "importance"))
})

test_that("ml_plot roc does not error for binary classification", {
  df    <- binary_df()
  s     <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "logistic", seed = 42L)
  tmp   <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp)
  on.exit({ grDevices::dev.off(); unlink(tmp) })
  expect_no_error(ml_plot(model, data = s$valid, kind = "roc"))
})

test_that("ml_plot confusion does not error for classification", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  tmp   <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp)
  on.exit({ grDevices::dev.off(); unlink(tmp) })
  expect_no_error(ml_plot(model, data = s$valid, kind = "confusion"))
})

test_that("ml_plot residual does not error for regression", {
  skip_if_not_installed("glmnet")
  s     <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "linear", seed = 42L)
  tmp   <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp)
  on.exit({ grDevices::dev.off(); unlink(tmp) })
  expect_no_error(ml_plot(model, data = s$valid, kind = "residual"))
})

test_that("ml_plot calibration does not error for binary classification", {
  df    <- binary_df()
  s     <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "logistic", seed = 42L)
  tmp   <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp)
  on.exit({ grDevices::dev.off(); unlink(tmp) })
  expect_no_error(ml_plot(model, data = s$valid, kind = "calibration"))
})
