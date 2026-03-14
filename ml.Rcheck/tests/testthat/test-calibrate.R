test_that("ml_calibrate returns ml_calibrated_model for binary classification", {
  df    <- binary_df()
  s     <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "logistic", seed = 42L)
  cal   <- ml_calibrate(model, data = s$valid)
  expect_s3_class(cal, "ml_calibrated_model")
  expect_s3_class(cal, "ml_model")
})

test_that("predict.ml_calibrated_model returns probability data.frame when proba=TRUE", {
  df    <- binary_df()
  s     <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "logistic", seed = 42L)
  cal   <- ml_calibrate(model, data = s$valid)
  proba <- predict(cal, newdata = s$valid, proba = TRUE)
  expect_true(is.data.frame(proba))
  expect_equal(ncol(proba), 2L)
  expect_true(all(proba >= 0 & proba <= 1))
})

test_that("ml_evaluate on calibrated model returns ml_metrics", {
  df    <- binary_df()
  s     <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "logistic", seed = 42L)
  cal   <- ml_calibrate(model, data = s$valid)
  m     <- ml_evaluate(cal, s$valid)
  expect_s3_class(m, "ml_metrics")
  expect_true("accuracy" %in% names(m))
  expect_true("roc_auc" %in% names(m))
})

test_that("ml_calibrate errors on multiclass classification", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_error(ml_calibrate(model, data = s$valid), class = "config_error")
})

test_that("ml_calibrate errors on regression model", {
  s     <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "linear", seed = 42L)
  expect_error(ml_calibrate(model, data = s$valid), class = "config_error")
})

test_that("print.ml_calibrated_model runs without error", {
  df    <- binary_df()
  s     <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "logistic", seed = 42L)
  cal   <- ml_calibrate(model, data = s$valid)
  expect_no_error(print(cal))
})
