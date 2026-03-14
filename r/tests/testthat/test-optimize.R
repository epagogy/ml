binary_clf <- function() {
  withr::local_seed(42L)
  n <- 200L
  data.frame(
    f1     = stats::rnorm(n),
    f2     = stats::rnorm(n),
    target = sample(c(0L, 1L), n, replace = TRUE)
  )
}

test_that("ml_optimize returns ml_optimize_result for binary clf", {
  df    <- binary_clf()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 42L)
  opt   <- ml_optimize(model, data = s$valid, metric = "f1")
  expect_s3_class(opt, "ml_optimize_result")
  expect_s3_class(opt, "ml_model")
  expect_type(opt$threshold, "double")
  expect_gte(opt$threshold, 0.0)
  expect_lte(opt$threshold, 1.0)
})

test_that("ml_optimize threshold is in (0, 1)", {
  df    <- binary_clf()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 42L)
  opt   <- ml_optimize(model, data = s$valid)
  expect_gt(opt$threshold, 0.0)
  expect_lt(opt$threshold, 1.0)
})

test_that("ml_optimize rejects regression model", {
  withr::local_seed(42L)
  n  <- 200L
  df <- data.frame(f1 = stats::rnorm(n), target = stats::rnorm(n))
  s  <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", seed = 42L)
  expect_error(
    ml_optimize(model, data = s$valid),
    class = "model_error"
  )
})

test_that("ml_optimize rejects ranking metric roc_auc", {
  df    <- binary_clf()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 42L)
  expect_error(
    ml_optimize(model, data = s$valid, metric = "roc_auc"),
    class = "config_error"
  )
})

test_that("ml_optimize rejects non-model input", {
  df <- binary_clf()
  expect_error(
    ml_optimize(df, data = df),
    class = "model_error"
  )
})

test_that("ml_optimize leaves original model unchanged", {
  df    <- binary_clf()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 42L)
  opt   <- ml_optimize(model, data = s$valid)
  expect_null(model[["threshold"]])
  expect_false(is.null(opt[["threshold"]]))
})
