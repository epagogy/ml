test_that("binary classification works — iris setosa vs versicolor > 0.80 acc", {
  data(iris)
  df  <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  df$Species <- droplevels(df$Species)
  X   <- as.matrix(df[, 1:4])
  y   <- as.integer(df$Species) - 1L
  fit <- .lr_fit(X, y, n_classes = 2L)
  pred <- .lr_predict(fit, X)
  acc  <- mean(pred == y)
  expect_gt(acc, 0.80)
})

test_that("multiclass works — full iris 3 classes > 0.70 acc", {
  data(iris)
  X   <- as.matrix(iris[, 1:4])
  y   <- as.integer(iris$Species) - 1L
  fit <- .lr_fit(X, y, n_classes = 3L)
  pred <- .lr_predict(fit, X)
  acc  <- mean(pred == y)
  expect_gt(acc, 0.70)
})

test_that("proba shape binary — ncol == 2", {
  data(iris)
  df  <- iris[iris$Species %in% c("setosa", "versicolor"), ]
  df$Species <- droplevels(df$Species)
  X   <- as.matrix(df[, 1:4])
  y   <- as.integer(df$Species) - 1L
  fit <- .lr_fit(X, y, n_classes = 2L)
  prob <- .lr_proba(fit, X)
  expect_equal(ncol(prob), 2L)
})

test_that("proba shape multiclass — ncol == 3", {
  data(iris)
  X   <- as.matrix(iris[, 1:4])
  y   <- as.integer(iris$Species) - 1L
  fit <- .lr_fit(X, y, n_classes = 3L)
  prob <- .lr_proba(fit, X)
  expect_equal(ncol(prob), 3L)
})

test_that("proba rows sum to 1", {
  data(iris)
  X   <- as.matrix(iris[, 1:4])
  y   <- as.integer(iris$Species) - 1L
  fit <- .lr_fit(X, y, n_classes = 3L)
  prob <- .lr_proba(fit, X)
  expect_true(all.equal(rowSums(prob), rep(1.0, nrow(prob)), tolerance = 1e-6))
})

test_that("ml_fit integration — algorithm='logistic' returns ml_model", {
  skip_if_not_installed("ml")
  s <- ml_split(iris, "Species", seed = 42L)
  m <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_s3_class(m, "ml_model")
  preds <- ml_predict(m, s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("regression target raises config error", {
  df <- iris
  df$target <- rnorm(nrow(df))
  expect_error(
    {
      s <- ml_split(df, "target", seed = 42L)
      ml_fit(s$train, "target", algorithm = "logistic", seed = 42L)
    },
    regexp = "classification"
  )
})

test_that("nnet not loaded after logistic fit", {
  # After fitting a logistic model, nnet should NOT be newly loaded
  before <- loadedNamespaces()
  s <- ml_split(iris, "Species", seed = 42L)
  ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  newly_loaded <- setdiff(loadedNamespaces(), before)
  expect_false("nnet" %in% newly_loaded)
})
