test_that("predict returns vector of correct length", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  preds <- predict(model, newdata = s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("predict returns class labels (not integers) for classification", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  preds <- predict(model, newdata = s$valid)
  expect_true(all(preds %in% levels(iris$Species)))
})

test_that("predict returns numeric for regression", {
  skip("logistic is clf-only — use linear for regression; tested separately")
})

test_that("predict returns numeric for regression (linear model)", {
  skip_if_not_installed("glmnet")
  s     <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "linear", seed = 42L)
  preds <- predict(model, newdata = s$valid)
  expect_true(is.numeric(preds))
})

test_that("target column auto-dropped from newdata", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  # Pass valid data WITH target column — should not error
  preds <- predict(model, newdata = s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("predict error on missing features", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  bad_data <- s$valid[, c("Sepal.Length", "Sepal.Width"), drop = FALSE]
  expect_error(predict(model, newdata = bad_data), class = "data_error")
})

test_that("predict with proba=TRUE returns data.frame (classification)", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  probs <- predict(model, newdata = s$valid, proba = TRUE)
  expect_true(is.data.frame(probs))
  expect_equal(nrow(probs), nrow(s$valid))
})

test_that("ml_predict_proba returns data.frame with correct columns", {
  skip_if_not_installed("ranger")
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42L)
  probs <- ml_predict_proba(model, s$valid)
  expect_true(is.data.frame(probs))
  expect_equal(ncol(probs), 3L)  # 3 species
  expect_equal(nrow(probs), nrow(s$valid))
})

test_that("ml_predict_proba probabilities sum to ~1.0 per row", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  probs <- ml_predict_proba(model, s$valid)
  row_sums <- rowSums(as.matrix(probs))
  expect_true(all(abs(row_sums - 1.0) < 1e-6))
})

test_that("ml_predict_proba errors on regression model", {
  skip_if_not_installed("glmnet")
  s     <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "linear", seed = 42L)
  expect_error(ml_predict_proba(model, s$valid), class = "model_error")
})

test_that("predict with extra features warns but succeeds", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  extra <- s$valid
  extra$EXTRA_COL <- 1L
  expect_warning(
    preds <- predict(model, newdata = extra),
    regexp = "Extra features"
  )
  expect_equal(length(preds), nrow(s$valid))
})

test_that("ml$predict_proba() module style works", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  probs <- ml$predict_proba(model, s$valid)
  expect_true(is.data.frame(probs))
})
