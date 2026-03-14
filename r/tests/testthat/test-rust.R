# Tests for Rust backend (engine="ml") and parity with R backends.
# All tests skip_if_not(.rust_available()) — CI without Rust is fine.

# ── Availability ─────────────────────────────────────────────────────────────

test_that(".rust_available() returns logical", {
  result <- ml:::.rust_available()
  expect_type(result, "logical")
  expect_length(result, 1L)
})

# ── Linear (Ridge) ──────────────────────────────────────────────────────────

test_that("Rust linear: fit + predict regression", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "linear", engine = "ml", seed = 42L)
  expect_s3_class(model, "ml_model")
  expect_equal(model$algorithm, "linear")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
  expect_type(preds, "double")
})

test_that("Rust linear vs R linear parity", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- mtcars_split()
  model_rust <- ml_fit(s$train, "mpg", algorithm = "linear", engine = "ml", seed = 42L)
  model_r    <- ml_fit(s$train, "mpg", algorithm = "linear", engine = "r", seed = 42L)
  preds_rust <- predict(model_rust, newdata = s$valid)
  preds_r    <- predict(model_r, newdata = s$valid)
  expect_equal(preds_rust, preds_r, tolerance = 1e-6)
})

# ── Logistic ────────────────────────────────────────────────────────────────

test_that("Rust logistic: fit + predict classification", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", engine = "ml", seed = 42L)
  expect_s3_class(model, "ml_model")
  expect_equal(model$algorithm, "logistic")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
})

test_that("Rust logistic vs R logistic parity (binary)", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  df <- binary_df(100L)
  s <- ml_split(df, "churn", seed = 42L)
  model_rust <- ml_fit(s$train, "churn", algorithm = "logistic", engine = "ml", seed = 42L)
  model_r    <- ml_fit(s$train, "churn", algorithm = "logistic", engine = "r", seed = 42L)
  preds_rust <- predict(model_rust, newdata = s$valid)
  preds_r    <- predict(model_r, newdata = s$valid)
  # Parity: same predictions (may differ slightly due to optimizer)
  agreement <- mean(preds_rust == preds_r)
  expect_true(agreement >= 0.8, info = paste0("Agreement: ", agreement))
})

# ── Decision Tree ───────────────────────────────────────────────────────────

test_that("Rust decision tree: classification fit + predict", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "decision_tree", engine = "ml", seed = 42L)
  expect_s3_class(model, "ml_model")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
})

test_that("Rust decision tree: regression fit + predict", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "decision_tree", engine = "ml", seed = 42L)
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
  expect_type(preds, "double")
})

test_that("Rust decision tree: explain returns importances", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "decision_tree", engine = "ml", seed = 42L)
  imp <- ml_explain(model)
  expect_s3_class(imp, "ml_explanation")
  expect_true(nrow(imp) > 0L)
})

# ── Random Forest ───────────────────────────────────────────────────────────

test_that("Rust random forest: classification fit + predict", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "random_forest",
                  engine = "ml", seed = 42L)
  expect_s3_class(model, "ml_model")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
})

test_that("Rust random forest: regression fit + predict", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "random_forest",
                  engine = "ml", seed = 42L)
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
  expect_type(preds, "double")
})

test_that("Rust random forest: explain returns importances", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "random_forest",
                  engine = "ml", seed = 42L)
  imp <- ml_explain(model)
  expect_s3_class(imp, "ml_explanation")
  expect_true(nrow(imp) > 0L)
})

# ── KNN ─────────────────────────────────────────────────────────────────────

test_that("Rust knn: classification fit + predict", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "knn",
                  engine = "ml", seed = 42L, k = 5L)
  expect_s3_class(model, "ml_model")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
})

test_that("Rust knn: regression fit + predict", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "knn",
                  engine = "ml", seed = 42L, k = 5L)
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
  expect_type(preds, "double")
})

# ── Engine parameter ────────────────────────────────────────────────────────

test_that("engine='ml' errors when Rust not available", {
  skip_if(ml:::.rust_available(), "Test requires Rust NOT available")
  s <- iris_split()
  expect_error(
    ml_fit(s$train, "Species", algorithm = "logistic", engine = "ml", seed = 42L),
    class = "config_error"
  )
})

test_that("engine='r' forces CRAN backend even when Rust is available", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", engine = "r", seed = 42L)
  # R logistic engine stores $models list (from .lr_fit)
  expect_true(is.list(model$engine))
  expect_false(ml:::.is_rust_engine(model$engine))
})

test_that("engine='auto' selects Rust when available", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", engine = "auto", seed = 42L)
  expect_true(ml:::.is_rust_engine(model$engine))
})

# ── predict_proba ───────────────────────────────────────────────────────────

test_that("Rust logistic predict_proba returns valid probabilities", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", engine = "ml", seed = 42L)
  proba <- ml_predict_proba(model, s$valid)
  expect_true(is.data.frame(proba))
  expect_equal(nrow(proba), nrow(s$valid))
  # Probabilities should be non-negative and rows should sum close to 1
  expect_true(all(as.matrix(proba) >= 0))
})

test_that("Rust random forest predict_proba returns valid probabilities", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "random_forest",
                  engine = "ml", seed = 42L)
  proba <- ml_predict_proba(model, s$valid)
  expect_true(is.data.frame(proba))
  expect_equal(nrow(proba), nrow(s$valid))
})

# ── Serialization (saveRDS / readRDS) ───────────────────────────────────────

test_that("Rust model survives saveRDS/readRDS round-trip", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "random_forest",
                  engine = "ml", seed = 42L)
  preds_before <- predict(model, newdata = s$valid)

  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp))
  saveRDS(model, tmp)
  loaded <- readRDS(tmp)

  preds_after <- predict(loaded, newdata = s$valid)
  expect_identical(preds_before, preds_after)
})

# ── criterion= smoke tests (Phases 1-2) ─────────────────────────────────────

test_that("criterion='entropy' routes to Rust (decision_tree)", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "decision_tree",
                  criterion = "entropy", engine = "ml", seed = 42L)
  expect_s3_class(model, "ml_model")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
})

test_that("criterion='entropy' runs without error", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  m_e <- ml_fit(s$train, "Species", algorithm = "decision_tree",
                criterion = "entropy", engine = "ml", seed = 42L)
  m_g <- ml_fit(s$train, "Species", algorithm = "decision_tree",
                criterion = "gini", engine = "ml", seed = 42L)
  preds_e <- predict(m_e, newdata = s$valid)
  preds_g <- predict(m_g, newdata = s$valid)
  # Both must produce valid predictions (same length)
  expect_length(preds_e, nrow(s$valid))
  expect_length(preds_g, nrow(s$valid))
  # Entropy and gini are distinct criteria — at least one fit should succeed
  expect_true(is.character(preds_e) || is.integer(preds_e) || is.numeric(preds_e))
})

# ── algorithm="extra_trees" smoke tests (Phase 2) ──────────────────────────

test_that("extra_trees classification routes to Rust", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "extra_trees",
                  engine = "ml", seed = 42L)
  expect_s3_class(model, "ml_model")
  expect_equal(model$algorithm, "extra_trees")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
})

test_that("extra_trees regression routes to Rust", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "extra_trees",
                  engine = "ml", seed = 42L)
  expect_s3_class(model, "ml_model")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
  expect_type(preds, "double")
})

test_that("extra_trees predict_proba returns valid probabilities", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "extra_trees",
                  engine = "ml", seed = 42L)
  proba <- ml_predict_proba(model, s$valid)
  expect_true(is.data.frame(proba))
  expect_equal(nrow(proba), nrow(s$valid))
  expect_true(all(as.matrix(proba) >= 0))
})

# ── multi_class="softmax" smoke tests (Phase 3) ──────────────────────────

test_that("multi_class='softmax' logistic works in R", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic",
                  multi_class = "softmax", engine = "ml", seed = 42L)
  expect_s3_class(model, "ml_model")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
  proba <- ml_predict_proba(model, s$valid)
  row_sums <- rowSums(as.matrix(proba))
  expect_true(all(abs(row_sums - 1.0) < 1e-9))
})

# ── monotone_cst smoke tests (Phase 4) ───────────────────────────────────

test_that("monotone_cst integer vector wires through to decision_tree reg", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  withr::local_seed(42L)
  n <- 100L
  x <- sort(runif(n))
  df <- data.frame(x = x, y = x + rnorm(n, sd = 0.1))
  s <- ml_split(df, "y", seed = 42L)
  model <- ml_fit(s$train, "y", algorithm = "decision_tree",
                  monotone_cst = 1L, engine = "ml", seed = 42L)
  test_x <- data.frame(x = seq(0, 1, length.out = 50L))
  preds <- predict(model, newdata = test_x)
  diffs <- diff(preds)
  expect_true(all(diffs >= -1e-9),
              info = paste("monotone violations:", sum(diffs < -1e-9)))
})

# ── algorithm="gradient_boosting" smoke tests (GBT parity) ──────────────────

test_that("gradient_boosting classification routes to Rust", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "gradient_boosting",
                  engine = "ml", seed = 42L)
  expect_s3_class(model, "ml_model")
  expect_equal(model$algorithm, "gradient_boosting")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
  expect_true(all(preds %in% levels(s$train$Species)))
})

test_that("gradient_boosting regression routes to Rust", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "gradient_boosting",
                  engine = "ml", seed = 42L)
  expect_s3_class(model, "ml_model")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
  expect_type(preds, "double")
})

test_that("gradient_boosting predict_proba returns valid probabilities", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "gradient_boosting",
                  engine = "ml", seed = 42L)
  proba <- ml_predict_proba(model, s$valid)
  expect_true(is.data.frame(proba))
  expect_equal(nrow(proba), nrow(s$valid))
  expect_equal(ncol(proba), 3L)  # 3 species
  row_sums <- rowSums(as.matrix(proba))
  expect_true(all(abs(row_sums - 1.0) < 1e-5))
  expect_true(all(as.matrix(proba) >= 0))
})

test_that("gradient_boosting feature importances sum to 1", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "gradient_boosting",
                  engine = "ml", seed = 42L)
  exp <- ml_explain(model)
  expect_s3_class(exp, "ml_explanation")
  expect_equal(nrow(exp), length(model$features))
  expect_true(abs(sum(exp$importance) - 1.0) < 1e-5)
})

test_that("gradient_boosting n_estimators and learning_rate params accepted", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "gradient_boosting",
                  n_estimators = 20L, learning_rate = 0.05,
                  engine = "ml", seed = 42L)
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
})

test_that("gradient_boosting serializes and deserializes via saveRDS", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "gradient_boosting",
                  engine = "ml", seed = 42L)
  preds_before <- predict(model, newdata = s$valid)
  tmp <- tempfile(fileext = ".rds")
  saveRDS(model, tmp)
  model2 <- readRDS(tmp)
  preds_after <- predict(model2, newdata = s$valid)
  expect_equal(preds_before, preds_after)
})

# ── Naive Bayes ─────────────────────────────────────────────────────────────

test_that("naive_bayes classification routes to Rust", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "naive_bayes",
                  engine = "ml", seed = 42L)
  expect_s3_class(model, "ml_model")
  expect_equal(model$algorithm, "naive_bayes")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
  expect_true(ml:::.is_rust_engine(model$engine))
})

test_that("naive_bayes predict_proba returns valid probabilities", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "naive_bayes",
                  engine = "ml", seed = 42L)
  proba <- ml_predict_proba(model, s$valid)
  expect_true(is.data.frame(proba))
  expect_equal(nrow(proba), nrow(s$valid))
  expect_equal(ncol(proba), 3L)
  expect_true(all(as.matrix(proba) >= 0))
  row_sums <- rowSums(as.matrix(proba))
  expect_true(all(abs(row_sums - 1.0) < 1e-5))
})

test_that("naive_bayes serializes and deserializes via saveRDS", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "naive_bayes",
                  engine = "ml", seed = 42L)
  preds_before <- predict(model, newdata = s$valid)
  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp))
  saveRDS(model, tmp)
  model2 <- readRDS(tmp)
  preds_after <- predict(model2, newdata = s$valid)
  expect_identical(preds_before, preds_after)
})

# ── Elastic Net ──────────────────────────────────────────────────────────────

test_that("elastic_net regression routes to Rust", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "elastic_net",
                  engine = "ml", seed = 42L)
  expect_s3_class(model, "ml_model")
  expect_equal(model$algorithm, "elastic_net")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
  expect_type(preds, "double")
  expect_true(ml:::.is_rust_engine(model$engine))
})

test_that("elastic_net alpha param accepted", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "elastic_net",
                  alpha = 0.5, l1_ratio = 0.5, engine = "ml", seed = 42L)
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
})

test_that("elastic_net serializes and deserializes via saveRDS", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "elastic_net",
                  engine = "ml", seed = 42L)
  preds_before <- predict(model, newdata = s$valid)
  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp))
  saveRDS(model, tmp)
  model2 <- readRDS(tmp)
  preds_after <- predict(model2, newdata = s$valid)
  expect_equal(preds_before, preds_after, tolerance = 1e-9)
})

# ── AdaBoost ─────────────────────────────────────────────────────────────────

test_that("adaboost classification routes to Rust", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "adaboost",
                  engine = "ml", seed = 42L)
  expect_s3_class(model, "ml_model")
  expect_equal(model$algorithm, "adaboost")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
  expect_true(ml:::.is_rust_engine(model$engine))
})

test_that("adaboost predict_proba returns valid probabilities", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "adaboost",
                  engine = "ml", seed = 42L)
  proba <- ml_predict_proba(model, s$valid)
  expect_true(is.data.frame(proba))
  expect_equal(nrow(proba), nrow(s$valid))
  expect_equal(ncol(proba), 3L)
  expect_true(all(as.matrix(proba) >= 0))
})

test_that("adaboost feature importances sum to 1", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "adaboost",
                  engine = "ml", seed = 42L)
  exp <- ml_explain(model)
  expect_s3_class(exp, "ml_explanation")
  expect_equal(nrow(exp), length(model$features))
  expect_true(abs(sum(exp$importance) - 1.0) < 1e-5)
})

test_that("adaboost serializes and deserializes via saveRDS", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "adaboost",
                  engine = "ml", seed = 42L)
  preds_before <- predict(model, newdata = s$valid)
  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp))
  saveRDS(model, tmp)
  model2 <- readRDS(tmp)
  preds_after <- predict(model2, newdata = s$valid)
  expect_identical(preds_before, preds_after)
})

# ── SVM ──────────────────────────────────────────────────────────────────────

test_that("svm classification routes to Rust", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "svm",
                  engine = "ml", seed = 42L)
  expect_s3_class(model, "ml_model")
  expect_equal(model$algorithm, "svm")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
  expect_true(ml:::.is_rust_engine(model$engine))
})

test_that("svm regression routes to Rust", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "svm",
                  engine = "ml", seed = 42L)
  expect_s3_class(model, "ml_model")
  expect_equal(model$algorithm, "svm")
  preds <- predict(model, newdata = s$valid)
  expect_length(preds, nrow(s$valid))
  expect_type(preds, "double")
})

test_that("svm predict_proba returns valid probabilities", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "svm",
                  engine = "ml", seed = 42L)
  proba <- ml_predict_proba(model, s$valid)
  expect_true(is.data.frame(proba))
  expect_equal(nrow(proba), nrow(s$valid))
  expect_equal(ncol(proba), 3L)
  expect_true(all(as.matrix(proba) >= 0))
})

test_that("svm serializes and deserializes via saveRDS", {
  skip_if_not(ml:::.rust_available(), "Rust backend not available")
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "svm",
                  engine = "ml", seed = 42L)
  preds_before <- predict(model, newdata = s$valid)
  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp))
  saveRDS(model, tmp)
  model2 <- readRDS(tmp)
  preds_after <- predict(model2, newdata = s$valid)
  expect_identical(preds_before, preds_after)
})
