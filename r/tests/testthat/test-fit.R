test_that("holdout fit returns ml_model with correct algorithm", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_s3_class(model, "ml_model")
  expect_equal(model$algorithm, "logistic")
})

test_that("CV fit returns ml_model with $scores_ populated", {
  cv    <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  expect_s3_class(model, "ml_model")
  expect_false(is.null(model$scores_))
  expect_true(length(model$scores_) > 0L)
})

test_that("auto algorithm selection: xgboost if available, else random_forest", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", seed = 42L)
  expect_true(model$algorithm %in% c("xgboost", "random_forest", "logistic"))
})

test_that("logistic regression: classification on iris", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  preds <- predict(model, newdata = s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("logistic regression: regression raises config_error", {
  s <- mtcars_split()
  expect_error(
    ml_fit(s$train, "mpg", algorithm = "logistic", seed = 42L),
    class = "config_error"
  )
})

test_that("xgboost classification on iris", {
  skip_if_not_installed("xgboost")
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "xgboost", seed = 42L)
  expect_equal(model$algorithm, "xgboost")
  preds <- predict(model, newdata = s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("xgboost regression on mtcars", {
  skip_if_not_installed("xgboost")
  s     <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "xgboost", seed = 42L)
  expect_equal(model$task, "regression")
  preds <- predict(model, newdata = s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("random_forest classification on iris", {
  skip_if_not_installed("ranger")
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42L)
  expect_equal(model$algorithm, "random_forest")
})

test_that("random_forest regression on mtcars", {
  skip_if_not_installed("ranger")
  s     <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "random_forest", seed = 42L)
  expect_equal(model$task, "regression")
})

test_that("svm classification on iris", {
  skip_if_not_installed("e1071")
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "svm", seed = 42L)
  expect_equal(model$algorithm, "svm")
})

test_that("knn classification on iris", {
  skip_if_not_installed("kknn")
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "knn", seed = 42L)
  expect_equal(model$algorithm, "knn")
})

test_that("naive_bayes classification on iris", {
  skip_if_not_installed("naivebayes")
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "naive_bayes", seed = 42L)
  expect_equal(model$algorithm, "naive_bayes")
})

test_that("glmnet linear regression on mtcars", {
  skip_if_not_installed("glmnet")
  s     <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "linear", seed = 42L)
  expect_equal(model$algorithm, "linear")
  expect_equal(model$task, "regression")
})

test_that("elastic_net regression on mtcars", {
  skip_if_not_installed("glmnet")
  s     <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "elastic_net", seed = 42L)
  expect_equal(model$algorithm, "elastic_net")
})

test_that("decision_tree classification on iris", {
  skip_if_not_installed("rpart")
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "decision_tree", seed = 42L)
  expect_equal(model$algorithm, "decision_tree")
  expect_equal(model$task, "classification")
  preds <- predict(model, newdata = s$valid)
  expect_true(all(preds %in% levels(iris$Species)))
})

test_that("decision_tree regression on mtcars", {
  skip_if_not_installed("rpart")
  s     <- mtcars_split()
  model <- ml_fit(s$train, "mpg", algorithm = "decision_tree", seed = 42L)
  expect_equal(model$algorithm, "decision_tree")
  expect_equal(model$task, "regression")
  preds <- predict(model, newdata = s$valid)
  expect_true(is.numeric(preds))
})

test_that("elastic_net binary classification", {
  skip_if_not_installed("glmnet")
  df    <- binary_df()
  s     <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "elastic_net", seed = 42L)
  expect_equal(model$algorithm, "elastic_net")
  expect_equal(model$task, "classification")
  preds <- predict(model, newdata = s$valid)
  expect_true(all(preds %in% c("yes", "no")))
})

test_that("elastic_net multiclass classification (iris)", {
  skip_if_not_installed("glmnet")
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "elastic_net", seed = 42L)
  expect_equal(model$algorithm, "elastic_net")
  expect_equal(model$task, "classification")
  preds <- predict(model, newdata = s$valid)
  expect_true(all(preds %in% levels(iris$Species)))
})

test_that("seed reproducibility: two fits with same seed produce same predictions", {
  s     <- iris_split()
  m1    <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 7L)
  m2    <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 7L)
  p1    <- predict(m1, newdata = s$valid)
  p2    <- predict(m2, newdata = s$valid)
  expect_identical(p1, p2)
})

test_that("string target auto-encoded (character → labels)", {
  df    <- binary_df()
  s     <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "logistic", seed = 42L)
  preds <- predict(model, newdata = s$valid)
  expect_true(all(preds %in% c("yes", "no")))
})

test_that("$time >= 0", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_true(model$time >= 0)
})

test_that("$preprocessing_ populated", {
  s     <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_false(is.null(model$preprocessing_))
})

test_that("error on missing target column", {
  s <- iris_split()
  expect_error(
    ml_fit(s$train, "nonexistent", seed = 42L),
    class = "data_error"
  )
})

test_that("error on single-class target", {
  df <- data.frame(x = stats::rnorm(20L), y = rep("a", 20L),
                   stringsAsFactors = FALSE)
  expect_error(
    ml_fit(df, "y", seed = 42L),
    class = "data_error"
  )
})

test_that("ml$fit() module style works", {
  s     <- iris_split()
  model <- ml$fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_s3_class(model, "ml_model")
})

# ── balance= parameter ──────────────────────────────────────────────────────

test_that("balance=TRUE works with logistic", {
  df <- binary_df()
  s  <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "logistic", seed = 42L,
                  balance = TRUE)
  expect_s3_class(model, "ml_model")
  preds <- predict(model, newdata = s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("balance=TRUE works with xgboost", {
  skip_if_not_installed("xgboost")
  df <- binary_df()
  s  <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "xgboost", seed = 42L,
                  balance = TRUE)
  expect_s3_class(model, "ml_model")
})

test_that("balance=TRUE works with random_forest", {
  skip_if_not_installed("ranger")
  df <- binary_df()
  s  <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "random_forest", seed = 42L,
                  balance = TRUE)
  expect_s3_class(model, "ml_model")
})

test_that("balance=TRUE works with decision_tree", {
  skip_if_not_installed("rpart")
  df <- binary_df()
  s  <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "decision_tree", seed = 42L,
                  balance = TRUE)
  expect_s3_class(model, "ml_model")
})

test_that("balance=TRUE works with svm", {
  skip_if_not_installed("e1071")
  df <- binary_df()
  s  <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "svm", seed = 42L,
                  balance = TRUE)
  expect_s3_class(model, "ml_model")
})

test_that("balance=TRUE works with elastic_net", {
  skip_if_not_installed("glmnet")
  df <- binary_df()
  s  <- ml_split(df, "churn", seed = 42L)
  model <- ml_fit(s$train, "churn", algorithm = "elastic_net", seed = 42L,
                  balance = TRUE)
  expect_s3_class(model, "ml_model")
})

test_that("balance=TRUE errors on regression", {
  s <- mtcars_split()
  expect_error(
    ml_fit(s$train, "mpg", algorithm = "linear", seed = 42L, balance = TRUE),
    class = "config_error"
  )
})

test_that("balance=TRUE works with multiclass", {
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L,
                  balance = TRUE)
  expect_s3_class(model, "ml_model")
  preds <- predict(model, newdata = s$valid)
  expect_true(all(preds %in% levels(iris$Species)))
})

# ── engine= parameter ─────────────────────────────────────────────────────

test_that("engine='r' accepted and produces valid model", {
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", engine = "r",
                  seed = 42L)
  expect_s3_class(model, "ml_model")
  preds <- predict(model, newdata = s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("engine='auto' accepted and produces valid model", {
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", engine = "auto",
                  seed = 42L)
  expect_s3_class(model, "ml_model")
})

# ── Partition guards ──

test_that("fit errors on test-tagged data", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s <- iris_split()
  expect_error(
    ml_fit(s$test, "Species", algorithm = "logistic", seed = 42L),
    class = "partition_error"
  )
})

test_that("fit accepts train-tagged data", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s <- iris_split()
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_s3_class(model, "ml_model")
})

test_that("fit accepts dev-tagged data", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  s <- iris_split()
  model <- ml_fit(s$dev, "Species", algorithm = "logistic", seed = 42L)
  expect_s3_class(model, "ml_model")
})

test_that("fit rejects untagged data in strict mode", {
  ml_config(guards = "strict")
  withr::defer(ml_config(guards = "off"))
  untagged <- iris
  expect_null(attr(untagged, "_ml_partition"))
  expect_error(
    ml_fit(untagged, "Species", algorithm = "logistic", seed = 42L),
    class = "partition_error"
  )
})

test_that("fit allows untagged data with guards off", {
  ml_config(guards = "off")
  untagged <- iris
  model <- ml_fit(untagged, "Species", algorithm = "logistic", seed = 42L)
  expect_s3_class(model, "ml_model")
})
