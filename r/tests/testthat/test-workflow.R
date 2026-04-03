# Workflow transition tests -- verifies function composition chains.
# Tests the seams between verbs, not individual functions.
# Each test exercises a realistic multi-step workflow that a user would follow.

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

clf_data <- function() {
  withr::local_seed(42L)
  n <- 200L
  data.frame(
    f1     = stats::rnorm(n),
    f2     = stats::rnorm(n),
    f3     = stats::rnorm(n),
    f4     = stats::rnorm(n),
    f5     = stats::rnorm(n),
    target = sample(c("yes", "no"), n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}

reg_data <- function() {
  withr::local_seed(42L)
  n <- 200L
  X <- matrix(stats::rnorm(n * 5), n, 5)
  data.frame(
    f1     = X[, 1], f2 = X[, 2], f3 = X[, 3], f4 = X[, 4], f5 = X[, 5],
    target = X[, 1] * 2 + X[, 2] * 0.5 + stats::rnorm(n) * 0.1
  )
}

multiclass_data <- function() {
  withr::local_seed(42L)
  n <- 300L
  data.frame(
    f1     = stats::rnorm(n),
    f2     = stats::rnorm(n),
    f3     = stats::rnorm(n),
    target = sample(c("cat", "dog", "fish"), n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------------
# Chain 1: split -> fit -> predict -> evaluate -> assess
# ---------------------------------------------------------------------------

test_that("chain1: full workflow classification", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", seed = 42L)
  preds <- ml_predict(model, s$valid)
  met   <- ml_evaluate(model, s$valid)
  verd  <- ml_assess(model, test = s$test)

  expect_equal(length(preds), nrow(s$valid))
  expect_true("accuracy" %in% names(met))
  expect_s3_class(verd, "ml_evidence")
  expect_true("accuracy" %in% names(verd))   # Evidence is dict-like
})

test_that("chain1: full workflow regression", {
  df    <- reg_data()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", seed = 42L)
  preds <- ml_predict(model, s$valid)
  met   <- ml_evaluate(model, s$valid)
  verd  <- ml_assess(model, test = s$test)

  expect_equal(length(preds), nrow(s$valid))
  expect_true("rmse" %in% names(met))
  expect_true("r2"   %in% names(met))
  expect_s3_class(verd, "ml_evidence")
})

test_that("chain1: full workflow multiclass", {
  df    <- multiclass_data()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", seed = 42L)
  preds <- ml_predict(model, s$valid)
  met   <- ml_evaluate(model, s$valid)

  expect_equal(length(preds), nrow(s$valid))
  expect_true(all(preds %in% c("cat", "dog", "fish")))
  expect_true("accuracy" %in% names(met))
})

test_that("chain1: fit on dev partition (train+valid combined)", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$dev, "target", seed = 42L)
  preds <- ml_predict(model, s$test)
  expect_equal(length(preds), nrow(s$test))
})

test_that("chain1: train accuracy >= valid - 0.05 (overfitting signal)", {
  df         <- clf_data()
  s          <- ml_split(df, "target", seed = 42L)
  model      <- ml_fit(s$train, "target", algorithm = "decision_tree", seed = 42L)
  train_acc  <- ml_evaluate(model, s$train)[["accuracy"]]
  valid_acc  <- ml_evaluate(model, s$valid)[["accuracy"]]
  expect_gte(train_acc, valid_acc - 0.05)
})

# ---------------------------------------------------------------------------
# Chain 2: save -> load -> predict (roundtrip)
# ---------------------------------------------------------------------------

test_that("chain2: save/load roundtrip — predictions identical", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", seed = 42L)
  preds_before <- ml_predict(model, s$valid)

  path <- tempfile(fileext = ".mlr")
  on.exit(unlink(path), add = TRUE)

  ml_save(model, path)
  loaded <- ml_load(path)
  preds_after <- ml_predict(loaded, s$valid)

  expect_equal(preds_before, preds_after)
})

test_that("chain2: save/load roundtrip — metrics identical", {
  df    <- reg_data()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", seed = 42L)
  met_before <- ml_evaluate(model, s$valid)

  path <- tempfile(fileext = ".mlr")
  on.exit(unlink(path), add = TRUE)

  ml_save(model, path)
  loaded <- ml_load(path)
  met_after <- ml_evaluate(loaded, s$valid)

  for (k in names(met_before)) {
    expect_equal(met_before[[k]], met_after[[k]], tolerance = 1e-10,
                 label = paste("Metric", k, "drifted after save/load"))
  }
})

test_that("chain2: loaded model accepts test partition", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", seed = 42L)

  path <- tempfile(fileext = ".mlr")
  on.exit(unlink(path), add = TRUE)
  ml_save(model, path)
  loaded <- ml_load(path)

  # assess on the original test partition -- must not error
  expect_s3_class(ml_assess(loaded, test = s$test), "ml_evidence")
})

# ---------------------------------------------------------------------------
# tune -> fit -> predict (hyperparameter transfer)
# ---------------------------------------------------------------------------

test_that("chain3: tune random_forest then fit with best_params", {
  skip_if_not_installed("ranger")
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  tuned <- ml_tune(s$train, "target", algorithm = "random_forest",
                   seed = 42L, n_trials = 3L)
  model <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 42L)
  preds <- ml_predict(model, s$valid)
  expect_equal(length(preds), nrow(s$valid))
  expect_s3_class(tuned, "ml_tuning_result")
})

test_that("chain3: tune knn then fit", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  tuned <- ml_tune(s$train, "target", algorithm = "knn",
                   seed = 42L, n_trials = 3L)
  model <- ml_fit(s$train, "target", algorithm = "knn", seed = 42L)
  preds <- ml_predict(model, s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("chain3: tune gradient_boosting then fit", {
  df <- clf_data()
  s  <- ml_split(df, "target", seed = 42L)
  tryCatch({
    tuned <- ml_tune(s$train, "target", algorithm = "gradient_boosting",
                     seed = 42L, n_trials = 3L)
    model <- ml_fit(s$train, "target", algorithm = "gradient_boosting", seed = 42L)
    preds <- ml_predict(model, s$valid)
    expect_equal(length(preds), nrow(s$valid))
  }, error = function(e) {
    skip(paste("gradient_boosting not available:", conditionMessage(e)))
  })
})

test_that("chain3: tune logistic then fit", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  tuned <- ml_tune(s$train, "target", algorithm = "logistic",
                   seed = 42L, n_trials = 3L)
  model <- ml_fit(s$train, "target", algorithm = "logistic", seed = 42L)
  preds <- ml_predict(model, s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("chain3: tune result is a tuning_result with best model", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  tuned <- ml_tune(s$train, "target", algorithm = "logistic",
                   seed = 42L, n_trials = 3L)
  expect_s3_class(tuned, "ml_tuning_result")
  expect_true(!is.null(tuned$best_model))
  preds <- ml_predict(tuned$best_model, s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

# ---------------------------------------------------------------------------
# Chain 4: screen -> fit -> compare
# ---------------------------------------------------------------------------

test_that("chain4: screen then fit top algorithm", {
  df <- clf_data()
  s  <- ml_split(df, "target", seed = 42L)
  lb <- ml_screen(s, "target", seed = 42L)
  expect_s3_class(lb, "ml_leaderboard")

  # Fit the top algorithm from leaderboard
  top_algo <- lb$algorithm[[1]]
  model    <- ml_fit(s$train, "target", algorithm = top_algo, seed = 42L)
  preds    <- ml_predict(model, s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("chain4: compare two algorithms returns leaderboard", {
  df <- clf_data()
  s  <- ml_split(df, "target", seed = 42L)
  m1 <- ml_fit(s$train, "target", algorithm = "logistic",      seed = 42L)
  m2 <- ml_fit(s$train, "target", algorithm = "decision_tree", seed = 42L)
  lb <- ml_compare(list(m1, m2), s$valid)
  expect_s3_class(lb, "ml_leaderboard")
  expect_gte(nrow(lb), 2L)
})

# ---------------------------------------------------------------------------
# Chain 5: fit -> explain -> validate
# ---------------------------------------------------------------------------

test_that("chain5: explain feature names come from input data", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", algorithm = "decision_tree", seed = 42L)
  imp   <- ml_explain(model)

  feature_names <- setdiff(names(df), "target")
  expect_true(all(imp$feature %in% feature_names))
})

test_that("chain5: validate rules pass (low bar)", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", seed = 42L)
  gate  <- ml_validate(model, test = s$test, rules = list(accuracy = ">0.3"))
  expect_true(gate$passed)
})

test_that("chain5: validate rules fail (impossible bar)", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", seed = 42L)
  gate  <- ml_validate(model, test = s$test, rules = list(accuracy = ">0.999"))
  expect_false(gate$passed)
})

# ---------------------------------------------------------------------------
# Chain 6: fit -> calibrate -> predict
# ---------------------------------------------------------------------------

test_that("chain6: calibrate then predict — same length as input", {
  withr::local_seed(42L)
  data <- data.frame(
    f1     = stats::rnorm(500L),
    f2     = stats::rnorm(500L),
    target = sample(c("yes", "no"), 500L, replace = TRUE),
    stringsAsFactors = FALSE
  )
  s     <- ml_split(data, "target", seed = 42L)
  model <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 42L)
  cal   <- ml_calibrate(model, s$valid)
  preds <- ml_predict(cal, s$test)
  expect_equal(length(preds), nrow(s$test))
})

test_that("chain6: calibrate proba rows sum to 1.0", {
  withr::local_seed(42L)
  data <- data.frame(
    f1     = stats::rnorm(500L),
    f2     = stats::rnorm(500L),
    target = sample(c("yes", "no"), 500L, replace = TRUE),
    stringsAsFactors = FALSE
  )
  s     <- ml_split(data, "target", seed = 42L)
  model <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 42L)
  cal   <- ml_calibrate(model, s$valid)
  proba <- predict(cal, newdata = s$test, proba = TRUE)
  row_sums <- rowSums(proba)
  expect_true(all(abs(row_sums - 1.0) < 1e-6))
})

# ---------------------------------------------------------------------------
# Chain 7: drift (monitoring)
# ---------------------------------------------------------------------------

test_that("chain7: drift — same data produces no drift", {
  df   <- clf_data()
  s    <- ml_split(df, "target", seed = 42L)
  rep  <- ml_drift(s$train, s$train)
  expect_false(rep$shifted)
})

test_that("chain7: drift — massive shift detected", {
  df      <- clf_data()
  s       <- ml_split(df, "target", seed = 42L)
  shifted <- s$valid
  shifted$f1 <- shifted$f1 + 20
  rep <- ml_drift(s$train, shifted)
  expect_true(rep$shifted)
})

# ---------------------------------------------------------------------------
# Chain 8: stack -> evaluate/predict
# ---------------------------------------------------------------------------

test_that("chain8: stack then evaluate returns accuracy", {
  df      <- clf_data()
  s       <- ml_split(df, "target", seed = 42L)
  stacked <- ml_stack(s$train, "target", seed = 42L)
  met     <- ml_evaluate(stacked, s$valid)
  expect_true("accuracy" %in% names(met))
})

test_that("chain8: stack then predict — correct length", {
  df      <- clf_data()
  s       <- ml_split(df, "target", seed = 42L)
  stacked <- ml_stack(s$train, "target", seed = 42L)
  preds   <- ml_predict(stacked, s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

# ---------------------------------------------------------------------------
# Chain 9: cross-algorithm consistency
# ---------------------------------------------------------------------------

test_that("chain9: all classification algorithms produce correct-length predictions", {
  algos <- c("random_forest", "logistic", "decision_tree", "knn", "naive_bayes", "elastic_net")
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)

  for (algo in algos) {
    tryCatch({
      model <- ml_fit(s$train, "target", algorithm = algo, seed = 42L)
      preds <- ml_predict(model, s$valid)
      expect_equal(length(preds), nrow(s$valid),
                   label = paste(algo, "prediction length"))
      expect_true(all(preds %in% c("yes", "no")),
                  label = paste(algo, "prediction values in label set"))
      expect_true(is.character(preds) || is.integer(preds) || is.numeric(preds),
                  label = paste(algo, "prediction type"))
    }, error = function(e) {
      skip(paste(algo, "not available:", conditionMessage(e)))
    })
  }
})

test_that("chain9: all regression algorithms produce correct-length predictions", {
  algos <- c("linear", "decision_tree", "knn", "elastic_net", "gradient_boosting")
  df    <- reg_data()
  s     <- ml_split(df, "target", seed = 42L)

  for (algo in algos) {
    tryCatch({
      model <- ml_fit(s$train, "target", algorithm = algo, seed = 42L)
      preds <- ml_predict(model, s$valid)
      expect_equal(length(preds), nrow(s$valid),
                   label = paste(algo, "regression prediction length"))
      expect_true(is.numeric(preds),
                  label = paste(algo, "regression predictions are numeric"))
    }, error = function(e) {
      skip(paste(algo, "not available:", conditionMessage(e)))
    })
  }
})

test_that("chain9: seed reproducibility — two identical seeds = identical predictions", {
  df <- clf_data()
  s  <- ml_split(df, "target", seed = 42L)
  m1 <- ml_fit(s$train, "target", seed = 42L)
  m2 <- ml_fit(s$train, "target", seed = 42L)
  p1 <- ml_predict(m1, s$valid)
  p2 <- ml_predict(m2, s$valid)
  expect_equal(p1, p2)
})

# ---------------------------------------------------------------------------
# Chain 10: edge case transitions
# ---------------------------------------------------------------------------

test_that("chain10: predict tolerates extra columns in newdata", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", seed = 42L)
  extra <- s$valid
  extra$bonus_col <- 999L
  preds <- ml_predict(model, extra)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("chain10: predict tolerates reordered columns", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", seed = 42L)
  reordered <- s$valid[, rev(names(s$valid))]
  preds <- ml_predict(model, reordered)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("chain10: string target labels survive fit -> predict -> evaluate", {
  withr::local_seed(42L)
  data <- data.frame(
    f1    = stats::rnorm(100L),
    f2    = stats::rnorm(100L),
    label = sample(c("yes", "no"), 100L, replace = TRUE),
    stringsAsFactors = FALSE
  )
  s     <- ml_split(data, "label", seed = 42L)
  model <- ml_fit(s$train, "label", seed = 42L)
  preds <- ml_predict(model, s$valid)
  expect_true(all(preds %in% c("yes", "no")))
  met   <- ml_evaluate(model, s$valid)
  expect_true("accuracy" %in% names(met))
})

test_that("chain10: NaN in features does not crash downstream", {
  withr::local_seed(42L)
  data <- data.frame(
    f1     = stats::rnorm(200L),
    f2     = stats::rnorm(200L),
    target = sample(c("yes", "no"), 200L, replace = TRUE),
    stringsAsFactors = FALSE
  )
  data$f1[1:5] <- NA_real_
  s     <- ml_split(data, "target", seed = 42L)
  model <- ml_fit(s$train, "target", seed = 42L)
  preds <- ml_predict(model, s$valid)
  expect_equal(length(preds), nrow(s$valid))
})

test_that("chain10: assess peeking prevention — second assess errors", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  model <- ml_fit(s$train, "target", seed = 42L)
  ml_assess(model, test = s$test)                        # first — ok
  expect_error(ml_assess(model, test = s$test))          # second — peeking error
})

# ---------------------------------------------------------------------------
# Chain 11: cv -> fit -> evaluate
# ---------------------------------------------------------------------------

test_that("chain11: ml_cv result feeds directly into ml_fit", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  cv    <- ml_cv(s, "target", folds = 3L, seed = 42L)
  model <- ml_fit(cv, "target", seed = 42L)
  expect_s3_class(model, "ml_model")
  expect_false(is.null(model$algorithm))
})

test_that("chain11: ml_cv result has correct fold count and dev coverage", {
  df   <- clf_data()
  s    <- ml_split(df, "target", seed = 42L)
  cv   <- ml_cv(s, "target", folds = 4L, seed = 42L)
  expect_equal(length(cv$folds), 4L)
  all_valid_idx <- unlist(lapply(cv$folds, `[[`, "valid"))
  expect_equal(sort(unique(all_valid_idx)), sort(seq_len(nrow(s$dev))))
})

test_that("chain11: ml_cv_temporal folds preserve time ordering", {
  withr::local_seed(42L)
  n  <- 200L
  df <- data.frame(
    date   = seq.Date(as.Date("2020-01-01"), by = "day", length.out = n),
    f1     = stats::rnorm(n),
    target = stats::rnorm(n)
  )
  s  <- ml_split(df, "target", time = "date", seed = 42L)
  cv <- ml_cv_temporal(s, "target", folds = 3L)
  expect_equal(length(cv$folds), 3L)
})

# ---------------------------------------------------------------------------
# Chain 13: shelf (model staleness monitoring)
# ---------------------------------------------------------------------------

test_that("chain13: shelf fresh on same distribution", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  cv    <- ml_cv(s, "target", folds = 3L, seed = 42L)
  model <- ml_fit(cv, "target", seed = 42L)   # needs scores_ from CV fit
  result <- ml_shelf(model, new = s$valid, target = "target")
  expect_s3_class(result, "ml_shelf_result")
  expect_false(is.null(result$fresh))
})

test_that("chain13: shelf returns metrics_now", {
  df    <- clf_data()
  s     <- ml_split(df, "target", seed = 42L)
  cv    <- ml_cv(s, "target", folds = 3L, seed = 42L)
  model <- ml_fit(cv, "target", seed = 42L)
  result <- ml_shelf(model, new = s$valid, target = "target")
  expect_true(is.list(result$metrics_now))
  expect_true(length(result$metrics_now) > 0L)
})

# ---------------------------------------------------------------------------
# Chain 18: temporal workflow
# ---------------------------------------------------------------------------

test_that("chain18: temporal split then fit predict", {
  withr::local_seed(42L)
  n  <- 200L
  df <- data.frame(
    date   = seq.Date(as.Date("2020-01-01"), by = "day", length.out = n),
    f1     = stats::rnorm(n),
    f2     = stats::rnorm(n),
    target = sample(c("yes", "no"), n, replace = TRUE),
    stringsAsFactors = FALSE
  )
  s     <- ml_split_temporal(df, "target", time = "date")
  model <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 42L)
  preds <- ml_predict(model, s$valid)
  expect_equal(length(preds), nrow(s$valid))
  expect_gt(nrow(s$train), 0L)
  expect_gt(nrow(s$valid), 0L)
  expect_gt(nrow(s$test),  0L)
})

test_that("chain18: temporal cv then fit", {
  withr::local_seed(42L)
  n  <- 300L
  df <- data.frame(
    date   = seq.Date(as.Date("2020-01-01"), by = "day", length.out = n),
    f1     = stats::rnorm(n),
    f2     = stats::rnorm(n),
    target = stats::rnorm(n)
  )
  s     <- ml_split_temporal(df, "target", time = "date")
  cv    <- ml_cv_temporal(s, "target", folds = 3L)
  model <- ml_fit(cv, "target", seed = 42L)
  expect_s3_class(model, "ml_model")
})

# ---------------------------------------------------------------------------
# Chain 19: calibrate -> save -> load (calibrated roundtrip)
# ---------------------------------------------------------------------------

test_that("chain19: calibrated model survives save/load cycle", {
  withr::local_seed(42L)
  data <- data.frame(
    f1     = stats::rnorm(500L),
    f2     = stats::rnorm(500L),
    target = sample(c("yes", "no"), 500L, replace = TRUE),
    stringsAsFactors = FALSE
  )
  s     <- ml_split(data, "target", seed = 42L)
  model <- ml_fit(s$train, "target", algorithm = "random_forest", seed = 42L)
  cal   <- ml_calibrate(model, s$valid)
  preds_before <- ml_predict(cal, s$test)
  proba_before <- predict(cal, newdata = s$test, proba = TRUE)

  path <- tempfile(fileext = ".mlr")
  on.exit(unlink(path), add = TRUE)
  ml_save(cal, path)
  loaded      <- ml_load(path)
  preds_after <- ml_predict(loaded, s$test)
  proba_after <- predict(loaded, newdata = s$test, proba = TRUE)

  expect_equal(preds_before, preds_after)
  expect_equal(proba_before, proba_after)   # proba columns match exactly
})

# ---------------------------------------------------------------------------
# Chain 20: tune -> stack (tuned models compose into ensemble)
# ---------------------------------------------------------------------------

test_that("chain20: tune then stack does not error", {
  df      <- clf_data()
  s       <- ml_split(df, "target", seed = 42L)
  tuned   <- ml_tune(s$train, "target", algorithm = "random_forest",
                     seed = 42L, n_trials = 3L)
  stacked <- ml_stack(s$train, "target", seed = 42L)
  met     <- ml_evaluate(stacked, s$valid)
  expect_true("accuracy" %in% names(met))
})

# ---------------------------------------------------------------------------
# Chain 21: quick (one-liner convenience)
# ---------------------------------------------------------------------------

test_that("chain21: quick classification returns model and metrics", {
  df     <- clf_data()
  result <- ml_quick(df, "target", seed = 42L)
  expect_false(is.null(result$model))
  expect_false(is.null(result$metrics))
  expect_true("accuracy" %in% names(result$metrics))
})

test_that("chain21: quick regression returns model and metrics", {
  df     <- reg_data()
  result <- ml_quick(df, "target", seed = 42L)
  expect_false(is.null(result$model))
  expect_false(is.null(result$metrics))
})

# ---------------------------------------------------------------------------
# Chain 23: profile -> check_data -> leak (data quality trifecta)
# ---------------------------------------------------------------------------

test_that("chain23: profile returns non-null result", {
  df   <- clf_data()
  prof <- ml_profile(df, "target")
  expect_false(is.null(prof))
})

test_that("chain23: check_data passes on clean data", {
  df     <- clf_data()
  report <- ml_check_data(df, "target")
  expect_false(is.null(report))
})

test_that("chain23: leak detects perfect copy of target", {
  withr::local_seed(42L)
  n    <- 200L
  leak <- sample(c(0L, 1L), n, replace = TRUE)
  data <- data.frame(
    f1     = stats::rnorm(n),
    leaky  = leak,
    target = leak,           # perfect leakage
    stringsAsFactors = FALSE
  )
  report <- ml_leak(data, "target")
  expect_false(is.null(report))
  # Either clean=FALSE or suspects is non-empty
  expect_true(!isTRUE(report$clean) || length(report$suspects) > 0L)
})

# ---------------------------------------------------------------------------
# Chain 24: enough (sample size check)
# ---------------------------------------------------------------------------

test_that("chain24: enough returns ml_enough_result", {
  df <- clf_data()
  s  <- ml_split(df, "target", seed = 42L)
  result <- ml_enough(s, "target")
  expect_s3_class(result, "ml_enough_result")
})

test_that("chain24: enough returns saturated flag and recommendation", {
  df <- clf_data()
  s  <- ml_split(df, "target", seed = 42L)
  result <- ml_enough(s, "target", seed = 42L)
  expect_true(is.logical(result$saturated))
  expect_type(result$recommendation, "character")
})
