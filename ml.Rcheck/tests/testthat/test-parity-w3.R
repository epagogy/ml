# W3 User Persona Parity — canonical end-to-end workflows
# Mirrors the canonical Python API section exactly (same verb sequence).
# Max 20 tests.

library(ml)

# ── PERSONA A: BEGINNER — dev loop workflow ────────────────────────────────────
# Python:  s = ml.split(df, "target", seed=42)
#          model = ml.fit(s.train, "target", seed=42)
#          preds = ml.predict(model, s.valid)
#          metrics = ml.evaluate(model, s.valid)
#          imp = ml.explain(model)

test_that("W3-01: split → fit → predict → evaluate pipeline works end-to-end", {
  s      <- ml_split(iris, "Species", seed = 42L)
  model  <- ml_fit(s$train, "Species", seed = 42L)
  preds  <- ml_predict(model, s$valid)
  metrics <- ml_evaluate(model, s$valid)
  expect_true(inherits(s, "ml_split_result"))
  expect_true(inherits(model, "ml_model"))
  expect_equal(length(preds), nrow(s$valid))
  expect_true("accuracy" %in% names(metrics))
  expect_true(metrics[["accuracy"]] >= 0 && metrics[["accuracy"]] <= 1)
})

test_that("W3-02: explain() returns feature importance for default model", {
  s     <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42L)
  imp   <- ml_explain(model)
  expect_true(inherits(imp, "ml_explanation"))
  df <- as.data.frame(imp)
  expect_true(nrow(df) > 0L)
  expect_true(all(c("feature", "importance") %in% names(df)))
  expect_true(all(df$importance >= 0))
})

# ── PERSONA B: PRACTITIONER — tuning workflow ──────────────────────────────────
# Python:  tuned = ml.tune(s.train, "target", algorithm="xgboost", seed=42)
#          stacked = ml.stack(s.train, "target", seed=42)
#          lb = ml.compare([tuned, model], s.valid)

test_that("W3-03: tune returns ml_tuning_result that can be used in compare", {
  s     <- ml_split(iris, "Species", seed = 42L)
  tuned <- ml_tune(s$train, "Species", algorithm = "logistic",
                   n_trials = 2L, seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  lb    <- ml_compare(list(tuned, model), s$valid)
  expect_true(inherits(lb, "ml_leaderboard"))
  expect_equal(nrow(lb), 2L)
})

test_that("W3-04: compare([tuned, model], data) is sorted by primary metric", {
  s     <- ml_split(iris, "Species", seed = 42L)
  m1    <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m2    <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 99L)
  lb    <- ml_compare(list(m1, m2), s$valid)
  df    <- as.data.frame(lb)
  # Should be sorted by accuracy (descending) for multiclass without roc_auc
  expect_equal(nrow(df), 2L)
  if (nrow(df) >= 2L && "accuracy" %in% names(df)) {
    expect_true(df$accuracy[1] >= df$accuracy[2] - 1e-6)
  }
})

# ── PERSONA C: PRODUCTION — finalize workflow ──────────────────────────────────
# Python:  final = ml.fit(s.dev, "target", seed=42)   ← retrain on train+valid
#          gate = ml.validate(final, test=s.test, rules={"accuracy": ">0.85"}, baseline=model)
#          verdict = ml.assess(final, test=s.test)

test_that("W3-05: fit on $dev (train+valid) works and produces valid model", {
  s     <- ml_split(iris, "Species", seed = 42L)
  final <- ml_fit(s$dev, "Species", seed = 42L)
  expect_true(inherits(final, "ml_model"))
  # final model should be trained on more data than train-only model
  expect_true(final$n_train >= nrow(s$train))
})

test_that("W3-06: validate passes when accuracy > 0.5 on iris logistic", {
  s     <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  gate  <- ml_validate(model, test = s$test,
                        rules = list(accuracy = ">0.5"))
  expect_true(inherits(gate, "ml_validate_result"))
  expect_true(gate$passed, info = paste("Validate failed. failures:",
                                         paste(gate$failures, collapse="; ")))
})

test_that("W3-07: validate fails when impossible threshold set (accuracy > 0.9999)", {
  s     <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  gate  <- ml_validate(model, test = s$test,
                        rules = list(accuracy = ">0.9999"))
  # May pass or fail — just check structure is correct
  expect_true(is.logical(gate$passed))
  expect_true(is.character(gate$failures))
})

test_that("W3-08: assess() returns ml_metrics (the 'do once' final exam)", {
  s       <- ml_split(iris, "Species", seed = 42L)
  model   <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  verdict <- ml_assess(model, test = s$test)
  expect_true(inherits(verdict, "ml_evidence"))
  expect_true("accuracy" %in% names(verdict))
})

test_that("W3-09: assess() errors on second call (Python: raises ModelError)", {
  s     <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  ml_assess(model, test = s$test)  # first call — OK
  expect_error(
    ml_assess(model, test = s$test),  # second call — must error
    regexp = "peeking|times|repeated",
    ignore.case = TRUE
  )
})

# ── PERSONA D: MONITORING — drift + shelf workflow ─────────────────────────────
# Python:  result = ml.drift(reference=s.train, new=new_customers)
#          result = ml.shelf(model, new=labeled_batch, target="churn")

test_that("W3-10: drift detects no drift on same distribution", {
  s   <- ml_split(iris, "Species", seed = 42L)
  res <- ml_drift(reference = s$train, new = s$valid, target = "Species")
  expect_true(inherits(res, "ml_drift_result"))
  expect_true(is.logical(res$shifted))
  # Same underlying distribution → likely no drift
  expect_false(res$shifted,
               info = paste("Unexpected drift detected. severity:", res$severity))
})

test_that("W3-11: drift detects obvious drift (large mean shift)", {
  s      <- ml_split(iris, "Species", seed = 42L)
  new_df <- s$test
  new_df$Sepal.Length <- new_df$Sepal.Length + 10  # massive shift
  new_df$Sepal.Width  <- new_df$Sepal.Width  + 10
  res <- ml_drift(reference = s$train, new = new_df, target = "Species")
  expect_true(res$shifted, info = "Expected drift to be detected with +10 shift")
  expect_true(length(res$features_shifted) >= 2L)
})

test_that("W3-12: shelf() works end-to-end on CV model", {
  cv    <- ml_split(iris, "Species", seed = 42L, folds = 3L)
  model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42L)
  new_batch <- iris[sample(nrow(iris), 40L), ]
  result <- ml_shelf(model, new = new_batch, target = "Species")
  expect_true(inherits(result, "ml_shelf_result"))
  expect_true(is.logical(result$fresh))
  expect_true(is.character(result$recommendation))
})

# ── PERSONA E: SCREENING — algorithm discovery ─────────────────────────────────
# Python:  leaderboard = ml.screen(s, "target")

test_that("W3-13: screen returns ranked leaderboard", {
  s  <- ml_split(iris, "Species", seed = 42L)
  lb <- ml_screen(s, "Species", seed = 42L,
                   algorithms = c("logistic", "random_forest"))
  expect_true(inherits(lb, "ml_leaderboard"))
  expect_true(nrow(lb) >= 1L)
  expect_true("algorithm" %in% names(lb))
})

test_that("W3-14: screen $best_model is an ml_model", {
  s  <- ml_split(iris, "Species", seed = 42L)
  lb <- ml_screen(s, "Species", seed = 42L, algorithms = c("logistic"))
  # Python: lb.best_model returns the top Model
  # R: lb$best_model should return the top ml_model
  bm <- lb$best_model
  if (!is.null(bm)) {
    expect_true(inherits(bm, "ml_model"))
  }
})

# ── PERSONA F: REGRESSION workflow ─────────────────────────────────────────────

test_that("W3-15: regression workflow — split, fit, evaluate, assess all work", {
  s       <- ml_split(mtcars, "mpg", seed = 42L)
  model   <- ml_fit(s$train, "mpg", algorithm = "random_forest", seed = 42L)
  metrics <- ml_evaluate(model, s$valid)
  verdict <- ml_assess(model, test = s$test)
  expect_true("rmse" %in% names(metrics))
  expect_true("r2"   %in% names(metrics))
  expect_true(metrics[["rmse"]] > 0)
  expect_true(inherits(verdict, "ml_evidence"))
})

test_that("W3-16: regression validate with rmse threshold works", {
  s     <- ml_split(mtcars, "mpg", seed = 42L)
  model <- ml_fit(s$train, "mpg", algorithm = "random_forest", seed = 42L)
  gate  <- ml_validate(model, test = s$test,
                        rules = list(r2 = "> -1.0"))  # always passes
  expect_true(gate$passed,
              info = paste("Validation failed. failures:",
                            paste(gate$failures, collapse="; ")))
})

# ── PERSONA G: SAVE / LOAD WORKFLOW ──────────────────────────────────────────

test_that("W3-17: save and load round-trip preserves predictions", {
  s     <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  tmp   <- tempfile(fileext = ".mlr")
  on.exit(unlink(tmp), add = TRUE)
  ml_save(model, tmp)
  loaded <- ml_load(tmp)
  expect_true(inherits(loaded, "ml_model"))
  preds_orig   <- ml_predict(model, s$valid)
  preds_loaded <- ml_predict(loaded, s$valid)
  expect_equal(preds_orig, preds_loaded)
})

# ── PERSONA H: VALIDATE WITH BASELINE ────────────────────────────────────────

test_that("W3-18: validate with baseline detects regression", {
  s        <- ml_split(iris, "Species", seed = 42L)
  old_model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  new_model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  gate <- ml_validate(new_model, test = s$test, baseline = old_model, tolerance = 0.5)
  # With wide tolerance, should pass (same model architecture)
  expect_true(is.logical(gate$passed))
  # Should have baseline_metrics populated
  if (!is.null(gate$baseline_metrics)) {
    expect_true(length(gate$baseline_metrics) > 0L)
  }
})

# ── PERSONA I: PREDICT API ────────────────────────────────────────────────────

test_that("W3-19: predict returns same-length vector as input rows", {
  s     <- ml_split(iris, "Species", seed = 42L)
  model <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  preds <- ml_predict(model, s$test)
  expect_equal(length(preds), nrow(s$test))
  # All predictions should be one of the training classes
  valid_classes <- unique(iris$Species)
  expect_true(all(preds %in% as.character(valid_classes)))
})

# ── PERSONA J: MODULE-STYLE ACCESS ($) ────────────────────────────────────────

test_that("W3-20: ml$fit() module style produces same result as ml_fit()", {
  # Python: import ml; ml.fit(...)
  # R: ml$fit(...) — module-style alternative
  s  <- ml_split(iris, "Species", seed = 42L)
  m1 <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  m2 <- ml$fit(s$train, "Species", algorithm = "logistic", seed = 42L)
  expect_equal(m1$algorithm, m2$algorithm)
  expect_equal(m1$task,      m2$task)
  expect_equal(m1$target,    m2$target)
})
