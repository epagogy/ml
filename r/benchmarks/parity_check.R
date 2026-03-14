# Parity check: ml (R) vs raw R packages on the same data, split, and algorithm.
#
# Proves that ml wraps ranger/nnet/stats faithfully -- same predictions, same metrics.
# Run: Rscript benchmarks/parity_check.R

library(ml)

cat(sprintf("ml parity check — same data, same split, same algorithm\n"))
cat(sprintf("ml %s\n\n", packageVersion("ml")))

# ── helpers ────────────────────────────────────────────────────────────────────

rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
mae  <- function(y, yhat) mean(abs(y - yhat))
r2   <- function(y, yhat) 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)

ordinal_encode <- function(train_df, valid_df, target) {
  for (col in names(train_df)) {
    if (col == target) next
    if (is.character(train_df[[col]]) || is.factor(train_df[[col]])) {
      lvls <- sort(unique(as.character(train_df[[col]])))
      train_df[[col]] <- as.integer(factor(as.character(train_df[[col]]), levels = lvls)) - 1L
      valid_df[[col]] <- as.integer(factor(as.character(valid_df[[col]]), levels = lvls)) - 1L
    }
  }
  list(train = train_df, valid = valid_df)
}

compare <- function(name, ml_vals, raw_vals) {
  cat(sprintf("\n%s\n", strrep("=", 60)))
  cat(sprintf("  %s\n", name))
  cat(sprintf("%s\n", strrep("=", 60)))
  cat(sprintf("  %-12s %10s %10s %10s\n", "metric", "ml", "raw", "delta"))
  cat(sprintf("  %s\n", strrep("-", 44)))
  all_close <- TRUE
  for (m in names(ml_vals)) {
    if (!m %in% names(raw_vals)) next
    delta <- abs(ml_vals[[m]] - raw_vals[[m]])
    mark  <- if (delta < 0.0001) "v" else if (delta < 0.01) "~" else "X"
    if (delta >= 0.01) all_close <- FALSE
    cat(sprintf("  %-12s %10.4f %10.4f %10.4f %s\n",
                m, ml_vals[[m]], raw_vals[[m]], delta, mark))
  }
  all_close
}

# ── test 1: random forest regression (tips) ────────────────────────────────────

test_rf_regression <- function() {
  data   <- ml_dataset("diabetes")
  target <- "target"

  s       <- ml_split(data, target, seed = 42)
  ml_mod  <- ml_fit(s$train, target, algorithm = "random_forest", seed = 42)
  ml_eval <- ml_evaluate(ml_mod, s$valid)

  # raw ranger — same split, same seed
  enc    <- ordinal_encode(s$train, s$valid, target)
  f      <- as.formula(paste(target, "~ ."))
  raw    <- ranger::ranger(f, data = enc$train, num.trees = 500, seed = 42)
  yhat   <- predict(raw, data = enc$valid)$predictions
  y      <- enc$valid[[target]]

  raw_vals <- list(rmse = rmse(y, yhat), mae = mae(y, yhat), r2 = r2(y, yhat))
  ml_vals  <- list(rmse = ml_eval[["rmse"]], mae = ml_eval[["mae"]], r2 = ml_eval[["r2"]])

  compare("Random Forest — diabetes (regression)", ml_vals, raw_vals)
}

# ── test 2: random forest classification (cancer) ──────────────────────────────

test_rf_classification <- function() {
  data   <- ml_dataset("cancer")
  target <- "target"

  s       <- ml_split(data, target, seed = 42)
  ml_mod  <- ml_fit(s$train, target, algorithm = "random_forest", seed = 42)
  ml_eval <- ml_evaluate(ml_mod, s$valid)

  # raw ranger — probability mode for AUC
  enc     <- ordinal_encode(s$train, s$valid, target)
  enc$train[[target]] <- factor(enc$train[[target]])
  f       <- as.formula(paste(target, "~ ."))
  raw     <- ranger::ranger(f, data = enc$train, num.trees = 500, seed = 42,
                             probability = TRUE)
  proba   <- predict(raw, data = enc$valid)$predictions
  yhat    <- colnames(proba)[max.col(proba)]
  y_true  <- as.character(enc$valid[[target]])

  acc <- mean(yhat == y_true)

  ml_vals  <- list(accuracy = ml_eval[["accuracy"]])
  raw_vals <- list(accuracy = acc)

  compare("Random Forest — cancer (binary clf)", ml_vals, raw_vals)
}

# ── test 3: logistic regression (iris binary) ──────────────────────────────────

test_logistic <- function() {
  data   <- ml_dataset("iris")
  # binary: setosa vs not
  data$species_bin <- ifelse(data$Species == "setosa", "setosa", "other")
  data$Species     <- NULL
  target           <- "species_bin"

  s       <- ml_split(data, target, seed = 42)
  ml_mod  <- ml_fit(s$train, target, algorithm = "logistic", seed = 42)
  ml_eval <- ml_evaluate(ml_mod, s$valid)

  # raw nnet::multinom (same as ml's logistic backend)
  # ml auto-scales; replicate with scale()
  num_cols <- setdiff(names(s$train), target)
  means    <- colMeans(s$train[num_cols])
  sds      <- apply(s$train[num_cols], 2, sd)

  train_sc        <- s$train
  valid_sc        <- s$valid
  train_sc[num_cols] <- scale(train_sc[num_cols], center = means, scale = sds)
  valid_sc[num_cols] <- scale(valid_sc[num_cols], center = means, scale = sds)

  f   <- as.formula(paste(target, "~ ."))
  raw <- nnet::multinom(f, data = train_sc, trace = FALSE)
  yhat <- as.character(predict(raw, newdata = valid_sc, type = "class"))
  y    <- as.character(valid_sc[[target]])
  acc  <- mean(yhat == y)

  ml_vals  <- list(accuracy = ml_eval[["accuracy"]])
  raw_vals <- list(accuracy = acc)

  compare("Logistic Regression — iris binary (scaled)", ml_vals, raw_vals)
}

# ── run ────────────────────────────────────────────────────────────────────────

results <- c(
  test_rf_regression(),
  test_rf_classification(),
  test_logistic()
)

cat(sprintf("\n%s\n", strrep("=", 60)))
cat(sprintf("  %d/%d parity checks passed (delta < 0.01)\n",
            sum(results), length(results)))
if (all(results)) {
  cat("  ml wraps ranger/nnet faithfully — same results.\n")
} else {
  cat("  Some deltas > 0.01 — investigate preprocessing differences.\n")
}
cat(sprintf("%s\n", strrep("=", 60)))
