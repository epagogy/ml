#' Optimize decision threshold for binary classification
#'
#' Sweeps thresholds from `min_threshold` to 0.95 in two phases (coarse
#' 0.05 steps, then fine 0.005 steps around the coarse best) and returns a
#' copy of the model with a tuned threshold. Subsequent `ml_predict()` calls
#' apply this threshold to positive-class probability instead of 0.5.
#'
#' @param model An `ml_model` or `ml_tuning_result` (binary classification only).
#' @param data A data.frame containing the target column used as true labels.
#' @param metric Character. Optimisation objective: `"f1"`, `"accuracy"`,
#'   `"precision"`, or `"recall"`. Ranking metrics (`"roc_auc"`, `"log_loss"`)
#'   are rejected because they are threshold-independent.
#' @param min_threshold Lower bound of the sweep. `"auto"` (default) computes
#'   `max(0.001, 1 / n_positives)` -- the minimum meaningful threshold for
#'   imbalanced data. Pass a numeric value to override.
#' @returns An `ml_optimize_result` (also an `ml_model`). The threshold is
#'   baked in -- every `ml_predict()` call uses it automatically.
#'   Inspect with `result$threshold`. The original model is unchanged.
#' @export
#' @examples
#' s     <- ml_split(iris[iris$Species != "virginica", ], "Species", seed = 42)
#' model <- ml_fit(s$train, "Species", seed = 42)
#' opt   <- ml_optimize(model, data = s$valid, metric = "f1")
#' opt$threshold
ml_optimize <- function(model, data, metric = "f1",
                        min_threshold = "auto") {
  # Auto-unwrap TuningResult
  if (inherits(model, "ml_tuning_result")) model <- model[["best_model"]]
  if (!inherits(model, "ml_model")) {
    model_error(paste0(
      "ml_optimize() requires an ml_model or ml_tuning_result. ",
      "Use ml_fit() or ml_tune() first."
    ))
  }

  # Task guard
  if (model[["task"]] != "classification") {
    model_error(paste0(
      "ml_optimize() only works for binary classification models. ",
      "This model's task is '", model[["task"]], "'."
    ))
  }

  # Binary-only guard
  classes <- model[["encoders"]][["label_levels"]]
  if (is.null(classes) || length(classes) != 2L) {
    n <- if (is.null(classes)) 0L else length(classes)
    model_error(paste0(
      "ml_optimize() requires binary classification (2 classes). ",
      "This model has ", n, " class(es). ",
      "For multiclass, use per-class thresholds manually."
    ))
  }

  # Ranking metric guard
  ranking_metrics <- c("roc_auc", "log_loss", "auc")
  if (is.character(metric) && metric %in% ranking_metrics) {
    config_error(paste0(
      "metric='", metric, "' is a ranking metric and cannot be ",
      "threshold-optimised. Use a label-based metric instead: ",
      "'f1', 'accuracy', 'precision', 'recall'."
    ))
  }

  # True labels
  target <- model[["target"]]
  if (!target %in% names(data)) {
    data_error(paste0(
      "Target column '", target, "' not found in data. ",
      "Available columns: ", paste(names(data), collapse = ", ")
    ))
  }
  y_true <- as.character(data[[target]])

  # Positive-class probabilities (last column = positive class, sklearn convention)
  proba_df  <- ml_predict_proba(model, data)
  pos_proba <- as.numeric(proba_df[, ncol(proba_df)])

  # Compute min threshold
  if (identical(min_threshold, "auto")) {
    n_pos     <- sum(y_true == classes[2L])
    min_thresh <- if (n_pos > 0L) max(0.001, 1.0 / n_pos) else 0.001
  } else {
    min_thresh <- as.numeric(min_threshold)
  }

  # Scorer closure
  scorer <- .threshold_scorer(metric, classes[2L])

  # Phase 1: coarse scan (0.05 steps from min_thresh to 0.95)
  coarse_thresholds <- seq(min_thresh, 0.95, by = 0.05)
  best_score <- -Inf
  best_thresh <- coarse_thresholds[[1L]]

  for (thresh in coarse_thresholds) {
    preds <- ifelse(pos_proba >= thresh, classes[2L], classes[1L])
    score <- tryCatch(scorer(y_true, preds), error = function(e) -Inf)
    if (!is.na(score) && score > best_score) {
      best_score <- score
      best_thresh <- thresh
    }
  }

  # Phase 2: fine scan +/-0.1 around coarse best (0.005 steps)
  fine_min <- max(min_thresh, best_thresh - 0.1)
  fine_max <- min(0.95, best_thresh + 0.1)
  fine_thresholds <- seq(fine_min, fine_max, by = 0.005)

  for (thresh in fine_thresholds) {
    preds <- ifelse(pos_proba >= thresh, classes[2L], classes[1L])
    score <- tryCatch(scorer(y_true, preds), error = function(e) -Inf)
    if (!is.na(score) && score > best_score) {
      best_score <- score
      best_thresh <- thresh
    }
  }

  # Build result: copy of model with threshold baked in
  optimized           <- model
  optimized[["threshold"]]        <- best_thresh
  optimized[["threshold_metric"]] <- metric
  optimized[["threshold_score"]]  <- best_score

  structure(
    c(list(threshold = best_thresh, metric = metric, score = best_score),
      optimized),
    class = c("ml_optimize_result", "ml_model")
  )
}

#' @export
print.ml_optimize_result <- function(x, ...) {
  cat(sprintf(
    "-- Threshold optimisation\n   metric: %s\n   threshold: %.4f  (score: %.4f)\n",
    x$metric, x$threshold, x$score
  ))
  invisible(x)
}

# -- Internal: build threshold scorer -----------------------------------------

.threshold_scorer <- function(metric, pos_label) {
  switch(metric,
    "accuracy" = function(truth, pred) mean(pred == truth, na.rm = TRUE),
    "f1" = function(truth, pred) {
      tp   <- sum(pred == pos_label & truth == pos_label)
      fp   <- sum(pred == pos_label & truth != pos_label)
      fn   <- sum(pred != pos_label & truth == pos_label)
      prec <- if (tp + fp == 0L) 0 else tp / (tp + fp)
      rec  <- if (tp + fn == 0L) 0 else tp / (tp + fn)
      if (prec + rec == 0) 0 else 2 * prec * rec / (prec + rec)
    },
    "precision" = function(truth, pred) {
      tp <- sum(pred == pos_label & truth == pos_label)
      fp <- sum(pred == pos_label & truth != pos_label)
      if (tp + fp == 0L) 0 else tp / (tp + fp)
    },
    "recall" = function(truth, pred) {
      tp <- sum(pred == pos_label & truth == pos_label)
      fn <- sum(pred != pos_label & truth == pos_label)
      if (tp + fn == 0L) 0 else tp / (tp + fn)
    },
    config_error(paste0(
      "Unknown metric='", metric, "'. ",
      "Choose from: 'f1', 'accuracy', 'precision', 'recall'."
    ))
  )
}
