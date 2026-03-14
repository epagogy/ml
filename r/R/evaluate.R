#' Evaluate model on validation data (iterate freely)
#'
#' The practice exam — call as many times as needed. For the one-time final
#' grade on held-out test data, use [ml_assess()].
#'
#' @param model An `ml_model` or `ml_tuning_result`
#' @param data A data.frame containing the target column
#' @returns An object of class `ml_metrics` (named numeric vector with print method)
#' @export
#' @examples
#' s <- ml_split(iris, "Species", seed = 42)
#' model <- ml_fit(s$train, "Species", seed = 42)
#' metrics <- ml_evaluate(model, s$valid)
#' metrics[["accuracy"]]
ml_evaluate <- function(model, data) {
  # Auto-unwrap TuningResult
  if (inherits(model, "ml_tuning_result")) model <- model[["best_model"]]
  if (!inherits(model, "ml_model")) model_error("Expected an ml_model or ml_tuning_result")

  data <- .coerce_data(data)

  # Partition guard — reject unsplit or test-tagged data
  if (.guards_active()) {
    .part <- .resolve_partition(data)
    if (is.null(.part)) {
      .guard_action(paste0(
        "ml_evaluate() received data without split provenance. ",
        "Split your data first: s <- ml_split(df, target, seed = 42), ",
        "then ml_evaluate(model, s$valid). ",
        "To disable: ml_config(guards = 'off')"
      ))
    } else if (.part == "test") {
      .guard_action(paste0(
        "ml_evaluate() received data tagged as 'test' partition. ",
        "ml_evaluate() is the practice exam -- use validation data. ",
        "For the final exam, use ml_assess(model, test = s$test)."
      ))
    }
  }

  .score_impl(model, data)
}


# Shared scoring logic — no guards. Caller is responsible for validation.
# Used by ml_evaluate, ml_assess, ml_compare, ml_validate, ml_screen, ml_shelf.
.score_impl <- function(model, data) {
  # Auto-unwrap TuningResult
  if (inherits(model, "ml_tuning_result")) model <- model[["best_model"]]
  if (!inherits(model, "ml_model")) model_error("Expected an ml_model or ml_tuning_result")

  # Route calibrated models through their own evaluate path (uses S3 predict)
  if (inherits(model, "ml_calibrated_model")) {
    return(.evaluate_calibrated_impl(model, data))
  }

  data <- .coerce_data(data)

  if (!model[["target"]] %in% names(data)) {
    data_error(paste0(
      "target column '", model[["target"]], "' not found in data. Available: ",
      paste(names(data), collapse = ", ")
    ))
  }

  t_start <- proc.time()[["elapsed"]]
  preds   <- .predict_impl(model, data)
  y_true  <- data[[model[["target"]]]]

  # Use engine + X for roc_auc computation (binary)
  X <- data[, model[["features"]][model[["features"]] %in% names(data)], drop = FALSE]
  X_enc <- tryCatch(.transform(X, model[["encoders"]]), error = function(e) NULL)

  metrics <- .compute_metrics(preds, y_true, model[["task"]],
                               engine = model[["engine"]],
                               X      = X_enc,
                               algorithm = model[["algorithm"]],
                               norm   = model[["encoders"]])
  t_end <- proc.time()[["elapsed"]]
  time_s <- as.numeric(t_end - t_start)

  # Round to 4dp — matches Python: {k: round(v, 4) for k, v in result.items()}
  raw <- unlist(metrics)
  raw <- round(raw, 4L)
  new_ml_metrics(raw, task = model[["task"]], time = time_s)
}
