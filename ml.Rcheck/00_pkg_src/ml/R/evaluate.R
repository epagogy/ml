#' Evaluate model on validation data (iterate freely)
#'
#' The practice exam -- call as many times as needed. For the one-time final
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
  .evaluate_impl(model = model, data = data, .guard = TRUE)
}

.evaluate_impl <- function(model, data, .guard = TRUE) {
  # Auto-unwrap TuningResult
  if (inherits(model, "ml_tuning_result")) model <- model[["best_model"]]
  if (!inherits(model, "ml_model")) model_error("Expected an ml_model or ml_tuning_result")

  # Route calibrated models through their own evaluate path (uses S3 predict)
  if (inherits(model, "ml_calibrated_model")) {
    return(.evaluate_calibrated_impl(model, data))
  }

  data <- .coerce_data(data)

  # Partition guard -- reject unsplit or test-tagged data
  if (isTRUE(.guard) && .guards_active()) {
    .part <- attr(data, "_ml_partition")
    if (is.null(.part)) {
      partition_error(paste0(
        "ml_evaluate() received data without split provenance. ",
        "Split your data first: s <- ml_split(df, target, seed = 42), ",
        "then ml_evaluate(model, s$valid). ",
        "Note: dplyr verbs (filter, mutate, select) strip partition tags. ",
        "Use base R subsetting or re-split after dplyr operations. ",
        "To disable: ml_config(guards = 'off')"
      ))
    } else if (.part == "test") {
      partition_error(paste0(
        "ml_evaluate() received data tagged as 'test' partition. ",
        "ml_evaluate() is the practice exam -- use validation data. ",
        "For the final exam, use ml_assess(model, test = s$test)."
      ))
    }
  }

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

  # Round to 4dp -- matches Python: {k: round(v, 4) for k, v in result.items()}
  raw <- unlist(metrics)
  raw <- round(raw, 4L)
  new_ml_metrics(raw, task = model[["task"]], time = time_s)
}
