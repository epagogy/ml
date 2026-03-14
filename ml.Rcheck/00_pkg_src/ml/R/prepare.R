#' Prepare data for ML: encode, impute, and scale
#'
#' Grammar primitive #2: DataFrame -> PreparedData.
#'
#' In the default workflow, `ml_fit()` calls preparation internally per fold.
#' Use `ml_prepare()` explicitly when you need manual control: inspect the
#' preprocessing state, apply the same encoding to external data, or chain
#' preparation with fitting.
#'
#' @param data A data.frame including the target column.
#' @param target Name of the target column (string).
#' @param algorithm Algorithm hint for encoding strategy: "auto", "random_forest",
#'   "logistic", etc. Tree-based algorithms use ordinal encoding; linear algorithms
#'   use one-hot encoding for low-cardinality categoricals.
#' @param task "classification", "regression", or "auto" (detected from target).
#'
#' @returns An `ml_prepared_data` object with:
#'   - `$data`   -- transformed data.frame (all-numeric, ready for ml_fit)
#'   - `$state`  -- NormState list; use .transform(state, X) on new data
#'   - `$target` -- target column name
#'   - `$task`   -- detected or provided task type
#'
#' @examples
#' \donttest{
#' s <- ml_split(iris, "Species", seed = 42)
#' p <- ml_prepare(s$train, "Species")
#' p$task       # "classification"
#' head(p$data) # encoded feature matrix
#' }
#'
#' @export
ml_prepare <- function(data, target, algorithm = "auto", task = "auto") {
  data <- .coerce_data(data)

  if (is.null(target) || !nzchar(target)) {
    config_error("target must be a non-empty string")
  }
  if (!target %in% names(data)) {
    config_error(paste0("target '", target, "' not found in data columns"))
  }

  X <- data[, setdiff(names(data), target), drop = FALSE]
  y <- data[[target]]

  detected_task <- .detect_task(y, task)

  norm_state  <- .prepare(X, y, algorithm = algorithm, task = detected_task)
  transformed <- .transform_fit(X, norm_state)
  X_enc       <- transformed$X_enc
  norm_state  <- transformed$norm_state

  structure(
    list(
      data   = X_enc,
      state  = norm_state,
      target = target,
      task   = detected_task
    ),
    class = "ml_prepared_data"
  )
}

#' Print prepared data summary
#'
#' @param x An `ml_prepared_data` object.
#' @param ... Additional arguments (ignored).
#' @returns The object `x`, invisibly.
#' @export
print.ml_prepared_data <- function(x, ...) {
  cat(sprintf(
    "PreparedData  %d rows x %d cols  task=%s  target='%s'\n",
    nrow(x$data), ncol(x$data), x$task, x$target
  ))
  invisible(x)
}
