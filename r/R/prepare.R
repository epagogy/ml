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
#' @return An `ml_prepared_data` object with:
#'   - `$data`   — transformed data.frame (all-numeric, ready for ml_fit)
#'   - `$state`  — NormState list; use .transform(state, X) on new data
#'   - `$target` — target column name
#'   - `$task`   — detected or provided task type
#'
#' @examples
#' \donttest{
#' df <- data.frame(x1 = rnorm(50), x2 = rnorm(50), y = rnorm(50))
#' s <- ml_split(df, "y", seed = 42)
#' p <- ml_prepare(s$train, "y")
#' p$task       # "classification" or "regression"
#' p$data       # encoded feature matrix
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

  # Partition guard — same as fit(): require split provenance, {train, valid, dev}
  if (.guards_active()) {
    .part <- .resolve_partition(data)
    if (is.null(.part)) {
      .guard_action(paste0(
        "ml_prepare() received data without split provenance. ",
        "Split your data first: s <- ml_split(df, target, seed = 42), ",
        "then ml_prepare(s$train, target). ",
        "To disable: ml_config(guards = 'off')"
      ))
    } else if (!.part %in% c("train", "valid", "dev")) {
      .guard_action(paste0(
        "ml_prepare() received data tagged as '", .part, "' partition. ",
        "ml_prepare() accepts train, valid, or dev data. ",
        "Use: ml_prepare(s$train, target)"
      ))
    }
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

#' @export
print.ml_prepared_data <- function(x, ...) {
  cat(sprintf(
    "PreparedData  %d rows x %d cols  task=%s  target='%s'\n",
    nrow(x$data), ncol(x$data), x$task, x$target
  ))
  invisible(x)
}
