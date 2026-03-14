#' Compare pre-fitted models on the same data
#'
#' Evaluates multiple fitted models on the same dataset without re-fitting.
#' All models must share the same target column and task.
#'
#' @param models A list of `ml_model` objects (or `ml_tuning_result`, auto-unwrapped)
#' @param data A data.frame containing the target column
#' @param sort_by "auto" or a metric name string
#' @returns An object of class `ml_leaderboard` (data.frame with formatted print)
#' @export
#' @examples
#' \donttest{
#' s <- ml_split(iris, "Species", seed = 42)
#' m1 <- ml_fit(s$train, "Species", algorithm = "logistic", seed = 42)
#' m2 <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42)
#' ml_compare(list(m1, m2), s$valid)
#' }
ml_compare <- function(models, data, sort_by = "auto") {
  .compare_impl(models = models, data = data, sort_by = sort_by)
}

.compare_impl <- function(models, data, sort_by = "auto") {
  if (!is.list(models) || length(models) == 0L) {
    config_error("models must be a non-empty list of ml_model objects")
  }

  # Auto-unwrap TuningResult → best_model
  models <- lapply(seq_along(models), function(i) {
    m <- models[[i]]
    if (inherits(m, "ml_tuning_result")) return(m[["best_model"]])
    if (inherits(m, "ml_model"))         return(m)
    config_error(paste0(
      "Expected ml_model or ml_tuning_result at index ", i,
      ", got: ", class(m)[[1]]
    ))
  })

  # Task/target consistency validation
  task   <- models[[1]][["task"]]
  target <- models[[1]][["target"]]

  for (i in seq_along(models)[-1]) {
    if (models[[i]][["task"]] != task) {
      config_error(paste0(
        "compare() item ", i, " is ", models[[i]][["task"]],
        " but item 1 is ", task,
        ". Cannot compare classifiers with regressors."
      ))
    }
    if (models[[i]][["target"]] != target) {
      config_error(paste0(
        "compare() item ", i, " has target='", models[[i]][["target"]],
        "' but item 1 has target='", target,
        "'. All models must share the same target."
      ))
    }
  }

  data <- .coerce_data(data)

  # Partition guard -- warn (not error) when test-tagged data is used
  if (.guards_active()) {
    .part <- attr(data, "_ml_partition")
    if (!is.null(.part) && .part == "test") {
      warning(
        "ml_compare() received test-tagged data. Comparing models on test data ",
        "constitutes implicit model selection on the test set. ",
        "Use validation data: ml_compare(models, s$valid). ",
        "Reserve test data for ml_assess().",
        call. = FALSE
      )
    }
  }

  rows <- list()
  for (model in models) {
    t0 <- proc.time()[["elapsed"]]
    result <- tryCatch({
      metrics <- .evaluate_impl(model, data, .guard = FALSE)
      list(success = TRUE, metrics = metrics, time = proc.time()[["elapsed"]] - t0)
    }, error = function(e) {
      list(success = FALSE, error = substr(conditionMessage(e), 1L, 120L),
           time = proc.time()[["elapsed"]] - t0)
    })

    if (!result$success) next

    row <- c(
      list(algorithm = model[["algorithm"]]),
      as.list(unclass(result$metrics)),
      list(time = round(result$time, 3))
    )
    rows <- c(rows, list(row))
  }

  if (length(rows) == 0L) {
    return(new_ml_leaderboard(
      data.frame(algorithm = character(0), stringsAsFactors = FALSE),
      context = paste(task, "* 0")
    ))
  }

  # Build leaderboard data.frame
  all_cols <- unique(unlist(lapply(rows, names)))
  lb_df <- as.data.frame(do.call(rbind, lapply(rows, function(r) {
    row_list <- lapply(all_cols, function(col) {
      val <- r[[col]]
      if (is.null(val)) NA_real_ else val
    })
    names(row_list) <- all_cols
    row_list
  })), stringsAsFactors = FALSE)

  for (col in setdiff(names(lb_df), "algorithm")) {
    lb_df[[col]] <- as.numeric(lb_df[[col]])
  }

  # Sort
  sort_metric <- if (sort_by == "auto") {
    if (task == "regression") "rmse" else "roc_auc"
  } else {
    sort_by
  }

  if (sort_metric %in% names(lb_df)) {
    desc_metrics <- c("accuracy", "f1", "f1_macro", "f1_weighted", "roc_auc",
                      "precision", "recall", "r2")
    decreasing <- sort_metric %in% desc_metrics
    lb_df <- lb_df[order(lb_df[[sort_metric]], decreasing = decreasing,
                         na.last = TRUE), , drop = FALSE]
  }
  rownames(lb_df) <- NULL

  context <- paste("Compare:", task, "*", nrow(lb_df))
  new_ml_leaderboard(lb_df, context = context)
}
