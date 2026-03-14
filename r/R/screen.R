#' Screen all algorithms on your data
#'
#' Fits every available algorithm on the training data and ranks by validation
#' performance. Use this to identify promising candidates before tuning.
#'
#' **Multiple comparison bias:** Selecting the best from N algorithms on the
#' same validation set produces optimistic estimates. The winning algorithm
#' benefits from selection bias. Use [ml_validate()] on held-out test data
#' for trustworthy comparisons.
#'
#' For imbalanced data, consider `sort_by = "f1"` — the default `roc_auc`
#' can hide failures on minority classes.
#'
#' @param data An `ml_split_result` (NOT a raw data.frame —
#'   split first to prevent overfitting)
#' @param target Target column name
#' @param algorithms Character vector of algorithm names, or NULL for all available
#' @param seed Random seed. NULL auto-generates.
#' @param sort_by "auto" (roc_auc for binary clf, f1_macro for multiclass,
#'   rmse for regression), or a metric name string
#' @param time_budget Maximum seconds for entire screen. Stops between
#'   algorithms (not mid-fit) when budget exceeded. NULL (default) = no limit.
#' @param keep_models If FALSE, discard fitted models after scoring to save
#'   memory. ml_best() will return NULL. Default TRUE.
#' @param ... Additional arguments passed to [ml_fit()]
#' @returns An object of class `ml_leaderboard` (data.frame with formatted print)
#' @export
#' @examples
#' \donttest{
#' s <- ml_split(iris, "Species", seed = 42)
#' lb <- ml_screen(s, "Species", seed = 42)
#' lb
#' }
ml_screen <- function(data, target, algorithms = NULL, seed = NULL,
                      sort_by = "auto", time_budget = NULL,
                      keep_models = TRUE, ...) {
  .screen_impl(data = data, target = target, algorithms = algorithms,
               seed = seed, sort_by = sort_by, time_budget = time_budget,
               keep_models = keep_models, ...)
}

.screen_impl <- function(data, target, algorithms = NULL, seed = NULL,
                         sort_by = "auto", time_budget = NULL,
                         keep_models = TRUE, ...) {
  # Validate: must be split result (forces split-first workflow)
  if (!inherits(data, "ml_split_result")) {
    config_error(paste0(
      "ml_screen() requires a split result, not raw data.\n",
      "  Quick fix: s <- ml_split(data, '", target, "', seed = 42)\n",
      "             lb <- ml_screen(s, '", target, "')\n",
      "  Why: screening on validation data prevents overfitting."
    ))
  }

  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)

  # Detect task BEFORE fitting any models
  src  <- data[["train"]]
  task <- .detect_task(src[[target]])

  # Filter algorithms by task compatibility
  all_algos <- if (is.null(algorithms)) .available_algorithms() else algorithms
  algos <- Filter(function(a) {
    if (task == "classification" && a %in% c("linear")) return(FALSE)
    if (task == "regression"     && a %in% c("logistic", "naive_bayes")) return(FALSE)
    TRUE
  }, all_algos)

  # Get validation data
  if (!is.null(data[["folds"]])) {
    # CV mode: use first fold for quick screening
    fdata <- data[[".folds_data"]]
    train_data <- fdata[data[["folds"]][[1]]$train, , drop = FALSE]
    eval_data  <- fdata[data[["folds"]][[1]]$valid, , drop = FALSE]
  } else {
    train_data <- data[["train"]]
    eval_data  <- data[["valid"]]
  }

  # Determine sort metric
  sort_metric <- if (sort_by == "auto") {
    if (task == "regression")                              "rmse"
    else if (length(unique(src[[target]])) > 2L)          "f1_macro"
    else                                                   "roc_auc"
  } else {
    sort_by
  }

  rows   <- list()
  models <- list()
  screen_start <- proc.time()[["elapsed"]]
  for (algo in algos) {
    # Time budget: stop between algorithms
    if (!is.null(time_budget) && (proc.time()[["elapsed"]] - screen_start) > time_budget) {
      break
    }
    t0 <- proc.time()[["elapsed"]]
    result <- tryCatch({
      model   <- .fit_impl(data = train_data, target = target,
                           algorithm = algo, seed = seed, ...)
      metrics <- .score_impl(model, eval_data)
      list(success = TRUE, model = model, metrics = metrics,
           time = proc.time()[["elapsed"]] - t0)
    }, error = function(e) {
      list(success = FALSE, error = substr(conditionMessage(e), 1L, 120L),
           time = proc.time()[["elapsed"]] - t0)
    })

    if (!result$success) next

    row <- c(
      list(algorithm = algo),
      as.list(unclass(result$metrics)),
      list(time = round(result$time, 3))
    )
    rows   <- c(rows, list(row))
    models <- c(models, list(if (keep_models) result$model else NULL))
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

  # Convert columns to correct types
  lb_df[["algorithm"]] <- as.character(lb_df[["algorithm"]])
  for (col in setdiff(names(lb_df), "algorithm")) {
    lb_df[[col]] <- as.numeric(lb_df[[col]])
  }

  # Sort (reorder models in sync)
  if (sort_metric %in% names(lb_df)) {
    desc_metrics <- c("accuracy", "f1", "f1_macro", "f1_weighted", "roc_auc",
                      "precision", "recall", "r2", "roc_auc_ovr")
    decreasing <- sort_metric %in% desc_metrics
    sort_idx <- order(lb_df[[sort_metric]], decreasing = decreasing, na.last = TRUE)
    lb_df  <- lb_df[sort_idx, , drop = FALSE]
    models <- models[sort_idx]
  }
  rownames(lb_df) <- NULL

  context <- paste(task, "*", nrow(lb_df))
  new_ml_leaderboard(lb_df, context = context, models = models)
}
