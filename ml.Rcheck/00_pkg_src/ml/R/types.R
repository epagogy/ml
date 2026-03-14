# ── ml_model ──────────────────────────────────────────────────────────────────
#
# S3 list with all immutable fields at top level.
# Mutable state (assess_count) stored in .mutable environment (reference semantics).
# No R6 required -- plain list + environment gives the same guarantees.

#' @keywords internal
new_ml_model <- function(engine, task, algorithm, target, features, seed,
                          scores_ = NULL, fold_scores_ = NULL,
                          preprocessing_ = NULL, n_train = 0L, time = 0,
                          hash = "", encoders = NULL, is_stacked = FALSE) {
  mutable <- new.env(parent = emptyenv())
  mutable$assess_count <- 0L

  structure(
    list(
      engine         = engine,
      task           = task,
      algorithm      = algorithm,
      target         = target,
      features       = features,
      seed           = seed,
      scores_        = scores_,
      fold_scores_   = fold_scores_,
      preprocessing_ = preprocessing_,
      n_train        = n_train,
      time           = time,
      hash           = hash,
      encoders       = encoders,
      is_stacked     = is_stacked,
      .mutable       = mutable
    ),
    class = "ml_model"
  )
}

# Helper to access and set assess_count (mutable reference)
.get_assess_count <- function(model) model[[".mutable"]][["assess_count"]]
.inc_assess_count <- function(model) {
  model[[".mutable"]][["assess_count"]] <- model[[".mutable"]][["assess_count"]] + 1L
  invisible(model)
}

#' Print an ml_model
#' @param x An ml_model object
#' @param ... Ignored
#' @returns The object \code{x}, invisibly.
#' @export
print.ml_model <- function(x, ...) {
  cat(sprintf("-- Model [%s * %s] --\n", x[["algorithm"]], x[["task"]]))
  cat(sprintf("  features : %d\n", length(x[["features"]])))
  cat(sprintf("  rows     : %d\n", x[["n_train"]]))
  cat(sprintf("  target   : %s\n", x[["target"]]))
  cat(sprintf("  seed     : %d\n", x[["seed"]]))
  cat(sprintf("  hash     : %s\n", x[["hash"]]))
  if (!is.null(x[["scores_"]])) {
    primary <- names(x[["scores_"]])[[1]]
    cat(sprintf("  cv_%s : %.4f\n", primary, x[["scores_"]][[primary]]))
  }
  cat(sprintf("  (%.2fs)\n", x[["time"]]))
  cat("\n")
  invisible(x)
}


# ── ml_split_result ────────────────────────────────────────────────────────────

#' @keywords internal
new_ml_split_result <- function(train, valid, test) {
  attr(train, "_ml_partition") <- "train"
  attr(valid, "_ml_partition") <- "valid"
  attr(test,  "_ml_partition") <- "test"
  structure(
    list(train = train, valid = valid, test = test),
    class = "ml_split_result"
  )
}

#' Access elements of a split result
#'
#' Provides `$train`, `$valid`, `$test` partitions, plus a computed `$dev`
#' partition (train + valid combined).
#'
#' @param x An `ml_split_result` object.
#' @param name Element name: `"train"`, `"valid"`, `"test"`, or `"dev"`.
#' @returns A data.frame partition. `$dev` is computed on-the-fly as
#'   `rbind(train, valid)`.
#' @export
`$.ml_split_result` <- function(x, name) {
  if (identical(name, "dev")) {
    result <- rbind(.subset2(x, "train"), .subset2(x, "valid"))
    attr(result, "_ml_partition") <- "dev"
    return(result)
  }
  .subset2(x, name)
}

#' Print an ml_split_result
#' @param x An ml_split_result object
#' @param ... Ignored
#' @returns The object \code{x}, invisibly.
#' @export
print.ml_split_result <- function(x, ...) {
  n_train <- nrow(x[["train"]])
  n_valid <- nrow(x[["valid"]])
  n_test  <- nrow(x[["test"]])
  n_total <- n_train + n_valid + n_test
  cat("-- Split --\n")
  cat(sprintf("  train : %d rows (%.0f%%)\n", n_train, 100 * n_train / n_total))
  cat(sprintf("  valid : %d rows (%.0f%%)\n", n_valid, 100 * n_valid / n_total))
  cat(sprintf("  test  : %d rows (%.0f%%)\n", n_test,  100 * n_test  / n_total))
  cat(sprintf("  dev   : %d rows (train + valid combined)\n", n_train + n_valid))
  cat("\n")
  invisible(x)
}

# ── ml_cv_result ───────────────────────────────────────────────────────────────

#' @keywords internal
new_ml_cv_result <- function(folds, data, target) {
  # k = number of folds -- matches Python CVResult.k attribute
  structure(
    list(folds = folds, data = data, target = target, k = length(folds)),
    class = "ml_cv_result"
  )
}

#' Access elements of a CV result
#'
#' Accessing `$train`, `$valid`, `$test`, or `$dev` raises an informative error
#' because CV results have no single validation set.
#'
#' @param x An `ml_cv_result` object.
#' @param name Element name to access.
#' @returns The requested element, or raises an error for partition names.
#' @export
`$.ml_cv_result` <- function(x, name) {
  if (name %in% c("train", "valid", "test", "dev")) {
    config_error(paste0(
      "No validation set -- this is a CVResult. ",
      "Use model$scores_ for CV metrics, or split without folds= for train/valid/test."
    ))
  }
  .subset2(x, name)
}

#' Print an ml_cv_result
#' @param x An ml_cv_result object
#' @param ... Ignored
#' @returns The object \code{x}, invisibly.
#' @export
print.ml_cv_result <- function(x, ...) {
  k <- length(x[["folds"]])
  n <- nrow(x[["data"]])
  cat(sprintf("-- CV Split [%d folds, %d rows] --\n", k, n))
  cat("  Use ml_fit(cv_result, target) for cross-validated training.\n")
  cat("\n")
  invisible(x)
}

# ── ml_tuning_result ───────────────────────────────────────────────────────────

#' @keywords internal
new_ml_tuning_result <- function(best_params, history, best_model) {
  structure(
    list(
      best_params_    = best_params,
      tuning_history_ = history,
      best_model      = best_model
    ),
    class = "ml_tuning_result"
  )
}

#' Predict from best model in a tuning result
#' @param object An ml_tuning_result
#' @param newdata A data.frame
#' @param ... Passed to predict.ml_model
#' @returns Predictions
#' @export
predict.ml_tuning_result <- function(object, newdata, ...) {
  predict(object[["best_model"]], newdata = newdata, ...)
}

#' Print an ml_tuning_result
#' @param x An ml_tuning_result object
#' @param ... Ignored
#' @returns The object \code{x}, invisibly.
#' @export
print.ml_tuning_result <- function(x, ...) {
  bm <- x[["best_model"]]
  cat(sprintf("-- Tuned [%s * %s] --\n", bm[["algorithm"]], bm[["task"]]))
  cat(sprintf("  trials : %d\n", nrow(x[["tuning_history_"]])))
  params <- x[["best_params_"]]
  if (length(params) > 0) {
    for (nm in names(params)) {
      cat(sprintf("  %-18s %s\n", nm, params[[nm]]))
    }
  }
  cat("\n")
  invisible(x)
}

# ── ml_metrics ─────────────────────────────────────────────────────────────────

#' @keywords internal
new_ml_metrics <- function(values, task, time = NULL) {
  structure(values, class = "ml_metrics", task = task, time = time)
}

#' Print ml_metrics
#' @param x An ml_metrics object
#' @param ... Ignored
#' @returns The object \code{x}, invisibly.
#' @export
print.ml_metrics <- function(x, ...) {
  task   <- attr(x, "task")
  time_s <- attr(x, "time")
  vals   <- unclass(x)
  nms    <- names(vals)
  cat(sprintf("-- Metrics [%s] --\n", task))
  if (length(nms) <= 6L) {
    for (nm in nms) cat(sprintf("  %-22s %.4f\n", paste0(nm, ":"), vals[[nm]]))
  } else {
    half  <- ceiling(length(nms) / 2)
    left  <- nms[seq_len(half)]
    right <- nms[seq(half + 1L, length(nms))]
    for (i in seq_along(left)) {
      rn <- if (i <= length(right)) right[[i]] else ""
      rv <- if (i <= length(right)) sprintf("%.4f", vals[[right[[i]]]]) else ""
      cat(sprintf("  %-22s %.4f   %-22s %s\n",
                  paste0(left[[i]], ":"), vals[[left[[i]]]], paste0(rn, ":"), rv))
    }
  }
  if (!is.null(time_s)) cat(sprintf("  (%.2fs)\n", time_s))
  cat("\n")
  invisible(x)
}

# ── ml_evidence ────────────────────────────────────────────────────────────────

#' @keywords internal
new_ml_evidence <- function(values, task, time = NULL) {
  # Sealed terminal type -- not substitutable for ml_metrics (Codd condition 7).
  # inherits(x, "ml_metrics") returns FALSE by design.
  structure(values, class = "ml_evidence", task = task, time = time)
}

#' Print ml_evidence
#' @param x An ml_evidence object
#' @param ... Ignored
#' @returns The object \code{x}, invisibly.
#' @export
print.ml_evidence <- function(x, ...) {
  task   <- attr(x, "task")
  time_s <- attr(x, "time")
  vals   <- unclass(x)
  nms    <- names(vals)
  cat(sprintf("-- Evidence [%s] --\n", task))
  for (nm in nms) cat(sprintf("  %-22s %.4f\n", paste0(nm, ":"), vals[[nm]]))
  if (!is.null(time_s)) cat(sprintf("  (%.2fs)\n", time_s))
  cat("\n")
  invisible(x)
}

# ── ml_explanation ─────────────────────────────────────────────────────────────

#' @keywords internal
new_ml_explanation <- function(df, algorithm) {
  structure(df, class = c("ml_explanation", "data.frame"), algorithm = algorithm)
}

#' Print ml_explanation
#' @param x An ml_explanation object
#' @param ... Ignored
#' @returns The object \code{x}, invisibly.
#' @export
print.ml_explanation <- function(x, ...) {
  algo    <- attr(x, "algorithm")
  max_bar <- 30L
  df      <- as.data.frame(x)
  top_n   <- min(nrow(df), 15L)
  cat(sprintf("-- Explain [%s] --\n", algo))
  for (i in seq_len(top_n)) {
    bar_len <- round(df$importance[[i]] * max_bar)
    bar     <- paste(rep("\u2588", bar_len), collapse = "")
    cat(sprintf("  %-20s %s %.3f\n", df$feature[[i]], bar, df$importance[[i]]))
  }
  cat("\n")
  invisible(x)
}

# ── ml_leaderboard ─────────────────────────────────────────────────────────────

#' @keywords internal
new_ml_leaderboard <- function(df, context = "", models = NULL) {
  structure(df, class = c("ml_leaderboard", "data.frame"),
            lb_context = context, lb_models = models)
}

#' Get the best model from a leaderboard
#'
#' Returns the top-ranked fitted model from screen() or compare().
#' NULL if no models were stored.
#'
#' @param lb An ml_leaderboard
#' @returns An ml_model or NULL
#' @export
#' @examples
#' \donttest{
#' s <- ml_split(iris, "Species", seed = 42)
#' lb <- ml_screen(s, "Species", seed = 42)
#' best <- ml_best(lb)
#' predict(best, s$valid)
#' }
ml_best <- function(lb) {
  if (!inherits(lb, "ml_leaderboard")) {
    config_error("ml_best() requires an ml_leaderboard from ml_screen() or ml_compare().")
  }
  models <- attr(lb, "lb_models")
  if (is.null(models) || length(models) == 0L) return(NULL)
  for (m in models) {
    if (!is.null(m)) return(m)
  }
  NULL
}

#' Print ml_leaderboard
#' @param x An ml_leaderboard object
#' @param ... Ignored
#' @returns The object \code{x}, invisibly.
#' @export
print.ml_leaderboard <- function(x, ...) {
  ctx <- attr(x, "lb_context")
  cat(sprintf("-- Leaderboard [%s] --\n", ctx))
  df <- as.data.frame(x)
  print.data.frame(df, digits = 4L, row.names = FALSE)
  cat("\n")
  invisible(x)
}

# ── ml_profile_result ──────────────────────────────────────────────────────────

#' @keywords internal
new_ml_profile_result <- function(shape, target, task, columns, warnings,
                                   condition_number = NULL) {
  # Expose n_rows / n_cols as top-level keys -- matches Python profile() dict keys
  structure(
    list(
      shape            = shape,
      n_rows           = shape[[1L]],
      n_cols           = shape[[2L]],
      target           = target,
      task             = task,
      columns          = columns,
      warnings         = warnings,
      condition_number = condition_number
    ),
    class = "ml_profile_result"
  )
}

#' Print ml_profile_result
#' @param x An ml_profile_result object
#' @param ... Ignored
#' @returns The object \code{x}, invisibly.
#' @export
print.ml_profile_result <- function(x, ...) {
  cat(sprintf("-- Profile [%s] --\n", x[["task"]]))
  cat(sprintf("  rows    : %d\n", x[["shape"]][[1]]))
  cat(sprintf("  columns : %d\n", x[["shape"]][[2]]))
  if (!is.null(x[["target"]])) cat(sprintf("  target  : %s\n", x[["target"]]))
  if (length(x[["warnings"]]) > 0L) {
    cat("  warnings:\n")
    for (w in x[["warnings"]]) cat(sprintf("    ! %s\n", w))
  }
  cat("\n")
  invisible(x)
}

# ── ml_validate_result ─────────────────────────────────────────────────────────

#' @keywords internal
new_ml_validate_result <- function(passed, metrics, failures, baseline_metrics,
                                    improvements, degradations) {
  structure(
    list(
      passed           = passed,
      metrics          = metrics,
      failures         = failures,
      baseline_metrics = baseline_metrics,
      improvements     = improvements,
      degradations     = degradations
    ),
    class = "ml_validate_result"
  )
}

#' Print ml_validate_result
#' @param x An ml_validate_result object
#' @param ... Ignored
#' @returns The object \code{x}, invisibly.
#' @export
print.ml_validate_result <- function(x, ...) {
  status <- if (x[["passed"]]) "PASSED" else "FAILED"
  cat(sprintf("-- Validate [%s] --\n", status))
  if (length(x[["failures"]]) > 0L) {
    cat("  failures:\n")
    for (f in x[["failures"]]) cat(sprintf("    x %s\n", f))
  }
  if (length(x[["improvements"]]) > 0L) {
    cat("  improvements:\n")
    for (i in x[["improvements"]]) cat(sprintf("    + %s\n", i))
  }
  if (length(x[["degradations"]]) > 0L) {
    cat("  degradations:\n")
    for (d in x[["degradations"]]) cat(sprintf("    - %s\n", d))
  }
  cat("\n")
  invisible(x)
}
