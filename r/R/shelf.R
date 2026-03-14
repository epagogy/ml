#' Check if a model is past its shelf life
#'
#' Evaluates the model on new labeled data and compares performance to the
#' model's original training metrics. Requires ground truth labels.
#'
#' Run this when outcome labels become available (e.g., daily/weekly batch
#' scoring, then wait for outcomes). Pair with [ml_drift()] for complete
#' monitoring:
#' - [ml_drift()]: input distribution shift (label-free, run always)
#' - [ml_shelf()]: performance degradation (needs labels, run periodically)
#'
#' **Requires** `model$scores_` from a cross-validated fit. If the model was
#' trained on a holdout split (no CV), `scores_` will be NULL and shelf()
#' raises a model_error.
#'
#' @param model An `ml_model` or `ml_tuning_result` with `$scores_` populated
#'   (i.e., trained with `ml_fit(cv_result, target)`)
#' @param new A data.frame — new labeled dataset including the target column
#' @param target Name of the target column in `new`
#' @param tolerance Allowed degradation per metric (default 0.05 = 5pp).
#'   Any key metric degrading beyond tolerance marks the model as stale.
#' @returns An object of class `ml_shelf_result` with:
#'   - `$fresh`: TRUE if model performance is within tolerance
#'   - `$stale`: inverse of fresh
#'   - `$metrics_then`: original training metrics (from model$scores_)
#'   - `$metrics_now`: current metrics on new data
#'   - `$degradation`: per-metric delta (negative = worse for higher-is-better)
#'   - `$recommendation`: human-readable guidance
#' @export
#' @examples
#' \donttest{
#' cv    <- ml_split(iris, "Species", seed = 42, folds = 3)
#' model <- ml_fit(cv, "Species", algorithm = "logistic", seed = 42)
#' # Simulate a new labeled batch
#' new_batch <- iris[sample(nrow(iris), 30), ]
#' result <- ml_shelf(model, new = new_batch, target = "Species")
#' result$fresh
#' result$degradation
#' }
ml_shelf <- function(model, new, target, tolerance = 0.05) {
  .shelf_impl(model = model, new = new, target = target, tolerance = tolerance)
}

# Lower-is-better metrics (same list as validate.R)
.SHELF_LOWER_IS_BETTER <- c("rmse", "mae", "mse", "log_loss")

.shelf_impl <- function(model, new, target, tolerance = 0.05) {
  # Auto-unwrap TuningResult
  if (inherits(model, "ml_tuning_result")) model <- model[["best_model"]]
  if (!inherits(model, "ml_model")) model_error("Expected an ml_model or ml_tuning_result")

  # Validate: must have CV scores_ to compare against
  scores_then <- model[["scores_"]]
  if (is.null(scores_then) || length(scores_then) == 0L) {
    model_error(paste0(
      "model has no training metrics (scores_ is NULL). ",
      "shelf() requires a cross-validated model. ",
      "Refit with: cv <- ml_split(data, target, folds=5); ml_fit(cv, target)"
    ))
  }

  # Validate new data
  new <- .coerce_data(new)
  if (nrow(new) < 5L) {
    data_error("new must have at least 5 rows for reliable shelf check")
  }
  if (!target %in% names(new)) {
    data_error(paste0("target='", target, "' not found in new data"))
  }

  # Target must match model's target
  if (model[["target"]] != target) {
    config_error(paste0(
      "target='", target, "' does not match model target='", model[["target"]], "'. ",
      "Both must be the same column."
    ))
  }

  # Evaluate model on new data
  metrics_now_obj <- .score_impl(model, new)
  metrics_now     <- as.list(unclass(metrics_now_obj))

  # Compute degradation for shared metrics
  # degradation = how much WORSE it got (positive = problem)
  shared_metrics <- intersect(names(scores_then), names(metrics_now))

  degradation  <- list()
  worse_count  <- 0L

  for (nm in shared_metrics) {
    then_val <- scores_then[[nm]]
    now_val  <- metrics_now[[nm]]
    if (is.null(then_val) || is.null(now_val)) next
    if (is.na(then_val)  || is.na(now_val))  next

    delta <- now_val - then_val  # positive = improved for higher-is-better

    # Direction-aware degradation
    if (nm %in% .SHELF_LOWER_IS_BETTER) {
      # rmse/mae: increase is bad
      deg <- now_val - then_val  # positive = worse
    } else {
      # accuracy/f1/r2: decrease is bad
      deg <- then_val - now_val  # positive = worse
    }

    degradation[[nm]] <- delta  # store raw delta (negative = worse for HIB)

    if (deg > tolerance) {
      worse_count <- worse_count + 1L
    }
  }

  fresh <- worse_count == 0L

  # Human-readable recommendation
  recommendation <- .shelf_recommendation(fresh, worse_count, degradation, tolerance)

  new_ml_shelf_result(
    fresh        = fresh,
    metrics_then = scores_then,
    metrics_now  = metrics_now,
    degradation  = degradation,
    recommendation = recommendation,
    n_new        = nrow(new),
    tolerance    = tolerance
  )
}

.shelf_recommendation <- function(fresh, worse_count, degradation, tolerance) {
  if (fresh) {
    max_deg <- if (length(degradation) > 0L) {
      max(vapply(degradation, function(d) if (is.numeric(d)) abs(d) else 0, numeric(1L)))
    } else 0
    if (max_deg < tolerance / 2) {
      "Model is fresh. Performance is stable within tolerance."
    } else {
      "Model is fresh. Minor degradation within tolerance -- monitor closely."
    }
  } else {
    paste0(
      "Model is stale. ", worse_count, " metric(s) degraded beyond tolerance (",
      tolerance, "). Consider retraining on recent data."
    )
  }
}

# ── S3 type ────────────────────────────────────────────────────────────────────

#' @keywords internal
new_ml_shelf_result <- function(fresh, metrics_then, metrics_now, degradation,
                                 recommendation, n_new, tolerance) {
  structure(
    list(
      fresh          = fresh,
      stale          = !fresh,
      metrics_then   = metrics_then,
      metrics_now    = metrics_now,
      degradation    = degradation,
      recommendation = recommendation,
      n_new          = n_new,
      tolerance      = tolerance
    ),
    class = "ml_shelf_result"
  )
}

#' Print ml_shelf_result
#' @param x An ml_shelf_result object
#' @param ... Ignored
#' @returns The object \code{x}, invisibly.
#' @export
print.ml_shelf_result <- function(x, ...) {
  status <- if (x[["fresh"]]) "FRESH" else "STALE"
  cat(sprintf("-- Shelf [%s] --\n", status))
  cat(sprintf("  n_new     : %d rows\n", x[["n_new"]]))
  cat(sprintf("  tolerance : %.3f\n", x[["tolerance"]]))

  metrics_then <- x[["metrics_then"]]
  metrics_now  <- x[["metrics_now"]]
  shared       <- intersect(names(metrics_then), names(metrics_now))

  if (length(shared) > 0L) {
    cat("  metrics:\n")
    for (nm in shared) {
      then_v <- metrics_then[[nm]]
      now_v  <- metrics_now[[nm]]
      delta  <- x[["degradation"]][[nm]]
      if (!is.null(delta)) {
        arrow <- if (delta >= 0) "+" else ""
        cat(sprintf("    %-20s %.4f -> %.4f (%s%.4f)\n",
                    nm, then_v, now_v, arrow, delta))
      }
    }
  }

  cat(sprintf("  %s\n", x[["recommendation"]]))
  cat("\n")
  invisible(x)
}
