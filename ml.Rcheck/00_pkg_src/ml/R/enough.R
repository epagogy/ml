#' Check if sample size is sufficient before training
#'
#' Assesses whether your training set is large enough for reliable estimates.
#' For classification: checks minimum samples per class. For regression:
#' checks overall sample count. Run this before ml_fit() to catch small-data
#' issues early.
#'
#' Note: This function checks minimum sample counts (30 per class for
#' classification, 50 overall for regression). It does NOT run learning curves.
#' The Python equivalent (`ml.enough()`) uses saturation detection on a learning
#' curve, which is more thorough but also more computationally expensive.
#'
#' @param s An `ml_split_result` from `ml_split()`
#' @param target Target column name
#' @param seed Ignored (for API parity with other ml functions)
#' @returns An `ml_enough_result` object (prints a summary)
#' @export
#' @examples
#' s <- ml_split(iris, "Species", seed = 42)
#' ml_enough(s, "Species")
ml_enough <- function(s, target, seed = NULL) {
  if (!inherits(s, "ml_split_result")) {
    data_error("s must be an ml_split_result from ml_split()")
  }

  train <- s$train
  n_train <- nrow(train)
  y <- train[[target]]
  task <- .detect_task(y, "auto")

  if (task == "classification") {
    counts  <- sort(table(y))
    min_n   <- as.integer(counts[[1L]])
    n_class <- length(counts)

    sufficient <- min_n >= 30L
    suggestion <- if (min_n < 10L) {
      paste0("Smallest class has only ", min_n, " samples -- too few to train reliably. Collect more data.")
    } else if (min_n < 30L) {
      paste0("Smallest class has ", min_n, " samples. Aim for >= 30 per class for stable estimates.")
    } else if (min_n < 100L) {
      paste0("Smallest class has ", min_n, " samples. Results are usable; more data would help.")
    } else {
      NULL
    }

    result <- structure(list(
      sufficient   = sufficient,
      task         = task,
      n_train      = n_train,
      n_classes    = n_class,
      min_class_n  = min_n,
      class_counts = as.list(counts),
      suggestion   = suggestion
    ), class = "ml_enough_result")

  } else {
    sufficient <- n_train >= 50L
    suggestion <- if (n_train < 30L) {
      paste0("Only ", n_train, " training samples. Consider cross-validation (ml_split folds=5).")
    } else if (n_train < 50L) {
      paste0(n_train, " training samples is low. Aim for >= 50 for reliable regression estimates.")
    } else if (n_train < 200L) {
      paste0(n_train, " training samples. Usable; more data would reduce variance.")
    } else {
      NULL
    }

    result <- structure(list(
      sufficient = sufficient,
      task       = task,
      n_train    = n_train,
      suggestion = suggestion
    ), class = "ml_enough_result")
  }

  result
}

#' Print a sample-size sufficiency result
#'
#' @param x An `ml_enough_result` object.
#' @param ... Additional arguments (ignored).
#' @returns The object `x`, invisibly.
#' @export
print.ml_enough_result <- function(x, ...) {
  status <- if (x$sufficient) "\u2713 sufficient" else "\u26a0 may not be enough"
  cat(sprintf("-- Sample size [%s] %s\n", x$task, status))
  cat(sprintf("   n_train: %d\n", x$n_train))
  if (x$task == "classification") {
    cat(sprintf("   classes: %d  (smallest: %d samples)\n", x$n_classes, x$min_class_n))
  }
  if (!is.null(x$suggestion)) {
    cat(sprintf("   suggestion: %s\n", x$suggestion))
  }
  invisible(x)
}
