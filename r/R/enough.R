#' Learning curve analysis -- do you need more data?
#'
#' Trains at increasing data sizes and reports train vs validation performance
#' at each step. Answers: is the model still learning (more data helps), or
#' saturated (more data unlikely to help)?
#'
#' @param s An `ml_split_result` from `ml_split()`
#' @param target Target column name
#' @param seed Random seed (optional in R; auto-generated if NULL)
#' @param algorithm Algorithm to use (default `"auto"`). Any algorithm
#'   supported by `ml_fit()`.
#' @param steps Integer >= 2. Number of data-size steps to evaluate, evenly
#'   spaced from ~10%% to 100%% of training data. Default 8.
#' @param cv Integer >= 2. Number of cross-validation folds for validation
#'   score at each step. Default 3.
#' @returns An `ml_enough_result` with fields:
#'   - `$saturated` -- logical, TRUE if curve plateaus (< 1%% gain in last half)
#'   - `$curve` -- data.frame: n_samples, train_score, val_score
#'   - `$metric` -- metric name used
#'   - `$n_current` -- total training rows in the full dataset
#'   - `$recommendation` -- human-readable action
#' @export
#' @examples
#' s <- ml_split(iris, "Species", seed = 42)
#' result <- ml_enough(s, "Species", seed = 42)
#' result$recommendation
ml_enough <- function(s, target, seed = NULL, algorithm = "auto",
                      steps = 8L, cv = 3L) {
  if (!inherits(s, "ml_split_result")) {
    data_error("s must be an ml_split_result from ml_split()")
  }

  steps <- as.integer(steps)
  cv    <- as.integer(cv)

  if (steps < 2L) config_error(paste0("steps must be >= 2, got ", steps, "."))
  if (cv < 2L)    config_error(paste0("cv must be >= 2, got ", cv, "."))

  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)

  train <- s$train
  n_total <- nrow(train)

  if (n_total < 50L) {
    data_error(paste0(
      "enough() requires at least 50 rows, got ", n_total, ". ",
      "Collect more data or use ml_validate() for small datasets."
    ))
  }

  y      <- train[[target]]
  task   <- .detect_task(y, "auto")
  metric <- if (task == "classification") "accuracy" else "r2"

  # Shuffle for reproducible random subsampling
  withr::local_seed(seed)
  perm <- sample.int(n_total)
  shuffled <- train[perm, , drop = FALSE]

  # Steps from ~10% to 100%, at least 20 samples minimum per step
  min_n     <- max(20L, n_total %/% (steps * 2L))
  ns        <- unique(as.integer(round(seq(min_n, n_total, length.out = steps))))

  records <- lapply(ns, function(n) {
    subset_data <- shuffled[seq_len(n), , drop = FALSE]

    # Train score: fit on subset, evaluate on same (in-sample, upper bound)
    train_score <- tryCatch({
      suppressWarnings({
        m    <- ml_fit(subset_data, target, algorithm = algorithm, seed = seed)
        eval <- ml_evaluate(m, subset_data)
        as.numeric(eval[[metric]])
      })
    }, error = function(e) NA_real_)

    # Validation score: manual k-fold on subset
    n_sub <- nrow(subset_data)
    k     <- min(cv, n_sub %/% 2L)
    if (k < 2L) {
      return(list(n_samples = n, train_score = round(train_score, 4L),
                  val_score = NA_real_))
    }

    # Stratified fold assignment for classification, random for regression
    y_sub <- subset_data[[target]]
    if (task == "classification") {
      fold_ids <- .stratified_fold_ids(y_sub, k)
    } else {
      fold_ids <- rep_len(sample.int(k), n_sub)
    }

    fold_scores <- vapply(seq_len(k), function(fold_k) {
      tr_idx  <- which(fold_ids != fold_k)
      val_idx <- which(fold_ids == fold_k)
      if (length(tr_idx) < 5L || length(val_idx) < 2L) return(NA_real_)
      tryCatch({
        suppressWarnings({
          fm   <- ml_fit(subset_data[tr_idx, , drop = FALSE], target,
                         algorithm = algorithm, seed = seed)
          fev  <- ml_evaluate(fm, subset_data[val_idx, , drop = FALSE])
          as.numeric(fev[[metric]])
        })
      }, error = function(e) NA_real_)
    }, numeric(1L))

    val_score <- mean(fold_scores, na.rm = TRUE)

    list(n_samples   = n,
         train_score = round(train_score, 4L),
         val_score   = round(val_score,   4L))
  })

  curve <- data.frame(
    n_samples   = vapply(records, `[[`, integer(1L),  "n_samples"),
    train_score = vapply(records, `[[`, numeric(1L), "train_score"),
    val_score   = vapply(records, `[[`, numeric(1L), "val_score")
  )

  # Saturation detection: val improvement in last half
  val_scores <- curve$val_score[!is.na(curve$val_score)]
  saturated  <- FALSE
  gain_str   <- ""

  if (length(val_scores) >= 4L) {
    mid            <- length(val_scores) %/% 2L
    first_half_max <- max(val_scores[seq_len(mid)],          na.rm = TRUE)
    second_half_max <- max(val_scores[seq(mid + 1L, length(val_scores))], na.rm = TRUE)
    gain     <- second_half_max - first_half_max
    gain_pct <- max(gain, 0.0) * 100
    gain_str <- sprintf("%.1f%%", gain_pct)
    saturated <- gain <= 0 || gain_pct < 1.0
  }

  last_val <- if (length(val_scores) > 0L) val_scores[length(val_scores)] else NA_real_
  last_n   <- curve$n_samples[nrow(curve)]

  recommendation <- if (saturated) {
    paste0("Model is saturated: ", metric, " improved < 1% adding more data. ",
           "Focus on feature engineering or a more powerful algorithm instead.")
  } else if (length(val_scores) < 4L) {
    "Insufficient data to determine saturation. Collect more labeled examples."
  } else {
    sprintf("Still learning: %s improved %s in last half of data. More data likely helps. Current %s=%.3f at n=%d.",
            metric, gain_str, metric, last_val, last_n)
  }

  structure(
    list(
      saturated      = saturated,
      curve          = curve,
      metric         = metric,
      n_current      = n_total,
      recommendation = recommendation
    ),
    class = "ml_enough_result"
  )
}

#' @export
print.ml_enough_result <- function(x, ...) {
  status <- if (x$saturated) "saturated" else "still learning"
  cat(sprintf("-- Learning curve [%s]\n", status))
  cat(sprintf("   metric: %s\n", x$metric))
  cat(sprintf("   n_current: %d\n", x$n_current))
  if (nrow(x$curve) > 0L) {
    last <- x$curve[nrow(x$curve), ]
    cat(sprintf("   final val_%s: %.4f  (train: %.4f)  at n=%d\n",
                x$metric, last$val_score, last$train_score, last$n_samples))
  }
  cat(sprintf("   recommendation: %s\n", x$recommendation))
  invisible(x)
}

# -- Internal helpers ----------------------------------------------------------

.stratified_fold_ids <- function(y, k) {
  y_char  <- as.character(y)
  classes <- unique(y_char)
  ids     <- integer(length(y_char))
  for (cl in classes) {
    idx        <- which(y_char == cl)
    ids[idx]   <- rep_len(seq_len(k), length(idx))
  }
  ids
}
