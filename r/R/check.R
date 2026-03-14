#' Verify bitwise reproducibility for a given dataset
#'
#' Fits the same model twice with the same seed and asserts predictions are
#' identical. Returns a list with \code{passed}, \code{algorithm}, \code{seed},
#' and \code{message}.
#'
#' @param data A data.frame with features and target
#' @param target Target column name
#' @param algorithm Algorithm to check (default "random_forest")
#' @param seed Random seed
#' @returns A list with \code{passed} (logical), \code{algorithm}, \code{seed},
#'   \code{message}. Supports \code{isTRUE(result$passed)} for assertions.
#' @export
#' @examples
#' result <- ml_check(iris, "Species", seed = 42)
#' result$passed
ml_check <- function(data, target, algorithm = "random_forest", seed) {
  s  <- ml_split(data, target, seed = seed)
  m1 <- ml_fit(data = s$train, target = target, algorithm = algorithm, seed = seed)
  m2 <- ml_fit(data = s$train, target = target, algorithm = algorithm, seed = seed)

  p1 <- predict(m1, newdata = s$valid)
  p2 <- predict(m2, newdata = s$valid)

  if (!identical(p1, p2)) {
    model_error(paste0(
      "ml_check() found non-reproducible predictions for algorithm='", algorithm,
      "', seed=", seed, ". This model is not deterministic. ",
      "Ensure seed is set correctly and the algorithm supports determinism."
    ))
  }

  list(
    passed    = TRUE,
    algorithm = algorithm,
    seed      = seed,
    message   = paste0(
      "Predictions are bitwise identical for algorithm='", algorithm,
      "', seed=", seed, "."
    )
  )
}

#' Pre-flight data quality checks
#'
#' Runs before fit() to catch common data quality issues that silently
#' degrade model performance.
#'
#' Checks performed:
#' \itemize{
#'   \item NaN in target (silently dropped by split)
#'   \item Inf in features
#'   \item ID columns (100\% unique integer/string columns)
#'   \item Zero-variance features (constant columns)
#'   \item High-null columns (>50\% missing)
#'   \item Severe class imbalance (<5\% minority, classification only)
#'   \item Duplicate rows (>10\% of data)
#'   \item Feature redundancy (|r| > 0.95)
#' }
#'
#' @param data A data.frame
#' @param target Target column name
#' @param severity "warn" (default) or "error". If "error", raises on any issue.
#' @returns A list with \code{warnings}, \code{errors}, \code{has_issues},
#'   \code{passed}. Supports \code{isTRUE(result$passed)} for assertions.
#' @export
#' @examples
#' report <- ml_check_data(iris, "Species")
#' report$passed
ml_check_data <- function(data, target, severity = "warn") {
  data <- .coerce_data(data)

  if (!target %in% names(data)) {
    data_error(paste0(
      "target='", target, "' not found in data. ",
      "Available columns: ", paste(names(data), collapse = ", ")
    ))
  }

  # Check duplicate column names
  dupes <- names(data)[duplicated(names(data))]
  if (length(dupes) > 0L) {
    data_error(paste0(
      "Duplicate column names found: ", paste(dupes, collapse = ", "), ". ",
      "Rename columns before calling ml_check_data()."
    ))
  }

  warns  <- character(0)
  errors <- character(0)

  y <- data[[target]]
  X <- data[, setdiff(names(data), target), drop = FALSE]

  # 0. NaN in target
  n_na <- sum(is.na(y))
  if (n_na > 0L) {
    warns <- c(warns, paste0(
      "Target '", target, "' has ", n_na, " NA value(s) (",
      round(100 * n_na / nrow(data)), "% of rows). ",
      "These rows are silently dropped by ml_split(). Remove them first."
    ))
  }

  # 0b. Inf in features
  for (col in names(X)) {
    if (is.numeric(X[[col]])) {
      n_inf <- sum(is.infinite(X[[col]]))
      if (n_inf > 0L) {
        warns <- c(warns, paste0(
          "Column '", col, "' has ", n_inf, " infinite value(s). ",
          "Most algorithms raise errors on Inf. Replace with NA or clip."
        ))
      }
    }
  }

  # 1. ID columns: 100% unique int/char columns
  for (col in names(X)) {
    if (nrow(X) > 10L && length(unique(X[[col]])) == nrow(X)) {
      is_id <- is.integer(X[[col]]) || is.character(X[[col]]) || is.factor(X[[col]])
      if (is_id) {
        warns <- c(warns, paste0(
          "Column '", col, "' has ", nrow(X), " unique values (100% unique). ",
          "Looks like an ID column. Consider dropping it before fitting."
        ))
      }
    }
  }

  # 2. Zero-variance features
  for (col in names(X)) {
    n_unique <- length(unique(X[[col]][!is.na(X[[col]])]))
    if (n_unique <= 1L) {
      warns <- c(warns, paste0(
        "Column '", col, "' has zero variance (constant). ",
        "It provides no predictive information. Consider dropping it."
      ))
    }
  }

  # 3. High-null columns (>50%)
  for (col in names(X)) {
    null_frac <- mean(is.na(X[[col]]))
    if (null_frac > 0.5) {
      warns <- c(warns, paste0(
        "Column '", col, "' has ", round(100 * null_frac), "% missing values. ",
        "Consider imputing or dropping columns with >50% missing."
      ))
    }
  }

  # 4. Class imbalance (classification only)
  task <- .detect_task(y)
  if (task == "classification") {
    n_unique <- length(unique(y[!is.na(y)]))
    if (n_unique <= 1L) {
      val_hint <- if (sum(!is.na(y)) > 0L) paste0(" (value: ", unique(y[!is.na(y)])[1], ")") else ""
      warns <- c(warns, paste0(
        "Target '", target, "' has only ", n_unique, " unique class(es)", val_hint, ". ",
        "ml_fit() will raise an error. Ensure at least 2 classes."
      ))
    } else {
      counts <- table(y[!is.na(y)])
      freqs  <- counts / sum(counts)
      for (i in seq_along(freqs)) {
        if (freqs[i] <= 0.05) {
          warns <- c(warns, paste0(
            "Class '", names(freqs)[i], "' represents only ",
            round(100 * freqs[i], 1), "% of samples (severe class imbalance). ",
            "Consider ml_fit(..., balance = TRUE) or oversampling."
          ))
        }
      }
    }
  }

  # 5. Duplicate rows (>10%)
  n_dup <- sum(duplicated(data))
  if (n_dup > 0.1 * nrow(data)) {
    warns <- c(warns, paste0(
      n_dup, " duplicate rows (", round(100 * n_dup / nrow(data)), "% of data). ",
      "Consider deduplication."
    ))
  }

  # 6. Feature redundancy (|r| > 0.95)
  num_cols <- names(X)[vapply(X, is.numeric, logical(1))]
  if (length(num_cols) >= 2L) {
    cm <- abs(stats::cor(X[, num_cols, drop = FALSE], use = "pairwise.complete.obs"))
    pairs <- character(0)
    for (i in seq_len(ncol(cm) - 1L)) {
      for (j in (i + 1L):ncol(cm)) {
        if (!is.na(cm[i, j]) && cm[i, j] > 0.95) {
          pairs <- c(pairs, paste0("'", num_cols[i], "' <-> '", num_cols[j],
                                   "' (r=", round(cm[i, j], 3), ")"))
        }
      }
    }
    if (length(pairs) > 0L) {
      shown <- if (length(pairs) > 5L) c(pairs[1:5], paste0("(+", length(pairs) - 5, " more)")) else pairs
      warns <- c(warns, paste0(
        length(pairs), " highly correlated feature pair(s) (|r| > 0.95): ",
        paste(shown, collapse = ", "), ". ",
        "Redundant features inflate importance. Consider dropping one from each pair."
      ))
    }
  }

  has_issues <- length(warns) > 0L || length(errors) > 0L
  passed     <- length(errors) == 0L

  if (severity == "error" && has_issues) {
    all_issues <- c(errors, warns)
    data_error(paste0(
      "ml_check_data() found ", length(all_issues), " issue(s):\n",
      paste0("  - ", all_issues, collapse = "\n")
    ))
  }

  structure(
    list(warnings = warns, errors = errors, has_issues = has_issues, passed = passed),
    class = "ml_check_report"
  )
}

#' @export
print.ml_check_report <- function(x, ...) {
  cat("Data Check Report\n")
  cat(strrep("=", 40), "\n")
  if (!x$has_issues) {
    cat("No issues found. Data looks clean.\n")
  }
  if (length(x$errors) > 0L) {
    cat(sprintf("\n%d ERROR(s):\n", length(x$errors)))
    for (e in x$errors) cat("  [ERROR]", e, "\n")
  }
  if (length(x$warnings) > 0L) {
    cat(sprintf("\n%d WARNING(s):\n", length(x$warnings)))
    for (w in x$warnings) cat("  [WARN]", w, "\n")
  }
  invisible(x)
}
