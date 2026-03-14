#' Detect potential data leakage
#'
#' Analyzes feature-target relationships before modeling. Runs pure data
#' introspection -- no model fitting.
#'
#' Checks performed:
#' \enumerate{
#'   \item Feature-target correlation (Pearson |r|, numeric features)
#'   \item High-cardinality ID columns
#'   \item Target name in feature names
#'   \item Duplicate rows between train and test (SplitResult only)
#' }
#'
#' @param data A data.frame or ml_split_result
#' @param target Target column name
#' @returns A list with \code{clean} (logical), \code{n_warnings},
#'   \code{checks} (list of check results), \code{suspects} (list of
#'   suspect features). Class \code{ml_leak_report}.
#' @export
#' @examples
#' s <- ml_split(iris, "Species", seed = 42)
#' report <- ml_leak(s, "Species")
#' report$clean
ml_leak <- function(data, target) {
  # 1. Unwrap SplitResult
  test_df <- NULL
  if (inherits(data, "ml_split_result")) {
    train_df <- data[["train"]]
    test_df  <- data[["test"]]
  } else if (is.data.frame(data)) {
    train_df <- .coerce_data(data)
  } else {
    data_error(paste0(
      "ml_leak() expects data.frame or ml_split_result, got ",
      class(data)[[1]]
    ))
  }

  # 2. Validate
  if (nrow(train_df) == 0L) data_error("Cannot analyze empty data (0 rows)")
  if (!target %in% names(train_df)) {
    data_error(paste0(
      "target='", target, "' not found in data. ",
      "Available: ", paste(names(train_df), collapse = ", ")
    ))
  }

  X <- train_df[, setdiff(names(train_df), target), drop = FALSE]
  y <- train_df[[target]]

  task <- .detect_task(y)

  # Encode target for correlation
  y_numeric <- .leak_encode_target(y)

  checks   <- list()
  suspects <- list()

 # Check 1: Feature-target correlation
  n_classes <- length(unique(y[!is.na(y)]))
  if (task == "classification" && n_classes > 2L) {
    checks <- c(checks, list(list(
      name = "Feature-target correlation",
      passed = TRUE,
      detail = paste0("skipped (multiclass, ", n_classes, " classes)"),
      severity = "ok"
    )))
  } else {
    cr <- .check_correlation(X, y_numeric)
    checks   <- c(checks, list(cr$check))
    suspects <- c(suspects, cr$suspects)
  }

  # Check 2: High-cardinality IDs
  cr2 <- .check_id_columns(X)
  checks   <- c(checks, list(cr2$check))
  suspects <- c(suspects, cr2$suspects)

  # Check 3: Target in feature names
  cr3 <- .check_target_names(names(X), target)
  checks   <- c(checks, list(cr3$check))
  suspects <- c(suspects, cr3$suspects)

  # Check 4: Duplicate rows (SplitResult only)
  if (!is.null(test_df)) {
    test_X <- test_df[, setdiff(names(test_df), target), drop = FALSE]
    cr4 <- .check_duplicates(X, test_X)
    checks   <- c(checks, list(cr4$check))
    suspects <- c(suspects, cr4$suspects)
  } else {
    checks <- c(checks, list(list(
      name = "Duplicate rows (train/test)",
      passed = TRUE,
      detail = "skipped (no split provided)",
      severity = "ok"
    )))
  }

  n_warnings <- sum(vapply(checks, function(c) !c$passed, logical(1)))

  structure(
    list(
      clean      = n_warnings == 0L,
      n_warnings = n_warnings,
      checks     = checks,
      suspects   = suspects
    ),
    class = "ml_leak_report"
  )
}

#' Print a leakage report
#'
#' @param x An `ml_leak_report` object.
#' @param ... Additional arguments (ignored).
#' @returns The object `x`, invisibly.
#' @export
print.ml_leak_report <- function(x, ...) {
  status <- if (x$clean) "CLEAN" else paste0(x$n_warnings, " WARNING(s)")
  cat(sprintf("-- Leak Report [%s] --\n", status))
  for (c in x$checks) {
    icon <- if (c$passed) "ok" else "!!"
    cat(sprintf("  [%s] %s: %s\n", icon, c$name, c$detail))
  }
  if (length(x$suspects) > 0L) {
    cat("\nSuspect features:\n")
    for (s in x$suspects) {
      cat(sprintf("  - %s (%s): %s\n", s$feature, s$check, s$detail))
    }
  }
  invisible(x)
}

# ── Helpers ──────────────────────────────────────────────────────────────────

.leak_encode_target <- function(y) {
  if (is.factor(y) || is.character(y) || is.logical(y)) {
    lvls <- sort(unique(as.character(y[!is.na(y)])))
    mapping <- stats::setNames(seq_along(lvls) - 1L, lvls)
    return(as.numeric(mapping[as.character(y)]))
  }
  as.numeric(y)
}

.WARN_CORR <- 0.8
.CRIT_CORR <- 0.95

.check_correlation <- function(X, y_numeric) {
  num_cols <- names(X)[vapply(X, is.numeric, logical(1))]

  if (length(num_cols) == 0L) {
    return(list(
      check = list(name = "Feature-target correlation", passed = TRUE,
                   detail = "no numeric features", severity = "ok"),
      suspects = list()
    ))
  }

  suspects  <- list()
  max_corr  <- 0
  max_col   <- ""

  for (col in num_cols) {
    r <- tryCatch(
      abs(stats::cor(X[[col]], y_numeric, use = "pairwise.complete.obs")),
      error = function(e) NA_real_
    )
    if (is.na(r)) next

    if (r > max_corr) {
      max_corr <- r
      max_col  <- col
    }

    if (r > .CRIT_CORR) {
      suspects <- c(suspects, list(list(
        feature = col, check = "Feature-target correlation",
        value = r, detail = sprintf("|r|=%.2f", r),
        action = "Verify not derived from target"
      )))
    } else if (r > .WARN_CORR) {
      suspects <- c(suspects, list(list(
        feature = col, check = "Feature-target correlation",
        value = r, detail = sprintf("|r|=%.2f", r),
        action = "Investigate if this feature uses target data"
      )))
    }
  }

  severity <- if (max_corr > .CRIT_CORR) "critical" else if (max_corr > .WARN_CORR) "warn" else "ok"
  passed   <- severity == "ok"
  detail   <- if (nzchar(max_col)) sprintf("max |r|=%.2f (%s)", max_corr, max_col) else "no valid correlations"

  list(
    check = list(name = "Feature-target correlation", passed = passed,
                 detail = detail, severity = severity),
    suspects = suspects
  )
}

.check_id_columns <- function(X) {
  suspects <- list()
  n_rows   <- nrow(X)
  id_pat   <- "(^id$|_id$|^index$|^key$|_key$)"

  for (col in names(X)) {
    n_unique     <- length(unique(X[[col]]))
    unique_ratio <- if (n_rows > 0L) n_unique / n_rows else 0
    is_discrete  <- is.integer(X[[col]]) || is.character(X[[col]]) || is.factor(X[[col]])
    is_high_card <- unique_ratio > 0.95 && is_discrete
    is_id_name   <- grepl(id_pat, col, ignore.case = TRUE)

    if (is_high_card || is_id_name) {
      reasons <- character(0)
      if (is_high_card) reasons <- c(reasons, sprintf("%.0f%% unique", unique_ratio * 100))
      if (is_id_name) reasons <- c(reasons, "name matches ID pattern")
      suspects <- c(suspects, list(list(
        feature = col, check = "High-cardinality IDs",
        value = unique_ratio, detail = paste(reasons, collapse = ", "),
        action = "Drop before modeling"
      )))
    }
  }

  passed <- length(suspects) == 0L
  if (passed) {
    detail <- "none found"
  } else {
    nms <- vapply(suspects[seq_len(min(3L, length(suspects)))], `[[`, character(1), "feature")
    detail <- paste0(length(suspects), " suspect: ", paste(nms, collapse = ", "))
  }

  list(
    check = list(name = "High-cardinality IDs", passed = passed,
                 detail = detail, severity = if (passed) "ok" else "warn"),
    suspects = suspects
  )
}

.check_target_names <- function(features, target) {
  suspects <- list()
  target_lower <- tolower(target)
  leak_pat <- "(^future_|_future$|^next_|_next$|_after$|_outcome$|_result$)"

  for (feat in features) {
    feat_lower <- tolower(feat)
    reasons <- character(0)

    if (grepl(target_lower, feat_lower, fixed = TRUE) && feat_lower != target_lower) {
      reasons <- c(reasons, paste0("contains target name '", target, "'"))
    }
    if (grepl(leak_pat, feat, ignore.case = TRUE)) {
      reasons <- c(reasons, "matches leakage name pattern")
    }

    if (length(reasons) > 0L) {
      suspects <- c(suspects, list(list(
        feature = feat, check = "Target in feature names",
        value = 1.0, detail = paste(reasons, collapse = ", "),
        action = paste0("Verify '", feat, "' doesn't encode the target")
      )))
    }
  }

  passed <- length(suspects) == 0L
  if (passed) {
    detail <- "none found"
  } else {
    nms <- vapply(suspects[seq_len(min(3L, length(suspects)))], `[[`, character(1), "feature")
    detail <- paste0(length(suspects), " suspect: ", paste(nms, collapse = ", "))
  }

  list(
    check = list(name = "Target in feature names", passed = passed,
                 detail = detail, severity = if (passed) "ok" else "warn"),
    suspects = suspects
  )
}

.check_duplicates <- function(train_X, test_X) {
  common_cols <- intersect(names(train_X), names(test_X))
  if (length(common_cols) == 0L) {
    return(list(
      check = list(name = "Duplicate rows (train/test)", passed = TRUE,
                   detail = "no common columns", severity = "ok"),
      suspects = list()
    ))
  }

  # Hash-based duplicate detection
  hash_rows <- function(df, cols) {
    apply(df[, cols, drop = FALSE], 1, function(r) paste(r, collapse = "|"))
  }

  train_hashes <- unique(hash_rows(train_X, common_cols))
  test_hashes  <- hash_rows(test_X, common_cols)
  shared <- sum(test_hashes %in% train_hashes)

  n_test   <- nrow(test_X)
  dup_ratio <- if (n_test > 0L) shared / n_test else 0

  suspects <- list()
  if (shared > 0L) {
    severity <- if (dup_ratio > 0.01) "critical" else "warn"
    suspects <- list(list(
      feature = "(rows)", check = "Duplicate rows (train/test)",
      value = dup_ratio,
      detail = sprintf("%d shared rows (%.1f%% of test)", shared, dup_ratio * 100),
      action = "Remove duplicate rows or verify intentional"
    ))
    detail <- sprintf("%d shared rows (%.1f%% of test)", shared, dup_ratio * 100)
    passed <- FALSE
  } else {
    severity <- "ok"
    detail   <- "0 shared rows"
    passed   <- TRUE
  }

  list(
    check = list(name = "Duplicate rows (train/test)", passed = passed,
                 detail = detail, severity = severity),
    suspects = suspects
  )
}
