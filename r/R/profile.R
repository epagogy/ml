#' Profile data before modeling
#'
#' Computes per-column statistics and emits warnings for common data quality
#' issues: missing values, constant columns, high cardinality, imbalanced
#' targets, and near-collinear features.
#'
#' @param data A data.frame (also accepts tibble or data.table)
#' @param target Optional target column name (enables task detection + distribution stats)
#' @returns An object of class `ml_profile_result` (list with formatted print)
#' @export
#' @examples
#' ml_profile(iris, "Species")
ml_profile <- function(data, target = NULL) {
  .profile_impl(data = data, target = target)
}

.profile_impl <- function(data, target = NULL) {
  if (is.list(data) && !is.data.frame(data)) {
    data_error(paste0("Expected data.frame, got ", class(data)[[1]]))
  }
  data <- .coerce_data(data)

  if (nrow(data) == 0L) data_error("Cannot profile empty data (0 rows)")

  if (!is.null(target) && !target %in% names(data)) {
    data_error(paste0("target='", target, "' not found. Available columns: ",
                      paste(names(data), collapse = ", ")))
  }

  n_rows <- nrow(data)
  n_cols <- ncol(data)
  warnings_list <- character(0)

  # --- Target info --------------------------------------------------------
  task <- "unknown"
  y <- NULL
  if (!is.null(target)) {
    y <- data[[target]]
    if (all(is.na(y))) {
      warnings_list <- c(warnings_list,
                         paste0("'", target, "' is entirely NA"))
      # Skip task detection
    } else {
      task <- .detect_task(y)

      if (task == "classification") {
        class_counts <- table(y[!is.na(y)])
        n_classes    <- length(class_counts)

        # Imbalanced target: minority class < 20% for binary, or mean - min > threshold
        if (n_classes == 2L) {
          minority_pct <- min(class_counts) / sum(class_counts) * 100
          if (minority_pct < 20) {
            minority_label <- names(class_counts)[which.min(class_counts)]
            warnings_list <- c(warnings_list, paste0(
              "Imbalanced target: minority class '", minority_label,
              "' is ", round(minority_pct, 1), "% of data"
            ))
          }
        }

        # Many classes warning
        if (n_classes > 20L) {
          warnings_list <- c(warnings_list, paste0(
            "Target has ", n_classes, " classes -- really classification?"
          ))
        }
      }
    }
  }

  # --- Small dataset warning -----------------------------------------------
  if (n_rows < 200L) {
    n_train_approx <- round(n_rows * 0.6)
    n_test_approx  <- round(n_rows * 0.2)
    warnings_list <- c(warnings_list, paste0(
      "Small dataset (", n_rows, " rows). ~", n_train_approx,
      " train, ~", n_test_approx, " test after split."
    ))
  }

  # --- Per-column stats ---------------------------------------------------
  columns <- list()
  feature_cols <- if (!is.null(target)) setdiff(names(data), target) else names(data)

  for (col in names(data)) {
    v         <- data[[col]]
    n_miss    <- sum(is.na(v))
    miss_pct  <- n_miss / n_rows * 100
    n_unique  <- length(unique(v[!is.na(v)]))

    if (n_miss > 0) {
      warnings_list <- c(warnings_list, paste0(
        n_miss, " rows (", round(miss_pct, 1), "%) have missing values in '", col, "'"
      ))
    }

    # Constant column
    if (n_unique <= 1L && !identical(col, target)) {
      warnings_list <- c(warnings_list, paste0("'", col, "' is constant (1 unique value)"))
    }

    # All NA column
    if (n_miss == n_rows) {
      warnings_list <- c(warnings_list, paste0("'", col, "' is entirely NA"))
    }

    col_stats <- list(
      dtype       = class(v)[[1]],
      missing     = n_miss,
      missing_pct = round(miss_pct, 2),
      unique      = n_unique
    )

    if (is.numeric(v)) {
      v_clean <- v[!is.na(v)]
      col_stats$mean   <- if (length(v_clean) > 0) mean(v_clean) else NA_real_
      col_stats$sd     <- if (length(v_clean) > 1) stats::sd(v_clean) else NA_real_
      col_stats$min    <- if (length(v_clean) > 0) min(v_clean) else NA_real_
      col_stats$max    <- if (length(v_clean) > 0) max(v_clean) else NA_real_
      col_stats$median <- if (length(v_clean) > 0) stats::median(v_clean) else NA_real_
    } else {
      tab <- table(v[!is.na(v)])
      if (length(tab) > 0) {
        col_stats$top      <- names(tab)[which.max(tab)]
        col_stats$top_freq <- as.integer(max(tab))
      }
      # High cardinality (categorical features only, not target)
      if (!identical(col, target) && n_unique > 50L) {
        warnings_list <- c(warnings_list, paste0(
          "'", col, "' high cardinality (", n_unique, "/", n_rows, " unique)"
        ))
      }
    }
    columns[[col]] <- col_stats
  }

  # --- Condition number (numeric features, not target) ---------------------
  condition_number <- NULL
  num_feature_cols <- feature_cols[vapply(feature_cols, function(col) is.numeric(data[[col]]), logical(1L))]
  if (length(num_feature_cols) >= 2L) {
    X_num <- as.matrix(data[, num_feature_cols, drop = FALSE])
    X_num <- X_num[complete.cases(X_num), , drop = FALSE]
    if (nrow(X_num) > 1L) {
      # Subsample to 5000 rows for performance
      if (nrow(X_num) > 5000L) X_num <- X_num[sample(nrow(X_num), 5000L), , drop = FALSE]
      tryCatch({
        cn <- kappa(X_num, exact = FALSE)
        condition_number <- cn
        if (!is.na(cn) && !is.infinite(cn) && cn > 1000) {
          warnings_list <- c(warnings_list, paste0(
            "High condition number (", round(cn, 0), "). Near-collinear features."
          ))
        }
      }, error = function(e) NULL)
    }
  }

  new_ml_profile_result(
    shape            = c(n_rows, n_cols),
    target           = target,
    task             = task,
    columns          = columns,
    warnings         = warnings_list,
    condition_number = condition_number
  )
}
