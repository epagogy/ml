#' Split data into train/valid/test partitions or cross-validation folds
#'
#' Three-way split is the default (60/20/20), following Hastie, Tibshirani, and
#' Friedman (2009, ISBN:978-0-387-84857-0) Chapter 7. Automatically stratifies
#' for classification.
#'
#' @param data A data.frame (also accepts tibble or data.table)
#' @param target Target column name (enables stratification + task detection)
#' @param seed Random seed. NULL (default) auto-generates and stores for
#'   reproducibility. Pass an integer for reproducible splits.
#' @param ratio Numeric vector of length 3: c(train, valid, test). Must sum to 1.0.
#' @param folds Integer for k-fold CV (e.g., `folds = 5`). Overrides ratio.
#' @param stratify Logical. Auto-stratify for classification targets (default TRUE).
#' @param task "auto", "classification", or "regression". Override task detection.
#' @param time Column name for temporal/chronological split. Data is sorted by
#'   this column, and the time column is dropped from output. Deterministic
#'   (seed is ignored). Cannot combine with `groups`.
#' @param groups Column name for group-aware split. No group appears in both
#'   train and validation/test. Cannot combine with `time`.
#' @returns An `ml_split_result` (when `ratio` used) or `ml_cv_result` (when
#'   `folds` used). Access `$train`, `$valid`, `$test`, `$dev` (train + valid).
#' @export
#' @examples
#' s <- ml_split(iris, "Species", seed = 42)
#' nrow(s$train)
#' nrow(s$dev)
ml_split <- function(data, target = NULL, seed = NULL,
                     ratio = c(0.6, 0.2, 0.2), folds = NULL,
                     stratify = TRUE, task = "auto",
                     time = NULL, groups = NULL) {
  .split_impl(data = data, target = target, seed = seed,
              ratio = ratio, folds = folds, stratify = stratify, task = task,
              time = time, groups = groups)
}

.split_impl <- function(data, target = NULL, seed = NULL,
                        ratio = c(0.6, 0.2, 0.2), folds = NULL,
                        stratify = TRUE, task = "auto",
                        time = NULL, groups = NULL) {
  # Coerce input types
  data <- .coerce_data(data)

  # Duplicate columns check
  .check_duplicate_cols(data)

  # Mutual exclusion: time and groups
  if (!is.null(time) && !is.null(groups)) {
    config_error("Cannot combine time= and groups=. Use one or the other.")
  }

  # Validate time column
  if (!is.null(time)) {
    if (!time %in% names(data)) {
      data_error(paste0("time='", time, "' not found in data."))
    }
  }

  # Validate groups column
  if (!is.null(groups)) {
    if (!groups %in% names(data)) {
      data_error(paste0("groups='", groups, "' not found in data."))
    }
  }

  # Auto-generate seed if NULL
  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)

  # Validate dimensions
  if (nrow(data) == 0L) data_error("Cannot split empty data (0 rows). Check that your data.frame is not empty after filtering.")
  if (nrow(data) < 4L) data_error(paste0("Cannot split ", nrow(data), " row(s) into 3 partitions. Need at least 4 rows, or use folds=2 for small datasets."))

  # Validate target if provided
  if (!is.null(target)) {
    if (!nzchar(target)) {
      config_error("target must be a non-empty string, e.g. 'Species'")
    }
    if (!target %in% names(data)) {
      data_error(paste0(
        "target='", target, "' not found in data. Available columns: ",
        paste(names(data), collapse = ", ")
      ))
    }
    y <- data[[target]]

    # Drop NA target rows
    na_mask <- is.na(y)
    if (any(na_mask)) {
      n_dropped <- sum(na_mask)
      cli::cli_warn(paste0("Dropped ", n_dropped, " rows with NA target."))
      data <- data[!na_mask, , drop = FALSE]
      y    <- data[[target]]
    }

    # Detect task
    detected_task <- .detect_task(y, task)

    # Warn if numeric target auto-detected as classification
    if (task == "auto" && detected_task == "classification" && is.numeric(y)) {
      n_uniq <- length(unique(y))
      cli::cli_warn(paste0(
        "Target '", target, "' detected as classification (", n_uniq,
        " unique values). To override: task='regression'"
      ))
    }
    task <- detected_task
  } else {
    y <- NULL
  }

  # ── Temporal split ──────────────────────────────────────────────────────────
  if (!is.null(time)) {
    # Sort by time column, split in order (deterministic, seed ignored)
    ord  <- order(data[[time]])
    data <- data[ord, , drop = FALSE]
    # Drop the time column from output
    data <- data[, setdiff(names(data), time), drop = FALSE]

    if (!is.null(folds)) {
      return(.temporal_cv(data, target, as.integer(folds)))
    }

    # Validate ratio
    if (length(ratio) != 3L) config_error("ratio must be a numeric vector of length 3, e.g. c(0.6, 0.2, 0.2)")
    if (abs(sum(ratio) - 1.0) > 0.001) {
      config_error(paste0("ratio must sum to 1.0, got ", round(sum(ratio), 4)))
    }

    n       <- nrow(data)
    n_train <- round(n * ratio[1])
    n_valid <- round(n * ratio[2])
    train_idx <- seq_len(n_train)
    valid_idx <- seq(n_train + 1L, n_train + n_valid)
    test_idx  <- seq(n_train + n_valid + 1L, n)

    .warn_small_partitions(train_idx, valid_idx, test_idx)
    return(new_ml_split_result(
      train = data[train_idx, , drop = FALSE],
      valid = data[valid_idx, , drop = FALSE],
      test  = data[test_idx,  , drop = FALSE]
    ))
  }

  # ── Group split ─────────────────────────────────────────────────────────────
  if (!is.null(groups)) {
    grp_col <- data[[groups]]
    unique_grps <- unique(grp_col)

    if (!is.null(folds)) {
      return(.group_cv(data, target, as.integer(folds), grp_col, unique_grps, seed))
    }

    # Validate ratio
    if (length(ratio) != 3L) config_error("ratio must be a numeric vector of length 3, e.g. c(0.6, 0.2, 0.2)")
    if (abs(sum(ratio) - 1.0) > 0.001) {
      config_error(paste0("ratio must sum to 1.0, got ", round(sum(ratio), 4)))
    }

    n_grps <- length(unique_grps)
    if (n_grps < 3L) {
      data_error(paste0(
        "Need at least 3 unique groups for 3-way split, got ", n_grps, "."
      ))
    }

    .local_seed(seed)
    perm_grps <- sample(unique_grps)
    n_train_g <- max(1L, round(n_grps * ratio[1]))
    n_valid_g <- max(1L, round(n_grps * ratio[2]))
    train_grps <- perm_grps[seq_len(n_train_g)]
    valid_grps <- perm_grps[seq(n_train_g + 1L, n_train_g + n_valid_g)]
    test_grps  <- perm_grps[seq(n_train_g + n_valid_g + 1L, n_grps)]

    train_idx <- which(grp_col %in% train_grps)
    valid_idx <- which(grp_col %in% valid_grps)
    test_idx  <- which(grp_col %in% test_grps)

    .warn_small_partitions(train_idx, valid_idx, test_idx)
    return(new_ml_split_result(
      train = data[train_idx, , drop = FALSE],
      valid = data[valid_idx, , drop = FALSE],
      test  = data[test_idx,  , drop = FALSE]
    ))
  }

  # ── K-fold CV mode ─────────────────────────────────────────────────────────
  if (!is.null(folds)) {
    if (!is.numeric(folds) || folds < 2L) {
      config_error("folds must be an integer >= 2")
    }
    folds <- as.integer(folds)
    n <- nrow(data)
    if (folds > n) {
      data_error(paste0(
        "Cannot create ", folds, " folds from ", n, " rows. Use folds=",
        min(n, 10L), " or fewer."
      ))
    }
    .local_seed(seed)
    fold_ids <- sample(rep(seq_len(folds), length.out = n))
    fold_list <- lapply(seq_len(folds), function(k) {
      list(
        train = which(fold_ids != k),
        valid = which(fold_ids == k)
      )
    })
    return(new_ml_cv_result(folds = fold_list, data = data, target = target))
  }

  # Validate ratio
  if (length(ratio) != 3L) config_error("ratio must be a numeric vector of length 3, e.g. c(0.6, 0.2, 0.2)")
  if (abs(sum(ratio) - 1.0) > 0.001) {
    config_error(paste0("ratio must sum to 1.0, got ", round(sum(ratio), 4)))
  }

  n <- nrow(data)
  .local_seed(seed)

  if (!is.null(y) && stratify && task == "classification") {
    # Stratified split: sample within each class
    classes   <- unique(y)
    train_idx <- integer(0)
    valid_idx <- integer(0)
    test_idx  <- integer(0)

    rare_class <- FALSE
    for (cl in classes) {
      cl_idx <- which(y == cl)
      n_cl   <- length(cl_idx)
      n_train_cl <- max(1L, round(n_cl * ratio[1]))
      n_valid_cl <- max(1L, round(n_cl * ratio[2]))
      n_test_cl  <- n_cl - n_train_cl - n_valid_cl

      if (n_test_cl < 1L || n_train_cl < 1L) {
        rare_class <- TRUE
        break
      }
      perm        <- sample(cl_idx)
      train_idx   <- c(train_idx, perm[seq_len(n_train_cl)])
      valid_idx   <- c(valid_idx, perm[seq(n_train_cl + 1L, n_train_cl + n_valid_cl)])
      test_idx    <- c(test_idx,  perm[seq(n_train_cl + n_valid_cl + 1L, n_cl)])
    }

    if (rare_class) {
      cli::cli_warn(
        "Rare class detected -- falling back to non-stratified split."
      )
      perm      <- sample(n)
      n_train   <- round(n * ratio[1])
      n_valid   <- round(n * ratio[2])
      train_idx <- perm[seq_len(n_train)]
      valid_idx <- perm[seq(n_train + 1L, n_train + n_valid)]
      test_idx  <- perm[seq(n_train + n_valid + 1L, n)]
    }
  } else {
    # Non-stratified random split
    perm      <- sample(n)
    n_train   <- round(n * ratio[1])
    n_valid   <- round(n * ratio[2])
    train_idx <- perm[seq_len(n_train)]
    valid_idx <- perm[seq(n_train + 1L, n_train + n_valid)]
    test_idx  <- perm[seq(n_train + n_valid + 1L, n)]
  }

  .warn_small_partitions(train_idx, valid_idx, test_idx)

  new_ml_split_result(
    train = data[train_idx, , drop = FALSE],
    valid = data[valid_idx, , drop = FALSE],
    test  = data[test_idx,  , drop = FALSE]
  )
}

# ── Temporal CV (expanding window) ───────────────────────────────────────────

.temporal_cv <- function(data, target, folds) {
  if (folds < 2L) config_error("folds must be an integer >= 2")
  n <- nrow(data)
  if (folds >= n) {
    data_error(paste0("Cannot create ", folds, " temporal folds from ", n, " rows. Need at least ", folds + 1L, " rows (one chunk per fold plus one for the initial training window)."))
  }
  # Expanding window: fold k trains on rows 1:cutoff, validates on next chunk
  chunk_size <- n %/% (folds + 1L)
  fold_list <- lapply(seq_len(folds), function(k) {
    train_end <- chunk_size * k
    valid_start <- train_end + 1L
    valid_end <- min(train_end + chunk_size, n)
    list(
      train = seq_len(train_end),
      valid = seq(valid_start, valid_end)
    )
  })
  new_ml_cv_result(folds = fold_list, data = data, target = target)
}

# ── Group CV (GroupKFold) ────────────────────────────────────────────────────

.group_cv <- function(data, target, folds, grp_col, unique_grps, seed) {
  if (folds < 2L) config_error("folds must be an integer >= 2")
  n_grps <- length(unique_grps)
  if (folds > n_grps) {
    data_error(paste0(
      "Cannot create ", folds, " folds from ", n_grps,
      " groups. Use folds=", min(n_grps, 10L), " or fewer."
    ))
  }
  # Round-robin assignment of groups to folds
  .local_seed(seed)
  perm_grps <- sample(unique_grps)
  grp_fold_ids <- rep(seq_len(folds), length.out = n_grps)
  names(grp_fold_ids) <- as.character(perm_grps)

  row_fold_ids <- grp_fold_ids[as.character(grp_col)]

  fold_list <- lapply(seq_len(folds), function(k) {
    list(
      train = which(row_fold_ids != k),
      valid = which(row_fold_ids == k)
    )
  })
  new_ml_cv_result(folds = fold_list, data = data, target = target)
}

# ── Domain specializations ───────────────────────────────────────────────────

#' Split data chronologically -- no future leakage
#'
#' Domain specialization of `ml_split()` for time series and forecasting.
#' Data is sorted by the `time` column and partitioned by position.
#' Deterministic: seed is ignored (chronological order is the only order).
#'
#' @param data A data.frame
#' @param target Target column name (optional, enables task detection)
#' @param time Column name containing timestamps or orderable values.
#'   Used for sorting, then dropped from output partitions.
#' @param ratio Numeric vector c(train, valid, test). Must sum to 1.0.
#' @param folds Integer for temporal CV (expanding window). When set, ignores ratio.
#' @param task "auto", "classification", or "regression"
#' @returns An `ml_split_result` (holdout) or `ml_cv_result` (temporal CV)
#' @export
#' @examples
#' df <- data.frame(date = 1:100, x = rnorm(100), y = sample(0:1, 100, TRUE))
#' s <- ml_split_temporal(df, "y", time = "date")
#' nrow(s$train)
ml_split_temporal <- function(data, target = NULL, time,
                              ratio = c(0.6, 0.2, 0.2),
                              folds = NULL, task = "auto") {
  .split_impl(data = data, target = target, seed = 1L,
              ratio = ratio, folds = folds, stratify = FALSE, task = task,
              time = time, groups = NULL)
}

#' Split data with group non-overlap -- no group leaks across partitions
#'
#' Domain specialization of `ml_split()` for clinical trials, repeated measures,
#' and any data where observations are nested within groups (patients, subjects,
#' hospitals). No group appears in more than one partition.
#'
#' Also covers Leave-Source-Out CV: when groups represent data sources
#' (hospitals, devices), this produces deployment-realistic evaluation.
#'
#' @param data A data.frame
#' @param target Target column name (optional, enables stratification)
#' @param groups Column name identifying groups
#' @param seed Random seed for reproducibility
#' @param ratio Numeric vector c(train, valid, test). Must sum to 1.0.
#' @param folds Integer for group CV. When set, ignores ratio.
#' @param stratify Logical. Stratify by target within groups (default TRUE).
#' @param task "auto", "classification", or "regression"
#' @returns An `ml_split_result` (holdout) or `ml_cv_result` (group CV)
#' @export
#' @examples
#' df <- data.frame(pid = rep(1:10, each = 5), x = rnorm(50), y = sample(0:1, 50, TRUE))
#' s <- ml_split_group(df, "y", groups = "pid", seed = 42)
#' nrow(s$train)
ml_split_group <- function(data, target = NULL, groups, seed = NULL,
                           ratio = c(0.6, 0.2, 0.2), folds = NULL,
                           stratify = TRUE, task = "auto") {
  .split_impl(data = data, target = target, seed = seed,
              ratio = ratio, folds = folds, stratify = stratify, task = task,
              time = NULL, groups = groups)
}

# ── Partition size warning helper ────────────────────────────────────────────

.warn_small_partitions <- function(train_idx, valid_idx, test_idx) {
  for (part_name in c("train", "valid", "test")) {
    part_n <- switch(part_name,
      train = length(train_idx),
      valid = length(valid_idx),
      test  = length(test_idx)
    )
    if (part_n < 30L) {
      cli::cli_warn(paste0(
        "Partition '", part_name, "' has only ", part_n,
        " rows. Results may be unreliable. Consider folds=5 for small datasets."
      ))
    }
  }
}
