# ── Safe index sequence (R's seq(a, b) returns descending when a > b)
.safe_seq <- function(from, to) {
  if (from > to) return(integer(0))
  seq.int(from, to)
}

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
#' @returns An `ml_split_result`. Access `$train`, `$valid`, `$test`,
#'   `$dev` (train + valid). When `folds` is set, also `$folds` (CV on dev).
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
  if (nrow(data) == 0L) data_error("Cannot split empty data (0 rows)")
  if (nrow(data) == 1L) data_error("Cannot split 1 row. Need at least 4.")

  # Validate target if provided
  if (!is.null(target)) {
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
      return(.temporal_cv(data, target, as.integer(folds), ratio))
    }

    # Validate ratio
    if (length(ratio) != 3L) config_error("ratio must be a numeric vector of length 3")
    if (abs(sum(ratio) - 1.0) > 0.001) {
      config_error(paste0("ratio must sum to 1.0, got ", round(sum(ratio), 4)))
    }

    n       <- nrow(data)
    sizes   <- .ml_partition_sizes(n, ratio)
    n_train <- sizes[1]
    n_valid <- sizes[2]
    train_idx <- seq_len(n_train)
    valid_idx <- .safe_seq(n_train + 1L, n_train + n_valid)
    test_idx  <- .safe_seq(n_train + n_valid + 1L, n)

    # Guard empty partitions
    if (length(test_idx) == 0L) {
      data_error("Test partition is empty. Increase n or adjust ratio.")
    }

    .warn_small_partitions(train_idx, valid_idx, test_idx)
    return(new_ml_split_result(
      train    = data[train_idx, , drop = FALSE],
      valid    = data[valid_idx, , drop = FALSE],
      test     = data[test_idx,  , drop = FALSE],
      seed     = seed,
      temporal = TRUE
    ))
  }

  # ── Group split ─────────────────────────────────────────────────────────────
  if (!is.null(groups)) {
    grp_col <- data[[groups]]
    # P0: NaN group values cause silent misassignment
    if (any(is.na(grp_col))) {
      data_error("groups column contains NA values. Drop or impute NA groups before splitting.")
    }
    unique_grps <- sort(unique(grp_col))

    if (!is.null(folds)) {
      return(.group_cv(data, target, as.integer(folds), grp_col, unique_grps,
                        seed, groups, ratio))
    }

    # Validate ratio
    if (length(ratio) != 3L) config_error("ratio must be a numeric vector of length 3")
    if (abs(sum(ratio) - 1.0) > 0.001) {
      config_error(paste0("ratio must sum to 1.0, got ", round(sum(ratio), 4)))
    }

    n_grps <- length(unique_grps)
    if (n_grps < 3L) {
      data_error(paste0(
        "Need at least 3 unique groups for 3-way split, got ", n_grps, "."
      ))
    }

    withr::local_seed(seed)
    perm_grps <- sample(unique_grps)
    n_train_g <- max(1L, round(n_grps * ratio[1]))
    n_valid_g <- max(1L, round(n_grps * ratio[2]))
    train_grps <- perm_grps[seq_len(n_train_g)]
    valid_grps <- perm_grps[.safe_seq(n_train_g + 1L, n_train_g + n_valid_g)]
    test_grps  <- perm_grps[.safe_seq(n_train_g + n_valid_g + 1L, n_grps)]

    train_idx <- which(grp_col %in% train_grps)
    valid_idx <- which(grp_col %in% valid_grps)
    test_idx  <- which(grp_col %in% test_grps)

    # P0: guard empty test (e.g., n_grps=3 with max(1L,...) consuming all groups)
    if (length(test_idx) == 0L) {
      data_error("Test partition is empty. Need more groups or adjust ratio.")
    }

    .warn_small_partitions(train_idx, valid_idx, test_idx)
    return(new_ml_split_result(
      train = data[train_idx, , drop = FALSE],
      valid = data[valid_idx, , drop = FALSE],
      test  = data[test_idx,  , drop = FALSE],
      seed  = seed
    ))
  }

  # ── K-fold CV mode ─────────────────────────────────────────────────────────
  # Grammar: Partition first (test held out), then Rotate on dev.
  # Returns ml_split_result with $folds on dev, $test sealed.
  if (!is.null(folds)) {
    if (!is.numeric(folds) || folds < 2L) {
      config_error("folds must be an integer >= 2")
    }
    folds <- as.integer(folds)

    # Hold out test first (terminal boundary)
    test_ratio <- ratio[3]
    n <- nrow(data)
    n_test <- max(1L, round(n * test_ratio))
    n_dev  <- n - n_test

    if (folds > n_dev) {
      data_error(paste0(
        "Cannot create ", folds, " folds from ", n_dev,
        " dev rows (", n_test, " held out for test). Use folds=",
        min(n_dev, 10L), " or fewer."
      ))
    }

    if (!is.null(y) && stratify && task == "classification") {
      perm <- .stratified_partition(y, test_ratio, seed)
      dev_idx  <- perm$dev
      test_idx <- perm$test
    } else {
      perm      <- .ml_shuffle(n, seed)
      dev_idx   <- perm[seq_len(n_dev)]
      test_idx  <- perm[.safe_seq(n_dev + 1L, n)]
    }

    dev_data  <- data[dev_idx, , drop = FALSE]
    test_data <- data[test_idx, , drop = FALSE]

    # Seed R's RNG for fold assignment — Rust shuffle doesn't touch R's RNG,
    # so without this, non-stratified fold assignment would be non-deterministic.
    withr::local_seed(seed)

    # Stratified k-fold on dev (preserve class ratio)
    y_dev <- if (!is.null(y)) y[dev_idx] else NULL
    if (!is.null(y_dev) && stratify && task == "classification") {
      fold_ids <- .stratified_kfold(y_dev, folds)
    } else {
      fold_ids <- sample(rep(seq_len(folds), length.out = nrow(dev_data)))
    }

    fold_list <- lapply(seq_len(folds), function(k) {
      list(
        train = which(fold_ids != k),
        valid = which(fold_ids == k)
      )
    })

    # Return split result with folds — test on the split, not on CV
    return(new_ml_split_result(
      train = dev_data, valid = dev_data[0L, , drop = FALSE], test = test_data,
      folds = fold_list, folds_data = dev_data, folds_target = target,
      seed  = seed
    ))
  }

  # Validate ratio
  if (length(ratio) != 3L) config_error("ratio must be a numeric vector of length 3")
  if (abs(sum(ratio) - 1.0) > 0.001) {
    config_error(paste0("ratio must sum to 1.0, got ", round(sum(ratio), 4)))
  }

  n <- nrow(data)
  withr::local_seed(seed)

  if (!is.null(y) && stratify && task == "classification") {
    # Stratified split: sample within each class
    classes   <- sort(unique(y))
    train_idx <- integer(0)
    valid_idx <- integer(0)
    test_idx  <- integer(0)

    rare_class <- FALSE
    ci <- 0L
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
      # Deterministic per-class shuffle via Rust PCG (cross-language parity)
      class_seed <- seed + ci
      ci <- ci + 1L
      perm_order  <- .ml_shuffle(n_cl, class_seed)
      perm        <- cl_idx[perm_order]
      train_idx   <- c(train_idx, perm[seq_len(n_train_cl)])
      valid_idx   <- c(valid_idx, perm[.safe_seq(n_train_cl + 1L, n_train_cl + n_valid_cl)])
      test_idx    <- c(test_idx,  perm[.safe_seq(n_train_cl + n_valid_cl + 1L, n_cl)])
    }

    if (rare_class) {
      cli::cli_warn(
        "Rare class detected -- falling back to non-stratified split."
      )
      perm      <- .ml_shuffle(n, seed)
      sizes     <- .ml_partition_sizes(n, ratio)
      n_train   <- sizes[1]
      n_valid   <- sizes[2]
      train_idx <- perm[seq_len(n_train)]
      valid_idx <- perm[.safe_seq(n_train + 1L, n_train + n_valid)]
      test_idx  <- perm[.safe_seq(n_train + n_valid + 1L, n)]
    }
  } else {
    # Non-stratified random split — Rust PCG for cross-language parity
    perm      <- .ml_shuffle(n, seed)
    sizes     <- .ml_partition_sizes(n, ratio)
    n_train   <- sizes[1]
    n_valid   <- sizes[2]
    train_idx <- perm[seq_len(n_train)]
    valid_idx <- perm[.safe_seq(n_train + 1L, n_train + n_valid)]
    test_idx  <- perm[.safe_seq(n_train + n_valid + 1L, n)]
  }

  # Guard empty partitions
  if (length(test_idx) == 0L) {
    data_error("Test partition is empty. Increase n or adjust ratio.")
  }

  .warn_small_partitions(train_idx, valid_idx, test_idx)

  new_ml_split_result(
    train = data[train_idx, , drop = FALSE],
    valid = data[valid_idx, , drop = FALSE],
    test  = data[test_idx,  , drop = FALSE],
    seed  = seed
  )
}

# ── Temporal CV (expanding window) ───────────────────────────────────────────

.temporal_cv <- function(data, target, folds, ratio = c(0.6, 0.2, 0.2)) {
  if (folds < 2L) config_error("folds must be an integer >= 2")
  n <- nrow(data)

  # Hold out test from END of time series (terminal boundary)
  test_ratio <- ratio[3]
  n_test <- max(1L, round(n * test_ratio))
  n_dev  <- n - n_test

  if (folds >= n_dev) {
    data_error(paste0(
      "Cannot create ", folds, " temporal folds from ", n_dev,
      " dev rows (", n_test, " held out for test)."
    ))
  }

  dev_data  <- data[seq_len(n_dev), , drop = FALSE]
  test_data <- data[.safe_seq(n_dev + 1L, n), , drop = FALSE]

  # Expanding window on dev (C1: last fold extends to n_dev, no remainder drop)
  chunk_size <- n_dev %/% (folds + 1L)
  fold_list <- lapply(seq_len(folds), function(k) {
    train_end <- chunk_size * k
    valid_start <- train_end + 1L
    valid_end <- if (k == folds) n_dev else min(train_end + chunk_size, n_dev)
    list(
      train = seq_len(train_end),
      valid = .safe_seq(valid_start, valid_end)
    )
  })
  new_ml_split_result(
    train    = dev_data, valid = dev_data[0L, , drop = FALSE], test = test_data,
    folds    = fold_list, folds_data = dev_data, folds_target = target,
    temporal = TRUE
  )
}

# ── Group CV (GroupKFold) ────────────────────────────────────────────────────

.group_cv <- function(data, target, folds, grp_col, unique_grps, seed,
                       group_col_name, ratio = c(0.6, 0.2, 0.2)) {
  if (folds < 2L) config_error("folds must be an integer >= 2")
  # P0: sort for deterministic ordering regardless of row order
  unique_grps <- sort(unique_grps)
  n_grps <- length(unique_grps)

  # Hold out test groups (terminal boundary + group integrity)
  test_ratio <- ratio[3]
  n_test_grps <- max(1L, round(n_grps * test_ratio))
  n_dev_grps  <- n_grps - n_test_grps

  if (folds > n_dev_grps) {
    data_error(paste0(
      "Cannot create ", folds, " folds from ", n_dev_grps,
      " dev groups (", n_test_grps, " held out for test). Use folds=",
      min(n_dev_grps, 10L), " or fewer."
    ))
  }

  withr::local_seed(seed)
  perm_grps  <- sample(unique_grps)
  dev_grps   <- perm_grps[seq_len(n_dev_grps)]
  test_grps  <- perm_grps[.safe_seq(n_dev_grps + 1L, n_grps)]

  dev_idx    <- which(grp_col %in% dev_grps)
  test_idx   <- which(grp_col %in% test_grps)
  dev_data   <- data[dev_idx, , drop = FALSE]
  test_data  <- data[test_idx, , drop = FALSE]

  # Round-robin assignment of dev groups to folds
  dev_grp_col <- dev_data[[group_col_name]]
  grp_fold_ids <- rep(seq_len(folds), length.out = n_dev_grps)
  # P0: use match() instead of named vector lookup — safe for factors and all types
  row_fold_ids <- grp_fold_ids[match(as.character(dev_grp_col), as.character(dev_grps))]

  fold_list <- lapply(seq_len(folds), function(k) {
    list(
      train = which(row_fold_ids != k),
      valid = which(row_fold_ids == k)
    )
  })
  new_ml_split_result(
    train = dev_data, valid = dev_data[0L, , drop = FALSE], test = test_data,
    folds = fold_list, folds_data = dev_data, folds_target = target
  )
}

# ── Domain specializations ───────────────────────────────────────────────────

#' Split data chronologically — no future leakage
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
#' @returns An `ml_split_result`. When `folds` is set, includes `$folds` and `$test`.
#' @export
#' @examples
#' df <- data.frame(date = 1:100, x = rnorm(100), y = sample(0:1, 100, TRUE))
#' s <- ml_split_temporal(df, "y", time = "date")
#' nrow(s$train)
ml_split_temporal <- function(data, target = NULL, time,
                              ratio = c(0.6, 0.2, 0.2),
                              folds = NULL, task = "auto") {
  # Temporal splits are deterministic (sorted by time); seed is only used
  # internally and never affects output. Use 0L to avoid clobbering R's RNG
  # with a meaningful seed value.
  .split_impl(data = data, target = target, seed = 0L,
              ratio = ratio, folds = folds, stratify = FALSE, task = task,
              time = time, groups = NULL)
}

#' Split data with group non-overlap — no group leaks across partitions
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
#' @returns An `ml_split_result`. When `folds` is set, includes `$folds` and `$test`.
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

# ── Stratified k-fold assignment (preserve class ratio per fold) ─────────

.stratified_kfold <- function(y, k) {
  n <- length(y)
  fold_ids <- integer(n)
  classes <- sort(unique(y))
  for (cl in classes) {
    cl_idx <- which(y == cl)
    cl_idx <- sample(cl_idx)
    for (j in seq_along(cl_idx)) {
      fold_ids[cl_idx[j]] <- ((j - 1L) %% k) + 1L
    }
  }
  fold_ids
}

# ── Stratified dev/test partition (preserve class ratio in test holdout) ─

.stratified_partition <- function(y, test_ratio, seed) {
  n <- length(y)
  dev_idx  <- integer(0)
  test_idx <- integer(0)
  classes  <- sort(unique(y))
  ci <- 0L
  for (cl in classes) {
    cl_idx <- which(y == cl)
    n_cl   <- length(cl_idx)
    n_test_cl <- max(1L, round(n_cl * test_ratio))
    # Singleton class: entire class goes to test, absent from dev/folds
    if (n_test_cl >= n_cl && n_cl <= 1L) {
      cli::cli_warn(paste0(
        "Class '", cl, "' has only ", n_cl,
        " member(s) -- absent from dev folds, present only in test."
      ))
    }
    # Deterministic per-class shuffle via Rust PCG (cross-language parity)
    class_seed <- seed + ci
    ci <- ci + 1L
    perm_order <- .ml_shuffle(n_cl, class_seed)
    perm <- cl_idx[perm_order]
    test_idx <- c(test_idx, perm[seq_len(n_test_cl)])
    dev_idx  <- c(dev_idx, perm[.safe_seq(n_test_cl + 1L, n_cl)])
  }
  list(dev = dev_idx, test = test_idx)
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
