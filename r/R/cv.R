# -- Cross-validation primitives -----------------------------------------------
#
# cv() creates k-fold rotations *within* an existing split's dev partition.
# Test stays sealed on the original split for ml_assess().
#
# Grammar: split() creates boundaries, cv() creates rotations.

#' Create k-fold cross-validation from a split
#'
#' Takes an existing `ml_split_result` and creates k-fold rotations within its
#' dev partition (train + valid). The test partition stays sealed on the original
#' split for `ml_assess()`.
#'
#' Two primitives, strict separation of concerns: `ml_split()` creates the
#' three-way boundary, `ml_cv()` creates rotations within that boundary.
#'
#' @param s An `ml_split_result` from `ml_split()`
#' @param target Target column name (string)
#' @param folds Number of folds (default 5)
#' @param seed Random seed for fold assignment
#' @param stratify Logical. Stratify folds by target for classification (default TRUE)
#' @returns An `ml_cv_result` that `ml_fit()` accepts directly.
#'   The original split's `$test` remains available via `s$test` for `ml_assess()`.
#' @export
#' @examples
#' s <- ml_split(iris, "Species", seed = 42)
#' c <- ml_cv(s, "Species", folds = 5, seed = 42)
#' model <- ml_fit(c, "Species", seed = 42)
#' model$scores_
ml_cv <- function(s, target, folds = 5L, seed = NULL, stratify = TRUE) {
  if (!inherits(s, "ml_split_result")) {
    config_error("ml_cv() requires an ml_split_result from ml_split().")
  }
  if (isTRUE(s$.temporal)) {
    config_error(
      "temporal split detected -- use ml_cv_temporal() for time-series CV instead."
    )
  }
  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)
  folds <- as.integer(folds)
  if (folds < 2L) config_error("folds must be an integer >= 2")

  # Use dev = train + valid

  dev_data <- s$dev

  if (!target %in% names(dev_data)) {
    config_error(paste0(
      "target='", target, "' not found in data. Available columns: ",
      paste(names(dev_data), collapse = ", ")
    ))
  }

  n_dev <- nrow(dev_data)
  if (folds > n_dev) {
    config_error(paste0(
      "Cannot create ", folds, " folds from ", n_dev, " dev rows. ",
      "Use folds=", min(n_dev, 10L), " or fewer."
    ))
  }

  withr::local_seed(seed)

  y <- dev_data[[target]]
  task <- .detect_task(y, "auto")

  if (stratify && task == "classification") {
    fold_ids <- .stratified_kfold(y, folds)
  } else {
    fold_ids <- sample(rep(seq_len(folds), length.out = n_dev))
  }

  fold_list <- lapply(seq_len(folds), function(k) {
    list(
      train = which(fold_ids != k),
      valid = which(fold_ids == k)
    )
  })

  cv_result <- list(
    folds  = fold_list,
    data   = dev_data,
    target = target
  )
  class(cv_result) <- "ml_cv_result"
  cv_result
}

#' Create temporal cross-validation from a split
#'
#' Expanding-window CV for time series. Data must already be sorted
#' chronologically (use `ml_split_temporal()` first).
#'
#' @param s An `ml_split_result` from `ml_split_temporal()`
#' @param target Target column name (string)
#' @param folds Number of folds (default 5)
#' @returns An `ml_cv_result` with expanding-window folds
#' @export
#' @examples
#' @param embargo Integer. Number of rows to skip between train end and valid
#'   start (gap to prevent temporal leakage from autocorrelation). Default 0.
#'   Must be >= 0.
#' @param window `"expanding"` (default, all prior rows as train) or
#'   `"sliding"` (fixed-size training window). When `"sliding"`,
#'   `window_size` must also be supplied.
#' @param window_size Integer. Required when `window = "sliding"`. Number of
#'   rows in each training window.
#' @examples
#' df <- data.frame(date = 1:100, x = rnorm(100), y = sample(0:1, 100, TRUE))
#' s <- ml_split_temporal(df, "y", time = "date")
#' c <- ml_cv_temporal(s, "y", folds = 5)
#' # With embargo to prevent autocorrelation leakage:
#' c2 <- ml_cv_temporal(s, "y", folds = 5, embargo = 5L)
ml_cv_temporal <- function(s, target, folds = 5L, embargo = 0L,
                            window = "expanding", window_size = NULL) {
  if (!inherits(s, "ml_split_result")) {
    config_error("ml_cv_temporal() requires an ml_split_result.")
  }
  folds   <- as.integer(folds)
  embargo <- as.integer(embargo)
  if (folds < 2L) config_error("folds must be an integer >= 2")
  if (embargo < 0L) {
    config_error(paste0(
      "embargo must be >= 0, got ", embargo, ". ",
      "Negative embargo would create temporal leakage."
    ))
  }
  if (!window %in% c("expanding", "sliding")) {
    config_error(paste0(
      "window='", window, "' not valid. Choose from: 'expanding', 'sliding'."
    ))
  }
  if (identical(window, "sliding") && is.null(window_size)) {
    config_error(paste0(
      "window_size= is required when window='sliding'. ",
      "Example: ml_cv_temporal(s, 'y', folds=5, window='sliding', window_size=100)"
    ))
  }

  dev_data <- s$dev
  n_dev    <- nrow(dev_data)

  if (folds >= n_dev) {
    config_error(paste0(
      "Cannot create ", folds, " temporal folds from ", n_dev, " dev rows."
    ))
  }

  chunk_size <- n_dev %/% (folds + 1L)
  if (chunk_size < 1L) {
    config_error(paste0(
      "Too many folds (", folds, ") for ", n_dev, " dev rows. ",
      "Use folds=", max(2L, n_dev %/% 2L - 1L), " or fewer."
    ))
  }

  fold_list <- vector("list", folds)
  n_valid_folds <- 0L

  for (k in seq_len(folds)) {
    if (!is.null(window_size) && identical(window, "sliding")) {
      train_start <- max(1L, chunk_size * k - as.integer(window_size) + 1L)
    } else {
      train_start <- 1L
    }
    train_end   <- chunk_size * k
    valid_start <- train_end + embargo + 1L
    valid_end   <- if (k == folds) n_dev else min(train_end + chunk_size, n_dev)

    if (valid_start > valid_end) next  # embargo consumed all validation data

    fold_list[[n_valid_folds + 1L]] <- list(
      train = .safe_seq(train_start, train_end),
      valid = .safe_seq(valid_start, valid_end)
    )
    n_valid_folds <- n_valid_folds + 1L
  }

  if (n_valid_folds == 0L) {
    config_error(paste0(
      "embargo (", embargo, ") is too large -- no validation data remains. ",
      "Reduce embargo or use fewer folds."
    ))
  }

  fold_list <- fold_list[seq_len(n_valid_folds)]

  cv_result <- list(
    folds  = fold_list,
    data   = dev_data,
    target = target
  )
  class(cv_result) <- "ml_cv_result"
  cv_result
}

#' Create group-aware cross-validation from a split
#'
#' No group appears in both train and validation within any fold.
#' Prevents leakage from repeated measurements (patients, stores, sensors).
#'
#' @param s An `ml_split_result` from `ml_split()` or `ml_split_group()`
#' @param target Target column name (string)
#' @param groups Column name identifying groups
#' @param folds Number of folds (default 5)
#' @param seed Random seed for group assignment
#' @returns An `ml_cv_result` with group-aware folds
#' @export
#' @examples
#' df <- data.frame(pid = rep(1:20, each = 5), x = rnorm(100), y = sample(0:1, 100, TRUE))
#' s <- ml_split(df, "y", seed = 42)
#' c <- ml_cv_group(s, "y", groups = "pid", folds = 5, seed = 42)
ml_cv_group <- function(s, target, groups, folds = 5L, seed = NULL) {
  if (!inherits(s, "ml_split_result")) {
    config_error("ml_cv_group() requires an ml_split_result.")
  }
  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)
  folds <- as.integer(folds)
  if (folds < 2L) config_error("folds must be an integer >= 2")

  dev_data <- s$dev

  if (!groups %in% names(dev_data)) {
    config_error(paste0(
      "groups='", groups, "' not found in data. Available columns: ",
      paste(names(dev_data), collapse = ", ")
    ))
  }

  grp_col <- dev_data[[groups]]
  if (any(is.na(grp_col))) {
    config_error("groups column contains NA values. Drop or impute before CV.")
  }

  unique_grps <- sort(unique(grp_col))
  n_grps <- length(unique_grps)

  if (folds > n_grps) {
    config_error(paste0(
      "Cannot create ", folds, " folds from ", n_grps, " groups. ",
      "Use folds=", min(n_grps, 10L), " or fewer."
    ))
  }

  withr::local_seed(seed)
  perm_grps <- sample(unique_grps)

  # Round-robin assignment of groups to folds
  grp_fold_ids <- rep(seq_len(folds), length.out = n_grps)
  row_fold_ids <- grp_fold_ids[match(as.character(grp_col), as.character(perm_grps))]

  fold_list <- lapply(seq_len(folds), function(k) {
    list(
      train = which(row_fold_ids != k),
      valid = which(row_fold_ids == k)
    )
  })

  cv_result <- list(
    folds  = fold_list,
    data   = dev_data,
    target = target
  )
  class(cv_result) <- "ml_cv_result"
  cv_result
}

# Block access to train/valid/test/dev — CVResult exposes only folds/data/target
#' @export
`$.ml_cv_result` <- function(x, name) {
  if (name %in% c("train", "valid", "dev")) {
    config_error(paste0(
      "No $", name, " on a CVResult. CVResult exposes $folds and $data only. ",
      "Train/valid splits are inside each fold: cv$folds[[1]]$train"
    ))
  }
  if (identical(name, "test")) {
    config_error(
      "CVResult has no $test partition. Test stays on the SplitResult: s$test"
    )
  }
  .subset2(x, name)
}

#' Print ml_cv_result
#' @param x An ml_cv_result object
#' @param ... Ignored
#' @returns The object \code{x}, invisibly.
#' @export
print.ml_cv_result <- function(x, ...) {
  k <- length(x[["folds"]])
  n <- nrow(x[["data"]])
  cat(sprintf("-- CV [%d folds, %d rows] --\n", k, n))
  cat(sprintf("  target : %s\n", x[["target"]]))
  # Show fold sizes
  for (i in seq_len(min(k, 3L))) {
    f <- x[["folds"]][[i]]
    cat(sprintf("  fold %d : %d train, %d valid\n", i, length(f$train), length(f$valid)))
  }
  if (k > 3L) cat(sprintf("  ... (%d more folds)\n", k - 3L))
  cat("\n")
  invisible(x)
}
