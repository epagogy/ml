# ── Content-Addressed Partition Identity ──────────────────────────────────────
#
# Survives dplyr, merge, rbind — anything that strips attributes.
# Mirrors Python's _provenance.py: fingerprint → registry → guards query by hash.
#
# Design: rlang::hash() on sorted column names + column values gives a
# deterministic identity. Package-level environment acts as session-scoped
# registry (same as Python's module-level singleton).

# Package-level registry — survives across function calls within a session
.partition_registry <- new.env(parent = emptyenv())
.partition_registry$store   <- list()  # fingerprint → role
.partition_registry$lineage <- list()  # fingerprint → split_id
.partition_registry$receipts <- list()  # split_id → receipt hash
.partition_registry$assessed <- character(0)  # fingerprints of assessed test partitions
.partition_registry$max_entries <- 10000L

#' @keywords internal
.fingerprint <- function(df) {
  # Content-addressed fingerprint of a data.frame.
  # Same data = same fingerprint regardless of attributes/rownames.
  # Uses rlang::hash() which is fast and deterministic within an R session.
  n <- nrow(df)
  cols <- sort(names(df))

  if (n == 0L) {
    return(rlang::hash(list("empty", cols)))
  }

  # For large datasets, stride-sample like Python (>100K rows)
  if (n > 100000L) {
    stride <- max(1L, n %/% 2000L)
    idx <- seq(1L, n, by = stride)
    sample_df <- df[idx, cols, drop = FALSE]
    return(rlang::hash(list(sample_df, cols, dim(df))))
  }

  # Hash sorted columns for column-order agnosticism
  rlang::hash(list(df[, cols, drop = FALSE], cols))
}

#' @keywords internal
.register_partition <- function(df, role, split_id) {
  fp <- .fingerprint(df)
  .partition_registry$store[[fp]]   <- role
  .partition_registry$lineage[[fp]] <- split_id

  # Evict oldest if too large
  if (length(.partition_registry$store) > .partition_registry$max_entries) {
    oldest <- names(.partition_registry$store)[[1L]]
    .partition_registry$store[[oldest]]   <- NULL
    .partition_registry$lineage[[oldest]] <- NULL
  }

  fp
}

#' @keywords internal
.identify_partition <- function(df) {
  fp <- .fingerprint(df)
  role <- .partition_registry$store[[fp]]
  if (is.null(role)) return(NULL)
  role
}

#' @keywords internal
.new_split_id <- function() {
  paste0(
    format(Sys.time(), "%Y%m%d%H%M%S"),
    sprintf("%04x", sample.int(65535L, 1L))
  )
}

#' Mark a test partition as assessed (per-holdout enforcement)
#' @keywords internal
.mark_assessed <- function(df) {
  fp <- .fingerprint(df)
  if (!fp %in% .partition_registry$assessed) {
    .partition_registry$assessed <- c(.partition_registry$assessed, fp)
  }
  invisible(NULL)
}

#' Check if a test partition has already been assessed by any model
#' @keywords internal
.is_assessed <- function(df) {
  fp <- .fingerprint(df)
  fp %in% .partition_registry$assessed
}

#' @keywords internal
.clear_registry <- function() {
  .partition_registry$store   <- list()
  .partition_registry$lineage <- list()
  .partition_registry$receipts <- list()
  .partition_registry$assessed <- character(0)
  invisible(NULL)
}

#' @keywords internal
.registry_size <- function() {

  length(.partition_registry$store)
}

#' Resolve partition role: fingerprint first, attr fallback
#' @keywords internal
.resolve_partition <- function(df) {
  # Layer 1: content-addressed (survives dplyr/merge/rbind)
  role <- .identify_partition(df)
  if (!is.null(role)) return(role)

  # Layer 2: attribute (cheap, but stripped by dplyr)
  attr(df, "_ml_partition")
}

# ── Layer 2: Cross-Verb Provenance ───────────────────────────────────────────
#
# fit() stores training fingerprint + split lineage in the model.
# assess() verifies test data comes from the same split.
# Catches split-shopping: fit on split A, assess on split B's test set.

#' Build provenance metadata from training data for storage in a Model.
#' @keywords internal
.build_provenance <- function(data) {
  fp <- .fingerprint(data)
  split_id <- .partition_registry$lineage[[fp]]
  # Include split receipt if available (cross-language verifiable identity)
  receipt <- if (!is.null(split_id)) .partition_registry$receipts[[split_id]] else NULL
  list(
    train_fingerprint = fp,
    split_id = split_id,
    split_receipt = receipt,
    fit_timestamp = as.numeric(Sys.time())
  )
}

#' Check cross-verb provenance. Errors on split-shopping.
#' @keywords internal
.check_provenance <- function(model_provenance, test) {
  if (is.null(model_provenance)) return(invisible(NULL))

  # Check: test data from same split as training data
  test_fp <- .fingerprint(test)
  test_split <- .partition_registry$lineage[[test_fp]]
  train_split <- model_provenance$split_id

  if (!is.null(test_split) && !is.null(train_split) && test_split != train_split) {
    .guard_action(paste0(
      "ml_assess() test data comes from a different split than ml_fit() training data. ",
      "This enables test-set shopping. Use test data from the same split: ",
      "s <- ml_split(...); model <- ml_fit(s$train, ...); ml_assess(model, test = s$test)"
    ))
  }

  invisible(NULL)
}
