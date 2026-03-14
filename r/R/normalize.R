# ── Constants ──────────────────────────────────────────────────────────────────

.ONEHOT_MAX_CARDINALITY <- 50L
.NAN_SENTINEL <- "__ml_nan__"
.TREE_ALGORITHMS <- c("xgboost", "random_forest")

# ── Task detection ─────────────────────────────────────────────────────────────

#' Detect task type from target vector
#' @keywords internal
.detect_task <- function(y, task = "auto") {
  if (task != "auto") {
    if (!task %in% c("classification", "regression")) {
      config_error(paste0("task must be 'auto', 'classification', or 'regression', got '", task, "'"))
    }
    return(task)
  }
  # Factor, character, logical → classification
  if (is.factor(y) || is.character(y) || is.logical(y)) return("classification")
  # Numeric: heuristic (≤20 unique AND <5% unique ratio → classification)
  n_unique <- length(unique(y[!is.na(y)]))
  if (n_unique <= 20L && n_unique / length(y) < 0.05) return("classification")
  "regression"
}

# ── Data coercion ──────────────────────────────────────────────────────────────

#' Coerce tibble/data.table to data.frame
#' @keywords internal
.coerce_data <- function(data) {
  if (inherits(data, "tbl_df"))     data <- as.data.frame(data)
  if (inherits(data, "data.table")) data <- as.data.frame(data)
  if (!is.data.frame(data)) data_error(paste0("Expected data.frame, got ", class(data)[[1]]))
  data
}

# ── prepare() ─────────────────────────────────────────────────────────────────
#
# Called ONCE per training set (or once per CV fold from training rows only).
# Returns a NormState list with all encoding/scaling state.
# Key invariant: stats come from TRAINING data only.

#' Fit encoding and scaling state from training data
#'
#' @param X Feature matrix (data.frame, numeric and/or character/factor columns)
#' @param y Target vector
#' @param algorithm Algorithm name (determines encoding/scaling strategy)
#' @param task "classification" or "regression"
#' @returns A named list (NormState) for use with .transform() and .encode_target()
#' @keywords internal
.prepare <- function(X, y, algorithm = "auto", task = "auto") {
  use_onehot <- algorithm %in% .LINEAR_ALGORITHMS
  use_scale  <- algorithm %in% .LINEAR_ALGORITHMS &&
                !algorithm %in% .NO_SCALE_ALGORITHMS

  # --- Label encoding for target -----------------------------------------
  label_map     <- NULL
  label_levels  <- NULL

  if (task == "classification" || task == "auto") {
    if (is.factor(y) || is.character(y) || is.logical(y)) {
      lvls <- sort(unique(as.character(y[!is.na(y)])))
      label_map <- stats::setNames(seq_along(lvls) - 1L, lvls)
      label_levels <- lvls
    } else if (task != "regression") {
      # Numeric classification: remap labels to 0..K-1 (required by XGBoost)
      raw_levels <- sort(unique(y[!is.na(y)]))
      label_map  <- stats::setNames(seq_along(raw_levels) - 1L, as.character(raw_levels))
      label_levels <- as.character(raw_levels)
    }
  }

  # --- Feature column classification ------------------------------------
  cat_cols   <- character(0)
  num_cols   <- character(0)
  order_cols <- character(0)  # ordered factors → always ordinal

  for (col in names(X)) {
    v <- X[[col]]
    if (is.ordered(v)) {
      order_cols <- c(order_cols, col)
    } else if (is.factor(v) || is.character(v) || is.logical(v)) {
      cat_cols <- c(cat_cols, col)
    } else {
      num_cols <- c(num_cols, col)
    }
  }

  # --- Ordinal encoding -------------------------------------------------
  category_maps   <- list()
  category_levels <- list()

  # Always ordinal for ordered factors
  for (col in order_cols) {
    v <- X[[col]]
    lvls <- if (is.ordered(v)) levels(v) else sort(unique(as.character(v[!is.na(v)])))
    mp <- stats::setNames(seq_along(lvls) - 1L, lvls)
    category_maps[[col]]   <- mp
    category_levels[[col]] <- lvls
  }

  # For tree algorithms, all unordered categoricals also get ordinal encoding
  if (!use_onehot) {
    for (col in cat_cols) {
      v    <- X[[col]]
      lvls <- sort(unique(as.character(v[!is.na(v)])))
      mp   <- stats::setNames(seq_along(lvls) - 1L, lvls)
      category_maps[[col]]   <- mp
      category_levels[[col]] <- lvls
    }
  }

  # --- One-hot encoding (linear models only) ----------------------------
  onehot_cols    <- character(0)
  onehot_levels  <- list()
  high_card_cols <- character(0)

  if (use_onehot) {
    for (col in cat_cols) {
      v    <- X[[col]]
      lvls <- sort(unique(as.character(v[!is.na(v)])))
      card <- length(lvls)
      if (card <= .ONEHOT_MAX_CARDINALITY) {
        onehot_cols           <- c(onehot_cols, col)
        onehot_levels[[col]]  <- lvls
      } else {
        high_card_cols <- c(high_card_cols, col)
        # Fall back to ordinal for high-cardinality
        mp <- stats::setNames(seq_along(lvls) - 1L, lvls)
        category_maps[[col]]   <- mp
        category_levels[[col]] <- lvls
        cli::cli_warn(paste0(
          "'", col, "' high cardinality (", card, " unique values > ", .ONEHOT_MAX_CARDINALITY,
          ") -- using ordinal encoding instead of one-hot."
        ))
      }
    }
  }

  # --- Imputation state (non-tree algorithms only) ----------------------
  medians <- NULL
  if (!algorithm %in% .TREE_ALGORITHMS) {
    # Compute medians on NUMERIC columns only (post ordinal-encode, pre-scale)
    # Will be filled properly after ordinal encoding in .transform_fit()
    medians <- list()
  }

  # --- Scaling state (will be computed after encoding in .transform_fit) -
  center <- NULL
  scale  <- NULL

  list(
    label_map      = label_map,
    label_levels   = label_levels,
    category_maps  = category_maps,
    category_levels = category_levels,
    onehot_cols    = onehot_cols,
    onehot_levels  = onehot_levels,
    medians        = medians,
    center         = center,
    scale          = scale,
    algorithm      = algorithm,
    task           = task,
    use_onehot     = use_onehot,
    use_scale      = use_scale
  )
}

# ── transform_fit() ────────────────────────────────────────────────────────────
#
# Apply encoding to training data AND compute/store imputation + scaling state.
# Returns list(X_enc, norm_state) where norm_state now has medians/center/scale.

#' Fit and apply encoding to training features
#' @keywords internal
.transform_fit <- function(X, norm) {
  X <- .coerce_data(X)

  # --- Inf check --------------------------------------------------------
  num_X <- X[, vapply(X, is.numeric, logical(1L)), drop = FALSE]
  if (ncol(num_X) > 0) {
    inf_mask <- is.infinite(as.matrix(num_X))
    if (any(inf_mask)) {
      data_error("Infinite values (Inf/-Inf) found in features. Replace with finite values before training.")
    }
  }

  # --- Ordinal encoding ------------------------------------------------
  X <- .apply_ordinal(X, norm$category_maps)

  # --- One-hot encoding ------------------------------------------------
  if (norm$use_onehot && length(norm$onehot_cols) > 0) {
    out <- .apply_onehot_fit(X, norm$onehot_cols, norm$onehot_levels)
    X   <- out$X
    norm$onehot_expanded_cols <- out$expanded_cols
  }

  # --- Imputation (non-tree) -------------------------------------------
  if (!is.null(norm$medians)) {
    num_cols  <- names(X)[vapply(X, is.numeric, logical(1L))]
    meds      <- vapply(num_cols, function(col) {
      stats::median(X[[col]], na.rm = TRUE)
    }, numeric(1L))
    meds[is.nan(meds)] <- 0  # all-NA column
    norm$medians <- as.list(stats::setNames(meds, num_cols))
    na_count <- sum(is.na(as.matrix(X[, num_cols, drop = FALSE])))
    if (na_count > 0) {
      cli::cli_warn(paste0(na_count, " NA value(s) in features -- auto-imputed with column medians."))
    }
    X <- .apply_imputation(X, norm$medians)
  }

  # --- Scaling ---------------------------------------------------------
  if (norm$use_scale) {
    num_cols <- names(X)[vapply(X, is.numeric, logical(1L))]
    norm$center <- vapply(num_cols, function(col) mean(X[[col]], na.rm = TRUE), numeric(1L))
    norm$scale  <- vapply(num_cols, function(col) {
      s <- stats::sd(X[[col]], na.rm = TRUE)
      if (is.na(s) || s == 0) 1 else s
    }, numeric(1L))
    norm$scale_cols <- num_cols
    X <- .apply_scaling(X, norm$center, norm$scale, num_cols)
  }

  norm$feature_names <- names(X)
  list(X = X, norm = norm)
}

# ── .transform() ──────────────────────────────────────────────────────────────
#
# Apply STORED state to new data (validation / prediction path).

#' Apply stored encoding + scaling to new features
#' @keywords internal
.transform <- function(X, norm) {
  X <- .coerce_data(X)

  # --- Inf check --------------------------------------------------------
  num_X <- X[, vapply(X, is.numeric, logical(1L)), drop = FALSE]
  if (ncol(num_X) > 0) {
    inf_mask <- is.infinite(as.matrix(num_X))
    if (any(inf_mask)) {
      data_error("Infinite values (Inf/-Inf) found in features. Replace with finite values before prediction.")
    }
  }

  # --- NA check for predict (warning only) ------------------------------
  na_count <- sum(is.na(as.matrix(X)))
  if (na_count > 0) {
    cli::cli_warn(paste0(na_count, " row(s) contain NA. Predictions may be unreliable."))
  }

  # --- Ordinal encoding ------------------------------------------------
  X <- .apply_ordinal(X, norm$category_maps)

  # --- One-hot encoding ------------------------------------------------
  if (isTRUE(norm$use_onehot) && length(norm$onehot_cols) > 0) {
    X <- .apply_onehot_transform(X, norm$onehot_cols, norm$onehot_levels,
                                  norm$onehot_expanded_cols)
  }

  # --- Imputation (non-tree) -------------------------------------------
  if (!is.null(norm$medians) && length(norm$medians) > 0) {
    X <- .apply_imputation(X, norm$medians)
  }

  # --- Scaling ---------------------------------------------------------
  if (isTRUE(norm$use_scale) && !is.null(norm$center)) {
    X <- .apply_scaling(X, norm$center, norm$scale, norm$scale_cols)
  }

  # --- Column alignment -----------------------------------------------
  if (!is.null(norm$feature_names)) {
    missing_cols <- setdiff(norm$feature_names, names(X))
    if (length(missing_cols) > 0) {
      data_error(paste0("Missing features in prediction data: ", paste(missing_cols, collapse = ", ")))
    }
    X <- X[, norm$feature_names, drop = FALSE]
  }

  X
}

# ── .encode_target() ──────────────────────────────────────────────────────────

#' Encode target vector using stored label_map
#' @keywords internal
.encode_target <- function(y, norm) {
  if (is.null(norm$label_map)) return(as.numeric(y))
  y_chr <- as.character(y)
  encoded <- norm$label_map[y_chr]
  # Unknown labels → NA; set to 0 as fallback
  encoded[is.na(encoded)] <- 0L
  as.integer(encoded)
}

# ── .decode() ─────────────────────────────────────────────────────────────────

#' Decode integer predictions back to original labels
#' @keywords internal
.decode <- function(predictions, norm) {
  if (is.null(norm$label_levels)) return(predictions)
  # predictions are 0-based integer indices
  lvls <- norm$label_levels
  decoded <- lvls[as.integer(predictions) + 1L]
  decoded[is.na(decoded)] <- as.character(predictions[is.na(decoded)])
  # Return factor if original was factor, else character
  decoded
}

# ── Private helpers ────────────────────────────────────────────────────────────

.apply_ordinal <- function(X, category_maps) {
  if (length(category_maps) == 0) return(X)
  for (col in names(category_maps)) {
    if (!col %in% names(X)) next
    v    <- as.character(X[[col]])
    mp   <- category_maps[[col]]
    # 3-step ordinal:
    # 1. Store original NA mask
    # 2. Map known values; unmapped → NA
    # 3. Fill NA from mapping miss with -1, then restore original NA mask
    orig_na <- is.na(X[[col]])
    encoded <- mp[v]
    # names() lookup returns NA for unknown values
    unseen_mask <- is.na(encoded) & !orig_na
    encoded[unseen_mask] <- -1L
    encoded[orig_na]     <- NA_integer_
    X[[col]] <- as.integer(encoded)
  }
  X
}

.apply_onehot_fit <- function(X, onehot_cols, onehot_levels) {
  # Remove original categorical columns, add dummy columns
  expanded_cols <- character(0)
  for (col in onehot_cols) {
    if (!col %in% names(X)) next
    lvls <- onehot_levels[[col]]
    # Fill NA with sentinel before encoding
    v <- as.character(X[[col]])
    v[is.na(v)] <- .NAN_SENTINEL

    for (lvl in lvls) {
      new_col  <- paste0(col, "_", lvl)
      # Collision detection
      if (new_col %in% names(X) && !new_col %in% onehot_cols) {
        cli::cli_warn(paste0(
          "One-hot column name '", new_col,
          "' collides with existing feature. Predictions may be incorrect."
        ))
      }
      X[[new_col]] <- as.integer(v == lvl)
      expanded_cols <- c(expanded_cols, new_col)
    }
    X[[col]] <- NULL
  }
  list(X = X, expanded_cols = expanded_cols)
}

.apply_onehot_transform <- function(X, onehot_cols, onehot_levels, expanded_cols) {
  for (col in onehot_cols) {
    if (!col %in% names(X)) next
    lvls <- onehot_levels[[col]]
    v <- as.character(X[[col]])
    v[is.na(v)] <- .NAN_SENTINEL
    for (lvl in lvls) {
      new_col <- paste0(col, "_", lvl)
      X[[new_col]] <- as.integer(v == lvl)
    }
    X[[col]] <- NULL
  }
  # Ensure same columns as training (fill missing with 0)
  if (!is.null(expanded_cols)) {
    for (col in expanded_cols) {
      if (!col %in% names(X)) X[[col]] <- 0L
    }
  }
  X
}

.apply_imputation <- function(X, medians) {
  for (col in names(medians)) {
    if (!col %in% names(X)) next
    na_mask <- is.na(X[[col]])
    if (any(na_mask)) X[[col]][na_mask] <- medians[[col]]
  }
  X
}

.apply_scaling <- function(X, center, scale, scale_cols) {
  for (col in scale_cols) {
    if (!col %in% names(X)) next
    X[[col]] <- (X[[col]] - center[[col]]) / scale[[col]]
  }
  X
}

# ── Zero-variance check ────────────────────────────────────────────────────────

.check_zero_variance <- function(X) {
  num_cols <- names(X)[vapply(X, is.numeric, logical(1L))]
  zv <- vapply(num_cols, function(col) {
    v <- X[[col]]
    stats::var(v, na.rm = TRUE) == 0 || all(is.na(v))
  }, logical(1L))
  zero_var_cols <- num_cols[zv]
  if (length(zero_var_cols) > 0) {
    cli::cli_warn(paste0(
      length(zero_var_cols), " feature(s) have zero variance: ",
      paste(zero_var_cols, collapse = ", ")
    ))
  }
  invisible(NULL)
}

# ── Duplicate column check ─────────────────────────────────────────────────────

.check_duplicate_cols <- function(data) {
  dupes <- names(data)[duplicated(names(data))]
  if (length(dupes) > 0) {
    data_error(paste0("Duplicate column names: ", paste(unique(dupes), collapse = ", ")))
  }
}
