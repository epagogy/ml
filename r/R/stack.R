#' Ensemble stacking
#'
#' Trains a stacking ensemble with out-of-fold meta-features. Base models
#' generate out-of-fold predictions, which are used to train a meta-learner.
#'
#' **Note:** This function uses global normalization (not per-fold), because
#' the stacking CV is internal to the meta-learner training. This is the
#' one exception to the per-fold normalization rule.
#'
#' @param data A data.frame with features and target
#' @param target Target column name
#' @param models Character vector of base algorithm names, or NULL for defaults
#' @param meta Meta-learner algorithm. Default: "logistic" (classification) or
#'   "linear" (regression)
#' @param cv_folds Number of CV folds for generating out-of-fold predictions
#' @param seed Random seed
#' @returns An `ml_model` with `$is_stacked = TRUE`
#' @export
#' @examples
#' \donttest{
#' s <- ml_split(iris, "Species", seed = 42)
#' stacked <- ml_stack(s$train, "Species", seed = 42)
#' predict(stacked, s$valid)
#' }
ml_stack <- function(data, target, models = NULL, meta = NULL,
                     cv_folds = 5L, seed = NULL) {
  .stack_impl(data = data, target = target, models = models, meta = meta,
              cv_folds = cv_folds, seed = seed)
}

.stack_impl <- function(data, target, models = NULL, meta = NULL,
                         cv_folds = 5L, seed = NULL) {
  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)

  data <- .coerce_data(data)

  # Partition guard — stack is a training verb
  if (.guards_active()) {
    .part <- .resolve_partition(data)
    if (!is.null(.part) && !.part %in% c("train", "valid", "dev")) {
      .guard_action(paste0(
        "ml_stack() received data tagged as '", .part, "' partition. ",
        "ml_stack() accepts train, valid, or dev data. ",
        "Use: ml_stack(s$train, ...) or ml_stack(s$dev, ...)"
      ))
    }
  }

  if (!target %in% names(data)) data_error(paste0("target='", target, "' not found in data"))

  y            <- data[[target]]
  detected_task <- .detect_task(y)
  features     <- setdiff(names(data), target)
  n            <- nrow(data)

  # Default base algorithms — avoid classification-only for regression
  if (is.null(models)) {
    avail  <- .available_algorithms()
    if (detected_task == "classification") {
      models <- intersect(avail, c("xgboost", "random_forest", "logistic"))
    } else {
      models <- intersect(avail, c("xgboost", "random_forest", "linear"))
    }
    if (length(models) == 0L) models <- avail[seq_len(min(2L, length(avail)))]
  }

  # Default meta-learner
  if (is.null(meta)) {
    meta <- if (detected_task == "classification") "logistic" else "linear"
  }

  # For regression with logistic: override to linear
  if (detected_task == "regression" && meta == "logistic") meta <- "linear"

  # GLOBAL normalization (exception to per-fold rule — see spec)
  X_all   <- data[, features, drop = FALSE]
  fit_out <- .transform_fit(X_all, .prepare(X_all, y, algorithm = models[[1L]], task = detected_task))
  X_enc   <- fit_out$X
  norm    <- fit_out$norm
  y_enc   <- .encode_target(y, norm)

  withr::local_seed(seed)
  fold_ids <- sample(rep(seq_len(cv_folds), length.out = n))

  # Generate OOF (out-of-fold) meta-features
  # Classification with proba-capable algos: use probabilities (better meta-features)
  # Regression / non-proba algos: use raw predictions
  n_classes <- if (detected_task == "classification") length(unique(y_enc)) else 0L
  use_proba <- detected_task == "classification"

  # Compute columns per model: binary clf=1 prob col, multiclass=n_classes, regression=1
  cols_per_model <- if (use_proba && n_classes > 2L) n_classes else 1L
  total_cols     <- length(models) * cols_per_model

  oof_preds <- matrix(NA_real_, nrow = n, ncol = total_cols)
  col_names <- character(total_cols)
  for (j in seq_along(models)) {
    start_col <- (j - 1L) * cols_per_model + 1L
    if (cols_per_model == 1L) {
      col_names[start_col] <- paste0("base_", j)
    } else {
      col_names[start_col:(start_col + cols_per_model - 1L)] <- paste0("base_", j, "_c", seq_len(cols_per_model))
    }
  }
  colnames(oof_preds) <- col_names

  base_engines <- vector("list", length(models))

  for (j in seq_along(models)) {
    algo <- models[[j]]

    # Validate compatibility
    if (detected_task == "regression" && algo %in% c("logistic", "naive_bayes")) next
    if (detected_task == "classification" && algo == "linear") next

    algo_has_proba <- algo %in% .PROBA_ALGORITHMS
    start_col <- (j - 1L) * cols_per_model + 1L

    for (k in seq_len(cv_folds)) {
      tr_idx  <- which(fold_ids != k)
      val_idx <- which(fold_ids == k)

      X_fold_tr  <- X_enc[tr_idx, , drop = FALSE]
      y_fold_tr  <- y_enc[tr_idx]
      X_fold_val <- X_enc[val_idx, , drop = FALSE]

      tryCatch({
        eng <- .fit_engine(X_fold_tr, y_fold_tr, detected_task, seed + k + j * 100L, algo)
        if (use_proba && algo_has_proba) {
          pm <- .predict_proba_engine(eng, X_fold_val, detected_task, algo)
          pm <- as.matrix(pm)
          if (n_classes == 2L) {
            # Binary: use positive-class probability only
            oof_preds[val_idx, start_col] <- pm[, ncol(pm)]
          } else {
            # Multiclass: use all class probabilities
            end_col <- start_col + cols_per_model - 1L
            oof_preds[val_idx, start_col:end_col] <- pm
          }
        } else {
          p <- .predict_engine(eng, X_fold_val, detected_task, algo)
          oof_preds[val_idx, start_col] <- as.numeric(p)
        }
      }, error = function(e) NULL)
    }

    # Fit base model on ALL encoded data
    base_engines[[j]] <- tryCatch(
      .fit_engine(X_enc, y_enc, detected_task, seed + j, algo),
      error = function(e) NULL
    )
  }

  # Impute any remaining NA in OOF (failed folds)
  for (j in seq_len(ncol(oof_preds))) {
    na_idx <- is.na(oof_preds[, j])
    if (any(na_idx)) {
      col_mean <- mean(oof_preds[, j], na.rm = TRUE)
      oof_preds[na_idx, j] <- if (is.nan(col_mean)) 0 else col_mean
    }
  }

  # Remove all-NA columns (base models that completely failed)
  valid_cols <- which(colSums(!is.na(oof_preds)) > 0)
  if (length(valid_cols) == 0L) valid_cols <- seq_len(ncol(oof_preds))
  oof_preds  <- oof_preds[, valid_cols, drop = FALSE]

  # Train meta-learner on OOF predictions
  # Use DECODED original labels for classification meta-learner
  oof_df <- as.data.frame(oof_preds)
  if (detected_task == "classification") {
    meta_y <- .decode(y_enc, norm)
    meta_y_factor <- factor(meta_y)
    # Check we have enough classes for the meta-learner
    if (length(levels(meta_y_factor)) < 2L) {
      # Fallback: use simple majority prediction
      meta_engine <- list(majority = names(sort(table(meta_y), decreasing = TRUE))[[1]])
      meta_algo   <- "majority"
    } else {
      meta_norm_fit <- .transform_fit(oof_df, .prepare(oof_df, meta_y, algorithm = meta,
                                                        task = detected_task))
      meta_X_enc    <- meta_norm_fit$X
      meta_norm     <- meta_norm_fit$norm
      meta_y_enc    <- .encode_target(meta_y, meta_norm)
      meta_engine   <- tryCatch(
        .fit_engine(meta_X_enc, meta_y_enc, detected_task, seed, meta),
        error = function(e) {
          # Fallback: majority class
          list(majority = names(sort(table(meta_y), decreasing = TRUE))[[1]])
        }
      )
      meta_algo <- meta
    }
  } else {
    meta_y <- as.numeric(y)
    meta_norm_fit <- .transform_fit(oof_df, .prepare(oof_df, meta_y, algorithm = meta,
                                                      task = detected_task))
    meta_X_enc  <- meta_norm_fit$X
    meta_norm   <- meta_norm_fit$norm
    meta_engine <- tryCatch(
      .fit_engine(meta_X_enc, meta_y, detected_task, seed, meta),
      error = function(e) NULL
    )
    meta_algo <- meta
  }

  if (is.null(meta_norm_fit)) meta_norm <- NULL

  t_hash <- .make_hash("stack", detected_task, target, seed, nrow(data), features)

  model <- new_ml_model(
    engine         = list(
      base_models    = base_engines,
      base_algos     = models[valid_cols],
      base_norm      = norm,
      meta_engine    = meta_engine,
      meta_algo      = meta_algo,
      meta_norm      = if (exists("meta_norm")) meta_norm else NULL,
      use_proba      = use_proba,
      n_classes      = n_classes,
      cols_per_model = cols_per_model
    ),
    task           = detected_task,
    algorithm      = "stack",
    target         = target,
    features       = features,
    seed           = seed,
    n_train        = nrow(data),
    hash           = t_hash,
    encoders       = norm
  )
  model[["is_stacked"]] <- TRUE
  model
}
