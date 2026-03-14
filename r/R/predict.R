#' Predict from a fitted model
#'
#' @param object An `ml_model` object
#' @param newdata A data.frame with the same features used for training
#' @param proba Logical. If TRUE, return class probabilities instead of labels
#'   (classification only). Default FALSE.
#' @param ... Ignored
#' @returns A vector of predicted class labels (classification) or numeric values
#'   (regression). If `proba = TRUE`, returns a data.frame with one column per
#'   class; values are probabilities summing to 1.0 per row.
#' @export
#' @examples
#' s <- ml_split(iris, "Species", seed = 42)
#' model <- ml_fit(s$train, "Species", seed = 42)
#' preds <- predict(model, newdata = s$valid)
#' head(preds)
predict.ml_model <- function(object, newdata, proba = FALSE, ...) {
  if (!inherits(object, "ml_model")) {
    model_error("Expected an ml_model object")
  }
  if (proba) {
    return(.predict_proba_impl(object, newdata))
  }
  .predict_impl(object, newdata)
}

#' Predict from a fitted model (ml_predict style)
#'
#' Alias for `predict(model, newdata = ...)`. Matches Python `ml.predict()`.
#'
#' @param model An `ml_model` or `ml_tuning_result`
#' @param new_data A data.frame with the same features used for training
#' @returns A vector of predicted class labels (classification) or numeric values
#'   (regression).
#' @export
#' @examples
#' s <- ml_split(iris, "Species", seed = 42)
#' model <- ml_fit(s$train, "Species", seed = 42)
#' preds <- ml_predict(model, s$valid)
#' head(preds)
ml_predict <- function(model, new_data) {
  # Auto-unwrap TuningResult
  if (inherits(model, "ml_tuning_result")) model <- model[["best_model"]]
  if (!inherits(model, "ml_model")) model_error("Expected an ml_model or ml_tuning_result")
  # Calibrated model: delegate to S3 method (engine lives inside $model, not at top level)
  if (inherits(model, "ml_calibrated_model")) {
    return(predict(model, newdata = .coerce_data(new_data)))
  }
  .predict_impl(model, new_data)
}

.predict_impl <- function(model, new_data) {
  new_data <- .coerce_data(new_data)

  # Auto-drop target column if present
  if (!is.null(model[["target"]]) && model[["target"]] %in% names(new_data)) {
    new_data <- new_data[, setdiff(names(new_data), model[["target"]]), drop = FALSE]
  }

  # Feature validation
  missing_feats <- setdiff(model[["features"]], names(new_data))
  if (length(missing_feats) > 0) {
    data_error(paste0("Missing features: ", paste(missing_feats, collapse = ", ")))
  }
  extra_feats <- setdiff(names(new_data), model[["features"]])
  if (length(extra_feats) > 0) {
    cli::cli_warn(paste0("Extra features ignored: ", paste(extra_feats, collapse = ", ")))
  }

  # Reorder columns to training order
  X <- new_data[, model[["features"]], drop = FALSE]

  # Stacked model: special dispatch
  if (isTRUE(model[["is_stacked"]])) {
    return(.predict_stacked(model, X))
  }

  # Apply stored preprocessing
  X_enc <- .transform(X, model[["encoders"]])

  # Predict
  raw_preds <- .predict_engine(model[["engine"]], X_enc, model[["task"]], model[["algorithm"]])

  # Decode back to original labels
  .decode(raw_preds, model[["encoders"]])
}

#' Predict class probabilities
#'
#' @param model An `ml_model` object (classification only)
#' @param new_data A data.frame with the same features used for training
#' @returns A data.frame with one column per class. Values are probabilities
#'   summing to 1.0 per row. Column names are the original class labels.
#' @export
#' @examples
#' s <- ml_split(iris, "Species", seed = 42)
#' model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42)
#' probs <- ml_predict_proba(model, s$valid)
#' head(probs)
ml_predict_proba <- function(model, new_data) {
  # Calibrated model: delegate to S3 method (engine lives inside $model)
  if (inherits(model, "ml_calibrated_model")) {
    return(predict(model, newdata = .coerce_data(new_data), proba = TRUE))
  }
  .predict_proba_impl(model, new_data)
}

.predict_proba_impl <- function(model, new_data) {
  if (!inherits(model, "ml_model")) {
    model_error("Expected an ml_model object")
  }
  if (model[["task"]] != "classification") {
    model_error("predict_proba is for classification only")
  }
  if (!model[["algorithm"]] %in% .PROBA_ALGORITHMS) {
    model_error(paste0(
      "ml_predict_proba() not supported for algorithm='", model[["algorithm"]], "'.\n",
      "  Try algorithm='xgboost' or 'random_forest'."
    ))
  }

  new_data <- .coerce_data(new_data)

  # Auto-drop target
  if (!is.null(model[["target"]]) && model[["target"]] %in% names(new_data)) {
    new_data <- new_data[, setdiff(names(new_data), model[["target"]]), drop = FALSE]
  }

  # Feature validation
  missing_feats <- setdiff(model[["features"]], names(new_data))
  if (length(missing_feats) > 0) {
    data_error(paste0("Missing features: ", paste(missing_feats, collapse = ", ")))
  }

  X <- new_data[, model[["features"]], drop = FALSE]
  X_enc <- .transform(X, model[["encoders"]])

  proba_df <- .predict_proba_engine(model[["engine"]], X_enc, model[["task"]], model[["algorithm"]])

  # Normalize rows to sum=1 (XGBoost floating-point drift)
  row_sums <- rowSums(as.matrix(proba_df))
  row_sums[row_sums == 0] <- 1
  proba_norm <- as.data.frame(as.matrix(proba_df) / row_sums)

  # Rename columns to original class labels
  if (!is.null(model[["encoders"]]$label_levels)) {
    colnames(proba_norm) <- model[["encoders"]]$label_levels
  }

  proba_norm
}

# ── Stacked model prediction ───────────────────────────────────────────────────

.predict_stacked <- function(model, X) {
  st    <- model[["engine"]]  # stack_state list
  norm  <- st[["base_norm"]]

  # Apply base normalization to input data
  X_enc <- tryCatch(.transform(X, norm), error = function(e) X)

  # Generate base model predictions as meta-features (matching OOF structure)
  n              <- nrow(X_enc)
  n_base         <- length(st[["base_models"]])
  use_proba      <- isTRUE(st[["use_proba"]])
  n_classes      <- st[["n_classes"]] %||% 0L
  cols_per_model <- st[["cols_per_model"]] %||% 1L
  total_cols     <- n_base * cols_per_model

  oof_test <- matrix(NA_real_, nrow = n, ncol = total_cols)
  col_names <- character(total_cols)
  for (j in seq_len(n_base)) {
    start_col <- (j - 1L) * cols_per_model + 1L
    if (cols_per_model == 1L) {
      col_names[start_col] <- paste0("base_", j)
    } else {
      col_names[start_col:(start_col + cols_per_model - 1L)] <- paste0("base_", j, "_c", seq_len(cols_per_model))
    }
  }
  colnames(oof_test) <- col_names

  for (j in seq_along(st[["base_models"]])) {
    eng  <- st[["base_models"]][[j]]
    algo <- st[["base_algos"]][[j]]
    start_col <- (j - 1L) * cols_per_model + 1L
    if (!is.null(eng)) {
      tryCatch({
        if (use_proba && algo %in% .PROBA_ALGORITHMS) {
          pm <- as.matrix(.predict_proba_engine(eng, X_enc, model[["task"]], algo))
          if (n_classes == 2L) {
            oof_test[, start_col] <- pm[, ncol(pm)]
          } else {
            end_col <- start_col + cols_per_model - 1L
            oof_test[, start_col:end_col] <- pm
          }
        } else {
          preds <- .predict_engine(eng, X_enc, model[["task"]], algo)
          oof_test[, start_col] <- as.numeric(preds)
        }
      }, error = function(e) NULL)
    }
  }

  # Fill NA (failed engines) with column means
  for (j in seq_len(ncol(oof_test))) {
    na_idx <- is.na(oof_test[, j])
    if (any(na_idx)) oof_test[na_idx, j] <- mean(oof_test[, j], na.rm = TRUE)
  }

  # Predict with meta-learner
  meta_df <- as.data.frame(oof_test)

  # Align to stored feature names (in case valid_cols filtered some)
  meta_norm <- st[["meta_norm"]]
  if (!is.null(meta_norm) && !is.null(meta_norm$feature_names)) {
    avail_cols <- intersect(meta_norm$feature_names, names(meta_df))
    if (length(avail_cols) > 0L) meta_df <- meta_df[, avail_cols, drop = FALSE]
  }

  meta_enc  <- tryCatch(.transform(meta_df, meta_norm), error = function(e) meta_df)
  raw_preds <- .predict_engine(st[["meta_engine"]], meta_enc, model[["task"]], st[["meta_algo"]])
  .decode(raw_preds, norm)
}
