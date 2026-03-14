#' Calibrate predicted probabilities
#'
#' Applies Platt scaling (logistic regression on raw probabilities) to produce
#' better-calibrated class probability estimates. Use validation data for
#' calibration -- never training data.
#'
#' Binary classification only.
#'
#' @param model An `ml_model` from `ml_fit()`
#' @param data A data.frame of calibration data (use validation set)
#' @returns An `ml_calibrated_model` that behaves like an `ml_model` but
#'   returns calibrated probabilities
#' @export
#' @examples
#' s <- ml_split(ml_dataset("cancer"), "target", seed = 42)
#' model <- ml_fit(s$train, "target", algorithm = "xgboost", seed = 42)
#' cal <- ml_calibrate(model, data = s$valid)
#' ml_evaluate(cal, s$valid)
ml_calibrate <- function(model, data = NULL) {
  if (inherits(model, "ml_tuning_result")) model <- model[["best_model"]]
  if (!inherits(model, "ml_model")) {
    model_error("Expected an ml_model object")
  }
  if (model$task != "classification") {
    config_error("ml_calibrate() is for classification models only")
  }
  if (is.null(data)) {
    config_error("data= required. Pass validation data -- never training data.")
  }

  classes <- sort(unique(data[[model$target]]))
  if (length(classes) != 2L) {
    config_error("ml_calibrate() supports binary classification only")
  }

  # Get raw probabilities on calibration data
  proba <- predict(model, newdata = data, proba = TRUE)
  pos_class <- as.character(classes[2L])
  raw_prob  <- proba[[pos_class]]
  y_true    <- as.integer(data[[model$target]] == classes[2L])

  # Platt scaling: fit logistic regression on raw probabilities
  cal_df    <- data.frame(y = y_true, p = raw_prob)
  cal_glm   <- stats::glm(y ~ p, data = cal_df, family = stats::binomial())

  structure(list(
    model      = model,
    cal_glm    = cal_glm,
    classes    = classes,
    pos_class  = pos_class,
    target     = model$target,
    task       = model$task,
    algorithm  = model$algorithm,
    features   = model$features,
    seed       = model$seed,
    calibrated = TRUE
  ), class = c("ml_calibrated_model", "ml_model"))
}

#' @export
predict.ml_calibrated_model <- function(object, newdata, proba = FALSE, ...) {
  # Get raw probabilities from base model
  raw_proba <- predict(object$model, newdata = newdata, proba = TRUE)
  pos_class <- object$pos_class
  raw_prob  <- raw_proba[[pos_class]]

  # Apply Platt calibration
  cal_prob <- stats::predict(object$cal_glm,
                             newdata = data.frame(p = raw_prob),
                             type = "response")

  if (proba) {
    df <- data.frame(v0 = 1 - cal_prob, v1 = cal_prob)
    names(df) <- as.character(object$classes)
    return(df)
  }

  # Return class labels
  predicted_class <- ifelse(cal_prob >= 0.5, object$pos_class,
                            as.character(object$classes[1L]))
  # Return as same type as original labels
  y_ref <- newdata[[object$target]]
  if (!is.null(y_ref) && is.factor(y_ref)) {
    factor(predicted_class, levels = levels(y_ref))
  } else {
    predicted_class
  }
}

# Internal: metrics for calibrated models (bypasses engine-based predict)
.evaluate_calibrated_impl <- function(model, data) {
  data   <- .coerce_data(data)
  preds  <- predict(model, newdata = data)
  y_true <- data[[model$target]]
  task   <- model$task

  if (task == "classification") {
    proba   <- predict(model, newdata = data, proba = TRUE)
    pos     <- model$pos_class
    y_bin   <- as.integer(y_true == model$classes[2L])
    p_bin   <- as.integer(as.character(preds) == pos)
    acc     <- mean(as.character(preds) == as.character(y_true))
    tp <- sum(p_bin == 1L & y_bin == 1L)
    fp <- sum(p_bin == 1L & y_bin == 0L)
    fn <- sum(p_bin == 0L & y_bin == 1L)
    prec <- if ((tp + fp) == 0L) 0 else tp / (tp + fp)
    rec  <- if ((tp + fn) == 0L) 0 else tp / (tp + fn)
    f1   <- if ((prec + rec) == 0) 0 else 2 * prec * rec / (prec + rec)
    prob_pos <- proba[[pos]]
    n_pos <- sum(y_bin == 1L); n_neg <- sum(y_bin == 0L)
    auc <- if (n_pos == 0L || n_neg == 0L) NA_real_ else {
      r <- rank(prob_pos)
      (sum(r[y_bin == 1L]) - n_pos * (n_pos + 1L) / 2L) / (n_pos * n_neg)
    }
    raw <- c(accuracy = round(acc, 4L), f1 = round(f1, 4L), roc_auc = round(auc, 4L))
  } else {
    res  <- as.numeric(y_true) - as.numeric(preds)
    rmse <- sqrt(mean(res^2))
    mae  <- mean(abs(res))
    r2   <- 1 - sum(res^2) / sum((as.numeric(y_true) - mean(as.numeric(y_true)))^2)
    raw  <- c(rmse = round(rmse, 4L), mae = round(mae, 4L), r2 = round(r2, 4L))
  }

  structure(raw, class = "ml_metrics", task = task, time_s = 0)
}

#' @export
print.ml_calibrated_model <- function(x, ...) {
  cat(sprintf("-- Calibrated ml_model [%s | %s | Platt scaling]\n",
              x$task, x$algorithm))
  invisible(x)
}
