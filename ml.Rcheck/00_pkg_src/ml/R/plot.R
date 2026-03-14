#' Visual diagnostics for a fitted model
#'
#' Produces diagnostic plots using base R graphics. No extra packages required.
#'
#' Available kinds:
#' - `"importance"` -- feature importance bar chart
#' - `"roc"` -- ROC curve (classification)
#' - `"confusion"` -- confusion matrix heatmap (classification)
#' - `"residual"` -- residuals vs fitted (regression)
#' - `"calibration"` -- predicted vs actual probabilities (classification)
#'
#' @param model An `ml_model` from `ml_fit()`
#' @param data A data.frame for computing predictions (required for all
#'   except `"importance"`)
#' @param kind Plot type. One of `"importance"`, `"roc"`, `"confusion"`,
#'   `"residual"`, `"calibration"`. Default: `"importance"`.
#' @param ... Passed to the underlying base R plot call
#' @returns Invisibly returns NULL (called for its side effect)
#' @export
#' @examples
#' \donttest{
#' s <- ml_split(iris, "Species", seed = 42)
#' model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42)
#' ml_plot(model, kind = "importance")
#' ml_plot(model, data = s$valid, kind = "confusion")
#' }
ml_plot <- function(model, data = NULL, kind = "importance", ...) {
  if (inherits(model, "ml_tuning_result")) model <- model[["best_model"]]
  if (!inherits(model, "ml_model") && !inherits(model, "ml_calibrated_model")) {
    model_error("Expected an ml_model object")
  }

  switch(kind,
    importance  = .plot_importance(model, ...),
    roc         = .plot_roc(model, data, ...),
    confusion   = .plot_confusion(model, data, ...),
    residual    = .plot_residual(model, data, ...),
    calibration = .plot_calibration(model, data, ...),
    config_error(paste0(
      "kind='", kind, "' not available.\n",
      "  Choose from: importance, roc, confusion, residual, calibration"
    ))
  )
  invisible(NULL)
}

# ── plot helpers ───────────────────────────────────────────────────────────────

.plot_importance <- function(model, n = 20L, ...) {
  imp <- tryCatch(ml_explain(model), error = function(e) {
    model_error("ml_plot(kind='importance') requires a model that supports ml_explain()")
  })
  top_n <- min(n, nrow(imp))
  imp   <- imp[seq_len(top_n), , drop = FALSE]
  imp   <- imp[order(imp$importance), ]

  old_mar <- graphics::par(mar = c(4, 10, 3, 2))
  on.exit(graphics::par(old_mar))

  graphics::barplot(
    imp$importance,
    names.arg = imp$feature,
    horiz     = TRUE,
    las       = 1,
    col       = "#4f46e5",
    border    = NA,
    xlab      = "Importance",
    main      = paste0("Feature importance (", model$algorithm, ")"),
    ...
  )
}

.plot_roc <- function(model, data, ...) {
  if (is.null(data)) config_error("data= required for kind='roc'")
  if (model$task != "classification") {
    config_error("kind='roc' is for classification models only")
  }

  classes <- sort(unique(data[[model$target]]))
  if (length(classes) != 2L) {
    config_error("kind='roc' supports binary classification only")
  }

  pos_class <- as.character(classes[2L])
  proba     <- predict(model, newdata = data, proba = TRUE)[[pos_class]]
  y_true    <- as.integer(data[[model$target]] == classes[2L])

  # Compute ROC
  thresholds <- sort(unique(proba), decreasing = TRUE)
  tpr <- vapply(thresholds, function(t) mean(proba[y_true == 1L] >= t), numeric(1L))
  fpr <- vapply(thresholds, function(t) mean(proba[y_true == 0L] >= t), numeric(1L))
  auc <- sum(diff(c(0, fpr, 1)) * c(0, tpr, 1)[-length(c(0, tpr, 1))])

  graphics::plot(
    c(0, fpr, 1), c(0, tpr, 1),
    type = "l", col = "#4f46e5", lwd = 2,
    xlab = "False Positive Rate", ylab = "True Positive Rate",
    main = paste0("ROC curve (AUC = ", round(auc, 3), ")"),
    xlim = c(0, 1), ylim = c(0, 1), ...
  )
  graphics::abline(0, 1, lty = 2, col = "#9ca3af")
}

.plot_confusion <- function(model, data, ...) {
  if (is.null(data)) config_error("data= required for kind='confusion'")
  if (model$task != "classification") {
    config_error("kind='confusion' is for classification models only")
  }

  preds  <- predict(model, newdata = data)
  actual <- data[[model$target]]
  cm     <- table(Actual = actual, Predicted = preds)
  cm_pct <- cm / rowSums(cm)

  n <- nrow(cm)
  graphics::image(
    seq_len(n), seq_len(n), t(cm_pct[n:1, ]),
    col  = grDevices::colorRampPalette(c("#ffffff", "#4f46e5"))(20),
    xaxt = "n", yaxt = "n",
    xlab = "Predicted", ylab = "Actual",
    main = "Confusion matrix",
    ...
  )
  graphics::axis(1, at = seq_len(n), labels = colnames(cm))
  graphics::axis(2, at = seq_len(n), labels = rev(rownames(cm)), las = 1)

  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      graphics::text(j, n + 1L - i,
                     labels = cm[i, j],
                     col    = if (cm_pct[i, j] > 0.5) "white" else "#222222")
    }
  }
}

.plot_residual <- function(model, data, ...) {
  if (is.null(data)) config_error("data= required for kind='residual'")
  if (model$task != "regression") {
    config_error("kind='residual' is for regression models only")
  }

  fitted    <- predict(model, newdata = data)
  actual    <- data[[model$target]]
  residuals <- actual - fitted

  graphics::plot(
    fitted, residuals,
    col  = "#4f46e5", pch = 16, cex = 0.7,
    xlab = "Fitted values", ylab = "Residuals",
    main = paste0("Residuals vs fitted (", model$algorithm, ")"),
    ...
  )
  graphics::abline(h = 0, lty = 2, col = "#9ca3af")
  graphics::lines(stats::lowess(fitted, residuals), col = "#ef4444", lwd = 2)
}

.plot_calibration <- function(model, data, bins = 10L, ...) {
  if (is.null(data)) config_error("data= required for kind='calibration'")
  if (model$task != "classification") {
    config_error("kind='calibration' is for classification models only")
  }

  classes   <- sort(unique(data[[model$target]]))
  if (length(classes) != 2L) {
    config_error("kind='calibration' supports binary classification only")
  }

  pos_class <- as.character(classes[2L])
  proba     <- predict(model, newdata = data, proba = TRUE)[[pos_class]]
  y_true    <- as.integer(data[[model$target]] == classes[2L])

  breaks  <- seq(0, 1, length.out = bins + 1L)
  bin_idx <- cut(proba, breaks, include.lowest = TRUE)
  mean_pred <- tapply(proba,  bin_idx, mean)
  mean_true <- tapply(y_true, bin_idx, mean)

  graphics::plot(
    mean_pred, mean_true,
    type = "b", col = "#4f46e5", pch = 16, lwd = 2,
    xlim = c(0, 1), ylim = c(0, 1),
    xlab = "Mean predicted probability",
    ylab = "Fraction of positives",
    main = "Calibration curve",
    ...
  )
  graphics::abline(0, 1, lty = 2, col = "#9ca3af")
  graphics::legend("topleft", legend = c("Model", "Perfect"),
                   col = c("#4f46e5", "#9ca3af"), lty = c(1, 2), bty = "n")
}
