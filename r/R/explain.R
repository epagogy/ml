#' Explain model via feature importance
#'
#' Returns a data frame of feature importances, normalized to sum to 1.0,
#' sorted descending. Uses tree-based impurity importance for 'xgboost' and
#' 'random_forest', absolute coefficients for 'logistic', 'linear', and
#' 'elastic_net'. Not supported for 'svm' or 'knn'.
#'
#' @param model An `ml_model` or `ml_tuning_result`
#' @returns An object of class `ml_explanation` (a data.frame with columns
#'   `feature` and `importance`; custom print shows a bar chart)
#' @export
#' @examples
#' s <- ml_split(iris, "Species", seed = 42)
#' model <- ml_fit(s$train, "Species", algorithm = "random_forest", seed = 42)
#' ml_explain(model)
ml_explain <- function(model) {
  .explain_impl(model = model)
}

.explain_impl <- function(model) {
  # Auto-unwrap TuningResult
  if (inherits(model, "ml_tuning_result")) model <- model[["best_model"]]
  if (!inherits(model, "ml_model")) model_error("Expected an ml_model or ml_tuning_result")

  algo <- model[["algorithm"]]

  imp <- tryCatch(
    .extract_importance(model[["engine"]], algo, model[["features"]]),
    error = function(e) {
      model_error(paste0(
        "ml_explain() requires a model with feature importances.\n",
        "  algorithm='", algo, "' does not support this. ",
        "Try algorithm='xgboost' or 'random_forest'."
      ))
    }
  )

  if (is.null(imp)) {
    model_error(paste0(
      "ml_explain() requires a model with feature importances.\n",
      "  algorithm='", algo, "' does not support this. ",
      "Try algorithm='xgboost' or 'random_forest'."
    ))
  }

  # Normalize to sum=1
  total <- sum(abs(imp), na.rm = TRUE)
  if (total == 0) total <- 1
  imp_norm <- abs(imp) / total

  df <- data.frame(
    feature    = names(imp_norm),
    importance = as.numeric(imp_norm),
    stringsAsFactors = FALSE
  )
  df <- df[order(df$importance, decreasing = TRUE), , drop = FALSE]
  rownames(df) <- NULL

  new_ml_explanation(df, algo)
}

.extract_importance <- function(engine, algo, features) {
  # Rust engines: extract from JSON via Rust bindings
  if (.is_rust_engine(engine)) {
    if (algo %in% c("random_forest", "decision_tree", "extra_trees",
                    "gradient_boosting", "histgradient", "adaboost")) {
      imp_fn <- switch(algo,
        random_forest     = .importances_rust_forest,
        extra_trees       = .importances_rust_extra_trees,
        gradient_boosting = ,  # fall through
        histgradient      = .importances_rust_gbt,
        adaboost          = .importances_rust_ada,
        .importances_rust_tree
      )
      imp <- imp_fn(engine)
      if (length(imp) == length(features)) names(imp) <- features
      return(imp)
    }
    if (algo == "linear") {
      coefs <- .coef_rust_linear(engine)
      # coefs = [intercept, w1, w2, ...]
      imp <- abs(coefs[-1])
      if (length(imp) == length(features)) names(imp) <- features
      return(imp)
    }
    if (algo == "logistic") {
      imp <- .coef_rust_logistic(engine)
      if (length(imp) == length(features)) names(imp) <- features
      return(imp)
    }
    if (algo == "knn") return(NULL)
    return(NULL)
  }

  if (algo == "xgboost") {
    sc <- xgboost::xgb.importance(model = engine)
    if (is.null(sc) || nrow(sc) == 0L) return(NULL)
    imp <- stats::setNames(sc$Gain, sc$Feature)
    # Fill 0 for any features not in importance output
    all_imp <- stats::setNames(rep(0, length(features)), features)
    all_imp[names(imp)] <- imp
    return(all_imp)
  }

  if (algo == "random_forest") {
    # ranger stores variable.importance
    vi <- engine$variable.importance
    if (is.null(vi) || length(vi) == 0L) return(NULL)
    return(vi)
  }

  if (algo == "extra_trees") {
    # CRAN fallback via ranger with splitrule="extratrees"
    vi <- engine$variable.importance
    if (is.null(vi) || length(vi) == 0L) return(NULL)
    return(vi)
  }

  if (algo == "logistic") {
    # Native .lr_fit: engine$models is a list of {coef=[bias, w1..wp]}
    # coef[1] = bias (skip), coef[-1] = feature weights
    if (!is.list(engine) || is.null(engine[["models"]])) return(NULL)
    coef_mat <- do.call(rbind, lapply(engine[["models"]], function(m) m[["coef"]][-1]))
    if (is.null(coef_mat) || length(coef_mat) == 0L) return(NULL)
    # Average absolute coefficients across OvR classifiers; name by features
    coefs <- colMeans(abs(coef_mat))
    if (!is.null(features) && length(features) == length(coefs)) {
      names(coefs) <- features
    }
    return(coefs)
  }

  if (algo == "linear") {
    # Native .lm_fit: engine$coef is a numeric vector (length = n_features)
    if (!is.list(engine) || is.null(engine[["coef"]])) return(NULL)
    coefs <- abs(as.numeric(engine[["coef"]]))
    if (!is.null(features) && length(features) == length(coefs)) {
      names(coefs) <- features
    }
    return(coefs)
  }

  if (algo == "elastic_net") {
    # glmnet: use coefficients at middle lambda
    coef_mat <- as.matrix(stats::coef(engine))
    mid       <- ceiling(ncol(coef_mat) / 2)
    coefs     <- coef_mat[, mid]
    coefs     <- coefs[names(coefs) != "(Intercept)"]
    return(abs(coefs))
  }

  if (algo == "naive_bayes") {
    # Not supported for Naive Bayes — no well-defined feature importance
    return(NULL)
  }

  if (algo == "decision_tree") {
    vi <- engine$variable.importance
    if (is.null(vi) || length(vi) == 0L) return(NULL)
    return(vi)
  }

  if (algo %in% c("svm", "knn")) {
    return(NULL)
  }

  NULL
}
