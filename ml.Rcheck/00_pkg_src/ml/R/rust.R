# R wrappers for Rust ml backends.
# Mirrors py/ml/_rust.py -- same algorithms, same JSON serialization.
# All functions are internal (dot-prefix). Called by engines.R dispatch.

# Suppress R CMD check NOTEs for extendr-generated .Call symbols.
# These are registered by R_init_ml_rust_extendr() at load time when
# the Rust backend is compiled; without Rust they are never called.
globalVariables(c(
  "wrap__ml_rust_available",
  "wrap__ml_rust_linear_fit", "wrap__ml_rust_linear_predict",
  "wrap__ml_rust_linear_coef",
  "wrap__ml_rust_logistic_fit", "wrap__ml_rust_logistic_predict",
  "wrap__ml_rust_logistic_predict_proba", "wrap__ml_rust_logistic_coef",
  "wrap__ml_rust_tree_fit_clf", "wrap__ml_rust_tree_fit_reg",
  "wrap__ml_rust_tree_predict_clf", "wrap__ml_rust_tree_predict_reg",
  "wrap__ml_rust_tree_predict_proba", "wrap__ml_rust_tree_importances",
  "wrap__ml_rust_forest_fit_clf", "wrap__ml_rust_forest_fit_reg",
  "wrap__ml_rust_forest_predict_clf", "wrap__ml_rust_forest_predict_reg",
  "wrap__ml_rust_forest_predict_proba", "wrap__ml_rust_forest_importances",
  "wrap__ml_rust_extra_trees_fit_clf", "wrap__ml_rust_extra_trees_fit_reg",
  "wrap__ml_rust_knn_fit_clf", "wrap__ml_rust_knn_fit_reg",
  "wrap__ml_rust_knn_predict_clf", "wrap__ml_rust_knn_predict_reg",
  "wrap__ml_rust_knn_predict_proba",
  "wrap__ml_rust_gbt_fit_clf", "wrap__ml_rust_gbt_fit_reg",
  "wrap__ml_rust_gbt_predict_clf", "wrap__ml_rust_gbt_predict_reg",
  "wrap__ml_rust_gbt_predict_proba", "wrap__ml_rust_gbt_importances",
  "wrap__ml_rust_nb_fit", "wrap__ml_rust_nb_predict",
  "wrap__ml_rust_nb_predict_proba",
  "wrap__ml_rust_en_fit", "wrap__ml_rust_en_predict",
  "wrap__ml_rust_ada_fit", "wrap__ml_rust_ada_predict",
  "wrap__ml_rust_ada_predict_proba", "wrap__ml_rust_ada_importances",
  "wrap__ml_rust_svm_fit_clf", "wrap__ml_rust_svm_fit_reg",
  "wrap__ml_rust_svm_predict_clf", "wrap__ml_rust_svm_predict_reg",
  "wrap__ml_rust_svm_predict_proba"
))

# ── Cache ────────────────────────────────────────────────────────────────────

.ml_rust_cache <- new.env(parent = emptyenv())
.ml_rust_cache$available <- NULL

# ── Availability ─────────────────────────────────────────────────────────────

#' Check if Rust backend is available (cached).
#' @returns A logical scalar: `TRUE` if the Rust shared library is loaded.
#' @keywords internal
.rust_available <- function() {
  if (!is.null(.ml_rust_cache$available)) return(.ml_rust_cache$available)
  result <- tryCatch(
    { .Call(wrap__ml_rust_available); TRUE },
    error = function(e) FALSE
  )
  .ml_rust_cache$available <- result
  result
}

# ── Proba reshape helper ────────────────────────────────────────────────────

.rust_proba_to_df <- function(result) {
  # Rust returns list(data=col_major_flat, nrow=n, ncol=k)
  matrix(result$data, nrow = result$nrow, ncol = result$ncol)
}

# ── Linear (Ridge) ──────────────────────────────────────────────────────────

.fit_rust_linear <- function(X, y, alpha = 1.0, sample_weight = NULL) {
  Xm <- as.matrix(X)
  json <- .Call(wrap__ml_rust_linear_fit,
                as.numeric(Xm), nrow(Xm), ncol(Xm),
                as.numeric(y), alpha, sample_weight)
  list(json = json, type = "rust_linear")
}

.predict_rust_linear <- function(engine, X) {
  Xm <- as.matrix(X)
  .Call(wrap__ml_rust_linear_predict,
        engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
}

.coef_rust_linear <- function(engine) {
  # Returns [intercept, coef1, coef2, ...]
  .Call(wrap__ml_rust_linear_coef, engine$json)
}

# ── Logistic (OvR + L-BFGS) ────────────────────────────────────────────────

.fit_rust_logistic <- function(X, y, C = 1.0, max_iter = 1000L,
                               sample_weight = NULL, multi_class = "ovr") {
  Xm <- as.matrix(X)
  y_int <- as.integer(y)
  json <- .Call(wrap__ml_rust_logistic_fit,
                as.numeric(Xm), nrow(Xm), ncol(Xm),
                y_int, C, as.integer(max_iter), sample_weight,
                as.character(multi_class))
  list(json = json, type = "rust_logistic")
}

.predict_rust_logistic <- function(engine, X) {
  Xm <- as.matrix(X)
  .Call(wrap__ml_rust_logistic_predict,
        engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
}

.proba_rust_logistic <- function(engine, X) {
  Xm <- as.matrix(X)
  result <- .Call(wrap__ml_rust_logistic_predict_proba,
                  engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  pm <- .rust_proba_to_df(result)
  as.data.frame(pm)
}

.coef_rust_logistic <- function(engine) {
  # Returns avg absolute OvR coefficients [feature_1, ..., feature_p]
  .Call(wrap__ml_rust_logistic_coef, engine$json)
}

# ── Decision Tree (CART) ────────────────────────────────────────────────────

.fit_rust_tree <- function(X, y, task, seed, max_depth = 500L,
                           min_samples_split = 2L, min_samples_leaf = 1L,
                           sample_weight = NULL, criterion = "gini",
                           monotone_cst = NULL,
                           min_impurity_decrease = 0.0, ccp_alpha = 0.0) {
  Xm <- as.matrix(X)
  if (task == "classification") {
    y_int <- as.integer(y)
    json <- .Call(wrap__ml_rust_tree_fit_clf,
                  as.numeric(Xm), nrow(Xm), ncol(Xm),
                  y_int,
                  as.integer(max_depth), as.integer(min_samples_split),
                  as.integer(min_samples_leaf), as.integer(seed),
                  sample_weight, as.character(criterion),
                  as.numeric(min_impurity_decrease), as.numeric(ccp_alpha))
  } else {
    mc <- if (is.null(monotone_cst)) NULL else as.integer(monotone_cst)
    json <- .Call(wrap__ml_rust_tree_fit_reg,
                  as.numeric(Xm), nrow(Xm), ncol(Xm),
                  as.numeric(y),
                  as.integer(max_depth), as.integer(min_samples_split),
                  as.integer(min_samples_leaf), as.integer(seed),
                  sample_weight, as.character(criterion), mc,
                  as.numeric(min_impurity_decrease), as.numeric(ccp_alpha))
  }
  list(json = json, type = paste0("rust_tree_", task), task = task)
}

.predict_rust_tree <- function(engine, X, task) {
  Xm <- as.matrix(X)
  if (task == "classification") {
    .Call(wrap__ml_rust_tree_predict_clf,
          engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  } else {
    .Call(wrap__ml_rust_tree_predict_reg,
          engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  }
}

.proba_rust_tree <- function(engine, X) {
  Xm <- as.matrix(X)
  result <- .Call(wrap__ml_rust_tree_predict_proba,
                  engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  pm <- .rust_proba_to_df(result)
  as.data.frame(pm)
}

.importances_rust_tree <- function(engine) {
  .Call(wrap__ml_rust_tree_importances, engine$json)
}

# ── Random Forest ───────────────────────────────────────────────────────────

.fit_rust_forest <- function(X, y, task, seed, n_trees = 100L,
                             max_depth = 500L, min_samples_split = 2L,
                             min_samples_leaf = 1L,
                             sample_weight = NULL, criterion = "gini",
                             monotone_cst = NULL,
                             min_impurity_decrease = 0.0) {
  Xm <- as.matrix(X)
  if (task == "classification") {
    y_int <- as.integer(y)
    json <- .Call(wrap__ml_rust_forest_fit_clf,
                  as.numeric(Xm), nrow(Xm), ncol(Xm),
                  y_int,
                  as.integer(n_trees), as.integer(max_depth),
                  as.integer(min_samples_split), as.integer(min_samples_leaf),
                  as.integer(seed), sample_weight, as.character(criterion),
                  as.numeric(min_impurity_decrease))
  } else {
    mc <- if (is.null(monotone_cst)) NULL else as.integer(monotone_cst)
    json <- .Call(wrap__ml_rust_forest_fit_reg,
                  as.numeric(Xm), nrow(Xm), ncol(Xm),
                  as.numeric(y),
                  as.integer(n_trees), as.integer(max_depth),
                  as.integer(min_samples_split), as.integer(min_samples_leaf),
                  as.integer(seed), sample_weight, as.character(criterion), mc,
                  as.numeric(min_impurity_decrease))
  }
  list(json = json, type = paste0("rust_forest_", task), task = task)
}

# ── Extra Trees ──────────────────────────────────────────────────────────────

.fit_rust_extra_trees <- function(X, y, task, seed, n_trees = 100L,
                                  max_depth = 500L, min_samples_split = 2L,
                                  min_samples_leaf = 1L,
                                  sample_weight = NULL, criterion = "gini",
                                  monotone_cst = NULL,
                                  min_impurity_decrease = 0.0) {
  Xm <- as.matrix(X)
  if (task == "classification") {
    y_int <- as.integer(y)
    json <- .Call(wrap__ml_rust_extra_trees_fit_clf,
                  as.numeric(Xm), nrow(Xm), ncol(Xm),
                  y_int,
                  as.integer(n_trees), as.integer(max_depth),
                  as.integer(min_samples_split), as.integer(min_samples_leaf),
                  as.integer(seed), sample_weight, as.character(criterion),
                  as.numeric(min_impurity_decrease))
  } else {
    mc <- if (is.null(monotone_cst)) NULL else as.integer(monotone_cst)
    json <- .Call(wrap__ml_rust_extra_trees_fit_reg,
                  as.numeric(Xm), nrow(Xm), ncol(Xm),
                  as.numeric(y),
                  as.integer(n_trees), as.integer(max_depth),
                  as.integer(min_samples_split), as.integer(min_samples_leaf),
                  as.integer(seed), sample_weight, as.character(criterion), mc,
                  as.numeric(min_impurity_decrease))
  }
  # Extra Trees serializes as the same RandomForestModel format -- reuse forest predict/proba
  list(json = json, type = paste0("rust_extra_trees_", task), task = task)
}

.predict_rust_forest <- function(engine, X, task) {
  Xm <- as.matrix(X)
  if (task == "classification") {
    .Call(wrap__ml_rust_forest_predict_clf,
          engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  } else {
    .Call(wrap__ml_rust_forest_predict_reg,
          engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  }
}

.proba_rust_forest <- function(engine, X) {
  Xm <- as.matrix(X)
  result <- .Call(wrap__ml_rust_forest_predict_proba,
                  engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  pm <- .rust_proba_to_df(result)
  as.data.frame(pm)
}

.importances_rust_forest <- function(engine) {
  .Call(wrap__ml_rust_forest_importances, engine$json)
}

.predict_rust_extra_trees <- function(engine, X, task) {
  .predict_rust_forest(engine, X, task)
}

.proba_rust_extra_trees <- function(engine, X) {
  .proba_rust_forest(engine, X)
}

.importances_rust_extra_trees <- function(engine) {
  .importances_rust_forest(engine)
}

# ── KNN ─────────────────────────────────────────────────────────────────────

.fit_rust_knn <- function(X, y, task, k = 5L, weights = "uniform") {
  Xm <- as.matrix(X)
  if (task == "classification") {
    y_int <- as.integer(y)
    json <- .Call(wrap__ml_rust_knn_fit_clf,
                  as.numeric(Xm), nrow(Xm), ncol(Xm),
                  y_int, as.integer(k), as.character(weights))
  } else {
    json <- .Call(wrap__ml_rust_knn_fit_reg,
                  as.numeric(Xm), nrow(Xm), ncol(Xm),
                  as.numeric(y), as.integer(k), as.character(weights))
  }
  list(json = json, type = paste0("rust_knn_", task), task = task)
}

.predict_rust_knn <- function(engine, X, task) {
  Xm <- as.matrix(X)
  if (task == "classification") {
    .Call(wrap__ml_rust_knn_predict_clf,
          engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  } else {
    .Call(wrap__ml_rust_knn_predict_reg,
          engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  }
}

.proba_rust_knn <- function(engine, X) {
  Xm <- as.matrix(X)
  result <- .Call(wrap__ml_rust_knn_predict_proba,
                  engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  pm <- .rust_proba_to_df(result)
  as.data.frame(pm)
}

# ── Gradient-Boosted Trees (GBT) ────────────────────────────────────────────

.fit_rust_gbt <- function(X, y, task, seed, n_estimators = 100L,
                          learning_rate = 0.1, max_depth = 3L,
                          min_samples_split = 2L, min_samples_leaf = 1L,
                          subsample = 1.0, sample_weight = NULL,
                          reg_lambda = 0.0, gamma = 0.0,
                          colsample_bytree = 1.0, min_child_weight = 1.0,
                          n_iter_no_change = NULL,
                          validation_fraction = 0.1) {
  Xm <- as.matrix(X)
  # Convert NULL → R NULL (Nullable<i32>::Null on Rust side)
  nic <- if (is.null(n_iter_no_change)) NULL else as.integer(n_iter_no_change)
  if (task == "classification") {
    y_int <- as.integer(y)
    json <- .Call(wrap__ml_rust_gbt_fit_clf,
                  as.numeric(Xm), nrow(Xm), ncol(Xm),
                  y_int,
                  as.integer(n_estimators), as.numeric(learning_rate),
                  as.integer(max_depth), as.integer(min_samples_split),
                  as.integer(min_samples_leaf), as.numeric(subsample),
                  as.integer(seed), sample_weight,
                  as.numeric(reg_lambda), as.numeric(gamma),
                  as.numeric(colsample_bytree), as.numeric(min_child_weight),
                  nic, as.numeric(validation_fraction))
  } else {
    json <- .Call(wrap__ml_rust_gbt_fit_reg,
                  as.numeric(Xm), nrow(Xm), ncol(Xm),
                  as.numeric(y),
                  as.integer(n_estimators), as.numeric(learning_rate),
                  as.integer(max_depth), as.integer(min_samples_split),
                  as.integer(min_samples_leaf), as.numeric(subsample),
                  as.integer(seed), sample_weight,
                  as.numeric(reg_lambda), as.numeric(gamma),
                  as.numeric(colsample_bytree), as.numeric(min_child_weight),
                  nic, as.numeric(validation_fraction))
  }
  list(json = json, type = paste0("rust_gbt_", task), task = task)
}

.predict_rust_gbt <- function(engine, X, task) {
  Xm <- as.matrix(X)
  if (task == "classification") {
    .Call(wrap__ml_rust_gbt_predict_clf,
          engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  } else {
    .Call(wrap__ml_rust_gbt_predict_reg,
          engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  }
}

.proba_rust_gbt <- function(engine, X) {
  Xm <- as.matrix(X)
  result <- .Call(wrap__ml_rust_gbt_predict_proba,
                  engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  pm <- .rust_proba_to_df(result)
  as.data.frame(pm)
}

.importances_rust_gbt <- function(engine) {
  .Call(wrap__ml_rust_gbt_importances, engine$json)
}

# ── Naive Bayes ──────────────────────────────────────────────────────────────

.fit_rust_nb <- function(X, y, task, seed, var_smoothing = 1e-9,
                         sample_weight = NULL) {
  Xm <- as.matrix(X)
  y_int <- as.integer(y)
  json <- .Call(wrap__ml_rust_nb_fit,
                as.numeric(Xm), nrow(Xm), ncol(Xm),
                y_int, as.numeric(var_smoothing), sample_weight)
  list(json = json, type = "rust_nb_classification", task = "classification")
}

.predict_rust_nb <- function(engine, X, task) {
  Xm <- as.matrix(X)
  .Call(wrap__ml_rust_nb_predict,
        engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
}

.proba_rust_nb <- function(engine, X) {
  Xm <- as.matrix(X)
  result <- .Call(wrap__ml_rust_nb_predict_proba,
                  engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  as.data.frame(.rust_proba_to_df(result))
}

# ── Elastic Net ───────────────────────────────────────────────────────────────

.fit_rust_en <- function(X, y, task, seed, alpha = 1.0, l1_ratio = 0.5,
                         max_iter = 1000L, tol = 1e-4, sample_weight = NULL) {
  Xm <- as.matrix(X)
  json <- .Call(wrap__ml_rust_en_fit,
                as.numeric(Xm), nrow(Xm), ncol(Xm),
                as.numeric(y),
                as.numeric(alpha), as.numeric(l1_ratio),
                as.integer(max_iter), as.numeric(tol),
                sample_weight)
  list(json = json, type = "rust_en_regression", task = "regression")
}

.predict_rust_en <- function(engine, X, task) {
  Xm <- as.matrix(X)
  .Call(wrap__ml_rust_en_predict,
        engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
}

# ── AdaBoost ──────────────────────────────────────────────────────────────────

.fit_rust_ada <- function(X, y, task, seed, n_estimators = 50L,
                          learning_rate = 1.0) {
  Xm <- as.matrix(X)
  y_int <- as.integer(y)
  json <- .Call(wrap__ml_rust_ada_fit,
                as.numeric(Xm), nrow(Xm), ncol(Xm),
                y_int, as.integer(n_estimators), as.numeric(learning_rate),
                as.integer(seed))
  list(json = json, type = "rust_ada_classification", task = "classification")
}

.predict_rust_ada <- function(engine, X, task) {
  Xm <- as.matrix(X)
  .Call(wrap__ml_rust_ada_predict,
        engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
}

.proba_rust_ada <- function(engine, X) {
  Xm <- as.matrix(X)
  result <- .Call(wrap__ml_rust_ada_predict_proba,
                  engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  as.data.frame(.rust_proba_to_df(result))
}

.importances_rust_ada <- function(engine) {
  .Call(wrap__ml_rust_ada_importances, engine$json)
}

# ── SVM ───────────────────────────────────────────────────────────────────────

.fit_rust_svm <- function(X, y, task, seed, C = 1.0, epsilon = 0.1,
                          tol = 1e-3, max_iter = 1000L,
                          sample_weight = NULL,
                          kernel = "linear", gamma = 0.0,
                          degree = 3L, coef0 = 0.0) {
  Xm <- as.matrix(X)
  if (task == "classification") {
    y_int <- as.integer(y)
    json <- .Call(wrap__ml_rust_svm_fit_clf,
                  as.numeric(Xm), nrow(Xm), ncol(Xm),
                  y_int, as.numeric(C), as.numeric(tol), as.integer(max_iter),
                  sample_weight,
                  as.character(kernel), as.numeric(gamma),
                  as.integer(degree), as.numeric(coef0))
  } else {
    json <- .Call(wrap__ml_rust_svm_fit_reg,
                  as.numeric(Xm), nrow(Xm), ncol(Xm),
                  as.numeric(y), as.numeric(C), as.numeric(epsilon),
                  as.numeric(tol), as.integer(max_iter),
                  sample_weight,
                  as.character(kernel), as.numeric(gamma),
                  as.integer(degree), as.numeric(coef0))
  }
  list(json = json, type = paste0("rust_svm_", task), task = task)
}

.predict_rust_svm <- function(engine, X, task) {
  Xm <- as.matrix(X)
  if (task == "classification") {
    .Call(wrap__ml_rust_svm_predict_clf,
          engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  } else {
    .Call(wrap__ml_rust_svm_predict_reg,
          engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  }
}

.proba_rust_svm <- function(engine, X) {
  Xm <- as.matrix(X)
  result <- .Call(wrap__ml_rust_svm_predict_proba,
                  engine$json, as.numeric(Xm), nrow(Xm), ncol(Xm))
  as.data.frame(.rust_proba_to_df(result))
}

# ── Rust engine type detection ──────────────────────────────────────────────

.is_rust_engine <- function(engine) {
  is.list(engine) && !is.null(engine$type) && startsWith(engine$type, "rust_")
}
