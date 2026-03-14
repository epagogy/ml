# extendr-wrappers.R -- placeholder.
# Rust function calls (.Call) are in R/rust.R.
# This file exists for convention compatibility with rextendr-generated packages.

#' @useDynLib ml, .registration = TRUE
NULL

# .Call entry points registered via the compiled Rust shared library.
# Suppresses R CMD check NOTE: no visible binding for global variable.
utils::globalVariables(c(
  "wrap__ml_rust_ada_fit",
  "wrap__ml_rust_ada_importances",
  "wrap__ml_rust_ada_predict",
  "wrap__ml_rust_ada_predict_proba",
  "wrap__ml_rust_available",
  "wrap__ml_rust_en_fit",
  "wrap__ml_rust_en_predict",
  "wrap__ml_rust_extra_trees_fit_clf",
  "wrap__ml_rust_extra_trees_fit_reg",
  "wrap__ml_rust_forest_fit_clf",
  "wrap__ml_rust_forest_fit_reg",
  "wrap__ml_rust_forest_importances",
  "wrap__ml_rust_forest_predict_clf",
  "wrap__ml_rust_forest_predict_proba",
  "wrap__ml_rust_forest_predict_reg",
  "wrap__ml_rust_gbt_fit_clf",
  "wrap__ml_rust_gbt_fit_reg",
  "wrap__ml_rust_gbt_importances",
  "wrap__ml_rust_gbt_predict_clf",
  "wrap__ml_rust_gbt_predict_proba",
  "wrap__ml_rust_gbt_predict_reg",
  "wrap__ml_rust_knn_fit_clf",
  "wrap__ml_rust_knn_fit_reg",
  "wrap__ml_rust_knn_predict_clf",
  "wrap__ml_rust_knn_predict_proba",
  "wrap__ml_rust_knn_predict_reg",
  "wrap__ml_rust_linear_coef",
  "wrap__ml_rust_linear_fit",
  "wrap__ml_rust_linear_predict",
  "wrap__ml_rust_logistic_coef",
  "wrap__ml_rust_logistic_fit",
  "wrap__ml_rust_logistic_predict",
  "wrap__ml_rust_logistic_predict_proba",
  "wrap__ml_rust_nb_fit",
  "wrap__ml_rust_nb_predict",
  "wrap__ml_rust_nb_predict_proba",
  "wrap__ml_rust_partition_sizes",
  "wrap__ml_rust_shuffle",
  "wrap__ml_rust_svm_fit_clf",
  "wrap__ml_rust_svm_fit_reg",
  "wrap__ml_rust_svm_predict_clf",
  "wrap__ml_rust_svm_predict_proba",
  "wrap__ml_rust_svm_predict_reg",
  "wrap__ml_rust_tree_fit_clf",
  "wrap__ml_rust_tree_fit_reg",
  "wrap__ml_rust_tree_importances",
  "wrap__ml_rust_tree_predict_clf",
  "wrap__ml_rust_tree_predict_proba",
  "wrap__ml_rust_tree_predict_reg"
))
