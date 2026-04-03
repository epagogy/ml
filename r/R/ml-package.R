#' ml: Machine Learning Workflows Made Simple
#'
#' @description
#' Implements the split-fit-evaluate-assess workflow from Hastie, Tibshirani,
#' and Friedman (2009, ISBN:978-0-387-84857-0) 'The Elements of Statistical
#' Learning', Chapter 7. Provides three-way data splitting with automatic
#' stratification, mandatory seeds for reproducibility, automatic data type
#' handling, and 8 algorithms out of the box.
#'
#' ## Workflow
#'
#' ```r
#' library(ml)
#'
#' s       <- ml_split(iris, "Species", seed = 42)
#' model   <- ml_fit(s$train, "Species", seed = 42)
#' metrics <- ml_evaluate(model, s$valid)
#' verdict <- ml_assess(model, test = s$test)
#' ```
#'
#' ## API
#'
#' All functions are available both as standalone `ml_verb()` functions and as
#' `ml$verb()` module-style calls. Both styles are equivalent.
#'
#' | Function | What it does |
#' |----------|-------------|
#' | [ml_split()] | Three-way train/valid/test split |
#' | [ml_fit()] | Fit a model |
#' | [ml_evaluate()] | Evaluate on validation data (iterate freely) |
#' | [ml_assess()] | Assess on test data (do once) |
#' | [ml_predict_proba()] | Class probabilities |
#' | [ml_explain()] | Feature importance |
#' | [ml_screen()] | Compare all algorithms quickly |
#' | [ml_compare()] | Compare fitted models |
#' | [ml_tune()] | Hyperparameter tuning |
#' | [ml_stack()] | Ensemble stacking |
#' | [ml_validate()] | Validation gate with rules |
#' | [ml_profile()] | Data profiling and warnings |
#' | [ml_save()] / [ml_load()] | Model serialization (.mlr format) |
#' | [ml_algorithms()] | List available algorithms |
#' | [ml_dataset()] | Built-in datasets |
#'
#' ## Algorithms
#'
#' | Algorithm | Classification | Regression | Package |
#' |-----------|:---:|:---:|---------|
#' | "xgboost" | yes | yes | 'xgboost' |
#' | "random_forest" | yes | yes | 'ranger' |
#' | "logistic" | yes | — | base R |
#' | "linear" (Ridge) | — | yes | 'glmnet' |
#' | "elastic_net" | — | yes | 'glmnet' |
#' | "svm" | yes | yes | 'e1071' |
#' | "knn" | yes | yes | 'kknn' |
#' | "naive_bayes" | yes | — | 'naivebayes' |
#'
#' LightGBM is available in Python 'mlw'. R support is planned for v1.1.
#'
#' ## Notes
#'
#' - Formula interfaces are not supported. Pass the data frame and target column
#'   name as a string: `ml_fit(data, "target", seed = 42)`.
#' - Seeds are optional (default NULL auto-generates) but recommended for
#'   reproducibility.
#'
#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @import cli
#' @import rlang
#' @importFrom stats predict sd median var coef setNames complete.cases
#' @importFrom utils packageVersion
#' @importFrom withr local_seed
## usethis namespace: end
NULL
