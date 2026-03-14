#' ml: A Grammar of Machine Learning Workflows
#'
#' @description
#' A structured workflow for supervised learning on tabular data. Core verbs
#' encode the workflow from Hastie, Tibshirani, and Friedman (2009,
#' ISBN:978-0-387-84857-0) 'The Elements of Statistical Learning',
#' Chapter 7: split data into train, validation, and test partitions;
#' fit a model; evaluate on validation data (iterate freely); assess on
#' test data (once, final). This evaluate-assess boundary prevents data
#' leakage by design.
#'
#' ## Workflow
#'
#' ```r
#' library(ml)
#'
#' s       <- ml_split(iris, "Species", seed = 42)
#' model   <- ml_fit(s$train, "Species", seed = 42)
#' metrics <- ml_evaluate(model, s$valid)   # iterate freely
#' verdict <- ml_assess(model, test = s$test)   # once, final
#' ```
#'
#' ## API
#'
#' All functions are available both as standalone `ml_verb()` functions and as
#' `ml$verb()` module-style calls. Both styles produce identical results.
#'
#' | Function | What it does |
#' |----------|-------------|
#' | [ml_split()] | Three-way train/valid/test split |
#' | [ml_fit()] | Fit a model |
#' | [ml_evaluate()] | Evaluate on validation data (iterate freely) |
#' | [ml_assess()] | Assess on test data (do once) |
#' | [ml_predict()] | Point predictions |
#' | [ml_predict_proba()] | Class probabilities |
#' | [ml_explain()] | Feature importance |
#' | [ml_screen()] | Compare all algorithms quickly |
#' | [ml_compare()] | Compare fitted models |
#' | [ml_tune()] | Hyperparameter tuning |
#' | [ml_stack()] | Ensemble stacking |
#' | [ml_validate()] | Validation gate with rules |
#' | [ml_drift()] | Data drift detection |
#' | [ml_profile()] | Data profiling and warnings |
#' | [ml_calibrate()] | Probability calibration |
#' | [ml_save()] / [ml_load()] | Model serialization (.mlr format) |
#' | [ml_algorithms()] | List available algorithms |
#' | [ml_dataset()] | Built-in datasets |
#'
#' ## Algorithms
#'
#' 13 algorithms are available. Backends in Suggests are loaded when needed;
#' without them, algorithms fall back to the optional Rust engine or report
#' as unavailable via [ml_algorithms()].
#'
#' | Algorithm | Classification | Regression | Backend |
#' |-----------|:---:|:---:|---------|
#' | "logistic" | yes | -- | nnet (base R) |
#' | "decision_tree" | yes | yes | 'rpart' |
#' | "random_forest" | yes | yes | 'ranger' |
#' | "extra_trees" | yes | yes | Rust engine |
#' | "gradient_boosting" | yes | yes | Rust engine |
#' | "xgboost" | yes | yes | 'xgboost' |
#' | "linear" (Ridge) | -- | yes | 'glmnet' |
#' | "elastic_net" | -- | yes | 'glmnet' |
#' | "svm" | yes | yes | 'e1071' |
#' | "knn" | yes | yes | 'kknn' |
#' | "naive_bayes" | yes | -- | 'naivebayes' |
#' | "adaboost" | yes | -- | Rust engine |
#' | "histgradient" | yes | yes | Rust engine |
#'
#' ## Notes
#'
#' - Formula interfaces are not supported. Pass the data frame and target column
#'   name as a string: `ml_fit(data, "target", seed = 42)`.
#' - Seeds are optional (default NULL auto-generates and stores) but recommended
#'   for reproducibility.
#'
#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @importFrom cli cli_warn cli_abort
#' @importFrom rlang abort
#' @importFrom stats predict sd median var coef setNames complete.cases
#' @importFrom utils packageVersion
## usethis namespace: end
NULL
