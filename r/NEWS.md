# ml 0.2.0

## Changed
* `ml_assess()` terminal constraint now enforced per-partition via content-addressed
  fingerprinting. Serialization and deepcopy bypasses are closed.

## New
* `ml_cv()`, `ml_cv_temporal()`, `ml_cv_group()` for cross-validation.
* `ml_verify()` for post-fit model verification.
* `ml_prepare()` for explicit preprocessing.
* Content-addressed provenance registry (`rlang::hash` fingerprinting).
* Cross-verb provenance checks (train/test from same split).

## Fixed
* Per-holdout assess enforcement: `ml_assess()` now rejects a second call on
  the same test partition regardless of which model calls it.
* Fixed `ml_prepare()` return value extraction (X and norm fields).

# ml 0.1.2

## New
* `ml_cv()`, `ml_cv_temporal()`, `ml_cv_group()` for cross-validation.
* `ml_verify()` for post-fit model verification.
* `ml_prepare()` for explicit preprocessing.
* Content-addressed provenance registry (`rlang::hash` fingerprinting).
* Cross-verb provenance checks (train/test from same split).

## Fixed
* Per-holdout assess enforcement: `ml_assess()` now rejects a second call on
  the same test partition regardless of which model calls it. The provenance
  registry tracks spent holdouts via content-addressed fingerprinting.
* Fixed `ml_prepare()` return value extraction (X and norm fields).

# ml 1.0.0

First stable CRAN release. The Hastie workflow in R.

## Core workflow

* Four-verb grammar: `ml_split()`, `ml_fit()`, `ml_evaluate()`, `ml_assess()`.
  The evaluate/assess boundary prevents data leakage by separating iterative
  model selection from final generalization estimates.
* Three-way data splitting (train/valid/test) with automatic stratification
  for classification targets.
* Mandatory reproducibility: seeds are auto-generated and stored when not
  provided explicitly.
* Per-fold preprocessing: encoding and scaling fit on training folds only.

## Algorithms

* 13 algorithms: logistic regression, linear regression (ridge), decision
  tree, random forest, extra trees, gradient boosting, histogram gradient
  boosting, XGBoost, KNN, naive Bayes, SVM, elastic net, AdaBoost.
* Optional backends via Suggests: xgboost, ranger, glmnet, e1071, kknn,
  naivebayes, rpart. Without these, algorithms fall back to the optional
  Rust engine or report as unavailable.

## Additional verbs

* `ml_screen()` for rapid algorithm comparison across all available backends.
* `ml_tune()` for hyperparameter tuning with random search and
  cross-validation.
* `ml_stack()` for model ensembling via out-of-fold stacking.
* `ml_drift()` for data drift detection (KS test).
* `ml_shelf()` for model staleness monitoring.
* `ml_explain()` for feature importance (impurity-based and coefficient-based).
* `ml_calibrate()` for probability calibration (Platt scaling).
* `ml_validate()` for pass/fail gating against user-defined rules.
* `ml_profile()` for dataset profiling.
* `ml_save()` / `ml_load()` for model persistence in `.mlr` format.

## Rust backend

* Optional compiled Rust engine via extendr bindings. Detected at install
  time by `configure`; no Rust required for a fully functional installation.
