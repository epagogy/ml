# ml <a href="https://epagogy.ai"><img src="https://epagogy.ai/assets/img/hex-ml-logo.svg" align="right" width="120" alt="ml hex logo"/></a>

A grammar of machine learning workflows for R.

[![R-CMD-check](https://github.com/epagogy/ml/actions/workflows/r-ci.yml/badge.svg)](https://github.com/epagogy/ml/actions/workflows/r-ci.yml)
[![MIT](https://img.shields.io/badge/license-MIT-4d7e7e)](https://github.com/epagogy/ml/blob/main/LICENSE)
[![epagogy.ai](https://img.shields.io/badge/_-epagogy.ai-555?style=flat&labelColor=0C0C0A&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCI+Cjxwb2x5Z29uIHBvaW50cz0iMjQsOCAzNy45LDE2IDM3LjksMzIgMjQsNDAgMTAuMSwzMiAxMC4xLDE2IgogICAgICAgICBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIG9wYWNpdHk9IjAuNyIvPgo8Y2lyY2xlIGN4PSIyNCIgY3k9IjI0IiByPSIxNiIKICAgICAgICBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIuNSIgb3BhY2l0eT0iMC45Ii8+Cjwvc3ZnPg==&logoColor=white)](https://epagogy.ai)

Split, fit, evaluate, assess — four verbs that encode the workflow from
Hastie, Tibshirani & Friedman (*The Elements of Statistical Learning*,
Ch. 7). The evaluate/assess boundary makes data leakage inexpressible:
`ml_evaluate()` runs on validation data and can be called freely;
`ml_assess()` runs on held-out test data and locks after one use.

## Installation

```r
install.packages("ml")
```

R >= 4.1.0. Optional backends: 'xgboost', 'ranger', 'glmnet', 'kknn',
'e1071', 'naivebayes', 'rpart'.

## Usage

```r
library(ml)

s <- ml_split(iris, "Species", seed = 42)

model <- ml_fit(s$train, "Species", seed = 42)
ml_evaluate(model, s$valid)       # check performance, tweak, repeat

final <- ml_fit(s$dev, "Species", seed = 42)
ml_assess(final, test = s$test)   # final exam — second call errors
```

`s$dev` is train + valid combined, used for the final refit before
assessment. This three-way split (train 60 / valid 20 / test 20) with a
`.dev` convenience accessor follows the textbook protocol exactly.

## Core verbs

| | |
|---|---|
| `ml_split()` | Stratified three-way split → `$train`, `$valid`, `$test`, `$dev` |
| `ml_fit()` | Train a model (per-fold preprocessing, deterministic seeding) |
| `ml_evaluate()` | Validation metrics — repeat freely |
| `ml_assess()` | Test metrics — once, final, locks after use |

These four are the grammar. Everything else extends it:

| | |
|---|---|
| `ml_screen()` | Algorithm leaderboard |
| `ml_tune()` | Hyperparameter search |
| `ml_stack()` | OOF ensemble stacking |
| `ml_predict()` | Class labels or probabilities |
| `ml_explain()` | Feature importance |
| `ml_compare()` | Side-by-side model comparison |
| `ml_validate()` | Pass/fail deployment gate |
| `ml_drift()` | Distribution shift detection (KS, chi-squared) |
| `ml_calibrate()` | Probability calibration (Platt, isotonic) |
| `ml_profile()` | Dataset summary |
| `ml_save()` / `ml_load()` | Serialize to `.mlr` |

## Algorithms

13 families. `engine = "auto"` uses the Rust backend when available;
`engine = "r"` forces the R package backend.

| Algorithm | String | Clf | Reg | Backend |
|---|---|:---:|:---:|---|
| Logistic | `"logistic"` | Y | | nnet |
| Decision Tree | `"decision_tree"` | Y | Y | rpart |
| Random Forest | `"random_forest"` | Y | Y | ranger |
| Extra Trees | `"extra_trees"` | Y | Y | Rust |
| Gradient Boosting | `"gradient_boosting"` | Y | Y | Rust |
| XGBoost | `"xgboost"` | Y | Y | xgboost |
| Ridge | `"linear"` | | Y | glmnet |
| Elastic Net | `"elastic_net"` | | Y | glmnet |
| SVM | `"svm"` | Y | Y | e1071 |
| KNN | `"knn"` | Y | Y | kknn |
| Naive Bayes | `"naive_bayes"` | Y | | naivebayes |
| AdaBoost | `"adaboost"` | Y | | Rust |
| Hist. Gradient Boosting | `"histgradient"` | Y | Y | Rust |

## Design notes

**Seeds.** `seed = NULL` auto-generates a seed and stores it on the
result for reproducibility. `seed = 42` gives full deterministic control.

**Per-fold preprocessing.** Scaling and encoding fit on training folds
only, never on validation or test. No information leaks across the
split boundary.

**Error messages.** Wrong column name? `ml_fit()` tells you what columns
exist. Wrong algorithm string? It lists the valid ones. Errors aim to
fix themselves.

## Citation

```
Roth, S. (2026). A Grammar of Machine Learning Workflows.
doi:10.5281/zenodo.18905073
```

## License

MIT. Simon Roth, 2026.
