# ml (R)

**A grammar of machine learning workflows.** Seven primitives. Four constraints. The evaluate/assess boundary prevents data leakage at call time.

[![CRAN](https://img.shields.io/cran/v/ml?color=4f46e5)](https://cran.r-project.org/package=ml)
[![R](https://img.shields.io/badge/R-%3E%3D4.1-blue)](https://cran.r-project.org/package=ml)
[![CI](https://github.com/epagogy/ml/actions/workflows/r-ci.yml/badge.svg)](https://github.com/epagogy/ml/actions/workflows/r-ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](../LICENSE)

Paper: [Roth (2026)](https://doi.org/10.5281/zenodo.18905073) | Python: [`mlw`](../python/) | Rust: [`ml`](../../al/) | [epagogy.ai](https://epagogy.ai)

---

## Install

```r
install.packages("ml")
```

Optional algorithm backends:

```r
install.packages(c("xgboost", "ranger", "glmnet", "kknn", "e1071", "naivebayes"))
```

**Requires R ≥ 4.1.0.**

---

## Quick start

```r
library(ml)

data   <- ml_dataset("tips")               # bundled — no internet needed
s      <- ml_split(data, "tip", seed = 42)

lb     <- ml_screen(s, "tip", seed = 42)   # rank all algorithms

# Iterate on validation
model  <- ml_fit(s$train, "tip", seed = 42)
ml_evaluate(model, s$valid)               # practice exam — repeat freely

# Finalize
final  <- ml_fit(s$dev, "tip", seed = 42) # retrain on train + valid
gate   <- ml_validate(final, test = s$test, rules = list(rmse = "<1.0"))
verdict <- ml_assess(final, test = s$test) # final exam — once only
ml_save(final, "tip.mlr")
```

---

## API

Two styles, same behavior:

```r
# Prefix style (primary — idiomatic R)
model <- ml_fit(s$train, "tip", seed = 42)

# Module style (mirrors Python ml.fit())
model <- ml$fit(s$train, "tip", seed = 42)
```

### Verbs

| Verb | Function | What it does |
|------|----------|-------------|
| Split | `ml_split()` | Three-way split — `$train`, `$valid`, `$test`, `$dev` |
| Profile | `ml_profile()` | Dataset summary + warnings |
| Screen | `ml_screen()` | Algorithm leaderboard |
| Fit | `ml_fit()` | Train with CV or holdout |
| Predict | `ml_predict()` | Labels or probabilities (`proba = TRUE`) |
| Evaluate | `ml_evaluate()` | Metrics on validation data — repeat freely |
| Assess | `ml_assess()` | Final metrics on test data — once only |
| Explain | `ml_explain()` | Feature importance |
| Tune | `ml_tune()` | Hyperparameter search |
| Stack | `ml_stack()` | OOF ensemble stacking |
| Compare | `ml_compare()` | Fair side-by-side of fitted models |
| Validate | `ml_validate()` | Pass/fail gate with rules |
| Drift | `ml_drift()` | Distribution drift detection |
| Shelf | `ml_shelf()` | Model freshness check |
| Save / Load | `ml_save()` / `ml_load()` | Persist models (`.mlr` format) |

### Algorithms

11 algorithm families with a native Rust backend (via [extendr](../../al/)). `engine = "auto"` picks Rust; `engine = "r"` forces the R fallback.

| Algorithm | `algorithm =` | Engine | Clf | Reg |
|-----------|-------------|--------|:---:|:---:|
| Random Forest | `"random_forest"` | Rust | ✓ | ✓ |
| Extra Trees | `"extra_trees"` | Rust | ✓ | ✓ |
| Decision Tree | `"decision_tree"` | Rust | ✓ | ✓ |
| Gradient Boosting | `"gradient_boosting"` | Rust | ✓ | ✓ |
| Linear (Ridge) | `"linear"` | Rust | — | ✓ |
| Logistic | `"logistic"` | Rust | ✓ | — |
| Elastic Net | `"elastic_net"` | Rust | — | ✓ |
| KNN | `"knn"` | Rust | ✓ | ✓ |
| Naive Bayes | `"naive_bayes"` | Rust | ✓ | — |
| AdaBoost | `"adaboost"` | Rust | ✓ | — |
| SVM | `"svm"` | Rust | ✓ | ✓ |
| XGBoost | `"xgboost"` | optional | ✓ | ✓ |

Default: XGBoost if installed, else Random Forest.

---

## Key design decisions

**`evaluate()` vs `assess()`** — the most important distinction in the API.
- `ml_evaluate()` is the practice exam. Run it on validation data as many times as you like.
- `ml_assess()` is the final exam. Run it once on held-out test data. ml warns if you repeat it.

This enforces the [methodology grammar](https://doi.org/10.5281/zenodo.18905073) (Roth, 2026), grounded in Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, Ch. 7.

**Seeds are optional but encouraged.** R has no keyword-only arguments, so `seed = NULL` auto-generates a seed and stores it in the model for reproducibility. Specify `seed = 42` for full control.

**Per-fold normalization.** All preprocessing (scaling, encoding) is fit on training folds only and applied to validation folds. No preprocessing leakage.

---

## Cross-language parity

The R package implements the same API contract as the Python package (`pip install mlw`). Same Rust algorithm kernels, same semantics. Numerical results match to 4 decimal places on the same data and seed.

---

## License

MIT — [Simon Roth](https://epagogy.ai), 2026.
