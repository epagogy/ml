# mlw <a href="https://epagogy.ai"><img src="https://epagogy.ai/assets/img/hex-ml-logo.svg" align="right" width="120" alt="ml hex logo"/></a>

**A grammar of machine learning workflows for Python.**
Four verbs prevent data leakage by construction. 16 algorithms, 11 Rust-native.

<p align="center">
  <a href="https://pypi.org/project/mlw"><img src="https://img.shields.io/pypi/v/mlw?color=4f46e5" alt="PyPI"/></a>
  <a href="https://pypi.org/project/mlw"><img src="https://img.shields.io/pypi/pyversions/mlw" alt="Python"/></a>
  <a href="https://github.com/epagogy/ml/actions/workflows/ci.yml"><img src="https://github.com/epagogy/ml/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/license-MIT-4d7e7e" alt="MIT"/></a>
  <a href="https://epagogy.ai"><img src="https://img.shields.io/badge/_-epagogy.ai-555?style=flat&labelColor=0C0C0A&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCI+Cjxwb2x5Z29uIHBvaW50cz0iMjQsOCAzNy45LDE2IDM3LjksMzIgMjQsNDAgMTAuMSwzMiAxMC4xLDE2IgogICAgICAgICBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIG9wYWNpdHk9IjAuNyIvPgo8Y2lyY2xlIGN4PSIyNCIgY3k9IjI0IiByPSIxNiIKICAgICAgICBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIuNSIgb3BhY2l0eT0iMC45Ii8+Cjwvc3ZnPg==&logoColor=white" alt="epagogy.ai"/></a>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.19023838">Paper</a> ·
  <a href="../r/">R</a> ·
  <a href="../al/">Rust engine</a> ·
  <a href="https://epagogy.ai">epagogy.ai</a>
</p>

## Install

```bash
pip install mlw                       # core (11 Rust-native algorithms)
pip install "mlw[xgboost]"            # + XGBoost
pip install "mlw[all]"                # everything
```

Python 3.10+. Also available: `lightgbm`, `catboost`, `plots`, `optuna`, `dev`.

## Quickstart

```python
import ml

data = ml.dataset("churn")
s = ml.split(data, "churn", seed=42)

lb = ml.screen(s, "churn", seed=42)          # rank all algorithms
model = ml.fit(s.train, "churn", seed=42)
ml.evaluate(model, s.valid)                   # iterate freely

final = ml.fit(s.dev, "churn", seed=42)       # retrain on train+valid
ml.assess(final, test=s.test)                 # once — second call errors
```

## Why ml

**The evaluate/assess boundary.** `evaluate` runs on validation data —
call it as often as you like. `assess` runs on held-out test data and
locks after one use. No discipline required; the API makes leakage
inexpressible. This encodes the protocol from Hastie, Tibshirani &
Friedman (*ESL*, Ch. 7).

**Three-way split with `.dev`.** Train (60%), valid (20%), test (20%).
`s.dev` = train + valid combined for the final refit before assessment.

**47 verbs, one import.** From `check_data` and `split` through `tune`,
`stack`, `explain`, `drift`, and `shelf`. Everything returns plain
objects you can inspect, compare, or serialize.

**168 datasets.** `tips` and `flights` are bundled. The rest download
from OpenML on first use and cache locally.

## Highlights

**Tune.** Random, Bayesian (`mlw[optuna]`), or grid search.

```python
tuned = ml.tune(s.train, "churn", algorithm="xgboost", seed=42, n_trials=50)
ml.evaluate(tuned, s.valid)
```

**Ship gate.** Hard pass/fail contracts before deployment.

```python
ml.validate(final, test=s.test, rules={"accuracy": ">0.85"})
```

**Drift.** Catch distribution shift before users notice.

```python
ml.drift(reference=s.train, new=live_data).shifted
```

## Algorithms

16 families. `engine="auto"` picks Rust when available. `engine="sklearn"`
forces scikit-learn fallback.

| Algorithm | String | Engine | Clf | Reg |
|---|---|---|:---:|:---:|
| Random Forest | `"random_forest"` | Rust | Y | Y |
| Extra Trees | `"extra_trees"` | Rust | Y | Y |
| Gradient Boosting | `"gradient_boosting"` | Rust | Y | Y |
| Hist. Gradient Boosting | `"histgradient"` | Rust | Y | Y |
| Decision Tree | `"decision_tree"` | Rust | Y | Y |
| Ridge | `"linear"` | Rust | · | Y |
| Logistic | `"logistic"` | Rust | Y | · |
| Elastic Net | `"elastic_net"` | Rust | · | Y |
| KNN | `"knn"` | Rust | Y | Y |
| Naive Bayes | `"naive_bayes"` | Rust | Y | · |
| AdaBoost | `"adaboost"` | Rust | Y | · |
| SVM | `"svm"` | Rust | Y | Y |
| XGBoost | `"xgboost"` | optional | Y | Y |
| LightGBM | `"lightgbm"` | optional | Y | Y |
| CatBoost | `"catboost"` | optional | Y | Y |
| TabPFN | `"tabpfn"` | optional | Y | · |

## Citation

```
Roth, S. (2026). A Grammar of Machine Learning Workflows.
doi:10.5281/zenodo.19023838
```

## License

MIT. [Simon Roth](https://epagogy.ai), 2026.

