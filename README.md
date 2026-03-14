# ml <a href="https://epagogy.ai"><img src="https://epagogy.ai/assets/img/hex-ml-logo.svg" align="right" width="120" alt="ml hex logo"/></a>

A grammar of machine learning.
Split, fit, evaluate, assess — in Python and R.

<p align="center">
  <a href="https://github.com/epagogy/ml/actions/workflows/ci.yml"><img src="https://github.com/epagogy/ml/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-4d7e7e" alt="MIT"/></a>
  <a href="https://epagogy.ai"><img src="https://img.shields.io/badge/_-epagogy.ai-555?style=flat&labelColor=0C0C0A&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCI+Cjxwb2x5Z29uIHBvaW50cz0iMjQsOCAzNy45LDE2IDM3LjksMzIgMjQsNDAgMTAuMSwzMiAxMC4xLDE2IgogICAgICAgICBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIG9wYWNpdHk9IjAuNyIvPgo8Y2lyY2xlIGN4PSIyNCIgY3k9IjI0IiByPSIxNiIKICAgICAgICBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIuNSIgb3BhY2l0eT0iMC45Ii8+Cjwvc3ZnPg==&logoColor=white" alt="epagogy.ai"/></a>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.18905073">Paper</a> ·
  <a href="https://epagogy.ai">Website</a> ·
  <a href="python/">Python</a> ·
  <a href="r/">R</a>
</p>

---

Every ML textbook distinguishes validation from test.
294 published papers don't ([Kapoor & Narayanan, 2023](https://doi.org/10.1016/j.patter.2023.100804)).
ml makes the mistake inexpressible.

```python
import ml

s = ml.split(data, "churn", seed=42)

model = ml.fit(s.train, "churn", seed=42)
ml.evaluate(model, s.valid)              # iterate freely
ml.assess(model, test=s.test)            # once — second call errors
```

Same four verbs in R (`ml_fit`, `ml_assess`).
Same Rust engine underneath. Same result.

---

Beyond the core four: **screen** algorithms, **tune** hyperparameters,
**stack** ensembles, **validate** against deployment rules, monitor
**drift** in production. 11 Rust-native algorithms, 38 verbs, 173
bundled datasets.

| | Install | Docs |
|---|---|---|
| **Python** | `pip install mlw` | [python/](python/) |
| **R** | `remotes::install_github("epagogy/ml", subdir="r")` | [r/](r/) |

## Research

Roth, S. (2026). *A Grammar of Machine Learning Workflows.*
[doi:10.5281/zenodo.18905073](https://doi.org/10.5281/zenodo.18905073)

## License

MIT. [Simon Roth](https://epagogy.ai), 2026.

