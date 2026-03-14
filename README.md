# ml

**A grammar of machine learning workflows.** Seven primitives. Four constraints. The evaluate/assess boundary prevents data leakage at call time.

[![PyPI](https://img.shields.io/pypi/v/mlw?label=PyPI&color=4f46e5)](https://pypi.org/project/mlw)
[![Python](https://img.shields.io/pypi/pyversions/mlw?label=Python)](https://pypi.org/project/mlw)
[![R](https://img.shields.io/badge/R-%3E%3D4.1-blue)](https://github.com/epagogy/ml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Paper: [Roth (2026)](https://doi.org/10.5281/zenodo.18905073) | [epagogy.ai](https://epagogy.ai)

---

## Ecosystem

```
epagogy/ml
├── al/              Shared algorithm engine (Rust)
│   ├── core/        11 algorithm families, pure Rust
│   ├── py/          PyO3 bindings → pip install ml-py
│   └── r/           extendr bindings → built by R configure
├── ml/              ML Grammar
│   ├── python/      pip install mlw
│   ├── r/           library(ml)
└── site/            epagogy.ai
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

---

## Python

```python
import ml

data = ml.dataset("tips")                        # bundled — no internet needed
s    = ml.split(data, "tip", seed=42)

leaderboard = ml.screen(s, "tip", seed=42)       # rank all algorithms
model  = ml.fit(s.train, "tip", seed=42)
ml.evaluate(model, s.valid)                       # practice exam — iterate freely

final  = ml.fit(s.dev, "tip", seed=42)           # retrain on all dev data
gate   = ml.validate(final, test=s.test)          # rules gate
verdict = ml.assess(final, test=s.test)           # final exam — do once
ml.save(final, "tip.mlw")
```

```bash
pip install mlw                  # core
pip install "mlw[xgboost]"       # + XGBoost (recommended)
pip install "mlw[all]"           # everything
```

> [Python package](ml/python/) · [PyPI](https://pypi.org/project/mlw)

---

## R

```r
library(ml)

data   <- ml_dataset("tips")                 # bundled — no internet needed
s      <- ml_split(data, "tip", seed = 42)
lb     <- ml_screen(s, "tip", seed = 42)
model  <- ml_fit(s$train, "tip", seed = 42)
ml_evaluate(model, s$valid)

final  <- ml_fit(s$dev, "tip", seed = 42)
verdict <- ml_assess(final, test = s$test)
```

```r
# Requires Rust toolchain + devtools (CRAN submission in progress)
remotes::install_github("epagogy/ml", subdir = "ml/r")
```

> [R package](ml/r/) · [GitHub](https://github.com/epagogy/ml)

---

## Algorithms

11 algorithm families with a native Rust backend. Zero sklearn dependency for the default path.

| Algorithm | Engine | Classification | Regression |
|-----------|--------|:-:|:-:|
| Random Forest | Rust | Y | Y |
| Extra Trees | Rust | Y | Y |
| Gradient Boosting | Rust | Y | Y |
| Decision Tree | Rust | Y | Y |
| Ridge / Linear | Rust | — | Y |
| Logistic | Rust | Y | — |
| Elastic Net | Rust | — | Y |
| Naive Bayes | Rust | Y | — |
| KNN | Rust | Y | Y |
| AdaBoost | Rust | Y | — |
| SVM | Rust | Y | Y |
| XGBoost | optional | Y | Y |

`engine="auto"` picks Rust. `engine="sklearn"` available as fallback.

---

## What ships

| Verb | Python | R | What it does |
|------|:------:|:-:|--------------|
| `split` | Y | Y | Three-way split, `.dev` property |
| `fit` | Y | Y | CV + holdout, 11 algorithm families |
| `predict` | Y | Y | Labels + probabilities |
| `evaluate` | Y | Y | Practice exam — iterate freely |
| `assess` | Y | Y | Final exam — do once |
| `explain` | Y | Y | Feature importance |
| `screen` | Y | Y | Algorithm leaderboard |
| `compare` | Y | Y | Fair side-by-side |
| `tune` | Y | Y | Random / Bayesian HPO |
| `stack` | Y | Y | OOF stacking |
| `validate` | Y | Y | Rules gate |
| `profile` | Y | Y | Dataset profiling |
| `drift` | Y | Y | KS + adversarial |
| `calibrate` | Y | Y | Platt / isotonic |
| `save` / `load` | Y | Y | `.mlw` / `.mlr` |

---

## Why ml?

**Where documentation fails, structure holds.** The evaluate/assess distinction is in every textbook and violated in 294 published papers ([Kapoor & Narayanan, 2023](https://doi.org/10.1016/j.patter.2023.100804)). The API makes the violation inexpressible.

**Same contract across languages.** A Python team and an R team run the same experiment, get the same result (1e-6 tolerance). Cross-language parity is a tested invariant, not a promise.

**Rust engine, no sklearn required.** 11 algorithm families in Rust via `al/`. The default `engine="auto"` path has zero sklearn dependency.

---

## Links

- **Homepage:** [epagogy.ai](https://epagogy.ai)
- **Paper:** [doi:10.5281/zenodo.18905073](https://doi.org/10.5281/zenodo.18905073)
- **Python:** [pypi.org/project/mlw](https://pypi.org/project/mlw)
- **R:** [github.com/epagogy/ml](https://github.com/epagogy/ml)
- **Issues:** [github.com/epagogy/ml/issues](https://github.com/epagogy/ml/issues)

---

## License

MIT — [Simon Roth](https://epagogy.ai), 2026.
