# ml (Rust)

**Native algorithm kernels for the ml ecosystem.** 11 algorithm families implemented from first principles in Rust — no BLAS, no system dependencies.

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Paper: [Roth (2026)](https://doi.org/10.5281/zenodo.19023838) | Python: [`mlw`](../ml/python/) | R: [`ml`](../ml/r/) | Julia: [`ML.jl`](../ml/julia/) | [epagogy.ai](https://epagogy.ai)

---

## Architecture

- **`ml`** (core) — Pure Rust algorithm implementations. No FFI. nalgebra for linear algebra.
- **`ml-py`** — [PyO3](https://pyo3.rs/) bindings. `import ml` uses this via `engine="auto"`.
- **`ml-r`** — [extendr](https://extendr.github.io/) bindings. `library(ml)` uses this via `engine="auto"`.

All models serialize to JSON for cross-language debuggability. Users see `engine="auto"` (Rust) or `engine="sklearn"` / `engine="r"` (fallback).

## Crates

| Crate | Purpose |
|-------|---------|
| `ml` | Core math: all 11 algorithm families |
| `ml-py` | PyO3 Python bindings (ships as `ml-py` on PyPI) |
| `ml-r` | extendr R bindings (built by R package `configure` script) |

## Algorithms

| Algorithm | Module | Classification | Regression |
|-----------|--------|:-:|:-:|
| Linear (Ridge) | `ml::linear` | — | Y |
| Logistic (OvR, L-BFGS) | `ml::logistic` | Y | — |
| Decision Tree (CART) | `ml::tree` | Y | Y |
| Random Forest | `ml::forest` | Y | Y |
| Extra Trees | `ml::forest` | Y | Y |
| Gradient Boosting (Newton) | `ml::gbt` | Y | Y |
| KNN (KD-tree) | `ml::knn` | Y | Y |
| Naive Bayes (Gaussian) | `ml::naive_bayes` | Y | — |
| Elastic Net | `ml::elastic_net` | — | Y |
| AdaBoost (SAMME) | `ml::adaboost` | Y | — |
| SVM (linear SMO) | `ml::svm` | Y | Y |

## Build

Requires Rust 1.87+ (`rustup update stable`).

```bash
cargo build
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

## Key implementation notes

**Ridge regression** — Closed-form via augmented normal equations. Cholesky for alpha > 0, LU fallback for collinear features. Sample weights via sqrt-row scaling.

**Logistic regression** — One-vs-Rest with L-BFGS (Nocedal-Wright Algorithm 7.4, m=10 history, weak Wolfe line search). Regularization uses the sklearn C convention.

**Gradient Boosting** — Newton leaves with histogram splits. Shared kernel for `gradient_boosting` and `histgradient` aliases.

**Random Forest / Extra Trees** — Parallel tree construction via rayon. Deterministic with seed.

## License

MIT — [Simon Roth](https://epagogy.ai), 2026.
