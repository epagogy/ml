# Architecture

This document records the foundational decisions for the ml ecosystem. These are locked — changes require explicit discussion and review.

## What is a "grammar"?

Inspired by Chomsky's formal grammars and Wilkinson's *Grammar of Graphics*, a **grammar** here means a finite set of typed composable primitives that generate all valid workflows in a domain while rejecting invalid ones at the type level. It is not a library of functions — it is a closed algebraic system where composition is guaranteed safe.

What makes this different from a library: closure (combining verbs produces valid workflows), generativity (finite verbs generate infinite valid pipelines), and rejection (invalid compositions fail at definition time, not at runtime).

## Repository Structure

```
epagogy/ml
├── al/                    # Algorithm Layer — shared engine (Rust workspace)
│   ├── core/              #   Core algorithms: fit, predict, serialize
│   ├── py/                #   PyO3 bindings → ml-py on PyPI
│   └── r/                 #   extendr bindings → linked into R package
├── ml/                    # ML Grammar — first workflow grammar
│   ├── python/            #   import ml — PyPI package "mlw"
│   ├── r/                 #   library(ml) — CRAN package
│   └── datasets/          #   Bundled test/benchmark data
├── site/                  # epagogy.ai — project website
├── .github/workflows/     # CI/CD
├── ARCHITECTURE.md        # This file
├── README.md
├── LICENSE                # MIT
└── CONTRIBUTING.md
```

Future grammars land as siblings to `ml/`:

```
├── rl/                    # Reinforcement Learning Grammar (planned)
│   ├── python/
│   └── r/
├── ag/                    # Agent Grammar (planned)
│   ├── python/
│   └── r/
```

## 12 Locked Decisions

### 1. Layer Model

**AL** (Algorithm Layer) implements Represent × Objective × Search × Compose — the universal engine inside every grammar. **ML**, **RL**, **AG** are workflow shells that encode domain-specific rules as types.

AL is validated for supervised ML. Extension to RL and AG is hypothesized but unproven. If no second grammar ships within 18 months of ml v1.0.0, revisit the al/ naming.

### 2. Monorepo

One repository, multiple grammar packages. Atomic commits across grammar + engine. Cargo workspace for Rust, independent Python/R packages per grammar.

### 3. AL = Spec, Not Implementation

Rust is one implementation of the AL spec. The `engine=` parameter already dispatches between backends (`auto`, `ml`, `sklearn`, `native`). Future backends (GPU, other languages) plug into the same contract.

### 4. Package Naming

Grammar = package = import name. `ml` in Python/R, `rl` when it ships, `al` for the engine. PyPI: `mlw` (pure Python wrapper), `ml-py` (Rust wheels). CRAN: `ml`.

### 5. AL Primitive Contract

Every algorithm implements exactly four operations:

```
fit(X, y, params) → ModelState (JSON-serializable)
predict(ModelState, X) → predictions
serialize(ModelState) → bytes
deserialize(bytes) → ModelState
```

Everything else (cross-validation, feature importance, calibration) lives in the grammar layer.

### 6. Error Taxonomy

Two error classes, never mixed:
- **AL errors**: `SingularMatrix`, `EmptyData`, `ConvergenceFailed` — algorithm-level
- **Grammar errors**: `LeakageDetected`, `AssessmentRepeated`, `SeedMissing` — workflow-level

### 7. Testing Contract

- Math invariants (Ridge KKT, logistic simplex, CART memorization)
- Cross-language parity: 1e-6 tolerance for tabular, directional for neural
- Roundtrip: serialize → deserialize → predict = bitwise identical
- Lattice validity: every cell in the algorithm×task lattice tested

### 8. Versioning

Independent semver per package. Breaking AL change = major version bump for all grammars that depend on it. Grammar changes don't affect AL version.

### 9. Lattice Artifact

Machine-readable YAML describing every algorithm×task cell:

```yaml
- algorithm: random_forest
  task: binary_classification
  status: supported
  engine: ml  # Rust
  reason: null
- algorithm: logistic
  task: regression
  status: rejected
  reason: "Logistic regression is classification-only"
```

### 10. Grammar Admission Criteria

A domain qualifies as a grammar when it has:
1. **Typed primitives** — finite verb set with typed inputs/outputs
2. **Closure** — composing verbs produces valid workflows
3. **Generativity** — finite verbs generate infinite valid pipelines
4. **Rejection** — invalid compositions fail at definition time
5. **Terminal boundary** — clear "done" state (assess for ML, converge for RL)

### 11. License

MIT for all packages. Code is open.

### 12. Directory Convention

`{grammar}/{language}/` for workflow packages, `al/` for the shared engine. Language directories contain self-contained packages that can be installed independently.

## Intentional Tech Debt

- **`al/core` crate name**: The Rust crate is still `name = "ml"` in Cargo.toml to avoid touching 20+ `use ml::` import statements across the Rust codebase. The directory is `al/core/` but the crate identity stays `ml`. This is cosmetic debt — no user impact, no correctness impact.

## Precedents

- **LLVM**: Shared backend serving multiple language frontends (Clang, Rust, Swift). AL serves multiple grammar frontends.
- **Chomsky hierarchy**: Formal grammars as algebraic systems with generative capacity and rejection. Direct inspiration for grammar admission criteria.
- **scikit-learn Estimator protocol**: `fit`/`predict`/`transform` as the universal ML contract. AL's primitive contract is a stricter, serialization-first evolution.
- **Cargo workspaces**: Independent semver for crates within a single repository. Same pattern here.
- **tidyverse**: Grammar-based package family (`ggplot2` = grammar of graphics, `dplyr` = grammar of data manipulation). ML grammar follows this philosophy for machine learning.
