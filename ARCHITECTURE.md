# Architecture

Foundational decisions for the ml ecosystem. These are locked. Changes require explicit design review.

## What is a "grammar"?

Inspired by Chomsky's formal grammars and Wilkinson's GoG (*The Grammar of Graphics*): a finite set of typed composable primitives that generate all valid workflows in a domain while rejecting invalid ones at the type level.

What makes this different from a library: **closure** (combining verbs produces valid workflows), **generativity** (finite verbs generate infinite valid pipelines), **rejection** (invalid compositions fail at definition time, not runtime).

## Repository structure

```
epagogy/ml
├── al/                    Shared Rust engine (workspace)
│   ├── core/              11 algorithm families, pure Rust
│   ├── py/                PyO3 bindings
│   └── r/                 extendr bindings
├── python/                pip install mlw
├── r/                     library(ml)
├── datasets/              Golden test fixtures
└── LICENSE                MIT
```

Future grammars land as siblings: `rl/` (reinforcement learning), `ag/` (agents).

## 12 Locked Decisions

### 1. Layer model

**AL** (Algorithm Layer) implements Represent x Objective x Search x Compose. **ML**, **RL**, **AG** are workflow shells encoding domain-specific rules as types.

### 2. Monorepo

One repository, multiple packages. Atomic commits across grammar + engine.

### 3. AL = Spec, not implementation

Rust is one implementation. The `engine=` parameter dispatches between backends. Future backends plug into the same contract.

### 4. Package naming

Grammar = package = import name. PyPI: `mlw` (Python wrapper), `ml-py` (Rust wheels). CRAN: `ml`.

### 5. AL primitive contract

Every algorithm implements exactly four operations:

```
fit(X, y, params)        -> ModelState (JSON-serializable)
predict(ModelState, X)   -> predictions
serialize(ModelState)    -> bytes
deserialize(bytes)       -> ModelState
```

Everything else (CV, feature importance, calibration) lives in the grammar layer.

### 6. Error taxonomy

Two classes, never mixed:
- **AL errors**: `SingularMatrix`, `EmptyData`, `ConvergenceFailed`
- **Grammar errors**: `LeakageDetected`, `AssessmentRepeated`, `SeedMissing`

### 7. Testing contract

- Math invariants (Ridge KKT, logistic simplex, CART memorization)
- Cross-language parity: 1e-6 tolerance
- Roundtrip: serialize -> deserialize -> predict = bitwise identical
- Lattice: every algorithm x task cell tested

### 8. Versioning

Independent semver per package. Breaking AL change = major bump for all dependents.

### 9. Lattice artifact

Machine-readable YAML for every algorithm x task cell:

```yaml
- algorithm: random_forest
  task: binary_classification
  status: supported
  engine: ml
- algorithm: logistic
  task: regression
  status: rejected
  reason: "Classification-only"
```

### 10. Domain admission criteria

A domain qualifies when it has:
1. **Typed verbs** with typed inputs/outputs
2. **Closure** across compositions
3. **Generativity** from finite verbs
4. **Rejection** at definition time
5. **Terminal boundary** (assess for ML, converge for RL)

### 11. License

MIT for all packages.

### 12. Directory convention

`{language}/` for workflow packages, `al/` for the engine, `datasets/` for test data.

## Precedents

- **LLVM**: shared backend, multiple frontends
- **Chomsky hierarchy**: formal grammars as algebraic systems
- **scikit-learn Estimator protocol**: `fit`/`predict`/`transform`, evolved here with serialization-first design
- **tidyverse**: grammar-based package family (`ggplot2`, `dplyr`)
