# Changelog

All notable changes to `mlw` will be documented in this file.

## [1.2.0] - 2026-03-28

### Changed
- `assess()` is now fully independent of `evaluate()` — no call relationship. Both call `_compute_metrics` directly for scoring. Paper claim verified.
- Terminal assess constraint: per-partition content-addressed enforcement. Serialization and `copy.deepcopy` bypasses are closed — partition-level guard fires regardless of model identity.
- Atomic check-and-mark in provenance registry prevents TOCTOU races between concurrent threads.
- **Validation metering K** is now per-partition, not per-model. `evaluate(model, s.valid)` meters the valid partition's fingerprint in the provenance registry. `assess()` reads K from the valid partition of the same split and seals it into `Evidence._K`. Three models evaluating on the same `s.valid` → K=3. Train evaluations do not inflate K.

### Added
- `cv`, `cv_temporal`, `cv_group` exported in `__all__` (were imported but inaccessible via `ml.cv()`)
- `verify` exported in `__all__` (provenance integrity check)
- `Evidence` and `audit` exported in `__all__`
- `Evidence._K`: selection pressure meter — number of `evaluate()` calls on the validation partition before terminal assessment

### Fixed
- `assess()` on wrong partition (e.g., `s.train`) no longer burns the one-shot counter
- `_split_receipt` AttributeError in `split.py` (removed non-existent kwarg)
- `ALGORITHM_ALIASES` missing from `_engines.py` (ImportError on fit)
- `histgradient` sklearn fallback no longer passes Rust-only kwargs
- `_fingerprint()` handles non-DataFrame inputs gracefully (returns None instead of crashing)

## [1.1.2] - 2026-03-14

### Fixed
- Per-holdout assess enforcement: `assess()` now rejects a second call on the same test partition regardless of which model calls it (provenance registry tracks spent holdouts)
- `histgradient` sklearn fallback no longer passes Rust-only kwargs (`reg_lambda`, `gamma`, etc.)
- `_fingerprint()` handles non-DataFrame inputs gracefully (returns None instead of crashing)

## [1.0.0] - 2026-03-07

First stable release. The Hastie workflow in code.

### Core Workflow (10 verbs)
- `split`, `split_temporal`, `split_group` — stratified, temporal, and group-aware holdout
- `fit` — train with automatic per-fold normalization and cross-validation
- `predict`, `predict_proba` — point predictions and calibrated probabilities
- `evaluate` — practice exam (iterate freely)
- `assess` — final exam (once per model, enforced)
- `explain` — permutation importance, coefficients, or tree-based importance
- `save`, `load` — skops-based serialization (no pickle)

### Screening and Tuning (5 verbs)
- `screen` — leaderboard across algorithm families
- `compare` — side-by-side model comparison without re-fitting
- `tune` — Optuna hyperparameter search
- `stack` — blended ensemble with automatic base learner selection
- `validate` — rules gate before deployment

### Monitoring (3 verbs)
- `drift` — label-free distribution shift detection
- `shelf` — model staleness scoring
- `calibrate` — Platt scaling and isotonic calibration

### Preprocessing (7 verbs)
- `tokenize` — TF-IDF text vectorization (native, no sklearn)
- `scale`, `encode`, `impute` — column-aware transformers
- `pipe` — composable preprocessing pipelines
- `discretize` — equal-width/equal-frequency binning
- `null_flags` — missing-indicator columns

### Analysis (10 verbs)
- `profile` — dataset statistics and quality warnings
- `interact` — pairwise feature interaction detection
- `enough` — learning curve saturation analysis
- `leak` — target leakage detection
- `check`, `check_data` — data quality checks
- `optimize` — threshold optimization on OOF predictions
- `nested_cv` — unbiased performance estimation
- `blend` — prediction blending
- `cluster_features`, `select` — feature reduction
- `plot`, `report` — visualization and summary

### Algorithms (14 families)
- **Rust backend (ml-py):** decision_tree, random_forest, extra_trees, gradient_boosting, histgradient, logistic, linear, ridge, elastic_net, naive_bayes, knn, adaboost, svm
- **Optional:** xgboost, lightgbm, catboost

### Grammar
- 4-state workflow DFA: CREATED → FITTED → EVALUATED → ASSESSED (terminal)
- Content-addressed data fingerprinting (SHA-256)
- Partition guards: `fit()` rejects test data, `assess()` rejects training data
- Mandatory seeds for reproducibility

### Languages
- Python (`mlw` on PyPI)
- R (`ml` on CRAN)
