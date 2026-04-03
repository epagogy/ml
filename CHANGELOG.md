# Changelog

All notable changes to `mlw` will be documented in this file.

## [1.2.0] — 2026-04-03

### Changed
- `assess()` is now fully independent of `evaluate()` — both call `_compute_metrics` directly
- Terminal assess constraint: per-partition content-addressed enforcement
- Validation metering K is now per-partition, not per-model

### Added
- `cv`, `cv_temporal`, `cv_group` exported (were imported but not in `__all__`)
- `verify`, `Evidence`, `audit` exported
- `deflate` — leaderboard deflation analysis
- `sparse`, `SparseFrame` — sparse text workflow (fused tokenize + sparse fit)
- Grammar conformance tests (CC1-CC8) in Python and R

### Fixed
- `assess()` on wrong partition no longer burns the one-shot counter
- `_split_receipt` AttributeError in `split.py`
- `ALGORITHM_ALIASES` missing from `_engines.py`
- `histgradient` sklearn fallback no longer passes Rust-only kwargs
- `_fingerprint()` handles non-DataFrame inputs gracefully
- Save/load for Rust tokenizer (`_RustVectorizer.max_features` AttributeError)

## [1.1.2] — 2026-03-14

### Fixed
- Per-holdout assess enforcement: `assess()` now rejects a second call on the same test partition
- `histgradient` sklearn fallback no longer passes Rust-only kwargs
- `_fingerprint()` handles non-DataFrame inputs gracefully

## [1.0.0] — 2026-02-25

First stable release. The Hastie workflow in code.

### Core Workflow
- `split`, `split_temporal`, `split_group` — stratified, temporal, and group-aware holdout
- `fit` — train with automatic per-fold normalization
- `predict`, `predict_proba` — point predictions and calibrated probabilities
- `evaluate` — practice exam (iterate freely)
- `assess` — final exam (once per model, enforced)
- `explain` — feature importance
- `save`, `load` — skops-based serialization

### Screening and Tuning
- `screen`, `compare`, `tune`, `stack`, `validate`

### Monitoring
- `drift`, `shelf`, `calibrate`

### Preprocessing
- `tokenize`, `scale`, `encode`, `impute`, `pipe`, `discretize`, `null_flags`

### Analysis
- `profile`, `interact`, `enough`, `leak`, `check`, `check_data`
- `optimize`, `nested_cv`, `blend`, `cluster_features`, `select`
- `plot`, `report`

### Algorithms
- 13 families: decision_tree, random_forest, extra_trees, gradient_boosting, histgradient, logistic, linear, elastic_net, naive_bayes, knn, adaboost, svm
- Optional: xgboost, lightgbm, catboost
