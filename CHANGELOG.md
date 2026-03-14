# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] — 2026-02-25

### First public release

- Version reset to 1.0.0 for PyPI publication (internal development versions were 4.x)
- All changes below represent the cumulative work leading to this release
- Load compatibility: models saved with internal 4.x versions load correctly in 1.0.0
- Dependency bounds updated: `scikit-learn>=1.4,<2.0`, `skops>=0.9,<2.0`, `tabpfn>=0.1,<2.0`
- Dataset note: `tips` and `flights` are bundled; most others download from OpenML on first use

## [4.1.0] — 2026-02-24 (internal)

### Bug fixes (post-audit, W29–W35)

- `encode(method='target')` — string target columns (e.g. "yes"/"no") no longer crash; uses `pd.factorize` before target statistics
- `blend()` — proba DataFrames from `predict(proba=True)` now return correct `(n,)` shape instead of `(2n,)` flattened
- `blend()` — now returns `pd.Series` with index alignment instead of bare `numpy.ndarray`
- `check()` — raises `ModelError` (not `AssertionError`) on reproducibility failure; returns `CheckResult` dataclass instead of bare `bool`
- `check_data()` — integer-dtype ID columns now detected; `.errors` list populated before `DataError` is raised
- `compare()` — test-data peeking warning throttled to once per unique dataset per session
- `drift().distinguishable` — now always `bool` (was `None` for statistical method)
- `enough()` — spurious internal UserWarnings from evaluate() calls suppressed
- `model.cv_score` — renamed from `model.score` to avoid sklearn API collision; holdout-fitted models now return an early-stopping/OOB estimate instead of `None`
- `null_flags()` — accepts `target=` parameter to exclude the target column from null indicators (prevents leakage)
- `plot()` — SVM/KNN models with `data=` fall back to permutation importance cleanly; `seed=` parameter added
- `report()` — returns absolute file path via `os.path.abspath()`
- `scale()` — `columns=` now optional; auto-detects numeric columns when omitted (matches `impute()` behaviour)
- `shelf()` — returns `fresh=True` for freshly deployed holdout models instead of `None`
- `stack(weights='cv_score')` — OOF-weighted stacking now implemented; was previously `KeyError`
- `tune()` — `timeout=` now emits `DeprecationWarning`; passing both `timeout=` and `time_budget=` raises `ConfigError`
- `tune()` — per-trial seed variation applied correctly (`seed + trial_idx * 1000 + fold_idx`)
- `predict(confidence=X)` — warns when `confidence=` is set but `intervals=False`

## [4.1.0-initial] — 2026-02-24 (internal)

### Added

- `ml.plot()` — 10 visualization types: importance, confusion, ROC, calibration, residual, learning_curve, leaderboard, drift, waterfall, pdp
- `ml.select()` — feature selection (importance, correlation, permutation methods)
- `ml.quick()` — one-call workflow: split + screen + fit + evaluate
- `ml.help()` — interactive inline documentation for all verbs
- `ml.check()` — reproducibility verification (fit twice, assert bitwise identical)
- `ml.check_data()` — pre-flight data quality checks (ID columns, zero-variance, high-null, class imbalance, duplicates)
- `ml.report()` — automated HTML training report with metrics, feature importance, confusion matrix
- `ml.config()` — global configuration (n_jobs, verbose)
- `ml.quiet()` / `ml.verbose()` — warning control
- Polars DataFrame support via `narwhals`
- `gpu=` parameter on fit/tune/screen (True/False/"auto")
- Brier score in classification evaluation metrics
- PDP/ICE plots via `ml.plot(model, data=, kind="pdp", feature=)` — competitor parity with PyCaret/sklearn 1.5+
- Prediction intervals for regression: `ml.predict(model, data, intervals=True, confidence=0.90)`
- Datetime feature decomposition: `ml.encode(data, columns=["date"], method="datetime")`
- `time_budget=` on `tune()` — wall-clock timeout (FLAML-style)
- `patience=` on `tune(method="bayesian")` — early stopping
- Fuzzy matching in error messages ("Did you mean 'xgboost'?")
- `@deprecated` decorator infrastructure for future deprecations
- Structured debug logging via `logging.getLogger("ml")`
- `py.typed` PEP 561 marker
- `Model.score` property returning best CV metric

### Fixed

- `shelf()` sign convention corrected
- `drift()` accepts `target=` to exclude target column from both datasets
- `stack()` warns and continues on failed base models (was silent failure)
- `stack()` default algorithms now include linear for diversity
- `stack(balance=True)` — balanced class weights on meta-learner
- `calibrate()` double-calibration is idempotent (warns, returns as-is)
- `encode("one-hot")` / `encode("one_hot")` aliases now work
- `algorithms()` returns pandas DataFrame with task/dep/gpu metadata (was flat list)
- `DriftResult.feature_tests` is now a public attribute
- `evaluate()` / `assess()` docstrings clarify the develop-vs-ceremonial intent
- 25 edge case guards with helpful, actionable error messages
- Empty DataFrame → DataError at fit/predict/evaluate (not silent crash)
- Target missing → DataError listing available columns
- Infinite values → DataError listing affected columns

### Stats

- **929+ public tests** (728 → 929+)
- **893+ private moat tests** (7 previously-uncovered verbs now covered)
- **1,822+ total tests** across both suites
- **38 verbs** + 3 data helpers
- **14 algorithms** (all unchanged, no breaking changes)
- **10 plot types**

## [4.0.0b1] — 2026-02-01

Initial beta release.

- 31 verbs, 14 algorithms, per-fold normalization
- OOF stacking, GroupKFold, temporal splits
- Leakage detection, peeking guards
- Bayesian HPO (Optuna), SHAP explain, TTA
- Target encoding with fold alignment
- 728 public tests, 0 failures
