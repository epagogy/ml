# mlw

**A grammar of machine learning workflows.** Seven primitives. Four constraints. The evaluate/assess boundary prevents data leakage at call time.

[![PyPI](https://img.shields.io/pypi/v/mlw?color=4f46e5)](https://pypi.org/project/mlw)
[![Python](https://img.shields.io/pypi/pyversions/mlw)](https://pypi.org/project/mlw)
[![CI](https://github.com/epagogy/ml/actions/workflows/ci.yml/badge.svg)](https://github.com/epagogy/ml/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](../LICENSE)

Paper: [Roth (2026)](https://doi.org/10.5281/zenodo.18905073) | R: [`ml`](../r/) | Rust: [`ml`](../../al/) | [epagogy.ai](https://epagogy.ai)

```python
import ml

data = ml.dataset("tips")               # bundled — no internet needed
s    = ml.split(data, "tip", seed=42)

# Explore
ml.check_data(data, "tip")              # sanity check: missing, leakage, imbalance
leaderboard = ml.screen(s, "tip", seed=42)   # rank all algorithms in 60s

# Iterate on validation (free to repeat)
model   = ml.fit(s.train,  "tip", seed=42)
tuned   = ml.tune(s.train, "tip", algorithm="xgboost", seed=42)
stacked = ml.stack(s.train, "tip", seed=42)
ml.evaluate(tuned, s.valid)             # metrics — iterate freely here
ml.compare([tuned, stacked], s.valid)   # fair side-by-side

# Finalize
final   = ml.fit(s.dev, "tip", seed=42)     # retrain on train+valid
gate    = ml.validate(final, test=s.test)    # rules gate — optional
verdict = ml.assess(final, test=s.test)      # final exam — do once
ml.save(final, "tip.mlw")
```

## Install

```bash
pip install mlw                       # core: scikit-learn, pandas, skops
pip install "mlw[xgboost]"            # + XGBoost (recommended)
pip install "mlw[lightgbm]"           # + LightGBM
pip install "mlw[catboost]"           # + CatBoost
pip install "mlw[plots]"              # + matplotlib for ml.plot()
pip install "mlw[optuna]"             # + Bayesian HPO (faster than random search)
pip install "mlw[all]"                # everything above
pip install "mlw[dev]"                # + pytest, ruff (contributing)
```

**Requires Python 3.10+.** See [notebooks/](notebooks/) for a 4-part tutorial.

> **Datasets:** `tips` and `flights` ship with the package (no internet needed). Most others (including `churn`, `fraud`) download from OpenML on first use. Run `ml.datasets()` to see all 173 available.

---

## The Full Workflow

Every step maps to the [methodology grammar](https://doi.org/10.5281/zenodo.18905073). Follow in order; skip what you don't need.

```python
import ml

data = ml.dataset("churn")

# ── 1. Sanity check ───────────────────────────────────────────────────────────
report = ml.check_data(data, "churn")   # missing values, leakage, imbalance
# report.issues   → list of warnings
# report.passed   → bool

# ── 2. Split ─────────────────────────────────────────────────────────────────
s = ml.split(data, "churn", seed=42)
# s.train  (60%)  — fit here
# s.valid  (20%)  — evaluate here, iterate freely
# s.test   (20%)  — assess here, once
# s.dev           — train + valid combined, for final refit

# ── 3. Screen ────────────────────────────────────────────────────────────────
lb = ml.screen(s, "churn", seed=42)    # quick rank of all algorithms
#   algorithm       accuracy   f1     roc_auc   time_s
#   xgboost         0.87       0.84   0.91      1.2
#   random_forest   0.85       0.82   0.89      0.8
#   logistic        0.83       0.80   0.87      0.1

# ── 4. Fit ───────────────────────────────────────────────────────────────────
model = ml.fit(s.train, "churn", seed=42)
model = ml.fit(s.train, "churn", algorithm="xgboost", seed=42)
model = ml.fit(s.train, "churn", algorithm="xgboost", seed=42,
               max_depth=6, learning_rate=0.1)   # direct hyperparams

# ── 5. Evaluate (iterate freely) ─────────────────────────────────────────────
metrics = ml.evaluate(model, s.valid)   # dict[str, float]
# {"accuracy": 0.87, "f1": 0.84, "roc_auc": 0.91}

# ── 6. Tune ──────────────────────────────────────────────────────────────────
tuned = ml.tune(s.train, "churn", algorithm="xgboost", seed=42, n_trials=50)
tuned.best_params_     # {"max_depth": 6, "learning_rate": 0.12, ...}
tuned.tuning_history_  # DataFrame: trial, score, params
ml.evaluate(tuned, s.valid)

# ── 7. Stack ─────────────────────────────────────────────────────────────────
stacked = ml.stack(s.train, "churn", seed=42)
stacked = ml.stack(s.train, "churn", seed=42,
    models=["xgboost", "random_forest", "logistic"],
    meta="logistic")

# ── 8. Compare ───────────────────────────────────────────────────────────────
ml.compare([model, tuned, stacked], s.valid)   # no re-fitting

# ── 9. Finalize + Validate ────────────────────────────────────────────────────
final = ml.fit(s.dev, "churn", seed=42)        # retrain on train+valid
gate  = ml.validate(final, test=s.test,        # optional pass/fail gate
                    rules={"accuracy": ">0.80"})
# gate.passed → True / False

# ── 10. Assess (do once) ─────────────────────────────────────────────────────
verdict = ml.assess(final, test=s.test)        # warns on repeat

# ── 11. Explain ──────────────────────────────────────────────────────────────
ml.explain(model)                              # feature importances

# ── 12. Save / Load ──────────────────────────────────────────────────────────
ml.save(final, "churn.mlw")
loaded = ml.load("churn.mlw")
```

---

## Algorithms

11 algorithm families with a native Rust backend (via [PyO3](../../al/)). Optional sklearn and XGBoost backends available.

| Algorithm | String | Engine | Clf | Reg | `explain()` | Auto-scale |
|-----------|--------|--------|:---:|:---:|:-----------:|:----------:|
| Random Forest | `"random_forest"` | Rust | yes | yes | yes | no |
| Extra Trees | `"extra_trees"` | Rust | yes | yes | yes | no |
| Decision Tree | `"decision_tree"` | Rust | yes | yes | yes | no |
| Gradient Boosting | `"gradient_boosting"` | Rust | yes | yes | yes | no |
| Ridge (linear) | `"linear"` | Rust | — | yes | yes\* | yes |
| Logistic | `"logistic"` | Rust | yes | — | yes\* | yes |
| Elastic Net | `"elastic_net"` | Rust | — | yes | yes\* | yes |
| KNN | `"knn"` | Rust | yes | yes | no | yes |
| Naive Bayes | `"naive_bayes"` | Rust | yes | — | no | yes |
| AdaBoost | `"adaboost"` | Rust | yes | — | no | no |
| SVM | `"svm"` | Rust | yes | yes | no | yes |
| XGBoost | `"xgboost"` | optional | yes | yes | yes | no |
| LightGBM | `"lightgbm"` | optional | yes | yes | yes | no |
| CatBoost | `"catboost"` | optional | yes | yes | yes | no |
| Auto | `"auto"` | — | yes | yes | yes | — |

\* Linear models use absolute coefficients as importance.
Auto selects XGBoost if installed, else Random Forest.
SVM/KNN/Logistic/Linear models are auto-scaled; tree models never are.
`engine="auto"` picks the Rust backend. `engine="sklearn"` forces the sklearn fallback.

```python
model = ml.fit(s.train, "churn", algorithm="random_forest", seed=42)
```

---

## Verb Cheat Sheet (38 verbs)

### Core workflow

| Verb | What it does |
|------|-------------|
| `ml.check_data(data, target)` | Sanity check: missing, leakage, imbalance |
| `ml.split(data, target, seed=)` | Three-way split → `.train` / `.valid` / `.test` / `.dev` |
| `ml.screen(s, target, seed=)` | Quick algorithm ranking (default hyperparams) |
| `ml.fit(data, target, seed=)` | Train a model (auto algorithm, auto dtype handling) |
| `ml.predict(model, data)` | Predictions as `pd.Series` |
| `ml.evaluate(model, data)` | Metrics on validation → `dict[str, float]` |
| `ml.tune(data, target, algorithm=, seed=)` | Hyperparameter search |
| `ml.stack(data, target, seed=)` | Ensemble stacking |
| `ml.compare(models, data)` | Fair side-by-side evaluation |
| `ml.assess(model, test=)` | Final exam metrics (keyword-only, warns on repeat) |
| `ml.explain(model)` | Feature importances |
| `ml.validate(model, test=, rules=)` | Guarantee gate (thresholds + regression check) |
| `ml.save(model, path)` | Persist to disk (skops, no pickle) |
| `ml.load(path)` | Restore from disk |

### Preprocessing

| Verb | What it does |
|------|-------------|
| `ml.scale(data, method=)` | Feature scaling: `"standard"` / `"minmax"` / `"robust"` |
| `ml.encode(data, columns=)` | Categorical encoding: `"ordinal"` / `"onehot"` |
| `ml.impute(data, strategy=)` | Missing value imputation |
| `ml.tokenize(data, column=)` | Text → TF-IDF features |
| `ml.null_flags(data)` | Add binary columns marking NaN positions |
| `ml.discretize(data, column=)` | Bin continuous features into intervals |
| `ml.pipe(*steps)` | Chain preprocessing steps into a pipeline |

### Analysis & diagnostics

| Verb | What it does |
|------|-------------|
| `ml.profile(data, target)` | Shape, types, missing, warnings |
| `ml.leak(data, target)` | Target leakage detection |
| `ml.enough(s, target, seed=)` | Sample size sufficiency check |
| `ml.interact(model, data=, seed=)` | Feature interaction effects (SHAP-free) |
| `ml.report(model, data=)` | Full model report (PDF / HTML) |
| `ml.plot(model, data=)` | Visual diagnostics (requires `mlw[plots]`) |
| `ml.check(model)` | Post-fit integrity check |

### Production monitoring

| Verb | What it does |
|------|-------------|
| `ml.drift(reference=, new=)` | Label-free drift detection (KS / chi²) |
| `ml.shelf(model, new=, target=)` | Model freshness check (needs labels) |
| `ml.calibrate(model, data=)` | Probability calibration (isotonic / Platt) |

### Advanced

| Verb | What it does |
|------|-------------|
| `ml.nested_cv(data, target, seed=)` | Nested cross-validation (unbiased estimate) |
| `ml.optimize(data, target, seed=)` | Full HPO with Optuna (requires `mlw[optuna]`) |
| `ml.blend(models, data=)` | Soft-vote blending across models |
| `ml.select(data, target)` | Automated feature selection |
| `ml.cluster_features(data)` | Group correlated features |

### Utilities

| Verb | What it does |
|------|-------------|
| `ml.dataset(name)` | Load a built-in dataset by name |
| `ml.datasets()` | Searchable metadata table of all 173 datasets |
| `ml.algorithms()` | List all available algorithm strings |
| `ml.help()` | Print verb cheat sheet in terminal |
| `ml.config` | Global configuration object |

---

## sklearn Migration

Before (12 lines):

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X = data.drop(columns=["churn"])
y = le.fit_transform(data["churn"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(accuracy_score(y_test, preds), f1_score(y_test, preds), roc_auc_score(y_test, preds))
```

After (4 lines):

```python
import ml
s     = ml.split(data, "churn", seed=42)
model = ml.fit(s.train, "churn", seed=42)
print(ml.evaluate(model, s.test))
```

What you gain: three-way split (no test-set contamination), auto-encoding, auto-scaling, all metrics in one call, reproducibility by default.

---

## Hyperparameter Tuning

```python
# Random search (default, no extra deps)
tuned = ml.tune(s.train, "churn", algorithm="xgboost", seed=42, n_trials=50)

# Bayesian search — faster convergence (requires mlw[optuna])
tuned = ml.tune(s.train, "churn", algorithm="xgboost", seed=42,
                method="bayesian", n_trials=50)

# Grid search — exhaustive, auditable
tuned = ml.tune(s.train, "churn", algorithm="xgboost", seed=42,
                method="grid",
                params={"max_depth": [3, 5, 7], "n_estimators": [100, 200]})

# Custom search space
tuned = ml.tune(s.train, "churn", algorithm="xgboost", seed=42,
                params={"max_depth": (3, 8), "learning_rate": (0.01, 0.3)})

tuned.best_params_      # {"max_depth": 6, "learning_rate": 0.12, ...}
tuned.tuning_history_   # DataFrame: trial, score, params
ml.evaluate(tuned, s.valid)   # works like any model
ml.explain(tuned)             # feature importance of best model
```

---

## Ensemble Stacking

```python
stacked = ml.stack(s.train, "churn", seed=42)   # default: xgboost + random_forest
stacked = ml.stack(s.train, "churn", seed=42,
    models=["xgboost", "random_forest", "logistic"],
    meta="logistic")

ml.evaluate(stacked, s.valid)
ml.explain(stacked)   # shows which base model is trusted most
```

---

## Production Monitoring

After deployment, detect when your model needs retraining:

```python
# Label-free: compare input distributions (run daily / continuously)
result = ml.drift(reference=s.train, new=live_data)
result.shifted           # False
result.features_shifted  # ["monthly_charges"]
result.severity          # "low"

# Label-required: check performance on labeled batches (run periodically)
result = ml.shelf(model, new=labeled_batch, target="churn")
result.fresh          # True
result.degradation    # {"accuracy": -0.02, "roc_auc": -0.01}
result.recommendation # "Model stable. Minor degradation within tolerance."

# Pattern: drift (early warning) → shelf (confirmation) → retrain
```

---

## Guarantee Gate

Ship with a hard contract on model quality:

```python
# Absolute thresholds
gate = ml.validate(model, test=s.test, rules={"accuracy": ">0.85", "roc_auc": ">=0.90"})
gate.passed    # True / False
gate.failures  # ["Rule accuracy > 0.85 FAILED: actual=0.82"]

# Regression against a production baseline
gate = ml.validate(new_model, test=s.test, baseline=prod_model, tolerance=0.02)
gate.improvements  # ["roc_auc: 0.88 → 0.91 (+0.03)"]
gate.degradations  # [] — empty means no regression

# Combine both modes
gate = ml.validate(new_model, test=s.test,
                   rules={"accuracy": ">0.80"},
                   baseline=prod_model)
```

---

## Cross-Validation

```python
s     = ml.split(data, "churn", folds=5, seed=42)   # CVResult
model = ml.fit(s, "churn", seed=42)                  # per-fold normalization, refit on all
model.scores_   # {"accuracy_mean": 0.85, "accuracy_std": 0.02, ...}
```

---

## Predicted Probabilities

```python
probs = model.predict_proba(s.valid)   # DataFrame with class columns
probs.columns.tolist()                  # ["no", "yes"]
#         no       yes
# 0    0.823     0.177
# 1    0.145     0.855
```

Classification only. Raises `ModelError` for regression.

---

## Why Three-Way Split?

> "In a data-rich situation, the best approach is to randomly divide the dataset into three parts."
> — Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, Ch. 7

| Partition | Purpose | Size |
|-----------|---------|------|
| `.train` | Fit here | 60% |
| `.valid` | Evaluate here, iterate freely | 20% |
| `.test` | Assess here, once | 20% |
| `.dev` | `.train` + `.valid` combined, for final refit | 80% |

Two-partition splits (train/test only) conflate model selection with model assessment. [tidymodels deprecated it in 2023.](https://rsample.tidymodels.org/reference/initial_validation_split.html)

---

## NaN Handling

- **Tree models** (xgboost, lightgbm, catboost, random_forest, decision_tree): NaN passed through natively — no imputation needed.
- **Other models** (svm, knn, logistic, linear, naive_bayes, elastic_net): NaN auto-imputed with column medians before fitting.
- **Target column**: Rows with NaN target are dropped automatically by `split()` with a warning.

All algorithms handle missing values out of the box.

---

## Design Decisions

- **`evaluate()` returns `dict[str, float]`** — always. Not a DataFrame, not mixed types.
- **`assess(test=)` is keyword-only** — the friction is intentional. You should pause before calling it.
- **Seeds are required** — reproducibility is explicit. Always pass `seed=42` (or any int).
- **Auto-stratification** — classification targets are stratified across all partitions automatically.
- **Auto dtype handling** — string targets, categorical features, scaling for SVM/KNN — all automatic.
- **No pickle** — save/load uses [skops](https://skops.readthedocs.io/) for safe serialization.

---

## Errors That Help

```python
ml.split(data, "nonexistent")
# DataError: target='nonexistent' not found in data.
#   Available columns: ['age', 'income', 'churn']

ml.fit(s.train, "price", algorithm="magic")
# ConfigError: algorithm='magic' not available.
#   Choose from: ['xgboost', 'random_forest', 'lightgbm', 'svm', 'knn',
#     'logistic', 'linear', 'catboost', 'naive_bayes', 'elastic_net']
```

---

## Built-in Datasets

**173 datasets** across synthetic, sklearn built-in, and both full OpenML benchmark suites (CC18 + CTR23). All CC-BY-4.0 or public domain.

Use `ml.datasets()` for a searchable metadata table.

```python
# ── Synthetic (deterministic, no download) ──
ml.dataset("churn")            #   7043 × 20  binary clf — real IBM Telco churn
ml.dataset("fraud")            #   9992 × 30  binary clf — real credit card fraud
ml.dataset("patients")         #    492 × 7   binary clf — for groups= parameter
ml.dataset("sales")            #    730 × 8   regression — daily revenue
ml.dataset("reviews")          #   2000 × 8   binary clf — text + numeric + boolean
ml.dataset("ecommerce")        #  12330 × 18  binary clf — real online shoppers
ml.dataset("tips")             #    244 × 7   regression — restaurant tips
ml.dataset("flights")          #    144 × 4   regression — time series, trend+season
ml.dataset("survival")         #    137 × 8   regression — VA lung cancer survival
ml.dataset("sentiment")        #   3000 × 10  binary clf — AI sentiment detection
ml.dataset("hallucination")    #   4000 × 11  binary clf — AI hallucination detection
ml.dataset("friction")         #   2500 × 11  4-class clf — decision friction levels
ml.dataset("decisions")        #   5000 × 10  regression — decision quality scoring
ml.dataset("cognitive_bias")   #   3500 × 11  6-class clf — cognitive bias types
ml.dataset("antipattern")      #   4000 × 11  6-class clf — code anti-patterns
ml.dataset("agentic")          #   3000 × 11  5-class clf — AI agent behavior modes

# ── sklearn built-in (no download) ──
ml.dataset("iris")             #    150 × 5   3-class clf
ml.dataset("wine")             #    178 × 14  3-class clf
ml.dataset("cancer")           #    569 × 31  binary clf
ml.dataset("diabetes")         #    442 × 11  regression
ml.dataset("houses")           #  20640 × 9   regression — California housing

# ── OpenML — binary classification (downloads once, cached) ──
ml.dataset("titanic")          #   1309 × 8   NaN, categoricals
ml.dataset("heart")            #    270 × 14  heart disease
ml.dataset("adult")            #  48842 × 14  income >50K
ml.dataset("bank")             #  45211 × 17  bank marketing
ml.dataset("spam")             #   4601 × 58  email spam
ml.dataset("credit")           #   1000 × 21  German credit risk
ml.dataset("phishing")         #  11055 × 31  phishing website detection
ml.dataset("electricity")      #  45312 × 9   electricity pricing
ml.dataset("mushroom")         #   8124 × 23  all categoricals, NaN
ml.dataset("eeg_eye_state")    #  14980 × 15  EEG brain-computer interface
# ... and 140+ more — see ml.datasets() for the full table

# ── OpenML — multiclass classification ──
ml.dataset("penguins")         #    344 × 7   3-class — categoricals, NaN
ml.dataset("letter")           #  20000 × 17  26-class — letter recognition
ml.dataset("har")              #  10299 × 562 6-class — human activity recognition
ml.dataset("arrhythmia")       #    452 × 280 13-class — heart arrhythmia (high-dim)

# ── OpenML — regression ──
ml.dataset("diamonds")         #  53940 × 10  diamond prices
ml.dataset("ames")             #   1460 × 81  Ames housing, NaN
ml.dataset("bike")             #  17379 × 13  hourly bike rentals
ml.dataset("concrete")         #   1030 × 9   compressive strength
ml.dataset("superconduct")     #  21263 × 82  superconductor critical temperature
```

---

## Python 3.10+

mlw requires Python 3.10+ (`str | None` union syntax).

```bash
# Check your version
python3 --version

# If < 3.10, use pyenv (macOS / Linux):
brew install pyenv
pyenv install 3.12.2
pyenv local 3.12.2

# Or conda:
conda create -n ml python=3.12
conda activate ml
pip install mlw

# Corporate / university IT often ships Python 3.9 — use a virtualenv:
python3.12 -m venv .venv && source .venv/bin/activate && pip install mlw
```

---

## Status

v1.0.0 — production stable. Core API frozen. Native Rust backend for 11 algorithm families.

## Research

Roth, S. (2026). *A Grammar of Machine Learning Workflows: Typed Primitives for Structural Leakage Prevention.* EPAGOGY. [doi:10.5281/zenodo.18905073](https://doi.org/10.5281/zenodo.18905073)

## License

MIT — [Simon Roth](https://epagogy.ai), 2026.
