"""Rust-backed model wrappers for ml.

ml provides CART, Random Forest, Ridge, Logistic Regression, and KNN
implemented in Rust with rayon parallelism and zero-copy shared data.
These wrappers provide fit/predict/predict_proba and serialization
for the raw ml_py C extension types.

Usage:
    from ml._rust import HAS_RUST
    if HAS_RUST:
        from ml._rust import _RustRandomForestClassifier
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._types import ConfigError

__all__ = ["HAS_RUST", "HAS_RUST_GBT", "HAS_RUST_NB", "HAS_RUST_EN", "HAS_RUST_ADA", "HAS_RUST_SVM"]

# Lazy availability flag — import once, cache result.
# Validates ALL core classes exist (guards against old ml_py versions).
try:
    import ml_py as _ax

    HAS_RUST = all(
        hasattr(_ax, name)
        for name in ("DecisionTree", "RandomForest", "ExtraTrees", "Linear", "Logistic", "KNN")
    )
except ImportError:
    HAS_RUST = False

# Separate sentinel for GradientBoosting — do NOT merge into HAS_RUST.
# Adding to HAS_RUST would break all existing Rust backends in envs
# that have an old ml_py build without GradientBoosting.
# Short-circuit: hasattr never evaluated if HAS_RUST is False (so _ax may be unbound).
HAS_RUST_GBT: bool = HAS_RUST and hasattr(_ax, "GradientBoosting")

# Separate sentinel for NaiveBayes — do NOT merge into HAS_RUST.
HAS_RUST_NB: bool = HAS_RUST and hasattr(_ax, "NaiveBayes")

# Separate sentinel for ElasticNet — do NOT merge into HAS_RUST.
HAS_RUST_EN: bool = HAS_RUST and hasattr(_ax, "ElasticNet")

# Separate sentinel for AdaBoost — do NOT merge into HAS_RUST.
HAS_RUST_ADA: bool = HAS_RUST and hasattr(_ax, "AdaBoost")

# Separate sentinel for SVM — do NOT merge into HAS_RUST.
HAS_RUST_SVM: bool = HAS_RUST and hasattr(_ax, "SvmClassifier") and hasattr(_ax, "SvmRegressor")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_contiguous_f64(X) -> np.ndarray:
    """Ensure array is C-contiguous float64 (required by Rust PyO3 bindings)."""
    if hasattr(X, "to_numpy"):
        # Pandas DataFrame/Series: use to_numpy so nullable dtypes (Int64, Float64,
        # boolean) are handled correctly — pd.NA becomes np.nan instead of crashing.
        return np.ascontiguousarray(X.to_numpy(dtype=np.float64, na_value=np.nan))
    return np.ascontiguousarray(X, dtype=np.float64)


def _compute_balanced_weights(y: np.ndarray) -> np.ndarray:
    """Compute balanced sample weights from class frequencies.

    Each sample's weight = n_samples / (n_classes * n_samples_for_class).
    Equivalent to sklearn's compute_sample_weight("balanced", y).
    """
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    n_classes = len(classes)
    weight_map = {c: n / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[label] for label in y], dtype=np.float64)


# ---------------------------------------------------------------------------
# Linear (Ridge Regression)
# ---------------------------------------------------------------------------


class _RustLinear:
    """Ridge regression backed by Rust solver."""

    def __init__(self, alpha: float = 1.0, **_ignored: Any) -> None:
        self.alpha = float(alpha)
        self._model: Any = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _RustLinear:
        X = _ensure_contiguous_f64(X)
        y = _ensure_contiguous_f64(y).ravel()
        sw = _ensure_contiguous_f64(sample_weight) if sample_weight is not None else None
        self._model = _ax.Linear(alpha=self.alpha)
        self._model.fit(X, y, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        return self._model.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        return self._model.coef

    @property
    def intercept_(self) -> float:
        return self._model.intercept

    def __getstate__(self) -> dict:
        return {
            "params": {"alpha": self.alpha},
            "json": self._model._to_json() if self._model else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        if state["json"] is not None:
            self._model = _ax.Linear._from_json(state["json"])


# ---------------------------------------------------------------------------
# Logistic Regression (OvR + L-BFGS)
# ---------------------------------------------------------------------------


class _RustLogistic:
    """Logistic Regression backed by Rust L-BFGS."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 100,
        n_jobs: int = 1,
        class_weight: str | None = None,
        multi_class: str = "ovr",
        **_ignored: Any,
    ) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.n_jobs = n_jobs  # accepted, ignored (Rust uses rayon)
        self.class_weight = class_weight
        self.multi_class = multi_class
        self.classes_: np.ndarray = np.array([])
        self._model: Any = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _RustLogistic:
        X = _ensure_contiguous_f64(X)
        # Encode labels to 0-based integers
        self.classes_ = np.unique(y)
        y_encoded = np.searchsorted(self.classes_, y).astype(np.int64)
        y_encoded = np.ascontiguousarray(y_encoded)
        sw = _ensure_contiguous_f64(sample_weight) if sample_weight is not None else None
        # Apply class_weight="balanced" via sample weights
        if self.class_weight == "balanced":
            balanced_sw = _compute_balanced_weights(y_encoded)
            sw = balanced_sw if sw is None else sw * balanced_sw
            sw = _ensure_contiguous_f64(sw)
        self._model = _ax.Logistic(c=self.C, max_iter=self.max_iter, multi_class=self.multi_class)
        self._model.fit(X, y_encoded, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        raw = self._model.predict(X)
        return self.classes_[raw]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    @property
    def coef_(self) -> np.ndarray:
        """Coefficient matrix. Binary: (1, p). Multiclass: (k, p)."""
        coefs = self._model.coefs  # list of 1-D arrays [bias, w1, ..., wp]
        if len(self.classes_) == 2:
            # Binary: 1 classifier, strip bias -> (1, p)
            w = np.array(coefs[0])
            return w[1:].reshape(1, -1)
        else:
            # Multiclass: k classifiers -> (k, p)
            return np.vstack([np.array(c)[1:] for c in coefs])

    @property
    def intercept_(self) -> np.ndarray:
        """Intercept vector. Binary: (1,). Multiclass: (k,)."""
        coefs = self._model.coefs
        if len(self.classes_) == 2:
            return np.array([coefs[0][0]])
        else:
            return np.array([c[0] for c in coefs])

    def __getstate__(self) -> dict:
        return {
            "params": {
                "C": self.C,
                "max_iter": self.max_iter,
                "n_jobs": self.n_jobs,
                "class_weight": self.class_weight,
                "multi_class": self.multi_class,
            },
            "classes": self.classes_,
            "json": self._model._to_json() if self._model else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        self.classes_ = state["classes"]
        if state["json"] is not None:
            self._model = _ax.Logistic._from_json(state["json"])


# ---------------------------------------------------------------------------
# Decision Tree (Classification)
# ---------------------------------------------------------------------------


class _RustDecisionTreeClassifier:
    """CART classifier backed by Rust."""

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | str | None = None,
        random_state: int | None = None,
        criterion: str = "gini",
        **_ignored: Any,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state
        self.criterion = criterion
        self.classes_: np.ndarray = np.array([])
        self._model: Any = None

    def _resolve_max_features(self, n_features: int) -> int | None:
        mf = self.max_features
        if mf is None:
            return None
        if isinstance(mf, int):
            return mf
        if mf == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(math.log2(n_features)))
        return None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _RustDecisionTreeClassifier:
        X = _ensure_contiguous_f64(X)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            # Degenerate: single class — predict it always (matches sklearn)
            self._model = None
            return self
        y_encoded = np.searchsorted(self.classes_, y).astype(np.int64)
        y_encoded = np.ascontiguousarray(y_encoded)
        sw = _ensure_contiguous_f64(sample_weight) if sample_weight is not None else None
        depth = self.max_depth if self.max_depth is not None else 500
        mf = self._resolve_max_features(X.shape[1])
        seed = self.random_state if self.random_state is not None else 0
        self._model = _ax.DecisionTree(
            max_depth=depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=mf,
            seed=seed,
            criterion=self.criterion,
        )
        self._model.fit_clf(X, y_encoded, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        if self._model is None:
            return np.full(len(np.asarray(X)), self.classes_[0])
        raw = self._model.predict_clf(X)
        return self.classes_[raw]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        if self._model is None:
            n = len(np.asarray(X))
            proba = np.zeros((n, len(self.classes_)))
            proba[:, 0] = 1.0
            return proba
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.array(self._model.feature_importances)

    def __getstate__(self) -> dict:
        return {
            "params": {
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
                "random_state": self.random_state,
                "criterion": self.criterion,
            },
            "classes": self.classes_,
            "json": self._model._to_json() if self._model else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        self.classes_ = state["classes"]
        if state["json"] is not None:
            self._model = _ax.DecisionTree._from_json(state["json"])


# ---------------------------------------------------------------------------
# Decision Tree (Regression)
# ---------------------------------------------------------------------------


class _RustDecisionTreeRegressor:
    """CART regressor backed by Rust."""

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | str | None = None,
        random_state: int | None = None,
        criterion: str = "mse",
        monotone_cst: list[int] | None = None,
        **_ignored: Any,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state
        self.criterion = criterion
        self.monotone_cst = list(monotone_cst) if monotone_cst is not None else None
        self._model: Any = None

    def _resolve_max_features(self, n_features: int) -> int | None:
        mf = self.max_features
        if mf is None:
            return None
        if isinstance(mf, int):
            return mf
        if mf == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(math.log2(n_features)))
        return None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _RustDecisionTreeRegressor:
        X = _ensure_contiguous_f64(X)
        y = _ensure_contiguous_f64(y).ravel()
        sw = _ensure_contiguous_f64(sample_weight) if sample_weight is not None else None
        if self.monotone_cst is not None and len(self.monotone_cst) != X.shape[1]:
            raise ConfigError(
                f"monotone_cst has {len(self.monotone_cst)} entries but data has "
                f"{X.shape[1]} features"
            )
        depth = self.max_depth if self.max_depth is not None else 500
        mf = self._resolve_max_features(X.shape[1])
        seed = self.random_state if self.random_state is not None else 0
        cst = [int(v) for v in self.monotone_cst] if self.monotone_cst is not None else None
        self._model = _ax.DecisionTree(
            max_depth=depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=mf,
            seed=seed,
            criterion=self.criterion,
            monotone_cst=cst,
        )
        self._model.fit_reg(X, y, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        return np.asarray(self._model.predict_reg(X), dtype=np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.array(self._model.feature_importances)

    def __getstate__(self) -> dict:
        return {
            "params": {
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
                "random_state": self.random_state,
                "criterion": self.criterion,
                "monotone_cst": self.monotone_cst,
            },
            "json": self._model._to_json() if self._model else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        if state["json"] is not None:
            self._model = _ax.DecisionTree._from_json(state["json"])


# ---------------------------------------------------------------------------
# Random Forest (Classification)
# ---------------------------------------------------------------------------


class _RustRandomForestClassifier:
    """Random Forest classifier backed by Rust + rayon."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | str | None = "sqrt",
        random_state: int | None = None,
        n_jobs: int = 1,
        bootstrap: bool = True,
        oob_score: bool = True,
        criterion: str = "gini",
        class_weight: str | None = None,
        **_ignored: Any,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs  # accepted, ignored (Rust uses rayon)
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.criterion = criterion
        self.class_weight = class_weight
        self.classes_: np.ndarray = np.array([])
        self._model: Any = None

    def _resolve_max_features(self, n_features: int) -> int | None:
        mf = self.max_features
        if mf is None:
            return None  # Rust defaults to sqrt(p) for clf
        if isinstance(mf, int):
            return mf
        if mf == "sqrt":
            return None  # Rust default for clf IS sqrt(p)
        if mf == "log2":
            return max(1, int(math.log2(n_features)))
        return None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _RustRandomForestClassifier:
        X = _ensure_contiguous_f64(X)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            # Degenerate: single class — predict it always (matches sklearn)
            self._model = None
            return self
        y_encoded = np.searchsorted(self.classes_, y).astype(np.int64)
        y_encoded = np.ascontiguousarray(y_encoded)
        sw = _ensure_contiguous_f64(sample_weight) if sample_weight is not None else None
        # Apply class_weight="balanced" via sample weights
        if self.class_weight == "balanced":
            balanced_sw = _compute_balanced_weights(y_encoded)
            sw = balanced_sw if sw is None else sw * balanced_sw
            sw = _ensure_contiguous_f64(sw)
        depth = self.max_depth if self.max_depth is not None else 500
        mf = self._resolve_max_features(X.shape[1])
        seed = self.random_state if self.random_state is not None else 42
        self._model = _ax.RandomForest(
            n_trees=self.n_estimators,
            max_depth=depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=mf,
            seed=seed,
            compute_oob=self.oob_score,
            criterion=self.criterion,
        )
        self._model.fit_clf(X, y_encoded, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        if self._model is None:
            return np.full(len(np.asarray(X)), self.classes_[0])
        raw = self._model.predict_clf(X)
        return self.classes_[raw]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        if self._model is None:
            n = len(np.asarray(X))
            proba = np.zeros((n, len(self.classes_)))
            proba[:, 0] = 1.0
            return proba
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.array(self._model.feature_importances)

    @property
    def oob_score_(self) -> float | None:
        if self._model is None:
            return None
        return self._model.oob_score

    def __getstate__(self) -> dict:
        return {
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "bootstrap": self.bootstrap,
                "oob_score": self.oob_score,
                "criterion": self.criterion,
                "class_weight": self.class_weight,
            },
            "classes": self.classes_,
            "json": self._model._to_json() if self._model else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        self.classes_ = state["classes"]
        if state["json"] is not None:
            self._model = _ax.RandomForest._from_json(state["json"])


# ---------------------------------------------------------------------------
# Random Forest (Regression)
# ---------------------------------------------------------------------------


class _RustRandomForestRegressor:
    """Random Forest regressor backed by Rust + rayon."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | str | None = None,
        random_state: int | None = None,
        n_jobs: int = 1,
        bootstrap: bool = True,
        oob_score: bool = True,
        criterion: str = "mse",
        class_weight: str | None = None,
        monotone_cst: list[int] | None = None,
        **_ignored: Any,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.criterion = criterion
        self.class_weight = class_weight
        self.monotone_cst = list(monotone_cst) if monotone_cst is not None else None
        self._model: Any = None

    def _resolve_max_features(self, n_features: int) -> int | None:
        mf = self.max_features
        if mf is None:
            return None  # Rust defaults to p/3 for reg
        if isinstance(mf, int):
            return mf
        if mf == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(math.log2(n_features)))
        return None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _RustRandomForestRegressor:
        X = _ensure_contiguous_f64(X)
        y = _ensure_contiguous_f64(y).ravel()
        sw = _ensure_contiguous_f64(sample_weight) if sample_weight is not None else None
        if self.monotone_cst is not None and len(self.monotone_cst) != X.shape[1]:
            raise ConfigError(
                f"monotone_cst has {len(self.monotone_cst)} entries but data has "
                f"{X.shape[1]} features"
            )
        depth = self.max_depth if self.max_depth is not None else 500
        mf = self._resolve_max_features(X.shape[1])
        seed = self.random_state if self.random_state is not None else 42
        cst = [int(v) for v in self.monotone_cst] if self.monotone_cst is not None else None
        self._model = _ax.RandomForest(
            n_trees=self.n_estimators,
            max_depth=depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=mf,
            seed=seed,
            compute_oob=self.oob_score,
            criterion=self.criterion,
            monotone_cst=cst,
        )
        self._model.fit_reg(X, y, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        return np.asarray(self._model.predict_reg(X), dtype=np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.array(self._model.feature_importances)

    @property
    def oob_score_(self) -> float | None:
        if self._model is None:
            return None
        return self._model.oob_score

    def __getstate__(self) -> dict:
        return {
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "bootstrap": self.bootstrap,
                "oob_score": self.oob_score,
                "criterion": self.criterion,
                "class_weight": self.class_weight,
                "monotone_cst": self.monotone_cst,
            },
            "json": self._model._to_json() if self._model else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        if state["json"] is not None:
            self._model = _ax.RandomForest._from_json(state["json"])


# ---------------------------------------------------------------------------
# Extra Trees (Classification)
# ---------------------------------------------------------------------------


class _RustExtraTreesClassifier:
    """Extra Trees classifier backed by Rust (Geurts et al. 2006)."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | str | None = "sqrt",
        random_state: int | None = None,
        n_jobs: int = 1,
        oob_score: bool = False,
        criterion: str = "gini",
        class_weight: str | None = None,
        **_ignored: Any,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        self.criterion = criterion
        self.class_weight = class_weight
        self.classes_: np.ndarray = np.array([])
        self._model: Any = None

    def _resolve_max_features(self, n_features: int) -> int | None:
        mf = self.max_features
        if mf is None:
            return None
        if isinstance(mf, int):
            return mf
        if mf == "sqrt":
            return None  # Rust default for clf IS sqrt(p)
        if mf == "log2":
            return max(1, int(math.log2(n_features)))
        return None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _RustExtraTreesClassifier:
        X = _ensure_contiguous_f64(X)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            self._model = None
            return self
        y_encoded = np.searchsorted(self.classes_, y).astype(np.int64)
        y_encoded = np.ascontiguousarray(y_encoded)
        sw = _ensure_contiguous_f64(sample_weight) if sample_weight is not None else None
        if self.class_weight == "balanced":
            balanced_sw = _compute_balanced_weights(y_encoded)
            sw = balanced_sw if sw is None else sw * balanced_sw
            sw = _ensure_contiguous_f64(sw)
        depth = self.max_depth if self.max_depth is not None else 500
        mf = self._resolve_max_features(X.shape[1])
        seed = self.random_state if self.random_state is not None else 42
        self._model = _ax.ExtraTrees(
            n_trees=self.n_estimators,
            max_depth=depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=mf,
            seed=seed,
            compute_oob=self.oob_score,
            criterion=self.criterion,
        )
        self._model.fit_clf(X, y_encoded, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        if self._model is None:
            return np.full(len(np.asarray(X)), self.classes_[0])
        raw = self._model.predict_clf(X)
        return self.classes_[raw]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        if self._model is None:
            n = len(np.asarray(X))
            proba = np.zeros((n, len(self.classes_)))
            proba[:, 0] = 1.0
            return proba
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.array(self._model.feature_importances)

    def __getstate__(self) -> dict:
        return {
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "oob_score": self.oob_score,
                "criterion": self.criterion,
                "class_weight": self.class_weight,
            },
            "classes": self.classes_,
            "json": self._model._to_json() if self._model else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        self.classes_ = state["classes"]
        if state["json"] is not None:
            self._model = _ax.ExtraTrees._from_json(state["json"])


# ---------------------------------------------------------------------------
# Extra Trees (Regression)
# ---------------------------------------------------------------------------


class _RustExtraTreesRegressor:
    """Extra Trees regressor backed by Rust (Geurts et al. 2006)."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | str | None = None,
        random_state: int | None = None,
        n_jobs: int = 1,
        oob_score: bool = False,
        criterion: str = "mse",
        class_weight: str | None = None,
        monotone_cst: list[int] | None = None,
        **_ignored: Any,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        self.criterion = criterion
        self.class_weight = class_weight
        self.monotone_cst = list(monotone_cst) if monotone_cst is not None else None
        self._model: Any = None

    def _resolve_max_features(self, n_features: int) -> int | None:
        mf = self.max_features
        if mf is None:
            return None  # Rust defaults to p/3 for reg
        if isinstance(mf, int):
            return mf
        if mf == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(math.log2(n_features)))
        return None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _RustExtraTreesRegressor:
        X = _ensure_contiguous_f64(X)
        y = _ensure_contiguous_f64(y).ravel()
        sw = _ensure_contiguous_f64(sample_weight) if sample_weight is not None else None
        if self.monotone_cst is not None and len(self.monotone_cst) != X.shape[1]:
            raise ConfigError(
                f"monotone_cst has {len(self.monotone_cst)} entries but data has "
                f"{X.shape[1]} features"
            )
        depth = self.max_depth if self.max_depth is not None else 500
        mf = self._resolve_max_features(X.shape[1])
        seed = self.random_state if self.random_state is not None else 42
        cst = [int(v) for v in self.monotone_cst] if self.monotone_cst is not None else None
        self._model = _ax.ExtraTrees(
            n_trees=self.n_estimators,
            max_depth=depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=mf,
            seed=seed,
            compute_oob=self.oob_score,
            criterion=self.criterion,
            monotone_cst=cst,
        )
        self._model.fit_reg(X, y, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        return np.asarray(self._model.predict_reg(X), dtype=np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.array(self._model.feature_importances)

    def __getstate__(self) -> dict:
        return {
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "oob_score": self.oob_score,
                "criterion": self.criterion,
                "class_weight": self.class_weight,
                "monotone_cst": self.monotone_cst,
            },
            "json": self._model._to_json() if self._model else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        if state["json"] is not None:
            self._model = _ax.ExtraTrees._from_json(state["json"])


# ---------------------------------------------------------------------------
# KNN (KD-tree classifier)
# ---------------------------------------------------------------------------


class _RustKNNClassifier:
    """KNN classifier backed by Rust KD-tree."""

    def __init__(self, *, n_neighbors: int = 5, n_jobs: int = 1):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs  # API compat, parallelism handled by rayon
        self._model = None
        self.classes_: np.ndarray | None = None

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> _RustKNNClassifier:
        X = _ensure_contiguous_f64(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_int = np.ascontiguousarray(np.searchsorted(self.classes_, y), dtype=np.int64)
        self._model = _ax.KNN(k=self.n_neighbors)
        self._model.fit_clf(X, y_int)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        pred_int = np.asarray(self._model.predict_clf(X))
        return self.classes_[pred_int]

    def predict_proba(self, X: Any) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    def __getstate__(self) -> dict:
        return {
            "params": {"n_neighbors": self.n_neighbors, "n_jobs": self.n_jobs},
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
            "json": self._model._to_json() if self._model else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        if state["classes"] is not None:
            self.classes_ = np.array(state["classes"])
        if state["json"] is not None:
            self._model = _ax.KNN._from_json(state["json"])


class _RustKNNRegressor:
    """KNN regressor backed by Rust KD-tree."""

    def __init__(self, *, n_neighbors: int = 5, n_jobs: int = 1):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self._model = None

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> _RustKNNRegressor:
        X = _ensure_contiguous_f64(X)
        y = _ensure_contiguous_f64(y).ravel()
        self._model = _ax.KNN(k=self.n_neighbors)
        self._model.fit_reg(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        return np.asarray(self._model.predict_reg(X), dtype=np.float64)

    def __getstate__(self) -> dict:
        return {
            "params": {"n_neighbors": self.n_neighbors, "n_jobs": self.n_jobs},
            "json": self._model._to_json() if self._model else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        if state["json"] is not None:
            self._model = _ax.KNN._from_json(state["json"])


# ---------------------------------------------------------------------------
# Gradient-Boosted Trees (Classification)
# ---------------------------------------------------------------------------


class _RustGBTClassifier:
    """Gradient-Boosted Trees classifier backed by Rust."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int | None = None,
        reg_lambda: float = 0.0,
        gamma: float = 0.0,
        colsample_bytree: float = 1.0,
        min_child_weight: float = 1.0,
        n_iter_no_change: int | None = None,
        validation_fraction: float = 0.1,
        reg_alpha: float = 0.0,
        max_delta_step: float = 0.0,
        base_score: float | None = None,
        monotone_cst: list[int] | None = None,
        max_bin: int = 254,
        grow_policy: str = "depthwise",
        max_leaves: int = 0,
        goss_top_rate: float = 1.0,
        goss_other_rate: float = 1.0,
        goss_min_n: int = 50_000,
        **_ignored: Any,
    ) -> None:
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be >= 1")
        if not (0.0 < float(subsample) <= 1.0):
            raise ValueError("subsample must be in (0, 1]")
        if grow_policy not in ("depthwise", "lossguide"):
            raise ValueError(f"grow_policy must be 'depthwise' or 'lossguide', got {grow_policy!r}")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.random_state = random_state
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.colsample_bytree = float(colsample_bytree)
        self.min_child_weight = float(min_child_weight)
        self.n_iter_no_change = int(n_iter_no_change) if n_iter_no_change is not None else None
        self.validation_fraction = float(validation_fraction)
        self.reg_alpha = float(reg_alpha)
        self.max_delta_step = float(max_delta_step)
        self.base_score = float(base_score) if base_score is not None else None
        self.monotone_cst = list(monotone_cst) if monotone_cst is not None else None
        self.max_bin = max(1, min(254, int(max_bin)))
        self.grow_policy = grow_policy
        self.max_leaves = int(max_leaves)
        self.goss_top_rate = float(goss_top_rate)
        self.goss_other_rate = float(goss_other_rate)
        self.goss_min_n = int(goss_min_n)
        self.classes_: np.ndarray = np.array([])
        self._model: Any = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _RustGBTClassifier:
        X = _ensure_contiguous_f64(X)
        if X.shape[0] == 0:
            raise ValueError("GBT requires at least 1 training sample")
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            self._model = None
            return self
        y_enc = np.searchsorted(self.classes_, y).astype(np.int64)
        y_enc = np.ascontiguousarray(y_enc)
        sw = _ensure_contiguous_f64(sample_weight) if sample_weight is not None else None
        seed = int(self.random_state) if self.random_state is not None else 42
        cst = [int(v) for v in self.monotone_cst] if self.monotone_cst is not None else None
        self._model = _ax.GradientBoosting(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            seed=seed,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            reg_alpha=self.reg_alpha,
            max_delta_step=self.max_delta_step,
            base_score=self.base_score,
            monotone_cst=cst,
            max_bin=self.max_bin,
            grow_policy=self.grow_policy,
            max_leaves=self.max_leaves,
            goss_top_rate=self.goss_top_rate,
            goss_other_rate=self.goss_other_rate,
            goss_min_n=self.goss_min_n,
        )
        self._model.fit_clf(X, y_enc, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        if self._model is None:
            return np.full(len(X), self.classes_[0])
        return self.classes_[self._model.predict_clf(X)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        if self._model is None:
            proba = np.zeros((len(X), len(self.classes_)))
            proba[:, 0] = 1.0
            return proba
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.array(self._model.feature_importances)

    @property
    def best_n_rounds_(self) -> int | None:
        """Actual rounds used (set when early stopping is active)."""
        if self._model is None:
            return None
        return self._model.best_n_rounds

    @property
    def best_iteration(self) -> int | None:
        """0-indexed best iteration (XGBoost API compat). Returns last tree when no early stopping."""
        if self._model is None:
            return None
        nr = self._model.best_n_rounds
        if nr is not None:
            return nr - 1
        return self.n_estimators - 1

    def __getstate__(self) -> dict:
        return {
            "params": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "subsample": self.subsample,
                "random_state": self.random_state,
                "reg_lambda": self.reg_lambda,
                "gamma": self.gamma,
                "colsample_bytree": self.colsample_bytree,
                "min_child_weight": self.min_child_weight,
                "n_iter_no_change": self.n_iter_no_change,
                "validation_fraction": self.validation_fraction,
                "reg_alpha": self.reg_alpha,
                "max_delta_step": self.max_delta_step,
                "base_score": self.base_score,
                "monotone_cst": self.monotone_cst,
                "max_bin": self.max_bin,
                "grow_policy": self.grow_policy,
                "max_leaves": self.max_leaves,
                "goss_top_rate": self.goss_top_rate,
                "goss_other_rate": self.goss_other_rate,
                "goss_min_n": self.goss_min_n,
            },
            "classes": self.classes_,
            "json": self._model._to_json() if self._model is not None else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        self.classes_ = state["classes"]
        if state["json"] is not None:
            self._model = _ax.GradientBoosting._from_json(state["json"])


# ---------------------------------------------------------------------------
# Gradient-Boosted Trees (Regression)
# ---------------------------------------------------------------------------


class _RustGBTRegressor:
    """Gradient-Boosted Trees regressor backed by Rust."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int | None = None,
        reg_lambda: float = 0.0,
        gamma: float = 0.0,
        colsample_bytree: float = 1.0,
        min_child_weight: float = 1.0,
        n_iter_no_change: int | None = None,
        validation_fraction: float = 0.1,
        reg_alpha: float = 0.0,
        max_delta_step: float = 0.0,
        base_score: float | None = None,
        monotone_cst: list[int] | None = None,
        max_bin: int = 254,
        grow_policy: str = "depthwise",
        max_leaves: int = 0,
        goss_top_rate: float = 1.0,
        goss_other_rate: float = 1.0,
        goss_min_n: int = 50_000,
        **_ignored: Any,
    ) -> None:
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be >= 1")
        if not (0.0 < float(subsample) <= 1.0):
            raise ValueError("subsample must be in (0, 1]")
        if grow_policy not in ("depthwise", "lossguide"):
            raise ValueError(f"grow_policy must be 'depthwise' or 'lossguide', got {grow_policy!r}")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.random_state = random_state
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.colsample_bytree = float(colsample_bytree)
        self.min_child_weight = float(min_child_weight)
        self.n_iter_no_change = int(n_iter_no_change) if n_iter_no_change is not None else None
        self.validation_fraction = float(validation_fraction)
        self.reg_alpha = float(reg_alpha)
        self.max_delta_step = float(max_delta_step)
        self.base_score = float(base_score) if base_score is not None else None
        self.monotone_cst = list(monotone_cst) if monotone_cst is not None else None
        self.max_bin = max(1, min(254, int(max_bin)))
        self.grow_policy = grow_policy
        self.max_leaves = int(max_leaves)
        self.goss_top_rate = float(goss_top_rate)
        self.goss_other_rate = float(goss_other_rate)
        self.goss_min_n = int(goss_min_n)
        self._model: Any = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _RustGBTRegressor:
        X = _ensure_contiguous_f64(X)
        if X.shape[0] == 0:
            raise ValueError("GBT requires at least 1 training sample")
        y = np.asarray(y, dtype=np.float64)
        y = np.ascontiguousarray(y)
        sw = _ensure_contiguous_f64(sample_weight) if sample_weight is not None else None
        seed = int(self.random_state) if self.random_state is not None else 42
        cst = [int(v) for v in self.monotone_cst] if self.monotone_cst is not None else None
        self._model = _ax.GradientBoosting(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            seed=seed,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            reg_alpha=self.reg_alpha,
            max_delta_step=self.max_delta_step,
            base_score=self.base_score,
            monotone_cst=cst,
            max_bin=self.max_bin,
            grow_policy=self.grow_policy,
            max_leaves=self.max_leaves,
            goss_top_rate=self.goss_top_rate,
            goss_other_rate=self.goss_other_rate,
            goss_min_n=self.goss_min_n,
        )
        self._model.fit_reg(X, y, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        if self._model is None:
            return np.zeros(len(X))
        return np.asarray(self._model.predict_reg(X), dtype=np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.array(self._model.feature_importances)

    @property
    def best_n_rounds_(self) -> int | None:
        """Actual rounds used (set when early stopping is active)."""
        if self._model is None:
            return None
        return self._model.best_n_rounds

    @property
    def best_iteration(self) -> int | None:
        """0-indexed best iteration (XGBoost API compat). Returns last tree when no early stopping."""
        if self._model is None:
            return None
        nr = self._model.best_n_rounds
        if nr is not None:
            return nr - 1
        return self.n_estimators - 1

    def __getstate__(self) -> dict:
        return {
            "params": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "subsample": self.subsample,
                "random_state": self.random_state,
                "reg_lambda": self.reg_lambda,
                "gamma": self.gamma,
                "colsample_bytree": self.colsample_bytree,
                "min_child_weight": self.min_child_weight,
                "n_iter_no_change": self.n_iter_no_change,
                "validation_fraction": self.validation_fraction,
                "reg_alpha": self.reg_alpha,
                "max_delta_step": self.max_delta_step,
                "base_score": self.base_score,
                "monotone_cst": self.monotone_cst,
                "max_bin": self.max_bin,
                "grow_policy": self.grow_policy,
                "max_leaves": self.max_leaves,
                "goss_top_rate": self.goss_top_rate,
                "goss_other_rate": self.goss_other_rate,
                "goss_min_n": self.goss_min_n,
            },
            "json": self._model._to_json() if self._model is not None else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        if state["json"] is not None:
            self._model = _ax.GradientBoosting._from_json(state["json"])


# ---------------------------------------------------------------------------
# Naive Bayes (Gaussian)
# ---------------------------------------------------------------------------


class _RustNaiveBayes:
    """Gaussian Naive Bayes classifier backed by Rust."""

    def __init__(self, var_smoothing: float = 1e-9, **_ignored: Any) -> None:
        self.var_smoothing = float(var_smoothing)
        self.classes_: np.ndarray = np.array([])
        self._model: Any = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _RustNaiveBayes:
        X = _ensure_contiguous_f64(X)
        self.classes_ = np.unique(y)
        y_encoded = np.searchsorted(self.classes_, y).astype(np.int64)
        y_encoded = np.ascontiguousarray(y_encoded)
        sw = _ensure_contiguous_f64(sample_weight) if sample_weight is not None else None
        self._model = _ax.NaiveBayes(var_smoothing=self.var_smoothing)
        self._model.fit_clf(X, y_encoded, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        raw = self._model.predict_clf(X)
        return self.classes_[raw]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    def __getstate__(self) -> dict:
        return {
            "params": {"var_smoothing": self.var_smoothing},
            "classes": self.classes_,
            "json": self._model._to_json() if self._model is not None else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        self.classes_ = state["classes"]
        if state["json"] is not None:
            self._model = _ax.NaiveBayes._from_json(state["json"])


# ---------------------------------------------------------------------------
# Elastic Net (coordinate descent, L1 + L2)
# ---------------------------------------------------------------------------


class _RustElasticNet:
    """Elastic Net regression backed by Rust coordinate descent."""

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-4,
        **_ignored: Any,
    ) -> None:
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._model: Any = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _RustElasticNet:
        X = _ensure_contiguous_f64(X)
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        sw = _ensure_contiguous_f64(sample_weight) if sample_weight is not None else None
        self._model = _ax.ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self._model.fit(X, y, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        return np.asarray(self._model.predict(X), dtype=np.float64)

    @property
    def coef_(self) -> np.ndarray:
        return np.asarray(self._model.coef, dtype=np.float64)

    @property
    def intercept_(self) -> float:
        return float(self._model.intercept)

    @property
    def n_iter_(self) -> int:
        return self._model.n_iter

    def __getstate__(self) -> dict:
        return {
            "params": {
                "alpha": self.alpha,
                "l1_ratio": self.l1_ratio,
                "max_iter": self.max_iter,
                "tol": self.tol,
            },
            "json": self._model._to_json() if self._model is not None else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        if state["json"] is not None:
            self._model = _ax.ElasticNet._from_json(state["json"])


# ---------------------------------------------------------------------------
# AdaBoost (SAMME)
# ---------------------------------------------------------------------------


class _RustAdaBoost:
    """AdaBoost classifier (SAMME) backed by Rust."""

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: int | None = None,
        **_ignored: Any,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state
        self.classes_: np.ndarray = np.array([])
        self._model: Any = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _RustAdaBoost:
        X = _ensure_contiguous_f64(X)
        self.classes_ = np.unique(y)
        y_encoded = np.searchsorted(self.classes_, y).astype(np.int64)
        y_encoded = np.ascontiguousarray(y_encoded)
        sw = _ensure_contiguous_f64(sample_weight) if sample_weight is not None else None
        seed = int(self.random_state) if self.random_state is not None else 42
        self._model = _ax.AdaBoost(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            seed=seed,
        )
        self._model.fit_clf(X, y_encoded, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        raw = self._model.predict_clf(X)
        return self.classes_[raw]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(self._model.feature_importances, dtype=np.float64)

    def __getstate__(self) -> dict:
        return {
            "params": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "random_state": self.random_state,
            },
            "classes": self.classes_,
            "json": self._model._to_json() if self._model is not None else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        self.classes_ = state["classes"]
        if state["json"] is not None:
            self._model = _ax.AdaBoost._from_json(state["json"])


# ---------------------------------------------------------------------------
# _RustSvmClassifier
# ---------------------------------------------------------------------------


class _RustSvmClassifier:
    """sklearn-compatible wrapper for Rust linear SVM classifier."""

    def __init__(self, C: float = 1.0, tol: float = 1e-3, max_iter: int = 1000, class_weight: str | None = None, **_ignored: Any) -> None:
        self.C = float(C)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.classes_: np.ndarray = np.array([])
        self._model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> _RustSvmClassifier:
        X = _ensure_contiguous_f64(X)
        self.classes_ = np.unique(y)
        y_encoded = np.searchsorted(self.classes_, y).astype(np.int64)
        y_encoded = np.ascontiguousarray(y_encoded)
        sw = np.ascontiguousarray(sample_weight, dtype=np.float64) if sample_weight is not None else None
        if self.class_weight == "balanced":
            balanced_sw = _compute_balanced_weights(y_encoded)
            sw = balanced_sw if sw is None else sw * balanced_sw
            sw = _ensure_contiguous_f64(sw)
        self._model = _ax.SvmClassifier(c=self.C, tol=self.tol, max_iter=self.max_iter)
        self._model.fit(X, y_encoded, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        raw = self._model.predict_clf(X)
        return self.classes_[raw]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    def __getstate__(self) -> dict:
        return {
            "params": {"C": self.C, "tol": self.tol, "max_iter": self.max_iter, "class_weight": self.class_weight},
            "classes": self.classes_,
            "json": self._model._to_json() if self._model is not None else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        self.classes_ = state["classes"]
        if state["json"] is not None:
            self._model = _ax.SvmClassifier._from_json(state["json"])


# ---------------------------------------------------------------------------
# _RustSvmRegressor
# ---------------------------------------------------------------------------


class _RustSvmRegressor:
    """sklearn-compatible wrapper for Rust linear SVR."""

    def __init__(self, C: float = 1.0, epsilon: float = 0.1, tol: float = 1e-3, max_iter: int = 1000, **_ignored: Any) -> None:
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self._model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> _RustSvmRegressor:
        X = _ensure_contiguous_f64(X)
        y = np.ascontiguousarray(y, dtype=np.float64)
        sw = np.ascontiguousarray(sample_weight, dtype=np.float64) if sample_weight is not None else None
        self._model = _ax.SvmRegressor(c=self.C, epsilon=self.epsilon, tol=self.tol, max_iter=self.max_iter)
        self._model.fit(X, y, sample_weight=sw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_contiguous_f64(X)
        return np.asarray(self._model.predict_reg(X), dtype=np.float64)

    def __getstate__(self) -> dict:
        return {
            "params": {"C": self.C, "epsilon": self.epsilon, "tol": self.tol, "max_iter": self.max_iter},
            "json": self._model._to_json() if self._model is not None else None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["params"])  # type: ignore[misc]
        if state["json"] is not None:
            self._model = _ax.SvmRegressor._from_json(state["json"])
