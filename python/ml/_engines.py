"""Algorithm wrapper layer.

Ownership: _engines.py handles engine-specific API quirks.
- SVM: auto-set probability=True (needed for predict_proba)
- XGBoost: eval_metric in constructor not fit() (version 2.1+)
- Algorithm selection and validation
- A13: GPU auto-detection for XGBoost/CatBoost (Deotte C3)

Does NOT handle dtypes (that's _normalize.py).
"""

from __future__ import annotations

from typing import Any

from ._types import ConfigError


def clone(estimator):
    """Clone an estimator without fitting data. Pure-Python replacement for sklearn.base.clone."""
    if hasattr(estimator, "get_params"):
        # sklearn estimators (XGBoost, SVM, etc.)
        return type(estimator)(**estimator.get_params())
    # Rust wrappers: reconstruct from stored constructor attributes
    params = {k: v for k, v in estimator.__dict__.items()
              if not k.startswith("_") and k != "classes_"}
    return type(estimator)(**params)


# Algorithms that support parallel fitting via n_jobs parameter.
# Used by screen() and tune() to enable n_jobs=-1 (all cores) by default.
# fit() keeps n_jobs=1 (deterministic single-thread default).
PARALLEL_ALGORITHMS: frozenset = frozenset(
    {"xgboost", "random_forest", "knn", "logistic", "lightgbm", "extra_trees"}
)


def _config_n_jobs() -> int:
    """Return n_jobs from global config, defaulting to 1 for safety."""
    try:
        from ._config import _CONFIG
        val = _CONFIG.get("n_jobs", 1)
        # -1 means "all cores" from config; preserve that intent
        return int(val)
    except Exception:
        return 1


def _screen_n_jobs() -> int:
    """n_jobs for parallel contexts (screen/tune/drift).

    Returns -1 (all cores) by default for maximum speed.
    Returns user's explicit value if they called ml.config(n_jobs=N).
    """
    try:
        from ._config import _CONFIG, _EXPLICITLY_SET
        if "n_jobs" in _EXPLICITLY_SET:
            return int(_CONFIG["n_jobs"])
        return -1
    except Exception:
        return -1


_GPU_CACHE: bool | None = None


def _detect_gpu() -> bool:
    """Detect whether a CUDA-capable GPU is available for XGBoost/CatBoost.

    Returns:
        True if a CUDA device is found, False otherwise.

    Note:
        Uses torch.cuda if available (fastest check), falls back to
        subprocess nvidia-smi, then returns False if neither is present.
        Result is cached after first call (GPU availability is stable).
    """
    global _GPU_CACHE
    if _GPU_CACHE is not None:
        return _GPU_CACHE

    # Fast path: torch is already imported in many ML environments
    try:
        import torch
        _GPU_CACHE = bool(torch.cuda.is_available())
        return _GPU_CACHE
    except ImportError:
        pass

    # Fallback: check nvidia-smi (works even without torch)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=2,
        )
        _GPU_CACHE = result.returncode == 0 and bool(result.stdout.strip())
        return _GPU_CACHE
    except Exception:
        pass

    _GPU_CACHE = False
    return _GPU_CACHE


_VALID_ENGINES = ("auto", "ml", "sklearn", "native")


def create(
    algorithm: str,
    *,
    task: str,
    seed: int,
    gpu: bool | str = "auto",
    engine: str = "auto",
    **kwargs,
) -> Any:
    """Factory — returns an UNFITTED estimator.

    Args:
        algorithm: "xgboost", "random_forest", "svm", "knn", "logistic", "linear", "auto"
        task: "classification" or "regression"
        seed: Random seed for reproducibility
        gpu: GPU override. "auto" uses _detect_gpu() logic.
        engine: Backend selection. "auto" prefers Rust (ml) > native > sklearn.
            "ml" forces Rust backend. "sklearn" forces sklearn.
            "native" forces Rust or pure-numpy (no sklearn).
        **kwargs: Passed directly to estimator constructor (e.g., max_depth=5)

    Returns:
        Unfitted estimator with .fit(X, y) and .predict(X) methods

    Raises:
        ConfigError: If algorithm not recognized, engine unavailable, or kwargs invalid
    """
    if engine not in _VALID_ENGINES:
        raise ConfigError(
            f"engine='{engine}' not recognised. Choose from: {list(_VALID_ENGINES)}"
        )

    # Handle "auto" fallback: LightGBM → XGBoost → RandomForest
    # LightGBM trains 3-5x faster than XGBoost on CPU for equivalent quality.
    if algorithm == "auto":
        try:
            import lightgbm  # noqa: F401
            algorithm = "lightgbm"
        except ImportError:
            try:
                import xgboost  # noqa: F401
                algorithm = "xgboost"
            except ImportError:
                algorithm = "random_forest"

    # Validate task
    if task not in ("classification", "regression"):
        raise ConfigError(
            f"task must be 'classification' or 'regression', got '{task}'"
        )

    # Create estimator with quirk fixes
    try:
        if algorithm == "xgboost":
            return _create_xgboost(task, seed, gpu=gpu, engine=engine, **kwargs)
        elif algorithm == "random_forest":
            return _create_random_forest(task, seed, engine=engine, **kwargs)
        elif algorithm == "svm":
            return _create_svm(task, seed, engine=engine, **kwargs)
        elif algorithm == "knn":
            return _create_knn(task, seed, engine=engine, **kwargs)
        elif algorithm == "logistic":
            return _create_logistic(task, seed, engine=engine, **kwargs)
        elif algorithm == "linear":
            return _create_linear(task, seed, engine=engine, **kwargs)
        elif algorithm == "lightgbm":
            return _create_lightgbm(task, seed, engine=engine, **kwargs)
        elif algorithm == "catboost":
            return _create_catboost(task, seed, gpu=gpu, engine=engine, **kwargs)
        elif algorithm == "naive_bayes":
            return _create_naive_bayes(task, seed, engine=engine, **kwargs)
        elif algorithm == "elastic_net":
            return _create_elastic_net(task, seed, engine=engine, **kwargs)
        elif algorithm == "histgradient":
            return _create_histgradient(task, seed, engine=engine, **kwargs)
        elif algorithm == "decision_tree":
            return _create_decision_tree(task, seed, engine=engine, **kwargs)
        elif algorithm == "tabpfn":
            return _create_tabpfn(task, seed, engine=engine, **kwargs)
        elif algorithm == "adaboost":
            return _create_adaboost(task, seed, engine=engine, **kwargs)
        elif algorithm == "gradient_boosting":
            return _create_gradient_boosting(task, seed, engine=engine, **kwargs)
        elif algorithm == "extra_trees":
            return _create_extra_trees(task, seed, engine=engine, **kwargs)
        else:
            available = [
                "xgboost", "random_forest", "lightgbm", "histgradient",
                "decision_tree", "svm", "knn", "logistic", "linear",
                "catboost", "naive_bayes", "elastic_net", "tabpfn",
                "adaboost", "gradient_boosting", "extra_trees", "auto",
            ]
            import difflib
            matches = difflib.get_close_matches(algorithm, available, n=1, cutoff=0.6)
            hint = f" Did you mean '{matches[0]}'?" if matches else ""
            raise ConfigError(
                f"algorithm='{algorithm}' not available.{hint} "
                f"Choose from: {available}"
            )
    except TypeError as e:
        # Invalid kwargs for the estimator
        raise ConfigError(
            f"{algorithm} does not accept some provided parameters. "
            f"Error: {e}"
        ) from e


def available() -> list[str]:
    """Return list of available algorithm strings for current gate.

    Returns:
        List of algorithm names

    Example:
        >>> available()
        ['xgboost', 'random_forest', 'svm', 'knn', 'logistic', 'linear', 'auto']
    """
    return [
        "xgboost", "random_forest", "lightgbm", "histgradient",
        "decision_tree", "svm", "knn", "logistic", "linear",
        "catboost", "naive_bayes", "elastic_net", "tabpfn",
        "adaboost", "gradient_boosting", "extra_trees", "auto",
    ]


# ===== PRIVATE FACTORIES =====


def _create_xgboost_cpp(task: str, seed: int, gpu: bool | str = "auto", **kwargs: Any) -> Any:
    """Create C++ XGBoost estimator (sklearn wrapper)."""
    import warnings

    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as e:
        raise ConfigError(
            "xgboost not installed. Install with: pip install xgboost"
        ) from e

    kwargs.setdefault("n_jobs", _config_n_jobs())
    kwargs.setdefault("n_estimators", 500)
    kwargs.setdefault("verbosity", 0)

    if "device" not in kwargs:
        if gpu is True:
            if not _detect_gpu():
                raise ConfigError(
                    "gpu=True requested but no CUDA GPU detected. "
                    "Use gpu=False to force CPU or gpu='auto' for automatic detection."
                )
            kwargs["device"] = "cuda"
            warnings.warn(
                "GPU mode enabled for XGBoost (device='cuda'). "
                "Results may vary slightly between runs despite fixed seed.",
                UserWarning,
                stacklevel=5,
            )
        elif gpu is False:
            kwargs["device"] = "cpu"
        elif gpu == "auto" and _detect_gpu():
            kwargs["device"] = "cuda"
            warnings.warn(
                "GPU mode enabled for XGBoost (device='cuda'). "
                "Results may vary slightly between runs despite fixed seed.",
                UserWarning,
                stacklevel=5,
            )

    if task == "classification":
        return XGBClassifier(random_state=seed, **kwargs)
    else:
        return XGBRegressor(random_state=seed, **kwargs)


def _create_xgboost(task: str, seed: int, gpu: bool | str = "auto", *, engine: str = "auto", **kwargs) -> Any:
    """Create XGBoost estimator.

    Routes to Rust GBT (engine='ml') when available and no GPU is requested.
    Falls back to C++ XGBoost (engine='sklearn') otherwise.

    engine='sklearn' → always C++ XGBoost.
    engine='ml'      → always Rust; raises ConfigError if ml-py unavailable or GPU requested.
    engine='auto'    → Rust when available, fallback to C++.
    """
    # 1. engine="sklearn" → always C++ XGBoost
    if engine == "sklearn":
        return _create_xgboost_cpp(task, seed, gpu, **kwargs)

    # 2. GPU requested → always C++ (Rust has no GPU)
    gpu_requested = gpu is True or kwargs.get("device") == "cuda"
    if gpu_requested:
        if engine == "ml":
            raise ConfigError(
                "engine='ml' does not support GPU. Use engine='sklearn' for GPU XGBoost."
            )
        return _create_xgboost_cpp(task, seed, gpu, **kwargs)

    # 3. engine="auto" or engine="ml" → try Rust
    if engine in ("auto", "ml"):
        try:
            from ._rust import HAS_RUST_GBT, _RustGBTClassifier, _RustGBTRegressor

            if HAS_RUST_GBT:
                _xgb_rust_params = {
                    "n_estimators", "learning_rate", "max_depth",
                    "min_samples_split", "min_samples_leaf", "subsample",
                    "reg_lambda", "reg_alpha", "gamma", "colsample_bytree",
                    "min_child_weight", "max_delta_step", "base_score",
                    "n_iter_no_change", "validation_fraction",
                    "grow_policy", "max_leaves", "max_bin", "monotone_cst",
                }
                filtered = {k: v for k, v in kwargs.items() if k in _xgb_rust_params}
                # XGBoost-specific defaults (differ from gradient_boosting defaults)
                filtered.setdefault("n_estimators", 500)
                filtered.setdefault("reg_lambda", 1.0)
                filtered.setdefault("learning_rate", 0.3)
                filtered.setdefault("grow_policy", "lossguide")
                filtered.setdefault("max_leaves", 31)
                # Clamp max_bin for u8 storage (max 254 for data bins; 255 reserved for NaN)
                if "max_bin" in filtered:
                    filtered["max_bin"] = min(filtered["max_bin"], 254)
                # Translate monotone_constraints tuple → monotone_cst list
                if "monotone_constraints" in kwargs and "monotone_cst" not in filtered:
                    filtered["monotone_cst"] = list(kwargs["monotone_constraints"])
                Cls = _RustGBTClassifier if task == "classification" else _RustGBTRegressor
                return Cls(random_state=seed, **filtered)
        except ImportError:
            pass

        if engine == "ml":
            raise ConfigError(
                "engine='ml' requires ml-py (Rust backend). "
                "Install with: pip install ml-py"
            )

    # 4. Fallback: C++ XGBoost
    return _create_xgboost_cpp(task, seed, gpu, **kwargs)


def _create_random_forest(task: str, seed: int, *, engine: str = "auto", **kwargs) -> Any:
    """Create Random Forest estimator.

    Prefers Rust ml backend (1.5-4.5x faster via rayon) when available.
    Falls back to sklearn when: Rust backend not installed, bootstrap=False
    (Rust always bootstraps), or criterion not supported.
    Rust supports: gini, entropy (clf); mse, squared_error, poisson (reg).
    sklearn fallback for: friedman_mse, log_loss, and any other criteria.
    """
    criterion = kwargs.get("criterion")
    _rust_criteria = {"gini", "entropy", "mse", "squared_error", "poisson"}
    _unsupported_criterion = criterion is not None and criterion not in _rust_criteria
    _no_bootstrap = kwargs.get("bootstrap", True) is False

    # engine="native" means Rust or error (no pure-numpy RF)
    _use_rust = engine in ("auto", "ml", "native")
    _use_sklearn = engine in ("auto", "sklearn")

    if _use_rust and not _unsupported_criterion and not _no_bootstrap:
        try:
            from ._rust import (
                HAS_RUST,
                _RustRandomForestClassifier,
                _RustRandomForestRegressor,
            )

            if HAS_RUST:
                _rust_params = {
                    "n_estimators", "max_depth", "min_samples_split",
                    "min_samples_leaf", "max_features", "random_state",
                    "n_jobs", "bootstrap", "oob_score", "criterion",
                    "class_weight", "monotone_cst",
                }
                filtered = {k: v for k, v in kwargs.items() if k in _rust_params}
                filtered.setdefault("n_jobs", _config_n_jobs())
                if kwargs.get("bootstrap", True):
                    filtered.setdefault("oob_score", True)
                if task == "classification":
                    return _RustRandomForestClassifier(random_state=seed, **filtered)
                else:
                    return _RustRandomForestRegressor(random_state=seed, **filtered)
            elif engine in ("ml", "native"):
                raise ConfigError(
                    f"engine='{engine}' requires ml-py (Rust backend). Install: pip install ml-py"
                )
        except ImportError as exc:
            if engine in ("ml", "native"):
                raise ConfigError(
                    f"engine='{engine}' requires ml-py (Rust backend). Install: pip install ml-py"
                ) from exc

    if not _use_sklearn:
        raise ConfigError(
            f"engine='{engine}' not available for random_forest with these parameters."
        )

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    kwargs.setdefault("n_jobs", _config_n_jobs())
    if kwargs.get("bootstrap", True):
        kwargs.setdefault("oob_score", True)

    if task == "classification":
        return RandomForestClassifier(random_state=seed, **kwargs)
    else:
        return RandomForestRegressor(random_state=seed, **kwargs)


def _create_svm(task: str, seed: int, *, engine: str = "auto", **kwargs) -> Any:
    """Create SVM estimator with quirk fixes.

    Prefers Rust linear SVM (no sklearn dep). Falls back to sklearn SVC/SVR.
    Quirk: sklearn SVM requires probability=True for predict_proba.
    """
    C = float(kwargs.pop("C", 1.0))
    tol = float(kwargs.pop("tol", 1e-3))
    max_iter = int(kwargs.pop("max_iter", 1000))
    epsilon = float(kwargs.pop("epsilon", 0.1))
    class_weight = kwargs.pop("class_weight", None)

    # Rust SVM is linear-only. Fall back to sklearn for non-linear kernels.
    _kernel = kwargs.get("kernel")
    _rust_eligible = _kernel in (None, "linear")

    if engine in ("auto", "ml") and _rust_eligible:
        try:
            from ._rust import HAS_RUST_SVM, _RustSvmClassifier, _RustSvmRegressor
            if HAS_RUST_SVM:
                if task == "classification":
                    return _RustSvmClassifier(C=C, tol=tol, max_iter=max_iter, class_weight=class_weight)
                else:
                    return _RustSvmRegressor(C=C, epsilon=epsilon, tol=tol, max_iter=max_iter)
        except ImportError as exc:
            if engine == "ml":
                raise ConfigError(
                    "engine='ml' requires ml-py (Rust backend). Install: pip install ml-py"
                ) from exc

    if engine == "ml":
        raise ConfigError(
            "engine='ml' not available for svm. "
            "ml-py Rust backend not found. Use engine='auto' or 'sklearn'."
        )

    if engine == "native":
        raise ConfigError(
            "engine='native' not available for svm. "
            "No native Python SVM implementation. Use engine='auto' or 'sklearn'."
        )

    from sklearn.svm import SVC, SVR

    if task == "classification":
        kwargs["probability"] = True
        if class_weight is not None:
            kwargs["class_weight"] = class_weight
        return SVC(C=C, tol=tol, max_iter=max_iter, random_state=seed, **kwargs)
    else:
        return SVR(C=C, epsilon=epsilon, tol=tol, max_iter=max_iter, **kwargs)


def _create_knn(task: str, seed: int, *, engine: str = "auto", **kwargs) -> Any:
    """Create KNN estimator.

    Prefers Rust KD-tree (O(k log n) queries with rayon parallelism).
    Falls back to native brute-force numpy, then sklearn.
    Note: KNN has no random_state parameter (deterministic).
    """
    n_neighbors = kwargs.get("n_neighbors", 5)
    n_jobs = kwargs.get("n_jobs", _config_n_jobs())

    if engine in ("auto", "ml"):
        try:
            from ._rust import HAS_RUST, _RustKNNClassifier, _RustKNNRegressor

            if HAS_RUST:
                if task == "classification":
                    return _RustKNNClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
                else:
                    return _RustKNNRegressor(n_neighbors=n_neighbors, n_jobs=n_jobs)
            elif engine == "ml":
                raise ConfigError("engine='ml' requires ml-py (Rust backend). Install: pip install ml-py")
        except ImportError as exc:
            if engine == "ml":
                raise ConfigError("engine='ml' requires ml-py (Rust backend). Install: pip install ml-py") from exc

    if engine in ("auto", "native"):
        from ._knn import _KNNClassifier, _KNNRegressor
        if task == "classification":
            return _KNNClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
        else:
            return _KNNRegressor(n_neighbors=n_neighbors, n_jobs=n_jobs)

    # engine == "sklearn"
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    kwargs.setdefault("n_jobs", _config_n_jobs())
    if task == "classification":
        return KNeighborsClassifier(**kwargs)
    else:
        return KNeighborsRegressor(**kwargs)


def _create_logistic(task: str, seed: int, *, engine: str = "auto", **kwargs) -> Any:
    """Create Logistic Regression estimator.

    Prefers Rust backend (L-BFGS with rayon) when available.
    Falls back to native Python L-BFGS implementation.
    """
    if task != "classification":
        raise ConfigError(
            "algorithm='logistic' only supports classification. "
            "For regression, use algorithm='linear'."
        )

    if kwargs.get("penalty") == "l1":
        raise ConfigError(
            "Native logistic only supports L2 regularisation. "
            "For L1, install scikit-learn and use LogisticRegression directly."
        )

    C = kwargs.get("C", 1.0)
    max_iter = kwargs.get("max_iter", 1000)
    n_jobs = kwargs.get("n_jobs", 1)
    class_weight = kwargs.get("class_weight")
    multi_class = kwargs.get("multi_class", "ovr")

    if engine in ("auto", "ml"):
        try:
            from ._rust import HAS_RUST, _RustLogistic

            if HAS_RUST:
                return _RustLogistic(C=C, max_iter=max_iter, n_jobs=n_jobs,
                                     class_weight=class_weight, multi_class=multi_class)
            elif engine == "ml":
                raise ConfigError("engine='ml' requires ml-py (Rust backend). Install: pip install ml-py")
        except ImportError as exc:
            if engine == "ml":
                raise ConfigError("engine='ml' requires ml-py (Rust backend). Install: pip install ml-py") from exc

    if engine in ("auto", "native"):
        from ._logistic import _LogisticModel
        return _LogisticModel(C=C, max_iter=max_iter)

    if engine == "sklearn":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=C, max_iter=max_iter, random_state=seed, n_jobs=n_jobs,
            class_weight=class_weight,
        )

    from ._logistic import _LogisticModel
    return _LogisticModel(C=C, max_iter=max_iter)


def _create_linear(task: str, seed: int, *, engine: str = "auto", **kwargs) -> Any:
    """Create Ridge regression estimator.

    Prefers Rust backend (Cholesky solver) when available.
    Falls back to native Python normal equations implementation.
    """
    if task != "regression":
        raise ConfigError(
            "algorithm='linear' only supports regression. "
            "For classification, use algorithm='logistic'."
        )

    alpha = kwargs.get("alpha", 1.0)

    if engine in ("auto", "ml"):
        try:
            from ._rust import HAS_RUST, _RustLinear

            if HAS_RUST:
                return _RustLinear(alpha=alpha)
            elif engine == "ml":
                raise ConfigError("engine='ml' requires ml-py (Rust backend). Install: pip install ml-py")
        except ImportError as exc:
            if engine == "ml":
                raise ConfigError("engine='ml' requires ml-py (Rust backend). Install: pip install ml-py") from exc

    if engine in ("auto", "native"):
        from ._linear import _LinearModel
        return _LinearModel(alpha=alpha)

    if engine == "sklearn":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=alpha, random_state=seed)

    from ._linear import _LinearModel
    return _LinearModel(alpha=alpha)


def _create_lightgbm(task: str, seed: int, *, engine: str = "auto", **kwargs) -> Any:
    """Create LightGBM estimator.

    engine='ml'      → Rust GBT (leaf-wise + GOSS); raises if ml-py unavailable.
    engine='auto'    → Rust when available, fallback to C++ LightGBM.
    engine='sklearn' → always C++ LightGBM.

    Optional C++ dependency: pip install lightgbm
    """
    # Try Rust when engine="auto" (if available) or engine="ml"
    _try_rust = engine == "ml" or engine == "auto"
    if _try_rust:
        try:
            from ._rust import HAS_RUST_GBT, _RustGBTClassifier, _RustGBTRegressor

            if HAS_RUST_GBT:
                # max_depth: LightGBM uses -1 for unlimited; Rust usize can't hold -1
                raw_depth = kwargs.get("max_depth", -1)
                rust_kwargs: dict[str, Any] = dict(
                    n_estimators=kwargs.get(
                        "n_estimators", kwargs.get("num_boost_round", 500)
                    ),
                    learning_rate=kwargs.get(
                        "learning_rate", kwargs.get("eta", 0.1)
                    ),
                    max_leaves=kwargs.get(
                        "max_leaves", kwargs.get("num_leaves", 31)
                    ),
                    reg_lambda=kwargs.get(
                        "reg_lambda", kwargs.get("lambda_l2", 0.0)
                    ),
                    reg_alpha=kwargs.get(
                        "reg_alpha", kwargs.get("lambda_l1", 0.0)
                    ),
                    subsample=kwargs.get(
                        "subsample", kwargs.get("bagging_fraction", 1.0)
                    ),
                    colsample_bytree=kwargs.get(
                        "colsample_bytree", kwargs.get("feature_fraction", 1.0)
                    ),
                    goss_top_rate=kwargs.get("top_rate", 0.2),
                    goss_other_rate=kwargs.get("other_rate", 0.1),
                    goss_min_n=kwargs.get("goss_min_n", 50_000),
                    grow_policy=kwargs.get("grow_policy", "lossguide"),
                )
                if raw_depth != -1:
                    rust_kwargs["max_depth"] = raw_depth
                # min_child_weight: only override Rust default if user explicitly set it
                if "min_child_weight" in kwargs or "min_sum_hessian_in_leaf" in kwargs:
                    rust_kwargs["min_child_weight"] = kwargs.get(
                        "min_child_weight", kwargs.get("min_sum_hessian_in_leaf")
                    )
                # early stopping mapping
                if "early_stopping_rounds" in kwargs or "n_iter_no_change" in kwargs:
                    rust_kwargs["n_iter_no_change"] = kwargs.get(
                        "n_iter_no_change", kwargs.get("early_stopping_rounds")
                    )
                Cls = _RustGBTClassifier if task == "classification" else _RustGBTRegressor
                return Cls(random_state=seed, **rust_kwargs)
        except ImportError:
            pass

        if engine == "ml":
            raise ConfigError(
                "engine='ml' requires ml-py (Rust backend). "
                "Install with: pip install ml-py"
            )

    # C++ LightGBM path (engine="sklearn" or engine="auto" without Rust)
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as e:
        raise ConfigError(
            "lightgbm not installed. Install with: pip install lightgbm"
        ) from e

    # Default: n_jobs from global config (1 = deterministic single-thread), verbose=-1 to suppress output
    kwargs.setdefault("n_jobs", _config_n_jobs())
    kwargs.setdefault("verbose", -1)

    # Note: force_col_wise is set conditionally in fit.py based on feature count
    # (> 500 features: force_col_wise=True; narrow data: LightGBM auto-selects)

    # Early stopping: raise tree budget, stop when overfitting
    kwargs.setdefault("n_estimators", 500)

    if task == "classification":
        return LGBMClassifier(random_state=seed, **kwargs)
    else:
        return LGBMRegressor(random_state=seed, **kwargs)


def _create_catboost(task: str, seed: int, gpu: bool | str = "auto", *, engine: str = "auto", **kwargs) -> Any:
    """Create CatBoost estimator.

    Optional dependency: pip install catboost
    Handles both classification and regression.
    A13: Auto-enables GPU if detected (Deotte C3).
    """
    import warnings

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as e:
        raise ConfigError(
            "catboost not installed. Install with: pip install catboost"
        ) from e

    # Suppress CatBoost verbose output by default
    kwargs.setdefault("verbose", 0)

    # GPU override: True=force, False=CPU, "auto"=detect
    if "task_type" not in kwargs:
        if gpu is True:
            if not _detect_gpu():
                raise ConfigError(
                    "gpu=True requested but no CUDA GPU detected. "
                    "Use gpu=False to force CPU or gpu='auto' for automatic detection."
                )
            kwargs["task_type"] = "GPU"
            warnings.warn(
                "GPU mode enabled for CatBoost (task_type='GPU'). "
                "Results may vary slightly between runs despite fixed seed.",
                UserWarning,
                stacklevel=4,
            )
        elif gpu is False:
            pass  # CatBoost default is CPU; no need to set task_type
        elif gpu == "auto" and _detect_gpu():
            kwargs["task_type"] = "GPU"
            warnings.warn(
                "GPU mode enabled for CatBoost (task_type='GPU'). "
                "Results may vary slightly between runs despite fixed seed.",
                UserWarning,
                stacklevel=4,
            )

    if task == "classification":
        return CatBoostClassifier(random_seed=seed, **kwargs)
    else:
        return CatBoostRegressor(random_seed=seed, **kwargs)


def _create_naive_bayes(task: str, seed: int, *, engine: str = "auto", **kwargs) -> Any:
    """Create Naive Bayes estimator.

    Classification only. Prefers native GaussianNB (zero sklearn dependency).
    Falls back to sklearn when engine='sklearn'.
    """
    if task != "classification":
        raise ConfigError(
            "algorithm='naive_bayes' only supports classification. "
            "For regression, use algorithm='linear' or 'xgboost'."
        )

    var_smoothing = kwargs.get("var_smoothing", 1e-9)

    if engine in ("auto", "ml"):
        from ._rust import HAS_RUST_NB, _RustNaiveBayes
        if HAS_RUST_NB:
            return _RustNaiveBayes(var_smoothing=var_smoothing)
        if engine == "ml":
            raise ConfigError(
                "engine='ml' not available for naive_bayes in this build. "
                "Use engine='auto', 'native', or 'sklearn'."
            )
        # auto fallback → native Python
        from ._naive_bayes import _NaiveBayesModel
        return _NaiveBayesModel(var_smoothing=var_smoothing)

    if engine == "native":
        from ._naive_bayes import _NaiveBayesModel
        return _NaiveBayesModel(var_smoothing=var_smoothing)

    # engine == "sklearn"
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB(**kwargs)


def _create_elastic_net(task: str, seed: int, *, engine: str = "auto", **kwargs) -> Any:
    """Create Elastic Net estimator.

    Regression only. Prefers native coordinate descent (zero sklearn dependency).
    Falls back to sklearn when engine='sklearn'.
    """
    if task != "regression":
        raise ConfigError(
            "algorithm='elastic_net' only supports regression. "
            "For classification, use algorithm='logistic'."
        )

    alpha = kwargs.get("alpha", 1.0)
    l1_ratio = kwargs.get("l1_ratio", 0.5)
    max_iter = kwargs.get("max_iter", 1000)
    tol = kwargs.get("tol", 1e-4)

    if engine in ("auto", "ml"):
        from ._rust import HAS_RUST_EN, _RustElasticNet
        if HAS_RUST_EN:
            return _RustElasticNet(
                alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol,
            )
        if engine == "ml":
            raise ConfigError(
                "engine='ml' not available for elastic_net (Rust backend not built). "
                "Use engine='auto', 'native', or 'sklearn'."
            )
        from ._elastic_net import _ElasticNetModel
        return _ElasticNetModel(
            alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol,
        )

    if engine == "native":
        from ._elastic_net import _ElasticNetModel
        return _ElasticNetModel(
            alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol,
        )

    # engine == "sklearn"
    from sklearn.linear_model import ElasticNet
    return ElasticNet(random_state=seed, **kwargs)


def _create_histgradient(task: str, seed: int, *, engine: str = "auto", **kwargs) -> Any:
    """Create HistGradientBoosting estimator.

    Rust backend (engine='auto' or 'ml'): routes to gradient_boosting Rust (Newton leaves).
    sklearn fallback (engine='sklearn'): HistGradientBoostingClassifier/Regressor.

    Note: Rust backend does NOT handle NaN passthrough.
    If your data contains NaN, use engine='sklearn' or impute first.
    """
    n_estimators = int(kwargs.get("n_estimators", 100))
    learning_rate = float(kwargs.get("learning_rate", 0.1))
    max_depth = int(kwargs.get("max_depth", 3))
    min_samples_split = int(kwargs.get("min_samples_split", 2))
    min_samples_leaf = int(kwargs.get("min_samples_leaf", 1))
    subsample = float(kwargs.get("subsample", 1.0))

    if engine in ("auto", "ml"):
        from ._rust import HAS_RUST_GBT, _RustGBTClassifier, _RustGBTRegressor
        if HAS_RUST_GBT:
            _gbt_extra = dict(
                reg_lambda=kwargs.get("reg_lambda", kwargs.get("lambda", 0.0)),
                gamma=kwargs.get("gamma", 0.0),
                colsample_bytree=kwargs.get("colsample_bytree", 1.0),
                min_child_weight=kwargs.get("min_child_weight", 1.0),
                n_iter_no_change=kwargs.get("n_iter_no_change"),
                validation_fraction=kwargs.get("validation_fraction", 0.1),
            )
            if task == "classification":
                return _RustGBTClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    subsample=subsample,
                    random_state=seed,
                    **_gbt_extra,
                )
            else:
                return _RustGBTRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    subsample=subsample,
                    random_state=seed,
                    **_gbt_extra,
                )
        if engine == "ml":
            raise ConfigError(
                "engine='ml' not available for histgradient (Rust backend not built). "
                "Use engine='auto' or 'sklearn'."
            )

    # engine == "sklearn" or Rust unavailable
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    if task == "classification":
        return HistGradientBoostingClassifier(random_state=seed, **kwargs)
    else:
        return HistGradientBoostingRegressor(random_state=seed, **kwargs)


def _create_decision_tree(task: str, seed: int, *, engine: str = "auto", **kwargs) -> Any:
    """Create Decision Tree estimator.

    Prefers Rust backend when available. Falls back to sklearn for unsupported criteria.
    Rust supports: gini, entropy (clf); mse, squared_error, poisson (reg).
    sklearn fallback for: friedman_mse, log_loss, and any other criteria.
    """
    criterion = kwargs.get("criterion")
    _rust_criteria = {"gini", "entropy", "mse", "squared_error", "poisson"}
    _unsupported_criterion = criterion is not None and criterion not in _rust_criteria

    _use_rust = engine in ("auto", "ml", "native")
    _use_sklearn = engine in ("auto", "sklearn")

    if _use_rust and not _unsupported_criterion:
        try:
            from ._rust import (
                HAS_RUST,
                _RustDecisionTreeClassifier,
                _RustDecisionTreeRegressor,
            )

            if HAS_RUST:
                _rust_params = {
                    "max_depth", "min_samples_split", "min_samples_leaf",
                    "max_features", "random_state", "criterion", "monotone_cst",
                }
                filtered = {k: v for k, v in kwargs.items() if k in _rust_params}
                if task == "classification":
                    return _RustDecisionTreeClassifier(random_state=seed, **filtered)
                else:
                    return _RustDecisionTreeRegressor(random_state=seed, **filtered)
            elif engine in ("ml", "native"):
                raise ConfigError(
                    f"engine='{engine}' requires ml-py (Rust backend). Install: pip install ml-py"
                )
        except ImportError as exc:
            if engine in ("ml", "native"):
                raise ConfigError(
                    f"engine='{engine}' requires ml-py (Rust backend). Install: pip install ml-py"
                ) from exc

    if not _use_sklearn:
        raise ConfigError(
            f"engine='{engine}' not available for decision_tree with these parameters."
        )

    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    if task == "classification":
        return DecisionTreeClassifier(random_state=seed, **kwargs)
    else:
        return DecisionTreeRegressor(random_state=seed, **kwargs)


def _create_tabpfn(task: str, seed: int, *, engine: str = "auto", **kwargs) -> Any:
    """Create TabPFN estimator.

    TabPFN is a foundation model for tabular data. Pre-trained on millions
    of synthetic datasets. Zero-shot: no hyperparameter tuning needed.
    Competitive with tuned GBDT on datasets <10K rows, <100 features.

    Limitations:
    - Max ~10,000 training rows (v2.5 handles ~100K but slower)
    - Max ~100 features (after encoding)
    - Classification only (regression support is experimental)
    - Requires: pip install tabpfn

    When dataset exceeds limits, a UserWarning is emitted by fit.py.
    """
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as e:
        raise ConfigError(
            "tabpfn not installed. Install with: pip install tabpfn"
        ) from e

    if task != "classification":
        raise ConfigError(
            "algorithm='tabpfn' currently supports classification only. "
            "For regression, use 'xgboost', 'lightgbm', or 'catboost'."
        )

    n_ensembles = kwargs.pop("n_ensembles", 32)
    return TabPFNClassifier(
        device="cpu",
        N_ensemble_configurations=n_ensembles,
        seed=seed,
        **kwargs,
    )


def _create_adaboost(task: str, seed: int, *, engine: str = "auto", **kwargs) -> Any:
    """Create AdaBoost estimator.

    Classification only.
    Rust backend (engine='auto' or 'ml'): native SAMME implementation.
    sklearn fallback (engine='sklearn'): AdaBoostClassifier.
    """
    if task != "classification":
        raise ConfigError(
            "algorithm='adaboost' only supports classification. "
            "For regression boosting, use 'xgboost' or 'gradient_boosting'."
        )

    n_estimators = kwargs.get("n_estimators", 50)
    learning_rate = kwargs.get("learning_rate", 1.0)

    if engine in ("auto", "ml"):
        from ._rust import HAS_RUST_ADA, _RustAdaBoost
        if HAS_RUST_ADA:
            return _RustAdaBoost(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=seed,
            )
        if engine == "ml":
            raise ConfigError(
                "engine='ml' not available for adaboost (Rust backend not built). "
                "Use engine='auto' or 'sklearn'."
            )

    # engine == "sklearn" or Rust unavailable in "auto" mode
    from sklearn.ensemble import AdaBoostClassifier
    return AdaBoostClassifier(random_state=seed, **kwargs)


def _create_gradient_boosting(task: str, seed: int, *, engine: str = "auto", **kwargs) -> Any:
    """Create Gradient Boosting estimator.

    Rust backend (engine='auto' or 'ml'): native Rust histogram GBT.
    sklearn fallback (engine='sklearn'): GradientBoostingClassifier/Regressor.
    """
    _use_rust = engine != "sklearn"
    if _use_rust:
        try:
            from ._rust import HAS_RUST_GBT, _RustGBTClassifier, _RustGBTRegressor

            if HAS_RUST_GBT:
                _rust_params = {
                    "n_estimators", "learning_rate", "max_depth",
                    "min_samples_split", "min_samples_leaf", "subsample",
                    "reg_lambda", "lambda", "gamma", "colsample_bytree",
                    "min_child_weight", "n_iter_no_change", "validation_fraction",
                    "reg_alpha", "max_delta_step", "base_score", "monotone_cst", "max_bin",
                    "grow_policy", "max_leaves",
                }
                filtered = {k: v for k, v in kwargs.items() if k in _rust_params}
                # Normalize "lambda" → "reg_lambda" for Python keyword safety
                if "lambda" in filtered and "reg_lambda" not in filtered:
                    filtered["reg_lambda"] = filtered.pop("lambda")
                elif "lambda" in filtered:
                    filtered.pop("lambda")
                if task == "classification":
                    return _RustGBTClassifier(random_state=seed, **filtered)
                return _RustGBTRegressor(random_state=seed, **filtered)
            elif engine in ("ml", "native"):
                from ._types import ConfigError
                raise ConfigError(
                    "engine='ml' requested but Rust GradientBoosting not available. "
                    "Rebuild ml_py with `cargo build --release -p ml-py`."
                )
        except ImportError as exc:
            if engine in ("ml", "native"):
                from ._types import ConfigError
                raise ConfigError(f"Rust backend unavailable: {exc}") from exc

    # sklearn fallback
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )

    kwargs.setdefault("n_estimators", 100)

    if task == "classification":
        return GradientBoostingClassifier(random_state=seed, **kwargs)
    return GradientBoostingRegressor(random_state=seed, **kwargs)


def _create_extra_trees(task: str, seed: int, *, engine: str = "auto", **kwargs) -> Any:
    """Create Extra Trees estimator.

    Extremely Randomized Trees — differs from Random Forest in Search primitive:
    random split thresholds instead of greedy best-split search.
    Routes to Rust for gini/entropy/mse criteria; falls back to sklearn for others.
    """
    from ._rust import HAS_RUST, _RustExtraTreesClassifier, _RustExtraTreesRegressor

    _use_rust = HAS_RUST and engine in ("auto", "ml")
    _rust_criteria = {"gini", "entropy", "mse", "squared_error"}
    criterion = kwargs.get("criterion")
    _unsupported = criterion is not None and criterion not in _rust_criteria

    if _use_rust and not _unsupported:
        _rust_params = {
            "n_estimators", "max_depth", "min_samples_split",
            "min_samples_leaf", "max_features", "random_state",
            "n_jobs", "oob_score", "criterion", "class_weight", "monotone_cst",
        }
        filtered = {k: v for k, v in kwargs.items() if k in _rust_params}
        if task == "classification":
            return _RustExtraTreesClassifier(random_state=seed, **filtered)
        return _RustExtraTreesRegressor(random_state=seed, **filtered)

    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    kwargs.setdefault("n_jobs", _config_n_jobs())
    kwargs.setdefault("n_estimators", 100)

    if task == "classification":
        return ExtraTreesClassifier(random_state=seed, **kwargs)
    else:
        return ExtraTreesRegressor(random_state=seed, **kwargs)
