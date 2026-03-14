"""Save and load models.

Uses skops format (secure, no pickle).
File extension: .ml
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._types import Model
    from .embed import Embedder


def save(model: Model | Embedder, path: str) -> str:
    """Save model, tuning result, or embedder to file.

    Uses skops format (secure, no pickle/joblib).
    File extension: .ml

    Args:
        model: Fitted Model, TuningResult, or Embedder to save
        path: File path (e.g., "model.ml")

    Example:
        >>> ml.save(model, "churn_model.ml")
        >>> ml.save(tuned, "tuned.ml")
        >>> ml.save(embedder, "embedder.ml")
    """
    import os
    from datetime import datetime, timezone

    import skops.io as sio

    from . import __version__
    from ._types import Model, TuningResult
    from .embed import Embedder
    from .encode import Encoder
    from .impute import Imputer
    from .pipeline import Pipeline
    from .scale import Scaler
    from .tokenize import Tokenizer

    # Path validation: warn on suspicious paths (traversal or /tmp), not all absolute paths
    path = os.path.abspath(os.path.normpath(path))
    _suspicious = (
        ".." in os.path.relpath(path) and not os.path.isabs(path)
        or path.startswith(os.path.join(os.sep, "tmp") + os.sep)
        or path.startswith(os.path.join(os.sep, "var", "tmp") + os.sep)
    )
    if _suspicious:
        warnings.warn(
            f"Saving to a temporary or traversal path: {path}. "
            "Ensure this path is intentional.",
            UserWarning,
            stacklevel=2,
        )

    # Accept .pyml or legacy .ml/.mlw extension, default to .pyml
    if not (path.endswith(".pyml") or path.endswith(".ml") or path.endswith(".mlw")):
        path = f"{path}.pyml"

    # Build save dict based on object type
    from ._types import OptimizeResult
    if isinstance(model, OptimizeResult):
        # Save OptimizeResult — delegates to the underlying Model with threshold recorded
        model = model.model  # unwrap; threshold is baked in as _threshold
    if isinstance(model, TuningResult):
        # Save TuningResult (model + tuning metadata)
        m = model.best_model
        save_dict = {
            "__type__": "TuningResult",
            "_model": m._model,
            "_task": m._task,
            "_algorithm": m._algorithm,
            "_features": m._features,
            "_target": m._target,
            "_seed": m._seed,
            "_label_encoder": m._label_encoder,
            "_feature_encoder": m._feature_encoder,
            "_preprocessor": m._preprocessor,
            "_n_train": m._n_train,
            "_balance": m._balance,
            "_calibrated": m._calibrated,
            "_sample_weight_col": m._sample_weight_col,  # A12
            "scores_": m.scores_,
            "fold_scores_": m.fold_scores_,
            "_provenance": getattr(m, "_provenance", None),  # Layer 2: cross-verb lineage
            "best_params_": model.best_params_,
            "tuning_history_": model.tuning_history_.to_dict(orient="list"),
            "__version__": __version__,
        }
    elif isinstance(model, Model):
        # Save Model (exclude _assess_count — not persisted)
        # Note: _preprocessor (lambda/function) may not be serializable by skops.
        # If save fails, user must remove preprocessor or wrap it in a serializable class.

        # A8: Seed-averaged models store sub-models separately (ensemble wrapper
        # is not skops-serializable). Set _model=None, store sub-model dicts.
        from .fit import _SeedAverageEnsemble
        if model._ensemble is not None and isinstance(model._model, _SeedAverageEnsemble):
            ensemble_dicts = [
                {
                    "_model": m._model,
                    "_task": m._task,
                    "_algorithm": m._algorithm,
                    "_features": m._features,
                    "_target": m._target,
                    "_seed": m._seed,
                    "_label_encoder": m._label_encoder,
                    "_feature_encoder": m._feature_encoder,
                    "_n_train": m._n_train,
                    "_balance": m._balance,
                }
                for m in model._ensemble
            ]
            engine_to_save = None  # reconstructed at load time
        else:
            ensemble_dicts = None
            engine_to_save = model._model

        save_dict = {
            "__type__": "Model",
            "_model": engine_to_save,
            "_task": model._task,
            "_algorithm": model._algorithm,
            "_features": model._features,
            "_target": model._target,
            "_seed": model._seed,
            "_label_encoder": model._label_encoder,
            "_feature_encoder": model._feature_encoder,
            "_preprocessor": model._preprocessor,
            "_n_train": model._n_train,
            "_balance": model._balance,
            "_calibrated": model._calibrated,
            "_sample_weight_col": model._sample_weight_col,  # A12
            "_ensemble_dicts": ensemble_dicts,  # A8
            "_seed_scores": model._seed_scores,  # A8
            "_seed_std": model._seed_std,  # A8
            "_threshold": model._threshold,  # A4
            "_holdout_score": model._holdout_score,  # early-stopping eval score
            "_provenance": getattr(model, "_provenance", None),  # Layer 2: cross-verb lineage
            "scores_": model.scores_,
            "fold_scores_": model.fold_scores_,
            "__version__": __version__,
        }
    elif isinstance(model, Embedder):
        # Save Embedder — decompose into plain serializable parts
        # (skops can't serialize TfidfVectorizer or DataFrame Cython internals)
        vec = model._vectorizer
        vectorizer_parts = {
            "vocabulary_": {str(k): int(v) for k, v in vec.vocabulary_.items()},
            "idf_": vec.idf_.tolist(),
            "max_features": vec.max_features,
        }
        # Store vectors as columns dict + column names (avoid DataFrame Cython)
        vectors_data = {
            "values": model.vectors.values.tolist(),
            "columns": list(model.vectors.columns),
        }
        save_dict = {
            "__type__": "Embedder",
            "vectors_data": vectors_data,
            "method": model.method,
            "vocab_size": model.vocab_size,
            "_vectorizer_parts": vectorizer_parts,
            "__version__": __version__,
        }
    elif isinstance(model, Tokenizer):
        # Save Tokenizer — decompose each column's TfidfVectorizer into plain parts
        vectorizer_parts = {}
        for col, vec in model._vectorizers.items():
            vectorizer_parts[col] = {
                "vocabulary_": {str(k): int(v) for k, v in vec.vocabulary_.items()},
                "idf_": vec.idf_.tolist(),
                "max_features": vec.max_features,
            }
        save_dict = {
            "__type__": "Tokenizer",
            "columns": model.columns,
            "max_features": model.max_features,
            "_vectorizer_parts": vectorizer_parts,
            "__version__": __version__,
        }
    elif isinstance(model, Scaler):
        # Save Scaler — decompose each sklearn scaler into plain numeric parts
        scaler_parts = {}
        for col, scaler in model._scalers.items():
            s_name = type(scaler).__name__
            if s_name == "StandardScaler":
                scaler_parts[col] = {
                    "type": "StandardScaler",
                    "mean_": scaler.mean_.tolist(),
                    "scale_": scaler.scale_.tolist(),
                    "var_": scaler.var_.tolist(),
                }
            elif s_name == "MinMaxScaler":
                scaler_parts[col] = {
                    "type": "MinMaxScaler",
                    "min_": scaler.min_.tolist(),
                    "scale_": scaler.scale_.tolist(),
                    "data_min_": scaler.data_min_.tolist(),
                    "data_max_": scaler.data_max_.tolist(),
                    "data_range_": scaler.data_range_.tolist(),
                    "feature_range": list(scaler.feature_range),
                }
            else:  # RobustScaler
                scaler_parts[col] = {
                    "type": "RobustScaler",
                    "center_": scaler.center_.tolist(),
                    "scale_": scaler.scale_.tolist(),
                }
        save_dict = {
            "__type__": "Scaler",
            "columns": model.columns,
            "method": model.method,
            "_scaler_parts": scaler_parts,
            "__version__": __version__,
        }
    elif isinstance(model, Encoder):
        save_dict = {
            "__type__": "Encoder",
            "columns": model.columns,
            "method": model.method,
            "_categories": model._categories,
            "_te_mapping": model._te_mapping,
            "_te_global": model._te_global,
            "_te_classes": model._te_classes,
            "_fold_indices": model._fold_indices,
            "_freq_mapping": model._freq_mapping,
            "_woe_mapping": getattr(model, "_woe_mapping", {}),
            "_iv_scores": getattr(model, "_iv_scores", {}),
            "_dt_include_hour": getattr(model, "_dt_include_hour", {}),
            "__version__": __version__,
        }
    elif isinstance(model, Imputer):
        # Serialize fill_values — convert numpy scalars to plain Python
        serialized_fills = {}
        for col, val in model._fill_values.items():
            if hasattr(val, "item"):
                serialized_fills[col] = val.item()
            else:
                serialized_fills[col] = val
        save_dict = {
            "__type__": "Imputer",
            "columns": model.columns,
            "strategy": model.strategy,
            "fill_value": model.fill_value,
            "_fill_values": serialized_fills,
            "__version__": __version__,
        }
    elif isinstance(model, Pipeline):
        # Save Pipeline — serialize each step by type into plain dicts
        from ._types import ConfigError as _CE

        steps_data = []
        for step in model.steps:
            if isinstance(step, Tokenizer):
                vp = {}
                for col, vec in step._vectorizers.items():
                    vp[col] = {
                        "vocabulary_": {str(k): int(v) for k, v in vec.vocabulary_.items()},
                        "idf_": vec.idf_.tolist(),
                        "max_features": vec.max_features,
                    }
                steps_data.append({
                    "__step_type__": "Tokenizer",
                    "columns": step.columns,
                    "max_features": step.max_features,
                    "_vectorizer_parts": vp,
                })
            elif isinstance(step, Scaler):
                sp = {}
                for col, scaler in step._scalers.items():
                    s_name = type(scaler).__name__
                    if s_name == "StandardScaler":
                        sp[col] = {
                            "type": "StandardScaler",
                            "mean_": scaler.mean_.tolist(),
                            "scale_": scaler.scale_.tolist(),
                            "var_": scaler.var_.tolist(),
                        }
                    elif s_name == "MinMaxScaler":
                        sp[col] = {
                            "type": "MinMaxScaler",
                            "min_": scaler.min_.tolist(),
                            "scale_": scaler.scale_.tolist(),
                            "data_min_": scaler.data_min_.tolist(),
                            "data_max_": scaler.data_max_.tolist(),
                            "data_range_": scaler.data_range_.tolist(),
                            "feature_range": list(scaler.feature_range),
                        }
                    else:  # RobustScaler
                        sp[col] = {
                            "type": "RobustScaler",
                            "center_": scaler.center_.tolist(),
                            "scale_": scaler.scale_.tolist(),
                        }
                steps_data.append({
                    "__step_type__": "Scaler",
                    "columns": step.columns,
                    "method": step.method,
                    "_scaler_parts": sp,
                })
            elif isinstance(step, Encoder):
                steps_data.append({
                    "__step_type__": "Encoder",
                    "columns": step.columns,
                    "method": step.method,
                    "_categories": step._categories,
                    "_te_mapping": step._te_mapping,
                    "_te_global": step._te_global,
                    "_te_classes": step._te_classes,
                    "_fold_indices": step._fold_indices,
                    "_freq_mapping": step._freq_mapping,
                })
            elif isinstance(step, Imputer):
                sf = {
                    col: val.item() if hasattr(val, "item") else val
                    for col, val in step._fill_values.items()
                }
                steps_data.append({
                    "__step_type__": "Imputer",
                    "columns": step.columns,
                    "strategy": step.strategy,
                    "fill_value": step.fill_value,
                    "_fill_values": sf,
                })
            else:
                raise _CE(
                    f"Pipeline step {type(step).__name__} cannot be saved. "
                    "Only Scaler, Encoder, Imputer, Tokenizer are supported."
                )
        save_dict = {
            "__type__": "Pipeline",
            "steps": steps_data,
            "__version__": __version__,
        }
    else:
        from ._types import ConfigError

        raise ConfigError(
            f"save() requires Model, TuningResult, Embedder, Tokenizer, "
            f"Scaler, Encoder, or Imputer. Got {type(model).__name__}"
        )

    # Warn if preprocessor is a lambda/local function (load will fail)
    pp = save_dict.get("_preprocessor")
    if pp is not None and callable(pp):
        name = getattr(pp, "__qualname__", getattr(pp, "__name__", ""))
        if "<lambda>" in name or "<locals>" in name:
            warnings.warn(
                f"Saving model with lambda/local preprocessor '{name}'. "
                "load() will fail because lambdas cannot be safely deserialized. "
                "Wrap your preprocessor in a named function or class.",
                UserWarning,
                stacklevel=2,
            )

    # Add metadata
    save_dict["__timestamp__"] = datetime.now(timezone.utc).isoformat()

    # Save using skops
    sio.dump(save_dict, path)
    return path


def load(path: str) -> Model | Embedder:
    """Load model or embedder from file.

    .. warning:: **Security**
       load() deserializes model files which may contain executable code.
       Never load .ml files from untrusted sources. While skops provides
       type-filtering (safer than pickle), the whitelisted types include
       sklearn estimators which can have custom __reduce__ methods.
       Treat model files like executables.

    Args:
        path: File path (e.g., "model.ml")

    Returns:
        Loaded Model or Embedder

    Raises:
        FileNotFoundError: If file doesn't exist
        VersionError: If major version mismatch

    Example:
        >>> model = ml.load("churn_model.ml")
        >>> model.algorithm
        'xgboost'
        >>> embedder = ml.load("embedder.ml")
        >>> embedder.method
        'tfidf'
    """
    import pandas as pd
    import skops.io as sio

    from . import __version__
    from ._types import Model, TuningResult, VersionError
    from .embed import Embedder
    from .encode import Encoder
    from .impute import Imputer
    from .scale import Scaler
    from .tokenize import Tokenizer

    # Check file exists (try with .ml extension if not found)
    p = Path(path)
    if not p.exists() and not (path.endswith(".pyml") or path.endswith(".ml") or path.endswith(".mlw")):
        p = Path(f"{path}.pyml")
    if not p.exists():
        import difflib
        import os as _os
        directory = _os.path.dirname(path) or "."
        try:
            candidates = [f for f in _os.listdir(directory) if f.endswith((".pyml", ".ml", ".mlw"))]
            matches = difflib.get_close_matches(_os.path.basename(path), candidates, n=1)
            suggestion = f" Did you mean '{_os.path.join(directory, matches[0])}'?" if matches else ""
        except Exception:
            suggestion = ""
        raise FileNotFoundError(f"Model file '{path}' not found.{suggestion}")
    path = str(p)

    # Load using skops with narrowed type whitelist
    # Security: only trust specific sklearn modules actually used by mlw.
    # Broad "sklearn." prefix replaced with specific module prefixes to
    # limit attack surface through __reduce__ chains.
    _SAFE_PREFIXES = (
        # sklearn — only modules mlw actually uses
        "sklearn.ensemble.",      # RandomForest, Stacking
        "sklearn.tree.",          # DecisionTree (internal to RF)
        "sklearn.linear_model.",  # Logistic, Ridge, ElasticNet
        "sklearn.svm.",           # SVC, SVR
        "sklearn.neighbors.",     # KNN
        "sklearn.metrics.",       # KNN distance metrics (EuclideanDistance64)
        "sklearn.naive_bayes.",   # GaussianNB
        "sklearn.preprocessing.", # LabelEncoder, OneHotEncoder, StandardScaler
        "sklearn.calibration.",   # CalibratedClassifierCV (legacy models)
        "sklearn.isotonic.",      # IsotonicRegression (legacy models)
        "sklearn.frozen.",        # FrozenEstimator (legacy models)
        "sklearn.impute.",        # SimpleImputer
        "sklearn.pipeline.",      # Pipeline (stacking)
        "sklearn._loss.",         # Loss functions (HistGradientBoosting internals)
        "sklearn.feature_extraction.",  # TfidfVectorizer (legacy models)
        "sklearn.utils.",         # sklearn internals (Bunch, validation)
        "sklearn.base.",          # BaseEstimator
        # numpy, scipy, pandas — data types
        "numpy.", "scipy.", "pandas.",
        # optional ML backends
        "xgboost.", "lightgbm.", "catboost.",
        # ml internals
        "ml._rust.",      # Rust-backed wrappers
        "ml._logistic.",      # native logistic regression
        "ml._linear.",        # native linear (Ridge) regression
        "ml._naive_bayes.",   # native Gaussian Naive Bayes
        "ml._knn.",           # native K-Nearest Neighbors
        "ml._elastic_net.",   # native Elastic Net
        "ml._normalize.", "ml._types.", "ml._transforms.",
        "ml.calibrate.",  # _CalibratedModel (own calibration wrapper)
        "ml.scale.", "ml.encode.", "ml.impute.", "ml.tokenize.", "ml.stack.",
        "ml.pipeline.",  # Pipeline used as model._preprocessor
        # collections (OrderedDict in sklearn internals)
        "collections.",
    )
    # Explicit builtins whitelist (not builtins.* — that would trust eval/exec)
    _SAFE_BUILTINS = frozenset({
        "builtins.list", "builtins.dict", "builtins.tuple", "builtins.set",
        "builtins.frozenset", "builtins.bytes", "builtins.bytearray",
        "builtins.str", "builtins.int", "builtins.float", "builtins.bool",
        "builtins.complex", "builtins.slice", "builtins.range",
        "builtins.NoneType",
    })
    try:
        unknown_types = sio.get_untrusted_types(file=path)
    except Exception as exc:
        from ._types import ConfigError

        raise ConfigError(
            f"Model file '{path}' appears corrupted or is not a valid .ml archive. "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    trusted = [
        t for t in unknown_types
        if any(t.startswith(p) for p in _SAFE_PREFIXES) or t in _SAFE_BUILTINS
    ]
    untrusted = [t for t in unknown_types if t not in trusted]
    if untrusted:
        from ._types import ConfigError

        raise ConfigError(
            f"Model file contains untrusted types: {untrusted}. "
            "This may indicate a tampered file. If you trust the source, "
            "use skops.io.load() directly with trusted=<types>."
        )
    try:
        loaded = sio.load(path, trusted=trusted)
    except Exception as exc:
        from ._types import ConfigError

        raise ConfigError(
            f"Failed to load model file '{path}'. File may be corrupted. "
            f"({type(exc).__name__}: {exc})"
        ) from exc

    # Check version compatibility
    saved_version = loaded.get("__version__", "0.0.0")
    current_major = __version__.split(".")[0]
    saved_major = saved_version.split(".")[0]

    # Migration shim: accept models saved with internal 4.x versions before public 1.0
    _legacy_majors = {"4"}
    if current_major != saved_major and saved_major not in _legacy_majors:
        raise VersionError(
            f"Object was saved with ml version {saved_version}, "
            f"but current version is {__version__}. "
            f"Major version mismatch — may not load correctly."
        )

    # Determine object type and rebuild
    obj_type = loaded.get("__type__", "Model")  # default to Model for backward compat

    if obj_type == "Model":
        # Build Model (reset _assess_count to 0)
        loaded_model = Model(
            _model=loaded["_model"],
            _task=loaded["_task"],
            _algorithm=loaded["_algorithm"],
            _features=loaded["_features"],
            _target=loaded["_target"],
            _seed=loaded["_seed"],
            _label_encoder=loaded["_label_encoder"],
            _feature_encoder=loaded["_feature_encoder"],
            _preprocessor=loaded.get("_preprocessor"),
            _n_train=loaded["_n_train"],
            _balance=loaded.get("_balance", False),
            _calibrated=loaded.get("_calibrated", False),
            _sample_weight_col=loaded.get("_sample_weight_col"),  # A12
            _seed_scores=loaded.get("_seed_scores"),  # A8
            _seed_std=loaded.get("_seed_std"),  # A8
            _threshold=loaded.get("_threshold"),  # A4
            _holdout_score=loaded.get("_holdout_score"),  # early-stopping eval score
            scores_=loaded.get("scores_"),
            fold_scores_=loaded.get("fold_scores_"),
            _assess_count=0,  # reset on load
        )
        # Layer 2: Restore provenance for cross-verb checks
        loaded_model._provenance = loaded.get("_provenance") or {}
        # A8: reconstruct seed-averaged ensemble
        ensemble_dicts = loaded.get("_ensemble_dicts")
        if ensemble_dicts is not None:
            from .fit import _SeedAverageEnsemble
            sub_models = []
            for ed in ensemble_dicts:
                sub_m = Model(
                    _model=ed["_model"],
                    _task=ed["_task"],
                    _algorithm=ed["_algorithm"],
                    _features=ed["_features"],
                    _target=ed["_target"],
                    _seed=ed["_seed"],
                    _label_encoder=ed["_label_encoder"],
                    _feature_encoder=ed["_feature_encoder"],
                    _n_train=ed["_n_train"],
                    _balance=ed.get("_balance", False),
                )
                sub_models.append(sub_m)
            loaded_model._ensemble = sub_models
            loaded_model._model = _SeedAverageEnsemble(sub_models)
        return loaded_model
    elif obj_type == "TuningResult":
        # Build Model then wrap in TuningResult
        best_model = Model(
            _model=loaded["_model"],
            _task=loaded["_task"],
            _algorithm=loaded["_algorithm"],
            _features=loaded["_features"],
            _target=loaded["_target"],
            _seed=loaded["_seed"],
            _label_encoder=loaded["_label_encoder"],
            _feature_encoder=loaded["_feature_encoder"],
            _preprocessor=loaded.get("_preprocessor"),
            _n_train=loaded["_n_train"],
            _balance=loaded.get("_balance", False),
            _calibrated=loaded.get("_calibrated", False),
            _sample_weight_col=loaded.get("_sample_weight_col"),  # A12
            _holdout_score=loaded.get("_holdout_score"),  # early-stopping eval score
            scores_=loaded.get("scores_"),
            fold_scores_=loaded.get("fold_scores_"),
            _assess_count=0,
        )
        best_model._provenance = loaded.get("_provenance") or {}
        tuning_history_raw = loaded.get("tuning_history_")
        if tuning_history_raw is None:
            tuning_history = pd.DataFrame(columns=["trial", "score"])
        elif isinstance(tuning_history_raw, dict):
            tuning_history = pd.DataFrame(tuning_history_raw)
        else:
            tuning_history = tuning_history_raw
        return TuningResult(
            best_model=best_model,
            best_params_=loaded.get("best_params_", {}),
            tuning_history_=tuning_history,
        )
    elif obj_type == "Embedder":
        # Reconstruct Embedder — rebuild vectorizer and DataFrame from parts
        import numpy as np

        from ._transforms import TfidfVectorizer

        parts = loaded["_vectorizer_parts"]
        vec = TfidfVectorizer(max_features=parts["max_features"])
        vec.vocabulary_ = parts["vocabulary_"]
        vec.idf_ = np.array(parts["idf_"])

        vdata = loaded["vectors_data"]
        vectors = pd.DataFrame(vdata["values"], columns=vdata["columns"])

        return Embedder(
            vectors=vectors,
            method=loaded["method"],
            vocab_size=loaded["vocab_size"],
            _vectorizer=vec,
        )
    elif obj_type == "Tokenizer":
        # Reconstruct Tokenizer — rebuild each column's TfidfVectorizer from parts
        import numpy as np

        from ._transforms import TfidfVectorizer

        vectorizers = {}
        for col, parts in loaded["_vectorizer_parts"].items():
            vec = TfidfVectorizer(max_features=parts["max_features"])
            vec.vocabulary_ = parts["vocabulary_"]
            vec.idf_ = np.array(parts["idf_"])
            vectorizers[col] = vec

        return Tokenizer(
            columns=loaded["columns"],
            max_features=loaded["max_features"],
            _vectorizers=vectorizers,
        )
    elif obj_type == "Scaler":
        import numpy as np

        from ._transforms import MinMaxScaler, RobustScaler, StandardScaler

        _cls_map = {
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "RobustScaler": RobustScaler,
        }
        scalers = {}
        for col, parts in loaded["_scaler_parts"].items():
            t = parts["type"]
            s = _cls_map[t]()
            if t == "StandardScaler":
                s.mean_ = np.array(parts["mean_"])
                s.scale_ = np.array(parts["scale_"])
                s.var_ = np.array(parts["var_"])
                s.n_features_in_ = 1
                s.n_samples_seen_ = 1  # sentinel — not used after fit
            elif t == "MinMaxScaler":
                s.min_ = np.array(parts["min_"])
                s.scale_ = np.array(parts["scale_"])
                s.data_min_ = np.array(parts["data_min_"])
                s.data_max_ = np.array(parts["data_max_"])
                s.data_range_ = np.array(parts["data_range_"])
                s.feature_range = tuple(parts["feature_range"])
                s.n_features_in_ = 1
            else:  # RobustScaler
                s.center_ = np.array(parts["center_"])
                s.scale_ = np.array(parts["scale_"])
                s.n_features_in_ = 1
            scalers[col] = s

        return Scaler(
            columns=loaded["columns"],
            method=loaded["method"],
            _scalers=scalers,
        )
    elif obj_type == "Encoder":
        return Encoder(
            columns=loaded["columns"],
            method=loaded["method"],
            _categories=loaded["_categories"],
            _te_mapping=loaded.get("_te_mapping", {}),
            _te_global=loaded.get("_te_global", {}),
            _te_classes=loaded.get("_te_classes", None),
            _fold_indices=loaded.get("_fold_indices", None),
            _freq_mapping=loaded.get("_freq_mapping", {}),
            _woe_mapping=loaded.get("_woe_mapping", {}),
            _iv_scores=loaded.get("_iv_scores", {}),
            _dt_include_hour=loaded.get("_dt_include_hour", {}),
        )
    elif obj_type == "Imputer":
        return Imputer(
            columns=loaded["columns"],
            strategy=loaded["strategy"],
            fill_value=loaded["fill_value"],
            _fill_values=loaded["_fill_values"],
        )
    elif obj_type == "Pipeline":
        import numpy as np

        from ._transforms import MinMaxScaler, RobustScaler, StandardScaler, TfidfVectorizer
        from .pipeline import Pipeline as PipelineClass

        _cls_map = {
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "RobustScaler": RobustScaler,
        }
        steps = []
        for step_dict in loaded["steps"]:
            st = step_dict["__step_type__"]
            if st == "Tokenizer":
                vectorizers = {}
                for col, parts in step_dict["_vectorizer_parts"].items():
                    vec = TfidfVectorizer(max_features=parts["max_features"])
                    vec.vocabulary_ = parts["vocabulary_"]
                    vec.idf_ = np.array(parts["idf_"])
                    vectorizers[col] = vec
                steps.append(Tokenizer(
                    columns=step_dict["columns"],
                    max_features=step_dict["max_features"],
                    _vectorizers=vectorizers,
                ))
            elif st == "Scaler":
                scalers = {}
                for col, parts in step_dict["_scaler_parts"].items():
                    t = parts["type"]
                    s = _cls_map[t]()
                    if t == "StandardScaler":
                        s.mean_ = np.array(parts["mean_"])
                        s.scale_ = np.array(parts["scale_"])
                        s.var_ = np.array(parts["var_"])
                        s.n_features_in_ = 1
                        s.n_samples_seen_ = 1
                    elif t == "MinMaxScaler":
                        s.min_ = np.array(parts["min_"])
                        s.scale_ = np.array(parts["scale_"])
                        s.data_min_ = np.array(parts["data_min_"])
                        s.data_max_ = np.array(parts["data_max_"])
                        s.data_range_ = np.array(parts["data_range_"])
                        s.feature_range = tuple(parts["feature_range"])
                        s.n_features_in_ = 1
                    else:  # RobustScaler
                        s.center_ = np.array(parts["center_"])
                        s.scale_ = np.array(parts["scale_"])
                        s.n_features_in_ = 1
                    scalers[col] = s
                steps.append(Scaler(
                    columns=step_dict["columns"],
                    method=step_dict["method"],
                    _scalers=scalers,
                ))
            elif st == "Encoder":
                steps.append(Encoder(
                    columns=step_dict["columns"],
                    method=step_dict["method"],
                    _categories=step_dict["_categories"],
                    _te_mapping=step_dict.get("_te_mapping", {}),
                    _te_global=step_dict.get("_te_global", {}),
                    _te_classes=step_dict.get("_te_classes", None),
                    _fold_indices=step_dict.get("_fold_indices", None),
                    _freq_mapping=step_dict.get("_freq_mapping", {}),
                ))
            elif st == "Imputer":
                steps.append(Imputer(
                    columns=step_dict["columns"],
                    strategy=step_dict["strategy"],
                    fill_value=step_dict["fill_value"],
                    _fill_values=step_dict["_fill_values"],
                ))
        return PipelineClass(steps=steps)
    else:
        from ._types import ConfigError

        raise ConfigError(f"Unknown object type in file: {obj_type}")
