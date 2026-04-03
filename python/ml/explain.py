"""Explain model predictions via feature importance.

Gate 1: Native feature importance (tree MDI, linear coefficients).
A7:     SHAP values (method="shap") and permutation importance (method="permutation").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ._types import Explanation, Model, TuningResult


def explain(
    model: Model | TuningResult,
    *,
    data: pd.DataFrame | None = None,
    method: str = "auto",
    seed: int | None = None,
    feature_groups: list[list[str]] | None = None,
) -> Explanation:
    """Explain model via feature importance.

    Returns normalized importance scores (sum to 1.0), sorted descending.

    Args:
        model: Fitted Model or TuningResult.
        data: Evaluation data (DataFrame with target column). Required for
            method="permutation". Optional for method="shap" (uses training
            features as background when omitted).
        method: How to compute importance. One of:
            - "auto" (default): tree MDI for tree models, absolute coefficients
              for linear models. Fast, no extra data needed.
            - "shap": SHAP values via the ``shap`` package (must be installed).
              Requires ``pip install shap``. Returns unbiased interaction-aware
              scores. ``explanation.shap_values`` DataFrame is also populated.
            - "permutation": sklearn permutation importance. Requires ``data=``.
              Measures actual prediction degradation when each feature is shuffled.

              .. warning::
                  Permutation importance may underestimate importance of correlated
                  features — shuffling one while leaving its correlated partner intact
                  allows the model to recover signal. For correlated feature sets,
                  prefer method="shap" or conditional permutation.

        seed: Random seed for permutation (reproducibility). Ignored for "auto".
        feature_groups: Groups of features to permute simultaneously for grouped
            permutation importance. Each inner list is one group. Requires
            ``method="permutation"`` and ``data=``. Use ``ml.cluster_features()``
            to generate groups from correlated features. Returns one row per group.

    Returns:
        Explanation with columns: feature, importance.
        ``explanation.method`` reports which method was used.
        ``explanation.shap_values`` is populated when method="shap" (else None).

    Raises:
        ConfigError: If method is unknown, or data= missing for permutation.
        ConfigError: If shap is not installed and method="shap".
        ModelError: If auto/shap cannot find importances for this algorithm.

    Example:
        >>> imp = ml.explain(model)
        >>> imp.head()
          feature  importance
        0     age        0.35

        >>> imp = ml.explain(model, data=s.valid, method="permutation", seed=42)
        >>> imp.shap_values is None
        True

        >>> imp = ml.explain(model, data=s.valid, method="shap")
        >>> imp.shap_values.shape
        (n_samples, n_features)
    """
    from ._types import ConfigError, Model, TuningResult

    if not isinstance(model, (Model, TuningResult)):
        raise ConfigError(
            f"explain() requires a fitted Model or TuningResult, got {type(model).__name__}. "
            "Use: ml.explain(model) where model = ml.fit(s.train, 'target', seed=42)"
        )

    if method not in ("auto", "shap", "permutation"):
        raise ConfigError(
            f"method='{method}' not recognised. Choose from: 'auto', 'shap', 'permutation'."
        )

    # DFA state transition: explain is idempotent (state unchanged)
    import contextlib

    from ._types import check_workflow_transition
    with contextlib.suppress(Exception):
        model._workflow_state = check_workflow_transition(
            model._workflow_state, "explain"
        )

    # Unwrap TuningResult → Model
    if isinstance(model, TuningResult):
        model = model.best_model

    # Grouped permutation importance (Chain 4.7)
    if feature_groups is not None:
        if data is None:
            raise ConfigError(
                "explain() with feature_groups= requires data=. "
                "Example: ml.explain(model, data=s.valid, method='permutation', "
                "feature_groups=groups, seed=42)"
            )
        return _explain_grouped_permutation(model, data, seed, feature_groups)

    if method == "permutation":
        if data is None:
            raise ConfigError(
                "explain() with method='permutation' requires data=. "
                "Example: ml.explain(model, data=s.valid, method='permutation', seed=42)"
            )
        return _explain_permutation(model, data, seed)

    if method == "shap":
        return _explain_shap(model, data, seed)

    # method == "auto"

    # Stacked models: return meta-learner weights
    if model._algorithm == "stacked":
        return _explain_stacked(model)

    return _explain_native(model)


# ---------------------------------------------------------------------------
# Auto (native importance)
# ---------------------------------------------------------------------------

def _explain_native(model: Model) -> Explanation:
    """Tree MDI or absolute coefficients — fast, no extra data needed."""
    import numpy as np

    from ._types import ConfigError, Explanation, ModelError  # noqa: F401

    inner_model = _unwrap_calibrated(model._model)

    directions = None
    if hasattr(inner_model, "feature_importances_"):
        importances = inner_model.feature_importances_
        method_name = "tree_importance"
        import warnings
        enc = model._feature_encoder
        if enc is not None and hasattr(enc, "category_maps") and enc.category_maps:
            high_card = [col for col, cmap in enc.category_maps.items() if len(cmap) > 20]
            if high_card:
                names = ", ".join(high_card[:3])
                warnings.warn(
                    f"tree_importance (MDI) is biased toward high-cardinality features. "
                    f"Features with many categories ({names}) may appear more important "
                    f"than they are. Consider permutation importance for unbiased results.",
                    UserWarning, stacklevel=3,
                )
    elif hasattr(inner_model, "coef_"):
        raw_coef = inner_model.coef_
        if len(raw_coef.shape) > 1:
            coef_avg = raw_coef.mean(axis=0)
            importances = np.abs(coef_avg)
            directions = np.sign(coef_avg)
        else:
            importances = np.abs(raw_coef)
            directions = np.sign(raw_coef)
        method_name = "abs_coefficients"
    else:
        raise ModelError(
            f"explain() requires a model with feature importances. "
            f"algorithm='{model._algorithm}' does not support this. "
            f"Try algorithm='xgboost' or 'random_forest', or use method='permutation'."
        )

    total = importances.sum()
    normalized = importances / total if total > 0 else np.zeros_like(importances)
    feature_names = _get_encoded_feature_names(model)

    df_data: dict = {"feature": feature_names, "importance": [float(s) for s in normalized]}
    if directions is not None:
        df_data["direction"] = [float(d) for d in directions]
    df = pd.DataFrame(df_data).sort_values("importance", ascending=False).reset_index(drop=True)
    return Explanation(df, algorithm=model._algorithm, method=method_name)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def _explain_shap(model: Model, data: pd.DataFrame | None, seed: int | None) -> Explanation:
    """SHAP-based feature importance. Requires the ``shap`` package."""
    import numpy as np

    from ._types import ConfigError, Explanation

    try:
        import shap
    except ImportError:
        raise ConfigError(
            "SHAP is not installed. Install it with: pip install shap\n"
            "Or: pip install 'ml[shap]'"
        ) from None

    enc = model._feature_encoder
    if data is not None:
        X_raw = data.drop(columns=[model._target]) if model._target in data.columns else data
        X = enc.transform(X_raw) if enc is not None else X_raw
    else:
        # No data: use a zero-background (works for tree models; warns for linear)
        n_features = len(model._features)
        X = pd.DataFrame(
            np.zeros((1, n_features)), columns=_get_encoded_feature_names(model)
        )

    inner = _unwrap_calibrated(model._model)
    feature_names = _get_encoded_feature_names(model)

    import warnings as _warn
    with _warn.catch_warnings():
        _warn.filterwarnings("ignore")

        if hasattr(inner, "feature_importances_"):
            try:
                explainer = shap.TreeExplainer(inner)
                raw = explainer.shap_values(X)
            except Exception as e:
                raise ConfigError(
                    f"SHAP TreeExplainer does not support algorithm='{model._algorithm}'. "
                    "Use method='permutation' instead."
                ) from e
        elif hasattr(inner, "coef_"):
            # LinearExplainer needs a background dataset
            bg = X if len(X) <= 100 else X.sample(100, random_state=seed or 0)
            explainer = shap.LinearExplainer(inner, bg)
            raw = explainer.shap_values(X)
        else:
            raise ConfigError(
                f"SHAP does not support algorithm='{model._algorithm}' via TreeExplainer or "
                "LinearExplainer. Use method='auto' or method='permutation' instead."
            )

    # Multiclass: shap returns list of arrays (one per class) → mean abs across classes
    if isinstance(raw, list):
        abs_shap = np.mean([np.abs(sv) for sv in raw], axis=0)
    else:
        abs_shap = np.abs(raw)

    # Mean absolute SHAP per feature (global importance)
    mean_abs = abs_shap.mean(axis=0)
    total = mean_abs.sum()
    normalized = mean_abs / total if total > 0 else np.zeros_like(mean_abs)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": [float(v) for v in normalized],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Build shap_values DataFrame (rows = samples, cols = features)
    shap_df = pd.DataFrame(abs_shap, columns=feature_names)

    return Explanation(df, algorithm=model._algorithm, method="shap", shap_values=shap_df)


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def _explain_permutation(model: Model, data: pd.DataFrame, seed: int | None) -> Explanation:
    """Permutation importance — unbiased but slower, requires data=."""
    import warnings

    import numpy as np

    from ._types import DataError, Explanation

    target = model._target
    if target not in data.columns:
        raise DataError(
            f"target='{target}' not found in data. "
            f"Available columns: {data.columns.tolist()}"
        )

    X_raw = data.drop(columns=[target])
    y = data[target]
    enc = model._feature_encoder
    X = enc.transform(X_raw) if enc is not None else X_raw
    y_enc = enc.encode_target(y) if enc is not None else y

    feature_names = _get_encoded_feature_names(model)

    # Warn if correlated features detected (Puget C5 bias)
    if len(X.columns) >= 2:
        corr = X.corr().abs()
        # Upper triangle excluding diagonal
        upper = corr.where(
            pd.DataFrame(
                np.triu(np.ones(corr.shape), k=1).astype(bool),
                index=corr.index, columns=corr.columns,
            )
        )
        max_corr = upper.max().max() if not upper.empty else 0.0
        if max_corr > 0.7:
            warnings.warn(
                "Permutation importance may underestimate importance of correlated features "
                f"(max |r|={max_corr:.2f} detected). Shuffling one feature while its "
                "correlated partner remains intact lets the model partially recover signal. "
                "Consider method='shap' for correlated feature sets.",
                UserWarning, stacklevel=3,
            )

    # Pure-numpy permutation importance (no sklearn dependency)
    inner = model._model
    is_clf = model._task == "classification"
    rng = np.random.RandomState(seed if seed is not None else 0)
    n_repeats = 5

    def _score(est, X_s, y_s):
        """Score an estimator — works with or without .score()."""
        if hasattr(est, "score"):
            return est.score(X_s, y_s)
        preds = est.predict(X_s)
        if is_clf:
            return float(np.mean(preds == y_s))
        y_arr = np.asarray(y_s, dtype=np.float64).ravel()
        ss_res = np.sum((y_arr - preds) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    try:
        baseline_score = _score(inner, X, y_enc)
    except Exception:
        baseline_score = 0.5

    importances = np.zeros(len(feature_names))
    for j in range(len(feature_names)):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(len(X))
            if hasattr(X_perm, "iloc"):
                X_perm.iloc[:, j] = X.iloc[:, j].values[perm_idx]
            else:
                X_perm[:, j] = X[perm_idx, j]
            try:
                scores.append(_score(inner, X_perm, y_enc))
            except Exception:
                scores.append(baseline_score)
        importances[j] = baseline_score - float(np.mean(scores))

    importances = importances.clip(min=0)  # clip negatives (noise → 0)
    total = importances.sum()
    normalized = importances / total if total > 0 else np.zeros_like(importances)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": [float(v) for v in normalized],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return Explanation(df, algorithm=model._algorithm, method="permutation")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap_calibrated(inner_model):
    """Unwrap calibrated model to access the base estimator."""
    from .calibrate import _CalibratedModel

    if isinstance(inner_model, _CalibratedModel):
        return inner_model.base_estimator
    # Legacy: sklearn CalibratedClassifierCV (pre-existing saved models)
    try:
        from sklearn.calibration import CalibratedClassifierCV
        if isinstance(inner_model, CalibratedClassifierCV):
            base = inner_model.calibrated_classifiers_[0].estimator
            if hasattr(base, "estimator"):
                base = base.estimator
            return base
    except ImportError:
        pass
    return inner_model


def _get_encoded_feature_names(model: Model) -> list[str]:
    """Get post-encoding feature names matching model coefficient dimensions."""
    enc = model._feature_encoder
    if enc is not None and enc.onehot_encoder is not None and enc.onehot_columns:
        names = [f for f in enc.feature_names if f not in enc.onehot_columns]
        names.extend(enc.onehot_encoder.get_feature_names_out(enc.onehot_columns))
        return names
    return model._features


def _explain_grouped_permutation(
    model: Model,
    data: pd.DataFrame,
    seed: int | None,
    feature_groups: list[list[str]],
) -> Explanation:
    """Grouped permutation importance — permutes all features in a group simultaneously.

    Avoids the underestimation bias that occurs when shuffling one correlated
    feature while leaving its partner intact (Puget C5).
    """
    import numpy as np

    from ._types import DataError, Explanation

    target = model._target
    if target not in data.columns:
        raise DataError(
            f"target='{target}' not found in data. "
            f"Available: {data.columns.tolist()}"
        )

    X_raw = data.drop(columns=[target])
    y = data[target]
    enc = model._feature_encoder
    X = enc.transform(X_raw) if enc is not None else X_raw
    y_enc = enc.encode_target(y) if enc is not None else y

    inner = model._model
    is_clf = model._task == "classification"
    rng = np.random.RandomState(seed if seed is not None else 0)

    def _score(est, X_s, y_s):
        if hasattr(est, "score"):
            return est.score(X_s, y_s)
        preds = est.predict(X_s)
        if is_clf:
            return float(np.mean(preds == y_s))
        y_arr = np.asarray(y_s, dtype=np.float64).ravel()
        ss_res = np.sum((y_arr - preds) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    try:
        baseline_score = _score(inner, X, y_enc)
    except Exception:
        baseline_score = 0.5

    group_importances: list[float] = []
    group_names: list[str] = []

    n_repeats = 5
    for group in feature_groups:
        valid = [f for f in group if f in X.columns]
        if not valid:
            continue

        repeat_scores: list[float] = []
        for _ in range(n_repeats):
            perm_idx = rng.permutation(len(X))
            X_perm = X.copy()
            for feat in valid:
                X_perm[feat] = X[feat].values[perm_idx]
            try:
                s = _score(inner, X_perm, y_enc)
            except Exception:
                s = baseline_score
            repeat_scores.append(s)

        importance = max(0.0, baseline_score - float(np.mean(repeat_scores)))
        group_importances.append(importance)
        group_names.append("+".join(valid))

    total = sum(group_importances)
    n_groups = len(group_importances)
    if total > 0:
        normalized = [v / total for v in group_importances]
    elif n_groups > 0:
        # All groups equally (un)important — use uniform weights
        normalized = [1.0 / n_groups] * n_groups
    else:
        normalized = []

    df = pd.DataFrame({
        "feature": group_names,
        "importance": normalized,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return Explanation(df, algorithm=model._algorithm, method="permutation_grouped")


def _explain_stacked(model: Model) -> Explanation:
    """Explain stacked model by returning meta-learner weights."""
    import numpy as np

    from ._types import Explanation

    stacking_est = model._model
    meta = stacking_est.final_estimator_
    base_names = [name for name, _ in stacking_est.estimators]

    if hasattr(meta, "coef_"):
        weights = np.abs(meta.coef_)
        if len(weights.shape) > 1:
            weights = weights.mean(axis=0)
    elif hasattr(meta, "feature_importances_"):
        weights = meta.feature_importances_
    else:
        weights = np.ones(len(base_names)) / len(base_names)

    n_base = len(base_names)
    if len(weights) > n_base and len(weights) % n_base == 0:
        n_per_base = len(weights) // n_base
        weights = np.array([
            weights[i * n_per_base:(i + 1) * n_per_base].sum()
            for i in range(n_base)
        ])

    total = weights.sum()
    normalized = weights / total if total > 0 else np.zeros_like(weights)

    df = pd.DataFrame({
        "feature": base_names,
        "importance": [float(w) for w in normalized],
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return Explanation(df, algorithm="stacked", method="meta_weights")
