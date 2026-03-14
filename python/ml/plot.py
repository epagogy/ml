"""Visualization — ml.plot().

All plots require matplotlib. Install with:
    pip install ml[plots]
    # or: pip install matplotlib

Returns matplotlib.figure.Figure in all cases.
Close figures explicitly to avoid memory leaks:
    fig = ml.plot(model, data=s.valid, kind="roc")
    fig.savefig("roc.png")
    import matplotlib.pyplot as plt; plt.close(fig)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure


def plot(
    obj,
    *,
    data=None,
    kind: str | None = None,
    feature: str | list[str] | None = None,
    top_n: int = 20,
    idx: int = 0,
    figsize: tuple[int, int] = (8, 6),
    seed: int | None = None,
) -> matplotlib.figure.Figure:
    """Visualize ML objects.

    Args:
        obj: Model, Explanation, EnoughResult, Leaderboard, or DriftResult
        data: DataFrame for model-based plots (confusion, roc, calibration, residual, pdp)
        kind: Plot type. Auto-detected from obj type if not provided.
            "importance"    — Feature importance bar chart (Model or Explanation)
            "confusion"     — Confusion matrix heatmap (Model, requires data=)
            "roc"           — ROC curve (Model, requires data=)
            "calibration"   — Reliability diagram (Model, requires data=)
            "residual"      — Residual plot (Model regression, requires data=)
            "learning_curve"— Learning curve (EnoughResult)
            "leaderboard"   — Leaderboard bar chart (Leaderboard)
            "drift"         — Feature drift chart (DriftResult)
            "waterfall"     — SHAP waterfall (Explanation with shap_values, idx=)
            "pdp"           — Partial Dependence Plot (Model, requires data=, feature=)
        feature: Feature name(s) for pdp kind. Single str or list of 2.
        top_n: Max features to show in importance plot (default 20).
        idx: Sample index for waterfall plot (default 0).
        figsize: Figure size (default (8, 6)).
        seed: Random seed for permutation-based importance plots (required for
            SVM/KNN/models that need data=). If None, defaults to 0.

    Returns:
        matplotlib.figure.Figure

    Raises:
        ConfigError: If matplotlib is not installed
        ConfigError: If kind cannot be auto-detected
        DataError: If data= is required but not provided
        ConfigError: If kind is unsupported
    """
    import importlib.util
    if importlib.util.find_spec("matplotlib") is None:
        from ._types import ConfigError
        raise ConfigError(
            "ml.plot() requires matplotlib. Install with: pip install ml[plots]"
        )

    # Auto-detect kind from object type
    if kind is None:
        kind = _auto_detect_kind(obj)

    kind = kind.lower().replace("-", "_")

    dispatchers = {
        "importance": _plot_importance,
        "confusion": _plot_confusion,
        "roc": _plot_roc,
        "calibration": _plot_calibration,
        "residual": _plot_residual,
        "learning_curve": _plot_learning_curve,
        "leaderboard": _plot_leaderboard,
        "drift": _plot_drift,
        "waterfall": _plot_waterfall,
        "pdp": _plot_pdp,
    }

    if kind not in dispatchers:
        from ._types import ConfigError
        valid = sorted(dispatchers.keys())
        raise ConfigError(
            f"Unknown plot kind {kind!r}. Valid kinds: {valid}"
        )

    return dispatchers[kind](obj, data=data, feature=feature, top_n=top_n, idx=idx, figsize=figsize, seed=seed)


def _auto_detect_kind(obj) -> str:
    """Auto-detect plot kind from object type."""
    from ._types import (
        ConfigError,
        Explanation,  # noqa: F401 — used in isinstance check below
        Model,
    )

    type_name = type(obj).__name__

    if isinstance(obj, Model):
        return "importance"
    if type_name == "Explanation":
        return "importance"
    if type_name == "EnoughResult":
        return "learning_curve"
    if type_name == "Leaderboard":
        return "leaderboard"
    if type_name == "DriftResult":
        return "drift"

    raise ConfigError(
        f"Cannot auto-detect plot kind from {type_name}. "
        "Pass kind= explicitly: ml.plot(obj, kind='importance')"
    )


def _get_style() -> str:
    """Return best available matplotlib style."""
    import matplotlib.pyplot as plt
    available = plt.style.available
    if "seaborn-v0_8" in available:
        return "seaborn-v0_8"
    if "seaborn" in available:
        return "seaborn"
    return "default"


def _plot_importance(obj, *, data, feature, top_n, idx, figsize, seed):
    """Bar chart of feature importances."""
    import matplotlib.pyplot as plt

    from ._types import ConfigError, Explanation, Model

    if isinstance(obj, Model):
        from ._types import ConfigError as _CE
        from .explain import explain
        try:
            if data is None:
                exp = explain(obj)
            else:
                exp = explain(obj, data=data, method="permutation", seed=seed if seed is not None else 0)
        except Exception as exc:
            raise _CE(
                f"importance plot failed for algorithm '{obj._algorithm}'. "
                "For SVM/KNN models, pass data= to use permutation importance: "
                "ml.plot(model, data=s.valid, kind='importance')"
            ) from exc
    elif isinstance(obj, Explanation):
        exp = obj
    else:
        raise ConfigError(
            f"importance plot requires Model or Explanation, got {type(obj).__name__}"
        )

    # Explanation wraps a DataFrame; get importance Series sorted descending
    imp_df = exp._df  # has columns: feature, importance
    vals = (
        imp_df.set_index("feature")["importance"]
        .sort_values(ascending=False)
        .head(top_n)
    )

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=figsize)
        vals.plot.barh(ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance (top {len(vals)})")
        fig.tight_layout()

    return fig


def _plot_confusion(obj, *, data, feature, top_n, idx, figsize, seed):
    """Confusion matrix heatmap."""
    import matplotlib.pyplot as plt
    import numpy as np

    from ._types import ConfigError, DataError, Model
    from .predict import _predict_impl

    if not isinstance(obj, Model):
        raise ConfigError(f"confusion plot requires Model, got {type(obj).__name__}")
    if data is None:
        raise DataError("confusion plot requires data=. Pass data=s.valid.")
    if obj._task != "classification":
        raise ConfigError("confusion plot is only for classification models.")
    if obj._target not in data.columns:
        raise DataError(f"data= must contain target column {obj._target!r}.")

    preds = _predict_impl(obj, data)
    y_true = data[obj._target]

    classes = sorted(set(y_true.unique()) | set(preds.unique()), key=str)
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    class_idx = {c: i for i, c in enumerate(classes)}
    for true, pred in zip(y_true, preds):
        cm[class_idx[true], class_idx[pred]] += 1

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, aspect="auto", cmap="Blues")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([str(c) for c in classes], rotation=45, ha="right")
        ax.set_yticklabels([str(c) for c in classes])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        max_val = cm.max() if cm.max() > 0 else 1
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > max_val / 2 else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()

    return fig


def _plot_roc(obj, *, data, feature, top_n, idx, figsize, seed):
    """ROC curve. Binary: single curve. Multiclass: OVR + macro."""
    import matplotlib.pyplot as plt
    import numpy as np

    from ._scoring import _auc_trapz, _roc_curve_impl
    from ._types import ConfigError, DataError, Model
    from .predict import _predict_proba

    if not isinstance(obj, Model):
        raise ConfigError(f"roc plot requires Model, got {type(obj).__name__}")
    if data is None:
        raise DataError("roc plot requires data=. Pass data=s.valid.")
    if obj._task != "classification":
        raise ConfigError("roc plot is only for classification models.")
    if obj._target not in data.columns:
        raise DataError(f"data= must contain target column {obj._target!r}.")

    y_true = data[obj._target]
    proba = _predict_proba(obj, data)
    classes = list(proba.columns)

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=figsize)

        if len(classes) == 2:
            # Binary
            pos_class = classes[-1]
            fpr, tpr, _ = _roc_curve_impl((y_true == pos_class).astype(int), proba[pos_class].values)
            roc_auc = _auc_trapz(fpr, tpr)
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        else:
            # Multiclass OVR
            macro_fpr = np.linspace(0, 1, 100)
            macro_tpr = np.zeros(100)
            for cls in classes:
                y_bin = (y_true == cls).astype(int)
                fpr, tpr, _ = _roc_curve_impl(y_bin.values, proba[cls].values)
                roc_auc = _auc_trapz(fpr, tpr)
                tpr_interp = np.interp(macro_fpr, fpr, tpr)
                macro_tpr += tpr_interp
                ax.plot(fpr, tpr, alpha=0.4, label=f"{cls} AUC={roc_auc:.2f}")
            macro_tpr /= len(classes)
            macro_auc = _auc_trapz(macro_fpr, macro_tpr)
            ax.plot(macro_fpr, macro_tpr, "k--", lw=2, label=f"Macro AUC={macro_auc:.3f}")

        ax.plot([0, 1], [0, 1], "r--", alpha=0.3, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()

    return fig


def _plot_calibration(obj, *, data, feature, top_n, idx, figsize, seed):
    """Reliability diagram."""
    import matplotlib.pyplot as plt
    import numpy as np

    from ._types import ConfigError, DataError, Model
    from .predict import _predict_proba

    if not isinstance(obj, Model):
        raise ConfigError(f"calibration plot requires Model, got {type(obj).__name__}")
    if data is None:
        raise DataError("calibration plot requires data=. Pass data=s.valid.")
    if obj._task != "classification":
        raise ConfigError("calibration plot is only for classification models.")
    if obj._target not in data.columns:
        raise DataError(f"data= must contain target column {obj._target!r}.")

    y_true = data[obj._target]
    proba = _predict_proba(obj, data)
    classes = list(proba.columns)
    pos_class = classes[-1]
    y_bin = (y_true == pos_class).astype(int).values
    y_prob = proba[pos_class].values

    # Compute calibration curve manually
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    fraction_of_positives = []
    mean_predicted = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            fraction_of_positives.append(float(y_bin[mask].mean()))
            mean_predicted.append(float(y_prob[mask].mean()))

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        if mean_predicted:
            ax.plot(mean_predicted, fraction_of_positives, "s-", label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration Curve ({pos_class})")
        ax.legend()
        fig.tight_layout()

    return fig


def _plot_residual(obj, *, data, feature, top_n, idx, figsize, seed):
    """Residual plot for regression."""
    import matplotlib.pyplot as plt

    from ._types import ConfigError, DataError, Model
    from .predict import _predict_impl

    if not isinstance(obj, Model):
        raise ConfigError(f"residual plot requires Model, got {type(obj).__name__}")
    if data is None:
        raise DataError("residual plot requires data=. Pass data=s.valid.")
    if obj._task != "regression":
        raise ConfigError("residual plot is only for regression models.")
    if obj._target not in data.columns:
        raise DataError(f"data= must contain target column {obj._target!r}.")

    preds = _predict_impl(obj, data)
    y_true = data[obj._target]
    residuals = y_true.values - preds.values

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(preds.values, residuals, alpha=0.4, s=10)
        ax.axhline(0, color="red", lw=1, linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual (actual - predicted)")
        ax.set_title("Residual Plot")
        fig.tight_layout()

    return fig


def _plot_learning_curve(obj, *, data, feature, top_n, idx, figsize, seed):
    """Learning curve from EnoughResult."""
    import matplotlib.pyplot as plt

    from ._types import ConfigError

    type_name = type(obj).__name__
    if type_name != "EnoughResult":
        raise ConfigError(f"learning_curve plot requires EnoughResult, got {type_name}")

    # EnoughResult.curve is a DataFrame with columns: n_samples, train_score, val_score
    curve = getattr(obj, "curve", None)

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=figsize)
        if curve is not None and hasattr(curve, "columns"):
            if "n_samples" in curve.columns and "val_score" in curve.columns:
                ax.plot(curve["n_samples"], curve["val_score"], "o-", label="Validation")
                if "train_score" in curve.columns:
                    ax.plot(curve["n_samples"], curve["train_score"], "s--", label="Train")
                ax.set_xlabel("Training samples")
                metric = getattr(obj, "metric", "score")
                ax.set_ylabel(metric)
                ax.legend()
            else:
                ax.text(0.5, 0.5, "Learning curve data", ha="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Learning curve", ha="center", transform=ax.transAxes)
        ax.set_title("Learning Curve")
        fig.tight_layout()

    return fig


def _plot_leaderboard(obj, *, data, feature, top_n, idx, figsize, seed):
    """Horizontal bar chart of leaderboard scores."""
    import matplotlib.pyplot as plt

    from ._types import ConfigError

    type_name = type(obj).__name__
    if type_name != "Leaderboard":
        raise ConfigError(f"leaderboard plot requires Leaderboard, got {type_name}")

    # Leaderboard wraps a DataFrame in ._df
    lb_df = obj._df

    # Get numeric columns for score (skip string cols like algorithm, error)
    numeric_cols = lb_df.select_dtypes("number").columns.tolist()
    if not numeric_cols:
        raise ConfigError("Leaderboard has no numeric columns to plot.")

    # Prefer roc_auc or accuracy, else first numeric col
    preferred = ["roc_auc", "roc_auc_ovr", "accuracy", "r2", "rmse"]
    score_col = next((c for c in preferred if c in numeric_cols), numeric_cols[0])

    # Get algorithm labels
    label_col = "algorithm" if "algorithm" in lb_df.columns else None

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=figsize)
        scores = lb_df[score_col].values
        labels = lb_df[label_col].values if label_col else lb_df.index.tolist()
        ax.barh(range(len(scores)), scores)
        ax.set_yticks(range(len(scores)))
        ax.set_yticklabels([str(lb) for lb in labels])
        ax.set_xlabel(score_col)
        ax.set_title("Leaderboard")
        fig.tight_layout()

    return fig


def _plot_drift(obj, *, data, feature, top_n, idx, figsize, seed):
    """Feature drift bar chart from DriftResult."""
    import matplotlib.pyplot as plt

    from ._types import ConfigError

    type_name = type(obj).__name__
    if type_name != "DriftResult":
        raise ConfigError(f"drift plot requires DriftResult, got {type_name}")

    # DriftResult.features is dict[str, float] — per-feature p-values
    feat_dict = getattr(obj, "features", None)

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=figsize)
        if feat_dict and isinstance(feat_dict, dict):
            items = sorted(feat_dict.items(), key=lambda x: x[1])[:top_n]
            if items:
                feat_names = [str(k) for k, _ in items]
                p_vals = [float(v) for _, v in items]
                ax.barh(feat_names, p_vals)
                ax.axvline(0.05, color="red", linestyle="--", label="p=0.05")
                ax.set_xlabel("p-value")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No feature data", ha="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No feature drift data available", ha="center",
                    transform=ax.transAxes)
        severity = getattr(obj, "severity", "unknown")
        ax.set_title(f"Feature Drift (severity={severity!r})")
        fig.tight_layout()

    return fig


def _plot_waterfall(obj, *, data, feature, top_n, idx, figsize, seed):
    """SHAP waterfall for a single prediction."""
    import matplotlib.pyplot as plt

    from ._types import ConfigError, Explanation

    if not isinstance(obj, Explanation):
        raise ConfigError(f"waterfall plot requires Explanation, got {type(obj).__name__}")

    shap_vals = getattr(obj, "shap_values", None)
    if shap_vals is None:
        raise ConfigError(
            "waterfall plot requires SHAP values. "
            "Use ml.explain(model, data=data, method='shap') first."
        )

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=figsize)
        try:
            row = shap_vals.iloc[idx] if hasattr(shap_vals, "iloc") else shap_vals[idx]
            features = list(row.index) if hasattr(row, "index") else list(range(len(row)))
            values = list(row.values) if hasattr(row, "values") else list(row)
            # Sort by absolute value, take top_n
            pairs = sorted(zip(features, values), key=lambda x: abs(x[1]), reverse=True)
            pairs = pairs[:top_n]
            feat_names = [str(p[0]) for p in pairs]
            feat_vals = [p[1] for p in pairs]
            colors = ["firebrick" if v > 0 else "steelblue" for v in feat_vals]
            ax.barh(feat_names, feat_vals, color=colors)
            ax.axvline(0, color="black", lw=0.5)
            ax.set_xlabel("SHAP value")
            ax.set_title(f"SHAP Waterfall (sample {idx})")
        except Exception as exc:
            ax.text(0.5, 0.5, f"Waterfall error: {exc}",
                    ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()

    return fig


def _plot_pdp(obj, *, data, feature, top_n, idx, figsize, seed):
    """Partial Dependence Plot — manual grid prediction, no sklearn dependency."""
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np

    from ._types import ConfigError, DataError, Model

    if not isinstance(obj, Model):
        raise ConfigError(f"pdp plot requires Model, got {type(obj).__name__}")
    if data is None:
        raise DataError("pdp plot requires data=. Pass data=s.valid.")
    if feature is None:
        raise ConfigError(
            "pdp plot requires feature=. "
            "Example: ml.plot(model, data=s.valid, kind='pdp', feature='age')"
        )

    # Get feature matrix (drop target if present)
    if obj._target in data.columns:
        X = data.drop(columns=[obj._target])
    else:
        X = data
    X = X[obj._features]

    # Transform features
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        X_clean = obj._feature_encoder.transform(X)

    encoded_names = list(X_clean.columns) if hasattr(X_clean, "columns") else [f"f{i}" for i in range(X_clean.shape[1])]

    is_2d = isinstance(feature, list) and len(feature) == 2

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=figsize)
        try:
            if is_2d:
                # 2D PDP — contour plot
                f1, f2 = feature
                idx1 = encoded_names.index(f1) if f1 in encoded_names else obj._features.index(f1)
                idx2 = encoded_names.index(f2) if f2 in encoded_names else obj._features.index(f2)
                grid1 = np.percentile(X_clean.iloc[:, idx1], np.linspace(0, 100, 20))
                grid2 = np.percentile(X_clean.iloc[:, idx2], np.linspace(0, 100, 20))
                grid1 = np.unique(grid1)
                grid2 = np.unique(grid2)
                pdp_vals = np.zeros((len(grid2), len(grid1)))
                for i, v2 in enumerate(grid2):
                    for j, v1 in enumerate(grid1):
                        X_mod = X_clean.copy()
                        X_mod.iloc[:, idx1] = v1
                        X_mod.iloc[:, idx2] = v2
                        if hasattr(obj._model, "predict_proba") and obj._task == "classification":
                            preds = obj._model.predict_proba(X_mod)
                            pdp_vals[i, j] = preds[:, 1].mean() if preds.ndim > 1 else preds.mean()
                        else:
                            pdp_vals[i, j] = obj._model.predict(X_mod).mean()
                ax.contourf(grid1, grid2, pdp_vals, levels=20, cmap="viridis")
                ax.set_xlabel(f1)
                ax.set_ylabel(f2)
                fig.colorbar(ax.collections[0], ax=ax)
            else:
                # 1D PDP
                feat_name = feature if isinstance(feature, str) else feature[0]
                if feat_name in encoded_names:
                    idx = encoded_names.index(feat_name)
                else:
                    idx = obj._features.index(feat_name)
                grid = np.percentile(X_clean.iloc[:, idx], np.linspace(0, 100, 50))
                grid = np.unique(grid)
                pdp_vals = np.zeros(len(grid))
                for i, val in enumerate(grid):
                    X_mod = X_clean.copy()
                    X_mod.iloc[:, idx] = val
                    if hasattr(obj._model, "predict_proba") and obj._task == "classification":
                        preds = obj._model.predict_proba(X_mod)
                        pdp_vals[i] = preds[:, 1].mean() if preds.ndim > 1 else preds.mean()
                    else:
                        pdp_vals[i] = obj._model.predict(X_mod).mean()
                ax.plot(grid, pdp_vals)
                ax.set_xlabel(feat_name)
                ax.set_ylabel("Partial Dependence")
        except Exception as exc:
            ax.text(0.5, 0.5, f"PDP failed: {exc}",
                    ha="center", va="center", transform=ax.transAxes)
        feature_label = feature if isinstance(feature, str) else str(feature)
        ax.set_title(f"Partial Dependence: {feature_label}")
        fig.tight_layout()

    return fig
