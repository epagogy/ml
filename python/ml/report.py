"""Automated training report — ml.report().

Generates an HTML report with model summary, metrics, and feature importance.
Uses only stdlib + pandas. Matplotlib plots embedded as base64 if available.
"""

from __future__ import annotations

import base64
import io
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._types import Model


def report(
    model: Model,
    data,
    *,
    path: str = "report.html",
) -> str:
    """Generate an HTML training report.

    Creates a self-contained HTML file with model summary, evaluation metrics,
    and (if matplotlib is installed) feature importance chart and confusion matrix.

    Args:
        model: Fitted Model from ml.fit()
        data: Validation/test DataFrame (with target column for metrics)
        path: Output file path (default "report.html")

    Returns:
        Path to the generated HTML file.

    Raises:
        ConfigError: If model is not a fitted Model.
        DataError: If data is not a DataFrame.

    Example:
        >>> s = ml.split(data, "target", seed=42)
        >>> model = ml.fit(s.train, "target", seed=42)
        >>> ml.report(model, s.valid, path="my_report.html")
        'my_report.html'
    """
    path = os.path.abspath(path)

    from ._types import ConfigError, DataError
    from ._types import Model as ModelType

    if not isinstance(model, ModelType):
        raise ConfigError(
            f"report() requires a fitted Model, got {type(model).__name__}. "
            "Use ml.fit() first."
        )

    import pandas as pd


    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"report() requires a DataFrame, got {type(data).__name__}."
        )

    from .evaluate import evaluate

    # Compute evaluation metrics
    metrics = evaluate(model, data)

    sections: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    sections.append(
        f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ML Training Report</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
h1 {{ color: #1a1a2e; }}
h2 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #0f3460; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.metric-value {{ font-weight: bold; color: #0f3460; }}
.section {{ margin: 32px 0; }}
img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
</style>
</head>
<body>
<h1>ML Training Report</h1>
<p>Model: <code>{model._algorithm}</code> | Task: <code>{model._task}</code> | Features: <code>{len(model._features)}</code></p>
"""
    )

    # ── Model Summary ────────────────────────────────────────────────────────
    sections.append(
        """<div class="section">
<h2>Model Summary</h2>
<table>
<tr><th>Property</th><th>Value</th></tr>
"""
    )
    sections.append(
        f"<tr><td>Algorithm</td><td class='metric-value'>{model._algorithm}</td></tr>\n"
    )
    sections.append(
        f"<tr><td>Task</td><td class='metric-value'>{model._task}</td></tr>\n"
    )
    sections.append(
        f"<tr><td>Features</td><td class='metric-value'>{len(model._features)}</td></tr>\n"
    )
    classes_ = model.classes_ if model._task == "classification" else None
    if classes_:
        sections.append(
            f"<tr><td>Classes</td><td class='metric-value'>{classes_}</td></tr>\n"
        )
    sections.append("</table></div>\n")

    # ── Metrics ──────────────────────────────────────────────────────────────
    sections.append(
        """<div class="section">
<h2>Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
"""
    )
    for k, v in metrics.items():
        sections.append(
            f"<tr><td>{k}</td><td class='metric-value'>{v:.4f}</td></tr>\n"
        )
    sections.append("</table></div>\n")

    # ── Feature Importance (matplotlib optional) ─────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .explain import explain

        exp = explain(model)
        imp = exp.importances

        vals = None
        if isinstance(imp, pd.Series):
            vals = imp.sort_values(ascending=False).head(20)
        elif isinstance(imp, pd.DataFrame):
            numeric_cols = imp.select_dtypes("number").columns
            if len(numeric_cols) > 0:
                vals = imp[numeric_cols[0]].sort_values(ascending=False).head(20)

        if vals is not None and len(vals) > 0:
            fig, ax = plt.subplots(figsize=(8, max(4, len(vals) * 0.35)))
            vals.plot.barh(ax=ax)
            ax.invert_yaxis()
            ax.set_title("Feature Importance (top 20)")
            ax.set_xlabel("Importance")
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()

            sections.append(
                f"""<div class="section">
<h2>Feature Importance</h2>
<img src="data:image/png;base64,{img_b64}" alt="Feature Importance">
</div>
"""
            )
    except Exception:
        sections.append(
            """<div class="section">
<h2>Feature Importance</h2>
<p><em>Install matplotlib for visual charts: pip install matplotlib</em></p>
</div>
"""
        )

    # ── Confusion Matrix (classification only, matplotlib required) ───────────
    if model._task == "classification" and model._target in data.columns:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            from .predict import _predict_impl

            preds = _predict_impl(model, data)
            y_true = data[model._target]
            classes = sorted(set(y_true.unique()) | set(preds.unique()),
                             key=lambda x: str(x))
            n = len(classes)
            cm = np.zeros((n, n), dtype=int)
            class_idx = {c: i for i, c in enumerate(classes)}
            for true_val, pred_val in zip(y_true, preds):
                cm[class_idx[true_val], class_idx[pred_val]] += 1

            fig, ax = plt.subplots(figsize=(max(5, n), max(4, n)))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels([str(c) for c in classes], rotation=45, ha="right")
            ax.set_yticklabels([str(c) for c in classes])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            for i in range(n):
                for j in range(n):
                    ax.text(
                        j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                    )
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()

            sections.append(
                f"""<div class="section">
<h2>Confusion Matrix</h2>
<img src="data:image/png;base64,{img_b64}" alt="Confusion Matrix">
</div>
"""
            )
        except Exception:
            pass

    # ── Footer ───────────────────────────────────────────────────────────────
    sections.append("</body></html>\n")

    html = "".join(sections)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    return path
