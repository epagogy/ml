"""Feature selection from fitted models."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ._types import Model


def select(
    model: Model,
    *,
    data: pd.DataFrame | None = None,
    method: str = "importance",
    threshold: float = 0.01,
    correlation_max: float = 0.95,
    seed: int | None = None,
) -> list[str]:
    """Select important features from a fitted model.

    Args:
        model: Fitted Model
        method: "importance" — keep features with importance >= threshold.
                "correlation" — additionally drop one of each correlated pair.
                "permutation" — use permutation importance (requires data= and seed=).
        threshold: Minimum importance to keep (for method="importance").
        correlation_max: Maximum allowed pairwise |r| (for method="correlation").
        data: Validation DataFrame required for method="permutation".
        seed: Required for method="permutation".

    Returns:
        List of feature names to keep.
    """
    from ._types import ConfigError, DataError, TuningResult
    from ._types import Model as ModelType

    if isinstance(model, TuningResult):
        model = model.best_model
    elif not isinstance(model, ModelType):
        raise ConfigError(
            f"select() requires a fitted Model. Got {type(model).__name__}."
        )

    features = list(model._features)

    if method == "permutation":
        if data is None:
            raise DataError("select(method='permutation') requires data=.")
        if seed is None:
            raise ConfigError("select(method='permutation') requires seed=.")
        from . import explain
        imp = explain(model, data=data, method="permutation", seed=seed)
        imp_dict = dict(imp.items())
    elif method in ("importance", "correlation"):
        from . import explain
        imp = explain(model)
        imp_dict = dict(imp.items())
    else:
        raise ConfigError(
            f"Unknown method '{method}'. Valid: 'importance', 'correlation', 'permutation'."
        )

    # Filter by threshold
    selected = [f for f in features if imp_dict.get(f, 0.0) >= threshold]

    # If threshold removes everything, keep the top feature
    if not selected:
        selected = [max(imp_dict, key=imp_dict.get)]

    # Drop correlated features
    if method == "correlation" and data is not None:
        X = data[[f for f in selected if f in data.columns]]
        if len(X.columns) > 1:
            corr = X.corr().abs()
            to_drop = set()
            cols = list(corr.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if corr.iloc[i, j] > correlation_max:
                        # Drop the one with lower importance
                        fi, fj = cols[i], cols[j]
                        drop = fi if imp_dict.get(fi, 0) < imp_dict.get(fj, 0) else fj
                        to_drop.add(drop)
            selected = [f for f in selected if f not in to_drop]

    return selected
