"""prepare — Grammar primitive #2: DataFrame -> PreparedData.

Called automatically by fit() per fold. Use explicitly when you need manual
control over encoding, imputation, or scaling before fitting.
"""

from __future__ import annotations

import pandas as pd

from ._types import ConfigError, PreparedData


def prepare(
    data: pd.DataFrame,
    target: str,
    *,
    algorithm: str = "auto",
    task: str = "auto",
) -> PreparedData:
    """Encode, impute, and scale training data.

    Grammar primitive #2. In the default workflow, fit() calls prepare()
    internally per fold — you never need to call this explicitly. Use it
    when you want manual control: inspect the preprocessing state, apply
    the same encoding to external data, or chain prepare() with fit().

    Args:
        data: Training DataFrame including the target column.
        target: Name of the target column.
        algorithm: Algorithm hint for encoding strategy ("auto", "random_forest",
            "logistic", etc.). Tree-based algorithms use ordinal encoding;
            linear algorithms use one-hot for low-cardinality categoricals.
        task: "classification", "regression", or "auto" (detected from target).

    Returns:
        PreparedData with:
            .data   — transformed DataFrame (all-numeric, ready for fit)
            .state  — NormState; call state.transform(X) to apply the same
                      encoding to validation or test data
            .target — target column name
            .task   — detected or provided task type

    Raises:
        ConfigError: if target is not in data.
        DataError: if data contains infinite values.

    Example:
        >>> import ml
        >>> s = ml.split(df, "y", seed=42)
        >>> p = ml.prepare(s.train, "y")
        >>> model = ml.fit(s.train, "y", seed=42)
        >>> # Apply same encoding to validation set:
        >>> X_val = p.state.transform(s.valid.drop(columns=["y"]))
    """
    if data is None:
        raise ConfigError("data must not be None")
    if target not in data.columns:
        raise ConfigError(f"target '{target}' not found in data columns: {list(data.columns)}")

    X = data.drop(columns=[target])
    y = data[target]

    from . import _normalize
    from .split import _detect_task

    detected_task = task if task != "auto" else _detect_task(y)
    state = _normalize.prepare(X, y, algorithm=algorithm, task=detected_task)

    # Consume the cached transformed training data from prepare()
    X_transformed = state.pop_train_data()
    if X_transformed is None:
        X_transformed = state.transform(X)

    # Reconstruct DataFrame with proper column names
    feature_names = state.feature_names
    if hasattr(X_transformed, "columns"):
        result_df = X_transformed
    else:
        result_df = pd.DataFrame(X_transformed, index=X.index, columns=feature_names)

    return PreparedData(
        data=result_df,
        state=state,
        target=target,
        task=detected_task,
    )
