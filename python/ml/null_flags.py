"""null_flags() — null indicator feature engineering.

Adds binary {col}_is_null columns for columns with missing values.
No fitting required — pure feature transformation. No target leakage.

Usage:
    >>> data_flagged = ml.null_flags(data)              # auto-detect nullable cols
    >>> data_flagged = ml.null_flags(data, columns=["age", "income"])
"""

from __future__ import annotations

import pandas as pd


def null_flags(
    data: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    target: str | None = None,
) -> pd.DataFrame:
    """Add binary ``{col}_is_null`` indicator columns for missing values.

    No fitting required — pure feature transformation. Safe to call on
    train/valid/test independently with identical results.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    columns : list[str], optional
        Columns to add null indicators for.
        If ``None`` (default), auto-detects all columns with any missing values.
    target : str, optional
        Name of the target column. When provided, ``{target}_is_null`` is NOT
        added even if the target has missing values — avoids target leakage.

    Returns
    -------
    pd.DataFrame
        New DataFrame with added ``{col}_is_null`` int8 columns (0 or 1).
        Original columns are preserved unchanged.

    Raises
    ------
    DataError
        If data is not a DataFrame or any specified column is missing.

    Examples
    --------
    >>> data_flagged = ml.null_flags(data)
    >>> model = ml.fit(data_flagged, "label", seed=42)

    >>> # Only flag specific columns:
    >>> data_flagged = ml.null_flags(data, columns=["age", "income"])
    """

    from ._types import DataError
    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"null_flags() expects DataFrame, got {type(data).__name__}."
        )

    if columns is None:
        columns = [
            c for c in data.columns
            if data[c].isna().any() and c != target
        ]
    else:
        missing_cols = [c for c in columns if c not in data.columns]
        if missing_cols:
            raise DataError(
                f"Columns not found in data: {missing_cols}. "
                f"Available: {list(data.columns)}"
            )

    out = data.copy()
    for col in columns:
        out[f"{col}_is_null"] = data[col].isna().astype("int8")

    return out
