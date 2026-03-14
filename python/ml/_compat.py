"""Compatibility layer for non-pandas DataFrames.

Auto-converts Polars (and any DataFrame with .to_pandas()) to pandas.
"""

from __future__ import annotations

import pandas as pd


def to_pandas(data):
    """Convert pandas, Polars, or interchange DataFrames to pandas.

    Supports:
    - pandas DataFrame (passthrough — same object, no copy)
    - narwhals-compatible frames (Polars, etc.) via narwhals.from_native
    - Any object with .to_pandas() method (duck-type fallback)
    - DataFrame interchange protocol (__dataframe__)

    Raises DataError for types that cannot be converted.
    """
    if isinstance(data, pd.DataFrame):
        return data

    # narwhals path: Polars and other narwhals-compatible frames
    try:
        import narwhals as nw
        return nw.from_native(data).to_pandas()
    except (ImportError, TypeError):
        pass

    # Duck-type fallback: anything with .to_pandas() gets converted
    if hasattr(data, "to_pandas") and callable(data.to_pandas):
        return data.to_pandas()

    # DataFrame interchange protocol (PEP 647)
    if hasattr(data, "__dataframe__"):
        try:
            return pd.api.interchange.from_dataframe(data)
        except Exception:
            pass

    # Pass through internal ml types (CVResult, Model, etc.) — let caller validate
    type_name = type(data).__name__
    if type_name in ("CVResult", "Model", "TuningResult", "SplitResult"):
        return data

    # Not a DataFrame and not an internal type — raise DataError with helpful hint
    from ._types import DataError
    raise DataError(
        f"Expected pandas or Polars DataFrame, got {type_name}. "
        "Install narwhals for Polars support: pip install mlw[polars]"
    )
