"""impute() — missing value imputation.

Learns fill values from training data, applies to any split.
No leakage: statistics computed on train only.

Strategies: "median", "mean", "mode", "constant"

Usage:
    >>> imp = ml.impute(s.train, columns=["age", "income"])
    >>> model = ml.fit(imp.transform(s.train), "label", seed=42)
    >>> preds = ml.predict(model, imp.transform(s.valid))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Imputer:
    """Fitted missing value imputer.

    Stores the fill value learned from training data per column.
    At transform time, replaces NaN with the stored fill value.
    Column names and dtypes are preserved where possible.

    Attributes
    ----------
    columns : list[str]
        Columns that were fitted.
    strategy : str
        Imputation strategy: "median", "mean", "mode", or "constant".
    fill_value : Any
        Fill value for strategy="constant".
    _fill_values : dict[str, Any]
        Learned fill value per column (from training data).
    """

    columns: list[str]
    strategy: str
    fill_value: Any
    _fill_values: dict[str, Any] = field(repr=False)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using fitted fill values.

        Replaces NaN in each fitted column with the value learned from
        training data. Other columns are passed through unchanged.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the fitted columns (may have NaN).

        Returns
        -------
        pd.DataFrame
            New DataFrame with NaN filled in fitted columns.

        Raises
        ------
        DataError
            If any fitted column is missing from data.
        """

        from ._types import DataError
        if not isinstance(data, pd.DataFrame):
            raise DataError(
                f"transform() expects DataFrame, got {type(data).__name__}."
            )

        missing_cols = [c for c in self.columns if c not in data.columns]
        if missing_cols:
            raise DataError(
                f"Columns missing from data: {missing_cols}. "
                f"Expected columns: {self.columns}"
            )

        out = data.copy()
        for col in self.columns:
            fill = self._fill_values[col]
            if fill is not None:
                out[col] = out[col].fillna(fill)

        return out

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Alias for transform() — allows using Imputer as a callable."""
        return self.transform(data)

    def __repr__(self) -> str:
        return (
            f"Imputer(columns={self.columns}, strategy='{self.strategy}')"
        )


def impute(
    data: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    strategy: str = "median",
    fill_value: Any = None,
) -> Imputer:
    """Fit a missing value imputer on training data.

    Learns fill values from ``data`` only — no leakage. Returns an
    ``Imputer`` that can fill NaN in any DataFrame using the stored values.

    Parameters
    ----------
    data : pd.DataFrame
        Training DataFrame to compute fill values from.
    columns : list[str] or None, default=None
        Columns to impute. If None, imputes all columns that have at
        least one missing value in ``data``.
    strategy : str, default="median"
        Fill strategy:
        - ``"median"``: fill with column median (numeric only).
        - ``"mean"``: fill with column mean (numeric only).
        - ``"mode"``: fill with most frequent value (numeric or categorical).
        - ``"constant"``: fill with ``fill_value`` parameter.
    fill_value : Any, default=None
        Value to use when ``strategy="constant"``.
        Ignored for other strategies.

    Returns
    -------
    Imputer
        Fitted imputer with ``.transform(df)`` method.

    Raises
    ------
    DataError
        If data is not a DataFrame, or columns are not in data.
    ConfigError
        If strategy is not recognized, or median/mean on non-numeric column,
        or strategy="constant" without fill_value.

    Examples
    --------
    >>> imp = ml.impute(s.train, columns=["age", "income"])
    >>> model = ml.fit(imp.transform(s.train), "label", seed=42)
    >>> preds = ml.predict(model, imp.transform(s.valid))

    Impute all columns with missing values automatically:
    >>> imp = ml.impute(s.train)

    Categorical column with mode:
    >>> imp = ml.impute(s.train, columns=["city"], strategy="mode")

    Constant fill:
    >>> imp = ml.impute(s.train, columns=["score"], strategy="constant",
    ...                 fill_value=0.0)

    Save and reload:
    >>> ml.save(imp, "imputer.pyml")
    >>> imp2 = ml.load("imputer.pyml")
    """
    import numpy as np

    from ._compat import to_pandas
    from ._types import ConfigError, DataError

    data = to_pandas(data)

    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"impute() expects DataFrame, got {type(data).__name__}."
        )

    _strategies = {"median", "mean", "mode", "constant"}
    if strategy not in _strategies:
        raise ConfigError(
            f"strategy='{strategy}' not recognized. "
            f"Choose from: {sorted(_strategies)}"
        )

    if strategy == "constant" and fill_value is None:
        raise ConfigError(
            "strategy='constant' requires fill_value=<value>. "
            "Example: ml.impute(data, strategy='constant', fill_value=0)"
        )

    # Resolve columns: auto-detect if not specified
    # For median/mean: only numeric columns with NaN
    # For mode/constant: any column with NaN
    if columns is None:
        _numeric_only = strategy in ("median", "mean")
        columns = [
            c for c in data.columns
            if data[c].isna().any()
            and (not _numeric_only or pd.api.types.is_numeric_dtype(data[c]))
        ]
        if not columns:
            # Nothing to impute — return no-op imputer
            return Imputer(
                columns=[],
                strategy=strategy,
                fill_value=fill_value,
                _fill_values={},
            )
    else:
        missing_cols = [c for c in columns if c not in data.columns]
        if missing_cols:
            raise DataError(
                f"Columns not found in data: {missing_cols}. "
                f"Available columns: {list(data.columns)}"
            )

    fill_values: dict[str, Any] = {}
    for col in columns:
        col_data = data[col].dropna()

        if strategy == "constant":
            fill_values[col] = fill_value
        elif strategy == "mean":
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ConfigError(
                    f"strategy='mean' requires numeric column, "
                    f"but '{col}' is {data[col].dtype}. "
                    "Use strategy='mode' for categorical columns."
                )
            fill_values[col] = float(col_data.mean()) if len(col_data) > 0 else np.nan
        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ConfigError(
                    f"strategy='median' requires numeric column, "
                    f"but '{col}' is {data[col].dtype}. "
                    "Use strategy='mode' for categorical columns."
                )
            fill_values[col] = float(col_data.median()) if len(col_data) > 0 else np.nan
        elif strategy == "mode":
            if len(col_data) == 0:
                fill_values[col] = None
            else:
                mode_val = col_data.mode()
                fill_values[col] = mode_val.iloc[0] if len(mode_val) > 0 else None

    # F4: warn when fill value is NaN/None (all-NaN column → transform is a no-op)
    import math

    for col, fv in fill_values.items():
        if fv is None or (isinstance(fv, float) and math.isnan(fv)):
            import warnings
            warnings.warn(
                f"Column '{col}' is entirely NaN — {strategy} fill value cannot be "
                "computed. transform() will be a no-op for this column. "
                "Check your data for columns with no non-missing values.",
                UserWarning,
                stacklevel=2,
            )

    return Imputer(
        columns=list(columns),
        strategy=strategy,
        fill_value=fill_value,
        _fill_values=fill_values,
    )
