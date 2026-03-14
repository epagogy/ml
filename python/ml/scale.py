"""scale() — numeric feature scaling.

Fits a scaler on training data, replaces numeric columns in-place.
No leakage: fitted on train, applied to any split.

Methods: "standard" (z-score), "minmax" (0–1), "robust" (median/IQR)

Usage:
    >>> scl = ml.scale(s.train, columns=["age", "income"])
    >>> model = ml.fit(scl.transform(s.train), "label", seed=42)
    >>> preds = ml.predict(model, scl.transform(s.valid))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Scaler:
    """Fitted numeric scaler.

    Replaces each column with its scaled value at transform time.
    Column names are preserved — values change, names don't.

    Attributes
    ----------
    columns : list[str]
        Numeric columns that were fitted.
    method : str
        Scaling method: "standard", "minmax", or "robust".
    _scalers : dict[str, Any]
        Internal: fitted sklearn scaler per column.
    """

    columns: list[str]
    method: str
    _scalers: dict[str, Any] = field(repr=False)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaling to data.

        Replaces each fitted column with its scaled value.
        Other columns are passed through unchanged.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the fitted columns.

        Returns
        -------
        pd.DataFrame
            New DataFrame with scaled columns (same column names).

        Raises
        ------
        DataError
            If any fitted column is missing from data.
        """
        import numpy as np

        from ._types import DataError
        if not isinstance(data, pd.DataFrame):
            raise DataError(
                f"transform() expects DataFrame, got {type(data).__name__}."
            )

        missing = [c for c in self.columns if c not in data.columns]
        if missing:
            raise DataError(
                f"Columns missing from data: {missing}. "
                f"Expected columns: {self.columns}"
            )

        out = data.copy()
        for col in self.columns:
            scaler = self._scalers[col]
            vals = data[[col]].values.astype(float)
            scaled = scaler.transform(vals).flatten()
            out[col] = scaled.astype(np.float64)

        return out

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Alias for transform() — allows using Scaler as a callable."""
        return self.transform(data)

    def __repr__(self) -> str:
        return f"Scaler(columns={self.columns}, method='{self.method}')"


def scale(
    data: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    method: str = "standard",
) -> Scaler:
    """Fit a numeric scaler on training data.

    Fits one scaler per column on ``data`` only — no leakage into
    validation or test sets. Returns a ``Scaler`` that transforms any
    DataFrame by replacing the fitted columns with scaled values.

    Parameters
    ----------
    data : pd.DataFrame
        Training DataFrame containing the numeric columns to scale.
    columns : list[str] or None, default=None
        Names of numeric columns to scale.
        If None, auto-detects all numeric (non-boolean) columns in data.
        Mirrors the behavior of ml.impute().
    method : str, default="standard"
        Scaling method:
        - ``"standard"``: z-score normalization (mean=0, std=1).
          Best for linear models, neural nets, PCA.
        - ``"minmax"``: scales to [0, 1] range.
          Best when you need bounded output.
        - ``"robust"``: median/IQR scaling, ignores outliers.
          Best for data with heavy-tailed distributions.

    Returns
    -------
    Scaler
        Fitted scaler with ``.transform(df)`` method.

    Raises
    ------
    DataError
        If data is not a DataFrame, columns list is empty, or any column
        is missing or non-numeric.
    ConfigError
        If method is not recognized.

    Examples
    --------
    >>> scl = ml.scale(s.train, columns=["age", "income"])
    >>> model = ml.fit(scl.transform(s.train), "label", seed=42)
    >>> preds = ml.predict(model, scl.transform(s.valid))

    Robust scaling for outlier-heavy data:
    >>> scl = ml.scale(s.train, columns=["salary"], method="robust")

    Save and reload:
    >>> ml.save(scl, "scaler.pyml")
    >>> scl2 = ml.load("scaler.pyml")
    """
    import warnings

    from ._compat import to_pandas
    from ._transforms import MinMaxScaler, RobustScaler, StandardScaler
    from ._types import ConfigError, DataError

    data = to_pandas(data)

    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"scale() expects DataFrame, got {type(data).__name__}."
        )
    # F2: auto-detect numeric columns when columns=None (mirrors impute() behavior)
    if columns is None:
        columns = [
            c for c in data.columns
            if pd.api.types.is_numeric_dtype(data[c])
            and not pd.api.types.is_bool_dtype(data[c])
        ]
        if not columns:
            raise DataError(
                "scale() found no numeric columns to scale. "
                "Pass columns=['col1', 'col2', ...] to specify columns explicitly."
            )
    elif not columns:
        raise DataError(
            "scale() requires at least one column. "
            "Pass columns=['col1', 'col2', ...]."
        )

    _methods = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
    if method not in _methods:
        raise ConfigError(
            f"method='{method}' not recognized. "
            f"Choose from: {sorted(_methods.keys())}"
        )

    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise DataError(
            f"Columns not found in data: {missing}. "
            f"Available columns: {list(data.columns)}"
        )

    # F3: non-numeric columns → DataError (not raw ValueError from astype(float))
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(data[c])]
    if non_numeric:
        raise DataError(
            f"scale() requires numeric columns, but {non_numeric} have non-numeric dtype. "
            "Use ml.encode() first to convert categorical columns to numbers."
        )

    # F2: NaN in training data → UserWarning (NaN will propagate to output)
    nan_cols = [c for c in columns if data[c].isna().any()]
    if nan_cols:
        warnings.warn(
            f"Columns {nan_cols} contain NaN values. "
            "scale() will propagate NaN to output. "
            "Run ml.impute() before ml.scale() to fill missing values first.",
            UserWarning,
            stacklevel=2,
        )

    scalers: dict[str, Any] = {}
    for col in columns:
        col_data = data[[col]].values.astype(float)
        scaler = _methods[method]()
        scaler.fit(col_data)
        scalers[col] = scaler

    return Scaler(columns=list(columns), method=method, _scalers=scalers)
