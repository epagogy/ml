"""discretize() — continuous feature binning.

Converts continuous features into discrete bins. Learned from training data,
applied consistently to new data (test/valid). No leakage.

Methods:
    "quantile" — equal-frequency bins (same number of samples per bin).
    "uniform"  — equal-width bins.

Usage:
    >>> binner = ml.discretize(s.train, columns=["age", "income"])
    >>> train_binned = binner.transform(s.train)
    >>> valid_binned = binner.transform(s.valid)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Binner:
    """Fitted feature binner.

    Attributes
    ----------
    columns : list[str]
        Numeric columns that were fitted.
    method : str
        Binning method: "quantile" or "uniform".
    n_bins : int
        Number of bins.
    bin_edges : dict[str, np.ndarray]
        Learned bin edges per column (interior edges, for inspection).
    """

    columns: list[str]
    method: str
    n_bins: int
    bin_edges: dict[str, np.ndarray] = field(repr=False)
    _fill_values: dict[str, float] = field(default_factory=dict, repr=False)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted binning to data.

        Replaces each fitted column with a bin index (0..n_bins-1).
        Values outside the training range clip to the nearest bin.
        NaN values are imputed with the training median before binning.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the fitted columns.

        Returns
        -------
        pd.DataFrame
            New DataFrame with integer bin-index columns.

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
        missing = [c for c in self.columns if c not in data.columns]
        if missing:
            raise DataError(f"Columns missing from data: {missing}.")

        out = data.copy()
        for col in self.columns:
            interior = self.bin_edges[col]  # interior edges (without -inf/+inf)
            fill = self._fill_values.get(col, 0.0)
            vals = data[col].fillna(fill).values.astype(float)
            # searchsorted: gives index in 0..len(interior) → clip to 0..n_bins-1
            idx = np.searchsorted(interior, vals, side="right")
            idx = np.clip(idx, 0, self.n_bins - 1)
            out[col] = idx.astype(int)
        return out

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Alias for transform()."""
        return self.transform(data)

    def __repr__(self) -> str:
        return (
            f"Binner(columns={self.columns!r}, method='{self.method}', "
            f"n_bins={self.n_bins})"
        )


def discretize(
    data: pd.DataFrame,
    columns: list[str],
    *,
    method: str = "quantile",
    n_bins: int = 5,
    seed: int | None = None,
) -> Binner:
    """Fit a feature binner on training data.

    Learns bin boundaries from training data only — no leakage.
    At transform time, values outside the training range clip to the nearest bin.

    Parameters
    ----------
    data : pd.DataFrame
        Training DataFrame containing the numeric columns.
    columns : list[str]
        Names of numeric columns to bin.
    method : str, default="quantile"
        Binning method:
        - ``"quantile"``: equal-frequency bins — each bin has ~equal samples.
          Useful for skewed distributions.
        - ``"uniform"``: equal-width bins. Preserves absolute scale.
    n_bins : int, default=5
        Number of bins (must be >= 2).
    seed : int, optional
        Only used for method="kmeans" (not yet implemented). For "quantile" and
        "uniform" methods, binning is deterministic and seed has no effect.
        Passing seed= with these methods emits a UserWarning.

    Returns
    -------
    Binner
        Fitted binner with ``.transform(df)`` method.

    Raises
    ------
    DataError
        If data is not a DataFrame, columns is empty, or any column is missing.
    ConfigError
        If method is not recognized or n_bins < 2.

    Examples
    --------
    >>> binner = ml.discretize(s.train, columns=["age", "income"])
    >>> model = ml.fit(binner.transform(s.train), "label", seed=42)
    >>> preds = ml.predict(model, binner.transform(s.valid))
    """
    from ._compat import to_pandas
    from ._types import ConfigError, DataError

    data = to_pandas(data)

    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"discretize() expects DataFrame, got {type(data).__name__}."
        )
    if not columns:
        raise DataError("discretize() requires at least one column.")

    _methods = {"quantile", "uniform"}
    if method not in _methods:
        raise ConfigError(
            f"method='{method}' not recognized. Choose from: {sorted(_methods)}"
        )
    if n_bins < 2:
        raise ConfigError(f"n_bins must be >= 2, got {n_bins}.")

    if seed is not None and method in ("quantile", "uniform"):
        import warnings
        warnings.warn(
            f"discretize(seed={seed!r}) has no effect for method={method!r}. "            "The 'quantile' and 'uniform' methods are deterministic and do not use a seed. "            "Pass seed=None or use method='kmeans' (when available) for stochastic binning.",
            UserWarning,
            stacklevel=2,
        )

    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise DataError(f"Columns not found in data: {missing}.")

    bin_edges: dict[str, np.ndarray] = {}
    fill_values: dict[str, float] = {}

    for col in columns:
        vals = data[col].dropna().astype(float).values

        if len(vals) == 0:
            # All NaN: use trivial edges
            interior = np.array([0.0])
            fill_values[col] = 0.0
            bin_edges[col] = interior
            continue

        fill_values[col] = float(np.median(vals))

        if method == "quantile":
            quantiles = np.linspace(0, 100, n_bins + 1)
            all_edges = np.percentile(vals, quantiles)
        else:  # uniform
            low, high = vals.min(), vals.max()
            if low == high:
                low, high = low - 0.5, high + 0.5
            all_edges = np.linspace(low, high, n_bins + 1)

        # Keep only the interior edges (indices 1..n_bins-1) — drop outer bounds.
        # This way searchsorted handles out-of-range values naturally:
        # values below first interior edge → bin 0
        # values above last interior edge → bin n_bins-1 (after clip)
        interior_edges = np.unique(all_edges[1:-1])  # deduplicate
        bin_edges[col] = interior_edges

    return Binner(
        columns=list(columns),
        method=method,
        n_bins=n_bins,
        bin_edges=bin_edges,
        _fill_values=fill_values,
    )
