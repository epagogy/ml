"""cluster_features() — group correlated features for honest permutation importance.

Features with |r| >= threshold are placed in the same cluster.
Use with ml.explain(feature_groups=...) to permute correlated features together,
avoiding the underestimation bias of individual-feature permutation.

Usage:
    >>> groups = ml.cluster_features(s.train.drop(columns=["target"]))
    >>> imp = ml.explain(model, data=s.valid, method="permutation",
    ...                  feature_groups=groups, seed=42)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_features(
    data: pd.DataFrame,
    *,
    threshold: float = 0.95,
    seed: int | None = None,
) -> list[list[str]]:
    """Group correlated features into clusters using hierarchical clustering.

    Features with absolute Pearson |r| >= threshold end up in the same cluster.
    Pass the result to ``ml.explain(feature_groups=groups)`` for grouped
    permutation importance — permuting correlated features together avoids the
    underestimation bias of individual feature permutation.

    Parameters
    ----------
    data : pd.DataFrame
        Feature DataFrame. Should NOT include the target column.
    threshold : float, default=0.95
        Correlation threshold. Features with |r| >= threshold are grouped together.
        Lower values create fewer, larger groups.
    seed : int, optional
        Reserved for future stochastic clustering methods. Current hierarchical
        clustering is deterministic; passing seed= emits a UserWarning.

    Returns
    -------
    list[list[str]]
        List of feature groups. Each group is a list of column names.
        Singleton groups (length 1) contain features with no correlated partner.
        Non-numeric columns are returned as singleton groups.

    Raises
    ------
    DataError
        If data is not a DataFrame.
    ConfigError
        If threshold is not in (0, 1].

    Examples
    --------
    >>> groups = ml.cluster_features(s.train.drop(columns=["target"]))
    >>> print(groups)
    [['age', 'income'], ['region'], ['score']]
    >>> imp = ml.explain(model, data=s.valid, method="permutation",
    ...                  feature_groups=groups, seed=42)
    """
    from ._compat import to_pandas
    from ._types import ConfigError, DataError

    data = to_pandas(data)

    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"cluster_features() expects DataFrame, got {type(data).__name__}."
        )
    if not (0 < threshold <= 1.0):
        raise ConfigError(
            f"threshold must be in (0, 1], got {threshold}."
        )

    if seed is not None:
        import warnings
        warnings.warn(
            f"cluster_features(seed={seed!r}) has no effect. "            "Current hierarchical clustering is deterministic and does not use a seed. "            "Pass seed=None to suppress this warning.",
            UserWarning,
            stacklevel=2,
        )

    # Separate numeric and non-numeric
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in data.columns if c not in numeric_cols]

    # Non-numeric columns → singleton groups
    singleton_groups: list[list[str]] = [[c] for c in non_numeric]

    if len(numeric_cols) == 0:
        return singleton_groups
    if len(numeric_cols) == 1:
        return singleton_groups + [[numeric_cols[0]]]

    # Absolute correlation matrix
    corr = data[numeric_cols].corr().abs().fillna(0.0)

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        # Distance = 1 - |r|, clipped for numerical safety
        dist = (1.0 - corr.values).clip(0, None)
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False).clip(0, None)

        Z = linkage(condensed, method="complete")
        labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    except ImportError:
        # scipy not available: greedy correlation-based grouping
        labels = _greedy_cluster(corr.values, threshold)

    n_clusters = int(labels.max())
    numeric_groups: list[list[str]] = []
    for k in range(1, n_clusters + 1):
        group = [numeric_cols[i] for i in range(len(numeric_cols)) if labels[i] == k]
        if group:
            numeric_groups.append(group)

    return singleton_groups + numeric_groups


def _greedy_cluster(corr_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Fallback greedy clustering when scipy is unavailable."""
    n = corr_matrix.shape[0]
    labels = np.zeros(n, dtype=int)
    next_label = 1
    for i in range(n):
        if labels[i] == 0:
            labels[i] = next_label
            for j in range(i + 1, n):
                if labels[j] == 0 and corr_matrix[i, j] >= threshold:
                    labels[j] = next_label
            next_label += 1
    return labels
