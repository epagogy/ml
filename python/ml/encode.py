"""encode() — categorical feature encoding.

Fits an encoder on training categories, transforms at predict time.
No leakage: unseen categories handled gracefully (mapped to unknown).

Methods:
    "onehot"    — dummy columns {col}_{category}. No leakage risk.
    "ordinal"   — integer codes 0..N-1. Unseen → -1.
    "target"    — cross-validated smoothed target encoding (A1).
                  Micci-Barreca empirical Bayes smoothing prevents leakage.
    "frequency" — category frequency encoding (A9).

Usage:
    >>> enc = ml.encode(s.train, columns=["city", "brand"])
    >>> model = ml.fit(enc.transform(s.train), "label", seed=42)
    >>> preds = ml.predict(model, enc.transform(s.valid))

    # Target encoding (cross-validated, no leakage):
    >>> cv = ml.split(data, "churn", folds=5, seed=42)
    >>> enc = ml.encode(s.train, columns=["city"], method="target",
    ...                 target="churn", cv=cv, seed=42)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Encoder:
    """Fitted categorical encoder.

    For ``method="onehot"``: drops the original column and adds
    ``{col}_{category}`` boolean columns for each known category.

    For ``method="ordinal"``: replaces the column with integer codes
    (0 to n_categories-1). Unseen categories mapped to -1.

    For ``method="target"``: replaces each column with cross-validated
    smoothed target encoding. Binary: one float column per categorical.
    Multiclass (C classes): C-1 columns named ``{col}_te_{k}`` per categorical.
    Unseen categories mapped to global mean (fully smoothed).

    For ``method="frequency"``: replaces column with category frequency (0..1).
    Unseen categories mapped to 0.

    Attributes
    ----------
    columns : list[str]
        Categorical columns that were fitted.
    method : str
        Encoding method: "onehot", "ordinal", "target", "frequency", or "woe".
    _categories : dict[str, list]
        Known categories per column (from training data).
    """

    columns: list[str]
    method: str
    _categories: dict[str, list] = field(repr=False)
    # Target encoding fields (only populated when method="target")
    _te_mapping: dict[str, Any] = field(default_factory=dict, repr=False)
    _te_global: dict[str, Any] = field(default_factory=dict, repr=False)
    _te_classes: list | None = field(default=None, repr=False)
    _fold_indices: list | None = field(default=None, repr=False)
    # Frequency encoding fields (only populated when method="frequency")
    _freq_mapping: dict[str, dict] = field(default_factory=dict, repr=False)
    # WOE encoding fields (only populated when method="woe")
    _woe_mapping: dict[str, dict] = field(default_factory=dict, repr=False)
    _iv_scores: dict[str, float] = field(default_factory=dict, repr=False)
    # Datetime encoding fields (only populated when method="datetime")
    _dt_include_hour: dict[str, bool] = field(default_factory=dict, repr=False)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoding to data.

        For one-hot: drops original column, adds {col}_{cat} dummies.
        For ordinal: replaces column with integer codes. Unseen → -1.
        For target: replaces column with smoothed target statistics.
            Unseen categories → global mean.
        For frequency: replaces column with training frequency (0..1).
            Unseen categories → 0.0.
        Other columns passed through unchanged.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the fitted categorical columns.

        Returns
        -------
        pd.DataFrame
            New DataFrame with encoded columns.

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
            raise DataError(
                f"Columns missing from data: {missing}. "
                f"Expected columns: {self.columns}"
            )

        out = data.copy()

        if self.method == "onehot":
            for col in self.columns:
                cats = self._categories[col]
                col_vals = out[col].astype(str)
                for cat in cats:
                    out[f"{col}_{cat}"] = (col_vals == str(cat)).astype(int)
                out = out.drop(columns=[col])

        elif self.method == "ordinal":
            for col in self.columns:
                cats = self._categories[col]
                cat_to_idx = {str(c): i for i, c in enumerate(cats)}
                out[col] = (
                    out[col].astype(str)
                    .map(cat_to_idx)
                    .fillna(-1)
                    .astype(int)
                )

        elif self.method == "target":
            for col in self.columns:
                mapping = self._te_mapping.get(col, {})
                global_val = self._te_global.get(col, 0.0)
                classes = self._te_classes

                if classes is None or len(classes) <= 2:
                    # Binary / regression: one column, replace in place
                    out[col] = (
                        out[col].astype(str)
                        .map(mapping)
                        .fillna(global_val if not isinstance(global_val, list) else global_val[0])
                        .astype(float)
                    )
                else:
                    # Multiclass: K-1 columns, drop original
                    n_out = len(classes) - 1
                    for k in range(n_out):
                        col_name = f"{col}_te_{k}"
                        gv = global_val[k] if isinstance(global_val, list) else 0.0
                        out[col_name] = (
                            out[col].astype(str)
                            .map({cat: vals[k] for cat, vals in mapping.items()})
                            .fillna(gv)
                            .astype(float)
                        )
                    out = out.drop(columns=[col])

        elif self.method == "frequency":
            for col in self.columns:
                freq = self._freq_mapping.get(col, {})
                out[col] = (
                    out[col].astype(str)
                    .map(freq)
                    .fillna(0.0)
                    .astype(float)
                )

        elif self.method == "woe":
            for col in self.columns:
                woe = self._woe_mapping.get(col, {})
                out[col] = (
                    out[col].astype(str)
                    .map(woe)
                    .fillna(0.0)  # unseen categories → 0.0 (neutral evidence)
                    .astype(float)
                )

        elif self.method == "datetime":
            for col in self.columns:
                include_hour = self._dt_include_hour.get(col, False)
                dt = pd.to_datetime(out[col], errors="coerce")
                out[f"{col}_year"] = dt.dt.year.astype("Int64").astype(float)
                out[f"{col}_month"] = dt.dt.month.astype("Int64").astype(float)
                out[f"{col}_day"] = dt.dt.day.astype("Int64").astype(float)
                out[f"{col}_dayofweek"] = dt.dt.dayofweek.astype("Int64").astype(float)
                out[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
                if include_hour:
                    out[f"{col}_hour"] = dt.dt.hour.astype("Int64").astype(float)
                out = out.drop(columns=[col])

        return out

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Alias for transform() — allows using Encoder as a callable."""
        return self.transform(data)

    def __repr__(self) -> str:
        col_summary = ", ".join(self.columns)
        return f"Encoder(columns=[{col_summary}], method='{self.method}')"


def encode(
    data: pd.DataFrame,
    *,
    columns: list[str],
    method: str = "onehot",
    target: str | None = None,
    smoothing: float | str = "auto",
    folds: int = 5,
    cv: Any = None,
    seed: int | None = None,
) -> Encoder:
    """Fit a categorical encoder on training data.

    Learns the category vocabulary from ``data`` only — no leakage.
    At transform time, unseen categories are handled gracefully:
    - one-hot: unseen category → all zeros
    - ordinal: unseen category → -1
    - target: unseen category → global mean (smoothed to prior)
    - frequency: unseen category → 0.0

    Parameters
    ----------
    data : pd.DataFrame
        Training DataFrame containing the categorical columns.
    columns : list[str]
        Names of categorical columns to encode.
    method : str, default="onehot"
        Encoding method:
        - ``"onehot"``: creates ``{col}_{category}`` boolean columns.
        - ``"ordinal"``: integer codes 0..N-1. Unseen → -1.
        - ``"target"``: cross-validated Micci-Barreca smoothed target
          encoding. Requires ``target=`` and ``seed=``. Leakage-free via
          K-fold CV. Unseen categories → global mean.
        - ``"frequency"``: replaces category with its training frequency
          (proportion of rows). Unseen → 0.0.
    target : str, optional
        Target column name. Required for method="target".
    smoothing : float or "auto", default="auto"
        Smoothing strength for target encoding.
        - ``"auto"``: Micci-Barreca empirical Bayes (k=5, f=1):
          ``weight = 1 / (1 + exp(-(count - 5) / 1))``
        - ``float``: additive smoothing: ``weight = count / (count + s)``
          Larger values → stronger regularization toward global mean.
    folds : int, default=5
        Number of CV folds for target encoding. Ignored when ``cv=`` provided.
    cv : CVResult, optional
        CVResult from ``ml.split(folds=N, seed=S)``. When provided, target
        encoding uses the same fold boundaries as downstream ``ml.fit(cv)``.
        This ensures consistent data splits and prevents fold-boundary leakage.
        If absent, internal random folds are used (warns about misalignment).
    seed : int, optional
        Random seed for internal fold generation. Required for method="target"
        when ``cv=`` is not provided.

    Returns
    -------
    Encoder
        Fitted encoder with ``.transform(df)`` method.

    Raises
    ------
    DataError
        If data is not a DataFrame, columns list is empty, or any column
        is missing from data.
    ConfigError
        If method is not recognized, or target/seed missing for method="target".

    Examples
    --------
    >>> enc = ml.encode(s.train, columns=["city", "brand"])
    >>> model = ml.fit(enc.transform(s.train), "label", seed=42)
    >>> preds = ml.predict(model, enc.transform(s.valid))

    Target encoding (fold-aligned, no leakage):
    >>> cv = ml.split(data, "churn", folds=5, seed=42)
    >>> s = ml.split(data, "churn", seed=42)
    >>> enc = ml.encode(s.train, columns=["city"], method="target",
    ...                 target="churn", cv=cv, seed=42)
    >>> model = ml.fit(enc.transform(s.train), "churn", seed=42)

    Frequency encoding:
    >>> enc = ml.encode(s.train, columns=["country"], method="frequency")
    """
    from ._compat import to_pandas
    from ._types import ConfigError, CVResult, DataError

    data = to_pandas(data)
    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"encode() expects DataFrame, got {type(data).__name__}."
        )
    if not columns:
        raise DataError(
            "encode() requires at least one column. "
            "Pass columns=['col1', 'col2', ...]."
        )

    # Normalize method aliases: 'one-hot', 'one_hot', 'OneHot' → 'onehot'
    method = method.replace("-", "").replace("_", "").lower()

    _methods = {"onehot", "ordinal", "target", "frequency", "woe", "datetime"}
    if method not in _methods:
        raise ConfigError(
            f"method='{method}' not recognized. "
            f"Choose from: {sorted(_methods)}"
        )

    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise DataError(
            f"Columns not found in data: {missing}. "
            f"Available columns: {list(data.columns)}"
        )

    categories: dict[str, list] = {}
    for col in columns:
        known = sorted(data[col].dropna().astype(str).unique().tolist())
        categories[col] = known

    if method == "woe":
        if target is None:
            raise ConfigError(
                "encode(method='woe') requires target=. "
                "Example: ml.encode(data, columns=['cat'], method='woe', target='label')"
            )
        if target not in data.columns:
            raise DataError(
                f"target='{target}' not found in data. "
                f"Available: {list(data.columns)}"
            )
        woe_mapping, iv_scores = _fit_woe_encoding(data, columns, target)
        return Encoder(
            columns=list(columns),
            method=method,
            _categories=categories,
            _woe_mapping=woe_mapping,
            _iv_scores=iv_scores,
        )

    if method == "target":
        if target is None:
            raise ConfigError(
                "encode(method='target') requires target=. "
                "Example: ml.encode(data, columns=['city'], method='target', "
                "target='churn', seed=42)"
            )
        if target not in data.columns:
            raise DataError(
                f"target='{target}' not found in data. "
                f"Available: {list(data.columns)}"
            )

        # cv= alignment: extract fold indices from CVResult
        fold_splits: list[tuple] | None = None
        stored_fold_indices: list | None = None
        if cv is not None:
            if not isinstance(cv, CVResult):
                raise ConfigError(
                    f"cv= must be a CVResult from ml.split(folds=N, seed=S), "
                    f"got {type(cv).__name__}."
                )
            fold_splits = cv.folds
            stored_fold_indices = [
                (fold_train.index.tolist(), fold_valid.index.tolist())
                for fold_train, fold_valid in cv.folds
            ]
        else:
            if seed is None:
                raise ConfigError(
                    "encode(method='target') requires seed= when cv= is not provided. "
                    "Example: ml.encode(data, columns=['city'], method='target', "
                    "target='churn', seed=42)"
                )
            import warnings
            warnings.warn(
                "encode(method='target') without cv=: using internal random folds. "
                "For fold-aligned target encoding, pass cv=ml.split(data, target, "
                f"folds={folds}, seed={seed}) to match downstream fit() splits.",
                UserWarning, stacklevel=2,
            )
            from .split import _kfold, _stratified_kfold
            y = data[target]
            try:
                fold_splits = list(_stratified_kfold(y.values, k=folds, seed=seed))
            except Exception:
                fold_splits = list(_kfold(len(data), k=folds, seed=seed))
            stored_fold_indices = [
                (train_idx.tolist(), val_idx.tolist())
                for train_idx, val_idx in fold_splits
            ]

        te_mapping, te_global, te_classes = _fit_target_encoding(
            data=data,
            columns=columns,
            target=target,
            smoothing=smoothing,
            fold_splits=fold_splits,
        )

        return Encoder(
            columns=list(columns),
            method=method,
            _categories=categories,
            _te_mapping=te_mapping,
            _te_global=te_global,
            _te_classes=te_classes,
            _fold_indices=stored_fold_indices,
        )

    if method == "frequency":
        freq_mapping: dict[str, dict] = {}
        n = len(data)
        for col in columns:
            counts = data[col].astype(str).value_counts()
            freq_mapping[col] = (counts / n).to_dict()
        return Encoder(
            columns=list(columns),
            method=method,
            _categories=categories,
            _freq_mapping=freq_mapping,
        )

    if method == "datetime":
        dt_include_hour: dict[str, bool] = {}
        for col in columns:
            # Parse to datetime, then check if any non-midnight hours exist
            dt = pd.to_datetime(data[col], errors="coerce")
            has_hours = bool((dt.dt.hour != 0).any())
            dt_include_hour[col] = has_hours
        return Encoder(
            columns=list(columns),
            method=method,
            _categories={},
            _dt_include_hour=dt_include_hour,
        )

    # onehot / ordinal
    return Encoder(columns=list(columns), method=method, _categories=categories)


# ---------------------------------------------------------------------------
# Target encoding internals
# ---------------------------------------------------------------------------

def _micci_barreca_weight(count: float, k: float = 5.0, f: float = 1.0) -> float:
    """Micci-Barreca empirical Bayes smoothing weight.

    weight = 1 / (1 + exp(-(count - k) / f))
    """
    import math
    try:
        return 1.0 / (1.0 + math.exp(-(count - k) / f))
    except OverflowError:
        return 0.0 if count < k else 1.0


def _smooth(count: float, cat_mean: float, global_mean: float,
            smoothing: float | str) -> float:
    """Blend category mean toward global mean via smoothing."""
    if smoothing == "auto":
        w = _micci_barreca_weight(count)
    else:
        s = float(smoothing)
        w = count / (count + s) if (count + s) > 0 else 0.0
    return w * cat_mean + (1.0 - w) * global_mean


def _fit_target_encoding(
    data: pd.DataFrame,
    columns: list[str],
    target: str,
    smoothing: float | str,
    fold_splits: list[tuple],
) -> tuple[dict, dict, list | None]:
    """Fit target encoding on all data (for transform-time statistics).

    Returns:
        te_mapping: {col: {category_str: encoded_value}}
            For multiclass: {col: {category_str: [v0, v1, ..., vK-2]}}
        te_global: {col: global_mean} or {col: [global_mean_0, ..., global_mean_K-2]}
        te_classes: list of classes (for multiclass), or None (binary/regression)
    """

    y_raw = data[target]
    classes: list | None = None

    # Detect multiclass: only for non-float targets with >2 unique values.
    # Float targets (regression) always use the scalar path even if many unique values.
    if not pd.api.types.is_float_dtype(y_raw):
        unique_vals = sorted(y_raw.dropna().unique().tolist())
        if len(unique_vals) > 2:
            classes = unique_vals  # multiclass categorical

    te_mapping: dict[str, dict] = {}
    te_global: dict[str, Any] = {}

    for col in columns:
        col_data = data[col].astype(str)

        if classes is None or len(classes) <= 2:
            # Binary classification or regression: encode toward P(y=1) or E[y]
            if classes is not None and len(classes) == 2:
                # Binary: encode toward positive class probability
                pos_class = classes[-1]
                y_bin = (y_raw == pos_class).astype(float)
            else:
                # Regression path: float targets convert directly.
                # Guard: if target is string/category (e.g. binary with non-standard
                # values like 'yes'/'no' passed without a cv= fold hint), factorize
                # to numeric labels (0, 1, ...) before computing target statistics.
                if y_raw.dtype == object or hasattr(y_raw.dtype, 'categories'):
                    y_bin = pd.Series(pd.factorize(y_raw)[0], index=y_raw.index, dtype=float)
                else:
                    y_bin = y_raw.astype(float)

            global_mean = float(y_bin.mean())
            te_global[col] = global_mean

            # Full-data statistics for transform-time use
            cat_stats: dict[str, tuple[float, float]] = {}  # cat -> (sum_y, count)
            for cat, y_val in zip(col_data, y_bin):
                if cat not in cat_stats:
                    cat_stats[cat] = (0.0, 0)
                s, c = cat_stats[cat]
                cat_stats[cat] = (s + y_val, c + 1)

            mapping: dict[str, float] = {}
            for cat, (s, c) in cat_stats.items():
                cat_mean = s / c if c > 0 else global_mean
                mapping[cat] = _smooth(c, cat_mean, global_mean, smoothing)

            te_mapping[col] = mapping

        else:
            # Multiclass: K-1 binary one-vs-rest encodings
            n_out = len(classes) - 1
            global_means: list[float] = []
            mappings_per_class: list[dict[str, float]] = []

            for k in range(n_out):
                ck = classes[k]
                y_bin_k = (y_raw == ck).astype(float)
                gm = float(y_bin_k.mean())
                global_means.append(gm)

                cat_stats_k: dict[str, tuple[float, float]] = {}
                for cat, y_val in zip(col_data, y_bin_k):
                    if cat not in cat_stats_k:
                        cat_stats_k[cat] = (0.0, 0)
                    s, c = cat_stats_k[cat]
                    cat_stats_k[cat] = (s + y_val, c + 1)

                map_k: dict[str, float] = {}
                for cat, (s, c) in cat_stats_k.items():
                    cat_mean = s / c if c > 0 else gm
                    map_k[cat] = _smooth(c, cat_mean, gm, smoothing)
                mappings_per_class.append(map_k)

            # Combine: mapping[cat] = [v0, v1, ..., vK-2]
            all_cats = set()
            for mk in mappings_per_class:
                all_cats.update(mk.keys())

            combined: dict[str, list] = {}
            for cat in all_cats:
                combined[cat] = [
                    mappings_per_class[k].get(cat, global_means[k])
                    for k in range(n_out)
                ]

            te_mapping[col] = combined
            te_global[col] = global_means

    return te_mapping, te_global, classes


# ---------------------------------------------------------------------------
# WOE encoding internals
# ---------------------------------------------------------------------------

def _fit_woe_encoding(
    data: pd.DataFrame,
    columns: list[str],
    target: str,
) -> tuple[dict, dict]:
    """Fit WOE (Weight of Evidence) encoding for binary classification.

    WOE(i) = ln(P_Events(i) / P_NonEvents(i))
    IV      = sum_i((P_Events(i) - P_NonEvents(i)) * WOE(i))

    Laplace smoothing (add 0.5 to each bin count) prevents log(0).
    Unseen categories return 0.0 (neutral: no evidence either way).

    Raises
    ------
    ConfigError
        If target is not binary (exactly 2 classes).
    """
    import math

    from ._types import ConfigError

    y = data[target]
    unique_classes = sorted(y.dropna().unique().tolist())

    if len(unique_classes) != 2:
        raise ConfigError(
            f"encode(method='woe') requires a binary target (2 classes), "
            f"got {len(unique_classes)} classes: {unique_classes[:5]}. "
            "WOE encoding is defined only for binary classification."
        )

    pos_class = unique_classes[-1]
    y_bin = (y == pos_class).astype(int)
    total_events = float(y_bin.sum())
    total_non_events = float((1 - y_bin).sum())

    if total_events == 0 or total_non_events == 0:
        raise ConfigError(
            "encode(method='woe') requires both classes to have at least one sample."
        )

    woe_mapping: dict[str, dict] = {}
    iv_scores: dict[str, float] = {}

    for col in columns:
        col_data = data[col].astype(str)
        unique_cats = col_data.unique()
        n_cats = len(unique_cats)

        # Laplace-smoothed totals
        total_ev_s = total_events + 0.5 * n_cats
        total_nev_s = total_non_events + 0.5 * n_cats

        woe_map: dict[str, float] = {}
        iv = 0.0

        for cat in unique_cats:
            mask = col_data == cat
            n_ev = float(y_bin[mask].sum()) + 0.5
            n_nev = float((1 - y_bin[mask]).sum()) + 0.5

            dist_ev = n_ev / total_ev_s
            dist_nev = n_nev / total_nev_s

            woe = math.log(dist_ev / dist_nev)
            woe_map[cat] = woe
            iv += (dist_ev - dist_nev) * woe

        woe_mapping[col] = woe_map
        iv_scores[col] = max(0.0, iv)  # IV is always non-negative in theory

    return woe_mapping, iv_scores
