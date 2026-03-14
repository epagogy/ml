"""cv() — cross-validation within split boundaries.

The 8th primitive. split() creates boundaries, cv() creates rotations.

    s = ml.split(data, "survived", seed=42)   # partition
    c = ml.cv(s, folds=5, seed=42)            # resample within dev
    model = ml.fit(c, "survived", seed=42)    # CV fit + refit on dev
    evidence = ml.assess(model, test=s.test)  # final exam (test on split)
"""

from __future__ import annotations

import numpy as np

from ._types import ConfigError, CVResult, DataError, SplitResult


def cv(
    s: SplitResult,
    *,
    folds: int = 5,
    seed: int,
    stratify: bool = True,
) -> CVResult:
    """Create cross-validation folds within a split's dev partition.

    Takes a SplitResult and creates k-fold rotations within ``.dev``
    (train + valid). The test partition is preserved for ``assess()``.

    This is a second-order primitive: it transforms a partition strategy,
    not raw data. The provenance chain is preserved — the model trained
    on these folds shares the same split_id as the test partition.

    Args:
        s: SplitResult from ``ml.split()``.
        folds: Number of CV folds (default 5).
        seed: Random seed (keyword-only).
        stratify: Stratify folds by target class (default True).
            Only applies to classification tasks.

    Returns:
        CVResult with ``.folds`` (rotations within dev) and ``.k``.
        Test stays on the SplitResult — use ``s.test`` for assess().

    Raises:
        ConfigError: If input is not a SplitResult, folds < 2, or
            no target was set in the original split.
        DataError: If dev partition is too small for requested folds.

    Example:
        >>> import ml
        >>> data = ml.dataset("titanic")
        >>> s = ml.split(data, "survived", seed=42)
        >>> c = ml.cv(s, folds=5, seed=42)
        >>> model = ml.fit(c, "survived", seed=42)
        >>> model.scores_
        {'accuracy_mean': 0.82, 'accuracy_std': 0.03, ...}
        >>> evidence = ml.assess(model, test=s.test)
    """
    # Validate input type
    if not isinstance(s, SplitResult):
        raise ConfigError(
            f"cv() expects a SplitResult from ml.split(), got {type(s).__name__}. "
            "Usage: s = ml.split(data, target, seed=42); c = ml.cv(s, folds=5, seed=42)"
        )

    # Block random folds on temporal data — use cv_temporal() instead
    if getattr(s, "_temporal", False):
        raise ConfigError(
            "cv() creates random folds — not valid for temporal data (future leakage). "
            "Use ml.cv_temporal() instead:\n"
            "  s = ml.split_temporal(data, target, time='col')\n"
            "  c = ml.cv_temporal(s, folds=5, embargo=0)"
        )

    # Validate seed
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(
            f"seed must be an integer, got {type(seed).__name__}: {seed!r}. "
            "Example: seed=42"
        )

    # Validate folds
    if folds < 2:
        raise ConfigError("folds must be >= 2")

    # Get dev data (train + valid)
    dev = s.dev
    target = s._target if hasattr(s, "_target") else None

    if target is None:
        raise ConfigError(
            "cv() requires a target column. "
            "Split with target: s = ml.split(data, 'target', seed=42)"
        )

    n_rows = len(dev)
    if folds > n_rows:
        raise DataError(
            f"Cannot create {folds} folds from {n_rows} dev rows. "
            f"Use folds={max(2, n_rows // 2)} or fewer."
        )

    # Guard against empty folds from stratified CV: folds must not exceed
    # the minority class count, otherwise round-robin allocation leaves
    # some folds with 0 validation rows.
    if stratify and target in dev.columns:
        from .split import _detect_task
        if _detect_task(dev[target]) == "classification":
            min_class_n = dev[target].value_counts().min()
            if folds > min_class_n:
                raise DataError(
                    f"Cannot create {folds} stratified folds: minority class has "
                    f"only {min_class_n} samples. Use folds={max(2, min_class_n)} "
                    f"or fewer, or set stratify=False."
                )

    # Import fold generators from split module
    from .split import _kfold, _stratified_kfold

    # Detect task for stratification
    do_stratify = stratify and target in dev.columns
    if do_stratify:
        from .split import _detect_task
        detected_task = _detect_task(dev[target])
        do_stratify = detected_task == "classification"

    # Create fold indices
    if do_stratify:
        fold_list = list(_stratified_kfold(dev[target].values, k=folds, seed=seed))
    else:
        fold_list = list(_kfold(len(dev), k=folds, seed=seed))

    # Build CVResult — folds on dev only, no test (test stays on SplitResult)
    result = CVResult(
        _data=dev,
        folds=fold_list,
        k=folds,
        target=target,
    )

    return result


def cv_temporal(
    s: SplitResult,
    *,
    folds: int = 5,
    embargo: int = 0,
    window: str = "expanding",
    window_size: int | None = None,
    groups: str | None = None,
) -> CVResult:
    """Create temporal cross-validation folds within a split's dev partition.

    Expanding or sliding window CV, respecting chronological order.
    No future leakage — each fold's training data precedes its validation data.

    When ``groups=`` is set, no group (patient, store, sensor) appears in
    both training and validation within any fold. Handles panel data —
    entities tracked over time.

    Args:
        s: SplitResult from ``ml.split_temporal()``.
        folds: Number of CV folds (default 5).
        embargo: Number of rows to skip between train and valid (gap to prevent
            temporal leakage from autocorrelation). Default 0.
        window: ``"expanding"`` (all prior data) or ``"sliding"`` (fixed window).
        window_size: Required when ``window="sliding"``. Number of rows in
            each training window.
        groups: Column name for group non-overlap (optional). When set,
            rows from the same group are kept together — prevents leakage
            from repeated measurements (patients, stores, devices).

    Returns:
        CVResult with ``.folds`` and ``._temporal=True``.
        Test stays on the SplitResult — use ``s.test`` for assess().

    Raises:
        ConfigError: If input is not a SplitResult, folds < 2, or
            sliding window requested without window_size.
        DataError: If dev partition is too small for requested folds.

    Example:
        >>> s = ml.split_temporal(data, "price", time="date")
        >>> c = ml.cv_temporal(s, folds=5, embargo=10)
        >>> model = ml.fit(c, "price", seed=42)
        >>> evidence = ml.assess(model, test=s.test)

        Panel data (entities over time):

        >>> s = ml.split_temporal(data, "mortality", time="admission_date")
        >>> c = ml.cv_temporal(s, folds=5, embargo=48, groups="patient_id")
    """
    if not isinstance(s, SplitResult):
        raise ConfigError(
            f"cv_temporal() expects a SplitResult, got {type(s).__name__}. "
            "Usage: s = ml.split_temporal(data, target, time='col'); "
            "c = ml.cv_temporal(s, folds=5)"
        )

    if folds < 2:
        raise ConfigError("folds must be >= 2")

    if window not in ("expanding", "sliding"):
        raise ConfigError(
            f"window='{window}' not valid. Choose from: ['expanding', 'sliding']"
        )

    if window == "sliding" and window_size is None:
        raise ConfigError(
            "window_size= is required when window='sliding'. "
            "Example: cv_temporal(s, folds=5, window='sliding', window_size=100)"
        )

    dev = s.dev
    target = s._target if hasattr(s, "_target") else None

    if target is None:
        raise ConfigError(
            "cv_temporal() requires a target column. "
            "Split with target: s = ml.split_temporal(data, 'target', time='col')"
        )

    # Validate groups column
    if groups is not None and groups not in dev.columns:
        raise ConfigError(
            f"groups='{groups}' not found in dev data. "
            f"Available columns: {list(dev.columns[:10])}"
        )

    if embargo < 0:
        raise ConfigError(
            f"embargo must be >= 0, got {embargo}. "
            "Negative embargo would create temporal leakage."
        )

    n = len(dev)
    if folds >= n:
        raise DataError(
            f"Cannot create {folds} temporal CV folds from {n} dev rows. "
            f"Use folds={min(5, n - 1)} or fewer."
        )

    # Build expanding or sliding window folds
    fold_size = n // (folds + 1)
    if fold_size < 1:
        raise DataError(
            f"Too many folds ({folds}) for {n} dev rows. "
            f"Use folds={max(2, n // 2 - 1)} or fewer."
        )

    fold_list = []
    for i in range(folds):
        if window == "sliding" and window_size is not None:
            train_start = max(0, (i + 1) * fold_size - window_size)
        else:
            train_start = 0

        train_end = (i + 1) * fold_size
        valid_start = train_end + embargo

        if i == folds - 1:
            valid_end = n
        else:
            valid_end = (i + 2) * fold_size

        if valid_start >= valid_end:
            continue  # embargo consumed all validation data

        train_idx = np.arange(train_start, train_end)
        valid_idx = np.arange(valid_start, valid_end)

        # Group non-overlap: remove rows from train whose group appears in valid
        if groups is not None:
            valid_groups = set(dev.iloc[valid_idx][groups].unique())
            train_mask = ~dev.iloc[train_idx][groups].isin(valid_groups)
            train_idx = train_idx[train_mask.values]

        fold_list.append((train_idx, valid_idx))

    if not fold_list:
        raise DataError(
            f"Embargo ({embargo}) is too large — no validation data remains. "
            f"Reduce embargo or use fewer folds."
        )

    result = CVResult(
        _data=dev,
        folds=fold_list,
        k=len(fold_list),
        target=target,
    )
    result._temporal = True

    return result


def cv_group(
    s: SplitResult,
    *,
    folds: int = 5,
    groups: str,
    seed: int,
    stratify: bool = True,
) -> CVResult:
    """Create group-aware cross-validation folds within a split's dev partition.

    No group appears in both train and valid within any fold — prevents
    leakage from repeated measurements (patients, subjects, stores, etc.).

    Args:
        s: SplitResult from ``ml.split_group()``.
        folds: Number of CV folds (default 5).
        groups: Column name containing group identifiers.
        seed: Random seed (keyword-only).
        stratify: Stratify groups by target class (default True).

    Returns:
        CVResult with ``.folds`` and ``.k``.
        Test stays on the SplitResult — use ``s.test`` for assess().

    Raises:
        ConfigError: If input is not a SplitResult, folds < 2, or
            groups column not found.
        DataError: If fewer groups than folds.

    Example:
        >>> s = ml.split_group(data, "outcome", groups="patient_id", seed=42)
        >>> c = ml.cv_group(s, folds=5, groups="patient_id", seed=42)
        >>> model = ml.fit(c, "outcome", seed=42)
        >>> evidence = ml.assess(model, test=s.test)
    """
    if not isinstance(s, SplitResult):
        raise ConfigError(
            f"cv_group() expects a SplitResult, got {type(s).__name__}. "
            "Usage: s = ml.split_group(data, target, groups='col', seed=42); "
            "c = ml.cv_group(s, folds=5, groups='col', seed=42)"
        )

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(
            f"seed must be an integer, got {type(seed).__name__}: {seed!r}."
        )

    if folds < 2:
        raise ConfigError("folds must be >= 2")

    dev = s.dev
    target = s._target if hasattr(s, "_target") else None

    if target is None:
        raise ConfigError(
            "cv_group() requires a target column."
        )

    if groups not in dev.columns:
        raise ConfigError(
            f"groups column '{groups}' not found in dev data. "
            f"Available columns: {list(dev.columns[:10])}"
        )

    group_values = dev[groups].values
    n_groups = len(np.unique(group_values))

    if folds > n_groups:
        raise DataError(
            f"Cannot create {folds} folds from {n_groups} groups. "
            f"Use folds<={n_groups}."
        )

    # Import group kfold from split module
    from .split import _group_kfold

    fold_list = list(_group_kfold(group_values, k=folds, seed=seed))

    result = CVResult(
        _data=dev,
        folds=fold_list,
        k=folds,
        target=target,
    )

    return result
