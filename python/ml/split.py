"""Split data into train/valid/test or cross-validation folds.

Three-way split is the default (Hastie ESL Ch.7).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._types import ConfigError, CVResult, DataError, SplitResult

# ---------------------------------------------------------------------------
# Cross-language deterministic RNG (Rust PCG-XSH-RR with numpy fallback)
# ---------------------------------------------------------------------------


def _ml_shuffle(n: int, seed: int) -> np.ndarray:
    """Deterministic shuffle of [0, 1, ..., n-1].

    Uses Rust PCG-XSH-RR when available (exact cross-language parity with R).
    Falls back to numpy RandomState when Rust backend is not installed.
    """
    try:
        import ml_py
        perm = ml_py.shuffle(n, seed)
        return np.array(perm, dtype=np.intp)
    except (ImportError, AttributeError, OverflowError):
        warnings.warn(
            "Rust backend unavailable for shuffle — falling back to numpy RNG. "
            "Cross-language parity not guaranteed.",
            UserWarning,
            stacklevel=3,
        )
        rng = np.random.RandomState(seed % (2**32))
        perm = np.arange(n)
        rng.shuffle(perm)
        return perm


def _ml_partition_sizes(n: int, ratio: tuple) -> tuple[int, int, int]:
    """Canonical partition sizes: (n_train, n_valid, n_test).

    Uses round(n * ratio) — identical formula across Python and R.
    """
    try:
        import ml_py
        return ml_py.partition_sizes(n, ratio[0], ratio[1], ratio[2])
    except (ImportError, AttributeError):
        n_train = round(n * ratio[0])
        n_valid = round(n * ratio[1])
        n_test = n - n_train - n_valid
        return (n_train, n_valid, n_test)


# ---------------------------------------------------------------------------
# Pure-numpy splitting helpers (no sklearn dependency)
# ---------------------------------------------------------------------------


def _train_test_split(*arrays, test_size=0.2, random_state=42, stratify=None,
                      train_size=None):
    """Shuffle-split arrays into train/test. Drop-in for sklearn train_test_split.

    Accepts DataFrames, Series, and numpy arrays. Returns same types.
    """
    n = len(arrays[0])
    if train_size is not None and test_size is not None:
        # train_size takes precedence when both given (sklearn compat)
        test_size = 1.0 - train_size
    elif train_size is not None:
        test_size = 1.0 - train_size

    n_test = max(1, int(round(n * test_size)))
    n_train = n - n_test

    rng = np.random.RandomState(random_state)

    if stratify is not None:
        # Stratified split: proportional allocation per class
        y_strat = np.asarray(stratify)
        classes, class_indices = np.unique(y_strat, return_inverse=True)
        class_groups = {c: np.where(class_indices == i)[0] for i, c in enumerate(classes)}

        train_idx = []
        test_idx = []
        for cls in classes:
            idx = class_groups[cls].copy()
            rng.shuffle(idx)
            n_cls_test = max(1, int(round(len(idx) * test_size)))
            # Ensure at least 1 in train if possible
            if n_cls_test >= len(idx) and len(idx) > 1:
                n_cls_test = len(idx) - 1
            test_idx.extend(idx[:n_cls_test])
            train_idx.extend(idx[n_cls_test:])

        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        # Shuffle within each split for randomness
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
    else:
        indices = np.arange(n)
        rng.shuffle(indices)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

    result = []
    for arr in arrays:
        if isinstance(arr, (pd.DataFrame, pd.Series)):
            result.append(arr.iloc[train_idx])
            result.append(arr.iloc[test_idx])
        else:
            a = np.asarray(arr)
            result.append(a[train_idx])
            result.append(a[test_idx])
    return result


def _kfold(n, k=5, seed=42):
    """Yield (train_idx, val_idx) for k-fold cross-validation.

    Args:
        n: Number of samples.
        k: Number of folds.
        seed: Random seed for shuffling.

    Yields:
        (train_indices, val_indices) numpy arrays.
    """
    indices = _ml_shuffle(n, seed)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1
    current = 0
    folds = []
    for size in fold_sizes:
        folds.append(indices[current:current + size])
        current += size
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx


def _stratified_kfold(y, k=5, seed=42):
    """Yield (train_idx, val_idx) for stratified k-fold cross-validation.

    Ensures each fold has approximately the same class distribution.

    Args:
        y: Target array (labels).
        k: Number of folds.
        seed: Random seed.

    Yields:
        (train_indices, val_indices) numpy arrays.
    """
    y = np.asarray(y)
    rng = np.random.RandomState(seed)

    classes, y_indices = np.unique(y, return_inverse=True)

    # Collect per-class indices, shuffled
    class_indices = []
    for cls_i in range(len(classes)):
        idx = np.where(y_indices == cls_i)[0]
        rng.shuffle(idx)
        class_indices.append(idx)

    # Allocate indices to folds round-robin per class
    fold_indices = [[] for _ in range(k)]
    for cls_idx in class_indices:
        for i, idx in enumerate(cls_idx):
            fold_indices[i % k].append(idx)

    fold_indices = [np.array(f) for f in fold_indices]

    for i in range(k):
        val_idx = fold_indices[i]
        train_idx = np.concatenate([fold_indices[j] for j in range(k) if j != i])
        yield train_idx, val_idx


def _group_kfold(groups, k=5, seed=42):
    """Yield (train_idx, val_idx) for group k-fold CV.

    No group appears in both train and validation within a fold.
    Groups are assigned to folds round-robin after shuffling.
    """
    unique_groups = np.array(sorted(set(groups)))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)

    # Round-robin assign groups to folds
    group_to_fold = {}
    for i, g in enumerate(unique_groups):
        group_to_fold[g] = i % k

    groups_arr = np.asarray(groups)
    for fold_i in range(k):
        val_mask = np.array([group_to_fold[g] == fold_i for g in groups_arr])
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        yield train_idx, val_idx


def _stratified_group_kfold(groups, y, k=5, seed=42):
    """Yield (train_idx, val_idx) for stratified group k-fold CV.

    No group in both train and validation. Class distribution in each fold
    approximates the global distribution via greedy bin-packing.
    """
    y = np.asarray(y)
    groups_arr = np.asarray(groups)
    rng = np.random.RandomState(seed)

    # Compute global class distribution
    classes = np.unique(y)
    n_total = len(y)
    global_dist = np.array([np.sum(y == c) / n_total for c in classes])

    # Per-group class counts
    unique_groups = np.array(sorted(set(groups_arr)))
    rng.shuffle(unique_groups)

    group_class_counts = {}
    for g in unique_groups:
        mask = groups_arr == g
        group_class_counts[g] = np.array([np.sum(y[mask] == c) for c in classes])

    # Greedy assignment: assign each group to the fold whose class distribution
    # is furthest from the global target (i.e., needs it most).
    fold_counts = np.zeros((k, len(classes)), dtype=np.float64)
    fold_totals = np.zeros(k, dtype=np.float64)
    group_to_fold = {}

    for g in unique_groups:
        gc = group_class_counts[g]
        # Find fold with worst class distribution match
        best_fold = 0
        best_deficit = -np.inf
        for f in range(k):
            total = fold_totals[f] + gc.sum()
            if total == 0:
                # Empty fold — assign here
                best_fold = f
                break
            current_dist = (fold_counts[f] + gc) / total
            deficit = np.sum(np.abs(global_dist - current_dist))
            # We want to minimize deficit, so pick fold with smallest post-add deficit
            # But for greedy: pick the fold that benefits most = currently worst match
            if f == 0 or deficit < best_deficit:
                best_deficit = deficit
                best_fold = f
        group_to_fold[g] = best_fold
        fold_counts[best_fold] += gc
        fold_totals[best_fold] += gc.sum()

    for fold_i in range(k):
        val_mask = np.array([group_to_fold[g] == fold_i for g in groups_arr])
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        yield train_idx, val_idx


def _group_shuffle_split(groups, train_size=0.6, seed=42):
    """Single shuffle split that respects group boundaries.

    Returns (train_idx, test_idx). No group in both partitions.
    """
    unique_groups = np.array(sorted(set(groups)))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)

    n_train_groups = max(1, int(round(len(unique_groups) * train_size)))
    train_groups = set(unique_groups[:n_train_groups])

    groups_arr = np.asarray(groups)
    train_mask = np.array([g in train_groups for g in groups_arr])
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(~train_mask)[0]
    return train_idx, test_idx


def _repeated_kfold(n, k=5, repeats=3, seed=42):
    """Yield (train_idx, val_idx) for repeated k-fold CV."""
    for r in range(repeats):
        yield from _kfold(n, k=k, seed=seed + r)


def _repeated_stratified_kfold(y, k=5, repeats=3, seed=42):
    """Yield (train_idx, val_idx) for repeated stratified k-fold CV."""
    for r in range(repeats):
        yield from _stratified_kfold(y, k=k, seed=seed + r)


def split(
    data: pd.DataFrame,
    target: str | None = None,
    *,
    seed: int = 42,
    ratio: tuple = (0.6, 0.2, 0.2),
    folds: int | None = None,
    splitter=None,
    stratify: bool = True,
    task: str = "auto",
    time: str | None = None,
    groups: str | None = None,
    repeats: int | None = None,
    embargo: int | None = None,
    window: str = "expanding",
    window_size: int | None = None,
) -> SplitResult | CVResult:
    """Split data into train/valid/test or cross-validation folds.

    Three-way split is the default (60/20/20). Automatically stratifies for classification.

    .. note:: **Cross-language reproducibility**
       When the Rust backend (ml-py) is installed, the same seed produces
       identical non-stratified splits across Python and R (PCG-XSH-RR
       deterministic RNG). Stratified splits use per-language RNG and may differ.
       Without the Rust backend, falls back to numpy — same correctness,
       different permutation.

    Args:
        data: DataFrame to split
        target: Target column name (enables stratification + size guidance)
        seed: Random seed for reproducibility (default: 42).
            Ignored for temporal splits (time=) — those are deterministic.
        ratio: (train, valid, test) fractions — must sum to 1.0
        folds: Set to k for CV (e.g., folds=5). Ignores ratio.
        splitter: Custom CV splitter. Must have .split(X, y)
            or .split(X) returning iterable of (train_idx, val_idx). When provided,
            folds= and stratify= are ignored. Returns CVResult.
        stratify: Auto-stratify when target is classification (default: True)
        task: "auto", "classification", or "regression" — override heuristic
        groups: Column name for group-aware splitting. No group appears in
            both train and valid/test. Essential for medical data (patients),
            time series (subjects), or any repeated-measures design. With
            folds=, uses GroupKFold. With holdout, uses GroupShuffleSplit.
        time: Column name for temporal (chronological) splitting. When set,
            data is sorted by this column and split by position — no future
            leakage. Works with datetime, int, or float columns. Ignores
            stratify= (chronological order takes precedence). With folds=,
            produces expanding-window temporal CV.

    Returns:
        SplitResult with .train, .valid, .test, .dev (three-way split)
        OR CVResult with .folds, .k (cross-validation)

    Raises:
        DataError: If data is invalid (empty, too few rows, duplicate columns)
        ConfigError: If ratio doesn't sum to 1.0, or splitter invalid

    Example:
        >>> s = ml.split(data, "churn", seed=42)
        >>> s.train.shape, s.valid.shape, s.test.shape
        ((3000, 5), (1000, 5), (1000, 5))

        >>> # Temporal split — chronological, no future leakage
        >>> s = ml.split(data, "target", time="timestamp")
        >>> # Time column dropped after ordering (used for sorting only)

        >>> # Temporal CV — expanding window
        >>> cv = ml.split(data, "target", time="timestamp", folds=5)
    """
    # Auto-convert Polars/other DataFrames to pandas
    from ._compat import to_pandas
    data = to_pandas(data)

    # Validate data type first — must be a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"split() expects a DataFrame, got {type(data).__name__}. "
            "Use: ml.split(df, 'target', seed=42)"
        )

    # Validate seed — must be a non-negative integer in u64 range
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(
            f"seed must be an integer, got {type(seed).__name__}: {seed!r}. "
            "Example: seed=42"
        )
    if seed < 0:
        raise ConfigError(
            f"seed must be non-negative, got {seed}. Example: seed=42"
        )
    if seed >= 2**64:
        raise ConfigError(
            f"seed must be < 2^64, got {seed}. Example: seed=42"
        )

    if data.shape[0] == 0:
        raise DataError("Cannot split empty data (0 rows)")

    if data.shape[0] == 1:
        raise DataError("Cannot split 1 row. Need at least 4.")

    # Check for duplicate column names
    if data.columns.duplicated().any():
        dupes = data.columns[data.columns.duplicated()].tolist()
        raise DataError(f"Duplicate column names: {dupes}")

    # Validate target if provided
    if target is not None:
        if target not in data.columns:
            available = data.columns.tolist()
            raise DataError(
                f"target='{target}' not found in data. Available columns: {available}"
            )

        # Drop rows with NaN target
        n_before = len(data)
        data = data.dropna(subset=[target]).reset_index(drop=True)
        n_after = len(data)
        if n_after < n_before:
            warnings.warn(
                f"Dropped {n_before - n_after} rows with NaN target.",
                UserWarning,
                stacklevel=2
            )

        if n_after == 0:
            raise DataError(f"Target column '{target}' is entirely NaN")

        # Check minimum rows for holdout split (after NaN drop)
        if data.shape[0] < 4 and folds is None:
            raise DataError(
                f"Cannot split {data.shape[0]} rows into train/valid/test. "
                "Need at least 4 rows. Consider using folds=2 for very small data."
            )

        # Check for single unique value — can't train on this
        if data[target].nunique() == 1:
            only_val = data[target].iloc[0]
            raise DataError(
                f"Target '{target}' has only 1 unique value ({only_val}). "
                "Need at least 2 classes for classification or variance for regression."
            )

        # Warn about class imbalance for classification targets
        detected = task if task != "auto" else _detect_task(data[target])
        if detected == "classification":
            class_counts = data[target].value_counts()
            majority = class_counts.iloc[0]
            minority = class_counts.iloc[-1]
            if minority > 0 and majority / minority > 10:
                imbalance_ratio = majority / minority
                warnings.warn(
                    f"Class imbalance detected: {majority}:{minority} "
                    f"({imbalance_ratio:.0f}:1 ratio). Minority class "
                    f"'{class_counts.index[-1]}' has only "
                    f"{minority / len(data):.1%} of rows. "
                    "Consider: stratify=True (default), "
                    "oversampling, or class weights.",
                    UserWarning,
                    stacklevel=2
                )

    # Mutual exclusion checks (column validation delegated to specializations)
    if groups is not None and time is not None:
        raise ConfigError("Cannot use groups= and time= together.")
    if groups is not None and splitter is not None:
        raise ConfigError("Cannot use groups= and splitter= together.")
    if time is not None and splitter is not None:
        raise ConfigError("Cannot use time= and splitter= together.")

    # folds= and repeats= removed from split() — use ml.cv() instead
    if repeats is not None:
        raise ConfigError(
            "split(repeats=) is removed. Use ml.cv() instead:\n"
            "  s = ml.split(data, target, seed=42)\n"
            "  c = ml.cv(s, folds=5, seed=42)"
        )

    if window not in ("expanding", "sliding"):
        raise ConfigError(
            f"window='{window}' not valid. Choose from: ['expanding', 'sliding']"
        )
    if window == "sliding" and window_size is None:
        raise ConfigError("window='sliding' requires window_size= to be set.")
    if window_size is not None and window_size < 1:
        raise ConfigError(f"window_size must be >= 1, got {window_size}.")

    if embargo is not None:
        if time is None:
            raise ConfigError("embargo= requires time= to be set.")
        if embargo < 0:
            raise ConfigError(f"embargo must be >= 0, got {embargo}.")

    # Route to domain specializations, custom splitter, or three-way
    if splitter is not None:
        return _split_custom(data, target, splitter, task)
    elif time is not None:
        if folds is not None:
            raise ConfigError(
                "split(time=, folds=) is removed. Use ml.cv_temporal() instead:\n"
                "  s = ml.split(data, target, time='col')\n"
                "  c = ml.cv_temporal(s, folds=5, embargo=0)\n"
                "This ensures a test holdout is always preserved."
            )
        return split_temporal(
            data, target, time=time, ratio=ratio, task=task,
        )
    elif groups is not None:
        if folds is not None:
            raise ConfigError(
                "split(groups=, folds=) is removed. Use ml.cv_group() instead:\n"
                "  s = ml.split(data, target, groups='col', seed=42)\n"
                "  c = ml.cv_group(s, folds=5, groups='col', seed=42)\n"
                "This ensures a test holdout is always preserved."
            )
        return split_group(
            data, target, groups=groups, seed=seed, ratio=ratio,
            stratify=stratify, task=task,
        )
    elif folds is not None:
        raise ConfigError(
            "split(folds=) is removed. Use ml.cv() instead:\n"
            "  s = ml.split(data, target, seed=42)\n"
            "  c = ml.cv(s, folds=5, seed=42)\n"
            "This ensures a test holdout is always preserved."
        )
    else:
        return _split_three_way(data, target, ratio, seed, stratify, task)


def _split_three_way(
    data: pd.DataFrame,
    target: str | None,
    ratio: tuple,
    seed: int,
    stratify: bool,
    task: str,
) -> SplitResult:
    """Three-way split: train/valid/test."""
    # Validate ratio
    if len(ratio) not in (2, 3):
        raise ConfigError(
            f"ratio must be 2-tuple or 3-tuple, got {len(ratio)}-tuple: {ratio}"
        )

    # Validate each value is positive (zero-valued ratios crash sklearn)
    for i, val in enumerate(ratio):
        if val <= 0:
            raise ConfigError(
                f"ratio values must be positive, got {val} at position {i}. "
                "Try ratio=(0.6, 0.2, 0.2)."
            )

    ratio_sum = sum(ratio)
    if not (0.999 <= ratio_sum <= 1.001):
        raise ConfigError(
            f"ratio must sum to 1.0, got {ratio_sum:.3f}. Try ratio=(0.6, 0.2, 0.2)."
        )

    # Detect task if auto
    detected_task = task
    if task == "auto" and target is not None:
        detected_task = _detect_task(data[target])

    # Determine stratification
    do_stratify = stratify and detected_task == "classification" and target is not None

    # Three-way split
    if len(ratio) == 3:
        train_frac, valid_frac, test_frac = ratio

        if do_stratify:
            # Stratified: single-pass per-class Rust PCG shuffle
            # (cross-language parity with R)
            y_arr = np.asarray(data[target])
            classes = np.sort(np.unique(y_arr))
            train_parts, valid_parts, test_parts = [], [], []
            rare_class = False

            for ci, cls in enumerate(classes):
                cl_idx = np.where(y_arr == cls)[0]
                n_cl = len(cl_idx)
                n_train_cl = max(1, round(n_cl * train_frac))
                n_valid_cl = max(1, round(n_cl * valid_frac))
                n_test_cl = n_cl - n_train_cl - n_valid_cl

                if n_test_cl < 1 or n_train_cl < 1:
                    rare_class = True
                    break

                # Deterministic per-class shuffle via Rust PCG
                class_seed = seed + ci
                perm = _ml_shuffle(n_cl, class_seed)
                shuffled = cl_idx[perm]

                train_parts.append(shuffled[:n_train_cl])
                valid_parts.append(
                    shuffled[n_train_cl:n_train_cl + n_valid_cl]
                )
                test_parts.append(shuffled[n_train_cl + n_valid_cl:])

            if rare_class:
                warnings.warn(
                    "Some classes have too few members for stratified split. "
                    "Falling back to non-stratified. Consider folds=5 or "
                    "filtering rare classes.",
                    UserWarning,
                    stacklevel=3
                )
                do_stratify = False
            else:
                train = data.iloc[np.concatenate(train_parts)]
                valid = data.iloc[np.concatenate(valid_parts)]
                test = data.iloc[np.concatenate(test_parts)]

        if not do_stratify:
            # Non-stratified: single-pass Rust PCG shuffle for cross-language parity
            n = len(data)
            perm = _ml_shuffle(n, seed)
            n_train, n_valid, n_test = _ml_partition_sizes(n, ratio)
            if n_test <= 0:
                raise DataError(
                    "Test partition is empty. Increase n or adjust ratio."
                )
            train_idx = perm[:n_train]
            valid_idx = perm[n_train:n_train + n_valid]
            test_idx = perm[n_train + n_valid:]
            train = data.iloc[train_idx]
            valid = data.iloc[valid_idx]
            test = data.iloc[test_idx]

        # Guard empty partitions (catches stratified n=4 binary edge case)
        for name, partition in [("train", train), ("valid", valid), ("test", test)]:
            if len(partition) == 0:
                raise DataError(
                    f"{name.title()} partition is empty. "
                    "Increase n or adjust ratio."
                )

        # Warn about small partitions (30 rows minimum for stable estimates)
        for name, partition in [("train", train), ("valid", valid), ("test", test)]:
            if len(partition) < 30:
                warnings.warn(
                    f"Partition '{name}' has only {len(partition)} rows. "
                    "Results may be unreliable. Consider folds=5 for small datasets.",
                    UserWarning,
                    stacklevel=3
                )

        # Warn about stratification issues with many classes
        if do_stratify and target is not None:
            for name, partition in [("train", train), ("valid", valid), ("test", test)]:
                class_counts = partition[target].value_counts()
                small_classes = (class_counts < 2).sum()
                if small_classes > 0:
                    warnings.warn(
                        f"{small_classes} classes have <2 samples in {name} partition. "
                        "Stratification may be unreliable. Consider folds=5.",
                        UserWarning,
                        stacklevel=3
                    )

        train_out = train.reset_index(drop=True)
        valid_out = valid.reset_index(drop=True)
        test_out = test.reset_index(drop=True)
        train_out.attrs["_ml_partition"] = "train"
        valid_out.attrs["_ml_partition"] = "valid"
        test_out.attrs["_ml_partition"] = "test"
        train_out.attrs["_ml_target"] = target
        valid_out.attrs["_ml_target"] = target
        test_out.attrs["_ml_target"] = target

        # Layer 1: Register partitions in provenance registry
        from ._provenance import (
            _fingerprint,
            _registry,
            audit_log,
            new_split_id,
            register_partition,
        )
        sid = new_split_id()
        register_partition(train_out, "train", sid)
        register_partition(valid_out, "valid", sid)
        register_partition(test_out, "test", sid)
        audit_log("split", _fingerprint(data), split_id=sid)

        result = SplitResult(
            train=train_out,
            valid=valid_out,
            test=test_out,
            _target=target,
            _task=detected_task,
            _seed=seed,
        )
        # Store receipt in registry for build_provenance() to find
        _registry.register_split(sid)
        return result

    else:  # Two-way split
        train_frac, test_frac = ratio

        warnings.warn(
            "2-element ratio creates an empty validation set. "
            "Use ratio=(0.6, 0.2, 0.2) for train/valid/test.",
            UserWarning,
            stacklevel=3,
        )

        if do_stratify:
            # Single-pass per-class Rust PCG shuffle (cross-language parity)
            y_arr = np.asarray(data[target])
            classes = np.sort(np.unique(y_arr))
            train_parts, test_parts = [], []
            for ci, cls in enumerate(classes):
                cl_idx = np.where(y_arr == cls)[0]
                n_cl = len(cl_idx)
                n_train_cl = max(1, round(n_cl * train_frac))
                class_seed = seed + ci
                perm = _ml_shuffle(n_cl, class_seed)
                shuffled = cl_idx[perm]
                train_parts.append(shuffled[:n_train_cl])
                test_parts.append(shuffled[n_train_cl:])
            train = data.iloc[np.concatenate(train_parts)]
            test = data.iloc[np.concatenate(test_parts)]
        else:
            train, test = _train_test_split(
                data,
                train_size=train_frac,
                random_state=seed
            )

        # Create SplitResult with valid = empty DataFrame
        # .dev will just return train
        empty_valid = pd.DataFrame(columns=data.columns)

        train_out = train.reset_index(drop=True)
        test_out = test.reset_index(drop=True)
        train_out.attrs["_ml_partition"] = "train"
        empty_valid.attrs["_ml_partition"] = "valid"
        test_out.attrs["_ml_partition"] = "test"
        train_out.attrs["_ml_target"] = target
        empty_valid.attrs["_ml_target"] = target
        test_out.attrs["_ml_target"] = target

        # Layer 1: Register partitions in provenance registry
        from ._provenance import (
            _fingerprint,
            _registry,
            audit_log,
            new_split_id,
            register_partition,
        )
        sid = new_split_id()
        register_partition(train_out, "train", sid)
        register_partition(test_out, "test", sid)
        audit_log("split", _fingerprint(data), split_id=sid)

        result = SplitResult(
            train=train_out,
            valid=empty_valid,
            test=test_out,
            _target=target,
            _task=detected_task,
            _seed=seed,
        )
        _registry.register_split(sid)
        return result


def _split_temporal(
    data: pd.DataFrame,
    target: str | None,
    time_col: str,
    ratio: tuple,
    task: str,
) -> SplitResult:
    """Chronological three-way split. Pure pandas, no sklearn.

    Note: deterministic (sort by time column). No seed parameter — randomness
    plays no role in temporal splits.
    """
    # Validate ratio (same checks as _split_three_way)
    if len(ratio) not in (2, 3):
        raise ConfigError(
            f"ratio must be 2-tuple or 3-tuple, got {len(ratio)}-tuple: {ratio}"
        )
    for i, val in enumerate(ratio):
        if val <= 0:
            raise ConfigError(
                f"ratio values must be positive, got {val} at position {i}. "
                "Try ratio=(0.6, 0.2, 0.2)."
            )
    ratio_sum = sum(ratio)
    if not (0.999 <= ratio_sum <= 1.001):
        raise ConfigError(
            f"ratio must sum to 1.0, got {ratio_sum:.3f}. "
            "Try ratio=(0.6, 0.2, 0.2)."
        )

    # Detect task
    detected_task = task
    if task == "auto" and target is not None:
        detected_task = _detect_task(data[target])

    # Sort chronologically (stable preserves ties' relative order)
    sorted_data = data.sort_values(by=time_col, kind="stable").reset_index(
        drop=True
    )
    n = len(sorted_data)

    # Compute cutpoints
    if len(ratio) == 3:
        train_frac, valid_frac, _ = ratio
    else:
        train_frac = ratio[0]
        valid_frac = 1.0 - ratio[0]

    # Canonical partition sizes — same formula across Python and R.
    n_train, n_valid, _ = _ml_partition_sizes(n, (train_frac, valid_frac, 1.0 - train_frac - valid_frac))
    train_end = n_train
    valid_end = n_train + n_valid

    # Drop time column — it served its ordering purpose.
    # Matches temporal CV path which also drops time_col from _data.
    sorted_data = sorted_data.drop(columns=[time_col])

    train = sorted_data.iloc[:train_end].reset_index(drop=True)
    valid = sorted_data.iloc[train_end:valid_end].reset_index(drop=True)
    test = sorted_data.iloc[valid_end:].reset_index(drop=True)

    # Guard empty test partition (only for 3-tuple ratio — 2-tuple intentionally has no test)
    if len(test) == 0 and len(ratio) == 3:
        raise DataError("Test partition is empty. Increase n or adjust ratio.")

    # Warn on small partitions
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        if len(part) < 30 and len(part) > 0:
            warnings.warn(
                f"Temporal split {name} has only {len(part)} rows. "
                "Consider using more data or adjusting ratio=.",
                UserWarning,
                stacklevel=3,
            )

    train.attrs["_ml_partition"] = "train"
    valid.attrs["_ml_partition"] = "valid"
    test.attrs["_ml_partition"] = "test"
    train.attrs["_ml_target"] = target
    valid.attrs["_ml_target"] = target
    test.attrs["_ml_target"] = target

    # Layer 1: Register partitions in provenance registry
    from ._provenance import _fingerprint, _registry, audit_log, new_split_id, register_partition
    sid = new_split_id()
    register_partition(train, "train", sid)
    register_partition(valid, "valid", sid)
    register_partition(test, "test", sid)
    audit_log("split", _fingerprint(sorted_data), split_id=sid)

    result = SplitResult(
        train=train,
        valid=valid,
        test=test,
        _target=target,
        _task=detected_task,
        _seed=None,  # temporal splits are deterministic (no seed used)
        _temporal=True,
    )
    _registry.register_split(sid)
    return result


def _split_temporal_cv(
    data: pd.DataFrame,
    target: str | None,
    time_col: str,
    folds: int,
    task: str,
    *,
    embargo: int | None = None,
    window: str = "expanding",
    window_size: int | None = None,
) -> CVResult:
    """Expanding-window or sliding-window temporal CV. Hand-rolled, no sklearn."""
    # Validate folds
    if folds < 2:
        raise ConfigError("folds must be >= 2")

    # Sort chronologically
    sorted_data = data.sort_values(by=time_col, kind="stable").reset_index(
        drop=True
    )
    n = len(sorted_data)

    if folds >= n:
        raise DataError(
            f"Cannot create {folds} temporal CV folds from {n} rows. "
            f"Use folds={min(5, n - 1)} or fewer."
        )

    # Expanding window: fold_size = minimum chunk for validation
    fold_size = n // (folds + 1)
    if fold_size < 1:
        raise DataError(
            f"Too many folds ({folds}) for {n} rows. "
            "Each fold needs at least 1 validation row. "
            f"Use folds={max(2, n // 2 - 1)} or fewer."
        )

    # Build window folds
    # Expanding: train grows each fold. Sliding: fixed window_size training.
    fold_list = []
    embargo_n = embargo if embargo is not None else 0
    for i in range(folds):
        valid_start = fold_size * (i + 1) + embargo_n
        if i == folds - 1:
            valid_end = n  # last fold captures remainder
        else:
            valid_end = valid_start + fold_size
        # Stop if there's no valid data left
        if valid_start >= n:
            break

        if window == "sliding" and window_size is not None:
            # Fixed-size training window ending before embargo
            train_end = fold_size * (i + 1)
            train_start = max(0, train_end - window_size)
            train_idx = list(range(train_start, train_end))
        else:
            # Expanding window: all data before valid_start (minus embargo)
            train_end = fold_size * (i + 1)
            train_idx = list(range(train_end))

        valid_idx = list(range(valid_start, valid_end))
        if not train_idx or not valid_idx:
            continue
        fold_list.append((train_idx, valid_idx))

    # Update k to actual number of folds produced (may be < requested)
    actual_folds = len(fold_list)
    if actual_folds < folds:
        warnings.warn(
            f"Only {actual_folds} temporal CV folds produced "
            f"(requested {folds}). With {n} rows and expanding "
            "windows, later folds have no validation data.",
            UserWarning,
            stacklevel=3,
        )

    # Drop time column from features — it was used for ordering only.
    # Keeping it would leak datetime dtypes into sklearn (DTypePromotionError).
    sorted_data = sorted_data.drop(columns=[time_col])

    return CVResult(
        _data=sorted_data,
        folds=fold_list,
        k=actual_folds,
        target=target,
        _temporal=True,
    )


def _split_cv(
    data: pd.DataFrame,
    target: str | None,
    folds: int,
    seed: int,
    stratify: bool,
    task: str,
) -> CVResult:
    """Cross-validation split (indices only)."""
    n_rows = len(data)

    # Validate folds
    if folds > n_rows:
        raise DataError(
            f"Cannot create {folds} folds from {n_rows} rows. "
            f"Use folds={max(2, n_rows // 2)} or fewer."
        )

    if folds < 2:
        raise ConfigError("folds must be >= 2")

    # Detect task if auto
    detected_task = task
    if task == "auto" and target is not None:
        detected_task = _detect_task(data[target])

    # Determine stratification
    do_stratify = stratify and detected_task == "classification" and target is not None

    # Create fold splits
    if do_stratify:
        fold_list = list(_stratified_kfold(data[target].values, k=folds, seed=seed))
    else:
        fold_list = list(_kfold(len(data), k=folds, seed=seed))

    return CVResult(
        _data=data,
        folds=fold_list,
        k=folds,
        target=target
    )


def _split_group_cv(
    data: pd.DataFrame,
    target: str | None,
    groups: str,
    folds: int,
    task: str,
    *,
    stratify: bool = False,
    seed: int = 42,
) -> CVResult:
    """Group cross-validation — no group leaks across folds.

    When stratify=True and target is classification, uses StratifiedGroupKFold.
    """
    n_groups = data[groups].nunique()
    if folds > n_groups:
        raise DataError(
            f"Cannot create {folds} folds from {n_groups} groups. "
            f"Use folds<={n_groups}."
        )

    if folds < 2:
        raise ConfigError("folds must be >= 2")

    group_values = data[groups]

    # A6: StratifiedGroupKFold when stratify=True + classification target
    detected_task = task
    if task == "auto" and target is not None:
        detected_task = _detect_task(data[target])

    use_stratified = (
        stratify
        and detected_task == "classification"
        and target is not None
    )

    if use_stratified:
        fold_list = list(_stratified_group_kfold(
            group_values, data[target].values, k=folds, seed=seed,
        ))
    else:
        fold_list = list(_group_kfold(group_values, k=folds, seed=seed))

    return CVResult(
        _data=data,
        folds=fold_list,
        k=folds,
        target=target,
    )


def _split_group_holdout(
    data: pd.DataFrame,
    target: str | None,
    groups: str,
    ratio: tuple,
    seed: int,
    task: str,
) -> SplitResult:
    """Group-aware holdout split — no group leaks across partitions."""
    # Validate ratio
    if len(ratio) not in (2, 3):
        raise ConfigError(
            f"ratio must be 2-tuple or 3-tuple, got {len(ratio)}-tuple: {ratio}"
        )
    for i, val in enumerate(ratio):
        if val <= 0:
            raise ConfigError(
                f"ratio values must be positive, got {val} at position {i}. "
                "Try ratio=(0.6, 0.2, 0.2)."
            )
    ratio_sum = sum(ratio)
    if not (0.999 <= ratio_sum <= 1.001):
        raise ConfigError(
            f"ratio must sum to 1.0, got {ratio_sum:.3f}. Try ratio=(0.6, 0.2, 0.2)."
        )

    group_values = data[groups]
    n_groups = group_values.nunique()
    if n_groups < 3:
        raise DataError(
            f"Need at least 3 unique groups for 3-way split, got {n_groups}. "
            "Use folds= for group cross-validation with fewer groups."
        )

    if len(ratio) == 3:
        train_frac, valid_frac, test_frac = ratio
    else:
        train_frac, test_frac = ratio
        valid_frac = 0.0

    # First split: train vs temp (valid+test)
    train_idx, temp_idx = _group_shuffle_split(
        group_values, train_size=train_frac, seed=seed,
    )
    train = data.iloc[train_idx]
    temp = data.iloc[temp_idx]

    if valid_frac > 0:
        # Second split: valid vs test from temp
        temp_groups = group_values.iloc[temp_idx]
        relative_valid = valid_frac / (valid_frac + test_frac)
        valid_sub, test_sub = _group_shuffle_split(
            temp_groups, train_size=relative_valid, seed=seed + 1,
        )
        valid = temp.iloc[valid_sub]
        test = temp.iloc[test_sub]
    else:
        valid = temp.iloc[:0]  # empty
        test = temp

    # Detect task
    detected_task = task
    if task == "auto" and target is not None:
        detected_task = _detect_task(data[target])

    # Guard empty partitions
    if len(test) == 0:
        raise DataError("Test partition is empty. Increase n or adjust ratio.")

    # Size warnings
    for name, partition in [("train", train), ("valid", valid), ("test", test)]:
        if len(partition) < 30 and len(partition) > 0:
            warnings.warn(
                f"Partition '{name}' has only {len(partition)} rows. "
                "Results may be unreliable. Consider folds=5 for small datasets.",
                UserWarning,
                stacklevel=3,
            )

    train_out = train.reset_index(drop=True)
    valid_out = valid.reset_index(drop=True)
    test_out = test.reset_index(drop=True)
    train_out.attrs["_ml_partition"] = "train"
    valid_out.attrs["_ml_partition"] = "valid"
    test_out.attrs["_ml_partition"] = "test"
    train_out.attrs["_ml_target"] = target
    valid_out.attrs["_ml_target"] = target
    test_out.attrs["_ml_target"] = target

    # Layer 1: Register partitions in provenance registry
    from ._provenance import _fingerprint, _registry, audit_log, new_split_id, register_partition
    sid = new_split_id()
    register_partition(train_out, "train", sid)
    register_partition(valid_out, "valid", sid)
    register_partition(test_out, "test", sid)
    audit_log("split", _fingerprint(data), split_id=sid)

    result = SplitResult(
        train=train_out,
        valid=valid_out,
        test=test_out,
        _target=target,
        _task=detected_task,
    )
    _registry.register_split(sid)
    return result


def _split_custom(
    data: pd.DataFrame,
    target: str | None,
    splitter,
    task: str,
) -> CVResult:
    """Split using a custom CV splitter (sklearn-compatible)."""
    import numpy as np

    # Guard: reject primitive types that happen to have .split() (e.g. str)
    if isinstance(splitter, (str, int, float, bool, type(None))):
        raise ConfigError(
            f"splitter must be a CV splitter object with a .split(X, y) method. "
            f"Got {type(splitter).__name__}: {splitter!r}. "
            "Pass any CV splitter with a .split(X, y) method."
        )

    # Validate splitter has .split() method
    if not hasattr(splitter, "split") or not callable(splitter.split):
        raise ConfigError(
            f"splitter must have a .split(X, y) method. "
            f"{type(splitter).__name__} does not. "
            "Pass any CV splitter with a .split(X, y) method."
        )

    # Generate fold indices
    X = data
    y = data[target] if target is not None else None

    try:
        if y is not None:
            fold_list = [(np.array(tr), np.array(val)) for tr, val in splitter.split(X, y)]
        else:
            fold_list = [(np.array(tr), np.array(val)) for tr, val in splitter.split(X)]
    except TypeError:
        # Some splitters only accept X
        fold_list = [(np.array(tr), np.array(val)) for tr, val in splitter.split(X)]

    if not fold_list:
        raise DataError("Custom splitter produced 0 folds.")

    n_folds = len(fold_list)

    # Use actual fold count (not splitter.n_splits which can differ, e.g. RepeatedKFold)
    k = n_folds

    return CVResult(
        _data=data,
        folds=fold_list,
        k=k,
        target=target,
    )


def _detect_task(target: pd.Series) -> str:
    """Detect classification vs regression from target column.

    Heuristic (pragmatic defaults, not theoretically grounded):
    - String, bool, category dtype → classification
    - Numeric with nunique <= 20 AND nunique/len <= 0.05 → classification
    - Everything else → regression

    Users should override with task="regression" or task="classification"
    when the heuristic gets it wrong. The warning message guides them.

    Warns when auto-detecting classification on numeric targets.
    """
    # String/bool/category → classification
    if pd.api.types.is_object_dtype(target) or \
       pd.api.types.is_string_dtype(target) or \
       isinstance(target.dtype, pd.CategoricalDtype) or \
       pd.api.types.is_bool_dtype(target):
        return "classification"

    # Numeric heuristic
    if pd.api.types.is_numeric_dtype(target):
        n_unique = target.nunique()
        n_total = len(target)

        # Float64 targets are usually regression (ratings, grades, prices stored as float)
        # Only apply classification heuristic to integer-typed columns, EXCEPT:
        # float columns with exactly {0.0, 1.0} are binary classification
        is_integer_typed = pd.api.types.is_integer_dtype(target)
        is_float_binary = (
            pd.api.types.is_float_dtype(target) and
            n_unique == 2 and
            set(target.dropna().unique()).issubset({0.0, 1.0})
        )

        if (is_integer_typed and (n_unique <= 2 or (n_unique <= 20 and n_unique / n_total <= 0.05))) \
                or is_float_binary:
            # Looks like classification
            warnings.warn(
                f"Target '{target.name}' detected as classification "
                f"({n_unique} unique values). To override: task='regression'",
                UserWarning,
                stacklevel=4
            )
            return "classification"

    return "regression"


def _split_repeated_cv(
    data: pd.DataFrame,
    target: str | None,
    folds: int,
    repeats: int,
    seed: int,
    stratify: bool,
    task: str,
) -> CVResult:
    """Repeated K-fold CV for variance reduction. A6.

    Creates R * K folds total. Variance reduction diminishes after R=3.
    """
    # Detect task
    detected_task = task
    if task == "auto" and target is not None:
        detected_task = _detect_task(data[target])

    do_stratify = stratify and detected_task == "classification" and target is not None

    if do_stratify:
        fold_list = list(_repeated_stratified_kfold(
            data[target].values, k=folds, repeats=repeats, seed=seed))
    else:
        fold_list = list(_repeated_kfold(
            len(data), k=folds, repeats=repeats, seed=seed))

    total_folds = folds * repeats
    return CVResult(
        _data=data,
        folds=fold_list,
        k=total_folds,
        target=target,
    )


# ---------------------------------------------------------------------------
# Public domain specializations
# ---------------------------------------------------------------------------


def split_temporal(
    data: pd.DataFrame,
    target: str | None = None,
    *,
    time: str,
    ratio: tuple = (0.6, 0.2, 0.2),
    folds: int | None = None,
    task: str = "auto",
) -> SplitResult:
    """Split data chronologically — no future leakage.

    Domain specialization of ``split`` for time series and forecasting.
    Data is sorted by the ``time`` column and partitioned by position.
    Deterministic: no seed parameter (chronological order is the only order).

    For temporal cross-validation, use ``ml.cv_temporal()`` after splitting:

        >>> s = ml.split_temporal(data, "price", time="date")
        >>> c = ml.cv_temporal(s, folds=5, embargo=10)

    Args:
        data: DataFrame to split
        target: Target column name (enables task detection)
        time: Column name containing timestamps or orderable values.
            Used for sorting, then dropped from output partitions.
        ratio: (train, valid, test) fractions for holdout split (default 60/20/20)
        task: ``"auto"``, ``"classification"``, or ``"regression"``

    Returns:
        SplitResult

    Example:
        >>> s = ml.split_temporal(data, "price", time="date")
    """
    from ._compat import to_pandas
    data = to_pandas(data)

    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"split_temporal() expects a DataFrame, got {type(data).__name__}."
        )
    if data.shape[0] == 0:
        raise DataError("Cannot split empty data (0 rows)")

    if time not in data.columns:
        available = data.columns.tolist()
        raise DataError(
            f"time='{time}' not found in data. Available columns: {available}"
        )

    # Validate target
    if target is not None:
        if target not in data.columns:
            raise DataError(
                f"target='{target}' not found in data. "
                f"Available columns: {data.columns.tolist()}"
            )
        n_before = len(data)
        data = data.dropna(subset=[target]).reset_index(drop=True)
        if len(data) < n_before:
            warnings.warn(
                f"Dropped {n_before - len(data)} rows with NaN target.",
                UserWarning, stacklevel=2,
            )
        if len(data) == 0:
            raise DataError(f"Target column '{target}' is entirely NaN")

    # NaN timestamp warning
    n_nat = data[time].isna().sum()
    if n_nat > 0:
        warnings.warn(
            f"{n_nat} rows have NaN/NaT in time='{time}'. "
            "NaN sorts to end (will land in test partition). "
            f"Fix: data = data.dropna(subset=['{time}'])",
            UserWarning, stacklevel=2,
        )

    if folds is not None:
        raise ConfigError(
            "split_temporal(folds=) is removed. Use ml.cv_temporal() instead:\n"
            "  s = ml.split_temporal(data, target, time='col')\n"
            "  c = ml.cv_temporal(s, folds=5, embargo=0)\n"
            "This ensures a test holdout is always preserved."
        )

    return _split_temporal(data, target, time, ratio, task)


def split_group(
    data: pd.DataFrame,
    target: str | None = None,
    *,
    groups: str,
    seed: int = 42,
    ratio: tuple = (0.6, 0.2, 0.2),
    folds: int | None = None,
    stratify: bool = True,
    task: str = "auto",
) -> SplitResult:
    """Split data with group non-overlap — no group leaks across partitions.

    Domain specialization of ``split`` for clinical trials, repeated measures,
    and any data where observations are nested within groups (patients, subjects,
    hospitals, devices). No group appears in more than one partition.

    For group cross-validation, use ``ml.cv_group()`` after splitting:

        >>> s = ml.split_group(data, "outcome", groups="patient_id", seed=42)
        >>> c = ml.cv_group(s, folds=5, groups="patient_id", seed=42)

    Args:
        data: DataFrame to split
        target: Target column name (enables stratification + task detection)
        groups: Column name identifying groups. Each unique value is kept
            together — never split across train/valid/test.
        seed: Random seed for reproducibility (default: 42)
        ratio: (train, valid, test) fractions for holdout split (default 60/20/20)
        stratify: Stratify by target within groups (classification only, default True)
        task: ``"auto"``, ``"classification"``, or ``"regression"``

    Returns:
        SplitResult

    Example:
        >>> s = ml.split_group(data, "outcome", groups="patient_id", seed=42)
    """
    from ._compat import to_pandas
    data = to_pandas(data)

    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"split_group() expects a DataFrame, got {type(data).__name__}."
        )
    if data.shape[0] == 0:
        raise DataError("Cannot split empty data (0 rows)")

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(
            f"seed must be an integer, got {type(seed).__name__}: {seed!r}. "
            "Example: seed=42"
        )
    if seed < 0:
        raise ConfigError(
            f"seed must be non-negative, got {seed}. Example: seed=42"
        )
    if seed >= 2**64:
        raise ConfigError(
            f"seed must be < 2^64, got {seed}. Example: seed=42"
        )

    if groups not in data.columns:
        available = data.columns.tolist()
        raise DataError(
            f"groups='{groups}' not found in data. Available columns: {available}"
        )

    # P0: NaN group values use object identity in set() — silent misassignment
    if data[groups].isna().any():
        raise DataError(
            f"groups='{groups}' contains NaN values. "
            "Drop or impute NaN groups before splitting."
        )

    # Validate target
    if target is not None:
        if target not in data.columns:
            raise DataError(
                f"target='{target}' not found in data. "
                f"Available columns: {data.columns.tolist()}"
            )
        n_before = len(data)
        data = data.dropna(subset=[target]).reset_index(drop=True)
        if len(data) < n_before:
            warnings.warn(
                f"Dropped {n_before - len(data)} rows with NaN target.",
                UserWarning, stacklevel=2,
            )
        if len(data) == 0:
            raise DataError(f"Target column '{target}' is entirely NaN")

    if folds is not None:
        raise ConfigError(
            "split_group(folds=) is removed. Use ml.cv_group() instead:\n"
            "  s = ml.split_group(data, target, groups='col', seed=42)\n"
            "  c = ml.cv_group(s, folds=5, groups='col', seed=42)\n"
            "This ensures a test holdout is always preserved."
        )
    # Stratification not implemented for group holdout — warn if expected
    if stratify and target is not None:
        detected = task if task != "auto" else _detect_task(data[target])
        if detected == "classification":
            warnings.warn(
                "stratify=True is ignored for group holdout splits. "
                "Group-level stratification is only available with folds=. "
                "Groups are assigned to partitions as whole units.",
                UserWarning, stacklevel=2,
            )
    return _split_group_holdout(data, target, groups, ratio, seed, task)
