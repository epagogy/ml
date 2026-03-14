"""profile() — data profiling before modeling.

Mostly pandas introspection. Delegates task detection to split._detect_task.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ._types import ProfileResult


PROFILE_SAMPLE_LIMIT = 100_000


def profile(
    data: pd.DataFrame,
    target: str | None = None,
) -> ProfileResult:
    """Profile a DataFrame to understand data before modeling.

    Args:
        data: Data to profile.
        target: Target column name. If provided, adds task detection and target distribution.

    Returns:
        Profile with shape, columns stats, warnings. Keys:
        shape, target, task, columns, warnings, target_distribution, target_balance, n_classes.

    Raises:
        DataError: If data is empty, target not found, or data is not a DataFrame.

    Example:
        >>> import ml
        >>> data = ml.dataset("churn")
        >>> prof = ml.profile(data)
        >>> prof["shape"]
        (5000, 12)
        >>> prof_with_target = ml.profile(data, "churn")
        >>> prof_with_target["task"]
        'classification'
    """
    from ._compat import to_pandas
    from ._types import DataError, ProfileResult

    # Auto-convert Polars/other DataFrames to pandas
    data = to_pandas(data)

    # Sample large datasets for speed
    if len(data) > PROFILE_SAMPLE_LIMIT:
        warnings.warn(
            f"profile() received {len(data):,} rows. Sampling {PROFILE_SAMPLE_LIMIT:,} rows for speed. "
            "Pass sample=False to disable.",
            UserWarning,
            stacklevel=2,
        )
        data = data.sample(PROFILE_SAMPLE_LIMIT, random_state=42)

    # Validation
    if not isinstance(data, pd.DataFrame):
        raise DataError(f"Expected DataFrame, got {type(data).__name__}")

    if len(data) == 0:
        raise DataError("Cannot profile empty data (0 rows)")

    if target is not None and target not in data.columns:
        available = list(data.columns)
        raise DataError(
            f"target='{target}' not found in data. Available columns: {available}"
        )

    # Edge case: all-NA target — warn but don't error (profile is diagnostic)
    if target is not None and data[target].isna().all():
        warnings.warn(
            f"Target '{target}' is entirely NA. Task detection and target "
            "distribution will be skipped.",
            UserWarning,
            stacklevel=2,
        )

    # Basic info
    nrows, ncols = data.shape
    result: dict = {
        "n_samples": nrows,
        "n_features": ncols - (1 if target else 0),
        "shape": (nrows, ncols),
        "target": target,
        "task": None,
        "columns": {},
        "warnings": [],
    }

    # Small sample warning
    if nrows < 500:
        train_rows = int(nrows * 0.6)
        test_rows = int(nrows * 0.2)
        result["warnings"].append(
            f"Small dataset ({nrows} rows). ~{train_rows} train, ~{test_rows} test after split."
        )

    # Task detection — delegate to split._detect_task (single source of truth)
    if target is not None:
        from .split import _detect_task

        target_col = data[target]
        n_unique = target_col.nunique()
        task = _detect_task(target_col)
        is_classification = task == "classification"

        result["task"] = task

        # Target distribution for classification
        if is_classification:
            counts = target_col.value_counts()
            result["target_distribution"] = counts.to_dict()
            result["n_classes"] = int(len(counts))
            minority_count = counts.min()
            result["target_balance"] = minority_count / len(data)

            # Warning: imbalanced target
            if result["target_balance"] < 0.20:
                minority_class = counts.idxmin()
                pct = result["target_balance"] * 100
                result["warnings"].append(
                    f"Imbalanced target: minority class '{minority_class}' is {pct:.0f}% of data"
                )

            # Warning: many classes
            if n_unique > 10:
                result["warnings"].append(
                    f"Target has {n_unique} classes — really classification?"
                )

    # Per-column stats
    for col in data.columns:
        col_data = data[col]
        dtype_str = str(col_data.dtype)
        n_missing = col_data.isna().sum()
        n_unique = col_data.nunique()

        col_stats = {
            "dtype": dtype_str,
            "missing": int(n_missing),
            "missing_pct": round(100 * n_missing / nrows, 1),
            "unique": int(n_unique),
        }

        # All NaN warning
        if n_missing == nrows:
            result["warnings"].append(f"'{col}' is entirely NaN")

        # Constant column warning
        if n_unique == 1 and n_missing < nrows:
            result["warnings"].append(
                f"'{col}' is constant (1 unique value)"
            )

        # Numeric stats
        if pd.api.types.is_numeric_dtype(col_data):
            col_stats["mean"] = float(col_data.mean()) if n_missing < nrows else None
            col_stats["std"] = float(col_data.std()) if n_missing < nrows else None
            col_stats["min"] = float(col_data.min()) if n_missing < nrows else None
            col_stats["max"] = float(col_data.max()) if n_missing < nrows else None
            col_stats["median"] = (
                float(col_data.median()) if n_missing < nrows else None
            )

        # Categorical stats
        else:
            if n_missing < nrows:
                top_val = col_data.mode()[0] if len(col_data.mode()) > 0 else None
                top_count = (col_data == top_val).sum() if top_val is not None else 0
                col_stats["top"] = top_val
                col_stats["top_freq"] = int(top_count)
            else:
                col_stats["top"] = None
                col_stats["top_freq"] = 0

            # High cardinality warning
            if n_unique > 0 and (n_unique / nrows) > 0.5:
                result["warnings"].append(
                    f"'{col}' high cardinality ({n_unique}/{nrows} unique)"
                )

        # Missing values warning (per column)
        if n_missing > 0 and n_missing < nrows:
            pct = col_stats["missing_pct"]
            result["warnings"].append(
                f"{n_missing} rows ({pct}%) have missing values in '{col}'"
            )

        result["columns"][col] = col_stats

    # Condition number for numeric features (multicollinearity diagnostic)
    # O(n*m^2) SVD — subsample rows to avoid hanging on large datasets
    _COND_MAX_ROWS = 5000
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if target is not None:
        numeric_cols = numeric_cols.drop(target, errors="ignore")
    if len(numeric_cols) >= 2:
        X_num = data[numeric_cols].dropna()
        if len(X_num) > 0:
            # Subsample for performance on large datasets
            if len(X_num) > _COND_MAX_ROWS:
                X_num = X_num.sample(n=_COND_MAX_ROWS, random_state=42)
            try:
                cond = float(np.linalg.cond(X_num.values))
                result["condition_number"] = cond
                if cond > 1000:
                    result["warnings"].append(
                        f"High condition number ({cond:.0f}). Near-collinear features."
                    )
            except (np.linalg.LinAlgError, ValueError):
                pass

    return ProfileResult(result)
