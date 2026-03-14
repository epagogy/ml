"""Pre-flight data quality checks — ml.check_data().

Inspired by EvalML's data checker. Runs before fit() to catch
common data quality issues that silently degrade model performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass
class CheckResult:
    """Result from check() — reproducibility verification.

    Returned by ml.check() to confirm that a model's predictions are
    bitwise identical across two independent fits with the same seed.

    Attributes:
        passed: True if predictions are bitwise identical (reproducible)
        algorithm: Algorithm used for the reproducibility check
        seed: Random seed used
        message: Human-readable summary of the check outcome

    Supports bool() for backward compatibility:
        assert ml.check(data, "target", seed=42)   # truthiness check
        result = ml.check(data, "target", seed=42)
        if result:  # works via __bool__
            print("Reproducible!")

    Example:
        >>> result = ml.check(data, "target", seed=42)
        >>> bool(result)
        True
        >>> result.algorithm
        'random_forest'
        >>> result.passed
        True
    """
    passed: bool
    algorithm: str
    seed: int
    message: str

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"CheckResult({status}, {self.algorithm}, seed={self.seed})"


@dataclass
class CheckReport:
    """Result from check_data()."""

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = ["Data Check Report", "=" * 40]
        if not self.warnings and not self.errors:
            lines.append("No issues found. Data looks clean.")
        if self.errors:
            lines.append(f"\n{len(self.errors)} ERROR(s):")
            for e in self.errors:
                lines.append(f"  [ERROR] {e}")
        if self.warnings:
            lines.append(f"\n{len(self.warnings)} WARNING(s):")
            for w in self.warnings:
                lines.append(f"  [WARN] {w}")
        return "\n".join(lines)

    @property
    def has_issues(self) -> bool:
        """True if any warnings or errors were found."""
        return bool(self.warnings or self.errors)

    @property
    def passed(self) -> bool:
        """True if no errors found (warnings allowed). Consistent with CheckResult.passed."""
        return len(self.errors) == 0

    def __bool__(self) -> bool:
        return not self.has_issues


def check_data(
    data: pd.DataFrame,
    target: str,
    *,
    severity: Literal["warn", "error"] = "warn",
) -> CheckReport:
    """Pre-flight data quality checks.

    Runs before fit() to catch common data quality issues that silently
    degrade model performance.

    Checks performed:
    - ID columns (100% unique values, any dtype — likely a row identifier)
    - Zero-variance features (constant columns)
    - High-null columns (>50% missing values)
    - Severe class imbalance (<5% minority class, classification only)
    - Duplicate rows (>10% of data)

    Args:
        data: Training DataFrame
        target: Target column name
        severity: "warn" (default) adds issues to .warnings.
            "error" raises DataError if any issue is found.

    Returns:
        CheckReport with .warnings, .errors, and .has_issues

    Raises:
        DataError: If severity="error" and issues are found.

    Example:
        >>> report = ml.check_data(data, "target")
        >>> print(report)
        >>> if report.has_issues:
        ...     print(report.warnings)
    """

    from ._types import DataError
    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"check_data() expects a DataFrame, got {type(data).__name__}."
        )
    if target not in data.columns:
        raise DataError(
            f"target='{target}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )

    # Check for duplicate column names (pandas returns DataFrame on df[col] when duplicated,
    # causing "truth value of a Series is ambiguous" in downstream checks)
    dupes = data.columns[data.columns.duplicated()].tolist()
    if dupes:
        raise DataError(
            f"Duplicate column names found: {dupes}. "
            "Rename columns before calling ml.check_data()."
        )

    report = CheckReport()
    X = data.drop(columns=[target])
    y = data[target]

    # 0. NaN in target — silently dropped by split(), causes invisible data loss
    n_nan = int(y.isna().sum())
    if n_nan > 0:
        report.warnings.append(
            f"Target '{target}' has {n_nan} NaN value(s) ({n_nan / len(data):.0%} of rows). "
            "These rows are silently dropped by ml.split(). Impute or remove them first."
        )

    # 0b. Inf in features — not caught by NaN checks, causes silent fit failures
    import numpy as np
    for col in data.columns:
        if data[col].dtype.kind in ("f", "i", "u"):
            n_inf = int(np.isinf(data[col].replace([np.nan], [0])).sum())
            if n_inf > 0:
                report.warnings.append(
                    f"Column '{col}' has {n_inf} infinite value(s) (±inf). "
                    "Most algorithms raise errors or produce NaN predictions on inf. "
                    "Replace with np.nan and impute, or clip with np.clip()."
                )

    # 1. ID columns: 100% unique integer or string columns — likely a row identifier
    # Float columns with 100% unique are normal continuous features, skip them.
    for col in X.columns:
        if len(X) > 10 and X[col].nunique() == len(X):
            is_int_like = X[col].dtype.kind in ("i", "u")  # signed/unsigned int
            is_str_like = X[col].dtype == object or str(X[col].dtype) in ("string", "str")
            if is_int_like or is_str_like:
                report.warnings.append(
                    f"Column '{col}' has {X[col].nunique()} unique values "
                    f"({X[col].nunique() / len(X):.0%} unique). "
                    "Looks like an ID column. Consider dropping it before fitting."
                )

    # 2. Zero-variance features (constant columns)
    for col in X.columns:
        if X[col].nunique(dropna=True) <= 1:
            report.warnings.append(
                f"Column '{col}' has zero variance (all values are constant). "
                "It provides no predictive information. Consider dropping it."
            )

    # 3. High-null columns (>50% missing)
    for col in X.columns:
        null_frac = X[col].isna().mean()
        if null_frac > 0.5:
            report.warnings.append(
                f"Column '{col}' has {null_frac:.0%} missing values. "
                "Consider imputing or dropping columns with >50% missing."
            )

    # 4. Class imbalance checks (classification only)
    task = _infer_task(y)
    if task == "classification":
        n_unique = y.nunique(dropna=True)
        # 4a. Degenerate target — single class causes fit() to raise DataError
        if n_unique <= 1:
            _non_null = y.dropna()
            _val_hint = f" (value: {_non_null.iloc[0]!r})" if len(_non_null) > 0 else ""
            report.warnings.append(
                f"Target '{target}' has only {n_unique} unique class(es){_val_hint}. "
                "ml.fit() will raise DataError. Ensure your data has at least 2 classes."
            )
        else:
            # 4b. Severe class imbalance (minority class <= 5%)
            counts = y.value_counts(normalize=True)
            for cls, frac in counts.items():
                if frac <= 0.05:
                    report.warnings.append(
                        f"Class '{cls}' represents only {frac:.1%} of samples "
                        "(severe class imbalance). "
                        "Consider ml.fit(..., balance=True) or oversampling."
                    )

    # 5. Duplicate rows (>10% duplicates)
    n_duplicates = int(data.duplicated().sum())
    if n_duplicates > 0.1 * len(data):
        report.warnings.append(
            f"{n_duplicates} duplicate rows ({n_duplicates / len(data):.0%} of data). "
            "Consider deduplication: data.drop_duplicates()."
        )

    # 6. Feature redundancy (pairwise |r| > 0.95)
    numeric_cols = X.select_dtypes(include="number").columns
    if len(numeric_cols) >= 2:
        corr_matrix = X[numeric_cols].corr().abs()
        # Zero out diagonal and lower triangle to avoid duplicates
        import numpy as np
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        redundant_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                if mask[i, j] and corr_matrix.iloc[i, j] > 0.95:
                    redundant_pairs.append(
                        (numeric_cols[i], numeric_cols[j], corr_matrix.iloc[i, j])
                    )
        if redundant_pairs:
            pair_strs = [f"'{a}' ↔ '{b}' (r={r:.3f})" for a, b, r in redundant_pairs[:5]]
            suffix = f" (+{len(redundant_pairs) - 5} more)" if len(redundant_pairs) > 5 else ""
            report.warnings.append(
                f"{len(redundant_pairs)} highly correlated feature pair(s) "
                f"(|r| > 0.95): {', '.join(pair_strs)}{suffix}. "
                "Redundant features inflate importance estimates and slow fitting. "
                "Consider dropping one from each pair."
            )

    if severity == "error" and report.has_issues:
        all_issues = report.errors + report.warnings
        report.errors = list(report.warnings)
        report.warnings = []
        raise DataError(
            f"check_data() found {len(all_issues)} issue(s):\n"
            + "\n".join(f"  - {i}" for i in all_issues)
        )

    return report


def _infer_task(y: pd.Series) -> str:
    """Infer task type from target series."""
    if y.dtype == object or str(y.dtype) == "category":
        return "classification"
    # Special case: degenerate single-class target is always classification
    # (ratio check below fails for n<=20: 1/20=0.05 is not <0.05)
    if y.nunique() <= 1:
        return "classification"
    if y.nunique() <= 20 and y.nunique() / max(len(y), 1) < 0.05:
        return "classification"
    return "regression"


def check(data, target, *, algorithm="random_forest", seed: int):
    """Verify bitwise reproducibility for a given dataset.

    Fits the same model twice and asserts predictions are bitwise identical.

    Args:
        data: DataFrame with features and target
        target: Target column name
        algorithm: Algorithm to use (default "random_forest")
        seed: Random seed

    Returns:
        True if reproducible

    Raises:
        ModelError: If predictions are not bitwise identical

    Example:
        >>> ml.check(data, "target")
        True
    """
    from .fit import fit
    from .predict import predict
    from .split import split

    s = split(data, target, seed=seed)
    model1 = fit(data=s.train, target=target, algorithm=algorithm, seed=seed)
    model2 = fit(data=s.train, target=target, algorithm=algorithm, seed=seed)

    preds1 = predict(model1, s.valid)
    preds2 = predict(model2, s.valid)

    if not (preds1.values == preds2.values).all():
        from ._types import ModelError
        raise ModelError(
            f"check() found non-reproducible predictions for algorithm='{algorithm}', "
            f"seed={seed}. This model is not deterministic. "
            "Ensure seed is set correctly and the algorithm supports determinism."
        )
    return CheckResult(
        passed=True,
        algorithm=algorithm,
        seed=seed,
        message=f"Predictions are bitwise identical for algorithm='{algorithm}', seed={seed}.",
    )
