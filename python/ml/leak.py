"""leak() — data leakage detection.

Pure data introspection. No model fitting. No dependencies on fit/evaluate.
Answers: "Is my data safe to model, or am I fooling myself?"

7 checks, two tiers (warn/critical):
1. Feature-target correlation (numeric, Pearson |r|)
2. Single-feature AUC (binary classification)
3. Nonlinear predictability (mutual information)
4. High-cardinality IDs
5. Target in feature names
6. Duplicate rows between train/test (SplitResult only)
7. Temporal ordering (SplitResult only)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ._types import LeakReport, SplitResult


def leak(
    data: pd.DataFrame | SplitResult,
    target: str,
) -> LeakReport:
    """Detect potential data leakage before modeling.

    Analyzes feature-target relationships to warn of leakage.

    Args:
        data: DataFrame or SplitResult to analyze.
            DataFrame: runs feature-target checks only.
            SplitResult: also checks train-test contamination.
        target: Target column name.

    Returns:
        LeakReport with .clean (bool), .checks, .suspects, .top_features

    Raises:
        DataError: If target not found or data is empty

    Example:
        >>> s = ml.split(data, "churn", seed=42)
        >>> report = ml.leak(s, "churn")
        >>> report.clean
        True
    """
    from ._types import (
        CheckResult,
        DataError,
        LeakReport,
        SplitResult,
        SuspectFeature,
    )

    # 1. Unwrap SplitResult → train_df + test_df
    test_df = None
    if isinstance(data, SplitResult):
        train_df = data.train
        test_df = data.test
    elif isinstance(data, pd.DataFrame):
        train_df = data
    else:
        raise DataError(
            f"leak() expects DataFrame or SplitResult, got {type(data).__name__}"
        )

    # 2. Validate
    if len(train_df) == 0:
        raise DataError("Cannot analyze empty data (0 rows)")
    if target not in train_df.columns:
        available = train_df.columns.tolist()
        raise DataError(
            f"target='{target}' not found in data. Available: {available}"
        )

    # 3. Separate X and y
    X = train_df.drop(columns=[target])
    y = train_df[target]

    # 4. Detect task
    task = _detect_task(y)

    # Encode target to numeric for correlation/MI
    y_numeric = _encode_target(y)

    # 5. Run checks
    checks: list[CheckResult] = []
    suspects: list[SuspectFeature] = []
    top_candidates: list[tuple] = []  # (feature, check_name, value, detail)

    # Check 1: Feature-target correlation
    # Skip Pearson |r| for multiclass targets — label encoding creates
    # arbitrary ordinal structure that inflates correlation (e.g., iris
    # petal width shows |r|=0.96 only because labels happen to be 0,1,2).
    # MI (check 3) handles multiclass correctly without this artefact.
    n_classes = y.nunique()
    if task == "classification" and n_classes > 2:
        from ._types import CheckResult as _CR
        c1 = _CR("Feature-target correlation", True,
                  f"skipped (multiclass, {n_classes} classes — use MI instead)", "ok")
        s1, t1 = [], []
    else:
        c1, s1, t1 = _check_correlation(X, y_numeric)
    checks.append(c1)
    suspects.extend(s1)
    top_candidates.extend(t1)

    # Check 2: Single-feature AUC (binary classification only)
    c2, s2, t2 = _check_single_auc(X, y, task)
    checks.append(c2)
    suspects.extend(s2)
    top_candidates.extend(t2)

    # Check 3: Nonlinear predictability (MI)
    c3, s3, t3 = _check_mutual_info(X, y, task)
    checks.append(c3)
    suspects.extend(s3)
    top_candidates.extend(t3)

    # Check 4: High-cardinality IDs
    c4, s4 = _check_id_columns(X)
    checks.append(c4)
    suspects.extend(s4)

    # Check 5: Target in feature names
    c5, s5 = _check_target_names(X.columns.tolist(), target)
    checks.append(c5)
    suspects.extend(s5)

    # Check 6: Duplicate rows (SplitResult only)
    if test_df is not None:
        test_X = test_df.drop(columns=[target], errors="ignore")
        c6, s6 = _check_duplicates(X, test_X)
        checks.append(c6)
        suspects.extend(s6)
    else:
        checks.append(CheckResult(
            name="Duplicate rows (train/test)",
            passed=True,
            detail="skipped (no split provided)",
            severity="ok",
        ))

    # Check 7: Temporal ordering (SplitResult only)
    if test_df is not None:
        c7, s7 = _check_temporal(train_df, test_df)
        checks.append(c7)
        suspects.extend(s7)
    else:
        checks.append(CheckResult(
            name="Temporal ordering",
            passed=True,
            detail="skipped (no split provided)",
            severity="ok",
        ))

    # 6. Build top-3 features by predictive strength (deduplicate by name)
    top_candidates.sort(key=lambda x: x[2], reverse=True)
    seen_names: set[str] = set()
    top_features: list[tuple] = []
    for name, check, _, detail in top_candidates:
        if name not in seen_names:
            seen_names.add(name)
            top_features.append((name, check, detail))
            if len(top_features) == 3:
                break

    n_warnings = sum(1 for c in checks if not c.passed)

    return LeakReport(
        clean=n_warnings == 0,
        n_warnings=n_warnings,
        checks=checks,
        suspects=suspects,
        top_features=top_features,
    )


# ===== HELPERS =====


def _detect_task(y: pd.Series) -> str:
    """Detect classification vs regression from target."""
    if y.dtype == object or y.dtype.name == "category" or y.dtype == bool:
        return "classification"
    n_unique = y.nunique()
    if n_unique <= 20 and (n_unique / len(y)) < 0.05:
        return "classification"
    return "regression"


def _encode_target(y: pd.Series) -> np.ndarray:
    """Encode target to numeric for correlation computation."""
    if y.dtype == object or y.dtype.name == "category" or str(y.dtype) in ("string", "str"):
        classes = sorted(y.dropna().unique())
        mapping = {c: i for i, c in enumerate(classes)}
        return np.array([mapping.get(v, -1) for v in y], dtype=np.float64)
    return y.values.astype(float)


# ===== CHECK IMPLEMENTATIONS =====


_WARN_CORR = 0.8
_CRIT_CORR = 0.95
_WARN_AUC = 0.8
_CRIT_AUC = 0.95


def _check_correlation(
    X: pd.DataFrame, y_numeric: np.ndarray
) -> tuple:
    """Check 1: Feature-target Pearson correlation for numeric features."""
    from ._types import CheckResult, SuspectFeature

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return (
            CheckResult("Feature-target correlation", True, "no numeric features", "ok"),
            [],
            [],
        )

    suspects = []
    top_candidates = []
    max_corr = 0.0
    max_col = ""

    y_series = pd.Series(y_numeric, index=X.index)

    for col in numeric_cols:
        # Suppress numpy RuntimeWarning from corrcoef on constant features
        # (std=0 → division by zero in Pearson correlation).
        # np.errstate handles numpy C-level warnings that bypass Python's warnings module.
        with np.errstate(invalid='ignore', divide='ignore'):
            r = abs(X[col].corr(y_series))
        if np.isnan(r):
            continue

        if r > max_corr:
            max_corr = r
            max_col = col

        top_candidates.append((col, "correlation", r, f"|r|={r:.2f}"))

        if r > _CRIT_CORR:
            suspects.append(SuspectFeature(
                feature=col,
                check="Feature-target correlation",
                value=float(r),
                detail=f"|r|={r:.2f}",
                action="Verify this feature is not derived from the target. "
                       "High correlation alone is not leakage — genuinely predictive features are expected",
            ))
        elif r > _WARN_CORR:
            suspects.append(SuspectFeature(
                feature=col,
                check="Feature-target correlation",
                value=float(r),
                detail=f"|r|={r:.2f}",
                action="Investigate if this feature uses target or future data. "
                       "High correlation with a legitimate feature is normal",
            ))

    severity = "ok"
    if max_corr > _CRIT_CORR:
        severity = "critical"
    elif max_corr > _WARN_CORR:
        severity = "warn"

    passed = severity == "ok"
    detail = f"max |r|={max_corr:.2f} ({max_col})" if max_col else "no valid correlations"

    return (
        CheckResult("Feature-target correlation", passed, detail, severity),
        suspects,
        top_candidates,
    )


def _check_single_auc(
    X: pd.DataFrame, y: pd.Series, task: str
) -> tuple:
    """Check 2: Single-feature AUC for binary classification."""
    from ._types import CheckResult, SuspectFeature

    # Only binary classification
    if task != "classification" or y.nunique() != 2:
        return (
            CheckResult("Single-feature AUC", True, "skipped (not binary)", "ok"),
            [],
            [],
        )

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return (
            CheckResult("Single-feature AUC", True, "no numeric features", "ok"),
            [],
            [],
        )

    from ._scoring import _roc_auc as roc_auc_score

    # Encode binary target to 0/1
    _classes = sorted(np.unique(y))
    _label_map = {c: i for i, c in enumerate(_classes)}
    y_bin = np.array([_label_map[v] for v in y])

    suspects = []
    top_candidates = []
    max_auc = 0.0
    max_col = ""

    for col in numeric_cols:
        vals = X[col].values
        # Skip if constant or has NaN
        if np.isnan(vals).any() or np.std(vals) == 0:
            continue

        try:
            auc = roc_auc_score(y_bin, vals)
            # AUC can be < 0.5 if negatively correlated; flip
            auc = max(auc, 1 - auc)
        except ValueError:
            continue

        if auc > max_auc:
            max_auc = auc
            max_col = col

        top_candidates.append((col, "auc", auc, f"AUC={auc:.2f}"))

        if auc > _CRIT_AUC:
            suspects.append(SuspectFeature(
                feature=col,
                check="Single-feature AUC",
                value=float(auc),
                detail=f"AUC={auc:.2f}",
                action="Verify this feature is not computed from the target or using future data. "
                       "A single feature predicting the target almost perfectly is unusual",
            ))
        elif auc > _WARN_AUC:
            suspects.append(SuspectFeature(
                feature=col,
                check="Single-feature AUC",
                value=float(auc),
                detail=f"AUC={auc:.2f}",
                action="Investigate if this feature uses target or future data. "
                       "Strong predictors are expected in good datasets",
            ))

    severity = "ok"
    if max_auc > _CRIT_AUC:
        severity = "critical"
    elif max_auc > _WARN_AUC:
        severity = "warn"

    passed = severity == "ok"
    detail = f"max AUC={max_auc:.2f} ({max_col})" if max_col else "no valid features"

    return (
        CheckResult("Single-feature AUC", passed, detail, severity),
        suspects,
        top_candidates,
    )


def _histogram_mutual_info(
    X: pd.DataFrame, y: pd.Series, task: str, n_bins: int = 10,
) -> np.ndarray:
    """Histogram-based mutual information for each feature vs target.

    Percentile-binned approximation. Sufficient for leak detection:
    leaky features get high MI, random features get ~0.

    Returns 1D array of MI values, one per column of X.
    """
    n = len(y)
    mi = np.zeros(X.shape[1])

    # Bin target
    if task == "classification":
        y_binned = np.asarray(y)
    else:
        # Percentile bins for regression target
        y_arr = np.asarray(y, dtype=np.float64)
        edges = np.percentile(y_arr, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        y_binned = np.digitize(y_arr, edges[1:-1]) if len(edges) > 1 else np.zeros(n)

    # H(Y) — target entropy
    _, y_counts = np.unique(y_binned, return_counts=True)
    py = y_counts / n
    h_y = -np.sum(py * np.log(py + 1e-12))

    for j in range(X.shape[1]):
        col = np.asarray(X.iloc[:, j], dtype=np.float64)
        # Percentile bins for feature
        edges = np.percentile(col, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) <= 1:
            mi[j] = 0.0
            continue
        x_binned = np.digitize(col, edges[1:-1])

        # Joint counts
        joint = {}
        for xi, yi in zip(x_binned, y_binned):
            key = (int(xi), yi)
            joint[key] = joint.get(key, 0) + 1

        # H(Y|X) — conditional entropy
        x_vals, x_counts = np.unique(x_binned, return_counts=True)

        h_y_given_x = 0.0
        for xi_idx, xi in enumerate(x_vals):
            nx = x_counts[xi_idx]
            if nx == 0:
                continue
            for yi in np.unique(y_binned):
                nxy = joint.get((int(xi), yi), 0)
                if nxy > 0:
                    h_y_given_x -= (nxy / n) * np.log(nxy / nx + 1e-12)

        mi[j] = max(0.0, h_y - h_y_given_x)

    return mi


def _check_mutual_info(
    X: pd.DataFrame, y: pd.Series, task: str
) -> tuple:
    """Check 3: Nonlinear predictability via mutual information."""
    from ._types import CheckResult, SuspectFeature

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return (
            CheckResult("Nonlinear predictability", True, "no numeric features", "ok"),
            [],
            [],
        )

    X_num = X[numeric_cols].copy()
    # Fill NaN for MI computation
    X_num = X_num.fillna(X_num.median())

    try:
        mi = _histogram_mutual_info(X_num, y, task)
    except Exception as e:
        return (
            CheckResult("Nonlinear predictability", False,
                        f"computation failed: {str(e)[:80]}", "warn"),
            [],
            [],
        )

    # Compute target entropy for threshold
    if task == "classification":
        probs = y.value_counts(normalize=True).values
        h_y = -np.sum(probs * np.log2(probs + 1e-12))
    else:
        # For regression, use differential entropy approximation
        h_y = np.log2(np.std(y) * np.sqrt(2 * np.pi * np.e) + 1e-12)

    threshold = 0.9 * max(h_y, 0.1)  # Floor to avoid near-zero threshold

    suspects = []
    top_candidates = []
    max_mi = 0.0
    max_col = ""

    for i, col in enumerate(numeric_cols):
        mi_val = mi[i]
        if mi_val > max_mi:
            max_mi = mi_val
            max_col = col

        top_candidates.append((col, "mi", mi_val, f"MI={mi_val:.2f}"))

        if mi_val > threshold:
            suspects.append(SuspectFeature(
                feature=col,
                check="Nonlinear predictability",
                value=float(mi_val),
                detail=f"MI={mi_val:.2f}",
                action="Feature captures most target information — verify it's not derived from target",
            ))

    passed = len(suspects) == 0
    severity = "critical" if suspects else "ok"
    detail = f"max MI={max_mi:.2f} ({max_col})" if max_col else "no valid features"

    return (
        CheckResult("Nonlinear predictability", passed, detail, severity),
        suspects,
        top_candidates,
    )


def _check_id_columns(X: pd.DataFrame) -> tuple:
    """Check 4: High-cardinality columns that look like IDs."""
    from ._types import CheckResult, SuspectFeature

    suspects = []
    id_pattern = re.compile(r"(^id$|_id$|^index$|^key$|_key$)", re.IGNORECASE)
    n_rows = len(X)

    for col in X.columns:
        n_unique = X[col].nunique()
        unique_ratio = n_unique / n_rows if n_rows > 0 else 0

        # Only flag high cardinality for integer/string/object columns (not continuous floats)
        dtype = X[col].dtype
        is_discrete = dtype.name in ("object", "category") or np.issubdtype(dtype, np.integer)
        is_high_card = unique_ratio > 0.95 and is_discrete
        is_id_name = bool(id_pattern.search(str(col)))

        if is_high_card or is_id_name:
            reason = []
            if is_high_card:
                reason.append(f"{unique_ratio:.0%} unique")
            if is_id_name:
                reason.append("name matches ID pattern")
            suspects.append(SuspectFeature(
                feature=str(col),
                check="High-cardinality IDs",
                value=float(unique_ratio),
                detail=", ".join(reason),
                action="Looks like a row identifier — drop before modeling",
            ))

    passed = len(suspects) == 0
    if passed:
        detail = "none found"
    else:
        names = ", ".join(s.feature for s in suspects[:3])
        detail = f"{len(suspects)} suspect: {names}"

    return (
        CheckResult("High-cardinality IDs", passed, detail, "warn" if suspects else "ok"),
        suspects,
    )


def _check_target_names(features: list[str], target: str) -> tuple:
    """Check 5: Feature names containing target name or leakage patterns."""
    from ._types import CheckResult, SuspectFeature

    suspects = []
    target_lower = target.lower()

    # Common leakage patterns in feature names
    leakage_patterns = re.compile(
        r"(^future_|_future$|^next_|_next$|_after$|_outcome$|_result$)",
        re.IGNORECASE,
    )

    for feat in features:
        feat_lower = feat.lower()
        reasons = []

        # Check if feature name contains target name (but isn't the target itself)
        if target_lower in feat_lower and feat_lower != target_lower:
            reasons.append(f"contains target name '{target}'")

        if leakage_patterns.search(feat):
            reasons.append("matches leakage name pattern")

        if reasons:
            suspects.append(SuspectFeature(
                feature=feat,
                check="Target in feature names",
                value=1.0,
                detail=", ".join(reasons),
                action=f"Verify '{feat}' doesn't encode the target or use future information",
            ))

    passed = len(suspects) == 0
    if passed:
        detail = "none found"
    else:
        names = ", ".join(s.feature for s in suspects[:3])
        detail = f"{len(suspects)} suspect: {names}"

    return (
        CheckResult("Target in feature names", passed, detail, "warn" if suspects else "ok"),
        suspects,
    )


def _check_duplicates(train_X: pd.DataFrame, test_X: pd.DataFrame) -> tuple:
    """Check 6: Exact duplicate rows between train and test features."""
    from ._types import CheckResult, SuspectFeature

    # Use common columns only
    common_cols = train_X.columns.intersection(test_X.columns).tolist()
    if not common_cols:
        return (
            CheckResult("Duplicate rows (train/test)", True, "no common columns", "ok"),
            [],
        )

    # Hash-based duplicate detection
    def _hash_rows(df: pd.DataFrame, cols: list[str]) -> set:
        return set(
            df[cols].astype(str).apply(lambda row: hash(tuple(row)), axis=1).values
        )

    train_hashes = _hash_rows(train_X, common_cols)
    test_hashes = _hash_rows(test_X, common_cols)
    shared = len(train_hashes & test_hashes)

    n_test = len(test_X)
    dup_ratio = shared / n_test if n_test > 0 else 0

    suspects = []
    if shared > 0:
        severity = "critical" if dup_ratio > 0.01 else "warn"
        suspects.append(SuspectFeature(
            feature="(rows)",
            check="Duplicate rows (train/test)",
            value=float(dup_ratio),
            detail=f"{shared} shared rows ({dup_ratio:.1%} of test)",
            action="Remove duplicate rows or verify this is intentional (e.g. augmentation)",
        ))
        detail = f"{shared} shared rows ({dup_ratio:.1%} of test)"
        passed = False
    else:
        severity = "ok"
        detail = "0 shared rows"
        passed = True

    return (
        CheckResult("Duplicate rows (train/test)", passed, detail, severity),
        suspects,
    )


def _check_temporal(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Check 7: Temporal ordering — detect datetime columns and check overlap."""
    from ._types import CheckResult, SuspectFeature

    # Find datetime columns in both
    datetime_cols = []
    for col in train_df.columns:
        if pd.api.types.is_datetime64_any_dtype(train_df[col]):
            if col in test_df.columns:
                datetime_cols.append(col)

    if not datetime_cols:
        return (
            CheckResult("Temporal ordering", True, "no datetime columns detected", "ok"),
            [],
        )

    suspects = []
    for col in datetime_cols:
        train_max = train_df[col].max()
        test_min = test_df[col].min()

        if pd.notna(train_max) and pd.notna(test_min) and test_min < train_max:
            # Overlap detected
            n_overlap = (test_df[col] < train_max).sum()
            suspects.append(SuspectFeature(
                feature=col,
                check="Temporal ordering",
                value=float(n_overlap),
                detail=f"{n_overlap} test rows before max train date",
                action=f"Use time-based split or verify {col} isn't the event timestamp",
            ))

    passed = len(suspects) == 0
    if passed:
        detail = f"{len(datetime_cols)} datetime col(s), ordering OK"
    else:
        names = ", ".join(s.feature for s in suspects)
        detail = f"overlap detected in {names}"

    return (
        CheckResult("Temporal ordering", passed, detail, "warn" if suspects else "ok"),
        suspects,
    )
