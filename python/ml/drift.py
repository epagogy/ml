"""drift() — label-free data drift detection.

Compares a reference dataset (training data) to new data using
per-feature statistical tests or adversarial validation.

Methods:
    "statistical" — per-feature KS/chi-squared tests (default). No labels needed.
    "adversarial" — train a classifier to distinguish reference from new data.
                    AUC measures distributional overlap. Returns per-row train_scores
                    for selecting validation rows that mirror test distribution.

Usage:
    >>> result = ml.drift(reference=s.train, new=new_customers)
    >>> result.shifted              # False (overall)
    >>> result.features_shifted     # ["monthly_charges"]
    >>> result.severity             # "low" / "medium" / "high"

    # Adversarial validation:
    >>> result = ml.drift(reference=s.train, new=test_data, method="adversarial", seed=42)
    >>> result.auc                  # 0.72 (> 0.6 = distinguishable)
    >>> result.train_scores         # pd.Series: per-row "looks like test" probability
    >>> val_idx = result.train_scores.nlargest(200).index  # high-fidelity val set
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DriftResult:
    """Result of a drift detection check.

    Attributes
    ----------
    shifted : bool
        True if drift detected (p < threshold for statistical; AUC > 0.6 for adversarial).
    features : dict[str, float]
        Statistical mode: per-feature p-values (low p = more likely drifted).
        Adversarial mode: per-feature importances (high = most discriminative).
    features_shifted : list[str]
        Statistical mode: features with p < threshold.
        Adversarial mode: top discriminative features.
    severity : str
        "none", "low", "medium", or "high".
    n_reference : int
        Number of rows in the reference dataset.
    n_new : int
        Number of rows in the new dataset.
    threshold : float
        p-value threshold (statistical mode, default 0.05).
    auc : float or None
        Adversarial mode only: AUC of the train-vs-test classifier.
        None for statistical mode.
    distinguishable : bool
        True if distributions are distinguishable.
        Statistical mode: True if any feature has p < threshold.
        Adversarial mode: True if AUC > 0.6.
    train_scores : pd.Series or None
        Adversarial mode only: per-row probability of "looks like test" for
        reference rows. Index matches reference DataFrame index.
        Use ``result.train_scores.nlargest(n).index`` to select validation
        rows that mirror the test distribution.
        None for statistical mode.
    """

    shifted: bool
    features: dict[str, float]
    features_shifted: list[str]
    severity: str
    n_reference: int
    n_new: int
    threshold: float = 0.05
    feature_tests: dict[str, str] = field(default_factory=dict, repr=False)
    # Adversarial mode only (None for statistical mode)
    auc: float | None = field(default=None, repr=False)
    distinguishable: bool = field(default=False, repr=False)
    train_scores: Any = field(default=None, repr=False)

    @property
    def drifted(self) -> bool:
        """Alias for shifted — verb-result symmetry: drift()→.drifted (like shelf()→.stale)."""
        return self.shifted

    def __bool__(self) -> bool:
        """Boolean truthiness mirrors .drifted — `if ml.drift(...):` works as expected."""
        return self.shifted

    def __repr__(self) -> str:
        shifted_str = f"shifted={self.shifted}"
        severity_str = f"severity='{self.severity}'"
        n_shifted = len(self.features_shifted)
        total = len(self.features)
        ratio = f"{n_shifted}/{total} features"
        return f"DriftResult({shifted_str}, {severity_str}, {ratio})"


def drift(
    *,
    reference: pd.DataFrame,
    new: pd.DataFrame,
    method: str = "statistical",
    threshold: float = 0.05,
    exclude: list[str] | None = None,
    target: str | None = None,
    seed: int | None = None,
    algorithm: str = "random_forest",
) -> DriftResult:
    """Detect data drift between reference and new data.

    Two methods available:

    **Statistical** (default): per-feature distribution tests.
    - Continuous features (numeric): Kolmogorov-Smirnov two-sample test
    - Categorical features (object/string): Chi-squared test on value counts
    - No labels needed — drift detection is purely distributional.

    **Adversarial**: train a classifier to distinguish reference from new data.
    - AUC ~= 0.5: distributions are similar (no meaningful drift)
    - AUC ~= 0.6: mild drift (``distinguishable=True``)
    - AUC ~= 1.0: very different distributions
    - ``result.train_scores``: per-row probability for reference rows that they
      "look like" the new/test data. Use to build a high-fidelity local CV:
      ``val_idx = result.train_scores.nlargest(n).index``
    - ``result.features``: most discriminative features (temporal leakage candidates)

    Parameters
    ----------
    reference : pd.DataFrame
        Reference dataset (typically training data). The baseline distribution.
    new : pd.DataFrame
        New data to compare against the reference.
    method : str, default="statistical"
        Detection method: ``"statistical"`` or ``"adversarial"``.
    threshold : float, default=0.05
        p-value threshold for ``method="statistical"``. Ignored for adversarial.
    exclude : list[str] or None
        Column names to skip (e.g., ID columns, timestamps).
    target : str or None
        Target column name. Automatically excluded from drift analysis.
    seed : int or None
        Random seed. Required for ``method="adversarial"``.
    algorithm : str, default="random_forest"
        Algorithm for adversarial classifier. Options: "random_forest", "xgboost".

    Returns
    -------
    DriftResult
        Statistical mode:
        - ``.shifted``: True if any feature is drifted
        - ``.features``: dict of feature → p-value
        - ``.features_shifted``: list of drifted feature names
        - ``.severity``: "none" / "low" / "medium" / "high"
        - ``.auc``, ``.distinguishable``, ``.train_scores``: None

        Adversarial mode:
        - ``.auc``: float classifier AUC (0.5 = no drift, 1.0 = full drift)
        - ``.distinguishable``: bool, True if AUC > 0.6
        - ``.train_scores``: pd.Series — per-row "looks like test" probability
          for reference rows. Index matches reference DataFrame.
        - ``.features``: dict of feature → importance (top discriminative)
        - ``.shifted``, ``.severity``: set based on AUC thresholds

    Raises
    ------
    DataError
        If reference or new is not a DataFrame, or if they share no columns.
    ConfigError
        If method is unknown, or seed= missing for adversarial.

    Examples
    --------
    >>> result = ml.drift(reference=s.train, new=new_customers)
    >>> result.shifted
    False

    Adversarial validation — build test-distribution-matched local CV:
    >>> result = ml.drift(reference=s.train, new=test_df, method="adversarial", seed=42)
    >>> result.auc            # 0.72 — distributions differ
    >>> result.train_scores.nlargest(200).index   # best local validation rows
    >>> result.features       # {"tenure": 0.31, "monthly_charges": 0.28, ...}
    """
    import numpy as np
    import pandas as pd

    from ._types import ConfigError, DataError

    if method not in ("statistical", "adversarial"):
        raise ConfigError(
            f"method='{method}' not recognized. Choose from: 'statistical', 'adversarial'."
        )

    if method == "adversarial":
        if seed is None:
            raise ConfigError(
                "drift(method='adversarial') requires seed=. "
                "Example: ml.drift(reference=s.train, new=test_df, "
                "method='adversarial', seed=42)"
            )

    if not isinstance(reference, pd.DataFrame):
        raise DataError(
            f"drift() reference must be a DataFrame, got {type(reference).__name__}."
        )
    if not isinstance(new, pd.DataFrame):
        raise DataError(
            f"drift() new must be a DataFrame, got {type(new).__name__}."
        )
    if len(reference) == 0:
        raise DataError("drift() reference dataset is empty.")
    if len(new) == 0:
        raise DataError("drift() new dataset is empty.")

    # F6: warn when new data is too small for reliable statistical tests
    if len(new) < 30:
        import warnings
        warnings.warn(
            f"drift() new data has only {len(new)} rows — insufficient for reliable "
            "distribution tests (KS test / chi-squared require n ≥ 30). "
            "Results may have low statistical power. Collect more data for reliable drift detection.",
            UserWarning,
            stacklevel=2,
        )

    exclude = set(exclude or [])
    if target is not None:
        exclude.add(target)
    shared_cols = [c for c in reference.columns if c in new.columns and c not in exclude]

    if not shared_cols:
        raise DataError(
            "drift() found no shared columns between reference and new data. "
            f"Reference columns: {list(reference.columns)[:10]}. "
            f"New columns: {list(new.columns)[:10]}."
        )

    # Dispatch to adversarial method
    if method == "adversarial":
        return _adversarial_drift(
            reference=reference,
            new=new,
            shared_cols=shared_cols,
            algorithm=algorithm,
            seed=seed,
            threshold=threshold,
        )


    from . import _stats

    p_values: dict[str, float] = {}
    test_used: dict[str, str] = {}

    for col in shared_cols:
        ref_col = reference[col].dropna()
        new_col = new[col].dropna()

        if len(ref_col) == 0 or len(new_col) == 0:
            # Warn about silently skipped all-NaN columns
            which = "reference" if len(ref_col) == 0 else "new"
            import warnings as _w
            _w.warn(
                f"Column '{col}' is entirely NaN in {which} data — "
                "skipped in drift detection. Fill or drop this column.",
                UserWarning,
                stacklevel=2,
            )
            continue

        if pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(new_col):
            # KS test for continuous features
            try:
                _, p = _stats.ks_2samp(ref_col.values, new_col.values)
                p_values[col] = float(p)
                test_used[col] = "ks"
            except Exception:
                pass
        else:
            # Chi-squared for categorical features
            # Build a contingency table using union of categories
            ref_counts = ref_col.astype(str).value_counts()
            new_counts = new_col.astype(str).value_counts()
            all_cats = sorted(set(ref_counts.index) | set(new_counts.index))
            if len(all_cats) < 2:
                # Only one category — no drift possible
                p_values[col] = 1.0
                test_used[col] = "chi2"
                continue
            ref_freq = np.array([ref_counts.get(c, 0) for c in all_cats], dtype=float)
            new_freq = np.array([new_counts.get(c, 0) for c in all_cats], dtype=float)
            # Avoid chi2 with all-zero expected
            if ref_freq.sum() == 0 or new_freq.sum() == 0:
                p_values[col] = 1.0
                test_used[col] = "chi2"
                continue
            # Normalize to proportions then scale to counts (scipy expects counts)
            try:
                # Use chi2 contingency on 2-row matrix: [reference, new]
                table = np.array([ref_freq, new_freq])
                _, p, _ = _stats.chi2_contingency(table)
                p_values[col] = float(p)
                test_used[col] = "chi2"
            except Exception:
                pass

    features_shifted = [col for col, p in p_values.items() if p < threshold]

    # Severity: fraction of features that shifted
    n_total = len(p_values)
    n_shifted = len(features_shifted)
    shifted = n_shifted > 0

    if n_total == 0 or n_shifted == 0:
        severity = "none"
    else:
        frac = n_shifted / n_total
        if frac < 0.1:
            severity = "low"
        elif frac < 0.3:
            severity = "medium"
        else:
            severity = "high"

    # F1: distinguishable is always bool — True if any feature has p < threshold
    distinguishable: bool = any(p < threshold for p in p_values.values())

    return DriftResult(
        shifted=shifted,
        features=p_values,
        features_shifted=sorted(features_shifted),
        severity=severity,
        n_reference=len(reference),
        n_new=len(new),
        threshold=threshold,
        feature_tests=test_used,
        distinguishable=distinguishable,
    )


# ---------------------------------------------------------------------------
# Adversarial validation
# ---------------------------------------------------------------------------

def _adversarial_drift(
    *,
    reference: pd.DataFrame,
    new: pd.DataFrame,
    shared_cols: list[str],
    algorithm: str,
    seed: int,
    threshold: float,
) -> DriftResult:
    """Train a classifier to distinguish reference from new data.

    Returns DriftResult with auc, distinguishable, and train_scores populated.
    """
    import warnings

    import numpy as np

    from ._scoring import _roc_auc as _roc_auc_score
    from .split import _stratified_kfold

    # Only numeric columns for the classifier
    numeric_cols = [
        c for c in shared_cols
        if pd.api.types.is_numeric_dtype(reference[c])
        and pd.api.types.is_numeric_dtype(new[c])
    ]
    if not numeric_cols:
        from ._types import DataError
        raise DataError(
            "drift(method='adversarial') found no numeric shared columns. "
            "Adversarial validation requires numeric features."
        )

    ref_X = reference[numeric_cols].copy()
    new_X = new[numeric_cols].copy()

    # Combine: reference = class 0, new = class 1
    combined_X = pd.concat([ref_X, new_X], ignore_index=True)
    y = np.array([0] * len(reference) + [1] * len(new))

    # Fill NaN with column median (computed on combined)
    col_medians = combined_X.median()
    combined_X = combined_X.fillna(col_medians)

    clf = _make_adversarial_clf(algorithm, seed)

    # OOF predictions for unbiased AUC estimate
    oof_proba = np.zeros(len(y), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, val_idx in _stratified_kfold(y, k=5, seed=seed):
            if hasattr(clf, "get_params"):
                _clf_fold = type(clf)(**clf.get_params())
            else:
                _p = {k: v for k, v in clf.__dict__.items()
                      if not k.startswith("_") and k != "classes_"}
                _clf_fold = type(clf)(**_p)
            _clf_fold.fit(combined_X.iloc[train_idx], y[train_idx])
            oof_proba[val_idx] = _clf_fold.predict_proba(combined_X.iloc[val_idx])[:, 1]

    auc = float(_roc_auc_score(y, oof_proba))
    distinguishable = auc > 0.6

    # Sanity check: AUC near 0.5 may indicate model failed to converge
    if auc < 0.55:
        warnings.warn(
            f"drift() adversarial AUC = {auc:.3f} (near 0.5) — train and test "
            "distributions appear similar. Adversarial validation may not be "
            "informative. Consider method='statistical' for per-feature analysis.",
            UserWarning,
            stacklevel=3,
        )

    # Feature importances: fit on full data
    clf_full = _make_adversarial_clf(algorithm, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_full.fit(combined_X, y)

    raw_imp = getattr(clf_full, "feature_importances_", None)
    if raw_imp is not None:
        imp_dict: dict[str, float] = dict(zip(numeric_cols, raw_imp.tolist()))
    else:
        imp_dict = {c: 1.0 / len(numeric_cols) for c in numeric_cols}

    # Top discriminative features sorted by importance
    top_features = sorted(imp_dict, key=lambda c: imp_dict[c], reverse=True)[:10]

    # Per-row train_scores: reference rows only, with original index
    train_scores = pd.Series(
        oof_proba[: len(reference)], index=reference.index, name="train_score"
    )

    # Severity based on AUC
    if auc < 0.55:
        severity = "none"
    elif auc < 0.65:
        severity = "low"
    elif auc < 0.80:
        severity = "medium"
    else:
        severity = "high"

    return DriftResult(
        shifted=distinguishable,
        features=imp_dict,
        features_shifted=top_features,
        severity=severity,
        n_reference=len(reference),
        n_new=len(new),
        threshold=threshold,
        auc=auc,
        distinguishable=distinguishable,
        train_scores=train_scores,
    )


def _make_adversarial_clf(algorithm: str, seed: int):
    """Create a classifier for adversarial validation."""
    if algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=100,
                random_state=seed,
                eval_metric="logloss",
                verbosity=0,
            )
        except ImportError:
            pass  # fall through to random_forest
    # Try Rust RF (zero sklearn dependency)
    try:
        from ._rust import HAS_RUST, _RustRandomForestClassifier
        if HAS_RUST:
            return _RustRandomForestClassifier(
                n_estimators=100, random_state=seed,
            )
    except ImportError:
        pass

    # Fall back to sklearn RF
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as e:
        from ._types import ConfigError
        raise ConfigError(
            "Adversarial drift detection requires xgboost, ml-py, or scikit-learn. "
            "Install one with: pip install xgboost  or  pip install scikit-learn"
        ) from e

    from . import _engines
    return RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        n_jobs=_engines._screen_n_jobs(),
    )
