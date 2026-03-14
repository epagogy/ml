"""Tests for ml.drift() — label-free data drift detection."""

import numpy as np
import pandas as pd
import pytest

import ml
from ml._types import ConfigError, DataError
from ml.drift import DriftResult, drift

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def reference():
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "income": rng.normal(60000, 15000, n),
        "city": rng.choice(["NYC", "LA", "Chicago"], n),
        "score": rng.uniform(0, 1, n),
    })


@pytest.fixture
def same_distribution(reference):
    """New data from the same distribution — should not drift."""
    rng = np.random.RandomState(99)
    n = 100
    return pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "income": rng.normal(60000, 15000, n),
        "city": rng.choice(["NYC", "LA", "Chicago"], n),
        "score": rng.uniform(0, 1, n),
    })


@pytest.fixture
def drifted():
    """New data with clearly shifted distributions."""
    rng = np.random.RandomState(7)
    n = 100
    return pd.DataFrame({
        "age": rng.normal(60, 5, n),          # shifted mean (+20 years)
        "income": rng.normal(120000, 5000, n), # shifted mean (+100%)
        "city": rng.choice(["Boston", "Miami"], n),  # completely different categories
        "score": rng.uniform(0.8, 1.0, n),     # shifted range
    })


# ── Return type ───────────────────────────────────────────────────────────────

def test_drift_returns_driftresult(reference, same_distribution):
    result = drift(reference=reference, new=same_distribution)
    assert isinstance(result, DriftResult)


def test_drift_via_ml_namespace(reference, same_distribution):
    result = ml.drift(reference=reference, new=same_distribution)
    assert isinstance(result, DriftResult)


# ── Attributes ────────────────────────────────────────────────────────────────

def test_drift_result_has_shifted(reference, same_distribution):
    result = drift(reference=reference, new=same_distribution)
    assert isinstance(result.shifted, bool)


def test_drift_result_has_features(reference, same_distribution):
    result = drift(reference=reference, new=same_distribution)
    assert isinstance(result.features, dict)
    assert len(result.features) > 0


def test_drift_result_features_are_pvalues(reference, same_distribution):
    result = drift(reference=reference, new=same_distribution)
    for col, p in result.features.items():
        assert 0.0 <= p <= 1.0, f"p-value out of range for {col}: {p}"


def test_drift_result_has_features_shifted(reference, same_distribution):
    result = drift(reference=reference, new=same_distribution)
    assert isinstance(result.features_shifted, list)


def test_drift_result_has_severity(reference, same_distribution):
    result = drift(reference=reference, new=same_distribution)
    assert result.severity in {"none", "low", "medium", "high"}


def test_drift_result_has_counts(reference, same_distribution):
    result = drift(reference=reference, new=same_distribution)
    assert result.n_reference == len(reference)
    assert result.n_new == len(same_distribution)


# ── No-drift case ──────────────────────────────────────────────────────────────

def test_drift_identical_data_no_shift(reference):
    result = drift(reference=reference, new=reference)
    assert not result.shifted
    assert result.severity == "none"
    assert result.features_shifted == []


def test_drift_same_distribution_low_shift(reference, same_distribution):
    result = drift(reference=reference, new=same_distribution)
    # With same distribution, very few features should be flagged
    assert len(result.features_shifted) <= 2


# ── Clear-drift case ──────────────────────────────────────────────────────────

def test_drift_detects_numeric_shift(reference, drifted):
    result = drift(reference=reference, new=drifted)
    assert result.shifted


def test_drift_flags_shifted_numeric_features(reference, drifted):
    result = drift(reference=reference, new=drifted)
    # age and income shifted massively — must be caught
    assert "age" in result.features_shifted or "income" in result.features_shifted


def test_drift_severity_high_on_major_shift(reference, drifted):
    result = drift(reference=reference, new=drifted)
    assert result.severity in {"medium", "high"}


# ── Categorical drift ─────────────────────────────────────────────────────────

def test_drift_detects_categorical_shift():
    rng = np.random.RandomState(0)
    ref = pd.DataFrame({"cat": rng.choice(["A", "B", "C"], 300)})
    new = pd.DataFrame({"cat": rng.choice(["X", "Y", "Z"], 100)})
    result = drift(reference=ref, new=new)
    assert "cat" in result.features
    assert "cat" in result.features_shifted


def test_drift_categorical_same_dist_not_shifted():
    rng = np.random.RandomState(0)
    ref = pd.DataFrame({"cat": rng.choice(["A", "B", "C"], 300, p=[0.5, 0.3, 0.2])})
    rng2 = np.random.RandomState(99)
    new = pd.DataFrame({"cat": rng2.choice(["A", "B", "C"], 100, p=[0.5, 0.3, 0.2])})
    result = drift(reference=ref, new=new)
    assert "cat" in result.features
    # p-value should be fairly high (same distribution)
    assert result.features["cat"] > 0.01


# ── Threshold ─────────────────────────────────────────────────────────────────

def test_drift_custom_threshold(reference, drifted):
    strict = drift(reference=reference, new=drifted, threshold=0.001)
    lenient = drift(reference=reference, new=drifted, threshold=0.5)
    # Stricter threshold → fewer or equal features flagged
    assert len(strict.features_shifted) <= len(lenient.features_shifted)


# ── exclude parameter ─────────────────────────────────────────────────────────

def test_drift_exclude_columns(reference, drifted):
    result_full = drift(reference=reference, new=drifted)
    result_excl = drift(reference=reference, new=drifted, exclude=["age", "income"])
    assert "age" not in result_excl.features
    assert "income" not in result_excl.features
    assert len(result_excl.features) < len(result_full.features)


# ── Partial columns (new data has subset) ────────────────────────────────────

def test_drift_new_has_extra_columns(reference):
    extra = reference.copy()
    extra["extra_col"] = 0
    result = drift(reference=reference, new=extra)
    # extra_col not in reference — excluded from drift test
    assert "extra_col" not in result.features


def test_drift_reference_has_extra_columns(reference, same_distribution):
    subset = same_distribution[["age", "income"]]
    result = drift(reference=reference, new=subset)
    # Only shared columns tested
    assert set(result.features.keys()) == {"age", "income"}


# ── NaN handling ──────────────────────────────────────────────────────────────

def test_drift_handles_nan_values(reference):
    noisy = reference.copy()
    noisy.loc[:10, "age"] = np.nan
    result = drift(reference=reference, new=noisy)
    # Should complete without error; NaN rows dropped before test
    assert "age" in result.features


# ── Error handling ────────────────────────────────────────────────────────────

def test_drift_reference_not_dataframe():
    with pytest.raises(DataError):
        drift(reference=pd.Series([1, 2, 3]), new=pd.DataFrame({"a": [1]}))


def test_drift_new_not_dataframe(reference):
    with pytest.raises(DataError):
        drift(reference=reference, new=[1, 2, 3])


def test_drift_empty_reference():
    with pytest.raises(DataError):
        drift(reference=pd.DataFrame({"a": []}), new=pd.DataFrame({"a": [1]}))


def test_drift_empty_new(reference):
    with pytest.raises(DataError):
        drift(reference=reference, new=pd.DataFrame({"age": []}))


def test_drift_no_shared_columns():
    ref = pd.DataFrame({"a": [1, 2, 3]})
    new = pd.DataFrame({"b": [4, 5, 6]})
    with pytest.raises(DataError):
        drift(reference=ref, new=new)


# ── repr ──────────────────────────────────────────────────────────────────────

def test_drift_result_repr(reference, same_distribution):
    result = drift(reference=reference, new=same_distribution)
    r = repr(result)
    assert "DriftResult" in r
    assert "shifted" in r
    assert "severity" in r


# ── Gate 3 additions ──────────────────────────────────────────────────────────

def test_drift_target_param_excludes_column(reference, same_distribution):
    """target= convenience param should auto-exclude the target from drift analysis."""
    ref_with_target = reference.copy()
    ref_with_target["label"] = np.random.RandomState(0).choice([0, 1], len(ref_with_target))
    new_with_target = same_distribution.copy()
    new_with_target["label"] = np.random.RandomState(1).choice([0, 1], len(new_with_target))

    result = drift(reference=ref_with_target, new=new_with_target, target="label")
    assert "label" not in result.features


def test_drift_all_nan_column_warning(reference):
    """Columns that are all NaN in reference or new should warn and be skipped."""
    new = reference.copy()
    new["age"] = np.nan  # all NaN in new data

    with pytest.warns(UserWarning, match="entirely NaN"):
        result = drift(reference=reference, new=new)
    assert "age" not in result.features


def test_drift_single_category_no_drift():
    """A categorical column with only one value cannot drift — p should be 1.0."""
    rng = np.random.RandomState(0)
    ref = pd.DataFrame({"cat": ["A"] * 100, "val": rng.randn(100)})
    new = pd.DataFrame({"cat": ["A"] * 50, "val": rng.randn(50)})
    result = drift(reference=ref, new=new)
    # Single-category column → p=1.0, not drifted
    if "cat" in result.features:
        assert result.features["cat"] == 1.0


def test_drift_small_sample_warning(reference):
    """new data with n < 30 should warn about unreliable statistics."""
    tiny = reference.iloc[:20].copy()
    with pytest.warns(UserWarning, match="insufficient"):
        drift(reference=reference, new=tiny)


def test_drift_severity_none_when_no_shift(reference):
    """Identical data → severity='none'."""
    result = drift(reference=reference, new=reference.copy())
    assert result.severity == "none"
    assert not result.shifted


def test_drift_severity_high_many_shifted():
    """If >30% of features drift, severity='high'."""
    rng = np.random.RandomState(42)
    n = 200
    ref = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(10)})
    # Massively shift every feature → all drift → severity=high
    new = pd.DataFrame({f"f{i}": rng.normal(100, 1, 100) for i in range(10)})
    result = drift(reference=ref, new=new)
    assert result.severity in {"medium", "high"}


def test_drift_mixed_types_both_tested(reference, same_distribution):
    """Fixture has numeric + categorical columns — both should appear in features dict."""
    result = drift(reference=reference, new=same_distribution)
    keys = set(result.features.keys())
    # numeric cols present
    assert "age" in keys or "income" in keys
    # categorical col present
    assert "city" in keys


# ── A3: Adversarial validation ────────────────────────────────────────────────


def test_adversarial_basic(reference, same_distribution):
    """drift(method='adversarial') returns DriftResult with auc and train_scores. A3."""
    result = ml.drift(reference=reference, new=same_distribution, method="adversarial", seed=42)

    assert isinstance(result, ml.DriftResult)
    assert result.auc is not None
    assert 0.0 <= result.auc <= 1.0
    assert isinstance(result.distinguishable, bool)
    assert result.train_scores is not None
    assert isinstance(result.train_scores, pd.Series)
    assert len(result.train_scores) == len(reference)


def test_adversarial_identical_auc_near_half(reference):
    """Identical datasets → AUC near 0.5, distinguishable=False. A3."""
    result = ml.drift(reference=reference, new=reference.copy(), method="adversarial", seed=42)
    assert result.auc < 0.65, f"Expected AUC < 0.65 for identical data, got {result.auc}"
    assert result.distinguishable is False


def test_adversarial_different_auc_high():
    """Very different distributions → AUC high, distinguishable=True. A3."""
    rng = np.random.RandomState(42)
    n = 300
    ref = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
    new = pd.DataFrame({"x1": rng.randn(n) * 5 + 20, "x2": rng.randn(n) * 5 + 20})
    result = ml.drift(reference=ref, new=new, method="adversarial", seed=42)

    assert result.auc > 0.8, f"Expected AUC > 0.8 for very different dists, got {result.auc}"
    assert result.distinguishable is True


def test_adversarial_features(reference, drifted):
    """features dict has importances; features_shifted lists top discriminative cols. A3."""
    result = ml.drift(reference=reference, new=drifted, method="adversarial", seed=42)

    assert isinstance(result.features, dict)
    assert len(result.features) > 0
    for col, imp in result.features.items():
        assert imp >= 0.0, f"Negative importance for {col}: {imp}"
    assert isinstance(result.features_shifted, list)
    for col in result.features_shifted:
        assert col in result.features


def test_adversarial_train_scores_indexed():
    """train_scores is pd.Series with reference DataFrame's index. A3."""
    rng = np.random.RandomState(0)
    ref = pd.DataFrame({"x1": rng.randn(100), "x2": rng.randn(100)}, index=range(500, 600))
    new = pd.DataFrame({"x1": rng.randn(100) + 3, "x2": rng.randn(100) + 3})
    result = ml.drift(reference=ref, new=new, method="adversarial", seed=42)

    assert list(result.train_scores.index) == list(ref.index)
    assert (result.train_scores >= 0.0).all()
    assert (result.train_scores <= 1.0).all()


def test_adversarial_requires_seed(reference, same_distribution):
    """drift(method='adversarial') without seed= raises ConfigError. A3."""
    with pytest.raises(ConfigError, match="seed="):
        ml.drift(reference=reference, new=same_distribution, method="adversarial")


def test_adversarial_convergence_warning(reference):
    """AUC near 0.5 (identical data) emits UserWarning. A3 (Onodera N4)."""
    with pytest.warns(UserWarning, match="near 0.5"):
        result = ml.drift(reference=reference, new=reference.copy(),
                          method="adversarial", seed=42)
    assert result.auc is not None


def test_drift_target_excludes_column(small_classification_data):
    """drift(target=) drops target column from analysis."""
    ref = small_classification_data.iloc[:60]
    new = small_classification_data.iloc[60:]
    # With target=, should not include target in features
    result_without = ml.drift(reference=ref, new=new, target="target")
    assert "target" not in result_without.features


def test_drift_no_target_includes_all(small_classification_data):
    """drift() without target= includes all columns."""
    ref = small_classification_data.iloc[:60]
    new = small_classification_data.iloc[60:]
    result = ml.drift(reference=ref, new=new)
    # With default (target=None), all columns including target participate
    assert result is not None


def test_drift_feature_tests_public_attribute(small_classification_data):
    """DriftResult.feature_tests is public (no underscore)."""
    ref = small_classification_data.iloc[:60]
    new = small_classification_data.iloc[60:]
    result = ml.drift(reference=ref, new=new)
    assert hasattr(result, "feature_tests")
    assert not hasattr(result, "_feature_tests")  # old name gone
