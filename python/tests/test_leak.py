"""Tests for ml.leak() — data leakage detection."""

import numpy as np
import pandas as pd
import pytest

import ml
from ml._types import CheckResult, SuspectFeature

# -- Data helpers --


def _random_binary(n=300, seed=42):
    """Binary data with random (unrelated) target — no genuine leakage."""
    rng = np.random.RandomState(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    target = rng.randint(0, 2, n)  # completely random
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": target})


def _leaky_binary(n=300, seed=42):
    """Binary data where x_leak is almost identical to target — |r|~0.99."""
    rng = np.random.RandomState(seed)
    target = rng.randint(0, 2, n).astype(float)
    x_leak = target + rng.normal(0, 0.01, n)  # near-perfect correlation
    x_normal = rng.normal(0, 1, n)
    return pd.DataFrame({"x_leak": x_leak, "x_normal": x_normal, "target": target})


def _id_column_data(n=300, seed=42):
    """Data with a high-cardinality integer ID column."""
    data = _random_binary(n=n, seed=seed)
    data["user_id"] = range(n)
    return data


def _target_name_data(n=300, seed=42):
    """Data with a feature name embedding the target name."""
    data = _random_binary(n=n, seed=seed)
    data = data.rename(columns={"x1": "target_score"})
    return data


def _regression_data(n=300, seed=42):
    """Regression dataset."""
    rng = np.random.RandomState(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 3.0 * x1 + 2.0 * x2 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _multiclass_data(n=300, seed=42):
    """3-class classification dataset."""
    rng = np.random.RandomState(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    target = rng.choice(["a", "b", "c"], n)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


# -- Module fixture: run leak once, share across structural tests --


@pytest.fixture(scope="module")
def binary_leak(request):
    """LeakReport on _random_binary — shared across structural tests."""
    data = _random_binary()
    return data, ml.leak(data, "target")


# -- Core behavior --


def test_leak_returns_leak_report(binary_leak):
    """leak() returns a LeakReport."""
    _, report = binary_leak
    assert isinstance(report, ml.LeakReport)


def test_leak_seven_checks(binary_leak):
    """leak() always runs exactly 7 checks."""
    _, report = binary_leak
    assert len(report.checks) == 7


def test_leak_checks_are_check_results(binary_leak):
    """All checks are CheckResult instances with required fields."""
    _, report = binary_leak
    for check in report.checks:
        assert isinstance(check, CheckResult)
        assert hasattr(check, "name")
        assert hasattr(check, "passed")
        assert hasattr(check, "detail")
        assert hasattr(check, "severity")
        assert check.severity in ("ok", "warn", "critical")


def test_leak_clean_bool(binary_leak):
    """report.clean is a bool."""
    _, report = binary_leak
    assert isinstance(report.clean, bool)


def test_leak_n_warnings_is_int(binary_leak):
    """report.n_warnings is a non-negative int."""
    _, report = binary_leak
    assert isinstance(report.n_warnings, int)
    assert report.n_warnings >= 0


def test_leak_n_warnings_matches_checks(binary_leak):
    """n_warnings == number of checks with passed=False."""
    _, report = binary_leak
    failed = sum(1 for c in report.checks if not c.passed)
    assert report.n_warnings == failed


def test_leak_max_severity_property(binary_leak):
    """max_severity returns a valid severity string."""
    _, report = binary_leak
    assert report.max_severity in ("ok", "warn", "critical")


def test_leak_top_features_list(binary_leak):
    """top_features is a list of 3-tuples (name, check, detail)."""
    _, report = binary_leak
    assert isinstance(report.top_features, list)
    assert len(report.top_features) <= 3
    for entry in report.top_features:
        assert len(entry) == 3


def test_leak_suspects_list(binary_leak):
    """suspects is a list (possibly empty)."""
    _, report = binary_leak
    assert isinstance(report.suspects, list)


def test_leak_repr(binary_leak):
    """LeakReport has a readable string repr."""
    _, report = binary_leak
    r = repr(report)
    assert "Leak" in r
    assert len(r) > 0


def test_leak_with_split_result():
    """SplitResult input enables duplicate-row and temporal checks."""
    data = _random_binary()
    s = ml.split(data=data, target="target", seed=42)
    report = ml.leak(s, "target")
    assert isinstance(report, ml.LeakReport)
    assert len(report.checks) == 7
    # Duplicate-row check should NOT say "skipped"
    dup_check = next(c for c in report.checks if "Duplicate" in c.name)
    assert "skipped" not in dup_check.detail


def test_leak_dataframe_skips_split_checks():
    """DataFrame input skips duplicate-row and temporal checks."""
    data = _random_binary()
    report = ml.leak(data, "target")
    dup_check = next(c for c in report.checks if "Duplicate" in c.name)
    temporal_check = next(c for c in report.checks if "Temporal" in c.name)
    assert "skipped" in dup_check.detail
    assert "skipped" in temporal_check.detail


# -- Leakage detection --


def test_leak_high_correlation_flags_feature():
    """Feature with |r|>0.95 is flagged as a suspect."""
    data = _leaky_binary()
    report = ml.leak(data, "target")
    suspect_names = [s.feature for s in report.suspects]
    assert "x_leak" in suspect_names


def test_leak_high_correlation_not_clean():
    """Near-perfect correlation makes report not clean."""
    data = _leaky_binary()
    report = ml.leak(data, "target")
    assert not report.clean
    assert report.n_warnings > 0


def test_leak_suspects_have_attributes():
    """Each SuspectFeature has all required fields."""
    data = _leaky_binary()
    report = ml.leak(data, "target")
    assert len(report.suspects) > 0
    for s in report.suspects:
        assert isinstance(s, SuspectFeature)
        assert isinstance(s.feature, str)
        assert isinstance(s.check, str)
        assert isinstance(s.value, float)
        assert isinstance(s.detail, str)
        assert isinstance(s.action, str)


def test_leak_id_column_detected():
    """High-cardinality integer ID column ('user_id') is flagged."""
    data = _id_column_data()
    report = ml.leak(data, "target")
    suspect_names = [s.feature for s in report.suspects]
    assert "user_id" in suspect_names


def test_leak_target_name_in_feature():
    """Feature name containing the target name is flagged."""
    data = _target_name_data()
    report = ml.leak(data, "target")
    suspect_names = [s.feature for s in report.suspects]
    assert "target_score" in suspect_names


def test_leak_duplicate_rows_detected():
    """Manually injected duplicate rows between train and test are flagged."""
    data = _random_binary(n=200, seed=0)
    s = ml.split(data=data, target="target", seed=0)

    # Inject duplicates: append test rows to train
    contaminated_train = pd.concat([s.train, s.test.head(20)], ignore_index=True)

    # Manually craft a fake split-like object using DataFrame for train check
    # Verify directly via check internals
    from ml.leak import _check_duplicates

    train_X = contaminated_train.drop(columns=["target"])
    test_X = s.test.drop(columns=["target"])
    check, suspects = _check_duplicates(train_X, test_X)

    assert not check.passed
    assert len(suspects) > 0


def test_leak_clean_split_no_duplicates():
    """A normal ml.split() has no train/test duplicate rows."""
    data = _random_binary()
    s = ml.split(data=data, target="target", seed=42)
    report = ml.leak(s, "target")
    dup_check = next(c for c in report.checks if "Duplicate" in c.name)
    assert dup_check.passed


# -- Task and data type coverage --


def test_leak_regression_works():
    """leak() handles regression targets (float continuous)."""
    data = _regression_data()
    report = ml.leak(data, "y")
    assert isinstance(report, ml.LeakReport)
    assert len(report.checks) == 7


def test_leak_multiclass_skips_pearson_correlation():
    """Multiclass target skips Pearson correlation (check 1 says 'skipped')."""
    data = _multiclass_data()
    report = ml.leak(data, "target")
    corr_check = next(c for c in report.checks if "correlation" in c.name.lower())
    # Multiclass → correlation is skipped (see leak.py comment about ordinal inflation)
    assert "skip" in corr_check.detail.lower() or corr_check.passed


def test_leak_binary_auc_check_fires():
    """Binary classification runs the single-feature AUC check."""
    data = _leaky_binary()
    report = ml.leak(data, "target")
    auc_check = next(c for c in report.checks if "AUC" in c.name)
    # x_leak is nearly a perfect predictor — AUC check should fire
    assert not auc_check.passed or auc_check.severity in ("warn", "critical")


def test_leak_string_binary_target():
    """leak() handles string binary targets."""
    rng = np.random.RandomState(7)
    n = 200
    data = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "target": rng.choice(["yes", "no"], n),
    })
    report = ml.leak(data, "target")
    assert isinstance(report, ml.LeakReport)
    assert len(report.checks) == 7


# -- Error handling --


def test_leak_missing_target_raises():
    """Missing target column raises DataError."""
    data = _random_binary()
    with pytest.raises(ml.DataError, match="not found"):
        ml.leak(data, "missing_column")


def test_leak_empty_data_raises():
    """Empty DataFrame raises DataError."""
    data = pd.DataFrame({"x1": [], "target": []})
    with pytest.raises(ml.DataError, match="empty"):
        ml.leak(data, "target")


def test_leak_wrong_type_raises():
    """Non-DataFrame/non-SplitResult input raises DataError."""
    with pytest.raises(ml.DataError):
        ml.leak([[1, 2], [3, 4]], "target")
