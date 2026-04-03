"""Tests for ml.check_data() — Chain 12."""

import numpy as np
import pandas as pd
import pytest

import ml


def _make_clean_data():
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "a": rng.randn(100),
        "b": rng.randn(100),
        "target": rng.choice(["yes", "no"], 100),
    })


def test_check_data_id_column():
    """check_data warns on columns with 100% unique string values (ID-like)."""
    data = _make_clean_data()
    data["user_id"] = [f"user_{i}" for i in range(100)]
    report = ml.check_data(data, "target")
    assert any("user_id" in w for w in report.warnings)


def test_check_data_zero_variance():
    """check_data warns on constant (zero-variance) columns."""
    data = _make_clean_data()
    data["constant"] = 42
    report = ml.check_data(data, "target")
    assert any("constant" in w for w in report.warnings)


def test_check_data_high_null():
    """check_data warns on columns with >50% missing values."""
    data = _make_clean_data()
    data["mostly_null"] = np.nan
    data.loc[data.index[:10], "mostly_null"] = 1.0  # 10% non-null
    report = ml.check_data(data, "target")
    assert any("mostly_null" in w for w in report.warnings)


def test_check_data_imbalance():
    """check_data warns on severe class imbalance (<5% minority class)."""
    rng = np.random.RandomState(42)
    labels = ["majority"] * 98 + ["rare"] * 2
    data = pd.DataFrame({
        "a": rng.randn(100),
        "target": labels,
    })
    report = ml.check_data(data, "target")
    assert any("rare" in w or "2%" in w or "imbalance" in w.lower()
               for w in report.warnings)


def test_check_data_duplicates():
    """check_data warns on >10% duplicate rows."""
    clean = _make_clean_data()
    # Add 20 duplicates of first row (>10% of 100 rows)
    dupes = pd.concat([clean] + [clean.iloc[[0]]] * 20, ignore_index=True)
    report = ml.check_data(dupes, "target")
    assert any("duplic" in w.lower() for w in report.warnings)


def test_check_data_clean_returns_report():
    """check_data returns a CheckReport even for clean data."""
    data = _make_clean_data()
    report = ml.check_data(data, "target")
    assert report is not None
    assert isinstance(report.warnings, list)
    assert isinstance(report.errors, list)


def test_check_data_returns_report():
    """check_data returns CheckReport with .warnings, .errors, .has_issues."""
    data = _make_clean_data()
    report = ml.check_data(data, "target")
    assert hasattr(report, "warnings")
    assert hasattr(report, "errors")
    assert hasattr(report, "has_issues")


def test_check_data_severity_error_raises():
    """check_data(severity='error') raises DataError when issues found."""
    data = _make_clean_data()
    data["constant"] = 99  # zero-variance column
    with pytest.raises(ml.DataError):
        ml.check_data(data, "target", severity="error")


def test_check_data_in_namespace():
    """check_data is accessible as ml.check_data."""
    assert hasattr(ml, "check_data")
    assert callable(ml.check_data)


def test_check_report_in_namespace():
    """CheckReport is accessible as ml.CheckReport."""
    assert hasattr(ml, "CheckReport")


def test_check_data_has_issues_false_for_clean():
    """check_data().has_issues is False for clean data."""
    # Clean numeric data with 2-class target
    rng = np.random.RandomState(99)
    data = pd.DataFrame({
        "a": rng.randn(200),
        "b": rng.randn(200),
        "c": rng.randn(200),
        "target": (rng.randn(200) > 0).astype(int),
    })
    report = ml.check_data(data, "target")
    # Clean data should have no warnings
    assert not report.has_issues


def test_check_data_report_repr():
    """CheckReport.__repr__ contains diagnostic info."""
    data = _make_clean_data()
    data["constant_col"] = 99
    report = ml.check_data(data, "target")
    text = repr(report)
    assert "WARNING" in text or "WARN" in text or "constant" in text.lower()


def test_check_data_invalid_target_raises():
    """check_data raises DataError for unknown target column."""
    data = _make_clean_data()
    with pytest.raises(ml.DataError):
        ml.check_data(data, "nonexistent_target")


def test_check_data_invalid_data_type():
    """check_data raises DataError when data is not a DataFrame."""
    with pytest.raises(ml.DataError):
        ml.check_data([1, 2, 3], "target")


def test_check_data_redundant_features():
    """check_data warns about highly correlated feature pairs (|r| > 0.95)."""
    import numpy as np
    rng = np.random.RandomState(42)
    x = rng.randn(100)
    data = pd.DataFrame({
        "x1": x,
        "x2": x * 1.01 + rng.randn(100) * 0.001,  # r > 0.99
        "x3": rng.randn(100),
        "target": rng.choice([0, 1], 100),
    })
    report = ml.check_data(data, "target")
    redundancy_warnings = [w for w in report.warnings if "correlated" in w.lower()]
    assert len(redundancy_warnings) == 1
    assert "x1" in redundancy_warnings[0] and "x2" in redundancy_warnings[0]


def test_check_data_no_redundancy_warning_for_uncorrelated():
    """check_data does NOT warn when features are uncorrelated."""
    import numpy as np
    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "x1": rng.randn(100),
        "x2": rng.randn(100),
        "target": rng.choice([0, 1], 100),
    })
    report = ml.check_data(data, "target")
    redundancy_warnings = [w for w in report.warnings if "correlated" in w.lower()]
    assert len(redundancy_warnings) == 0
