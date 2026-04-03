"""Tests for ml.report() — Chain 12."""

import os

import pytest

import ml


def test_report_generates_html(small_classification_data, tmp_path):
    """report() generates an HTML file."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    path = str(tmp_path / "report.html")
    result = ml.report(model, s.valid, path=path)
    assert os.path.exists(path)
    assert result == path


def test_report_returns_path(small_classification_data, tmp_path):
    """report() returns the path to the generated file."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    path = str(tmp_path / "my_report.html")
    result = ml.report(model, s.valid, path=path)
    assert isinstance(result, str)
    assert result == path


def test_report_contains_metrics(small_classification_data, tmp_path):
    """report HTML contains Metrics section."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    path = str(tmp_path / "report.html")
    ml.report(model, s.valid, path=path)
    with open(path) as f:
        html = f.read()
    assert "Metrics" in html or "metric" in html.lower()


def test_report_contains_model_summary(small_classification_data, tmp_path):
    """report HTML contains Model Summary section."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    path = str(tmp_path / "report.html")
    ml.report(model, s.valid, path=path)
    with open(path) as f:
        html = f.read()
    assert "Model Summary" in html or model._algorithm in html


def test_report_is_valid_html(small_classification_data, tmp_path):
    """report() generates a file that starts with HTML doctype."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    path = str(tmp_path / "report.html")
    ml.report(model, s.valid, path=path)
    with open(path) as f:
        html = f.read()
    assert html.strip().startswith("<!DOCTYPE html>") or html.strip().startswith("<html")


def test_report_regression(small_regression_data, tmp_path):
    """report() works for regression models."""
    import warnings
    s = ml.split(data=small_regression_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    path = str(tmp_path / "reg_report.html")
    result = ml.report(model, s.valid, path=path)
    assert os.path.exists(path)
    assert result == path


def test_report_invalid_model_raises(small_classification_data, tmp_path):
    """report() raises ConfigError when model is not a fitted Model."""
    path = str(tmp_path / "bad_report.html")
    with pytest.raises(ml.ConfigError):
        ml.report("not_a_model", small_classification_data, path=path)


def test_report_invalid_data_raises(small_classification_data, tmp_path):
    """report() raises DataError when data is not a DataFrame."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    path = str(tmp_path / "bad_report.html")
    with pytest.raises(ml.DataError):
        ml.report(model, [1, 2, 3], path=path)


def test_report_in_namespace():
    """report is accessible as ml.report."""
    assert hasattr(ml, "report")
    assert callable(ml.report)


def test_report_default_path(small_classification_data, tmp_path):
    """report() default path 'report.html' is returned as absolute path.

    W29 P1 fix: report() now converts path to absolute before returning.
    The file is still named 'report.html' in cwd, but the return value is absolute.
    """
    import os
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    orig_dir = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = ml.report(model, s.valid)
        # W29 fix: result is now an absolute path
        assert os.path.isabs(result), f"report() should return absolute path, got: {result}"
        assert result.endswith("report.html")
        assert os.path.exists(result)
    finally:
        os.chdir(orig_dir)
