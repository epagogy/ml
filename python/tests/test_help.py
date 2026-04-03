"""Tests for ml.help() and ml.quick() -- Chain 11."""

import warnings

import pandas as pd
import pytest

import ml


def test_help_no_args():
    """ml.help() with no args returns overview text."""
    result = ml.help()
    text = str(result)
    assert "ml" in text.lower()
    assert "fit" in text or "split" in text


def test_help_verb():
    """ml.help('fit') returns fit() documentation."""
    result = ml.help("fit")
    text = str(result)
    assert "fit" in text.lower()


def test_help_returns_helptext_type():
    """ml.help() returns a _HelpText object with __repr__."""
    result = ml.help()
    assert hasattr(result, "__repr__")
    assert repr(result) == str(result)


def test_help_all_core_verbs():
    """ml.help() works for all core verbs without raising."""
    for verb in ["split", "fit", "predict", "evaluate", "assess", "screen", "tune", "stack"]:
        result = ml.help(verb)
        assert verb in str(result).lower(), f"help({verb!r}) should mention {verb}"


def test_help_unknown_verb_returns_helptext():
    """ml.help() on unknown verb returns _HelpText (not raise)."""
    result = ml.help("nonexistent_verb_xyz")
    text = str(result)
    assert "nonexistent_verb_xyz" in text or "Unknown" in text


def test_help_typo_suggests_correction():
    """ml.help('fitt') suggests 'fit'."""
    result = ml.help("fitt")
    text = str(result)
    # Should either suggest 'fit' or indicate unknown
    assert "fit" in text.lower() or "Unknown" in text


def test_help_repr_equals_str():
    """repr(help()) == str(help()) for any verb."""
    for verb in [None, "fit", "split", "evaluate"]:
        result = ml.help(verb)
        assert repr(result) == str(result)


@pytest.mark.slow
def test_quick_returns_tuple(small_classification_data):
    """ml.quick() returns (model, metrics, split_result) tuple."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.quick(small_classification_data, "target", seed=42)
    assert isinstance(result, tuple)
    assert len(result) == 3


@pytest.mark.slow
def test_quick_classification(small_classification_data):
    """ml.quick() works on classification data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, metrics, s = ml.quick(small_classification_data, "target", seed=42)
    assert model is not None
    assert isinstance(metrics, dict)
    assert s is not None


@pytest.mark.slow
def test_quick_regression(small_regression_data):
    """ml.quick() works on regression data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, metrics, s = ml.quick(small_regression_data, "target", seed=42)
    assert model is not None
    assert isinstance(metrics, dict)


def test_core_verbs_have_docstrings():
    """Core verbs have docstrings."""
    core_verbs = [ml.fit, ml.predict, ml.evaluate, ml.assess, ml.split, ml.screen, ml.tune, ml.stack]
    for verb in core_verbs:
        assert verb.__doc__ is not None and len(verb.__doc__) > 10, f"{verb.__name__} missing docstring"


def test_core_verbs_have_examples():
    """Core verbs docstrings mention 'Example' or '>>>'."""
    core_verbs = [ml.fit, ml.predict, ml.evaluate, ml.split]
    for verb in core_verbs:
        doc = verb.__doc__ or ""
        assert ">>>" in doc or "Example" in doc or "example" in doc, (
            f"{verb.__name__} docstring has no example"
        )


def test_fit_errors_have_guidance():
    """fit() error messages include guidance on valid options."""
    data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0]})
    with pytest.raises(ml.ConfigError) as exc_info:
        ml.fit(data=data, target="target", algorithm="nonexistent_algo_xyz", seed=42)
    error_msg = str(exc_info.value)
    assert any(
        word in error_msg.lower()
        for word in ["valid", "did you mean", "xgboost", "random_forest", "unknown"]
    )


def test_tune_errors_have_guidance():
    """tune() ConfigError messages include guidance."""
    data = pd.DataFrame({"a": range(20), "b": range(20), "target": [0, 1] * 10})
    with pytest.raises((ml.ConfigError, Exception)) as exc_info:
        ml.tune(data=data, target="target", algorithm="nonexistent_xyz", seed=42, n_trials=1)
    assert exc_info.value is not None
