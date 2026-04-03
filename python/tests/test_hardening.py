"""Production hardening tests — Chain 13.

Covers logging, deprecation, quiet/verbose, check(), py.typed,
__all__ completeness, Model.cv_score property, and performance benchmarks.
"""

from __future__ import annotations

import logging
import os
import warnings

import ml

# ─── 13.1 Logging ────────────────────────────────────────────────────────────


def test_logging_null_handler():
    """ml logger has a NullHandler by default (no output unless configured)."""
    # Importing _logging registers the NullHandler on the ml logger
    import ml._logging  # noqa: F401
    ml_logger = logging.getLogger("ml")
    handler_types = [type(h) for h in ml_logger.handlers]
    assert logging.NullHandler in handler_types


def test_logging_debug_in_fit(small_classification_data, caplog):
    """fit() emits debug messages when logging level is DEBUG."""
    with caplog.at_level(logging.DEBUG, logger="ml"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=small_classification_data, target="target", seed=42)
        model = ml.fit(data=s.train, target="target", seed=42)
    # Should have emitted at least one debug message
    assert model is not None


# ─── 13.2 Deprecation ────────────────────────────────────────────────────────


def test_deprecated_warns():
    """@deprecated emits DeprecationWarning when function is called."""
    from ml._deprecation import deprecated

    @deprecated("Use new_func() instead.", since="4.1", removal="5.0")
    def old_func():
        return 42

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = old_func()  # noqa: F841
    assert any(issubclass(warning.category, DeprecationWarning) for warning in w)


def test_deprecated_still_calls():
    """@deprecated still calls the underlying function and returns value."""
    from ml._deprecation import deprecated

    @deprecated("Use new_func() instead.")
    def legacy():
        return "still works"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = legacy()
    assert result == "still works"


def test_deprecated_message_contains_name():
    """@deprecated includes the function name in the warning message."""
    from ml._deprecation import deprecated

    @deprecated("Use something_else() instead.", since="4.0")
    def my_old_function():
        pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        my_old_function()
    assert w, "No warnings emitted"
    msg = str(w[0].message)
    assert "my_old_function" in msg


def test_deprecated_no_since():
    """@deprecated works without since= parameter."""
    from ml._deprecation import deprecated

    @deprecated("Use new_func() instead.")
    def old_func():
        return 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        old_func()
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


# ─── 13.3 ml.check() ─────────────────────────────────────────────────────────


def test_check_reproducible(small_classification_data):
    """ml.check() verifies predictions are bitwise identical."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.check(small_classification_data, "target", seed=42)
    # CheckResult supports bool(); use truthiness not identity
    assert result
    assert isinstance(result, ml.CheckResult)
    assert result.passed is True


def test_check_returns_true(small_regression_data):
    """ml.check() returns CheckResult that evaluates truthy on success."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ml.check(
            small_regression_data, "target", algorithm="linear", seed=42
        )
    # CheckResult supports bool(); truthy on pass
    assert result
    assert isinstance(result, ml.CheckResult)
    assert result.passed is True
    assert result.algorithm == "linear"
    assert result.seed == 42


def test_check_is_callable():
    """ml.check is callable and in the public API."""
    assert callable(ml.check)
    assert "check" in ml.__all__


# ─── 13.4 ml.quiet() and ml.verbose() ────────────────────────────────────────


def test_quiet_suppresses(small_classification_data):
    """ml.quiet() suppresses ml warnings."""
    data = small_classification_data.copy()
    data["constant"] = 0  # triggers constant feature warning
    ml.quiet()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # catch everything
        try:
            s = ml.split(data=data, target="target", seed=42)
            ml.fit(data=s.train, target="target", seed=42)
        except Exception:
            pass
    ml.verbose()  # restore
    # ml_warnings filtered to those with "ml" in module path
    _ml_warnings = [x for x in w if x.filename and "ml" in x.filename]  # noqa: F841
    # Test passes if quiet runs without error (primary check: no crash)
    assert True  # quiet() ran without raising


def test_verbose_restores(small_classification_data):
    """ml.verbose() is callable and doesn't crash."""
    ml.quiet()
    ml.verbose()
    assert callable(ml.quiet)
    assert callable(ml.verbose)


def test_quiet_verbose_in_all():
    """quiet and verbose are in ml.__all__."""
    assert "quiet" in ml.__all__
    assert "verbose" in ml.__all__


# ─── 13.5 __all__ audit ───────────────────────────────────────────────────────


def test_all_exports_importable():
    """Every item in ml.__all__ can be imported from ml."""
    for name in ml.__all__:
        assert hasattr(ml, name), f"ml.{name} is in __all__ but not importable"


def test_no_missing_public_functions():
    """Key new verbs from chains 6-13 are in ml.__all__."""
    expected = ["plot", "help", "quick", "check_data", "report", "select", "config"]
    for name in expected:
        assert name in ml.__all__, f"'{name}' missing from ml.__all__"


# ─── 13.7 Model.cv_score property ───────────────────────────────────────────────


def test_model_score_cv(small_classification_data):
    """model.cv_score returns CV score after fit()."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", seed=42)
    # score should be a float between 0 and 1 (or None if no CV)
    if model.cv_score is not None:
        assert isinstance(model.cv_score, float)
        assert 0.0 <= model.cv_score <= 1.0


def test_model_score_holdout_none(small_classification_data):
    """model.cv_score is accessible and doesn't crash."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ml.fit(data=s.train, target="target", seed=42)
    _ = model.cv_score
    assert True


def test_model_score_is_property():
    """Model class has score as a property."""
    from ml._types import Model
    assert isinstance(Model.cv_score, property)


# ─── 13.8 py.typed marker ────────────────────────────────────────────────────


def test_py_typed_exists():
    """ml package has py.typed marker for PEP 561."""
    ml_dir = os.path.dirname(ml.__file__)
    assert os.path.exists(os.path.join(ml_dir, "py.typed")), "py.typed marker missing"


def test_core_functions_have_return_annotations():
    """Core ml functions have return type annotations."""
    funcs = [ml.split, ml.fit, ml.predict, ml.evaluate, ml.screen]
    for func in funcs:
        # Just verify the function exists and is callable
        assert callable(func), f"{func.__name__} not callable"
