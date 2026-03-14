"""Tests for ml.plot() — Chain 9 Visualization Layer."""

import numpy as np
import pytest

import ml

# ── Matplotlib backend for headless CI ─────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.figure
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MPL_AVAILABLE, reason="matplotlib not installed")


def _close(fig):
    """Close figure to avoid ResourceWarning."""
    import matplotlib.pyplot as plt
    plt.close(fig)


# ── 9.1 dispatcher + importance + confusion + ROC ─────────────────────────


def test_plot_matplotlib_not_installed(monkeypatch):
    """ConfigError if matplotlib not installed."""
    import sys
    from unittest import mock

    # Temporarily hide matplotlib by patching the import inside ml.plot
    with mock.patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None,
                                        "matplotlib.figure": None}):
        # Reimport plot module to trigger the ImportError path
        # Monkeypatch plot to force the ImportError path
        original_plot = ml.plot
        def _bad_plot(*args, **kwargs):
            raise ml.ConfigError("ml.plot() requires matplotlib. Install with: pip install ml[plots]")
        monkeypatch.setattr(ml, "plot", _bad_plot)
        with pytest.raises(ml.ConfigError, match="matplotlib"):
            ml.plot(object())
        monkeypatch.setattr(ml, "plot", original_plot)


def test_plot_importance(small_classification_data):
    """importance plot returns Figure for Model."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    fig = ml.plot(model, kind="importance")
    assert isinstance(fig, matplotlib.figure.Figure)
    _close(fig)


def test_plot_importance_top_n(small_classification_data):
    """importance plot respects top_n."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    fig = ml.plot(model, kind="importance", top_n=2)
    assert isinstance(fig, matplotlib.figure.Figure)
    _close(fig)


def test_plot_confusion(small_classification_data):
    """confusion plot returns Figure for Model + data."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    fig = ml.plot(model, data=s.valid, kind="confusion")
    assert isinstance(fig, matplotlib.figure.Figure)
    _close(fig)


def test_plot_roc_binary(small_classification_data):
    """roc plot for binary classification."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    fig = ml.plot(model, data=s.valid, kind="roc")
    assert isinstance(fig, matplotlib.figure.Figure)
    _close(fig)


def test_plot_roc_multiclass(multiclass_data):
    """roc plot for multiclass — OVR + macro."""
    s = ml.split(data=multiclass_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    fig = ml.plot(model, data=s.valid, kind="roc")
    assert isinstance(fig, matplotlib.figure.Figure)
    _close(fig)


# ── 9.2 calibration + residual + learning_curve ───────────────────────────


def test_plot_calibration(small_classification_data):
    """calibration plot returns Figure."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    fig = ml.plot(model, data=s.valid, kind="calibration")
    assert isinstance(fig, matplotlib.figure.Figure)
    _close(fig)


def test_plot_residual(small_regression_data):
    """residual plot for regression."""
    s = ml.split(data=small_regression_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="linear", seed=42)
    fig = ml.plot(model, data=s.valid, kind="residual")
    assert isinstance(fig, matplotlib.figure.Figure)
    _close(fig)


def test_plot_residual_regression_only(small_classification_data):
    """residual plot raises ConfigError for classification."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    with pytest.raises(ml.ConfigError, match="regression"):
        ml.plot(model, data=s.valid, kind="residual")


def test_plot_learning_curve(small_classification_data):
    """learning_curve plot from EnoughResult."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    result = ml.enough(data=s.train, target="target", seed=42)
    fig = ml.plot(result, kind="learning_curve")
    assert isinstance(fig, matplotlib.figure.Figure)
    _close(fig)


# ── 9.3 leaderboard + drift + waterfall + PDP ─────────────────────────────


def test_plot_leaderboard(small_classification_data):
    """leaderboard plot from compare() output."""
    import warnings
    s = ml.split(data=small_classification_data, target="target", seed=42)
    m1 = ml.fit(data=s.train, target="target", algorithm="logistic", seed=42)
    m2 = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb = ml.compare([m1, m2], data=s.valid, warn_test=False)
    fig = ml.plot(lb, kind="leaderboard")
    assert isinstance(fig, matplotlib.figure.Figure)
    _close(fig)


def test_plot_drift(small_classification_data):
    """drift plot from drift() output."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    result = ml.drift(reference=s.train, new=s.valid)
    fig = ml.plot(result, kind="drift")
    assert isinstance(fig, matplotlib.figure.Figure)
    _close(fig)


def test_plot_waterfall_shap(small_classification_data):
    """waterfall plot from SHAP Explanation."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    try:
        exp = ml.explain(model, data=s.valid, method="shap", seed=42)
        if exp.shap_values is None:
            pytest.skip("shap not available")
    except (ml.ConfigError, ImportError):
        pytest.skip("shap not available")
    fig = ml.plot(exp, kind="waterfall", idx=0)
    assert isinstance(fig, matplotlib.figure.Figure)
    _close(fig)


def test_plot_pdp_single(small_classification_data):
    """pdp plot for single feature."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
    # Get first numeric feature from model features
    numeric_features = [
        c for c in model._features
        if small_classification_data[c].dtype in [np.float64, np.int64, float, int]
    ]
    if not numeric_features:
        pytest.skip("No numeric features available")
    fig = ml.plot(model, data=s.valid, kind="pdp", feature=numeric_features[0])
    assert isinstance(fig, matplotlib.figure.Figure)
    _close(fig)


def test_plot_pdp_pair(small_classification_data):
    """pdp plot for feature pair (2D PDP)."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
    numeric_features = [
        c for c in model._features
        if small_classification_data[c].dtype in [np.float64, np.int64, float, int]
    ]
    if len(numeric_features) < 2:
        pytest.skip("Need at least 2 numeric features")
    fig = ml.plot(model, data=s.valid, kind="pdp", feature=numeric_features[:2])
    assert isinstance(fig, matplotlib.figure.Figure)
    _close(fig)


# ── 9.4 Export + auto-detection ───────────────────────────────────────────


def test_plot_in_all():
    """ml.plot is in __all__ and is callable."""
    assert "plot" in ml.__all__
    assert callable(ml.plot)


def test_plot_auto_selects_kind(small_classification_data):
    """Auto-detect kind from Model → importance."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    fig = ml.plot(model)  # no kind= → auto-detect → importance
    assert isinstance(fig, matplotlib.figure.Figure)
    _close(fig)


def test_plot_missing_data_error(small_classification_data):
    """DataError when data= is required but not given."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    with pytest.raises((ml.DataError, ml.ConfigError)):
        ml.plot(model, kind="confusion")  # needs data=
