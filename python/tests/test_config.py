"""Tests for ml.config()."""
import pytest

import ml


def test_config_set_get():
    """config() sets and gets values."""
    original = ml.config()
    ml.config(n_jobs=2)
    assert ml.config()["n_jobs"] == 2
    ml.config(n_jobs=original["n_jobs"])  # Restore


def test_config_unknown_key_error():
    """config() raises ConfigError for unknown keys."""
    with pytest.raises(ml.ConfigError, match="Unknown config key"):
        ml.config(nonexistent=42)


def test_config_n_jobs_propagates(small_classification_data):
    """config(n_jobs=1) affects parallelism without error."""
    original = ml.config()["n_jobs"]
    ml.config(n_jobs=1)
    try:
        s = ml.split(data=small_classification_data, target="target", seed=42)
        model = ml.fit(data=s.train, target="target", seed=42)
        assert model is not None
    finally:
        ml.config(n_jobs=original)  # Restore original value


def test_config_returns_dict():
    """config() with no args returns a dict with all keys."""
    cfg = ml.config()
    assert isinstance(cfg, dict)
    assert "n_jobs" in cfg
    assert "verbose" in cfg
