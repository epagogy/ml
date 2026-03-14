"""Tests for __repr__ on ml types."""
import ml


def test_model_repr(small_classification_data):
    """Model has informative __repr__."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    r = repr(model)
    assert "Model(" in r
    assert "algorithm" in r


def test_split_result_repr(small_classification_data):
    """SplitResult has informative __repr__."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    r = repr(s)
    assert "SplitResult" in r or "train" in r.lower()


def test_tuning_result_repr(small_classification_data):
    """TuningResult has informative __repr__."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    tuned = ml.tune(data=s.train, target="target", algorithm="random_forest", seed=42, n_trials=2)
    r = repr(tuned)
    assert "TuningResult" in r or "best" in r.lower()


def test_metrics_repr(small_classification_data):
    """Metrics has informative __repr__."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    metrics = ml.evaluate(model, s.valid)
    r = repr(metrics)
    assert len(r) > 5  # Has some content
