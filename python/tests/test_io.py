"""Tests for save() and load()."""

import pytest

import ml


def test_save_and_load_basic(small_classification_data, tmp_path):
    """Test basic save and load."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    path = tmp_path / "model.ml"
    ml.save(model=model, path=str(path))
    loaded = ml.load(path=str(path))

    assert loaded.task == model.task
    assert loaded.algorithm == model.algorithm
    assert loaded.features == model.features
    assert loaded.target == model.target
    assert loaded.seed == model.seed


def test_save_auto_adds_extension(small_classification_data, tmp_path):
    """Test save auto-adds .pyml extension."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    path = tmp_path / "model"  # no extension
    ml.save(model=model, path=str(path))

    # Should create model.pyml
    assert (tmp_path / "model.pyml").exists()


def test_load_predictions_match(small_classification_data, tmp_path):
    """Test loaded model produces same predictions."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    preds_before = model.predict(s.valid)

    path = tmp_path / "model.ml"
    ml.save(model=model, path=str(path))
    loaded = ml.load(path=str(path))
    preds_after = loaded.predict(s.valid)

    import pandas as pd
    pd.testing.assert_series_equal(preds_before, preds_after)


def test_load_file_not_found_error(tmp_path):
    """Test load raises on missing file."""
    path = tmp_path / "nonexistent.ml"

    with pytest.raises(FileNotFoundError, match="not found"):
        ml.load(path=str(path))


def test_save_load_cv_scores(small_classification_data, tmp_path):
    """Test CV scores are persisted."""
    cv = ml.split(data=small_classification_data, target="target", folds=2, seed=42)
    model = ml.fit(data=cv, target="target", seed=42)

    path = tmp_path / "model.ml"
    ml.save(model=model, path=str(path))
    loaded = ml.load(path=str(path))

    assert loaded.scores_ is not None
    assert loaded.scores_ == model.scores_


def test_save_load_version_stored(small_classification_data, tmp_path):
    """Test version is stored in save file."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    path = tmp_path / "model.ml"
    ml.save(model=model, path=str(path))

    # Load should succeed (same version)
    loaded = ml.load(path=str(path))
    assert loaded is not None


def test_save_load_n_train_persists(small_classification_data, tmp_path):
    """n_train is preserved through save/load."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    path = tmp_path / "model.ml"
    ml.save(model=model, path=str(path))
    loaded = ml.load(path=str(path))

    assert loaded.n_train == model.n_train
    assert loaded.n_train == len(s.train)


def test_save_accepts_path_object(small_classification_data, tmp_path):
    """save() accepts pathlib.Path, not just str."""

    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    path = tmp_path / "model.ml"
    ml.save(model=model, path=path)  # Path object, not str
    assert path.exists()

    loaded = ml.load(path=path)
    assert loaded.task == model.task


def test_load_rejects_untrusted_types(tmp_path):
    """Test load() rejects files with unknown types (security whitelist)."""
    import skops.io as sio

    # Create a file with a custom untrusted type
    class EvilType:
        pass

    save_dict = {"__type__": "Model", "evil": EvilType()}
    path = tmp_path / "evil.ml"
    sio.dump(save_dict, str(path))

    with pytest.raises(ml.ConfigError, match="untrusted types"):
        ml.load(path=str(path))
