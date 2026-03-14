"""Tests for ml.tokenize() — explicit text preprocessing verb."""

import numpy as np
import pandas as pd
import pytest

import ml
from ml._types import ConfigError, DataError
from ml.tokenize import Tokenizer, tokenize

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def text_df():
    """Small DataFrame with one text column and one numeric column."""
    return pd.DataFrame({
        "review": ["great product", "terrible quality", "okay I guess", "loved it", "meh"],
        "price": [10.0, 5.0, 8.0, 12.0, 7.0],
        "label": ["pos", "neg", "neg", "pos", "neg"],
    })


@pytest.fixture
def multi_text_df():
    """DataFrame with two text columns."""
    return pd.DataFrame({
        "title": ["Great buy", "Waste of money", "Average", "Excellent", "So-so"],
        "body": ["Works well", "Broke after a week", "Does the job", "Amazing", "Nothing special"],
        "score": [5, 1, 3, 5, 2],
        "label": ["pos", "neg", "neg", "pos", "neg"],
    })


# ── Basic interface ───────────────────────────────────────────────────────────

def test_tokenize_returns_textprep(text_df):
    prep = tokenize(text_df, columns=["review"])
    assert isinstance(prep, Tokenizer)


def test_tokenize_via_ml_namespace(text_df):
    prep = ml.tokenize(text_df, columns=["review"])
    assert isinstance(prep, Tokenizer)


def test_textprep_attributes(text_df):
    prep = tokenize(text_df, columns=["review"], max_features=20)
    assert prep.columns == ["review"]
    assert prep.max_features == 20
    assert "review" in prep._vectorizers


def test_tokenize_multi_column(multi_text_df):
    prep = tokenize(multi_text_df, columns=["title", "body"], max_features=10)
    assert prep.columns == ["title", "body"]
    assert "title" in prep._vectorizers
    assert "body" in prep._vectorizers


# ── transform() output shape and structure ────────────────────────────────────

def test_transform_drops_text_column(text_df):
    prep = tokenize(text_df, columns=["review"], max_features=5)
    out = prep.transform(text_df)
    assert "review" not in out.columns


def test_transform_preserves_numeric_columns(text_df):
    prep = tokenize(text_df, columns=["review"], max_features=5)
    out = prep.transform(text_df)
    assert "price" in out.columns
    assert "label" in out.columns


def test_transform_adds_tfidf_columns(text_df):
    prep = tokenize(text_df, columns=["review"], max_features=10)
    out = prep.transform(text_df)
    tfidf_cols = [c for c in out.columns if c.startswith("review_tfidf_")]
    assert len(tfidf_cols) > 0
    assert len(tfidf_cols) <= 10


def test_transform_multi_column_adds_both_blocks(multi_text_df):
    prep = tokenize(multi_text_df, columns=["title", "body"], max_features=5)
    out = prep.transform(multi_text_df)
    title_cols = [c for c in out.columns if c.startswith("title_tfidf_")]
    body_cols = [c for c in out.columns if c.startswith("body_tfidf_")]
    assert len(title_cols) > 0
    assert len(body_cols) > 0


def test_transform_preserves_index(text_df):
    text_df = text_df.copy()
    text_df.index = [10, 20, 30, 40, 50]
    prep = tokenize(text_df, columns=["review"], max_features=5)
    out = prep.transform(text_df)
    assert list(out.index) == [10, 20, 30, 40, 50]


def test_transform_output_is_numeric(text_df):
    prep = tokenize(text_df, columns=["review"], max_features=5)
    out = prep.transform(text_df)
    tfidf_cols = [c for c in out.columns if c.startswith("review_tfidf_")]
    assert out[tfidf_cols].dtypes.apply(lambda d: np.issubdtype(d, np.floating)).all()


def test_transform_preserves_row_count(text_df):
    prep = tokenize(text_df, columns=["review"], max_features=5)
    out = prep.transform(text_df)
    assert len(out) == len(text_df)


# ── No leakage: fit on train, transform valid ─────────────────────────────────

def test_tokenize_fit_on_train_transform_valid():
    """Vocabulary fitted only on train — valid words outside vocab silently ignored."""
    train = pd.DataFrame({
        "text": ["hello world", "foo bar", "baz qux"],
        "y": [0, 1, 0],
    })
    valid = pd.DataFrame({
        "text": ["hello friend", "foo unknown"],
        "y": [1, 0],
    })
    prep = tokenize(train, columns=["text"], max_features=10)
    out_train = prep.transform(train)
    out_valid = prep.transform(valid)

    # Same feature columns (vocab fixed at fit time)
    train_tfidf = [c for c in out_train.columns if c.startswith("text_tfidf_")]
    valid_tfidf = [c for c in out_valid.columns if c.startswith("text_tfidf_")]
    assert train_tfidf == valid_tfidf


# ── NaN handling ──────────────────────────────────────────────────────────────

def test_tokenize_handles_nan_in_text():
    df = pd.DataFrame({
        "review": ["good product", None, "bad service", np.nan],
        "y": [1, 0, 0, 1],
    })
    prep = tokenize(df, columns=["review"], max_features=5)
    out = prep.transform(df)
    assert len(out) == 4
    assert not out[[c for c in out.columns if "tfidf" in c]].isnull().any().any()


# ── save / load round-trip ────────────────────────────────────────────────────

def test_tokenize_save_load_roundtrip(tmp_path, text_df):
    prep = tokenize(text_df, columns=["review"], max_features=10)
    path = str(tmp_path / "prep.pyml")
    ml.save(prep, path)
    prep2 = ml.load(path)

    assert isinstance(prep2, Tokenizer)
    assert prep2.columns == prep.columns
    assert prep2.max_features == prep.max_features

    # Transforms must be identical
    out1 = prep.transform(text_df)
    out2 = prep2.transform(text_df)
    pd.testing.assert_frame_equal(out1, out2)


def test_tokenize_save_load_multi_column(tmp_path, multi_text_df):
    prep = tokenize(multi_text_df, columns=["title", "body"], max_features=5)
    path = str(tmp_path / "prep.pyml")
    ml.save(prep, path)
    prep2 = ml.load(path)
    assert prep2.columns == ["title", "body"]
    assert "title" in prep2._vectorizers
    assert "body" in prep2._vectorizers


# ── Integration: tokenize → fit → predict ─────────────────────────────────────

def test_tokenize_fit_predict_pipeline(text_df):
    """Full workflow: tokenize train, fit, tokenize valid, predict."""
    train = text_df.iloc[:4].copy()
    valid = text_df.iloc[4:].copy()

    prep = ml.tokenize(train, columns=["review"], max_features=20)
    model = ml.fit(data=prep.transform(train), target="label", seed=42)
    preds = ml.predict(model=model, data=prep.transform(valid))

    assert isinstance(preds, pd.Series)
    assert len(preds) == len(valid)
    assert set(preds.unique()).issubset({"pos", "neg"})


# ── Error handling ────────────────────────────────────────────────────────────

def test_tokenize_non_dataframe_raises():
    with pytest.raises(DataError):
        tokenize(pd.Series(["a", "b"]), columns=["col"])


def test_tokenize_empty_columns_raises(text_df):
    with pytest.raises(DataError):
        tokenize(text_df, columns=[])


def test_tokenize_missing_column_raises(text_df):
    with pytest.raises(DataError):
        tokenize(text_df, columns=["nonexistent"])


def test_tokenize_max_features_zero_raises(text_df):
    with pytest.raises(ConfigError):
        tokenize(text_df, columns=["review"], max_features=0)


def test_transform_missing_column_raises(text_df):
    prep = tokenize(text_df, columns=["review"], max_features=5)
    bad_df = text_df.drop(columns=["review"])
    with pytest.raises(DataError):
        prep.transform(bad_df)


def test_transform_non_dataframe_raises(text_df):
    prep = tokenize(text_df, columns=["review"], max_features=5)
    with pytest.raises(DataError):
        prep.transform(pd.Series(["a", "b"]))


# ── repr ──────────────────────────────────────────────────────────────────────

def test_textprep_repr(text_df):
    prep = tokenize(text_df, columns=["review"], max_features=50)
    r = repr(prep)
    assert "Tokenizer" in r
    assert "review" in r
    assert "50" in r
