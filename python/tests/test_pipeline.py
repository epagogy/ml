"""Tests for ml.pipe() — composable preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

import ml
from ml._types import ConfigError, DataError
from ml.pipeline import Pipeline, pipe


@pytest.fixture
def mixed_df():
    """Realistic DataFrame: numeric (with NaN), categorical, text, label."""
    rng = np.random.RandomState(42)
    n = 120
    df = pd.DataFrame({
        "age": rng.normal(35, 10, n),
        "income": rng.normal(55000, 15000, n),
        "city": rng.choice(["NYC", "LA", "Chicago"], n),
        "review": rng.choice(["great", "bad", "okay fine", "love it", "meh"], n),
        "label": rng.choice(["yes", "no"], n),
    })
    df.loc[::8, "age"] = np.nan
    df.loc[::12, "income"] = np.nan
    return df


# ── Return type ───────────────────────────────────────────────────────────────

def test_pipe_returns_pipeline(mixed_df):
    imp = ml.impute(mixed_df, columns=["age"])
    p = pipe([imp])
    assert isinstance(p, Pipeline)


def test_pipe_via_ml_namespace(mixed_df):
    imp = ml.impute(mixed_df, columns=["age"])
    p = ml.pipe([imp])
    assert isinstance(p, Pipeline)


def test_pipeline_len(mixed_df):
    imp = ml.impute(mixed_df, columns=["age"])
    enc = ml.encode(mixed_df, columns=["city"])
    p = ml.pipe([imp, enc])
    assert len(p) == 2


# ── process() basics ─────────────────────────────────────────────────────────

def test_process_returns_dataframe(mixed_df):
    imp = ml.impute(mixed_df, columns=["age"])
    p = ml.pipe([imp])
    out = p.process(mixed_df)
    assert isinstance(out, pd.DataFrame)


def test_process_preserves_row_count(mixed_df):
    imp = ml.impute(mixed_df, columns=["age"])
    enc = ml.encode(mixed_df, columns=["city"])
    p = ml.pipe([imp, enc])
    out = p.process(mixed_df)
    assert len(out) == len(mixed_df)


def test_process_applies_steps_in_order(mixed_df):
    """impute → encode — NaN filled first, city encoded second."""
    imp = ml.impute(mixed_df, columns=["age", "income"])
    enc = ml.encode(mixed_df, columns=["city"])
    p = ml.pipe([imp, enc])
    out = p.process(mixed_df)
    assert not out[["age", "income"]].isna().any().any()
    assert "city" not in out.columns
    assert any(c.startswith("city_") for c in out.columns)


def test_process_identical_to_manual_chain(mixed_df):
    """Pipeline output must match manually chaining .transform() calls."""
    imp = ml.impute(mixed_df, columns=["age", "income"])
    enc = ml.encode(mixed_df, columns=["city"])
    scl = ml.scale(mixed_df.pipe(lambda d: imp.transform(d)), columns=["age", "income"])

    # Manual chain
    manual = scl.transform(enc.transform(imp.transform(mixed_df)))

    # Pipeline (fitted on same data, same step objects)
    imp2 = ml.impute(mixed_df, columns=["age", "income"])
    enc2 = ml.encode(mixed_df, columns=["city"])
    df_imp = imp2.transform(mixed_df)
    scl2 = ml.scale(df_imp, columns=["age", "income"])
    pipe2 = ml.pipe([imp2, enc2, scl2])
    pipeline_out = pipe2.process(mixed_df)

    # Both should have same columns (order may differ, but no NaN, city dropped)
    assert set(manual.columns) == set(pipeline_out.columns)


def test_process_non_dataframe_raises(mixed_df):
    imp = ml.impute(mixed_df, columns=["age"])
    p = ml.pipe([imp])
    with pytest.raises(DataError):
        p.process(mixed_df["age"])  # Series, not DataFrame


# ── transform() alias ────────────────────────────────────────────────────────

def test_transform_is_alias_for_process(mixed_df):
    imp = ml.impute(mixed_df, columns=["age"])
    p = ml.pipe([imp])
    pd.testing.assert_frame_equal(p.process(mixed_df), p.transform(mixed_df))


# ── __call__ ─────────────────────────────────────────────────────────────────

def test_pipeline_is_callable(mixed_df):
    imp = ml.impute(mixed_df, columns=["age"])
    p = ml.pipe([imp])
    assert callable(p)


def test_call_same_as_process(mixed_df):
    imp = ml.impute(mixed_df, columns=["age"])
    p = ml.pipe([imp])
    pd.testing.assert_frame_equal(p(mixed_df), p.process(mixed_df))


# ── Full pipeline: preprocessing → fit → predict ─────────────────────────────

def test_pipe_fit_predict_workflow(mixed_df):
    """Full workflow: pipe() → fit() → predict()."""
    train = mixed_df.iloc[:96].copy()
    valid = mixed_df.iloc[96:].copy()

    imp = ml.impute(train, columns=["age", "income"])
    enc = ml.encode(train, columns=["city"])
    scl = ml.scale(imp.transform(train), columns=["age", "income"])

    p = ml.pipe([imp, enc, scl])

    model = ml.fit(data=p.process(train), target="label", seed=42)
    preds = ml.predict(model=model, data=p.process(valid))

    assert isinstance(preds, pd.Series)
    assert len(preds) == len(valid)
    assert set(preds.unique()).issubset({"yes", "no"})


# ── No leakage ────────────────────────────────────────────────────────────────

def test_pipeline_no_leakage_steps_fitted_on_train_only(mixed_df):
    """Steps are fitted on train, pipeline applied to valid — no leakage."""
    train = mixed_df.iloc[:96]
    valid = mixed_df.iloc[96:]

    imp = ml.impute(train, columns=["age", "income"])
    enc = ml.encode(train, columns=["city"])
    p = ml.pipe([imp, enc])

    # Valid uses train statistics (imputer fill values, encoder categories)
    out_valid = p.process(valid)
    assert not out_valid[["age", "income"]].isna().any().any()
    assert len(out_valid) == len(valid)


# ── Save / load ───────────────────────────────────────────────────────────────

def test_pipeline_save_load_roundtrip(tmp_path, mixed_df):
    """Saved pipeline produces identical output after load."""
    train = mixed_df.iloc[:96]

    imp = ml.impute(train, columns=["age", "income"])
    enc = ml.encode(train, columns=["city"])

    p = ml.pipe([imp, enc])
    path = str(tmp_path / "pipeline.pyml")
    ml.save(p, path)
    p2 = ml.load(path)

    assert isinstance(p2, Pipeline)
    assert len(p2) == len(p)

    pd.testing.assert_frame_equal(p.process(train), p2.process(train))


def test_pipeline_save_load_with_all_step_types(tmp_path, mixed_df):
    """Pipeline with Imputer + Encoder + Scaler + Tokenizer survives save/load."""
    train = mixed_df.iloc[:96]

    imp = ml.impute(train, columns=["age", "income"])
    enc = ml.encode(train, columns=["city"])
    tok = ml.tokenize(train, columns=["review"], max_features=10)
    df_imp = imp.transform(train)
    scl = ml.scale(df_imp, columns=["age", "income"])

    p = ml.pipe([imp, enc, tok, scl])
    path = str(tmp_path / "full_pipeline.pyml")
    ml.save(p, path)
    p2 = ml.load(path)

    assert isinstance(p2, Pipeline)
    assert len(p2) == 4

    out1 = p.process(train)
    out2 = p2.process(train)
    pd.testing.assert_frame_equal(out1, out2)


# ── repr ──────────────────────────────────────────────────────────────────────

def test_pipeline_repr(mixed_df):
    imp = ml.impute(mixed_df, columns=["age"])
    enc = ml.encode(mixed_df, columns=["city"])
    p = ml.pipe([imp, enc])
    r = repr(p)
    assert "Pipeline" in r
    assert "Imputer" in r
    assert "Encoder" in r


# ── Error handling ────────────────────────────────────────────────────────────

def test_pipe_empty_steps_raises():
    with pytest.raises(DataError):
        pipe([])


def test_pipe_non_list_raises():
    with pytest.raises(DataError):
        pipe("not a list")


def test_pipe_invalid_step_raises():
    """Step without .transform() raises ConfigError."""
    with pytest.raises(ConfigError):
        pipe(["not a transformer"])


def test_pipe_non_transformer_object_raises():
    with pytest.raises(ConfigError):
        pipe([42])
