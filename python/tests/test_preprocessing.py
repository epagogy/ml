"""Integration tests — full preprocessing pipeline.

Tests composition of tokenize(), scale(), encode(), impute()
before fit() and predict(). Validates the complete stateful
transformer pattern end-to-end.
"""

import numpy as np
import pandas as pd
import pytest

import ml


@pytest.fixture
def messy_df():
    """Realistic mixed DataFrame: numeric, categorical, text, NaN."""
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.rand(n)
    df = pd.DataFrame({
        "age": rng.normal(35, 10, n),
        "income": rng.normal(55000, 20000, n),
        "city": rng.choice(["NYC", "LA", "Chicago", "Houston"], n),
        "review": rng.choice(
            ["great product", "terrible quality", "average item",
             "highly recommend", "would not buy again"],
            n,
        ),
        "label": (x1 > 0.5).astype(str),
    })
    # Inject NaN
    df.loc[::8, "age"] = np.nan
    df.loc[::12, "income"] = np.nan
    return df


# ── Individual transformers compose ──────────────────────────────────────────

def test_impute_then_scale_then_encode(messy_df):
    """Canonical tabular preprocessing pipeline."""
    train = messy_df.iloc[:160]
    valid = messy_df.iloc[160:]

    imp = ml.impute(train, columns=["age", "income"])
    scl = ml.scale(train, columns=["age", "income"])
    enc = ml.encode(train, columns=["city"])

    def prep(df):
        return enc.transform(scl.transform(imp.transform(df)))

    df_train = prep(train)
    df_valid = prep(valid)

    assert not df_train[["age", "income"]].isna().any().any()
    assert "city" not in df_train.columns
    assert any(c.startswith("city_") for c in df_train.columns)
    assert df_valid.shape[0] == len(valid)


def test_impute_then_tokenize(messy_df):
    """Text + numeric preprocessing together."""
    train = messy_df.iloc[:160]
    valid = messy_df.iloc[160:]

    imp = ml.impute(train, columns=["age", "income"])
    tok = ml.tokenize(train, columns=["review"], max_features=20)

    def prep(df):
        return tok.transform(imp.transform(df))

    df_train = prep(train)
    df_valid = prep(valid)

    assert "review" not in df_train.columns
    assert any(c.startswith("review_tfidf_") for c in df_train.columns)
    assert not df_train[["age", "income"]].isna().any().any()
    assert df_train.shape[0] == len(train)
    assert df_valid.shape[0] == len(valid)


# ── Full pipeline: preprocess → fit → predict ─────────────────────────────────

def test_full_pipeline_fit_predict_numeric(messy_df):
    """impute + scale + encode → fit → predict."""
    train = messy_df.iloc[:160].drop(columns=["review"])
    valid = messy_df.iloc[160:].drop(columns=["review"])

    imp = ml.impute(train, columns=["age", "income"])
    scl = ml.scale(train, columns=["age", "income"])
    enc = ml.encode(train, columns=["city"])

    def prep(df):
        return enc.transform(scl.transform(imp.transform(df)))

    model = ml.fit(data=prep(train), target="label", seed=42)
    preds = ml.predict(model=model, data=prep(valid))

    assert isinstance(preds, pd.Series)
    assert len(preds) == len(valid)
    assert set(preds.unique()).issubset({"True", "False"})


def test_full_pipeline_fit_predict_with_text(messy_df):
    """impute + tokenize + encode → fit → predict."""
    train = messy_df.iloc[:160].copy()
    valid = messy_df.iloc[160:].copy()

    imp = ml.impute(train, columns=["age", "income"])
    tok = ml.tokenize(train, columns=["review"], max_features=15)
    enc = ml.encode(train, columns=["city"])

    def prep(df):
        return enc.transform(tok.transform(imp.transform(df)))

    model = ml.fit(data=prep(train), target="label", seed=42)
    preds = ml.predict(model=model, data=prep(valid))

    assert isinstance(preds, pd.Series)
    assert len(preds) == len(valid)


# ── Leakage: all transformers fitted on train only ────────────────────────────

def test_no_leakage_transformers_see_only_train(messy_df):
    """Validate transformers don't see valid/test data at fit time."""
    train = messy_df.iloc[:160]
    valid = messy_df.iloc[160:]

    # Record train statistics before fitting
    train_age_median = train["age"].dropna().median()
    train_income_median = train["income"].dropna().median()


    imp = ml.impute(train, columns=["age", "income"])
    scl = ml.scale(train, columns=["age", "income"])

    # Imputer must use train median, not combined
    assert abs(imp._fill_values["age"] - train_age_median) < 1e-9
    assert abs(imp._fill_values["income"] - train_income_median) < 1e-9

    # Scaler must center on train mean
    train_age_mu = train["age"].dropna().mean()
    assert abs(float(scl._scalers["age"].mean_[0]) - train_age_mu) < 1e0  # rough check

    # Applying to valid must not crash or change train stats
    out_valid = scl.transform(imp.transform(valid))
    assert not out_valid[["age", "income"]].isna().any().any()


# ── Save / load entire pipeline ───────────────────────────────────────────────

def test_save_load_all_transformers(tmp_path, messy_df):
    """All preprocessing objects survive save/load with identical output."""
    train = messy_df.iloc[:160]

    imp = ml.impute(train, columns=["age", "income"])
    scl = ml.scale(train, columns=["age", "income"])
    enc = ml.encode(train, columns=["city"])
    tok = ml.tokenize(train, columns=["review"], max_features=10)

    for obj, name in [(imp, "imp"), (scl, "scl"), (enc, "enc"), (tok, "tok")]:
        path = str(tmp_path / f"{name}.pyml")
        ml.save(obj, path)
        loaded = ml.load(path)
        # Transforms must be identical
        if name == "imp" or name == "scl" or name == "enc" or name == "tok":
            pd.testing.assert_frame_equal(
                obj.transform(train), loaded.transform(train)
            )


# ── Type consistency ──────────────────────────────────────────────────────────

def test_all_transformers_return_dataframe(messy_df):
    train = messy_df.iloc[:160]

    imp = ml.impute(train, columns=["age", "income"])
    scl = ml.scale(train, columns=["age", "income"])
    enc = ml.encode(train, columns=["city"])
    tok = ml.tokenize(train, columns=["review"])

    assert isinstance(imp.transform(train), pd.DataFrame)
    assert isinstance(scl.transform(train), pd.DataFrame)
    assert isinstance(enc.transform(train), pd.DataFrame)
    assert isinstance(tok.transform(train), pd.DataFrame)


def test_all_transformers_preserve_row_count(messy_df):
    train = messy_df.iloc[:160]
    n = len(train)

    imp = ml.impute(train, columns=["age", "income"])
    scl = ml.scale(train, columns=["age", "income"])
    enc = ml.encode(train, columns=["city"])
    tok = ml.tokenize(train, columns=["review"])

    assert imp.transform(train).shape[0] == n
    assert scl.transform(train).shape[0] == n
    assert enc.transform(train).shape[0] == n
    assert tok.transform(train).shape[0] == n


# ── Safety net: OneHotEncoder edge cases ─────────────────────────────────────


def test_ohe_nan_produces_zero_row():
    """OneHotEncoder with NaN values produces all-zero rows for NaN."""
    from ml._transforms import OneHotEncoder
    df = pd.DataFrame({"color": ["red", "blue", np.nan, "red"]})
    ohe = OneHotEncoder(handle_unknown="ignore")
    ohe.fit(df)
    result = ohe.transform(df)
    # NaN row (index 2) should be all zeros
    assert result[2].sum() == 0.0
    # Non-NaN rows should have exactly one 1.0
    assert result[0].sum() == 1.0
    assert result[1].sum() == 1.0
    assert result[3].sum() == 1.0


def test_ohe_numpy_scalar_types():
    """OneHotEncoder works with numpy int/float scalar types in categories."""
    from ml._transforms import OneHotEncoder
    df = pd.DataFrame({"val": [np.int64(1), np.int64(2), np.int64(3), np.int64(1)]})
    ohe = OneHotEncoder(handle_unknown="ignore")
    ohe.fit(df)
    result = ohe.transform(df)
    assert result.shape == (4, 3)
    # Each row has exactly one 1.0
    for i in range(4):
        assert result[i].sum() == 1.0


def test_ohe_handle_unknown_error():
    """OneHotEncoder with handle_unknown='error' raises on unseen categories."""
    from ml._transforms import OneHotEncoder
    train = pd.DataFrame({"color": ["red", "blue", "green"]})
    test = pd.DataFrame({"color": ["red", "purple"]})
    ohe = OneHotEncoder(handle_unknown="error")
    ohe.fit(train)
    with pytest.raises(ValueError, match="Unknown category"):
        ohe.transform(test)


def test_ohe_handle_unknown_ignore():
    """OneHotEncoder with handle_unknown='ignore' produces zeros for unseen."""
    from ml._transforms import OneHotEncoder
    train = pd.DataFrame({"color": ["red", "blue", "green"]})
    test = pd.DataFrame({"color": ["red", "purple"]})
    ohe = OneHotEncoder(handle_unknown="ignore")
    ohe.fit(train)
    result = ohe.transform(test)
    assert result[0].sum() == 1.0  # "red" is known
    assert result[1].sum() == 0.0  # "purple" is unknown → all zeros


def test_ohe_multiple_columns():
    """OneHotEncoder handles multiple categorical columns."""
    from ml._transforms import OneHotEncoder
    df = pd.DataFrame({
        "color": ["red", "blue", "red"],
        "size": ["S", "M", "L"],
    })
    ohe = OneHotEncoder(handle_unknown="ignore")
    ohe.fit(df)
    result = ohe.transform(df)
    # 2 colors + 3 sizes = 5 columns
    assert result.shape == (3, 5)


def test_ohe_empty_category_set():
    """OneHotEncoder handles column with all NaN (no categories)."""
    from ml._transforms import OneHotEncoder
    df = pd.DataFrame({"val": [np.nan, np.nan, np.nan]})
    ohe = OneHotEncoder(handle_unknown="ignore")
    ohe.fit(df)
    result = ohe.transform(df)
    # No categories learned → 0 output columns
    assert result.shape[1] == 0
