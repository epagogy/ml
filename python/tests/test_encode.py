"""Tests for ml.encode() — categorical feature encoding."""

import numpy as np
import pandas as pd
import pytest

import ml
from ml._types import ConfigError, DataError
from ml.encode import Encoder, encode


@pytest.fixture
def cat_df():
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "city": rng.choice(["NYC", "LA", "Chicago"], 100),
        "brand": rng.choice(["A", "B", "C", "D"], 100),
        "score": rng.rand(100),
        "label": rng.choice(["yes", "no"], 100),
    })


# Return type
def test_encode_returns_encoder(cat_df):
    enc = encode(cat_df, columns=["city"])
    assert isinstance(enc, Encoder)

def test_encode_via_ml_namespace(cat_df):
    enc = ml.encode(cat_df, columns=["city"])
    assert isinstance(enc, Encoder)

def test_encode_attributes(cat_df):
    enc = encode(cat_df, columns=["city", "brand"])
    assert enc.columns == ["city", "brand"]
    assert enc.method == "onehot"
    assert "city" in enc._categories
    assert "brand" in enc._categories

# One-hot transform
def test_onehot_drops_original_column(cat_df):
    enc = encode(cat_df, columns=["city"])
    out = enc.transform(cat_df)
    assert "city" not in out.columns

def test_onehot_adds_dummy_columns(cat_df):
    enc = encode(cat_df, columns=["city"])
    out = enc.transform(cat_df)
    assert "city_NYC" in out.columns
    assert "city_LA" in out.columns
    assert "city_Chicago" in out.columns

def test_onehot_columns_are_binary(cat_df):
    enc = encode(cat_df, columns=["city"])
    out = enc.transform(cat_df)
    dummy_cols = [c for c in out.columns if c.startswith("city_")]
    for col in dummy_cols:
        assert set(out[col].unique()).issubset({0, 1})

def test_onehot_each_row_sums_to_one(cat_df):
    enc = encode(cat_df, columns=["city"])
    out = enc.transform(cat_df)
    dummy_cols = [c for c in out.columns if c.startswith("city_")]
    row_sums = out[dummy_cols].sum(axis=1)
    assert (row_sums == 1).all()

def test_onehot_preserves_numeric_columns(cat_df):
    enc = encode(cat_df, columns=["city"])
    out = enc.transform(cat_df)
    assert "score" in out.columns
    assert "label" in out.columns

def test_onehot_preserves_index(cat_df):
    cat_df = cat_df.copy()
    cat_df.index = range(100, 200)
    enc = encode(cat_df, columns=["city"])
    out = enc.transform(cat_df)
    assert list(out.index) == list(cat_df.index)

# Ordinal transform
def test_ordinal_replaces_column(cat_df):
    enc = encode(cat_df, columns=["city"], method="ordinal")
    out = enc.transform(cat_df)
    assert "city" in out.columns

def test_ordinal_column_is_integer(cat_df):
    enc = encode(cat_df, columns=["city"], method="ordinal")
    out = enc.transform(cat_df)
    assert out["city"].dtype in (int, "int64", "int32")

def test_ordinal_values_in_range(cat_df):
    enc = encode(cat_df, columns=["city"], method="ordinal")
    out = enc.transform(cat_df)
    n_cats = len(enc._categories["city"])
    assert out["city"].min() >= 0
    assert out["city"].max() < n_cats

# Unseen category handling
def test_onehot_unseen_category_all_zeros(cat_df):
    enc = encode(cat_df, columns=["city"])
    new = pd.DataFrame({"city": ["UNKNOWN"], "score": [1.0], "label": ["yes"]})
    out = enc.transform(new)
    dummy_cols = [c for c in out.columns if c.startswith("city_")]
    assert out[dummy_cols].sum(axis=1).iloc[0] == 0

def test_ordinal_unseen_category_is_minus_one(cat_df):
    enc = encode(cat_df, columns=["city"], method="ordinal")
    new = pd.DataFrame({"city": ["UNKNOWN"], "score": [1.0], "label": ["yes"]})
    out = enc.transform(new)
    assert out["city"].iloc[0] == -1

# Multi-column
def test_encode_multi_column(cat_df):
    enc = encode(cat_df, columns=["city", "brand"])
    out = enc.transform(cat_df)
    assert "city" not in out.columns
    assert "brand" not in out.columns
    assert any(c.startswith("city_") for c in out.columns)
    assert any(c.startswith("brand_") for c in out.columns)

# No leakage: fit on train, apply to valid
def test_encode_fit_on_train_apply_to_valid(cat_df):
    train = cat_df.iloc[:80]
    valid = cat_df.iloc[80:]
    enc = encode(train, columns=["city"])
    out = enc.transform(valid)
    assert any(c.startswith("city_") for c in out.columns)

# save/load
def test_encode_save_load_onehot(tmp_path, cat_df):
    enc = encode(cat_df, columns=["city", "brand"])
    path = str(tmp_path / "enc.pyml")
    ml.save(enc, path)
    enc2 = ml.load(path)
    assert isinstance(enc2, Encoder)
    pd.testing.assert_frame_equal(enc.transform(cat_df), enc2.transform(cat_df))

def test_encode_save_load_ordinal(tmp_path, cat_df):
    enc = encode(cat_df, columns=["city"], method="ordinal")
    path = str(tmp_path / "enc.pyml")
    ml.save(enc, path)
    enc2 = ml.load(path)
    pd.testing.assert_frame_equal(enc.transform(cat_df), enc2.transform(cat_df))

# Error handling
def test_encode_non_dataframe_raises():
    with pytest.raises(DataError):
        encode(pd.Series(["a", "b"]), columns=["col"])

def test_encode_empty_columns_raises(cat_df):
    with pytest.raises(DataError):
        encode(cat_df, columns=[])

def test_encode_missing_column_raises(cat_df):
    with pytest.raises(DataError):
        encode(cat_df, columns=["nonexistent"])

def test_encode_bad_method_raises(cat_df):
    with pytest.raises(ConfigError):
        encode(cat_df, columns=["city"], method="target")

def test_encode_transform_missing_column_raises(cat_df):
    enc = encode(cat_df, columns=["city"])
    with pytest.raises(DataError):
        enc.transform(cat_df.drop(columns=["city"]))

# repr
def test_encode_repr(cat_df):
    enc = encode(cat_df, columns=["city"])
    assert "Encoder" in repr(enc)
    assert "city" in repr(enc)


# ── A1: Target encoding ───────────────────────────────────────────────────────


def test_encode_target_basic():
    """encode(method='target') returns Encoder with float-valued column. A1."""
    import warnings
    rng = np.random.RandomState(42)
    n = 200
    data = pd.DataFrame({
        "city": rng.choice(["NYC", "LA", "Chicago"], n),
        "label": rng.choice([0, 1], n),
    })
    s = ml.split(data=data, target="label", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enc = ml.encode(s.train, columns=["city"], method="target",
                        target="label", seed=42)

    assert isinstance(enc, Encoder)
    assert enc.method == "target"
    transformed = enc.transform(s.train)
    assert "city" in transformed.columns
    assert transformed["city"].dtype == float
    # Values should be probabilities in [0, 1]
    assert (transformed["city"] >= 0.0).all()
    assert (transformed["city"] <= 1.0).all()


def test_encode_target_regression():
    """encode(method='target') works for regression targets. A1."""
    import warnings
    rng = np.random.RandomState(42)
    n = 200
    data = pd.DataFrame({
        "city": rng.choice(["NYC", "LA", "Chicago"], n),
        "price": rng.rand(n) * 100,
    })
    s = ml.split(data=data, target="price", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enc = ml.encode(s.train, columns=["city"], method="target",
                        target="price", seed=42)

    transformed = enc.transform(s.valid)
    assert "city" in transformed.columns
    assert transformed["city"].dtype == float


def test_encode_target_unseen_category():
    """Unseen categories map to global mean — no KeyError. A1."""
    import warnings
    rng = np.random.RandomState(42)
    n = 100
    train = pd.DataFrame({
        "city": rng.choice(["NYC", "LA"], n),
        "label": rng.choice([0, 1], n),
    })
    test = pd.DataFrame({
        "city": ["NYC", "Boston", "UNSEEN"],  # Boston/UNSEEN not in train
        "label": [0, 1, 0],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enc = ml.encode(train, columns=["city"], method="target",
                        target="label", seed=42)

    result = enc.transform(test)
    global_mean = float(train["label"].mean())
    # Unseen categories should be close to global mean (smoothed toward prior)
    assert result.loc[result.index[1], "city"] == pytest.approx(global_mean, abs=0.1)
    assert not result["city"].isna().any()


def test_encode_target_no_leakage():
    """Target encoding on training data uses CV — no target leakage. A1."""
    import warnings
    # Perfect signal: if we leak target, encoding == target exactly
    n = 100
    data = pd.DataFrame({
        "cat": [str(i % 10) for i in range(n)],  # 10 unique categories
        "label": [i % 2 for i in range(n)],       # perfectly correlated with cat
    })
    s = ml.split(data=data, target="label", seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enc = ml.encode(s.train, columns=["cat"], method="target",
                        target="label", seed=42, folds=5)
    transformed = enc.transform(s.train)
    # If no leakage: values are NOT exactly 0.0 or 1.0 for every row
    # (smoothing + CV prevents perfect memorisation)
    unique_vals = transformed["cat"].unique()
    assert len(unique_vals) > 1


def test_encode_target_smoothing_auto():
    """smoothing='auto' uses Micci-Barreca — rare categories pulled toward mean. A1."""
    import warnings
    rng = np.random.RandomState(42)
    n = 500
    # One rare category (5 rows) and one frequent (495 rows)
    cats = ["rare"] * 5 + ["common"] * (n - 5)
    rng.shuffle(cats)
    data = pd.DataFrame({
        "cat": cats,
        "label": rng.choice([0, 1], n),
    })
    s = ml.split(data=data, target="label", seed=42)
    global_mean = float(s.train["label"].mean())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enc = ml.encode(s.train, columns=["cat"], method="target",
                        target="label", smoothing="auto", seed=42)

    transformed = enc.transform(s.train)
    rare_vals = transformed.loc[s.train["cat"] == "rare", "cat"]
    common_vals = transformed.loc[s.train["cat"] == "common", "cat"]
    # Rare category should be more smoothed toward global mean than common
    rare_mean = rare_vals.mean()
    common_mean = common_vals.mean()
    # Rare should be closer to global_mean than common (more regularised)
    assert abs(rare_mean - global_mean) <= abs(common_mean - global_mean) + 0.3


def test_encode_target_single_occurrence_category():
    """Single-occurrence category smoothed entirely toward global mean. A1."""
    import warnings
    rng = np.random.RandomState(42)
    n = 200
    cats = ["unique_once"] + rng.choice(["A", "B", "C"], n - 1).tolist()
    data = pd.DataFrame({
        "cat": cats,
        "label": [1] + rng.choice([0, 1], n - 1).tolist(),
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enc = ml.encode(data, columns=["cat"], method="target",
                        target="label", seed=42)

    transformed = enc.transform(data.iloc[:1])  # just the unique row
    global_mean = float(data["label"].mean())
    # Single occurrence: weight ≈ 0 → value ≈ global mean
    assert abs(transformed["cat"].iloc[0] - global_mean) < 0.3


def test_encode_target_multiclass_K_minus_1():
    """Multiclass target encoding produces K-1 columns per categorical. A1."""
    import warnings
    rng = np.random.RandomState(42)
    n = 300
    data = pd.DataFrame({
        "city": rng.choice(["NYC", "LA", "Chicago"], n),
        "label": rng.choice(["red", "green", "blue"], n),  # 3 classes → 2 cols
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enc = ml.encode(data, columns=["city"], method="target",
                        target="label", seed=42)

    transformed = enc.transform(data)
    assert "city" not in transformed.columns  # original dropped
    assert "city_te_0" in transformed.columns
    assert "city_te_1" in transformed.columns
    assert "city_te_2" not in transformed.columns  # K-1=2 columns only


def test_encode_target_cv_alignment():
    """cv= parameter aligns fold boundaries with downstream fit(). A1 (Onodera C1)."""
    import warnings
    rng = np.random.RandomState(42)
    n = 300
    data = pd.DataFrame({
        "city": rng.choice(["NYC", "LA", "Chicago"], n),
        "label": rng.choice([0, 1], n),
    })
    s = ml.split(data=data, target="label", seed=42)
    _s_cv = ml.split(data=data, target="label", seed=42)
    cv = ml.cv(_s_cv, folds=5, seed=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enc = ml.encode(s.train, columns=["city"], method="target",
                        target="label", cv=cv, seed=42)

    # Fold indices should be stored (not the DataFrame)
    assert enc._fold_indices is not None
    assert len(enc._fold_indices) == 5
    # Each entry is (train_list, val_list)
    train_idxs, val_idxs = enc._fold_indices[0]
    assert isinstance(train_idxs, list)
    assert isinstance(val_idxs, list)

    # Transform should still work
    transformed = enc.transform(s.valid)
    assert not transformed["city"].isna().any()


def test_encode_target_requires_target_param():
    """encode(method='target') without target= raises ConfigError. A1."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "city": rng.choice(["NYC", "LA"], 100),
        "label": rng.choice([0, 1], 100),
    })
    with pytest.raises(ConfigError, match="target="):
        ml.encode(data, columns=["city"], method="target", seed=42)


def test_encode_target_requires_seed():
    """encode(method='target') without cv= or seed= raises ConfigError. A1."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "city": rng.choice(["NYC", "LA"], 100),
        "label": rng.choice([0, 1], 100),
    })
    with pytest.raises(ConfigError, match="seed="):
        ml.encode(data, columns=["city"], method="target", target="label")


def test_encode_target_save_load(tmp_path):
    """Encoder with method='target' round-trips through save/load. A1."""
    import warnings
    rng = np.random.RandomState(42)
    n = 200
    data = pd.DataFrame({
        "city": rng.choice(["NYC", "LA", "Chicago"], n),
        "label": rng.choice([0, 1], n),
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enc = ml.encode(data, columns=["city"], method="target",
                        target="label", seed=42)

    path = str(tmp_path / "enc.pyml")
    ml.save(enc, path)
    enc2 = ml.load(path)

    assert isinstance(enc2, Encoder)
    assert enc2.method == "target"

    orig = enc.transform(data)
    loaded = enc2.transform(data)
    pd.testing.assert_frame_equal(orig, loaded)


# ---------------------------------------------------------------------------
# A9: Frequency encoding tests
# ---------------------------------------------------------------------------

def test_frequency_basic(cat_df):
    """encode(method='frequency') replaces categories with their train frequencies. A9."""
    enc = encode(cat_df, columns=["city"], method="frequency")
    assert enc.method == "frequency"
    result = enc.transform(cat_df)
    # Frequency values should be in (0, 1]
    assert result["city"].between(0.0, 1.0).all()
    # Number of distinct frequency values should equal number of distinct cities
    assert result["city"].nunique() == cat_df["city"].nunique()


def test_frequency_unseen_zero(cat_df):
    """encode(method='frequency') maps unseen categories to 0.0. A9."""
    enc = encode(cat_df, columns=["city"], method="frequency")
    # Add unseen category
    test_data = cat_df.copy()
    test_data.loc[test_data.index[0], "city"] = "UNSEEN_CITY"
    result = enc.transform(test_data)
    assert result["city"].iloc[0] == 0.0


def test_frequency_proportion_mode(cat_df):
    """Frequency encoding values sum correctly: n_i / n for each category. A9."""
    enc = encode(cat_df, columns=["brand"], method="frequency")
    result = enc.transform(cat_df)
    # Each brand's frequency should equal its count / total
    for brand in cat_df["brand"].unique():
        expected_freq = (cat_df["brand"] == brand).mean()
        actual_freq = result.loc[cat_df["brand"] == brand, "brand"].iloc[0]
        assert abs(actual_freq - expected_freq) < 1e-9


def test_frequency_multiple_columns(cat_df):
    """encode(method='frequency') handles multiple columns independently. A9."""
    enc = encode(cat_df, columns=["city", "brand"], method="frequency")
    result = enc.transform(cat_df)
    assert "city" in result.columns
    assert "brand" in result.columns
    # Both columns should have float values
    assert result["city"].dtype == float
    assert result["brand"].dtype == float


# ---------------------------------------------------------------------------
# Chain 4.1: WOE encoding
# ---------------------------------------------------------------------------


def test_woe_basic():
    """encode(method='woe') returns Encoder with float-valued column. Chain 4.1."""
    rng = np.random.RandomState(42)
    n = 200
    data = pd.DataFrame({
        "city": rng.choice(["NYC", "LA", "Chicago"], n),
        "label": rng.choice([0, 1], n),
    })
    enc = ml.encode(data, columns=["city"], method="woe", target="label")
    assert isinstance(enc, Encoder)
    assert enc.method == "woe"
    transformed = enc.transform(data)
    assert "city" in transformed.columns
    assert transformed["city"].dtype == float


def test_woe_iv_scores():
    """encode(method='woe') stores non-negative IV score per column. Chain 4.1."""
    rng = np.random.RandomState(42)
    n = 300
    data = pd.DataFrame({
        "city": rng.choice(["NYC", "LA", "Chicago"], n),
        "label": rng.choice([0, 1], n),
    })
    enc = ml.encode(data, columns=["city"], method="woe", target="label")
    assert "city" in enc._iv_scores
    assert enc._iv_scores["city"] >= 0.0


def test_woe_unseen_zero():
    """Unseen categories map to 0.0 (neutral evidence). Chain 4.1."""
    rng = np.random.RandomState(42)
    n = 100
    train = pd.DataFrame({
        "city": rng.choice(["NYC", "LA"], n),
        "label": rng.choice([0, 1], n),
    })
    test = pd.DataFrame({
        "city": ["Boston", "UNKNOWN"],
        "label": [0, 1],
    })
    enc = ml.encode(train, columns=["city"], method="woe", target="label")
    result = enc.transform(test)
    assert result["city"].iloc[0] == pytest.approx(0.0)
    assert result["city"].iloc[1] == pytest.approx(0.0)


def test_woe_requires_binary():
    """encode(method='woe') raises ConfigError for non-binary target. Chain 4.1."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "city": rng.choice(["NYC", "LA", "Chicago"], 100),
        "label": rng.choice(["A", "B", "C"], 100),  # 3 classes
    })
    with pytest.raises(ml.ConfigError, match="binary"):
        ml.encode(data, columns=["city"], method="woe", target="label")


def test_encode_one_hot_alias(small_classification_data):
    """encode(method='one-hot') accepted as alias for 'onehot'."""
    data = small_classification_data.copy()
    encoder = ml.encode(data=data, columns=["x3"], method="one-hot", seed=42)
    assert encoder is not None
    transformed = encoder.transform(data)
    # Should have one-hot columns for x3
    assert any("x3" in col for col in transformed.columns)

