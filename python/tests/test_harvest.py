"""Tests for harvest-driven features.

Features from GitHub harvest research:
1. Polars DataFrame auto-conversion
2. LightGBM force_col_wise default
3. Sample weights in fit()
4. Group k-fold in split()
"""

import numpy as np
import pandas as pd
import pytest

import ml  # noqa: I001

# --- 1. Polars auto-conversion ---

def test_polars_conversion_split():
    """Polars DataFrame auto-converts to pandas in split()."""
    pytest.importorskip("polars")
    import polars as pl

    df_pd = pd.DataFrame({
        "x": np.random.RandomState(42).rand(50),
        "y": np.random.RandomState(42).choice([0, 1], 50),
    })
    df_pl = pl.from_pandas(df_pd)

    s = ml.split(df_pl, "y", seed=42)
    assert len(s.train) + len(s.valid) + len(s.test) == 50


def test_polars_conversion_fit():
    """Polars DataFrame auto-converts to pandas in fit()."""
    pytest.importorskip("polars")
    import polars as pl

    rng = np.random.RandomState(42)
    df_pd = pd.DataFrame({
        "x": rng.rand(50),
        "y": rng.choice([0, 1], 50),
    })
    df_pl = pl.from_pandas(df_pd)

    model = ml.fit(df_pl, "y", seed=42)
    assert model.task == "classification"


def test_polars_conversion_predict():
    """Polars DataFrame auto-converts to pandas in predict()."""
    pytest.importorskip("polars")
    import polars as pl

    rng = np.random.RandomState(42)
    df = pd.DataFrame({"x": rng.rand(50), "y": rng.choice([0, 1], 50)})
    s = ml.split(df, "y", seed=42)
    model = ml.fit(s.train, "y", seed=42)

    valid_pl = pl.from_pandas(s.valid)
    preds = ml.predict(model, valid_pl)
    assert len(preds) == len(s.valid)


def test_polars_conversion_evaluate():
    """Polars DataFrame auto-converts to pandas in evaluate()."""
    pytest.importorskip("polars")
    import polars as pl

    rng = np.random.RandomState(42)
    df = pd.DataFrame({"x": rng.rand(50), "y": rng.choice([0, 1], 50)})
    s = ml.split(df, "y", seed=42)
    model = ml.fit(s.train, "y", seed=42)

    valid_pl = pl.from_pandas(s.valid)
    metrics = ml.evaluate(model, valid_pl)
    assert "accuracy" in metrics


def test_polars_conversion_profile():
    """Polars DataFrame auto-converts to pandas in profile()."""
    pytest.importorskip("polars")
    import polars as pl

    df_pd = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    df_pl = pl.from_pandas(df_pd)

    prof = ml.profile(df_pl, "y")
    assert prof["shape"] == (3, 2)


def test_duck_typed_to_pandas():
    """Any object with .to_pandas() method works."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"x": rng.rand(50), "y": rng.choice([0, 1], 50)})

    class FakeFrame:
        def to_pandas(self):
            return df

    s = ml.split(FakeFrame(), "y", seed=42)
    assert len(s.train) > 0


# --- 2. LightGBM force_col_wise (now conditional on feature count) ---

def test_lightgbm_narrow_data_no_force_col_wise():
    """LightGBM does NOT set force_col_wise for narrow data (<=500 features).

    Chain 1 fix: unconditional force_col_wise degrades performance on narrow
    datasets. Now only set when X.shape[1] > 500 (in fit.py, not _engines.py).
    """
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {f"x{i}": rng.rand(100) for i in range(10)},
    )
    df["y"] = rng.choice([0, 1], 100)
    model = ml.fit(df, "y", algorithm="lightgbm", seed=42)
    # force_col_wise should NOT be set for 10-feature data
    params = model._model.get_params()
    assert params.get("force_col_wise", False) is False


def test_lightgbm_wide_data_force_col_wise():
    """LightGBM DOES set force_col_wise for wide data (>500 features)."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {f"x{i}": rng.rand(200) for i in range(510)},
    )
    df["y"] = rng.choice([0, 1], 200)
    model = ml.fit(df, "y", algorithm="lightgbm", seed=42, early_stopping=False)
    # force_col_wise should be True for 510-feature data
    params = model._model.get_params()
    assert params.get("force_col_wise") is True


# --- 3. Sample weights ---

def test_weights_classification():
    """weights= parameter works for classification."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "x": rng.rand(100),
        "y": rng.choice([0, 1], 100),
        "w": rng.rand(100) + 0.1,
    })
    model = ml.fit(df, "y", weights="w", seed=42)
    assert model.task == "classification"
    # weights column excluded from features
    assert "w" not in model.features


def test_weights_regression():
    """weights= parameter works for regression."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "x": rng.rand(100),
        "y": rng.rand(100) * 100,
        "w": rng.rand(100) + 0.1,
    })
    model = ml.fit(df, "y", weights="w", algorithm="linear", seed=42)
    assert model.task == "regression"
    assert "w" not in model.features


def test_weights_missing_column():
    """weights= with missing column gives clear error."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"x": rng.rand(50), "y": rng.choice([0, 1], 50)})
    with pytest.raises(ml.DataError, match="not found"):
        ml.fit(df, "y", weights="nonexistent", seed=42)


def test_weights_nan_rejected():
    """weights= column with NaN gives clear error."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "x": rng.rand(50),
        "y": rng.choice([0, 1], 50),
        "w": [np.nan] + list(rng.rand(49)),
    })
    with pytest.raises(ml.DataError, match="NaN"):
        ml.fit(df, "y", weights="w", seed=42)


def test_weights_negative_rejected():
    """weights= column with negative values gives clear error."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "x": rng.rand(50),
        "y": rng.choice([0, 1], 50),
        "w": [-1.0] + list(rng.rand(49)),
    })
    with pytest.raises(ml.DataError, match="negative"):
        ml.fit(df, "y", weights="w", seed=42)


def test_weights_and_balance_conflict():
    """weights= and balance=True cannot be used together."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "x": rng.rand(50),
        "y": rng.choice([0, 1], 50),
        "w": rng.rand(50) + 0.1,
    })
    with pytest.raises(ml.ConfigError, match="Cannot use weights.*balance"):
        ml.fit(df, "y", weights="w", balance=True, seed=42)


def test_weights_cv():
    """weights= works with cross-validation."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "x": rng.rand(100),
        "y": rng.choice([0, 1], 100),
        "w": rng.rand(100) + 0.1,
    })
    cv = ml.split(df, "y", folds=3, seed=42)
    model = ml.fit(cv, "y", weights="w", seed=42)
    assert model.task == "classification"
    assert "w" not in model.features


# --- 4. Group k-fold ---

def test_group_holdout_no_leakage():
    """groups= prevents group leakage in holdout split."""
    rng = np.random.RandomState(42)
    patients = np.repeat(range(20), 5)
    df = pd.DataFrame({
        "patient": patients,
        "x": rng.rand(100),
        "y": rng.choice([0, 1], 100),
    })
    s = ml.split(df, "y", groups="patient", seed=42)

    train_groups = set(s.train["patient"])
    valid_groups = set(s.valid["patient"])
    test_groups = set(s.test["patient"])

    assert train_groups.isdisjoint(valid_groups)
    assert train_groups.isdisjoint(test_groups)
    assert valid_groups.isdisjoint(test_groups)


def test_group_cv_no_leakage():
    """groups= prevents group leakage in CV folds."""
    rng = np.random.RandomState(42)
    patients = np.repeat(range(20), 5)
    df = pd.DataFrame({
        "patient": patients,
        "x": rng.rand(100),
        "y": rng.choice([0, 1], 100),
    })
    cv = ml.split(df, "y", groups="patient", folds=5, seed=42)

    assert cv.k == 5
    for fold_train, fold_valid in cv.folds:
        train_g = set(fold_train["patient"])
        valid_g = set(fold_valid["patient"])
        assert train_g.isdisjoint(valid_g)


def test_group_cv_too_many_folds():
    """groups= CV with more folds than groups gives clear error."""
    rng = np.random.RandomState(42)
    patients = np.repeat(range(3), 10)
    df = pd.DataFrame({
        "patient": patients,
        "x": rng.rand(30),
        "y": rng.choice([0, 1], 30),
    })
    with pytest.raises(ml.DataError, match="3 groups"):
        ml.split(df, "y", groups="patient", folds=5, seed=42)


def test_group_missing_column():
    """groups= with missing column gives clear error."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"x": rng.rand(50), "y": rng.choice([0, 1], 50)})
    with pytest.raises(ml.DataError, match="not found"):
        ml.split(df, "y", groups="nonexistent", seed=42)


def test_group_holdout_all_rows_present():
    """groups= holdout preserves all rows."""
    rng = np.random.RandomState(42)
    patients = np.repeat(range(20), 5)
    df = pd.DataFrame({
        "patient": patients,
        "x": rng.rand(100),
        "y": rng.choice([0, 1], 100),
    })
    s = ml.split(df, "y", groups="patient", seed=42)
    assert len(s.train) + len(s.valid) + len(s.test) == 100


def test_group_and_time_conflict():
    """groups= and time= cannot be used together."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "patient": np.repeat(range(10), 5),
        "ts": pd.date_range("2020-01-01", periods=50),
        "x": rng.rand(50),
        "y": rng.choice([0, 1], 50),
    })
    with pytest.raises(ml.ConfigError, match="groups.*time"):
        ml.split(df, "y", groups="patient", time="ts", seed=42)


# --- 5. Dataset: patients (groups) ---

def test_patients_dataset_loads():
    """patients: real hospital readmission data (OpenML 4541, subsampled)."""
    p = ml.dataset("patients")
    assert p.shape == (424, 13)
    assert "patient_id" in p.columns
    assert "readmitted" in p.columns
    assert p["patient_id"].nunique() == 100
    assert "race" in p.columns  # real demographics
    assert "A1Cresult" in p.columns  # real lab results


def test_patients_groups_no_leakage():
    """patients dataset works with groups= and has no patient leakage."""
    p = ml.dataset("patients")
    s = ml.split(p, "readmitted", groups="patient_id", seed=42)

    train_patients = set(s.train["patient_id"])
    test_patients = set(s.test["patient_id"])
    assert train_patients.isdisjoint(test_patients)


def test_patients_groups_cv():
    """patients dataset works with groups= CV."""
    p = ml.dataset("patients")
    cv = ml.split(p, "readmitted", groups="patient_id", folds=5, seed=42)
    assert cv.k == 5


# --- 6. Dataset: sales (time) ---

def test_sales_dataset_loads():
    """sales dataset loads with expected shape and columns."""
    s = ml.dataset("sales")
    assert s.shape[0] == 730
    assert "date" in s.columns
    assert "revenue" in s.columns
    assert s["date"].dtype == "datetime64[ns]"


def test_sales_time_split():
    """sales dataset works with time= and preserves temporal order."""
    sa = ml.dataset("sales")
    s = ml.split(sa, "revenue", time="date", seed=42)

    assert len(s.train) + len(s.valid) + len(s.test) == 730
    # date column consumed by time= — but check train/valid/test are non-empty
    assert len(s.train) > 0
    assert len(s.valid) > 0
    assert len(s.test) > 0


def test_sales_end_to_end():
    """sales dataset: split → fit → evaluate full pipeline."""
    sa = ml.dataset("sales")
    s = ml.split(sa, "revenue", time="date", seed=42)
    model = ml.fit(s.train, "revenue", seed=42)
    assert model.task == "regression"
    metrics = ml.evaluate(model, s.valid)
    assert "rmse" in metrics
    assert metrics["r2"] > 0.0  # model learns something


# --- 7. New gap-closing synthetic datasets ---

def test_reviews_dataset():
    """reviews: text features, boolean, mixed types."""
    r = ml.dataset("reviews")
    assert r.shape == (2000, 8)
    assert "review_text" in r.columns
    assert "sentiment" in r.columns
    assert r["review_text"].dtype == "object"
    assert r["verified_purchase"].dtype == bool
    assert any(len(t) > 10 for t in r["review_text"])


def test_ecommerce_dataset():
    """ecommerce: real online shoppers intention (OpenML 42993)."""
    e = ml.dataset("ecommerce")
    assert e.shape == (12330, 18)
    assert "purchased" in e.columns
    assert e["TrafficType"].nunique() >= 10  # real high-cardinality
    assert e["Browser"].nunique() >= 5


def test_tips_dataset():
    """tips: real Bryant & Smith 1995, small dataset, regression."""
    t = ml.dataset("tips")
    assert t.shape == (244, 7)
    assert "smoker" in t.columns  # categorical Yes/No
    assert "tip" in t.columns
    assert t["total_bill"].dtype == float


def test_flights_dataset():
    """flights: time series with datetime, trend + seasonality."""
    f = ml.dataset("flights")
    assert f.shape == (144, 4)
    assert "datetime" in str(f["date"].dtype)
    assert "passengers" in f.columns
    assert f["passengers"].is_monotonic_increasing is not True  # has seasonality


def test_survival_dataset():
    """survival: real VA lung cancer data (OpenML 497)."""
    s = ml.dataset("survival")
    assert s.shape == (137, 8)
    assert "survival_days" in s.columns
    assert "Celltype" in s.columns  # real categorical with 4 levels
    assert "Treatment" in s.columns


def test_datasets_metadata():
    """ml.datasets() returns metadata DataFrame for all datasets."""
    meta = ml.datasets()
    assert len(meta) > 70
    assert "name" in meta.columns
    assert "rows" in meta.columns
    assert "task" in meta.columns
    assert "source" in meta.columns
    # Check sources present
    sources = set(meta["source"])
    assert "synthetic" in sources
    assert "sklearn" in sources
    assert "openml" in sources


def test_ecommerce_fit_with_high_cardinality():
    """ecommerce fits with real mixed dtypes and categoricals."""
    e = ml.dataset("ecommerce")
    s = ml.split(e, "purchased", seed=42)
    model = ml.fit(s.train, "purchased", seed=42)
    assert model.task == "classification"
    metrics = ml.evaluate(model, s.valid)
    assert "accuracy" in metrics


# --- 8. Real datasets (OpenML) ---

def test_titanic_loads_and_fits():
    """Titanic: real data, NaN, categoricals, binary clf."""
    t = ml.dataset("titanic")
    assert t.shape == (1309, 8)
    assert "survived" in t.columns
    assert t["age"].isna().sum() > 0  # real NaN

    s = ml.split(t, "survived", seed=42)
    model = ml.fit(s.train, "survived", seed=42)
    assert model.task == "classification"
    metrics = ml.evaluate(model, s.valid)
    # Majority class baseline = 61.8% ("no"). Threshold must exceed this.
    assert metrics["accuracy"] > 0.75


def test_penguins_loads_and_fits():
    """Penguins: real data, 3-class, categoricals."""
    p = ml.dataset("penguins")
    assert p.shape == (344, 7)
    assert "species" in p.columns
    assert p["species"].nunique() == 3

    s = ml.split(p, "species", seed=42)
    model = ml.fit(s.train, "species", seed=42)
    assert model.task == "classification"
    metrics = ml.evaluate(model, s.valid)
    assert metrics["accuracy"] > 0.8


@pytest.mark.slow
def test_adult_loads_and_fits():
    """Adult: real data, many categoricals, large, binary clf — server only (40K rows)."""
    a = ml.dataset("adult")
    assert a.shape[0] > 40000
    assert "income" in a.columns

    s = ml.split(a, "income", seed=42)
    model = ml.fit(s.train, "income", seed=42)
    assert model.task == "classification"
    metrics = ml.evaluate(model, s.valid)
    # Majority class baseline = 76.0% ("<=50K"). Threshold must exceed this.
    assert metrics["accuracy"] > 0.82


# --- 8. All OpenML datasets load and fit ---

@pytest.mark.slow
@pytest.mark.parametrize("name,target,task_type", [
    # Binary classification — medical/health
    ("heart", "heart_disease", "classification"),
    ("diabetes_pima", "diabetes", "classification"),
    ("haberman", "survived", "classification"),
    ("blood", "donated", "classification"),
    ("breast_w", "malignant", "classification"),
    # Binary classification — finance/business
    ("credit", "credit_risk", "classification"),
    ("australian", "approved", "classification"),
    ("bank", "subscribed", "classification"),
    # Binary classification — science/engineering
    ("sonar", "is_mine", "classification"),
    ("ionosphere", "ionosphere", "classification"),
    ("spam", "spam", "classification"),
    ("electricity", "price_up", "classification"),
    ("mushroom", "poisonous", "classification"),
    ("tic_tac_toe", "x_wins", "classification"),
    ("kr_vs_kp", "white_wins", "classification"),
    ("banknote", "authentic", "classification"),
    ("sick", "thyroid_sick", "classification"),
    ("phoneme", "phoneme", "classification"),
    ("credit_approval", "credit_approved", "classification"),
    ("jm1", "defect", "classification"),
    # Binary classification — environment/safety/medical/cyber
    ("wilt", "wilted", "classification"),
    ("climate", "crashed", "classification"),
    ("ozone", "ozone_day", "classification"),
    ("seismic", "hazardous", "classification"),
    ("mammography", "malignant", "classification"),
    ("ilpd", "liver_disease", "classification"),
    ("hypothyroid", "hypothyroid", "classification"),
    ("toxicity", "toxic", "classification"),
    ("phishing", "phishing", "classification"),
    ("qsar_biodeg", "biodegradable", "classification"),
    ("speed_dating", "match", "classification"),
    # Binary classification — neuroscience/IoT/tech
    ("eeg_eye_state", "eyes_open", "classification"),
    ("hill_valley", "hill", "classification"),
    ("kc2", "defective", "classification"),
    ("pc1", "defective", "classification"),
    ("amazon_employee", "access_granted", "classification"),
    ("nomao", "valid_place", "classification"),
    ("churn_telco", "churned", "classification"),
    ("jasmine", "positive", "classification"),
    # CC18 binary
    ("pc3", "defective", "classification"),
    ("pc4", "defective", "classification"),
    ("kc1", "defective", "classification"),
    ("wdbc", "malignant", "classification"),
    ("bioresponse", "active", "classification"),
    ("madelon", "positive", "classification"),
    ("cylinder_bands", "defective", "classification"),
    ("dresses_sales", "sold", "classification"),
    ("internet_ads", "is_ad", "classification"),
    # Binary — exotic/social/molecular
    ("vote", "party", "classification"),
    ("hepatitis", "survived", "classification"),
    ("musk", "musk", "classification"),
    ("qsar_oral_toxicity", "toxic", "classification"),
    ("speech", "spoken", "classification"),
    # Multiclass classification
    ("glass", "glass_type", "classification"),
    ("ecoli", "localization", "classification"),
    ("yeast", "localization", "classification"),
    ("segment", "segment", "classification"),
    ("vehicle", "vehicle_type", "classification"),
    ("vowel", "vowel", "classification"),
    ("car", "evaluation", "classification"),
    ("nursery", "recommendation", "classification"),
    ("letter", "letter", "classification"),
    ("satimage", "land_use", "classification"),
    ("optdigits", "digit", "classification"),
    ("page_blocks", "block_type", "classification"),
    ("cmc", "method", "classification"),
    ("dermatology", "disease", "classification"),
    ("waveform", "waveform", "classification"),
    ("balance_scale", "balance", "classification"),
    ("zoo", "animal_type", "classification"),
    ("splice", "splice_type", "classification"),
    ("pendigits", "digit", "classification"),
    ("eucalyptus", "utility", "classification"),
    ("steel_plates", "fault_type", "classification"),
    ("connect_4", "outcome", "classification"),
    ("mfeat_factors", "digit", "classification"),
    ("wall_robot", "direction", "classification"),
    ("texture", "texture_class", "classification"),
    ("jungle_chess", "outcome", "classification"),
    ("miniboone", "signal", "classification"),
    # Multiclass — science/biology
    ("dna", "dna_class", "classification"),
    ("soybean", "disease", "classification"),
    ("mice_protein", "genotype", "classification"),
    ("drug_consumption", "drug_use", "classification"),
    ("cardiotocography", "fetal_state", "classification"),
    ("seeds", "variety", "classification"),
    ("plants_margin", "species", "classification"),
    ("plants_shape", "species", "classification"),
    # Multiclass — signal/speech
    ("har", "activity", "classification"),
    ("isolet", "letter", "classification"),
    ("semeion", "digit", "classification"),
    ("gas_drift", "gas_type", "classification"),
    ("first_order", "theorem_class", "classification"),
    ("authorship", "author", "classification"),
    ("artificial_chars", "character", "classification"),
    # Multiclass — medical/exotic
    ("arrhythmia", "arrhythmia_type", "classification"),
    # CC18 multiclass
    ("mfeat_fourier", "digit", "classification"),
    ("mfeat_karhunen", "digit", "classification"),
    ("mfeat_morphological", "digit", "classification"),
    ("mfeat_zernike", "digit", "classification"),
    ("mfeat_pixel", "digit", "classification"),
    ("dental", "prevention", "classification"),
    ("cnae9", "category", "classification"),
    ("gesture", "phase", "classification"),
    # Regression
    ("diamonds", "price", "regression"),
    ("abalone", "rings", "regression"),
    ("mpg", "mpg", "regression"),
    ("bike", "count", "regression"),
    ("wine_quality", "quality", "regression"),
    ("ames", "price", "regression"),
    ("concrete", "strength", "regression"),
    ("kin8nm", "position", "regression"),
    ("cpu_act", "cpu_usage", "regression"),
    ("delta_elevators", "response", "regression"),
    ("bodyfat", "bodyfat", "regression"),
    ("machine_cpu", "performance", "regression"),
    ("stock", "stock_price", "regression"),
    ("auto_price", "price", "regression"),
    ("elevators", "response", "regression"),
    ("boston", "price", "regression"),
    ("energy", "heating_load", "regression"),
    ("superconduct", "critical_temp", "regression"),
    ("white_wine", "quality", "regression"),
    ("red_wine", "quality", "regression"),
    ("miami_housing", "price", "regression"),
    ("cars", "price", "regression"),
    ("house_16h", "price", "regression"),
    # CTR23 regression
    ("airfoil", "sound_pressure", "regression"),
    ("qsar_aquatic", "toxicity_time", "regression"),
    ("protein_structure", "rmsd", "regression"),
    ("communities", "latitude", "regression"),
    ("solar_flare", "flares", "regression"),  # numeric_target=True → float64 → regression
    ("gas_turbine", "decay", "regression"),
    ("electrical_grid", "stability", "regression"),
    ("kin_dynamics", "angular_accel", "regression"),
    ("wages", "wage", "regression"),
    ("house_sales", "price", "regression"),
    ("wind", "wind_power", "regression"),
    ("fps_benchmark", "fps", "regression"),
    ("work_hours", "hours_per_week", "regression"),
    ("fifa_wages", "wage", "regression"),
    ("debutanizer", "butane", "regression"),
    ("forest_fires", "burned_area", "regression"),
    # student_performance skipped — 649 rows / 17 classes = too sparse for stable split
    ("qsar_fish", "toxicity", "regression"),
    ("space_voting", "vote_share", "regression"),
    ("social_mobility", "occupation_count", "regression"),
])
def test_openml_dataset_roundtrip(name, target, task_type):
    """Every OpenML dataset: load → split → fit → evaluate."""
    df = ml.dataset(name)
    assert target in df.columns
    assert len(df) > 0

    s = ml.split(df, target, seed=42)
    model = ml.fit(s.train, target, seed=42)
    assert model.task == task_type

    metrics = ml.evaluate(model, s.valid)
    if task_type == "classification":
        assert "accuracy" in metrics
    else:
        assert "r2" in metrics


