"""Utility functions.

Not imported by core modules (split/fit/predict/evaluate/explain/io).
Only imported by __init__.py.
"""

from __future__ import annotations

import pandas as pd


def algorithms(task: str | None = None) -> pd.DataFrame:
    """List available algorithms as a DataFrame.

    Returns a DataFrame with one row per algorithm showing:
    algorithm, classification, regression, optional_dep, gpu.

    Args:
        task: Optional filter — "classification" or "regression".
              If provided, returns only algorithms that support that task.

    Returns:
        DataFrame: algorithm | classification | regression | optional_dep | gpu

    Example:
        >>> ml.algorithms()
        >>> ml.algorithms(task="classification")
    """
    rows = [
        {"algorithm": "random_forest",  "classification": True,  "regression": True,  "optional_dep": "",          "gpu": False},
        {"algorithm": "xgboost",        "classification": True,  "regression": True,  "optional_dep": "xgboost",   "gpu": True},
        {"algorithm": "lightgbm",       "classification": True,  "regression": True,  "optional_dep": "lightgbm",  "gpu": True},
        {"algorithm": "logistic",       "classification": True,  "regression": False, "optional_dep": "",          "gpu": False},
        {"algorithm": "linear",         "classification": False, "regression": True,  "optional_dep": "",          "gpu": False},
        {"algorithm": "knn",            "classification": True,  "regression": True,  "optional_dep": "",          "gpu": False},
        {"algorithm": "svm",            "classification": True,  "regression": True,  "optional_dep": "",          "gpu": False},
        {"algorithm": "naive_bayes",    "classification": True,  "regression": False, "optional_dep": "",          "gpu": False},
        {"algorithm": "elastic_net",    "classification": False, "regression": True,  "optional_dep": "",          "gpu": False},
        {"algorithm": "catboost",       "classification": True,  "regression": True,  "optional_dep": "catboost",  "gpu": True},
        {"algorithm": "tabpfn",         "classification": True,  "regression": False, "optional_dep": "tabpfn",    "gpu": False},
    ]
    df = pd.DataFrame(rows)
    if task == "classification":
        df = df[df["classification"]].reset_index(drop=True)
    elif task == "regression":
        df = df[df["regression"]].reset_index(drop=True)
    return df


DATASETS = [
    # Synthetic (deterministic, no download)
    "churn", "fraud", "patients", "sales",
    "reviews", "ecommerce", "tips", "flights", "survival",
    # sklearn built-in (no download)
    "iris", "wine", "cancer", "breast_cancer", "diabetes", "houses", "housing",
    # OpenML — binary classification
    "titanic", "heart", "diabetes_pima", "haberman", "blood", "breast_w",
    "credit", "australian", "bank", "adult",
    "sonar", "ionosphere", "spam", "electricity", "mushroom",
    "tic_tac_toe", "kr_vs_kp",
    "banknote", "sick", "phoneme", "credit_approval", "jm1",
    # OpenML — binary classification — environment/safety/medical/cyber
    "wilt", "climate", "ozone", "seismic",
    "mammography", "ilpd", "hypothyroid",
    "toxicity", "phishing", "qsar_biodeg", "speed_dating",
    "eeg_eye_state", "hill_valley",
    "kc2", "pc1", "amazon_employee", "nomao", "churn_telco", "jasmine",
    "pc3", "pc4", "kc1", "wdbc", "bioresponse", "madelon",
    "cylinder_bands", "dresses_sales", "internet_ads",
    # OpenML — regression / survival
    "veteran",
    # OpenML — binary classification — ecommerce/web
    "online_shoppers",
    # OpenML — binary classification — exotic/social/molecular
    "vote", "hepatitis", "musk", "qsar_oral_toxicity", "speech",
    # OpenML — multiclass classification
    "penguins", "glass", "ecoli", "yeast", "segment", "vehicle", "vowel",
    "car", "nursery", "letter", "satimage", "optdigits", "page_blocks",
    "cmc", "dermatology", "waveform", "balance_scale", "zoo",
    "splice", "pendigits", "eucalyptus", "steel_plates", "connect_4",
    "mfeat_factors",
    "wall_robot", "texture", "jungle_chess", "miniboone",
    "dna", "soybean", "mice_protein", "drug_consumption", "cardiotocography",
    "seeds", "plants_margin", "plants_shape",
    "har", "isolet", "semeion", "gas_drift", "first_order",
    "authorship", "artificial_chars",
    # OpenML — multiclass — medical/exotic
    "arrhythmia",
    "mfeat_fourier", "mfeat_karhunen", "mfeat_morphological",
    "mfeat_zernike", "mfeat_pixel",
    "dental", "cnae9", "gesture",
    # OpenML — regression
    "diamonds", "abalone", "mpg", "bike", "wine_quality", "ames",
    "concrete", "kin8nm", "cpu_act", "delta_elevators", "bodyfat",
    "machine_cpu", "stock", "auto_price", "elevators", "boston",
    "energy", "superconduct", "white_wine", "red_wine", "miami_housing",
    "cars", "house_16h",
    "airfoil", "qsar_aquatic", "protein_structure", "communities",
    "solar_flare", "gas_turbine", "electrical_grid", "kin_dynamics",
    "wages", "house_sales", "wind", "fps_benchmark", "work_hours",
    "fifa_wages", "debutanizer", "forest_fires", "student_performance",
    "qsar_fish", "space_voting", "social_mobility",
]


def dataset(name: str) -> pd.DataFrame:
    """Load example dataset.

    Self-contained quickstart (no external CSV needed).
    Includes synthetic data (churn) and sklearn classics.

    Args:
        name: Dataset name. Use ml.datasets() to see all available. Key examples:
              churn, titanic, penguins, iris, diabetes, houses, diamonds, adult.

    Returns:
        DataFrame ready for ml.split()

    Raises:
        DataError: If dataset name not recognized

    Example:
        >>> data = ml.dataset("churn")
        >>> data.shape
        (5000, 6)
        >>> data = ml.dataset("iris")
        >>> data.shape
        (150, 5)
    """
    if name == "churn":
        return _create_churn_dataset()
    elif name == "fraud":
        return _create_fraud_dataset()
    elif name == "iris":
        return _load_sklearn("iris")
    elif name == "wine":
        return _load_sklearn("wine")
    elif name in ("cancer", "breast_cancer"):
        return _load_sklearn("cancer")
    elif name == "diabetes":
        return _load_sklearn("diabetes")
    elif name in ("houses", "housing"):
        return _load_sklearn("houses")
    elif name == "patients":
        return _create_patients_dataset()
    elif name == "sales":
        return _create_sales_dataset()
    elif name == "reviews":
        return _create_reviews_dataset()
    elif name == "ecommerce":
        return _load_openml("online_shoppers")
    elif name == "tips":
        return _create_tips_dataset()
    elif name == "flights":
        return _create_flights_dataset()
    elif name == "survival":
        return _load_openml("veteran")
    elif name in DATASETS:
        return _load_openml(name)
    else:
        from ._types import DataError

        raise DataError(
            f"Dataset '{name}' not available. Choose from: {DATASETS}"
        )


def _create_churn_dataset() -> pd.DataFrame:
    """Load real IBM Telco customer churn dataset (OpenML 42178).

    7,043 customers × 20 features. 26.5% churn rate.
    Real-world quirks: TotalCharges stored as string, quoted categoricals,
    17 object columns, SeniorCitizen as 0/1 int, mixed dtypes.
    """
    import warnings

    try:
        from sklearn.datasets import fetch_openml
    except ImportError as e:
        from ._types import DataError
        raise DataError(
            "The 'churn' dataset requires scikit-learn for the initial download. "
            "Install it with: pip install scikit-learn"
        ) from e

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bunch = fetch_openml(data_id=42178, as_frame=True, parser="auto")

    df = bunch.frame

    # TotalCharges is stored as string — convert to numeric (real data quirk!)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Rename target to friendly name
    df = df.rename(columns={"Churn": "churn"})
    df["churn"] = df["churn"].map({"Yes": "yes", "No": "no"})

    return df


def _create_fraud_dataset() -> pd.DataFrame:
    """Load real credit card fraud dataset (OpenML 1597, subsampled).

    ~10,000 rows from 284K original. All 492 real frauds preserved +
    ~9,500 non-fraud sampled. PCA-anonymized features V1-V28, Amount, Time.
    Real 0.17% fraud rate → extreme class imbalance.
    """
    import warnings

    import numpy as np

    try:
        from sklearn.datasets import fetch_openml
    except ImportError as e:
        from ._types import DataError
        raise DataError(
            "The 'fraud' dataset requires scikit-learn for the initial download. "
            "Install it with: pip install scikit-learn"
        ) from e

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bunch = fetch_openml(data_id=1597, as_frame=True, parser="auto")

    df = bunch.frame
    # Separate fraud and non-fraud
    fraud_rows = df[df["Class"] == "1"]
    legit_rows = df[df["Class"] == "0"]

    # Keep ALL fraud rows + subsample non-fraud to ~9500
    rng = np.random.RandomState(42)
    legit_sample = legit_rows.sample(n=9500, random_state=rng)

    result = pd.concat([fraud_rows, legit_sample], ignore_index=True)
    # Shuffle
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    # Rename target
    result["fraud"] = result["Class"].map({"0": "no", "1": "yes"})
    result = result.drop(columns=["Class"])

    return result


def _load_sklearn(name: str) -> pd.DataFrame:
    """Load a bundled toy dataset as a tidy DataFrame.

    Datasets are bundled as csv.gz in ml/data/ — no sklearn required.
    """
    import pathlib

    _BUNDLED = {"iris", "wine", "cancer", "diabetes", "houses"}
    if name not in _BUNDLED:
        from ._types import DataError
        raise DataError(f"Unknown bundled dataset: {name}")

    data_dir = pathlib.Path(__file__).parent / "data"
    return pd.read_csv(data_dir / f"{name}.csv.gz")


# OpenML dataset registry: name → spec dict
# Keys: id (required), target, rename_target, keep, drop, map_target, numeric_target
_OPENML_REGISTRY: dict[str, dict] = {
    # ── Binary classification — medical/health ──
    "titanic": {
        "id": 40945, "target": "survived",
        "keep": ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "survived"],
        "map_target": {"1": "yes", "0": "no"},
    },
    "heart": {"id": 53, "target": "class", "rename_target": "heart_disease"},
    "diabetes_pima": {"id": 37, "target": "class", "rename_target": "diabetes"},
    "haberman": {"id": 43, "target": "Survival_status", "rename_target": "survived"},
    "blood": {"id": 1464, "target": "Class", "rename_target": "donated"},
    "breast_w": {"id": 15, "target": "Class", "rename_target": "malignant"},
    # ── Binary classification — finance/business ──
    "credit": {"id": 31, "target": "class", "rename_target": "credit_risk"},
    "australian": {"id": 40981, "target": "A15", "rename_target": "approved"},
    "bank": {"id": 1461, "target": "Class", "rename_target": "subscribed"},
    "adult": {"id": 1590, "target": "class", "rename_target": "income", "drop": ["fnlwgt"]},
    # ── Binary classification — science/engineering ──
    "sonar": {"id": 40, "target": "Class", "rename_target": "is_mine"},
    "ionosphere": {"id": 59, "target": "class", "rename_target": "ionosphere"},
    "spam": {"id": 44, "target": "class", "rename_target": "spam"},
    "electricity": {"id": 151, "target": "class", "rename_target": "price_up"},
    "mushroom": {"id": 24, "target": "class", "rename_target": "poisonous"},
    "tic_tac_toe": {"id": 50, "target": "Class", "rename_target": "x_wins"},
    "kr_vs_kp": {"id": 3, "target": "class", "rename_target": "white_wins"},
    "banknote": {"id": 1462, "target": "Class", "rename_target": "authentic"},
    "sick": {"id": 38, "target": "Class", "rename_target": "thyroid_sick"},
    "phoneme": {"id": 1489, "target": "Class", "rename_target": "phoneme"},
    "credit_approval": {"id": 29, "target": "class", "rename_target": "credit_approved"},
    "jm1": {"id": 1053, "target": "defects", "rename_target": "defect"},
    # ── Binary classification — environment/safety ──
    "wilt": {"id": 40983, "target": "class", "rename_target": "wilted"},
    "climate": {"id": 40994, "target": "outcome", "rename_target": "crashed"},
    "ozone": {"id": 1487, "target": "Class", "rename_target": "ozone_day"},
    "seismic": {"id": 40878, "target": "class", "rename_target": "hazardous"},
    # ── Binary classification — medical ──
    "mammography": {"id": 310, "target": "class", "rename_target": "malignant"},
    "ilpd": {"id": 1480, "target": "Class", "rename_target": "liver_disease"},
    "hypothyroid": {"id": 57, "target": "Class", "rename_target": "hypothyroid"},
    # ── Binary classification — cybersecurity/toxicity ──
    "toxicity": {"id": 44160, "target": "class", "rename_target": "toxic"},
    "phishing": {"id": 4534, "target": "Result", "rename_target": "phishing"},
    "qsar_biodeg": {"id": 1494, "target": "Class", "rename_target": "biodegradable"},
    # ── Binary classification — social/fun ──
    "speed_dating": {"id": 40536, "target": "match", "rename_target": "match"},
    # ── Binary classification — neuroscience/IoT ──
    "eeg_eye_state": {"id": 1471, "target": "Class", "rename_target": "eyes_open"},
    "hill_valley": {"id": 1479, "target": "Class", "rename_target": "hill"},
    # ── Binary classification — software/tech ──
    "kc2": {"id": 1063, "target": "problems", "rename_target": "defective"},
    "pc1": {"id": 1068, "target": "defects", "rename_target": "defective"},
    "amazon_employee": {"id": 4135, "target": "target", "rename_target": "access_granted"},
    "nomao": {"id": 1486, "target": "Class", "rename_target": "valid_place"},
    "churn_telco": {"id": 40701, "target": "class", "rename_target": "churned"},
    "jasmine": {"id": 41143, "target": "class", "rename_target": "positive"},
    # ── Binary classification — CC18 benchmark ──
    "pc3": {"id": 1050, "target": "c", "rename_target": "defective"},
    "pc4": {"id": 1049, "target": "c", "rename_target": "defective"},
    "kc1": {"id": 1067, "target": "defects", "rename_target": "defective"},
    "wdbc": {"id": 1510, "target": "Class", "rename_target": "malignant"},
    "bioresponse": {"id": 4134, "target": "target", "rename_target": "active"},
    "madelon": {"id": 1485, "target": "Class", "rename_target": "positive"},
    "cylinder_bands": {"id": 6332, "target": "band_type", "rename_target": "defective"},
    "dresses_sales": {"id": 23381, "target": "Class", "rename_target": "sold"},
    "internet_ads": {"id": 40978, "target": "class", "rename_target": "is_ad"},
    # ── Binary classification — ecommerce/web ──
    "online_shoppers": {"id": 42993, "target": "Revenue", "rename_target": "purchased"},
    # ── Binary classification — exotic/social/molecular ──
    "vote": {"id": 56, "target": "Class", "rename_target": "party"},
    "hepatitis": {"id": 55, "target": "Class", "rename_target": "survived"},
    "musk": {"id": 1116, "target": "class", "rename_target": "musk"},
    "qsar_oral_toxicity": {"id": 1504, "target": "Class", "rename_target": "toxic"},
    "speech": {"id": 40910, "target": "Target", "rename_target": "spoken"},
    # ── Multiclass classification ──
    "penguins": {"id": 42585, "target": "species"},
    "glass": {"id": 41, "target": "Type", "rename_target": "glass_type"},
    "ecoli": {"id": 39, "target": "class", "rename_target": "localization"},
    "yeast": {"id": 181, "target": "class_protein_localization", "rename_target": "localization"},
    "segment": {"id": 36, "target": "class", "rename_target": "segment"},
    "vehicle": {"id": 54, "target": "Class", "rename_target": "vehicle_type"},
    "vowel": {"id": 307, "target": "Class", "rename_target": "vowel"},
    "car": {"id": 40975, "target": "class", "rename_target": "evaluation"},
    "nursery": {"id": 26, "target": "class", "rename_target": "recommendation"},
    "letter": {"id": 6, "target": "class", "rename_target": "letter"},
    "satimage": {"id": 182, "target": "class", "rename_target": "land_use"},
    "optdigits": {"id": 28, "target": "class", "rename_target": "digit"},
    "page_blocks": {"id": 30, "target": "class", "rename_target": "block_type"},
    "cmc": {"id": 23, "target": "Contraceptive_method_used", "rename_target": "method"},
    "dermatology": {"id": 35, "target": "class", "rename_target": "disease"},
    "waveform": {"id": 60, "target": "class", "rename_target": "waveform"},
    "balance_scale": {"id": 11, "target": "class", "rename_target": "balance"},
    "zoo": {"id": 62, "target": "type", "rename_target": "animal_type"},
    "splice": {"id": 46, "target": "Class", "rename_target": "splice_type"},
    "pendigits": {"id": 32, "target": "class", "rename_target": "digit"},
    "eucalyptus": {"id": 188, "target": "Utility", "rename_target": "utility"},
    "steel_plates": {"id": 40982, "target": "target", "rename_target": "fault_type"},
    "connect_4": {"id": 40668, "target": "class", "rename_target": "outcome"},
    "mfeat_factors": {"id": 12, "target": "class", "rename_target": "digit"},
    "wall_robot": {"id": 1497, "target": "Class", "rename_target": "direction"},
    "texture": {"id": 40499, "target": "Class", "rename_target": "texture_class"},
    "jungle_chess": {"id": 41027, "target": "class", "rename_target": "outcome"},
    "miniboone": {"id": 41146, "target": "class", "rename_target": "signal"},
    # ── Multiclass — science/biology ──
    "dna": {"id": 40670, "target": "class", "rename_target": "dna_class"},
    "soybean": {"id": 42, "target": "class", "rename_target": "disease"},
    "mice_protein": {"id": 40966, "target": "class", "rename_target": "genotype"},
    "drug_consumption": {"id": 1560, "target": "Class", "rename_target": "drug_use"},
    "cardiotocography": {"id": 1466, "target": "Class", "rename_target": "fetal_state"},
    "seeds": {"id": 1499, "target": "Class", "rename_target": "variety"},
    "plants_margin": {"id": 1493, "target": "Class", "rename_target": "species"},
    "plants_shape": {"id": 1491, "target": "Class", "rename_target": "species"},
    # ── Multiclass — signal/speech/images ──
    "har": {"id": 1478, "target": "Class", "rename_target": "activity"},
    "isolet": {"id": 300, "target": "class", "rename_target": "letter"},
    "semeion": {"id": 1501, "target": "Class", "rename_target": "digit"},
    "gas_drift": {"id": 1476, "target": "Class", "rename_target": "gas_type"},
    "first_order": {"id": 1475, "target": "Class", "rename_target": "theorem_class"},
    "authorship": {"id": 458, "target": "Author", "rename_target": "author"},
    "artificial_chars": {"id": 1459, "target": "Class", "rename_target": "character"},
    # ── Multiclass — medical/exotic ──
    "arrhythmia": {"id": 5, "target": "class", "rename_target": "arrhythmia_type"},
    # ── Multiclass — CC18 benchmark ──
    "mfeat_fourier": {"id": 14, "target": "class", "rename_target": "digit"},
    "mfeat_karhunen": {"id": 16, "target": "class", "rename_target": "digit"},
    "mfeat_morphological": {"id": 18, "target": "class", "rename_target": "digit"},
    "mfeat_zernike": {"id": 22, "target": "class", "rename_target": "digit"},
    "mfeat_pixel": {"id": 40979, "target": "class", "rename_target": "digit"},
    "dental": {"id": 469, "target": "Prevention", "rename_target": "prevention"},
    "cnae9": {"id": 1468, "target": "Class", "rename_target": "category"},
    "gesture": {"id": 4538, "target": "Phase", "rename_target": "phase"},
    # ── Regression / survival ──
    "veteran": {"id": 497, "target": "Survival", "rename_target": "survival_days"},
    # ── Regression ──
    "diamonds": {"id": 42225, "target": "price"},
    "abalone": {"id": 183, "target": "Class_number_of_rings", "rename_target": "rings", "numeric_target": True},
    "mpg": {"id": 196, "target": "class", "rename_target": "mpg", "numeric_target": True},
    "bike": {"id": 42712, "target": "count"},
    "wine_quality": {"id": 287, "target": "quality", "numeric_target": True},
    "ames": {"id": 42165, "target": "SalePrice", "rename_target": "price"},
    "concrete": {"id": 4353, "target": "Concrete compressive strength(MPa. megapascals)", "rename_target": "strength"},
    "kin8nm": {"id": 189, "target": "y", "rename_target": "position"},
    "cpu_act": {"id": 197, "target": "usr", "rename_target": "cpu_usage", "numeric_target": True},
    "delta_elevators": {"id": 198, "target": "Se", "rename_target": "response", "numeric_target": True},
    "bodyfat": {"id": 560, "target": "class", "rename_target": "bodyfat", "numeric_target": True},
    "machine_cpu": {"id": 230, "target": "class", "rename_target": "performance", "numeric_target": True},
    "stock": {"id": 223, "target": "company10", "rename_target": "stock_price"},
    "auto_price": {"id": 195, "target": "price"},
    "elevators": {"id": 216, "target": "Goal", "rename_target": "response"},
    "boston": {"id": 531, "target": "MEDV", "rename_target": "price"},
    "energy": {"id": 44960, "target": "heating_load"},
    "superconduct": {"id": 44964, "target": "critical_temp"},
    "white_wine": {"id": 44971, "target": "quality", "numeric_target": True, "rename_target": "quality"},
    "red_wine": {"id": 44972, "target": "quality", "numeric_target": True, "rename_target": "quality"},
    "miami_housing": {"id": 44983, "target": "SALE_PRC", "rename_target": "price"},
    "cars": {"id": 44994, "target": "Price", "rename_target": "price"},
    "house_16h": {"id": 574, "target": "price"},
    # ── Regression — CTR23 benchmark ──
    "airfoil": {"id": 44957, "target": "sound_pressure"},
    "qsar_aquatic": {"id": 44958, "target": "verification.time", "rename_target": "toxicity_time"},
    "protein_structure": {"id": 44963, "target": "RMSD", "rename_target": "rmsd"},
    "communities": {"id": 44965, "target": "latitude"},
    "solar_flare": {"id": 44966, "target": "c_class_flares", "rename_target": "flares", "numeric_target": True},
    "gas_turbine": {"id": 44969, "target": "gt_compressor_decay_state_coefficient", "rename_target": "decay"},
    "electrical_grid": {"id": 44973, "target": "stab", "rename_target": "stability"},
    "kin_dynamics": {"id": 44981, "target": "thetadd6", "rename_target": "angular_accel"},
    "wages": {"id": 44984, "target": "wage"},
    "house_sales": {"id": 44989, "target": "price"},
    "wind": {"id": 44990, "target": "total", "rename_target": "wind_power"},
    "fps_benchmark": {"id": 44992, "target": "FPS", "rename_target": "fps"},
    "work_hours": {"id": 44993, "target": "whrswk", "rename_target": "hours_per_week"},
    "fifa_wages": {"id": 45012, "target": "wage_eur", "rename_target": "wage"},
    "debutanizer": {"id": 41021, "target": "RS", "rename_target": "butane"},
    "forest_fires": {"id": 44962, "target": "area", "rename_target": "burned_area"},
    "student_performance": {"id": 44967, "target": "G3", "rename_target": "final_grade", "numeric_target": True},
    "qsar_fish": {"id": 44970, "target": "LC50", "rename_target": "toxicity"},
    "space_voting": {"id": 45402, "target": "ln_votes_pop", "rename_target": "vote_share"},
    "social_mobility": {"id": 44987, "target": "counts_for_sons_current_occupation", "rename_target": "occupation_count"},
}


def _load_openml(name: str) -> pd.DataFrame:
    """Load a dataset from OpenML (cached after first download).

    Cleans columns: converts categoricals to strings, renames targets
    to friendly names, drops noisy columns where specified.
    Requires scikit-learn for the initial network download.
    """
    import warnings

    try:
        from sklearn.datasets import fetch_openml
    except ImportError as e:
        from ._types import DataError
        raise DataError(
            f"Loading '{name}' from OpenML requires scikit-learn for the initial download. "
            "Install it with: pip install scikit-learn"
        ) from e

    from ._types import DataError

    if name not in _OPENML_REGISTRY:
        raise DataError(f"Unknown OpenML dataset: {name}")

    spec = _OPENML_REGISTRY[name]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bunch = fetch_openml(data_id=spec["id"], as_frame=True, parser="auto")
        df = bunch.frame.copy()
    except Exception as e:
        raise DataError(
            f"Failed to load '{name}' from OpenML (requires internet on first use). "
            f"Error: {e}"
        ) from e

    # Keep only specified columns (if any)
    if "keep" in spec:
        df = df[[c for c in spec["keep"] if c in df.columns]]

    # Drop specified columns
    if "drop" in spec:
        df = df.drop(columns=[c for c in spec["drop"] if c in df.columns])

    # Convert categoricals to plain strings (cleaner for mlw)
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].replace("nan", None)

    # Map target values if specified
    if "map_target" in spec:
        target_col = spec["target"]
        if target_col in df.columns:
            df[target_col] = df[target_col].map(spec["map_target"])

    # Rename target column if specified
    if "rename_target" in spec:
        old_name = spec["target"]
        new_name = spec["rename_target"]
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})

    # Convert target to float64 if specified (e.g. abalone rings, mpg, grades)
    # float64 signals regression intent — avoids the classification heuristic
    # in ml.split() which triggers for integer columns with few unique values
    if spec.get("numeric_target"):
        target_col = spec.get("rename_target", spec["target"])
        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype("float64")

    return df


def _create_patients_dataset() -> pd.DataFrame:
    """Load real hospital readmission data (OpenML 4541, subsampled).

    ~424 rows, 100 patients with 3+ visits each. Real group structure
    for demonstrating groups= parameter in split().
    Binary classification: readmitted within 30 days (yes/no).
    """
    import warnings

    import numpy as np

    try:
        from sklearn.datasets import fetch_openml
    except ImportError as e:
        from ._types import DataError
        raise DataError(
            "The 'patients' dataset requires scikit-learn for the initial download. "
            "Install it with: pip install scikit-learn"
        ) from e

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bunch = fetch_openml(data_id=4541, as_frame=True, parser="auto")
    df = bunch.frame

    # Keep patients with 3+ visits (real group structure)
    visits = df["patient_nbr"].value_counts()
    multi_patients = visits[visits >= 3].index.tolist()

    # Sample 100 patients for manageable size
    rng = np.random.RandomState(42)
    sample_patients = rng.choice(multi_patients, size=100, replace=False)
    df = df[df["patient_nbr"].isin(sample_patients)].copy()

    # Select clinically meaningful columns
    keep_cols = [
        "patient_nbr", "race", "gender", "age",
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_emergency", "number_inpatient",
        "number_diagnoses", "A1Cresult", "readmitted",
    ]
    df = df[keep_cols].copy()

    # Binary target: <30 days = "yes", else "no"
    df["readmitted"] = df["readmitted"].map(
        {"<30": "yes", ">30": "no", "NO": "no"}
    )
    df = df.rename(columns={"patient_nbr": "patient_id"})
    return df.reset_index(drop=True)


def _create_sales_dataset() -> pd.DataFrame:
    """Create synthetic daily sales dataset with timestamps.

    730 rows (2 years of daily data). Natural time structure
    for demonstrating time= parameter in split().
    Regression: daily revenue.
    """
    import numpy as np

    rng = np.random.RandomState(42)
    n = 730  # 2 years

    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    day_of_week = dates.dayofweek  # 0=Mon, 6=Sun
    month = dates.month

    # Features
    is_weekend = (day_of_week >= 5).astype(int)
    temperature = 15 + 15 * np.sin(2 * np.pi * (month - 4) / 12) + rng.normal(0, 3, n)
    ad_spend = rng.lognormal(6, 0.8, size=n).clip(100, 10000)
    n_staff = rng.choice([3, 4, 5, 6, 7], size=n, p=[0.1, 0.2, 0.3, 0.25, 0.15])
    foot_traffic = (
        200 + is_weekend * 80 + (temperature > 20) * 40
        + rng.poisson(30, size=n)
    )

    # Target: daily revenue
    revenue = (
        500
        + foot_traffic * 2.5
        + ad_spend * 0.03
        + n_staff * 50
        + is_weekend * 150
        - (temperature < 5) * 100
        + rng.normal(0, 80, size=n)
    ).clip(200, 5000)

    df = pd.DataFrame({
        "date": dates,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "temperature": np.round(temperature, 1),
        "ad_spend": np.round(ad_spend, 2),
        "n_staff": n_staff,
        "foot_traffic": foot_traffic,
        "revenue": np.round(revenue, 2),
    })

    return df


def _create_reviews_dataset() -> pd.DataFrame:
    """Create synthetic product reviews dataset with TEXT features.

    2000 rows. Demonstrates text + numeric + categorical mixed features.
    Binary classification: positive/negative sentiment.
    Text columns need preprocessing before fit() — teaches real-world workflow.
    """
    import numpy as np

    rng = np.random.RandomState(42)
    n = 2000

    # Text building blocks
    pos_phrases = [
        "love this product", "works great", "highly recommend",
        "excellent quality", "best purchase ever", "very satisfied",
        "amazing value", "perfect fit", "exceeded expectations",
        "would buy again", "fantastic", "outstanding",
    ]
    neg_phrases = [
        "terrible quality", "waste of money", "broke after a week",
        "very disappointed", "would not recommend", "poor design",
        "cheaply made", "does not work", "horrible experience",
        "want a refund", "awful", "complete garbage",
    ]
    neutral_phrases = [
        "it's okay", "nothing special", "average product",
        "meets expectations", "decent for the price", "not bad",
    ]

    categories = ["electronics", "clothing", "home", "sports", "books"]
    sentiment = rng.choice(["positive", "negative"], n, p=[0.65, 0.35])

    # Generate review text based on sentiment
    review_text = []
    for s in sentiment:
        if s == "positive":
            phrases = rng.choice(pos_phrases, size=rng.randint(1, 4), replace=True)
        else:
            phrases = rng.choice(neg_phrases, size=rng.randint(1, 4), replace=True)
        # Occasionally mix in neutral
        if rng.rand() < 0.2:
            phrases = list(phrases) + list(rng.choice(neutral_phrases, 1))
        review_text.append(". ".join(phrases))

    rating = np.where(sentiment == "positive",
                      rng.choice([4, 5], n), rng.choice([1, 2], n))
    # Add noise — some ratings don't match sentiment
    noise_mask = rng.rand(n) < 0.05
    rating[noise_mask] = rng.randint(1, 6, size=noise_mask.sum())

    word_count = np.array([len(t.split()) for t in review_text])
    verified = rng.choice([True, False], n, p=[0.7, 0.3])
    category = rng.choice(categories, n)
    price = rng.lognormal(3.5, 1.0, n).clip(5, 500).round(2)
    helpful_votes = rng.poisson(3, n)

    df = pd.DataFrame({
        "review_text": review_text,
        "rating": rating,
        "word_count": word_count,
        "verified_purchase": verified,
        "category": category,
        "price": price,
        "helpful_votes": helpful_votes,
        "sentiment": sentiment,
    })

    return df


def _create_ecommerce_dataset() -> pd.DataFrame:
    """Create synthetic e-commerce dataset with HIGH-CARDINALITY categoricals.

    5000 rows. Demonstrates zip codes (500+ levels), product IDs (200+ levels),
    and other high-cardinality string features. Binary classification: returned.
    """
    import numpy as np

    rng = np.random.RandomState(42)
    n = 5000

    # High-cardinality: zip codes (500 unique)
    zip_pool = [f"{rng.randint(10000, 99999)}" for _ in range(500)]
    zip_code = rng.choice(zip_pool, n)

    # High-cardinality: product IDs (200 unique)
    product_pool = [f"SKU-{i:04d}" for i in range(200)]
    product_id = rng.choice(product_pool, n)

    # Medium-cardinality: brand (50 unique)
    brand_pool = [f"brand_{i}" for i in range(50)]
    brand = rng.choice(brand_pool, n)

    # Low-cardinality
    category = rng.choice(["electronics", "clothing", "home", "food", "sports"], n)
    channel = rng.choice(["web", "mobile", "store"], n, p=[0.5, 0.35, 0.15])

    # Numeric
    price = rng.lognormal(3.5, 1.2, n).clip(2, 2000).round(2)
    quantity = rng.choice([1, 1, 1, 2, 2, 3, 4, 5], n)
    discount_pct = rng.choice([0, 0, 0, 5, 10, 15, 20, 25, 30], n)

    # Target: return probability depends on features
    return_prob = (
        0.08
        + (price > 200) * 0.10
        + (discount_pct > 20) * 0.05
        + (quantity > 2) * 0.03
        + (np.array([c == "clothing" for c in category])) * 0.12
    )
    return_prob = return_prob.clip(0, 0.6)
    returned = np.where(rng.binomial(1, return_prob), "yes", "no")

    df = pd.DataFrame({
        "product_id": product_id,
        "brand": brand,
        "category": category,
        "channel": channel,
        "zip_code": zip_code,
        "price": price,
        "quantity": quantity,
        "discount_pct": discount_pct,
        "returned": returned,
    })

    return df


def _create_tips_dataset() -> pd.DataFrame:
    """Load real restaurant tips dataset (Bryant & Smith 1995).

    244 rows × 7 columns. One waiter's tips over 2.5 months.
    Regression: tip amount. Has string categoricals, numeric, small.
    """
    from pathlib import Path

    csv_path = Path(__file__).parent / "data" / "tips.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    # Fallback: fetch from web if CSV not bundled
    return pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    )


def _create_flights_dataset() -> pd.DataFrame:
    """Load real airline passenger dataset (Box & Jenkins 1976).

    144 rows (1949-1960 × 12 months). THE classic time series benchmark.
    Regression: monthly passenger count. Real trend + seasonality.
    """
    from pathlib import Path

    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }

    csv_path = Path(__file__).parent / "data" / "flights.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(
            "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv"
        )

    df["month"] = df["month"].map(month_map)
    df["date"] = pd.to_datetime(
        [f"{y}-{m:02d}-01" for y, m in zip(df["year"], df["month"])]
    )
    return df[["date", "year", "month", "passengers"]]


def _create_survival_dataset() -> pd.DataFrame:
    """Create synthetic survival/time-to-event dataset with CENSORED data.

    1000 rows. Medical trial with censoring — some patients still alive at study end.
    Regression on time column, but event column marks censoring (0=censored, 1=event).
    Demonstrates right-censored survival data pattern.
    """
    import numpy as np

    rng = np.random.RandomState(42)
    n = 1000

    # Patient features
    age = rng.randint(30, 80, n)
    treatment = rng.choice(["drug_A", "drug_B", "placebo"], n, p=[0.4, 0.4, 0.2])
    stage = rng.choice([1, 2, 3, 4], n, p=[0.3, 0.3, 0.25, 0.15])
    biomarker = rng.normal(50, 15, n).clip(10, 100).round(1)
    prior_therapy = rng.choice([True, False], n, p=[0.35, 0.65])

    # Generate survival times (Weibull-like)
    # Higher stage and age = shorter survival
    scale = (
        500
        - (age - 30) * 2
        - (stage - 1) * 80
        + (np.array([t == "drug_A" for t in treatment])) * 60
        + (np.array([t == "drug_B" for t in treatment])) * 40
        - (biomarker > 70) * 50
    ).clip(50, 800)
    true_time = rng.exponential(scale).clip(1, 2000).round(0).astype(int)

    # Censoring: study ends at day 730 (2 years)
    study_end = 730
    observed_time = np.minimum(true_time, study_end)
    event = (true_time <= study_end).astype(int)  # 1=event, 0=censored

    df = pd.DataFrame({
        "age": age,
        "treatment": treatment,
        "stage": stage,
        "biomarker": biomarker,
        "prior_therapy": prior_therapy,
        "time": observed_time,
        "event": event,
    })

    return df


def datasets() -> pd.DataFrame:
    """List all available datasets with quality metadata.

    Returns a DataFrame with one row per dataset showing:
    name, rows, columns, task, target, n_classes, nan_pct, source.

    Example:
        >>> ml.datasets()
        >>> ml.datasets().query("task == 'regression'")
    """
    # Metadata for synthetic datasets (no download needed)
    _SYNTHETIC_META = {
        "churn": ("churn", 7043, 20, "classification", 2, 0.2, "openml"),
        "fraud": ("fraud", 9992, 31, "classification", 2, 0.0, "openml"),
        "patients": ("readmitted", 424, 13, "classification", 2, 0.0, "openml"),
        "sales": ("revenue", 730, 8, "regression", None, 0.0, "synthetic"),
        "reviews": ("sentiment", 2000, 8, "classification", 2, 0.0, "synthetic"),
        "ecommerce": ("purchased", 12330, 18, "classification", 2, 0.0, "openml"),
        "tips": ("tip", 244, 7, "regression", None, 0.0, "csv"),
        "flights": ("passengers", 144, 4, "regression", None, 0.0, "csv"),
        "survival": ("survival_days", 137, 8, "regression", None, 0.0, "openml"),
    }
    _SKLEARN_META = {
        "iris": ("species", 150, 5, "classification", 3, 0.0, "sklearn"),
        "wine": ("cultivar", 178, 14, "classification", 3, 0.0, "sklearn"),
        "cancer": ("diagnosis", 569, 31, "classification", 2, 0.0, "sklearn"),
        "diabetes": ("progression", 442, 11, "regression", None, 0.0, "sklearn"),
        "houses": ("price", 20640, 9, "regression", None, 0.0, "sklearn"),
    }

    rows = []
    # Synthetic
    for name, (target, n_rows, n_cols, task, n_cls, nan_pct, source) in _SYNTHETIC_META.items():
        rows.append({
            "name": name, "rows": n_rows, "cols": n_cols, "task": task,
            "target": target, "n_classes": n_cls, "nan_pct": nan_pct, "source": source,
        })
    # sklearn
    for name, (target, n_rows, n_cols, task, n_cls, nan_pct, source) in _SKLEARN_META.items():
        rows.append({
            "name": name, "rows": n_rows, "cols": n_cols, "task": task,
            "target": target, "n_classes": n_cls, "nan_pct": nan_pct, "source": source,
        })
    # OpenML — derive from registry
    _CLF_TARGETS = {
        "heart_disease", "diabetes", "survived", "donated", "malignant",
        "credit_risk", "approved", "subscribed", "income", "is_mine",
        "ionosphere", "spam", "price_up", "poisonous", "x_wins",
        "white_wins", "authentic", "thyroid_sick", "phoneme",
        "credit_approved", "defect", "glass_type", "localization",
        "segment", "vehicle_type", "vowel", "evaluation",
        "recommendation", "letter", "land_use", "digit", "block_type",
        "method", "disease", "waveform", "balance", "animal_type",
        "splice_type", "utility", "fault_type", "outcome", "species",
        "wilted", "crashed", "ozone_day", "hazardous",
        "liver_disease", "hypothyroid", "toxic", "phishing",
        "biodegradable", "match", "direction", "texture_class", "signal",
        "eyes_open", "hill", "defective", "access_granted", "valid_place",
        "churned", "positive", "dna_class", "genotype",
        "drug_use", "fetal_state", "variety", "activity",
        "gas_type", "theorem_class", "author", "character",
        "active", "sold", "is_ad", "prevention", "category", "phase",
        "party", "musk", "spoken", "arrhythmia_type", "purchased",
    }
    for name, spec in _OPENML_REGISTRY.items():
        target = spec.get("rename_target", spec["target"])
        task = "classification" if target in _CLF_TARGETS else "regression"
        rows.append({
            "name": name, "rows": None, "cols": None, "task": task,
            "target": target, "n_classes": None, "nan_pct": None, "source": "openml",
        })

    return pd.DataFrame(rows)
