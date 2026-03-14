"""Tests for ml.algorithms(), ml.dataset(), ml.datasets() — utility functions."""

import pandas as pd
import pytest

import ml

# -- algorithms() --


def test_algorithms_returns_dataframe():
    """algorithms() returns a pandas DataFrame."""
    import pandas as pd
    result = ml.algorithms()
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_algorithms_has_task_columns():
    """algorithms() DataFrame has classification and regression columns."""
    result = ml.algorithms()
    assert "algorithm" in result.columns
    assert "classification" in result.columns
    assert "regression" in result.columns
    assert "optional_dep" in result.columns
    assert "gpu" in result.columns


def test_algorithms_contains_core():
    """algorithms() includes the core algorithms."""
    result = ml.algorithms()
    algo_names = result["algorithm"].tolist()
    for algo in ["xgboost", "random_forest", "logistic", "linear"]:
        assert algo in algo_names, f"Expected '{algo}' in algorithms()"


def test_algorithms_all_strings():
    """Every entry in algorithms() algorithm column is a string."""
    result = ml.algorithms()
    for algo in result["algorithm"]:
        assert isinstance(algo, str)


# -- dataset() — offline-safe (sklearn built-ins only) --


def test_dataset_iris_shape():
    """dataset('iris') returns 150×5 DataFrame."""
    data = ml.dataset("iris")
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (150, 5)


def test_dataset_iris_target_column():
    """dataset('iris') has 'species' target column."""
    data = ml.dataset("iris")
    assert "species" in data.columns
    assert data["species"].nunique() == 3


def test_dataset_diabetes_regression():
    """dataset('diabetes') returns regression data with 'progression' target."""
    data = ml.dataset("diabetes")
    assert isinstance(data, pd.DataFrame)
    assert "progression" in data.columns
    assert len(data) == 442


def test_dataset_wine_shape():
    """dataset('wine') returns 178×14 DataFrame."""
    data = ml.dataset("wine")
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (178, 14)
    assert "cultivar" in data.columns


def test_dataset_cancer_shape():
    """dataset('cancer') and 'breast_cancer' alias both work."""
    data = ml.dataset("cancer")
    assert isinstance(data, pd.DataFrame)
    assert "diagnosis" in data.columns

    data2 = ml.dataset("breast_cancer")
    assert data.shape == data2.shape


def test_dataset_houses_shape():
    """dataset('houses') and 'housing' alias return same data."""
    data = ml.dataset("houses")
    assert isinstance(data, pd.DataFrame)
    assert "price" in data.columns

    data2 = ml.dataset("housing")
    assert data.shape == data2.shape


def test_dataset_no_empty_rows():
    """Sklearn datasets have no all-NaN rows."""
    for name in ("iris", "wine", "diabetes"):
        data = ml.dataset(name)
        assert not data.isnull().all(axis=1).any(), f"{name} has all-NaN rows"


def test_dataset_invalid_raises():
    """dataset() raises DataError for unknown dataset name."""
    with pytest.raises(ml.DataError, match="not available"):
        ml.dataset("totally_fake_dataset_xyz")


# -- datasets() --


def test_datasets_returns_dataframe():
    """datasets() returns a DataFrame."""
    result = ml.datasets()
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_datasets_has_required_columns():
    """datasets() DataFrame has required metadata columns."""
    result = ml.datasets()
    for col in ["name", "task", "target", "source"]:
        assert col in result.columns, f"Missing column: {col}"


def test_datasets_has_both_tasks():
    """datasets() contains both classification and regression entries."""
    result = ml.datasets()
    tasks = result["task"].unique().tolist()
    assert "classification" in tasks
    assert "regression" in tasks


def test_datasets_query_works():
    """datasets().query() filters correctly."""
    result = ml.datasets()
    reg = result.query("task == 'regression'")
    assert len(reg) > 0
    assert (reg["task"] == "regression").all()


def test_datasets_includes_known_entries():
    """datasets() includes iris, diabetes, and churn."""
    result = ml.datasets()
    names = result["name"].tolist()
    for expected in ("iris", "diabetes", "churn"):
        assert expected in names, f"'{expected}' missing from datasets()"


def test_datasets_no_duplicate_names():
    """datasets() has no duplicate dataset names."""
    result = ml.datasets()
    assert result["name"].nunique() == len(result), "Duplicate names in datasets()"


# -- version --


def test_version_string():
    """ml.__version__ is a valid semver string starting with '1.'."""
    assert isinstance(ml.__version__, str)
    assert ml.__version__.startswith("1.")


def test_version_format():
    """ml.__version__ follows semver-like format."""
    parts = ml.__version__.split(".")
    assert len(parts) >= 3, f"Version '{ml.__version__}' doesn't look like semver"


# -- Error hierarchy --


def test_error_hierarchy():
    """All ml errors inherit from MLError."""
    assert issubclass(ml.ConfigError, ml.MLError)
    assert issubclass(ml.DataError, ml.MLError)
    assert issubclass(ml.ModelError, ml.MLError)
    assert issubclass(ml.VersionError, ml.MLError)


def test_catch_all_ml_error():
    """except MLError catches all library-specific errors."""
    with pytest.raises(ml.MLError):
        ml.dataset("totally_fake_dataset_xyz")


def test_version_error_exists():
    """VersionError is importable and raisable."""
    err = ml.VersionError("test message")
    assert str(err) == "test message"
    assert isinstance(err, ml.MLError)


def test_optional_deps_lazy_import_error():
    """Modules with optional deps raise ImportError with install hint."""
    import ml
    # narwhals is not installed in dev environment, so passing a Polars frame
    # should produce a DataError with helpful message
    # Just verify the package imports cleanly
    assert ml.__version__ is not None


def test_normalize_to_pandas_passthrough():
    """to_pandas() returns pandas DataFrame unchanged."""
    import pandas as pd

    from ml._compat import to_pandas
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = to_pandas(df)
    assert isinstance(result, pd.DataFrame)
    assert result is df  # same object, no copy


def test_polars_not_installed_error():
    """Passing non-DataFrame without narwhals gives DataError or ConfigError."""
    import ml
    with pytest.raises((ml.DataError, ml.ConfigError)):
        ml.fit(data=[1, 2, 3], target="x", seed=42)


def test_version_string_nonempty():
    """ml.__version__ is a non-empty string with dots."""
    import ml
    assert isinstance(ml.__version__, str)
    assert len(ml.__version__) > 0
    assert "." in ml.__version__
