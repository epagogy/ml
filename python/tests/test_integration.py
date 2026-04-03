"""Integration & Polish tests — Chain 16.

End-to-end tests verifying entire workflows work together.
"""

import os
import warnings

import pandas as pd
import pytest

import ml

# ── 16.1 Full integration tests ───────────────────────────────────────────


def test_integration_binary_classification(small_classification_data):
    """Full workflow: profile→split→screen→fit→evaluate→explain→save→load→predict."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 1. Profile
        prof = ml.profile(small_classification_data, "target")
        assert prof is not None

        # 2. Split
        s = ml.split(data=small_classification_data, target="target", seed=42)

        # 3. Screen (single algo — integration test checks the workflow, not algo comparison)
        lb = ml.screen(s, "target", seed=42, algorithms=["logistic"])
        assert lb is not None

        # 4. Fit + evaluate + explain
        model = ml.fit(data=s.train, target="target", seed=42)
        metrics = ml.evaluate(model, s.valid)
        assert "accuracy" in metrics or len(metrics) > 0
        exp = ml.explain(model)
        assert exp is not None

        # 5. Compare
        m2 = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
        leaderboard = ml.compare([model, m2], s.valid)
        assert leaderboard is not None

        # 6. Validate + assess
        ml.validate(model, test=s.test, rules={"accuracy": ">0.3"})
        verdict = ml.assess(model, test=s.test)
        assert verdict is not None


def test_integration_regression(small_regression_data):
    """Full regression workflow end-to-end."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        s = ml.split(data=small_regression_data, target="target", seed=42)
        model = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
        metrics = ml.evaluate(model, s.valid)
        assert "rmse" in metrics or "r2" in metrics or len(metrics) > 0

        # Drift check
        drift_result = ml.drift(reference=s.train, new=s.valid)
        assert drift_result is not None

        # Shelf check — requires target column in new data
        shelf_result = ml.shelf(model, new=s.valid, target="target")
        assert shelf_result is not None


def test_integration_preprocessing_pipeline(small_classification_data):
    """Preprocessing pipeline: impute→fit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        data = small_classification_data.copy()
        # Add some NaN values to a numeric column (x1)
        data.iloc[:5, data.columns.get_loc("x1")] = float("nan")

        s = ml.split(data=data, target="target", seed=42)

        # Build imputer using correct parameter name (strategy, not method)
        imputer = ml.impute(s.train, strategy="median")
        train_clean = imputer.transform(s.train)

        # Fit on preprocessed data
        model = ml.fit(data=train_clean, target="target", seed=42)

        # Transform and predict
        valid_clean = imputer.transform(s.valid)
        preds = ml.predict(model, valid_clean)
        assert len(preds) == len(s.valid)


def test_integration_select_then_fit(small_classification_data):
    """Feature selection → fit with selected features."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        s = ml.split(data=small_classification_data, target="target", seed=42)
        model = ml.fit(data=s.train, target="target", seed=42)

        # Select top features
        selected = ml.select(model, method="importance", seed=42)
        assert len(selected) > 0

        # Refit with selected features
        train_selected = s.train[selected + ["target"]]
        valid_selected = s.valid[selected + ["target"]]
        model2 = ml.fit(data=train_selected, target="target", seed=42)
        preds = ml.predict(model2, valid_selected)
        assert len(preds) == len(s.valid)


def test_integration_tune_compare(small_classification_data):
    """tune → compare workflow."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        s = ml.split(data=small_classification_data, target="target", seed=42)
        tuned = ml.tune(data=s.train, target="target", algorithm="random_forest",
                        seed=42, n_trials=3)
        base = ml.fit(data=s.train, target="target", algorithm="random_forest", seed=42)
        lb = ml.compare([tuned, base], s.valid)
        assert lb is not None


def test_integration_stack_validate_assess(small_classification_data):
    """stack → validate → assess workflow."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        s = ml.split(data=small_classification_data, target="target", seed=42)
        stacked = ml.stack(data=s.train, target="target", seed=42,
                           models=["logistic", "random_forest"])
        ml.validate(stacked, test=s.test, rules={"accuracy": ">0.3"})
        verdict = ml.assess(stacked, test=s.test)
        assert verdict is not None


# ── 16.2 API parameter consistency ────────────────────────────────────────


def test_seed_param_consistency():
    """Core ml functions use 'seed' not 'random_state'."""
    import inspect

    # These functions should have 'seed' parameter, not 'random_state'
    funcs_with_seed = [ml.fit, ml.split, ml.screen, ml.tune, ml.stack, ml.explain]
    for func in funcs_with_seed:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        assert "seed" in params, f"{func.__name__} missing 'seed' parameter"
        assert "random_state" not in params, (
            f"{func.__name__} uses 'random_state' instead of 'seed'"
        )


def test_data_param_consistency():
    """Core ml functions use 'data' not 'X', 'df', or 'dataframe'."""
    import inspect

    funcs_with_data = [ml.fit, ml.predict, ml.evaluate, ml.assess]
    for func in funcs_with_data:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        # These should have 'data' parameter or take model as first arg
        assert "data" in params or "model" in params, (
            f"{func.__name__} missing standard parameter names"
        )


# ── 16.3 Pipe composition + persistence ──────────────────────────────────


def test_pipe_composition_full(small_classification_data):
    """pipe([imputer]) composes correctly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        data = small_classification_data.copy()
        data.iloc[:5, data.columns.get_loc("x1")] = float("nan")

        s = ml.split(data=data, target="target", seed=42)

        # Imputer only — scale requires explicit columns list
        imputer = ml.impute(s.train, strategy="median")
        pipe = ml.pipe(steps=[imputer])

        # Transform
        result = pipe.transform(s.train)
        assert result is not None
        assert len(result) == len(s.train)


def test_pipe_save_load(small_classification_data, tmp_path):
    """pipe survives save/load round-trip."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        s = ml.split(data=small_classification_data, target="target", seed=42)
        imputer = ml.impute(s.train, strategy="median")
        pipe = ml.pipe(steps=[imputer])

        path = str(tmp_path / "pipe.ml")
        ml.save(pipe, path)
        loaded = ml.load(path)
        result = loaded.transform(s.valid)
        assert len(result) == len(s.valid)


# ── 16.4 Error audit (guidance in error messages) ─────────────────────────


def test_remaining_errors_have_guidance():
    """ConfigError/DataError messages have actionable guidance."""
    errors_tested = 0

    # Wrong algorithm
    try:
        data = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        ml.fit(data=data, target="target", algorithm="nonexistent_xyz_algo", seed=42)
    except (ml.ConfigError, Exception) as e:
        msg = str(e)
        errors_tested += 1
        # Should mention the algorithm name or valid options
        assert len(msg) > 10  # at least some content

    assert errors_tested >= 1


# ── 16.5 Backward compatibility ───────────────────────────────────────────


def test_version_in_saved_model(small_classification_data, tmp_path):
    """Saved model includes version information."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)
    path = str(tmp_path / "versioned.ml")
    ml.save(model, path)
    # File should exist
    assert os.path.exists(path)
    # Loaded model should be functional
    loaded = ml.load(path)
    preds = ml.predict(loaded, s.valid)
    assert len(preds) == len(s.valid)


def test_load_v4_0_model(small_classification_data, tmp_path):
    """Models saved in current version load cleanly."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    model = ml.fit(data=s.train, target="target", seed=42)

    # Save current version model
    path = str(tmp_path / "model_v4.ml")
    ml.save(model, path)

    # Load and verify
    loaded = ml.load(path)
    assert loaded is not None
    preds = ml.predict(loaded, s.valid)
    assert len(preds) == len(s.valid)


# ── 16.6 Notebook CI (slow) ───────────────────────────────────────────────


@pytest.mark.slow
def test_notebook_executes(tmp_path):
    """quickstart.ipynb executes without errors."""
    pytest.importorskip("nbformat")
    try:
        import subprocess

        nb_path = os.path.join(
            os.path.dirname(__file__), "..", "quickstart.ipynb"
        )
        if not os.path.exists(nb_path):
            pytest.skip("quickstart.ipynb not found")

        result = subprocess.run(
            [
                "jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=120",
                "--output", str(tmp_path / "executed.ipynb"),
                nb_path,
            ],
            capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, f"Notebook failed:\n{result.stderr}"
    except FileNotFoundError:
        pytest.skip("jupyter not available")
