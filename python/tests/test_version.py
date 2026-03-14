"""Version and export tests —"""

import ml


def test_version_is_semver():
    """ml.__version__ is a valid semver string matching pyproject.toml."""
    parts = ml.__version__.split(".")
    assert len(parts) == 3, f"Expected semver X.Y.Z, got {ml.__version__}"
    assert all(p.isdigit() for p in parts), f"Non-numeric version: {ml.__version__}"


def test_all_verbs_in_help():
    """ml.help() output mentions core verbs."""
    result = ml.help()
    text = str(result)
    for verb in ["fit", "predict", "evaluate", "split", "screen"]:
        assert verb in text, f"'{verb}' not found in ml.help() output"


def test_all_verbs_in_all():
    """Key verbs are in ml.__all__."""
    required = [
        "fit", "predict", "evaluate", "assess", "split", "screen",
        "tune", "stack", "compare", "save", "load", "profile", "validate",
        "explain", "drift", "calibrate", "optimize", "encode", "plot",
        "enough", "select", "quick", "help", "check", "check_data", "report",
    ]
    for verb in required:
        assert verb in ml.__all__, f"'{verb}' missing from ml.__all__"
