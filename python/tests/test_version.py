"""Version and export tests — Chain 17."""

import ml


def test_version_is_current():
    """ml.__version__ matches expected release."""
    assert ml.__version__ == "1.2.0"


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
