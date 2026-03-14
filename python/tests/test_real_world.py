"""Real-world dataset sweep
Floor-check: core algorithms must exceed minimum metrics on bundled real-world datasets.
Not golden pins — just "algorithm doesn't completely fail on real data."

Dataset loading smoke tests
Core algorithm floor checks (clf: accuracy; reg: r2)
"""

import warnings

import pytest

import ml

# ── Real-world suite configuration ────────────────────────────────────────────
# (dataset_name, target_col, task, metric, floor, algorithms_to_test)
_REAL_WORLD_SUITE = [
    # iris: 3-class classification, 150 rows, clean data
    ("iris",     "species",     "clf", "accuracy", 0.80, ["random_forest", "logistic", "knn"]),
    # wine: 3-class classification, 178 rows, clean data
    ("wine",     "cultivar",    "clf", "accuracy", 0.65, ["random_forest", "knn"]),
    # cancer: binary classification, 569 rows, clean data
    ("cancer",   "diagnosis",   "clf", "accuracy", 0.85, ["random_forest", "logistic", "gradient_boosting"]),
    # diabetes: regression, 442 rows, clean data
    ("diabetes", "progression", "reg", "r2",       0.25, ["random_forest", "linear", "gradient_boosting"]),
    # churn: binary clf, 7043 rows, may have NaN — use algorithms that are robust
    ("churn",    "churn",       "clf", "accuracy", 0.65, ["random_forest", "gradient_boosting"]),
    # titanic: binary clf, 1309 rows, has categorical + NaN columns
    ("titanic",  "survived",    "clf", "accuracy", 0.65, ["random_forest", "gradient_boosting"]),
]

# Flatten into (dataset, target, task, metric, floor, algorithm) tuples for parametrize
_CASES = []
_CASE_IDS = []
for ds, tgt, task, metric, floor, algos in _REAL_WORLD_SUITE:
    for algo in algos:
        _CASES.append((ds, tgt, task, metric, floor, algo))
        _CASE_IDS.append(f"{ds}/{algo}")


# ── Dataset loading smoke tests ────────────────────────────────────────

@pytest.mark.parametrize("name", ["iris", "wine", "cancer", "diabetes", "churn", "titanic"])
def test_dataset_loads(name):
    """Dataset loads successfully with positive shape."""
    df = ml.dataset(name)
    assert len(df) > 0, f"{name}: empty DataFrame"
    assert len(df.columns) >= 2, f"{name}: too few columns"


# ── Floor checks ────────────────────────────────────────────────────────

@pytest.mark.parametrize("dataset,target,task,metric,floor,algorithm", _CASES, ids=_CASE_IDS)
def test_real_world_floor(dataset, target, task, metric, floor, algorithm):
    """Core algorithm exceeds minimum metric threshold on real-world data.

    Floor values are conservative — any reasonable implementation should beat them.
    Tests that the algorithm doesn't completely fail on messy, real-world data.
    """
    df = ml.dataset(dataset)

    # Verify target column exists
    if target not in df.columns:
        pytest.skip(f"Target '{target}' not in {dataset} columns: {list(df.columns)}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            s = ml.split(data=df, target=target, seed=42)
            m = ml.fit(data=s.train, target=target, algorithm=algorithm, seed=42)
            metrics = ml.evaluate(model=m, data=s.valid)
        except Exception as e:
            pytest.fail(f"{algorithm} on {dataset}: unexpected exception: {e}")

    value = metrics.get(metric)
    if value is None:
        pytest.skip(f"Metric '{metric}' not in evaluate() output for {algorithm}/{task}")

    assert value >= floor, (
        f"{algorithm} on {dataset}: {metric}={value:.3f} < floor={floor:.3f}"
    )


# ── Profile smoke test ──────────────────────────────────────────────────

@pytest.mark.parametrize("name,target", [
    ("iris",   "species"),
    ("cancer", "diagnosis"),
])
def test_profile_runs_on_real_data(name, target):
    """ml.profile() completes without error on real-world datasets."""
    df = ml.dataset(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prof = ml.profile(data=df, target=target)
    assert isinstance(prof, dict), f"profile() should return dict, got {type(prof)}"
    assert len(prof) > 0, "profile() returned empty dict"


# ── Screen smoke test ───────────────────────────────────────────────────

def test_screen_runs_on_real_data():
    """ml.screen() completes and returns a leaderboard on iris."""
    df = ml.dataset("iris")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=df, target="species", seed=42)
        lb = ml.screen(data=s, target="species", seed=42)
    assert len(lb) > 0, "screen() returned empty leaderboard"


# ── Assess smoke test ───────────────────────────────────────────────────

def test_assess_runs_on_real_data():
    """ml.assess() completes on cancer dataset (single holdout, no repeated testing)."""
    df = ml.dataset("cancer")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ml.split(data=df, target="diagnosis", seed=42)
        m = ml.fit(data=s.dev, target="diagnosis", algorithm="random_forest", seed=42)
        verdict = ml.assess(model=m, test=s.test)
    assert "accuracy" in verdict, "assess() should return dict with 'accuracy'"
    assert verdict["accuracy"] >= 0.80, f"random_forest on cancer: assess accuracy={verdict['accuracy']:.3f} < 0.80"
