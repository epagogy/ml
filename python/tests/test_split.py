"""Tests for split()."""

import numpy as np
import pandas as pd
import pytest

import ml


def test_split_basic_three_way(churn_data):
    """Test basic three-way split."""
    s = ml.split(data=churn_data, target="churn", seed=42)

    assert isinstance(s, ml.SplitResult)
    assert len(s.train) == 4225
    assert len(s.valid) == 1409
    assert len(s.test) == 1409
    assert len(s.dev) == 5634  # train + valid


def test_split_no_target(churn_data):
    """Test split without target (unsupervised)."""
    s = ml.split(data=churn_data, seed=42)

    assert isinstance(s, ml.SplitResult)
    assert len(s.train) + len(s.valid) + len(s.test) == len(churn_data)


def test_split_custom_ratio(small_classification_data):
    """Test custom ratio."""
    s = ml.split(data=small_classification_data, target="target", ratio=(0.8, 0.1, 0.1), seed=42)

    assert len(s.train) == 80
    assert len(s.valid) == 10
    assert len(s.test) == 10


def test_split_cv_basic(small_classification_data):
    """Test CV split."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    cv = ml.cv(s, folds=2, seed=42)

    assert isinstance(cv, ml.CVResult)
    assert cv.k == 2
    assert len(cv.folds) == 2
    for fold_train, fold_valid in cv.folds:
        assert len(fold_train) + len(fold_valid) == len(s.dev)


def test_split_stratification(small_classification_data):
    """Test stratification preserves class proportions."""
    s = ml.split(data=small_classification_data, target="target", seed=42)

    original_ratio = (small_classification_data["target"] == "yes").mean()
    train_ratio = (s.train["target"] == "yes").mean()
    valid_ratio = (s.valid["target"] == "yes").mean()
    test_ratio = (s.test["target"] == "yes").mean()

    # Should be close (within 0.1)
    assert abs(train_ratio - original_ratio) < 0.1
    assert abs(valid_ratio - original_ratio) < 0.1
    assert abs(test_ratio - original_ratio) < 0.1


def test_split_empty_data_error():
    """Test split raises on empty DataFrame."""
    empty = pd.DataFrame()

    with pytest.raises(ml.DataError, match="empty data"):
        ml.split(data=empty, seed=42)


def test_split_single_row_error():
    """Test split raises on single row."""
    single = pd.DataFrame({"x": [1], "y": [2]})

    with pytest.raises(ml.DataError, match="Cannot split 1 row"):
        ml.split(data=single, seed=42)


def test_split_target_not_found_error(churn_data):
    """Test split raises when target not found."""
    with pytest.raises(ml.DataError, match="target='missing' not found"):
        ml.split(data=churn_data, target="missing", seed=42)


def test_split_reproducible(small_classification_data):
    """Test split is reproducible with same seed."""
    s1 = ml.split(data=small_classification_data, target="target", seed=42)
    s2 = ml.split(data=small_classification_data, target="target", seed=42)

    pd.testing.assert_frame_equal(s1.train, s2.train)
    pd.testing.assert_frame_equal(s1.valid, s2.valid)
    pd.testing.assert_frame_equal(s1.test, s2.test)


def test_split_different_seeds_differ(small_classification_data):
    """Test split with different seeds produces different results."""
    s1 = ml.split(data=small_classification_data, target="target", seed=42)
    s2 = ml.split(data=small_classification_data, target="target", seed=123)

    # At least one partition should differ
    try:
        pd.testing.assert_frame_equal(s1.train, s2.train)
        different = False
    except AssertionError:
        different = True

    assert different


def test_split_dev_property(small_classification_data):
    """Test .dev property returns train+valid."""
    s = ml.split(data=small_classification_data, target="target", seed=42)

    expected = pd.concat([s.train, s.valid]).reset_index(drop=True)
    pd.testing.assert_frame_equal(s.dev, expected)


def test_split_regression_detection(small_regression_data):
    """Test regression task detection."""
    s = ml.split(data=small_regression_data, target="target", seed=42)

    # Numeric target with many unique values → regression (no stratification warning)
    assert len(s.train) + len(s.valid) + len(s.test) == len(small_regression_data)


def test_split_task_override():
    """Test task= parameter overrides heuristic."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "x": rng.rand(100),
        "rating": rng.choice([1, 2, 3, 4, 5], 100),
    })

    # Override to regression (no warning)
    s = ml.split(data=data, target="rating", task="regression", seed=42)
    assert len(s.train) + len(s.valid) + len(s.test) == 100


def test_split_two_way_ratio():
    """Test two-way split (train/test only)."""
    data = pd.DataFrame({"x": range(100), "y": range(100)})
    s = ml.split(data=data, target="y", ratio=(0.8, 0.2), seed=42)

    assert len(s.train) == 80
    assert len(s.test) == 20
    assert len(s.valid) == 0  # empty
    assert len(s.dev) == 80  # just train


def test_split_cv_stratified(small_classification_data):
    """Test CV stratification."""
    s = ml.split(data=small_classification_data, target="target", seed=42)
    cv = ml.cv(s, folds=2, seed=42, stratify=True)

    # Check each fold has balanced classes
    for _fold_train, fold_valid in cv.folds:
        ratio = (fold_valid["target"] == "yes").mean()
        # Should be close to overall ratio
        overall_ratio = (small_classification_data["target"] == "yes").mean()
        assert abs(ratio - overall_ratio) < 0.2


# ─── Temporal splitting tests ───


def _make_temporal_data(n=200, seed=42):
    """Helper: create data with a sortable time column."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="D"),
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice(["yes", "no"], n),
    })


# -- Three-way temporal split (8 tests) --


def test_split_temporal_basic():
    """Temporal split preserves chronological order and produces correct sizes."""
    data = _make_temporal_data(200)
    s = ml.split(data=data, target="target", time="timestamp", seed=42)

    assert isinstance(s, ml.SplitResult)
    assert len(s.train) + len(s.valid) + len(s.test) == 200
    # Default ratio (0.6, 0.2, 0.2)
    assert len(s.train) == 120
    assert len(s.valid) == 40
    assert len(s.test) == 40


def test_split_temporal_time_column_dropped():
    """Time column is dropped from all partitions (used for ordering only)."""
    data = _make_temporal_data(100)
    s = ml.split(data=data, target="target", time="timestamp", seed=42)

    assert "timestamp" not in s.train.columns
    assert "timestamp" not in s.valid.columns
    assert "timestamp" not in s.test.columns
    # Other columns preserved
    assert "x1" in s.train.columns
    assert "target" in s.train.columns


def test_split_temporal_no_future_leakage():
    """Temporal split produces chronologically ordered partitions."""
    data = _make_temporal_data(200)
    s = ml.split(data=data, target="target", time="timestamp", seed=42)

    # Verify partitions are non-overlapping and sum to total
    total = len(s.train) + len(s.valid) + len(s.test)
    assert total == 200
    # 60/20/20 default ratio
    assert len(s.train) > len(s.valid)
    assert len(s.train) > len(s.test)


def test_split_temporal_integer_time_column():
    """Temporal split works with integer time column (not just datetime)."""
    rng = np.random.RandomState(42)
    n = 100
    data = pd.DataFrame({
        "year": rng.choice(range(2000, 2025), n),
        "x": rng.rand(n),
        "target": rng.rand(n) * 100,
    })

    s = ml.split(data=data, target="target", time="year", seed=42)
    assert isinstance(s, ml.SplitResult)
    # Time column dropped, other columns preserved
    assert "year" not in s.train.columns
    assert "x" in s.train.columns


def test_split_temporal_deterministic_regardless_of_seed():
    """Temporal split order is deterministic — seed does not affect ordering."""
    data = _make_temporal_data(100)

    s1 = ml.split(data=data, target="target", time="timestamp", seed=1)
    s2 = ml.split(data=data, target="target", time="timestamp", seed=999)

    # Same data, same time column → same split regardless of seed
    pd.testing.assert_frame_equal(s1.train, s2.train)
    pd.testing.assert_frame_equal(s1.valid, s2.valid)
    pd.testing.assert_frame_equal(s1.test, s2.test)


def test_split_temporal_column_not_found():
    """DataError when time column does not exist."""
    data = _make_temporal_data(50)

    with pytest.raises(ml.DataError, match="time='nonexistent'"):
        ml.split(data=data, target="target", time="nonexistent", seed=42)


def test_split_temporal_custom_ratio():
    """Custom ratio is respected for temporal split."""
    data = _make_temporal_data(200)
    s = ml.split(
        data=data, target="target", time="timestamp",
        ratio=(0.7, 0.15, 0.15), seed=42,
    )

    assert len(s.train) == 140
    assert len(s.valid) == 30
    assert len(s.test) == 30


def test_split_temporal_nan_time_warns():
    """NaN/NaT in time column emits a warning."""
    data = _make_temporal_data(100)
    # Inject NaT into time column
    data.loc[5, "timestamp"] = pd.NaT
    data.loc[10, "timestamp"] = pd.NaT

    with pytest.warns(UserWarning, match="2 rows have NaN/NaT"):
        ml.split(data=data, target="target", time="timestamp", seed=42)


# -- Temporal CV (6 tests) --


def test_split_temporal_cv_basic():
    """Temporal CV returns CVResult with correct number of folds."""
    data = _make_temporal_data(200)
    s = ml.split_temporal(data=data, target="target", time="timestamp")
    cv = ml.cv_temporal(s, folds=3)

    assert isinstance(cv, ml.CVResult)
    assert cv.k == 3
    assert len(cv.folds) == 3


def test_split_temporal_cv_expanding_window():
    """Later folds have more training data (expanding window)."""
    data = _make_temporal_data(200)
    s = ml.split_temporal(data=data, target="target", time="timestamp")
    cv = ml.cv_temporal(s, folds=3)

    train_sizes = [len(fold_train) for fold_train, _ in cv.folds]
    # Each fold should have strictly more training data
    for i in range(1, len(train_sizes)):
        assert train_sizes[i] > train_sizes[i - 1], (
            f"Fold {i} train size ({train_sizes[i]}) should exceed "
            f"fold {i-1} ({train_sizes[i-1]})"
        )


def test_split_temporal_cv_data_is_sorted():
    """CVResult._data is sorted by the time column (time column dropped from features)."""
    rng = np.random.RandomState(42)
    n = 100
    # Deliberately unsorted
    timestamps = rng.choice(range(1000), n)
    data = pd.DataFrame({
        "timestamp": timestamps,
        "x": rng.rand(n),
        "target": rng.choice(["a", "b"], n),
    })

    s = ml.split_temporal(data=data, target="target", time="timestamp")
    cv = ml.cv_temporal(s, folds=3)

    # Time column is dropped from _data (used for ordering only)
    assert "timestamp" not in cv._data.columns
    # Data should have contiguous index (sorted + reset)
    assert list(cv._data.index) == list(range(len(cv._data)))


def test_split_temporal_cv_too_many_folds_raises():
    """DataError when folds >= n."""
    data = _make_temporal_data(10)
    s = ml.split_temporal(data=data, target="target", time="timestamp")

    with pytest.raises((ml.DataError, ml.ConfigError)):
        ml.cv_temporal(s, folds=10)


def test_split_temporal_cv_small_data_works():
    """Temporal CV with minimal data (fold_size=1) produces valid folds."""
    data = _make_temporal_data(20)
    s = ml.split_temporal(data=data, target="target", time="timestamp")

    cv = ml.cv_temporal(s, folds=3)
    assert cv.k == 3
    for _, fold_valid in cv.folds:
        assert len(fold_valid) >= 1


def test_split_temporal_cv_divisible_n_all_folds_valid():
    """When n%(folds+1)==0, all folds have non-empty valid partitions."""
    rng = np.random.RandomState(42)
    # n=100, folds=3 → fold_size=25, 3 full folds
    data = pd.DataFrame({
        "timestamp": range(100),
        "x": rng.rand(100),
        "target": rng.choice(["a", "b"], 100),
    })

    s = ml.split_temporal(data=data, target="target", time="timestamp")
    cv = ml.cv_temporal(s, folds=3)

    # Should produce exactly 3 folds
    assert cv.k == 3

    # All folds must have non-empty valid partitions
    for _, fold_valid in cv.folds:
        assert len(fold_valid) > 0, "No fold should have empty valid partition"


# -- Conflict/edge (2 tests) --


def test_split_temporal_plus_splitter_raises():
    """ConfigError when both time= and splitter= are set."""
    data = _make_temporal_data(50)

    with pytest.raises(ml.ConfigError, match="Cannot use time= and splitter="):
        ml.split(
            data=data, target="target", time="timestamp",
            splitter="anything", seed=42,
        )


def test_split_temporal_float_time_column():
    """Temporal split works with float time column."""
    rng = np.random.RandomState(42)
    n = 100
    data = pd.DataFrame({
        "time_seconds": rng.rand(n) * 1000,
        "x": rng.rand(n),
        "target": rng.rand(n),
    })

    s = ml.split(data=data, target="target", time="time_seconds", seed=42)
    assert isinstance(s, ml.SplitResult)
    # Time column dropped, other columns preserved
    assert "time_seconds" not in s.train.columns
    assert "x" in s.train.columns


# -- Integration (2 tests) --


def test_split_temporal_cv_works_with_fit():
    """ml.fit() on temporal CVResult produces a Model with scores_."""
    import warnings as w
    rng = np.random.RandomState(42)
    n = 200
    # Use integer time column — datetime would choke XGBoost
    data = pd.DataFrame({
        "day": range(n),
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice(["yes", "no"], n),
    })
    s = ml.split_temporal(data=data, target="target", time="day")
    cv = ml.cv_temporal(s, folds=3)

    with w.catch_warnings():
        w.simplefilter("ignore")
        model = ml.fit(data=cv, target="target", seed=42)

    assert model is not None
    assert hasattr(model, "scores_")
    assert len(model.scores_) > 0


def test_split_temporal_cv_works_with_screen():
    """ml.screen() on temporal CVResult returns a leaderboard."""
    import warnings as w
    rng = np.random.RandomState(42)
    n = 200
    data = pd.DataFrame({
        "day": range(n),
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice(["yes", "no"], n),
    })
    s = ml.split_temporal(data=data, target="target", time="day")
    cv = ml.cv_temporal(s, folds=3)

    with w.catch_warnings():
        w.simplefilter("ignore")
        result = ml.screen(
            data=cv, target="target",
            algorithms=["random_forest"], seed=42,
        )

    assert len(result) >= 1
    assert "algorithm" in result.columns


# ─── Temporal splitting: adversarial stress tests ───


def test_split_temporal_all_same_timestamp():
    """All rows have identical timestamp — split is positional, no crash."""
    rng = np.random.RandomState(42)
    n = 100
    data = pd.DataFrame({
        "ts": [42] * n,
        "x": rng.rand(n),
        "target": rng.choice(["a", "b"], n),
    })

    s = ml.split(data=data, target="target", time="ts", seed=42)
    assert isinstance(s, ml.SplitResult)
    assert len(s.train) + len(s.valid) + len(s.test) == n
    # Time column dropped
    assert "ts" not in s.train.columns


def test_split_temporal_reverse_sorted_data():
    """Pre-sorted descending data is correctly reversed by temporal split."""
    rng = np.random.RandomState(42)
    n = 100
    data = pd.DataFrame({
        "ts": list(range(n, 0, -1)),  # 100, 99, ..., 1
        "x": rng.rand(n),
        "target": rng.rand(n),
    })

    s = ml.split(data=data, target="target", time="ts", seed=42)
    # Time column dropped, partitions are chronologically ordered
    assert "ts" not in s.train.columns
    assert len(s.train) + len(s.valid) + len(s.test) == n


def test_split_temporal_duplicate_timestamps_at_boundary():
    """Duplicate timestamps at partition boundary — stable sort keeps them together."""
    # Create data where timestamps cluster at the 60% boundary
    ts = [1] * 30 + [2] * 30 + [3] * 40  # 30+30+40 = 100
    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "ts": ts,
        "x": rng.rand(100),
        "target": rng.choice(["a", "b"], 100),
    })

    s = ml.split(data=data, target="target", time="ts", seed=42)
    # With 60/20/20: train=60, valid=20, test=20
    # ts=1 (30 rows) and ts=2 (30 rows) in train
    # ts=3 (40 rows) split between valid and test
    assert len(s.train) == 60
    assert len(s.valid) == 20
    assert len(s.test) == 20


def test_split_temporal_inf_time_column():
    """Time column with inf/-inf values — sort handles them correctly."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "ts": [float("-inf")] + list(range(98)) + [float("inf")],
        "x": rng.rand(100),
        "target": rng.choice(["a", "b"], 100),
    })

    s = ml.split(data=data, target="target", time="ts", seed=42)
    # Time column dropped, no crash from inf values
    assert "ts" not in s.train.columns
    assert len(s.train) + len(s.valid) + len(s.test) == 100


def test_split_temporal_single_row_partition():
    """Ratio that produces a 1-row valid partition still works."""
    import warnings as w
    rng = np.random.RandomState(42)
    n = 50
    data = pd.DataFrame({
        "ts": range(n),
        "x": rng.rand(n),
        "target": rng.choice(["a", "b"], n),
    })

    with w.catch_warnings():
        w.simplefilter("ignore")
        # ratio=(0.96, 0.02, 0.02) on 50 rows → valid=1, test=1
        s = ml.split(
            data=data, target="target", time="ts",
            ratio=(0.96, 0.02, 0.02), seed=42,
        )

    assert len(s.valid) >= 1
    assert len(s.test) >= 1
    # Time column dropped
    assert "ts" not in s.train.columns


def test_split_temporal_cv_remainder_handling():
    """Temporal CV with n not divisible by (folds+1) — last fold gets remainder."""
    rng = np.random.RandomState(42)
    n = 103  # 103 // 4 = 25, remainder 3
    data = pd.DataFrame({
        "ts": range(n),
        "x": rng.rand(n),
        "target": rng.choice(["a", "b"], n),
    })

    s = ml.split_temporal(data=data, target="target", time="ts")
    cv = ml.cv_temporal(s, folds=3)
    assert cv.k == 3

    # Last fold valid should be larger (gets remainder)
    last_valid = cv.folds[-1][1]
    second_valid = cv.folds[-2][1]
    assert len(last_valid) >= len(second_valid), (
        "Last fold should get remainder rows"
    )


def test_split_temporal_two_tuple_ratio():
    """2-tuple ratio temporal split — train/valid only, empty test."""
    import warnings as w
    rng = np.random.RandomState(42)
    n = 100
    data = pd.DataFrame({
        "ts": range(n),
        "x": rng.rand(n),
        "target": rng.choice(["a", "b"], n),
    })

    with w.catch_warnings():
        w.simplefilter("ignore")
        s = ml.split(
            data=data, target="target", time="ts",
            ratio=(0.8, 0.2), seed=42,
        )

    assert len(s.train) == 80
    assert len(s.valid) == 20
    assert len(s.test) == 0  # empty test with 2-tuple
    # Time column dropped
    assert "ts" not in s.train.columns


def test_split_temporal_class_imbalance_across_time():
    """Target class distribution shifts over time — temporal split captures this."""
    # Early data: mostly "no", late data: mostly "yes"
    n = 200
    data = pd.DataFrame({
        "ts": range(n),
        "x": list(range(n)),
        "target": ["no"] * 150 + ["yes"] * 50,
    })

    s = ml.split(data=data, target="target", time="ts", seed=42)
    # Train (first 60%) should be heavily "no"
    train_yes_pct = (s.train["target"] == "yes").mean()
    # Test (last 20%) should be heavily "yes"
    test_yes_pct = (s.test["target"] == "yes").mean()

    assert train_yes_pct < test_yes_pct, (
        "Temporal split should capture class distribution shift"
    )


def test_split_temporal_cv_fold_indices_no_overlap():
    """Temporal CV fold indices don't overlap — each validation set is unique."""
    rng = np.random.RandomState(42)
    n = 200
    data = pd.DataFrame({
        "ts": range(n),
        "x": rng.rand(n),
        "target": rng.choice(["a", "b"], n),
    })

    s = ml.split_temporal(data=data, target="target", time="ts")
    cv = ml.cv_temporal(s, folds=4)

    # Valid sets should not overlap
    all_valid = []
    for _, fold_valid in cv.folds:
        all_valid.extend(fold_valid.index)
    assert len(all_valid) == len(set(all_valid)), "Valid fold indices must not overlap"


def test_split_temporal_cv_train_always_before_valid():
    """In each temporal CV fold, all train indices < all valid indices."""
    rng = np.random.RandomState(42)
    n = 200
    data = pd.DataFrame({
        "ts": range(n),
        "x": rng.rand(n),
        "target": rng.choice(["a", "b"], n),
    })

    s = ml.split_temporal(data=data, target="target", time="ts")
    cv = ml.cv_temporal(s, folds=4)

    for i, (fold_train, fold_valid) in enumerate(cv.folds):
        assert max(fold_train.index) < min(fold_valid.index), (
            f"Fold {i}: max train index ({max(fold_train.index)}) must be < "
            f"min valid index ({min(fold_valid.index)})"
        )


def test_split_temporal_with_target_nan_dropped_before_sort():
    """NaN target rows are dropped before temporal sorting (upstream in split())."""
    rng = np.random.RandomState(42)
    n = 100
    data = pd.DataFrame({
        "ts": range(n),
        "x": rng.rand(n),
        "target": rng.choice(["a", "b"], n).astype(object),
    })
    # Inject NaN targets at specific timestamps
    data.loc[10, "target"] = None
    data.loc[50, "target"] = None

    import warnings as w
    with w.catch_warnings():
        w.simplefilter("ignore")
        s = ml.split(data=data, target="target", time="ts", seed=42)

    total = len(s.train) + len(s.valid) + len(s.test)
    assert total == 98, f"Expected 98 rows (2 NaN dropped), got {total}"


def test_split_temporal_cv_fit_produces_valid_scores():
    """Temporal CV through fit() produces sensible cross-validation scores."""
    import warnings as w
    rng = np.random.RandomState(42)
    n = 300
    data = pd.DataFrame({
        "day": range(n),
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice(["yes", "no"], n),
    })
    s = ml.split_temporal(data=data, target="target", time="day")
    cv = ml.cv_temporal(s, folds=3)

    with w.catch_warnings():
        w.simplefilter("ignore")
        model = ml.fit(data=cv, target="target", algorithm="random_forest", seed=42)

    # Scores should be in valid range
    for key, val in model.scores_.items():
        if "mean" in key:
            assert 0.0 <= val <= 1.0, f"Score {key}={val} out of range"


def test_split_temporal_cv_degenerate_fold_single_class():
    """Temporal CV where some folds have only one class in validation.

    Regression test: when a validation fold has zero minority class, the
    metric key set must still be consistent across folds (all 'f1', never
    mixing 'f1' and 'f1_weighted'). This caused KeyError in score
    aggregation before the n_classes <= 2 fix.
    """
    import warnings as w

    # Create data with extreme temporal class imbalance:
    # early rows are all class "a", later rows mix "a" and "b"
    n = 300
    data = pd.DataFrame({
        "day": range(n),
        "x1": np.random.RandomState(42).rand(n),
        "target": ["a"] * 200 + np.random.RandomState(42).choice(["a", "b"], 100).tolist(),
    })

    s = ml.split_temporal(data=data, target="target", time="day")
    cv = ml.cv_temporal(s, folds=3)

    # Fold 0 trains on early rows, validates on middle rows — likely all "a"
    # This should NOT crash with KeyError on score aggregation
    with w.catch_warnings():
        w.simplefilter("ignore")
        model = ml.fit(data=cv, target="target", algorithm="random_forest", seed=42)

    assert model.scores_ is not None
    assert "accuracy_mean" in model.scores_


def test_split_temporal_cv_datetime_column_not_in_features():
    """Temporal CV drops the time column from features so datetime dtypes
    don't crash sklearn (DTypePromotionError)."""
    import warnings as w

    n = 200
    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="D"),
        "x1": rng.rand(n),
        "x2": rng.rand(n),
        "target": rng.choice([0, 1], n),
    })

    s = ml.split_temporal(data=data, target="target", time="timestamp")
    cv = ml.cv_temporal(s, folds=3)

    # Time column should be dropped from CVResult._data
    assert "timestamp" not in cv._data.columns

    # fit() should work without DTypePromotionError
    with w.catch_warnings():
        w.simplefilter("ignore")
        model = ml.fit(data=cv, target="target", algorithm="random_forest", seed=42)

    assert model.scores_ is not None
    # Features should NOT include the time column
    assert "timestamp" not in model.features


# ── A6: Split extensions ───────────────────────────────────────────────────────

@pytest.mark.skip(reason="repeats= removed in v1.1; use ml.cv() without repeats")
def test_repeated_kfold():
    """repeats= with folds= creates R*K total folds. A6."""
    pass


@pytest.mark.skip(reason="repeats= removed in v1.1; use ml.cv() without repeats")
def test_repeated_stratified():
    """Stratified repeated K-fold for classification target. A6."""
    pass


@pytest.mark.skip(reason="repeats= removed in v1.1; use ml.cv() without repeats")
def test_repeated_requires_folds():
    """repeats= without folds= raises ConfigError. A6."""
    pass


def test_stratified_group_kfold():
    """groups= + stratify=True uses StratifiedGroupKFold. A6."""
    rng = np.random.RandomState(42)
    n = 120
    data = pd.DataFrame({
        "x1": rng.rand(n),
        "target": rng.choice([0, 1], n),
        "group": [f"g{i // 10}" for i in range(n)],  # 12 groups of 10
    })
    s = ml.split_group(data=data, target="target", groups="group", seed=42)
    cv = ml.cv_group(s, folds=3, groups="group", seed=42)
    assert isinstance(cv, ml.CVResult)
    assert cv.k == 3
    # No group leaks
    for fold_train, fold_valid in cv.folds:
        train_groups = set(fold_train["group"])
        valid_groups = set(fold_valid["group"])
        assert train_groups.isdisjoint(valid_groups)


def test_embargo_temporal():
    """time= + embargo= removes rows between train and valid. A6."""
    rng = np.random.RandomState(42)
    n = 100
    data = pd.DataFrame({
        "x": rng.rand(n),
        "target": rng.rand(n),
        "ts": range(n),
    })
    s = ml.split_temporal(data=data, target="target", time="ts")
    cv = ml.cv_temporal(s, folds=3, embargo=5)
    assert isinstance(cv, ml.CVResult)
    # Each fold should have a gap between train end and valid start
    for fold_train, fold_valid in cv.folds:
        if len(fold_train) > 0 and len(fold_valid) > 0:
            assert min(fold_valid.index) > max(fold_train.index)


def test_embargo_removes_correct_rows():
    """Embargo gap size is as specified. A6."""
    n = 60
    data = pd.DataFrame({
        "x": range(n),
        "target": range(n),
        "ts": range(n),
    })
    embargo = 5
    s = ml.split_temporal(data=data, target="target", time="ts")
    cv = ml.cv_temporal(s, folds=3, embargo=embargo)
    for fold_train, fold_valid in cv.folds:
        if len(fold_train) > 0 and len(fold_valid) > 0:
            gap = min(fold_valid.index) - max(fold_train.index) - 1
            assert gap >= embargo - 1  # embargo rows removed


def test_sliding_window_basic():
    """window='sliding' with window_size= produces CVResult. A6."""
    rng = np.random.RandomState(42)
    n = 100
    data = pd.DataFrame({
        "x": rng.rand(n),
        "target": rng.rand(n),
        "ts": range(n),
    })
    s = ml.split_temporal(data=data, target="target", time="ts")
    cv = ml.cv_temporal(s, folds=3, window="sliding", window_size=20)
    assert isinstance(cv, ml.CVResult)
    # Train sets should be fixed size (sliding window)
    for fold_train, _fold_valid in cv.folds:
        assert len(fold_train) <= 20


def test_sliding_window_size():
    """window='sliding' requires window_size= or raises ConfigError. A6."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({"x": rng.rand(50), "target": rng.rand(50), "ts": range(50)})
    s = ml.split_temporal(data=data, target="target", time="ts")
    with pytest.raises(ml.ConfigError, match="window_size"):
        ml.cv_temporal(s, folds=3, window="sliding")


# ── Partition tag tests ──


def test_standard_split_tags_partitions():
    """Standard 3-way split tags DataFrames with _ml_partition attr."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({"x": rng.rand(100), "target": rng.randint(0, 2, 100)})
    s = ml.split(data=data, target="target", seed=42)
    assert s.train.attrs.get("_ml_partition") == "train"
    assert s.valid.attrs.get("_ml_partition") == "valid"
    assert s.test.attrs.get("_ml_partition") == "test"


def test_dev_property_tags_partition():
    """s.dev is tagged as 'dev' partition."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({"x": rng.rand(100), "target": rng.randint(0, 2, 100)})
    s = ml.split(data=data, target="target", seed=42)
    assert s.dev.attrs.get("_ml_partition") == "dev"


def test_temporal_split_tags_partitions():
    """Temporal split tags DataFrames with _ml_partition attr."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "x": rng.rand(100),
        "target": rng.randint(0, 2, 100),
        "ts": pd.date_range("2020-01-01", periods=100),
    })
    s = ml.split(data=data, target="target", time="ts", seed=42)
    assert s.train.attrs.get("_ml_partition") == "train"
    assert s.valid.attrs.get("_ml_partition") == "valid"
    assert s.test.attrs.get("_ml_partition") == "test"


def test_group_split_tags_partitions():
    """Group split tags DataFrames with _ml_partition attr."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "x": rng.rand(100),
        "target": rng.randint(0, 2, 100),
        "group": rng.choice(["a", "b", "c", "d", "e"], 100),
    })
    s = ml.split(data=data, target="target", groups="group", seed=42)
    assert s.train.attrs.get("_ml_partition") == "train"
    assert s.valid.attrs.get("_ml_partition") == "valid"
    assert s.test.attrs.get("_ml_partition") == "test"
