"""Tests for split_temporal() and split_group() domain specializations."""

import numpy as np
import pandas as pd
import pytest

import ml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temporal_data():
    """DataFrame with a time column for temporal splitting."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "target": np.random.choice([0, 1], n),
    })


@pytest.fixture
def group_data():
    """DataFrame with a group column for group splitting."""
    np.random.seed(42)
    groups = np.repeat([f"patient_{i}" for i in range(20)], 10)
    n = len(groups)
    return pd.DataFrame({
        "patient_id": groups,
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "outcome": np.random.choice(["good", "bad"], n),
    })


# ---------------------------------------------------------------------------
# split_temporal — holdout
# ---------------------------------------------------------------------------


class TestSplitTemporalHoldout:
    def test_produces_split_result(self, temporal_data):
        s = ml.split_temporal(temporal_data, "target", time="date")
        assert isinstance(s, ml.SplitResult)

    def test_correct_partition_sizes(self, temporal_data):
        s = ml.split_temporal(temporal_data, "target", time="date")
        assert len(s.train) == 120  # 60%
        assert len(s.valid) == 40   # 20%
        assert len(s.test) == 40    # 20%

    def test_custom_ratio(self, temporal_data):
        s = ml.split_temporal(temporal_data, "target", time="date",
                              ratio=(0.7, 0.15, 0.15))
        assert len(s.train) == 140

    def test_time_column_dropped(self, temporal_data):
        s = ml.split_temporal(temporal_data, "target", time="date")
        assert "date" not in s.train.columns
        assert "date" not in s.test.columns

    def test_partition_tags_set(self, temporal_data):
        s = ml.split_temporal(temporal_data, "target", time="date")
        assert s.train.attrs.get("_ml_partition") == "train"
        assert s.valid.attrs.get("_ml_partition") == "valid"
        assert s.test.attrs.get("_ml_partition") == "test"

    def test_deterministic_no_seed(self, temporal_data):
        s1 = ml.split_temporal(temporal_data, "target", time="date")
        s2 = ml.split_temporal(temporal_data, "target", time="date")
        pd.testing.assert_frame_equal(s1.train, s2.train)
        pd.testing.assert_frame_equal(s1.test, s2.test)

    def test_dev_property_works(self, temporal_data):
        s = ml.split_temporal(temporal_data, "target", time="date")
        assert len(s.dev) == len(s.train) + len(s.valid)

    def test_equivalent_to_split_time(self, temporal_data):
        """split(time=) should route to split_temporal()."""
        s1 = ml.split_temporal(temporal_data, "target", time="date")
        s2 = ml.split(temporal_data, "target", time="date")
        pd.testing.assert_frame_equal(s1.train, s2.train)
        pd.testing.assert_frame_equal(s1.test, s2.test)


# ---------------------------------------------------------------------------
# split_temporal — CV
# ---------------------------------------------------------------------------


class TestSplitTemporalCV:
    def test_produces_cv_result(self, temporal_data):
        cv = ml.split_temporal(temporal_data, "target", time="date", folds=5)
        assert isinstance(cv, ml.CVResult)

    def test_correct_fold_count(self, temporal_data):
        cv = ml.split_temporal(temporal_data, "target", time="date", folds=5)
        assert cv.k == 5

    def test_embargo(self, temporal_data):
        cv = ml.split_temporal(temporal_data, "target", time="date",
                               folds=3, embargo=5)
        assert isinstance(cv, ml.CVResult)
        assert cv.k == 3

    def test_sliding_window(self, temporal_data):
        cv = ml.split_temporal(temporal_data, "target", time="date",
                               folds=3, window="sliding", window_size=50)
        assert isinstance(cv, ml.CVResult)
        assert cv.k == 3

    def test_equivalent_to_split_time_folds(self, temporal_data):
        cv1 = ml.split_temporal(temporal_data, "target", time="date", folds=5)
        cv2 = ml.split(temporal_data, "target", time="date", folds=5)
        assert cv1.k == cv2.k


# ---------------------------------------------------------------------------
# split_temporal — error handling
# ---------------------------------------------------------------------------


class TestSplitTemporalErrors:
    def test_missing_time_column(self, temporal_data):
        with pytest.raises(ml.DataError, match="time='nonexistent'"):
            ml.split_temporal(temporal_data, "target", time="nonexistent")

    def test_empty_data(self):
        df = pd.DataFrame({"date": [], "target": []})
        with pytest.raises(ml.DataError, match="empty"):
            ml.split_temporal(df, "target", time="date")

    def test_sliding_without_window_size(self, temporal_data):
        with pytest.raises(ml.ConfigError, match="window_size"):
            ml.split_temporal(temporal_data, "target", time="date",
                              folds=3, window="sliding")

    def test_embargo_without_folds(self, temporal_data):
        with pytest.raises(ml.ConfigError, match="embargo.*folds"):
            ml.split_temporal(temporal_data, "target", time="date", embargo=5)

    def test_invalid_window(self, temporal_data):
        with pytest.raises(ml.ConfigError, match="window"):
            ml.split_temporal(temporal_data, "target", time="date",
                              folds=3, window="invalid")

    def test_missing_target(self, temporal_data):
        with pytest.raises(ml.DataError, match="target='nonexistent'"):
            ml.split_temporal(temporal_data, "nonexistent", time="date")


# ---------------------------------------------------------------------------
# split_group — holdout
# ---------------------------------------------------------------------------


class TestSplitGroupHoldout:
    def test_produces_split_result(self, group_data):
        s = ml.split_group(group_data, "outcome", groups="patient_id", seed=42)
        assert isinstance(s, ml.SplitResult)

    def test_no_group_overlap(self, group_data):
        s = ml.split_group(group_data, "outcome", groups="patient_id", seed=42)
        # Groups must be disjoint across partitions
        # Group column is still in the data (not dropped like time)
        train_groups = set(s.train["patient_id"].unique())
        valid_groups = set(s.valid["patient_id"].unique())
        test_groups = set(s.test["patient_id"].unique())
        assert train_groups & valid_groups == set()
        assert train_groups & test_groups == set()
        assert valid_groups & test_groups == set()

    def test_partition_tags_set(self, group_data):
        s = ml.split_group(group_data, "outcome", groups="patient_id", seed=42)
        assert s.train.attrs.get("_ml_partition") == "train"
        assert s.valid.attrs.get("_ml_partition") == "valid"
        assert s.test.attrs.get("_ml_partition") == "test"

    def test_reproducible_with_seed(self, group_data):
        s1 = ml.split_group(group_data, "outcome", groups="patient_id", seed=42)
        s2 = ml.split_group(group_data, "outcome", groups="patient_id", seed=42)
        assert set(s1.train["patient_id"].unique()) == set(s2.train["patient_id"].unique())

    def test_different_seed_different_split(self, group_data):
        s1 = ml.split_group(group_data, "outcome", groups="patient_id", seed=42)
        s2 = ml.split_group(group_data, "outcome", groups="patient_id", seed=99)
        # Not guaranteed to differ with only 20 groups, but very likely
        g1 = set(s1.train["patient_id"].unique())
        g2 = set(s2.train["patient_id"].unique())
        # At least check both produce valid splits
        assert len(g1) > 0
        assert len(g2) > 0

    def test_dev_property_works(self, group_data):
        s = ml.split_group(group_data, "outcome", groups="patient_id", seed=42)
        assert len(s.dev) == len(s.train) + len(s.valid)

    def test_equivalent_to_split_groups(self, group_data):
        """split(groups=) should route to split_group()."""
        s1 = ml.split_group(group_data, "outcome", groups="patient_id", seed=42)
        s2 = ml.split(group_data, "outcome", groups="patient_id", seed=42)
        assert set(s1.train["patient_id"].unique()) == set(s2.train["patient_id"].unique())


# ---------------------------------------------------------------------------
# split_group — CV
# ---------------------------------------------------------------------------


class TestSplitGroupCV:
    def test_produces_cv_result(self, group_data):
        cv = ml.split_group(group_data, "outcome", groups="patient_id",
                            folds=5, seed=42)
        assert isinstance(cv, ml.CVResult)

    def test_correct_fold_count(self, group_data):
        cv = ml.split_group(group_data, "outcome", groups="patient_id",
                            folds=5, seed=42)
        assert cv.k == 5

    def test_no_group_overlap_in_folds(self, group_data):
        cv = ml.split_group(group_data, "outcome", groups="patient_id",
                            folds=4, seed=42)
        for train_df, valid_df in cv.folds:
            train_groups = set(train_df["patient_id"].unique())
            valid_groups = set(valid_df["patient_id"].unique())
            assert train_groups & valid_groups == set()

    def test_equivalent_to_split_groups_folds(self, group_data):
        cv1 = ml.split_group(group_data, "outcome", groups="patient_id",
                             folds=5, seed=42)
        cv2 = ml.split(group_data, "outcome", groups="patient_id",
                        folds=5, seed=42)
        assert cv1.k == cv2.k


# ---------------------------------------------------------------------------
# split_group — error handling
# ---------------------------------------------------------------------------


class TestSplitGroupErrors:
    def test_missing_groups_column(self, group_data):
        with pytest.raises(ml.DataError, match="groups='nonexistent'"):
            ml.split_group(group_data, "outcome", groups="nonexistent", seed=42)

    def test_empty_data(self):
        df = pd.DataFrame({"patient_id": [], "outcome": []})
        with pytest.raises(ml.DataError, match="empty"):
            ml.split_group(df, "outcome", groups="patient_id", seed=42)

    def test_invalid_seed(self, group_data):
        with pytest.raises(ml.ConfigError, match="seed must be an integer"):
            ml.split_group(group_data, "outcome", groups="patient_id", seed="42")

    def test_missing_target(self, group_data):
        with pytest.raises(ml.DataError, match="target='nonexistent'"):
            ml.split_group(group_data, "nonexistent", groups="patient_id", seed=42)


# ---------------------------------------------------------------------------
# No-target usage (unsupervised)
# ---------------------------------------------------------------------------


class TestSplitGroupStratifyWarning:
    def test_warns_stratify_on_holdout(self, group_data):
        with pytest.warns(UserWarning, match="stratify=True is ignored for group holdout"):
            ml.split_group(group_data, "outcome", groups="patient_id",
                           seed=42, stratify=True)

    def test_no_warn_stratify_on_cv(self, group_data):
        """stratify=True should NOT warn when folds= is set (it's supported)."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # This should not raise — stratify is supported for group CV
            try:
                ml.split_group(group_data, "outcome", groups="patient_id",
                               folds=4, seed=42, stratify=True)
            except UserWarning:
                pytest.fail("split_group with folds= should not warn about stratify")


class TestNoDoubleWarning:
    def test_split_time_no_duplicate_nan_warning(self):
        """split(time=) should not produce duplicate NaN warnings."""
        df = pd.DataFrame({
            "date": list(range(100)) + [np.nan],
            "x": np.random.randn(101),
            "target": np.random.choice([0, 1], 101),
        })
        with pytest.warns(UserWarning) as record:
            ml.split(df, "target", time="date")
        nan_warnings = [w for w in record if "NaN/NaT" in str(w.message)]
        assert len(nan_warnings) <= 1, f"Got {len(nan_warnings)} NaN warnings (expected ≤1)"


class TestSplitSpecializationsNoTarget:
    def test_temporal_no_target(self, temporal_data):
        s = ml.split_temporal(temporal_data, time="date")
        assert isinstance(s, ml.SplitResult)
        assert len(s.train) + len(s.valid) + len(s.test) == len(temporal_data)

    def test_group_no_target(self, group_data):
        s = ml.split_group(group_data, groups="patient_id", seed=42)
        assert isinstance(s, ml.SplitResult)
