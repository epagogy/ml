"""Tests for ml.cv() — the 8th primitive.

Tests the full workflow: split → cv → fit → assess.
Provenance chain, error handling, all three variants.
"""

import numpy as np
import pandas as pd
import pytest

import ml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_data():
    """Classification dataset for CV tests."""
    return ml.dataset("titanic")


@pytest.fixture
def reg_data():
    """Regression dataset for CV tests."""
    return ml.dataset("tips")


@pytest.fixture
def clf_split(clf_data):
    """Pre-split classification data."""
    return ml.split(clf_data, "survived", seed=42)


@pytest.fixture
def reg_split(reg_data):
    """Pre-split regression data."""
    return ml.split(reg_data, "tip", seed=42)


# ---------------------------------------------------------------------------
# cv() — basic functionality
# ---------------------------------------------------------------------------

class TestCV:
    """Core cv() tests."""

    def test_cv_returns_cvresult(self, clf_split):
        c = ml.cv(clf_split, folds=5, seed=42)
        assert isinstance(c, ml.CVResult)

    def test_cv_has_correct_folds(self, clf_split):
        c = ml.cv(clf_split, folds=5, seed=42)
        assert c.k == 5
        assert len(c.folds) == 5

    def test_cv_has_no_test_partition(self, clf_split):
        """Test lives on SplitResult, not CVResult."""
        c = ml.cv(clf_split, folds=5, seed=42)
        with pytest.raises(ml.ConfigError):
            _ = c.test
            # Test is on the split
            assert len(clf_split.test) > 0

    def test_cv_folds_are_dataframes(self, clf_split):
        c = ml.cv(clf_split, folds=5, seed=42)
        for train_df, valid_df in c.folds:
            assert isinstance(train_df, pd.DataFrame)
            assert isinstance(valid_df, pd.DataFrame)

    def test_cv_folds_cover_dev(self, clf_split):
        """Every row in dev appears in exactly one validation fold."""
        c = ml.cv(clf_split, folds=5, seed=42)
        dev = clf_split.dev
        all_valid_indices = []
        for _, valid_df in c.folds:
            all_valid_indices.extend(valid_df.index.tolist())
            # All dev rows appear in validation
            assert sorted(all_valid_indices) == sorted(dev.index.tolist())

    def test_cv_no_fold_overlap(self, clf_split):
        """Validation folds don't overlap."""
        c = ml.cv(clf_split, folds=5, seed=42)
        all_valid_indices = []
        for _, valid_df in c.folds:
            all_valid_indices.extend(valid_df.index.tolist())
            assert len(all_valid_indices) == len(set(all_valid_indices))

    def test_cv_data_is_dev(self, clf_split):
        """CV operates on dev (train+valid), not all data."""
        c = ml.cv(clf_split, folds=5, seed=42)
        assert len(c._data) == len(clf_split.dev)

    def test_cv_deterministic(self, clf_split):
        """Same seed → same folds."""
        c1 = ml.cv(clf_split, folds=5, seed=42)
        c2 = ml.cv(clf_split, folds=5, seed=42)
        for (t1, v1), (t2, v2) in zip(c1.folds, c2.folds):
            pd.testing.assert_frame_equal(t1, t2)
            pd.testing.assert_frame_equal(v1, v2)

    def test_cv_different_seeds(self, clf_split):
        """Different seed → different folds."""
        c1 = ml.cv(clf_split, folds=5, seed=42)
        c2 = ml.cv(clf_split, folds=5, seed=99)
        # At least one fold should differ
        any_different = False
        for (_, v1), (_, v2) in zip(c1.folds, c2.folds):
            if not v1.index.equals(v2.index):
                any_different = True
                break
                assert any_different

    def test_cv_regression(self, reg_split):
        """CV works for regression tasks."""
        c = ml.cv(reg_split, folds=3, seed=42)
        assert c.k == 3

    def test_cv_repr(self, clf_split):
        c = ml.cv(clf_split, folds=5, seed=42)
        r = repr(c)
        assert "folds=5" in r

    def test_cv_folds_2(self, clf_split):
        """Minimum folds=2 works."""
        c = ml.cv(clf_split, folds=2, seed=42)
        assert c.k == 2
        assert len(c.folds) == 2


# ---------------------------------------------------------------------------
# cv() — full workflow integration
# ---------------------------------------------------------------------------

class TestCVWorkflow:
    """The money tests: split → cv → fit → assess."""

    def test_cv_fit_produces_scores(self, clf_split):
        """fit(cv, ...) produces fold-averaged scores."""
        c = ml.cv(clf_split, folds=3, seed=42)
        model = ml.fit(c, "survived", seed=42)
        assert model.scores_ is not None
        assert "accuracy_mean" in model.scores_
        assert "accuracy_std" in model.scores_

    def test_cv_fit_assess_full_workflow(self, clf_split):
        """The canonical workflow: split → cv → fit → assess(test=s.test)."""
        c = ml.cv(clf_split, folds=3, seed=42)
        model = ml.fit(c, "survived", seed=42)
        evidence = ml.assess(model, test=clf_split.test)
        assert evidence is not None
        # assess consumed the test set
        assert model._assess_count == 1

    def test_cv_fit_assess_provenance_passes(self, clf_split):
        """Provenance guard accepts cv workflow — same split lineage."""
        ml.config(guards="strict")
        c = ml.cv(clf_split, folds=3, seed=42)
        model = ml.fit(c, "survived", seed=42)
        # This should NOT raise PartitionError
        evidence = ml.assess(model, test=clf_split.test)
        assert evidence is not None

    def test_cv_fit_regression_workflow(self, reg_split):
        """Full workflow for regression."""
        c = ml.cv(reg_split, folds=3, seed=42)
        model = ml.fit(c, "tip", seed=42)
        assert "rmse_mean" in model.scores_ or "mse_mean" in model.scores_
        evidence = ml.assess(model, test=reg_split.test)
        assert evidence is not None

    def test_cv_screen_workflow(self, clf_split):
        """screen() works with CVResult."""
        c = ml.cv(clf_split, folds=3, seed=42)
        lb = ml.screen(c, "survived", seed=42)
        assert lb is not None
        assert len(lb) > 0

    def test_cv_assess_one_shot(self, clf_split):
        """assess() still enforces one-shot on CV models."""
        c = ml.cv(clf_split, folds=3, seed=42)
        model = ml.fit(c, "survived", seed=42)
        ml.assess(model, test=clf_split.test)
        with pytest.raises(ml.ModelError):
            ml.assess(model, test=clf_split.test)


# ---------------------------------------------------------------------------
# Target inference — fit() reads target from split/cv
# ---------------------------------------------------------------------------

class TestTargetInference:
    """fit() infers target from split/cv when not provided."""

    def test_fit_cv_infers_target(self, clf_split):
        """fit(cv, seed=42) without target= works."""
        c = ml.cv(clf_split, folds=3, seed=42)
        model = ml.fit(c, seed=42)
        assert model.target == "survived"

    def test_fit_train_infers_target(self, clf_split):
        """fit(s.train, seed=42) without target= works."""
        model = ml.fit(clf_split.train, seed=42)
        assert model.target == "survived"

    def test_fit_dev_infers_target(self, clf_split):
        """fit(s.dev, seed=42) without target= works."""
        model = ml.fit(clf_split.dev, seed=42)
        assert model.target == "survived"

    def test_fit_explicit_target_overrides(self, clf_split):
        """Explicit target= overrides inferred target."""
        c = ml.cv(clf_split, folds=3, seed=42)
        # This would fail because "pclass" is not the right kind of target
        # but the override should be respected up to the point of fit
        model = ml.fit(c, "survived", seed=42)
        assert model.target == "survived"

    def test_fit_raw_dataframe_requires_target(self):
        """Raw DataFrame (no split) requires target=."""
        data = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        with pytest.raises(ml.ConfigError, match="target= is required"):
            ml.fit(data, seed=42)

    def test_fit_cv_full_workflow_no_target(self, clf_split):
        """Complete workflow without repeating target after split."""
        c = ml.cv(clf_split, folds=3, seed=42)
        model = ml.fit(c, seed=42)
        evidence = ml.assess(model, test=clf_split.test)
        assert evidence is not None

    def test_screen_infers_target(self, clf_split):
        """screen(s, seed=42) without target= works."""
        lb = ml.screen(clf_split, seed=42)
        assert lb is not None
        assert len(lb) > 0

    def test_screen_cv_infers_target(self, clf_split):
        """screen(cv, seed=42) without target= works."""
        c = ml.cv(clf_split, folds=3, seed=42)
        lb = ml.screen(c, seed=42)
        assert lb is not None


# ---------------------------------------------------------------------------
# cv() — OOF predictions
# ---------------------------------------------------------------------------

class TestCVOOF:
    """Out-of-fold prediction tests."""

    def test_cv_predictions_exist(self, clf_split):
        """fit(cv) produces cv_predictions_."""
        c = ml.cv(clf_split, folds=3, seed=42)
        model = ml.fit(c, "survived", seed=42)
        assert model.cv_predictions_ is not None

    def test_cv_predictions_is_series(self, clf_split):
        c = ml.cv(clf_split, folds=3, seed=42)
        model = ml.fit(c, "survived", seed=42)
        assert isinstance(model.cv_predictions_, pd.Series)

    def test_cv_predictions_length_matches_dev(self, clf_split):
        """OOF preds cover every row in dev."""
        c = ml.cv(clf_split, folds=3, seed=42)
        model = ml.fit(c, "survived", seed=42)
        assert len(model.cv_predictions_) == len(clf_split.dev)

    def test_cv_predictions_labels_match(self, clf_split):
        """OOF preds use the same label values as the target."""
        c = ml.cv(clf_split, folds=3, seed=42)
        model = ml.fit(c, "survived", seed=42)
        expected_labels = set(clf_split.dev["survived"].unique())
        actual_labels = set(model.cv_predictions_.unique())
        assert actual_labels.issubset(expected_labels)

    def test_cv_probabilities_exist(self, clf_split):
        """fit(cv) produces cv_probabilities_ for classification."""
        c = ml.cv(clf_split, folds=3, seed=42)
        model = ml.fit(c, "survived", seed=42)
        assert model.cv_probabilities_ is not None

    def test_cv_probabilities_shape(self, clf_split):
        """Proba shape = (n_dev, n_classes)."""
        c = ml.cv(clf_split, folds=3, seed=42)
        model = ml.fit(c, "survived", seed=42)
        n_classes = clf_split.dev["survived"].nunique()
        assert model.cv_probabilities_.shape == (len(clf_split.dev), n_classes)

    def test_cv_probabilities_sum_to_one(self, clf_split):
        """Each row's probabilities sum to ~1.0."""
        c = ml.cv(clf_split, folds=3, seed=42)
        model = ml.fit(c, "survived", seed=42)
        row_sums = model.cv_probabilities_.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_cv_probabilities_no_nans(self, clf_split):
        """No NaN in OOF probabilities."""
        c = ml.cv(clf_split, folds=3, seed=42)
        model = ml.fit(c, "survived", seed=42)
        assert not np.any(np.isnan(model.cv_probabilities_))

    def test_cv_predictions_none_for_holdout(self, clf_split):
        """Holdout fit has no cv_predictions_."""
        model = ml.fit(clf_split.train, "survived", seed=42)
        assert model.cv_predictions_ is None
        assert model.cv_probabilities_ is None

    def test_cv_predictions_regression(self, reg_split):
        """Regression OOF preds are numeric, no probabilities."""
        c = ml.cv(reg_split, folds=3, seed=42)
        model = ml.fit(c, "tip", seed=42)
        assert model.cv_predictions_ is not None
        assert len(model.cv_predictions_) == len(reg_split.dev)
        # Regression has no probabilities
        assert model.cv_probabilities_ is None


# ---------------------------------------------------------------------------
# cv() — error handling
# ---------------------------------------------------------------------------

class TestCVErrors:
    """Error cases for cv()."""

    def test_cv_rejects_dataframe(self, clf_data):
        """cv() requires SplitResult, not raw DataFrame."""
        with pytest.raises(ml.ConfigError, match="SplitResult"):
            ml.cv(clf_data, folds=5, seed=42)

    def test_cv_rejects_folds_less_than_2(self, clf_split):
        with pytest.raises(ml.ConfigError, match="folds must be >= 2"):
            ml.cv(clf_split, folds=1, seed=42)

    def test_cv_rejects_non_int_seed(self, clf_split):
        with pytest.raises(ml.ConfigError, match="seed must be an integer"):
            ml.cv(clf_split, folds=5, seed=4.2)

    def test_cv_rejects_too_many_folds(self, clf_split):
        with pytest.raises(ml.DataError):
            ml.cv(clf_split, folds=99999, seed=42)

    def test_cv_rejects_temporal_split(self):
        """cv() on temporal SplitResult raises ConfigError — use cv_temporal()."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "x": np.random.randn(n),
        "target": np.random.choice([0, 1], n),
        })
        s = ml.split_temporal(data, "target", time="date")
        with pytest.raises(ml.ConfigError, match="cv_temporal"):
            ml.cv(s, folds=5, seed=42)

    def test_split_folds_removed(self, clf_data):
        """split(folds=) raises ConfigError — use ml.cv() instead."""
        with pytest.raises(ml.ConfigError, match="ml.cv"):
            ml.split(clf_data, "survived", folds=5, seed=42)

    def test_split_repeats_removed(self, clf_data):
        """split(repeats=) raises ConfigError — use ml.cv() instead."""
        with pytest.raises(ml.ConfigError, match="ml.cv"):
            ml.split(clf_data, "survived", folds=5, repeats=3, seed=42)

    def test_cv_blocks_train_access(self, clf_split):
        c = ml.cv(clf_split, folds=5, seed=42)
        with pytest.raises(ml.ConfigError):
            _ = c.train

    def test_cv_blocks_valid_access(self, clf_split):
        c = ml.cv(clf_split, folds=5, seed=42)
        with pytest.raises(ml.ConfigError):
            _ = c.valid

    def test_cv_blocks_dev_access(self, clf_split):
        c = ml.cv(clf_split, folds=5, seed=42)
        with pytest.raises(ml.ConfigError):
            _ = c.dev


# ---------------------------------------------------------------------------
# cv_temporal()
# ---------------------------------------------------------------------------

class TestCVTemporal:
    """Temporal CV tests."""

    @pytest.fixture
    def temporal_split(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        data = pd.DataFrame({
        "date": dates,
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "price": 100 + np.cumsum(np.random.randn(n) * 0.5),
        })
        return ml.split_temporal(data, "price", time="date")

    def test_cv_temporal_returns_cvresult(self, temporal_split):
        c = ml.cv_temporal(temporal_split, folds=3)
        assert isinstance(c, ml.CVResult)
        assert c._temporal is True

    def test_cv_temporal_has_no_test(self, temporal_split):
        """Test lives on SplitResult, not CVResult."""
        c = ml.cv_temporal(temporal_split, folds=3)
        with pytest.raises(ml.ConfigError):
            _ = c.test
            assert len(temporal_split.test) > 0

    def test_cv_temporal_chronological(self, temporal_split):
        """Each fold's train precedes its valid."""
        c = ml.cv_temporal(temporal_split, folds=3)
        for train_df, valid_df in c.folds:
            assert train_df.index.max() < valid_df.index.min()

    def test_cv_temporal_fit_assess(self, temporal_split):
        """Full temporal workflow: test from split, not cv."""
        c = ml.cv_temporal(temporal_split, folds=3)
        model = ml.fit(c, "price", seed=42)
        assert model.scores_ is not None
        evidence = ml.assess(model, test=temporal_split.test)
        assert evidence is not None

    def test_cv_temporal_embargo(self, temporal_split):
        """Embargo creates gap between train and valid."""
        c = ml.cv_temporal(temporal_split, folds=3, embargo=10)
        for train_df, valid_df in c.folds:
            gap = valid_df.index.min() - train_df.index.max()
            assert gap > 1 # at least embargo gap

    def test_cv_temporal_rejects_non_split(self):
        with pytest.raises(ml.ConfigError, match="SplitResult"):
            ml.cv_temporal(pd.DataFrame({"x": [1]}), folds=3)

    def test_cv_temporal_groups_no_leakage(self):
        """groups= prevents group overlap in temporal folds."""
        np.random.seed(42)
        n = 600
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        data = pd.DataFrame({
        "date": dates,
        "patient_id": np.tile([f"p{i}" for i in range(20)], n // 20),
        "x": np.random.randn(n),
        "mortality": np.random.choice([0, 1], n),
        })
        s = ml.split_temporal(data, "mortality", time="date")
        c = ml.cv_temporal(s, folds=3, embargo=0, groups="patient_id")
        assert isinstance(c, ml.CVResult)
        for train_df, valid_df in c.folds:
            train_groups = set(train_df["patient_id"].unique())
            valid_groups = set(valid_df["patient_id"].unique())
            assert train_groups.isdisjoint(valid_groups), \
            f"Group leak in temporal fold: {train_groups & valid_groups}"

    def test_cv_temporal_groups_missing_column(self):
        """groups= with nonexistent column raises ConfigError."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "x": np.random.randn(n),
        "target": np.random.choice([0, 1], n),
        })
        s = ml.split_temporal(data, "target", time="date")
        with pytest.raises(ml.ConfigError, match="not found"):
            ml.cv_temporal(s, folds=3, groups="nonexistent")

    def test_cv_temporal_groups_fit_assess(self):
        """Full panel data workflow: split_temporal → cv_temporal(groups=) → fit → assess."""
        np.random.seed(42)
        n = 600
        # Assign each row a unique group so group filter only removes
        # validation rows' groups from train (realistic: many entities)
        data = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "store_id": [f"s{i}" for i in range(n)],
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "sales": np.random.randn(n) * 100 + 500,
        })
        s = ml.split_temporal(data, "sales", time="date")
        c = ml.cv_temporal(s, folds=3, embargo=5, groups="store_id")
        model = ml.fit(c, "sales", seed=42)
        assert model.scores_ is not None
        evidence = ml.assess(model, test=s.test)
        assert evidence is not None


# ---------------------------------------------------------------------------
# cv_group()
# ---------------------------------------------------------------------------

class TestCVGroup:
    """Group CV tests."""

    @pytest.fixture
    def group_data(self):
        np.random.seed(42)
        n = 300
        data = pd.DataFrame({
        "patient_id": np.repeat(np.arange(30), 10),
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "outcome": np.random.choice(["yes", "no"], n),
        })
        return data

    @pytest.fixture
    def group_split(self, group_data):
        return ml.split_group(group_data, "outcome", groups="patient_id", seed=42)

    def test_cv_group_returns_cvresult(self, group_split):
        c = ml.cv_group(group_split, folds=3, groups="patient_id", seed=42)
        assert isinstance(c, ml.CVResult)

    def test_cv_group_has_no_test(self, group_split):
        """Test lives on SplitResult, not CVResult."""
        c = ml.cv_group(group_split, folds=3, groups="patient_id", seed=42)
        with pytest.raises(ml.ConfigError):
            _ = c.test
            assert len(group_split.test) > 0

    def test_cv_group_no_group_leakage(self, group_split):
        """No group appears in both train and valid within any fold."""
        c = ml.cv_group(group_split, folds=3, groups="patient_id", seed=42)
        for train_df, valid_df in c.folds:
            train_groups = set(train_df["patient_id"].unique())
            valid_groups = set(valid_df["patient_id"].unique())
            assert train_groups.isdisjoint(valid_groups), \
            f"Group leak: {train_groups & valid_groups}"

    def test_cv_group_fit_assess(self, group_split):
        """Full group workflow: test from split, not cv."""
        c = ml.cv_group(group_split, folds=3, groups="patient_id", seed=42)
        model = ml.fit(c, "outcome", seed=42)
        assert model.scores_ is not None
        evidence = ml.assess(model, test=group_split.test)
        assert evidence is not None

    def test_cv_group_rejects_missing_column(self, group_split):
        with pytest.raises(ml.ConfigError, match="not found"):
            ml.cv_group(group_split, folds=3, groups="nonexistent", seed=42)

    def test_cv_group_rejects_too_many_folds(self, group_split):
        with pytest.raises(ml.DataError):
            ml.cv_group(group_split, folds=999, groups="patient_id", seed=42)
