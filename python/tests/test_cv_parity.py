"""CV parity tests — structural invariants against sklearn reference.

The backtester for the backtester. Compares ml.cv/cv_temporal/cv_group
against sklearn.model_selection splitters on STRUCTURAL PROPERTIES:

    1. Fold count matches
    2. Fold sizes balanced (±1 row)
    3. No validation overlap (partition property)
    4. Complete coverage (every dev row appears in exactly one valid fold)
    5. Stratification preserves class ratios per fold
    6. Group non-overlap (no group in both train and valid)
    7. Temporal ordering (train indices < valid indices)
    8. Train+valid = dev (no rows lost or invented)

We do NOT compare exact indices — implementations legitimately differ.
We compare the properties any correct implementation must satisfy.
"""

import numpy as np
import pandas as pd
import pytest

sklearn_ms = pytest.importorskip("sklearn.model_selection")

import ml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_200():
    """200-row binary classification dataset."""
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    data = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    data["target"] = y
    return data


@pytest.fixture
def reg_200():
    """200-row regression dataset."""
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 5)
    y = X[:, 0] * 2 + X[:, 1] + rng.randn(n) * 0.3
    data = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    data["target"] = y
    return data


@pytest.fixture
def group_200():
    """200-row grouped dataset (20 groups × 10 rows each)."""
    rng = np.random.RandomState(42)
    groups = np.repeat([f"g{i}" for i in range(20)], 10)
    n = len(groups)
    data = pd.DataFrame({
    "group_id": groups,
    "x1": rng.randn(n),
    "x2": rng.randn(n),
    "target": rng.choice([0, 1], n),
    })
    return data


@pytest.fixture
def temporal_500():
    """500-row temporal dataset."""
    rng = np.random.RandomState(42)
    n = 500
    data = pd.DataFrame({
    "date": pd.date_range("2020-01-01", periods=n, freq="D"),
    "x1": rng.randn(n),
    "x2": rng.randn(n),
    "target": rng.randn(n) * 10 + 100,
    })
    return data


# ---------------------------------------------------------------------------
# Helpers — structural invariant checks
# ---------------------------------------------------------------------------

def assert_fold_count(folds, k):
    """Invariant 1: correct number of folds."""
    assert len(folds) == k, f"Expected {k} folds, got {len(folds)}"


def assert_balanced_fold_sizes(folds, n_dev, k):
    """Invariant 2: fold sizes within ±1 of n_dev/k."""
    expected = n_dev / k
    for i, (_, valid_df) in enumerate(folds):
        size = len(valid_df)
        assert abs(size - expected) <= 1.5, \
        f"Fold {i}: valid size {size}, expected ~{expected:.0f}"


def assert_no_valid_overlap(folds):
    """Invariant 3: validation sets are disjoint."""
    seen = set()
    for i, (_, valid_df) in enumerate(folds):
        idx = set(valid_df.index.tolist())
        overlap = seen & idx
        assert len(overlap) == 0, \
        f"Fold {i} overlaps with earlier fold on {len(overlap)} rows"
        seen.update(idx)


def assert_complete_coverage(folds, dev_index):
    """Invariant 4: union of validation folds = dev."""
    all_valid = set()
    for _, valid_df in folds:
        all_valid.update(valid_df.index.tolist())
    expected = set(dev_index.tolist())
    missing = expected - all_valid
    extra = all_valid - expected
    assert len(missing) == 0, f"{len(missing)} dev rows never appear in validation"
    assert len(extra) == 0, f"{len(extra)} rows in validation not from dev"


def assert_train_valid_disjoint(folds):
    """Invariant 5: train and valid don't overlap within each fold."""
    for i, (train_df, valid_df) in enumerate(folds):
        overlap = set(train_df.index) & set(valid_df.index)
        assert len(overlap) == 0, \
        f"Fold {i}: {len(overlap)} rows in both train and valid"


def assert_stratification_ratio(folds, target_col, tolerance=0.10):
    """Invariant 6: class ratio in each fold's valid ≈ global ratio."""
    # Collect all valid targets to get the global ratio
    all_targets = pd.concat([v[target_col] for _, v in folds])
    global_ratio = all_targets.mean()

    for i, (_, valid_df) in enumerate(folds):
        fold_ratio = valid_df[target_col].mean()
        diff = abs(fold_ratio - global_ratio)
        assert diff < tolerance, \
        f"Fold {i}: class ratio {fold_ratio:.3f} vs global {global_ratio:.3f} " \
        f"(diff={diff:.3f} > {tolerance})"


def assert_group_non_overlap(folds, group_col):
    """Invariant 7: no group in both train and valid."""
    for i, (train_df, valid_df) in enumerate(folds):
        train_groups = set(train_df[group_col].unique())
        valid_groups = set(valid_df[group_col].unique())
        leak = train_groups & valid_groups
        assert len(leak) == 0, \
        f"Fold {i}: groups {leak} appear in both train and valid"


def assert_temporal_ordering(folds):
    """Invariant 8: all train indices < all valid indices."""
    for i, (train_df, valid_df) in enumerate(folds):
        if len(train_df) == 0 or len(valid_df) == 0:
            continue
        assert train_df.index.max() < valid_df.index.min(), \
            f"Fold {i}: train max index {train_df.index.max()} >= " \
            f"valid min index {valid_df.index.min()}"


def assert_no_rows_invented(folds, dev):
    """Invariant 9: every row in folds exists in dev (no fabrication)."""
    dev_idx = set(dev.index.tolist())
    for i, (train_df, valid_df) in enumerate(folds):
        train_extra = set(train_df.index) - dev_idx
        valid_extra = set(valid_df.index) - dev_idx
        assert len(train_extra) == 0, \
        f"Fold {i}: {len(train_extra)} train rows not in dev"
        assert len(valid_extra) == 0, \
        f"Fold {i}: {len(valid_extra)} valid rows not in dev"


# ---------------------------------------------------------------------------
# 1. cv() vs sklearn KFold / StratifiedKFold — structural parity
# ---------------------------------------------------------------------------

class TestCVvsSklearnKFold:
    """Both ml.cv() and sklearn KFold must satisfy identical invariants."""

    @pytest.mark.parametrize("k", [2, 3, 5, 10])
    def test_ml_cv_invariants(self, clf_200, k):
        """ml.cv() satisfies all structural invariants."""
        s = ml.split(clf_200, "target", seed=42)
        c = ml.cv(s, folds=k, seed=42)
        dev = s.dev

        assert_fold_count(c.folds, k)
        assert_balanced_fold_sizes(c.folds, len(dev), k)
        assert_no_valid_overlap(c.folds)
        assert_complete_coverage(c.folds, dev.index)
        assert_train_valid_disjoint(c.folds)
        assert_no_rows_invented(c.folds, dev)

    @pytest.mark.parametrize("k", [2, 3, 5, 10])
    def test_sklearn_kfold_invariants(self, clf_200, k):
        """sklearn KFold satisfies same invariants (reference proof)."""
        s = ml.split(clf_200, "target", seed=42)
        dev = s.dev
        X = dev.drop(columns=["target"])

        kf = sklearn_ms.KFold(n_splits=k, shuffle=True, random_state=42)
        folds = []
        for train_idx, valid_idx in kf.split(X):
            folds.append((dev.iloc[train_idx], dev.iloc[valid_idx]))

        assert_fold_count(folds, k)
        assert_balanced_fold_sizes(folds, len(dev), k)
        assert_no_valid_overlap(folds)
        assert_complete_coverage(folds, dev.index)
        assert_train_valid_disjoint(folds)

    def test_stratification_parity(self, clf_200):
        """Both ml.cv(stratify=True) and sklearn StratifiedKFold preserve class ratio."""
        s = ml.split(clf_200, "target", seed=42)
        c = ml.cv(s, folds=5, seed=42, stratify=True)
        assert_stratification_ratio(c.folds, "target", tolerance=0.08)

        # sklearn reference
        dev = s.dev
        X = dev.drop(columns=["target"])
        y = dev["target"]
        skf = sklearn_ms.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        sk_folds = []
        for train_idx, valid_idx in skf.split(X, y):
            sk_folds.append((dev.iloc[train_idx], dev.iloc[valid_idx]))
        assert_stratification_ratio(sk_folds, "target", tolerance=0.08)

    def test_fold_size_matches_sklearn(self, clf_200):
        """ml.cv(stratify=False) and sklearn KFold produce same fold sizes."""
        s = ml.split(clf_200, "target", seed=42)
        k = 5
        dev = s.dev

        # stratify=False → plain KFold, comparable to sklearn KFold
        c = ml.cv(s, folds=k, seed=42, stratify=False)
        ml_sizes = sorted([len(v) for _, v in c.folds])

        X = dev.drop(columns=["target"])
        kf = sklearn_ms.KFold(n_splits=k, shuffle=True, random_state=42)
        sk_sizes = sorted([len(vi) for _, vi in kf.split(X)])

        assert ml_sizes == sk_sizes, \
        f"ml fold sizes {ml_sizes} != sklearn {sk_sizes}"

    def test_stratified_fold_sizes_within_tolerance(self, clf_200):
        """ml.cv(stratify=True) fold sizes within ±2 of sklearn StratifiedKFold."""
        s = ml.split(clf_200, "target", seed=42)
        k = 5
        dev = s.dev

        c = ml.cv(s, folds=k, seed=42, stratify=True)
        ml_sizes = sorted([len(v) for _, v in c.folds])

        X = dev.drop(columns=["target"])
        y = dev["target"]
        skf = sklearn_ms.StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        sk_sizes = sorted([len(vi) for _, vi in skf.split(X, y)])

        for m, s_ in zip(ml_sizes, sk_sizes):
            assert abs(m - s_) <= 2, \
            f"ml stratified size {m} vs sklearn {s_} (diff > 2)"

    def test_regression_cv_invariants(self, reg_200):
        """Regression CV satisfies same invariants."""
        s = ml.split(reg_200, "target", seed=42)
        c = ml.cv(s, folds=5, seed=42)
        dev = s.dev

        assert_fold_count(c.folds, 5)
        assert_balanced_fold_sizes(c.folds, len(dev), 5)
        assert_no_valid_overlap(c.folds)
        assert_complete_coverage(c.folds, dev.index)
        assert_train_valid_disjoint(c.folds)


# ---------------------------------------------------------------------------
# 2. cv_group() vs sklearn GroupKFold — group non-overlap parity
# ---------------------------------------------------------------------------

class TestCVGroupvsSklearnGroupKFold:
    """Both must guarantee no group in both train and valid."""

    def test_ml_cv_group_invariants(self, group_200):
        """ml.cv_group() satisfies group + structural invariants."""
        s = ml.split_group(group_200, "target", groups="group_id", seed=42)
        c = ml.cv_group(s, folds=4, groups="group_id", seed=42)
        dev = s.dev

        assert_fold_count(c.folds, 4)
        assert_no_valid_overlap(c.folds)
        assert_complete_coverage(c.folds, dev.index)
        assert_train_valid_disjoint(c.folds)
        assert_group_non_overlap(c.folds, "group_id")
        assert_no_rows_invented(c.folds, dev)

    def test_sklearn_group_kfold_invariants(self, group_200):
        """sklearn GroupKFold satisfies same invariants (reference proof)."""
        s = ml.split_group(group_200, "target", groups="group_id", seed=42)
        dev = s.dev
        X = dev.drop(columns=["target"])
        groups = dev["group_id"]

        gkf = sklearn_ms.GroupKFold(n_splits=4)
        folds = []
        for train_idx, valid_idx in gkf.split(X, groups=groups):
            folds.append((dev.iloc[train_idx], dev.iloc[valid_idx]))

        assert_fold_count(folds, 4)
        assert_no_valid_overlap(folds)
        assert_complete_coverage(folds, dev.index)
        assert_train_valid_disjoint(folds)
        assert_group_non_overlap(folds, "group_id")

    def test_group_count_per_fold_matches(self, group_200):
        """Both frameworks distribute groups across folds comparably."""
        s = ml.split_group(group_200, "target", groups="group_id", seed=42)
        dev = s.dev
        k = 4

        c = ml.cv_group(s, folds=k, groups="group_id", seed=42)
        ml_groups_per_fold = sorted(
        [len(v["group_id"].unique()) for _, v in c.folds]
        )

        X = dev.drop(columns=["target"])
        gkf = sklearn_ms.GroupKFold(n_splits=k)
        sk_groups_per_fold = sorted(
        [len(dev.iloc[vi]["group_id"].unique())
        for _, vi in gkf.split(X, groups=dev["group_id"])]
        )

        # Both should distribute ~n_groups/k groups per fold (±2)
        n_groups = len(dev["group_id"].unique())
        expected = n_groups / k
        for sizes, name in [(ml_groups_per_fold, "ml"), (sk_groups_per_fold, "sklearn")]:
            for sz in sizes:
                assert abs(sz - expected) <= 3, \
                f"{name}: {sz} groups in fold, expected ~{expected:.0f}"


# ---------------------------------------------------------------------------
# 3. cv_temporal() vs sklearn TimeSeriesSplit — temporal ordering parity
# ---------------------------------------------------------------------------

class TestCVTemporalvsSklearnTSS:
    """Both must guarantee temporal ordering (train < valid)."""

    def test_ml_cv_temporal_invariants(self, temporal_500):
        """ml.cv_temporal() satisfies temporal + structural invariants."""
        s = ml.split_temporal(temporal_500, "target", time="date")
        c = ml.cv_temporal(s, folds=5)
        dev = s.dev

        assert_fold_count(c.folds, 5)
        assert_no_valid_overlap(c.folds)
        assert_train_valid_disjoint(c.folds)
        assert_temporal_ordering(c.folds)
        assert_no_rows_invented(c.folds, dev)

    def test_sklearn_tss_invariants(self, temporal_500):
        """sklearn TimeSeriesSplit satisfies same invariants (reference proof)."""
        s = ml.split_temporal(temporal_500, "target", time="date")
        dev = s.dev
        X = dev.drop(columns=["target"])

        tss = sklearn_ms.TimeSeriesSplit(n_splits=5)
        folds = []
        for train_idx, valid_idx in tss.split(X):
            folds.append((dev.iloc[train_idx], dev.iloc[valid_idx]))

        assert_fold_count(folds, 5)
        assert_no_valid_overlap(folds)
        assert_train_valid_disjoint(folds)
        assert_temporal_ordering(folds)

    def test_expanding_window_train_grows(self, temporal_500):
        """Expanding window: each fold's train >= previous fold's train."""
        s = ml.split_temporal(temporal_500, "target", time="date")
        c = ml.cv_temporal(s, folds=5, window="expanding")
        prev_train_size = 0
        for train_df, _ in c.folds:
            assert len(train_df) >= prev_train_size, \
            "Expanding window: train should grow monotonically"
            prev_train_size = len(train_df)

    def test_sklearn_tss_also_expands(self, temporal_500):
        """sklearn TimeSeriesSplit is expanding by default — same property."""
        s = ml.split_temporal(temporal_500, "target", time="date")
        dev = s.dev
        X = dev.drop(columns=["target"])

        tss = sklearn_ms.TimeSeriesSplit(n_splits=5)
        prev_train_size = 0
        for train_idx, _ in tss.split(X):
            assert len(train_idx) >= prev_train_size
            prev_train_size = len(train_idx)

    def test_embargo_gap_vs_sklearn_gap(self, temporal_500):
        """ml.cv_temporal(embargo=N) vs sklearn TimeSeriesSplit(gap=N)."""
        s = ml.split_temporal(temporal_500, "target", time="date")
        dev = s.dev
        gap = 10

        # ml
        c = ml.cv_temporal(s, folds=3, embargo=gap)
        for train_df, valid_df in c.folds:
            if len(train_df) > 0 and len(valid_df) > 0:
                actual_gap = valid_df.index.min() - train_df.index.max()
                assert actual_gap > 1, \
                f"ml embargo gap {actual_gap} should be > 1"

                # sklearn
                X = dev.drop(columns=["target"])
                tss = sklearn_ms.TimeSeriesSplit(n_splits=3, gap=gap)
                for train_idx, valid_idx in tss.split(X):
                    if len(train_idx) > 0 and len(valid_idx) > 0:
                        actual_gap = valid_idx.min() - train_idx.max()
                        assert actual_gap > 1, \
                        f"sklearn gap {actual_gap} should be > 1"


# ---------------------------------------------------------------------------
# 4. Sliding window — ml vs sklearn
# ---------------------------------------------------------------------------

class TestCVSlidingWindow:
    """Sliding window: fixed train size."""

    def test_sliding_window_train_bounded(self, temporal_500):
        """window='sliding' caps training size."""
        s = ml.split_temporal(temporal_500, "target", time="date")
        window_size = 50
        c = ml.cv_temporal(s, folds=3, window="sliding", window_size=window_size)
        for train_df, _ in c.folds:
            assert len(train_df) <= window_size, \
            f"Sliding window: train size {len(train_df)} > window_size {window_size}"

    def test_sklearn_tss_max_train_also_caps(self, temporal_500):
        """sklearn TimeSeriesSplit(max_train_size=N) caps training size."""
        s = ml.split_temporal(temporal_500, "target", time="date")
        dev = s.dev
        X = dev.drop(columns=["target"])
        window_size = 50

        tss = sklearn_ms.TimeSeriesSplit(n_splits=3, max_train_size=window_size)
        for train_idx, _ in tss.split(X):
            assert len(train_idx) <= window_size


# ---------------------------------------------------------------------------
# 5. Determinism parity
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Both frameworks produce deterministic results with same seed."""

    def test_ml_cv_deterministic(self, clf_200):
        s = ml.split(clf_200, "target", seed=42)
        c1 = ml.cv(s, folds=5, seed=42)
        c2 = ml.cv(s, folds=5, seed=42)
        for (_, v1), (_, v2) in zip(c1.folds, c2.folds):
            assert v1.index.tolist() == v2.index.tolist()

    def test_sklearn_kfold_deterministic(self, clf_200):
        s = ml.split(clf_200, "target", seed=42)
        dev = s.dev
        X = dev.drop(columns=["target"])

        kf1 = sklearn_ms.KFold(n_splits=5, shuffle=True, random_state=42)
        kf2 = sklearn_ms.KFold(n_splits=5, shuffle=True, random_state=42)
        for (_, v1), (_, v2) in zip(kf1.split(X), kf2.split(X)):
            assert list(v1) == list(v2)

    def test_ml_cv_group_deterministic(self, group_200):
        s = ml.split_group(group_200, "target", groups="group_id", seed=42)
        c1 = ml.cv_group(s, folds=4, groups="group_id", seed=42)
        c2 = ml.cv_group(s, folds=4, groups="group_id", seed=42)
        for (_, v1), (_, v2) in zip(c1.folds, c2.folds):
            assert v1.index.tolist() == v2.index.tolist()


# ---------------------------------------------------------------------------
# 6. Cross-framework score sanity — same data, similar scores
# ---------------------------------------------------------------------------

class TestScoreSanity:
    """ml and sklearn on same data should produce similar-range scores.

    Not exact match (different defaults), but same ballpark.
    This catches gross bugs like inverted splits or data leakage.
    """

    def test_cv_clf_score_in_sklearn_range(self, clf_200):
        """ml CV accuracy within 15% of sklearn CV accuracy."""
        pytest.importorskip("sklearn.linear_model")
        from sklearn.linear_model import LogisticRegression

        s = ml.split(clf_200, "target", seed=42)

        # ml
        c = ml.cv(s, folds=5, seed=42)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ml.fit(c, "target", algorithm="logistic", seed=42)
            ml_acc = model.scores_.get("accuracy_mean", 0)

            # sklearn reference
            dev = s.dev
            X = dev.drop(columns=["target"]).values
            y = dev["target"].values
            clf = LogisticRegression(random_state=42, max_iter=200)
            skf = sklearn_ms.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sk_scores = sklearn_ms.cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
                sk_acc = float(np.mean(sk_scores))

                assert abs(ml_acc - sk_acc) < 0.15, \
                f"ml accuracy {ml_acc:.3f} too far from sklearn {sk_acc:.3f}"

    def test_cv_reg_score_in_sklearn_range(self, reg_200):
        """ml CV R² within 0.20 of sklearn CV R²."""
        pytest.importorskip("sklearn.linear_model")
        from sklearn.linear_model import LinearRegression

        s = ml.split(reg_200, "target", seed=42)

        # ml
        c = ml.cv(s, folds=5, seed=42)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ml.fit(c, "target", algorithm="linear", seed=42)
            ml_r2 = model.scores_.get("r2_mean", 0)

            # sklearn reference
            dev = s.dev
            X = dev.drop(columns=["target"]).values
            y = dev["target"].values
            reg = LinearRegression()
            kf = sklearn_ms.KFold(n_splits=5, shuffle=True, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sk_scores = sklearn_ms.cross_val_score(reg, X, y, cv=kf, scoring="r2")
                sk_r2 = float(np.mean(sk_scores))

                assert abs(ml_r2 - sk_r2) < 0.20, \
                f"ml R² {ml_r2:.3f} too far from sklearn {sk_r2:.3f}"
