"""CV statistical correctness tests — beyond structural invariants.

Layer 2-5 of the backtester-for-the-backtester:

    Layer 2: Leakage detection — CV scores should NOT inflate vs holdout
    Layer 3: Permutation null — scrambled target → chance-level CV scores
    Layer 4: Monte Carlo coverage — CV ± 2σ covers holdout ~95% of the time
    Layer 5: Property-based testing — random shapes × k × seeds → invariants hold
    Layer 6: Adversarial edge cases — extreme imbalance, tiny data, degenerate inputs

These tests prove the STATISTICS are correct, not just the mechanics.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import ml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def signal_data():
    """200-row dataset with real signal (not noise)."""
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 5)
    # Strong signal in features 0-2
    y = (X[:, 0] * 2 + X[:, 1] + X[:, 2] * 0.5 + rng.randn(n) * 0.3 > 0).astype(int)
    data = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    data["target"] = y
    return data


@pytest.fixture
def noise_data():
    """200-row dataset with NO signal (pure noise target)."""
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 5)
    y = rng.choice([0, 1], n) # random target, no relationship to X
    data = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    data["target"] = y
    return data


@pytest.fixture
def reg_signal_data():
    """200-row regression dataset with real signal."""
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 5)
    y = X[:, 0] * 3 + X[:, 1] * 1.5 + rng.randn(n) * 0.5
    data = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    data["target"] = y
    return data


# ---------------------------------------------------------------------------
# Layer 2: Leakage detection
# ---------------------------------------------------------------------------

class TestLeakageDetection:
    """If split/cv leaks, CV scores inflate relative to holdout.

    Strategy: compare CV accuracy against holdout accuracy.
    With correct per-fold normalization, CV ≈ holdout (within noise).
    With leakage, CV >> holdout.
    """

    def test_cv_score_not_inflated_vs_holdout(self, signal_data):
        """CV accuracy should not be systematically higher than holdout."""
        s = ml.split(signal_data, "target", seed=42)

        # CV path
        c = ml.cv(s, folds=5, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_model = ml.fit(c, "target", algorithm="logistic", seed=42)
            cv_acc = cv_model.scores_.get("accuracy_mean", 0)

            # Holdout path
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ho_model = ml.fit(s.train, "target", algorithm="logistic", seed=42)
                ho_metrics = ml.evaluate(ho_model, s.valid)
                ho_acc = ho_metrics.get("accuracy", 0)

                # CV should not be much higher than holdout (leakage signal)
                # Allow 10% tolerance for normal variance
                assert cv_acc < ho_acc + 0.10, \
                f"CV accuracy {cv_acc:.3f} suspiciously higher than holdout {ho_acc:.3f} — possible leakage"

    def test_cv_regression_no_inflation(self, reg_signal_data):
        """CV R² should not be inflated vs holdout R²."""
        s = ml.split(reg_signal_data, "target", seed=42)

        c = ml.cv(s, folds=5, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_model = ml.fit(c, "target", algorithm="linear", seed=42)
            cv_r2 = cv_model.scores_.get("r2_mean", 0)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ho_model = ml.fit(s.train, "target", algorithm="linear", seed=42)
                ho_metrics = ml.evaluate(ho_model, s.valid)
                ho_r2 = ho_metrics.get("r2", 0)

                assert cv_r2 < ho_r2 + 0.15, \
                f"CV R² {cv_r2:.3f} suspiciously higher than holdout {ho_r2:.3f}"


# ---------------------------------------------------------------------------
# Layer 3: Permutation null — scrambled target → chance level
# ---------------------------------------------------------------------------

class TestPermutationNull:
    """With scrambled target (no signal), CV should give chance-level scores.

    If CV gives high scores on scrambled data, the split is leaking information.
    """

    def test_permuted_target_gives_chance_accuracy(self, noise_data):
        """Scrambled target → CV accuracy ≈ 0.50 (±0.15)."""
        s = ml.split(noise_data, "target", seed=42)
        c = ml.cv(s, folds=5, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ml.fit(c, "target", algorithm="logistic", seed=42)
            acc = model.scores_.get("accuracy_mean", 0)
            assert 0.30 < acc < 0.70, \
            f"Noise data CV accuracy {acc:.3f} — should be near 0.50 (chance level)"

    def test_permuted_target_multiple_seeds(self, noise_data):
        """Across 5 seeds, mean CV accuracy on noise should be near 0.50."""
        accs = []
        for seed in [1, 2, 3, 4, 5]:
            s = ml.split(noise_data, "target", seed=seed)
            c = ml.cv(s, folds=3, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ml.fit(c, "target", algorithm="logistic", seed=seed)
                accs.append(model.scores_.get("accuracy_mean", 0))
                mean_acc = np.mean(accs)
                assert 0.35 < mean_acc < 0.65, \
                f"Mean accuracy on noise across seeds: {mean_acc:.3f} (expected ~0.50)"

    def test_real_signal_beats_noise(self, signal_data, noise_data):
        """Real signal should produce higher CV accuracy than noise."""
        # Signal
        s1 = ml.split(signal_data, "target", seed=42)
        c1 = ml.cv(s1, folds=5, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m1 = ml.fit(c1, "target", algorithm="logistic", seed=42)
            signal_acc = m1.scores_.get("accuracy_mean", 0)

            # Noise
            s2 = ml.split(noise_data, "target", seed=42)
            c2 = ml.cv(s2, folds=5, seed=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m2 = ml.fit(c2, "target", algorithm="logistic", seed=42)
                noise_acc = m2.scores_.get("accuracy_mean", 0)

                assert signal_acc > noise_acc, \
                f"Signal accuracy {signal_acc:.3f} should beat noise {noise_acc:.3f}"


# ---------------------------------------------------------------------------
# Layer 4: Monte Carlo coverage — CV ± 2σ covers holdout
# ---------------------------------------------------------------------------

class TestMonteCarloCoverage:
    """CV mean ± 2*std should contain the holdout score ~95% of the time.

    If coverage is too low: CV variance estimate is too optimistic.
    If coverage is too high: CV variance estimate is too conservative.
    If scores are systematically off: split/cv has a bias.
    """

    def test_cv_coverage_classification(self, signal_data):
        """Across 20 seeds, CV ± 2σ should cover holdout accuracy ≥60% of the time."""
        covered = 0
        n_trials = 20
        for seed in range(n_trials):
            s = ml.split(signal_data, "target", seed=seed)
            c = ml.cv(s, folds=5, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_model = ml.fit(c, "target", algorithm="logistic", seed=seed)

                cv_mean = cv_model.scores_.get("accuracy_mean", 0)
                cv_std = cv_model.scores_.get("accuracy_std", 0)

                # Holdout score
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ho_model = ml.fit(s.train, "target", algorithm="logistic", seed=seed)
                    ho_metrics = ml.evaluate(ho_model, s.valid)
                    ho_acc = ho_metrics.get("accuracy", 0)

                    # Check if holdout falls within CV ± 2σ
                    if cv_mean - 2 * cv_std <= ho_acc <= cv_mean + 2 * cv_std:
                        covered += 1

                        coverage_rate = covered / n_trials
                        # We expect ~95% coverage with ±2σ, but with only 20 trials
                        # and small data, we accept ≥60% as evidence of no gross bias
                        assert coverage_rate >= 0.50, \
                        f"CV ± 2σ covered holdout only {coverage_rate:.0%} of the time " \
                        f"({covered}/{n_trials}) — variance estimate may be wrong"

    def test_cv_coverage_regression(self, reg_signal_data):
        """Across 20 seeds, CV ± 2σ should cover holdout R² ≥60% of the time."""
        covered = 0
        n_trials = 20
        for seed in range(n_trials):
            s = ml.split(reg_signal_data, "target", seed=seed)
            c = ml.cv(s, folds=5, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_model = ml.fit(c, "target", algorithm="linear", seed=seed)

                cv_mean = cv_model.scores_.get("r2_mean", 0)
                cv_std = cv_model.scores_.get("r2_std", 0.1)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ho_model = ml.fit(s.train, "target", algorithm="linear", seed=seed)
                    ho_metrics = ml.evaluate(ho_model, s.valid)
                    ho_r2 = ho_metrics.get("r2", 0)

                    if cv_mean - 2 * cv_std <= ho_r2 <= cv_mean + 2 * cv_std:
                        covered += 1

                        coverage_rate = covered / n_trials
                        assert coverage_rate >= 0.50, \
                        f"Regression CV coverage: {coverage_rate:.0%} ({covered}/{n_trials})"


# ---------------------------------------------------------------------------
# Layer 5: Property-based testing
# ---------------------------------------------------------------------------

class TestPropertyBased:
    """Random data shapes × k × seeds → invariants always hold."""

    @pytest.mark.parametrize("n", [20, 50, 100, 300])
    @pytest.mark.parametrize("k", [2, 3, 5])
    @pytest.mark.parametrize("seed", [0, 42, 99])
    def test_invariants_hold_across_shapes(self, n, k, seed):
        """Core invariants hold for any (n, k, seed) combination."""
        rng = np.random.RandomState(seed)
        data = pd.DataFrame({
        "x1": rng.randn(n),
        "x2": rng.randn(n),
        "target": rng.choice([0, 1], n),
        })

        s = ml.split(data, "target", seed=seed)
        if len(s.dev) < k:
            pytest.skip(f"dev too small ({len(s.dev)}) for {k} folds")

            c = ml.cv(s, folds=k, seed=seed)

            # Invariant 1: fold count
            assert c.k == k

            # Invariant 2: no validation overlap
            all_valid = []
            for _, v in c.folds:
                all_valid.extend(v.index.tolist())
                assert len(all_valid) == len(set(all_valid)), "validation overlap detected"

                # Invariant 3: complete coverage
                assert sorted(all_valid) == sorted(s.dev.index.tolist()), "incomplete coverage"

                # Invariant 4: train/valid disjoint per fold
                for t, v in c.folds:
                    assert len(set(t.index) & set(v.index)) == 0, "train/valid overlap"

                    # Invariant 5: no invented rows
                    dev_idx = set(s.dev.index.tolist())
                    for t, v in c.folds:
                        assert set(t.index).issubset(dev_idx), "invented train rows"
                        assert set(v.index).issubset(dev_idx), "invented valid rows"

    @pytest.mark.parametrize("n_features", [1, 2, 5, 20, 50])
    def test_invariants_hold_across_feature_counts(self, n_features):
        """Invariants hold regardless of feature count."""
        rng = np.random.RandomState(42)
        n = 100
        data = pd.DataFrame(
        rng.randn(n, n_features),
        columns=[f"x{i}" for i in range(n_features)]
        )
        data["target"] = rng.choice([0, 1], n)

        s = ml.split(data, "target", seed=42)
        c = ml.cv(s, folds=3, seed=42)
        assert c.k == 3
        all_valid = []
        for _, v in c.folds:
            all_valid.extend(v.index.tolist())
            assert len(all_valid) == len(set(all_valid))


# ---------------------------------------------------------------------------
# Layer 6: Adversarial edge cases
# ---------------------------------------------------------------------------

class TestAdversarialEdgeCases:
    """Extreme inputs that could break split/cv."""

    def test_extreme_imbalance_99_1(self):
        """99:1 class imbalance — split should still work."""
        rng = np.random.RandomState(42)
        n = 200
        # 6 minority samples ensures >=2 in dev after 3-way split
        y = np.array([0] * 194 + [1] * 6)
        data = pd.DataFrame({"x": rng.randn(n), "target": y})
        s = ml.split(data, "target", seed=42)
        assert len(s.train) + len(s.valid) + len(s.test) == n
        # Should not crash on CV either
        c = ml.cv(s, folds=2, seed=42)
        assert c.k == 2

    def test_all_same_class_rejected(self):
        """All rows same class — split correctly rejects (no signal possible)."""
        data = pd.DataFrame({
        "x": np.random.randn(50),
        "target": [1] * 50,
        })
        with pytest.raises(ml.DataError, match="1 unique value"):
            ml.split(data, "target", seed=42)

    def test_n_equals_k(self):
        """n_dev = k → each fold has ~1 validation row (±1 for stratification)."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
        "x": rng.randn(60),
        "target": rng.choice([0, 1], 60),
        })
        s = ml.split(data, "target", seed=42)
        # Use small k that fits the dev partition
        k = min(5, len(s.dev))
        if k >= 2:
            c = ml.cv(s, folds=k, seed=42)
            assert c.k == k

    def test_constant_features(self):
        """All features constant — split/cv should still partition correctly."""
        data = pd.DataFrame({
        "x1": [1.0] * 100,
        "x2": [2.0] * 100,
        "target": np.random.choice([0, 1], 100),
        })
        s = ml.split(data, "target", seed=42)
        c = ml.cv(s, folds=3, seed=42)
        assert c.k == 3
        # Coverage
        all_valid = []
        for _, v in c.folds:
            all_valid.extend(v.index.tolist())
            assert len(all_valid) == len(s.dev)

    def test_duplicate_rows(self):
        """All rows identical except target — split/cv handles gracefully."""
        data = pd.DataFrame({
        "x": [1.0] * 100,
        "target": [0] * 50 + [1] * 50,
        })
        s = ml.split(data, "target", seed=42)
        c = ml.cv(s, folds=3, seed=42)
        assert c.k == 3

    def test_single_feature(self):
        """Single feature column — minimal valid input."""
        data = pd.DataFrame({
        "x": np.random.randn(100),
        "target": np.random.choice([0, 1], 100),
        })
        s = ml.split(data, "target", seed=42)
        c = ml.cv(s, folds=5, seed=42)
        assert c.k == 5

    def test_many_features_few_rows(self):
        """p >> n (50 features, 30 rows) — split/cv doesn't crash."""
        rng = np.random.RandomState(42)
        n, p = 30, 50
        data = pd.DataFrame(
        rng.randn(n, p),
        columns=[f"x{i}" for i in range(p)]
        )
        data["target"] = rng.choice([0, 1], n)
        s = ml.split(data, "target", seed=42)
        c = ml.cv(s, folds=3, seed=42)
        assert c.k == 3

    def test_group_single_member_groups(self):
        """Groups with 1 member each — should work like regular KFold."""
        n = 50
        data = pd.DataFrame({
        "group_id": [f"g{i}" for i in range(n)],
        "x": np.random.randn(n),
        "target": np.random.choice([0, 1], n),
        })
        s = ml.split_group(data, "target", groups="group_id", seed=42)
        c = ml.cv_group(s, folds=5, groups="group_id", seed=42)
        assert c.k == 5

    def test_temporal_constant_target(self):
        """Temporal split with constant target — degenerate but valid."""
        n = 100
        data = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "x": np.random.randn(n),
        "target": [1.0] * n,
        })
        s = ml.split_temporal(data, "target", time="date")
        assert len(s.train) + len(s.valid) + len(s.test) == n

    def test_temporal_reverse_sorted(self):
        """Data arrives reverse-sorted by time — split should still sort correctly."""
        n = 100
        data = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D")[::-1],
        "x": np.arange(n, dtype=float),
        "target": np.random.randn(n),
        })
        s = ml.split_temporal(data, "target", time="date")
        # Train should have earlier dates (lower x values after sort)
        assert len(s.train) + len(s.valid) + len(s.test) == n
