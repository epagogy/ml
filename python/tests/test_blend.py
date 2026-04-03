"""Tests for ml.blend() — prediction blending. Chain 3.1."""

import numpy as np
import pytest

import ml


def test_blend_mean_equal_weights():
    """Default method is arithmetic mean with equal weights. Chain 3.1."""
    rng = np.random.RandomState(42)
    p1 = rng.rand(100)
    p2 = rng.rand(100)
    result = ml.blend([p1, p2])
    expected = (p1 + p2) / 2
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_blend_rank_invariant_to_calibration():
    """Rank blending is invariant to monotone rescaling of predictions. Chain 3.1."""
    rng = np.random.RandomState(42)
    p1 = rng.rand(100)
    # Miscalibrated: same rank order as p1, compressed near 0
    p1_miscal = p1 ** 10
    p2 = rng.rand(100)

    rank_original = ml.blend([p1, p2], method="rank")
    rank_miscal = ml.blend([p1_miscal, p2], method="rank")

    # Same rank order → identical rank blend
    np.testing.assert_allclose(rank_original, rank_miscal, rtol=1e-10)


def test_blend_geometric_log_space():
    """Geometric mean differs from arithmetic mean for non-uniform distributions. Chain 3.1."""
    p1 = np.array([0.1, 0.5, 0.9])
    p2 = np.array([0.9, 0.5, 0.1])
    result = ml.blend([p1, p2], method="geometric")
    arithmetic = (p1 + p2) / 2
    # Geometric != arithmetic for non-uniform values
    assert not np.allclose(result, arithmetic)
    # Result should be in [0, 1]
    assert np.all(result >= 0)
    assert np.all(result <= 1)


def test_blend_power_1_equals_mean():
    """power=1 gives the same result as arithmetic mean. Chain 3.1."""
    rng = np.random.RandomState(7)
    p1 = rng.rand(50)
    p2 = rng.rand(50)
    p3 = rng.rand(50)
    power1 = ml.blend([p1, p2, p3], method="power", power=1.0)
    mean_result = ml.blend([p1, p2, p3], method="mean")
    np.testing.assert_allclose(power1, mean_result, rtol=1e-6)


def test_blend_custom_weights():
    """Custom weights are respected in the blend. Chain 3.1."""
    rng = np.random.RandomState(3)
    p1 = rng.rand(80)
    p2 = rng.rand(80)
    result = ml.blend([p1, p2], method="mean", weights=[0.7, 0.3])
    expected = 0.7 * p1 + 0.3 * p2
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_blend_weights_sum_check():
    """weights that don't sum to 1.0 raise ConfigError. Chain 3.1."""
    rng = np.random.RandomState(5)
    p1 = rng.rand(50)
    p2 = rng.rand(50)
    with pytest.raises(ml.ConfigError, match="sum to 1"):
        ml.blend([p1, p2], weights=[0.6, 0.6])  # sums to 1.2
