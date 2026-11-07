"""Unit tests for statistical baseline confidence intervals and verification logic."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.run_ber_baseline import (
    compute_clopper_pearson_interval,
    compute_rule_of_three_upper_bound,
)


def test_clopper_pearson_zero_errors():
    """Verify exact two-sided Clopper-Pearson interval when zero errors are observed."""
    n_bits = 100_000
    low, high = compute_clopper_pearson_interval(0, n_bits, confidence=0.95)
    assert low == 0.0

    # Theoretical exact formula: 1 - (alpha/2)**(1/n) where alpha = 0.05
    expected_high = 1.0 - (0.025) ** (1.0 / n_bits)
    assert abs(high - expected_high) < 1e-12

    # Verify that Rule of Three one-sided upper bound differs from
    # two-sided Clopper-Pearson
    rot_upper = compute_rule_of_three_upper_bound(n_bits, confidence=0.95)
    expected_rot = -np.log(0.05) / n_bits  # ~ 2.996e-5
    assert abs(rot_upper - expected_rot) < 1e-12
    # Two-sided interval uses alpha/2 (0.025), so high is ~ 3.689e-5,
    # greater than rot_upper (~2.996e-5)
    assert high > rot_upper


def test_clopper_pearson_all_errors():
    """Verify exact two-sided Clopper-Pearson interval when all bits are in error."""
    n_bits = 100_000
    low, high = compute_clopper_pearson_interval(n_bits, n_bits, confidence=0.95)
    assert high == 1.0

    expected_low = (0.025) ** (1.0 / n_bits)
    assert abs(low - expected_low) < 1e-12
    assert low < 1.0


def test_clopper_pearson_standard_cases():
    """Verify standard nonzero error counts."""
    n_bits = 100_000
    k_errors = 1000
    low, high = compute_clopper_pearson_interval(k_errors, n_bits, confidence=0.95)
    assert 0.0 < low < (k_errors / n_bits) < high < 1.0
    assert abs((low + high) / 2.0 - (k_errors / n_bits)) < 1e-3


def test_clopper_pearson_edge_case_validations():
    """Verify exception handling on invalid inputs."""
    # Zero or negative bits
    with pytest.raises(ValueError, match="n_bits must be positive"):
        compute_clopper_pearson_interval(0, 0)

    with pytest.raises(ValueError, match="n_bits must be positive"):
        compute_clopper_pearson_interval(0, -100)

    # Negative error count
    with pytest.raises(ValueError, match="k_errors must be in"):
        compute_clopper_pearson_interval(-1, 1000)

    # Error count exceeding trial count
    with pytest.raises(ValueError, match="k_errors must be in"):
        compute_clopper_pearson_interval(1001, 1000)

    # Invalid confidence levels
    with pytest.raises(ValueError, match="confidence must be in"):
        compute_clopper_pearson_interval(10, 1000, confidence=1.5)

    with pytest.raises(ValueError, match="confidence must be in"):
        compute_clopper_pearson_interval(10, 1000, confidence=0.0)
