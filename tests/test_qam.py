"""Tests for Gray-coded 16-QAM modulation and demodulation."""

import numpy as np
import pytest

from wireless.qam import demodulate_16qam, modulate_16qam


def test_16qam_output_symbol_count():
    bits = np.array([0, 0, 0, 0, 0, 1, 1, 1])

    symbols = modulate_16qam(bits)

    assert len(symbols) == 2


def test_16qam_output_is_complex():
    bits = np.array([0, 0, 0, 0])

    symbols = modulate_16qam(bits)

    assert np.iscomplexobj(symbols)


def test_16qam_constellation_has_16_unique_points():
    all_bit_combinations = np.array(
        [[(number >> shift) & 1 for shift in (3, 2, 1, 0)] for number in range(16)],
        dtype=np.int_,
    )

    symbols = modulate_16qam(all_bit_combinations.reshape(-1))

    assert len(np.unique(symbols)) == 16


def test_16qam_average_symbol_energy_is_one():
    all_bit_combinations = np.array(
        [[(number >> shift) & 1 for shift in (3, 2, 1, 0)] for number in range(16)],
        dtype=np.int_,
    )

    symbols = modulate_16qam(all_bit_combinations.reshape(-1))

    average_energy = np.mean(np.abs(symbols) ** 2)

    assert average_energy == pytest.approx(1.0)


def test_16qam_noiseless_recovery():
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=4000, dtype=np.int_)

    symbols = modulate_16qam(bits)
    recovered_bits = demodulate_16qam(symbols)

    np.testing.assert_array_equal(recovered_bits, bits)


def test_16qam_gray_mapping():
    bits = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
        ],
        dtype=np.int_,
    )

    symbols = modulate_16qam(bits)

    expected_i = np.array([3, 1, -1, -3]) / np.sqrt(10)

    np.testing.assert_allclose(symbols.real, expected_i)


@pytest.mark.parametrize(
    "bits",
    [
        [0, 0, 0],
        [0, 0, 0, 0, 1],
    ],
)
def test_16qam_rejects_invalid_bit_length(bits):
    with pytest.raises(ValueError, match="divisible by 4"):
        modulate_16qam(bits)


@pytest.mark.parametrize(
    "bits",
    [
        [0, 1, 2, 0],
        [0, 1, -1, 0],
    ],
)
def test_16qam_rejects_non_binary_values(bits):
    with pytest.raises(ValueError, match="only 0 and 1"):
        modulate_16qam(bits)


def test_16qam_rejects_multidimensional_bits():
    bits = np.zeros((2, 2), dtype=np.int_)

    with pytest.raises(ValueError, match="one-dimensional"):
        modulate_16qam(bits)


def test_16qam_rejects_non_finite_received_symbols():
    symbols = np.array([complex(np.nan, 0)])

    with pytest.raises(ValueError, match="finite"):
        demodulate_16qam(symbols)


def test_16qam_empty_input():
    bits = np.array([], dtype=np.int_)

    symbols = modulate_16qam(bits)
    recovered = demodulate_16qam(symbols)

    assert len(symbols) == 0
    assert len(recovered) == 0
