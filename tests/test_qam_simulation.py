"""Tests for 16-QAM BER simulation."""

import numpy as np
import pytest

from wireless.qam_simulation import (
    add_16qam_awgn,
    simulate_16qam_ber,
    theoretical_ber_16qam_awgn,
)


@pytest.mark.parametrize("channel", ["AWGN", "Rayleigh", "Rician"])
def test_16qam_simulation_returns_valid_result(channel):
    result = simulate_16qam_ber(
        ebn0_db=[0, 4, 8],
        num_bits=20_000,
        random_seed=42,
        channel=channel,
    )

    assert result["modulation"] == "16QAM"
    assert result["channel"] == channel
    assert result["num_bits"] == 20_000
    assert len(result["ebn0_db"]) == 3
    assert len(result["ber_simulated"]) == 3
    assert len(result["error_counts"]) == 3

    assert all(0 <= ber <= 1 for ber in result["ber_simulated"])
    assert all(0 <= count <= 20_000 for count in result["error_counts"])


def test_16qam_awgn_includes_theoretical_reference():
    result = simulate_16qam_ber(
        ebn0_db=[0, 5, 10],
        num_bits=40_000,
        random_seed=42,
        channel="AWGN",
    )

    assert result["ber_theoretical"] is not None
    assert len(result["ber_theoretical"]) == 3
    assert result["theoretical_reference"] is not None


@pytest.mark.parametrize("channel", ["Rayleigh", "Rician"])
def test_16qam_fading_does_not_claim_awgn_theory(channel):
    result = simulate_16qam_ber(
        ebn0_db=[0, 5, 10],
        num_bits=20_000,
        random_seed=42,
        channel=channel,
    )

    assert result["ber_theoretical"] is None
    assert result["theoretical_reference"] is None


def test_16qam_awgn_ber_decreases_with_snr():
    result = simulate_16qam_ber(
        ebn0_db=[0, 5, 10, 15],
        num_bits=200_000,
        random_seed=42,
        channel="AWGN",
    )

    ber = result["ber_simulated"]

    assert ber[0] > ber[1] > ber[2] > ber[3]


def test_16qam_awgn_simulation_is_reproducible():
    first = simulate_16qam_ber(
        ebn0_db=[0, 5, 10],
        num_bits=20_000,
        random_seed=123,
        channel="AWGN",
    )

    second = simulate_16qam_ber(
        ebn0_db=[0, 5, 10],
        num_bits=20_000,
        random_seed=123,
        channel="AWGN",
    )

    assert first["ber_simulated"] == second["ber_simulated"]
    assert first["error_counts"] == second["error_counts"]


def test_16qam_simulation_uses_complete_bit_groups():
    result = simulate_16qam_ber(
        ebn0_db=[5],
        num_bits=10_003,
        random_seed=42,
    )

    assert result["num_bits"] == 10_000


def test_16qam_constellation_output_is_json_compatible():
    result = simulate_16qam_ber(
        ebn0_db=[0, 5],
        num_bits=1000,
        random_seed=42,
    )

    samples = result["constellation_received"]

    assert len(samples) <= 200
    assert all(len(sample) == 2 for sample in samples)

    for sample in samples:
        assert all(np.isfinite(value) for value in sample)


def test_16qam_noise_is_finite():
    symbols = np.ones(100, dtype=np.complex128)
    rng = np.random.default_rng(42)

    received = add_16qam_awgn(symbols, 10, rng)

    assert received.shape == symbols.shape
    assert np.all(np.isfinite(received))


@pytest.mark.parametrize(
    "invalid_snr",
    [
        [],
        [[0, 2]],
        [0, np.nan],
        [0, np.inf],
    ],
)
def test_16qam_rejects_invalid_snr(invalid_snr):
    with pytest.raises(ValueError):
        simulate_16qam_ber(ebn0_db=invalid_snr)


def test_16qam_rejects_unsupported_channel():
    with pytest.raises(ValueError, match="Unsupported channel"):
        simulate_16qam_ber(channel="Invalid")


@pytest.mark.parametrize("num_bits", [0, 1, 2, 3])
def test_16qam_rejects_too_few_bits(num_bits):
    with pytest.raises(ValueError, match="at least 4"):
        simulate_16qam_ber(num_bits=num_bits)


def test_16qam_rejects_non_integer_bit_count():
    with pytest.raises(ValueError, match="integer"):
        simulate_16qam_ber(num_bits=1000.5)


def test_16qam_awgn_theoretical_ber_decreases_with_snr():
    ber = theoretical_ber_16qam_awgn([0, 5, 10, 15, 20])

    assert np.all(np.isfinite(ber))
    assert np.all(ber >= 0)
    assert np.all(ber <= 0.5)
    assert np.all(np.diff(ber) < 0)


def test_16qam_awgn_theoretical_ber_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        theoretical_ber_16qam_awgn([0, np.inf])
