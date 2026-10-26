# new phase 1 tests

import numpy as np
import pytest

from wireless.simulation import (
    add_awgn,
    demodulate,
    modulate,
    plot_ber,
    simulate_ber,
    theoretical_ber,
)


def test_bpsk_modulation_and_demodulation_without_noise():
    bits = np.array([0, 1, 1, 0], dtype=np.int_)
    symbols = modulate(bits, "BPSK")

    assert np.allclose(symbols, [1, -1, -1, 1])
    assert np.array_equal(demodulate(symbols, "BPSK"), bits)


def test_qpsk_modulation_and_demodulation_without_noise():
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.int_)
    symbols = modulate(bits, "QPSK")

    assert len(symbols) == len(bits) // 2
    assert np.allclose(np.abs(symbols), 1.0)
    assert np.array_equal(demodulate(symbols, "QPSK"), bits)


def test_qpsk_rejects_odd_number_of_bits():
    with pytest.raises(ValueError, match="even number"):
        modulate(np.array([0, 1, 0]), "QPSK")


def test_theoretical_ber_decreases_with_ebn0():
    ber = theoretical_ber("BPSK", [0, 5, 10])

    assert ber[0] > ber[1] > ber[2]
    assert ber[2] > 0


@pytest.mark.parametrize("modulation", ["BPSK", "QPSK"])
def test_simulation_output_is_reproducible(modulation):
    first = simulate_ber(
        modulation=modulation,
        ebn0_db=[0, 4, 8],
        num_bits=20_000,
        random_seed=123,
    )
    second = simulate_ber(
        modulation=modulation,
        ebn0_db=[0, 4, 8],
        num_bits=20_000,
        random_seed=123,
    )

    assert first["ber_simulated"] == second["ber_simulated"]
    assert first["error_counts"] == second["error_counts"]
    assert first["channel"] == "AWGN"


@pytest.mark.parametrize("modulation", ["BPSK", "QPSK"])
def test_simulated_ber_is_reasonably_close_to_theory(modulation):
    result = simulate_ber(
        modulation=modulation,
        ebn0_db=[0, 4, 8],
        num_bits=200_000,
        random_seed=7,
    )

    simulated = np.asarray(result["ber_simulated"], dtype=float)
    theoretical = np.asarray(result["ber_theoretical"], dtype=float)
    num_bits = result["num_bits"]
    assert isinstance(num_bits, int)
    # Statistical tolerance; this is not a bit-for-bit equality test.
    tolerance = (
        5
        * np.sqrt(
            np.maximum(
                theoretical * (1 - theoretical) / num_bits,
                1 / num_bits**2,
            )
        )
        + 0.001
    )

    assert np.all(np.abs(simulated - theoretical) < tolerance)


def test_plot_ber_creates_png(tmp_path):
    result = simulate_ber(
        modulation="BPSK",
        ebn0_db=[0, 4, 8],
        num_bits=20_000,
        random_seed=42,
    )

    output = plot_ber(result, tmp_path / "ber.png")

    assert output.exists()
    assert output.stat().st_size > 0


def test_awgn_rejects_invalid_bits_per_symbol():
    rng = np.random.default_rng(1)

    with pytest.raises(ValueError, match="bits_per_symbol"):
        add_awgn(np.array([1 + 0j]), 0, 3, rng)
