import numpy as np
import pytest

from wireless.ofdm import (
    demodulate,
    modulate,
    ofdm_demodulate,
    ofdm_modulate,
    simulate_ofdm_ber,
)


@pytest.mark.parametrize("modulation", ["QPSK", "16QAM"])
def test_modulation_noiseless_recovery(modulation):
    bits_per_symbol = 2 if modulation == "QPSK" else 4
    bits = np.random.default_rng(42).integers(0, 2, size=128 * bits_per_symbol)

    symbols = modulate(bits, modulation)
    recovered = demodulate(symbols, modulation)

    np.testing.assert_array_equal(recovered, bits)


def test_ifft_fft_recovery():
    rng = np.random.default_rng(42)
    symbols = rng.normal(size=64) + 1j * rng.normal(size=64)

    time_signal = ofdm_modulate(symbols, 64, 16)
    recovered = ofdm_demodulate(time_signal, 64, 16)

    np.testing.assert_allclose(recovered, symbols, atol=1e-10)


def test_cyclic_prefix_copies_last_samples():
    symbols = np.arange(64) + 1j * np.arange(64)
    time_signal = np.fft.ifft(symbols)

    transmitted = ofdm_modulate(symbols, 64, 16)

    np.testing.assert_allclose(transmitted[:16], time_signal[-16:])


def test_ofdm_output_length():
    symbols = np.ones(64, dtype=np.complex128)

    transmitted = ofdm_modulate(symbols, 64, 16)

    assert len(transmitted) == 80


def test_ofdm_rejects_invalid_cp():
    symbols = np.ones(64, dtype=np.complex128)

    with pytest.raises(ValueError):
        ofdm_modulate(symbols, 64, 64)


@pytest.mark.parametrize("modulation", ["QPSK", "16QAM"])
@pytest.mark.parametrize("channel", ["AWGN", "Rayleigh"])
def test_ofdm_simulation_returns_valid_result(modulation, channel):
    result = simulate_ofdm_ber(
        modulation=modulation,
        channel=channel,
        ebn0_db=[0, 4, 8],
        num_bits=64 * 4 * 20,
        n_subcarriers=64,
        cp_length=16,
        num_taps=4 if channel == "Rayleigh" else 1,
        random_seed=42,
    )

    assert result["modulation"] == modulation
    assert result["channel"] == channel
    assert len(result["ber_simulated"]) == 3
    assert len(result["error_counts"]) == 3
    assert all(0 <= ber <= 1 for ber in result["ber_simulated"])
    assert result["num_bits"] > 0


def test_ofdm_simulation_is_reproducible():
    kwargs = {
        "modulation": "QPSK",
        "channel": "Rayleigh",
        "ebn0_db": [0, 4, 8],
        "num_bits": 64 * 2 * 10,
        "n_subcarriers": 64,
        "cp_length": 16,
        "num_taps": 4,
        "random_seed": 42,
    }

    first = simulate_ofdm_ber(**kwargs)
    second = simulate_ofdm_ber(**kwargs)

    assert first["ber_simulated"] == second["ber_simulated"]
    assert first["error_counts"] == second["error_counts"]


def test_rayleigh_requires_sufficient_cyclic_prefix():
    with pytest.raises(ValueError):
        simulate_ofdm_ber(
            channel="Rayleigh",
            num_bits=64 * 2,
            n_subcarriers=64,
            cp_length=2,
            num_taps=4,
        )


def test_awgn_rejects_multiple_taps():
    with pytest.raises(ValueError):
        simulate_ofdm_ber(
            channel="AWGN",
            num_bits=64 * 2,
            n_subcarriers=64,
            cp_length=16,
            num_taps=4,
        )


def test_incomplete_ofdm_block_is_not_used():
    result = simulate_ofdm_ber(
        modulation="QPSK",
        channel="AWGN",
        ebn0_db=[0],
        num_bits=300,
        n_subcarriers=64,
        cp_length=16,
        num_taps=1,
    )

    assert result["num_bits"] == 256
