"""OFDM simulation using QPSK or 16-QAM over AWGN and multipath Rayleigh."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

Modulation = Literal["QPSK", "16QAM"]
Channel = Literal["AWGN", "Rayleigh"]


def _validate_bits(bits: NDArray[np.int_]) -> NDArray[np.int_]:
    bits = np.asarray(bits)

    if bits.ndim != 1:
        raise ValueError("bits must be one-dimensional")

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only 0 and 1")

    return bits.astype(np.int_)


def modulate(bits: NDArray[np.int_], modulation: Modulation) -> NDArray[np.complex128]:
    """Map binary data to unit-average-energy QPSK or Gray-coded 16-QAM."""

    bits = _validate_bits(bits)

    if modulation == "QPSK":
        if len(bits) % 2:
            raise ValueError("QPSK requires a multiple of 2 bits")

        pairs = bits.reshape(-1, 2)
        i = 1 - 2 * pairs[:, 0]
        q = 1 - 2 * pairs[:, 1]

        return ((i + 1j * q) / np.sqrt(2)).astype(np.complex128)

    if modulation == "16QAM":
        if len(bits) % 4:
            raise ValueError("16QAM requires a multiple of 4 bits")

        groups = bits.reshape(-1, 4)

        # Gray mapping: 00 -> +3, 01 -> +1,
        #               11 -> -1, 10 -> -3.
        levels = np.array([3, 1, -3, -1], dtype=float)

        i_index = groups[:, 0] * 2 + groups[:, 1]
        q_index = groups[:, 2] * 2 + groups[:, 3]

        i = levels[i_index]
        q = levels[q_index]

        return ((i + 1j * q) / np.sqrt(10)).astype(np.complex128)

    raise ValueError(f"Unsupported modulation: {modulation}")


def demodulate(
    symbols: NDArray[np.complex128],
    modulation: Modulation,
) -> NDArray[np.int_]:
    """Hard-decision QPSK or Gray-coded 16-QAM demodulation."""

    symbols = np.asarray(symbols, dtype=np.complex128)

    if symbols.ndim != 1 or not np.all(np.isfinite(symbols)):
        raise ValueError("symbols must be a finite one-dimensional array")

    if modulation == "QPSK":
        bits = np.column_stack((symbols.real < 0, symbols.imag < 0)).astype(np.int_)

        return bits.ravel()

    if modulation == "16QAM":
        scaled = symbols * np.sqrt(10)

        def decode_axis(
            values: NDArray[np.float64],
        ) -> NDArray[np.int_]:
            result = np.zeros((len(values), 2), dtype=np.int_)

            # First bit: sign of the signal.
            result[:, 0] = values < 0

            # Second bit: 1 for inner levels, 0 for outer levels.
            result[:, 1] = np.abs(values) < 2

            return result

        i_bits = decode_axis(scaled.real)
        q_bits = decode_axis(scaled.imag)

        return np.column_stack((i_bits, q_bits)).ravel()

    raise ValueError(f"Unsupported modulation: {modulation}")


def create_channel(
    channel: Channel,
    num_taps: int,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """Create a normalized channel impulse response."""

    if num_taps < 1:
        raise ValueError("num_taps must be at least 1")

    if channel == "AWGN":
        if num_taps != 1:
            raise ValueError("AWGN requires exactly one channel tap")

        return np.array([1.0 + 0j], dtype=np.complex128)

    if channel == "Rayleigh":
        taps = (
            rng.standard_normal(num_taps) + 1j * rng.standard_normal(num_taps)
        ) / np.sqrt(2)

        return (taps / np.linalg.norm(taps)).astype(np.complex128)

    raise ValueError(f"Unsupported channel: {channel}")


def ofdm_modulate(
    symbols: NDArray[np.complex128],
    n_subcarriers: int = 64,
    cp_length: int = 16,
) -> NDArray[np.complex128]:
    """Convert frequency-domain symbols into a time-domain OFDM block."""

    symbols = np.asarray(symbols, dtype=np.complex128)

    if symbols.ndim != 1 or len(symbols) != n_subcarriers:
        raise ValueError("one symbol is required per subcarrier")

    if cp_length < 0 or cp_length >= n_subcarriers:
        raise ValueError("invalid cyclic prefix length")

    time_signal = np.fft.ifft(symbols)

    if cp_length:
        time_signal = np.concatenate((time_signal[-cp_length:], time_signal))

    return time_signal.astype(np.complex128)


def ofdm_demodulate(
    received: NDArray[np.complex128],
    n_subcarriers: int = 64,
    cp_length: int = 16,
) -> NDArray[np.complex128]:
    """Remove cyclic prefix and recover frequency-domain symbols."""

    received = np.asarray(received, dtype=np.complex128)

    expected_length = n_subcarriers + cp_length

    if received.ndim != 1 or len(received) != expected_length:
        raise ValueError("received OFDM block has an invalid length")

    useful_signal = received[cp_length:]

    return np.fft.fft(useful_signal).astype(np.complex128)


def simulate_ofdm_ber(
    modulation: Modulation = "QPSK",
    channel: Channel = "AWGN",
    ebn0_db: list[float] | tuple[float, ...] = (0, 2, 4, 6, 8, 10, 12, 14),
    num_bits: int = 100_000,
    n_subcarriers: int = 64,
    cp_length: int = 16,
    num_taps: int = 4,
    random_seed: int = 42,
) -> dict[str, object]:
    """Simulate OFDM BER with ideal frequency-domain channel knowledge."""

    if modulation not in ("QPSK", "16QAM"):
        raise ValueError("modulation must be QPSK or 16QAM")

    if channel not in ("AWGN", "Rayleigh"):
        raise ValueError("channel must be AWGN or Rayleigh")

    if n_subcarriers < 2:
        raise ValueError("n_subcarriers must be at least 2")

    if cp_length < 0 or cp_length >= n_subcarriers:
        raise ValueError("invalid cyclic prefix length")

    if num_taps < 1:
        raise ValueError("num_taps must be at least 1")

    if channel == "AWGN" and num_taps != 1:
        raise ValueError("AWGN requires num_taps=1")

    if channel == "Rayleigh" and cp_length < num_taps - 1:
        raise ValueError("cyclic prefix must be at least num_taps - 1")

    if not isinstance(num_bits, int) or num_bits <= 0:
        raise ValueError("num_bits must be a positive integer")

    bits_per_symbol = 2 if modulation == "QPSK" else 4
    bits_per_block = n_subcarriers * bits_per_symbol
    usable_bits = (num_bits // bits_per_block) * bits_per_block

    if usable_bits == 0:
        raise ValueError("num_bits must contain at least one complete OFDM block")

    snr_values = np.asarray(ebn0_db, dtype=float)

    if (
        snr_values.ndim != 1
        or len(snr_values) == 0
        or not np.all(np.isfinite(snr_values))
    ):
        raise ValueError("ebn0_db must be a non-empty finite one-dimensional list")

    rng = np.random.default_rng(random_seed)

    transmitted_bits = rng.integers(0, 2, size=usable_bits, dtype=np.int_)
    transmitted_symbols = modulate(transmitted_bits, modulation)
    symbol_blocks = transmitted_symbols.reshape(-1, n_subcarriers)

    ber_values: list[float] = []
    error_counts: list[int] = []
    last_received: NDArray[np.complex128] | None = None

    for snr_db in snr_values:
        gamma = 10 ** (float(snr_db) / 10)

        # Es = 1, Eb = Es / bits_per_symbol.
        # FFT-domain noise variance per complex sample = N0.
        n0 = 1 / (bits_per_symbol * gamma)

        total_errors = 0
        recovered_blocks: list[NDArray[np.complex128]] = []

        for block in symbol_blocks:
            h = create_channel(channel, num_taps, rng)

            time_signal = ofdm_modulate(block, n_subcarriers, cp_length)

            # Linear convolution is safe because the CP covers the channel.
            faded = np.convolve(time_signal, h)[: len(time_signal)]

            # Time-domain noise variance is N0 / Nfft.
            noise_scale = np.sqrt(n0 / (2 * n_subcarriers))

            noise = noise_scale * (
                rng.standard_normal(len(faded)) + 1j * rng.standard_normal(len(faded))
            )

            received = faded + noise

            frequency_received = ofdm_demodulate(received, n_subcarriers, cp_length)

            channel_frequency = np.fft.fft(h, n_subcarriers)

            if np.any(np.abs(channel_frequency) < 1e-12):
                raise ValueError("channel has a near-zero subcarrier response")

            equalized = frequency_received / channel_frequency

            detected_bits = demodulate(equalized, modulation)

            start = len(recovered_blocks) * bits_per_block
            original_block_bits = transmitted_bits[start : start + bits_per_block]

            total_errors += int(np.count_nonzero(original_block_bits != detected_bits))

            recovered_blocks.append(equalized)

        error_counts.append(total_errors)
        ber_values.append(total_errors / usable_bits)

        last_received = np.concatenate(recovered_blocks)

    return {
        "modulation": modulation,
        "channel": channel,
        "n_subcarriers": n_subcarriers,
        "cp_length": cp_length,
        "num_taps": num_taps,
        "receiver_equalization": channel == "Rayleigh",
        "perfect_channel_knowledge": channel == "Rayleigh",
        "metric": "Eb/N0 (dB)",
        "ebn0_db": snr_values.tolist(),
        "ber_simulated": ber_values,
        "error_counts": error_counts,
        "num_bits": usable_bits,
        "random_seed": random_seed,
        "constellation_received": (
            [[float(s.real), float(s.imag)] for s in last_received[:200]]
            if last_received is not None
            else []
        ),
        "theoretical_reference": None,
    }
