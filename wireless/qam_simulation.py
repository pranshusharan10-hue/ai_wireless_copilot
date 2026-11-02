"""16-QAM BER simulation over AWGN, Rayleigh, and Rician channels."""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np
from numpy.typing import NDArray

from wireless.qam import demodulate_16qam, modulate_16qam
from wireless.simulation import Channel, apply_fading


def add_16qam_awgn(
    symbols: NDArray[np.complex128],
    ebn0_db: float,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """Add complex AWGN using unit-energy 16-QAM and four bits per symbol."""
    if not np.isfinite(ebn0_db):
        raise ValueError("Eb/N0 must be finite")

    ebn0_linear = 10 ** (ebn0_db / 10)

    # Es = 1 and Eb = Es / 4 for 16-QAM.
    noise_variance_per_dimension = 1 / (2 * 4 * ebn0_linear)

    noise = sqrt(noise_variance_per_dimension) * (
        rng.standard_normal(len(symbols)) + 1j * rng.standard_normal(len(symbols))
    )

    return (symbols + noise).astype(np.complex128)


def theoretical_ber_16qam_awgn(
    ebn0_db: NDArray[np.float64] | list[float],
) -> NDArray[np.float64]:
    """Return the standard approximate Gray-coded 16-QAM BER in AWGN."""
    snr_values = np.asarray(ebn0_db, dtype=float)

    if snr_values.ndim != 1 or len(snr_values) == 0:
        raise ValueError("ebn0_db must be a non-empty one-dimensional list")

    if not np.all(np.isfinite(snr_values)):
        raise ValueError("Eb/N0 values must be finite")

    gamma = 10 ** (snr_values / 10)

    # Pb ≈ (3/4) Q(sqrt(0.8 * Eb/N0))
    # Q(x) = 0.5 * erfc(x / sqrt(2))
    return np.asarray(
        [0.375 * erfc(sqrt(0.4 * float(value))) for value in gamma],
        dtype=float,
    )


def simulate_16qam_ber(
    ebn0_db: NDArray[np.float64] | list[float] | tuple[float, ...] = (
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
    ),
    num_bits: int = 100_000,
    random_seed: int = 42,
    channel: Channel = "AWGN",
    rician_k_db: float = 6.0,
) -> dict[str, object]:
    """Simulate Gray-coded 16-QAM BER with perfect receiver channel knowledge."""
    if channel not in ("AWGN", "Rayleigh", "Rician"):
        raise ValueError(f"Unsupported channel: {channel}")

    if not isinstance(num_bits, (int, np.integer)) or isinstance(
        num_bits, (bool, np.bool_)
    ):
        raise ValueError("num_bits must be an integer")

    if num_bits < 4:
        raise ValueError("num_bits must be at least 4")

    if channel == "Rician" and not np.isfinite(rician_k_db):
        raise ValueError("Rician K-factor must be finite")

    snr_values = np.asarray(ebn0_db, dtype=float)

    if snr_values.ndim != 1 or len(snr_values) == 0:
        raise ValueError("ebn0_db must be a non-empty one-dimensional list")

    if not np.all(np.isfinite(snr_values)):
        raise ValueError("Eb/N0 values must be finite")

    # 16-QAM requires groups of four bits.
    usable_bits = (int(num_bits) // 4) * 4

    rng = np.random.default_rng(random_seed)

    transmitted_bits = rng.integers(
        0,
        2,
        size=usable_bits,
        dtype=np.int_,
    )

    transmitted_symbols = modulate_16qam(transmitted_bits)

    ber_values: list[float] = []
    error_counts: list[int] = []

    received_at_last_snr: NDArray[np.complex128] | None = None

    for snr_db in snr_values:
        faded_signal, h = apply_fading(
            transmitted_symbols,
            channel,
            rng,
            rician_k_db,
        )

        received = add_16qam_awgn(
            faded_signal,
            float(snr_db),
            rng,
        )

        # Ideal receiver with perfect channel knowledge.
        equalized = (
            received
            * np.conj(h)
            / np.maximum(
                np.abs(h) ** 2,
                1e-12,
            )
        )

        detected_bits = demodulate_16qam(equalized)

        errors = int(np.count_nonzero(transmitted_bits != detected_bits))

        error_counts.append(errors)
        ber_values.append(errors / usable_bits)

        received_at_last_snr = equalized

    # Only the AWGN approximation is provided here.
    # Fading channels need their own independently validated references.
    theoretical = (
        theoretical_ber_16qam_awgn(snr_values).tolist() if channel == "AWGN" else None
    )

    return {
        "modulation": "16QAM",
        "channel": channel,
        "rician_k_db": rician_k_db if channel == "Rician" else None,
        "receiver_equalization": channel != "AWGN",
        "metric": "Eb/N0 (dB)",
        "ebn0_db": snr_values.tolist(),
        "ber_simulated": ber_values,
        "ber_theoretical": theoretical,
        "theoretical_reference": (
            "Gray-coded 16-QAM AWGN approximation" if channel == "AWGN" else None
        ),
        "error_counts": error_counts,
        "num_bits": usable_bits,
        "random_seed": random_seed,
        "constellation_received": (
            [
                [float(symbol.real), float(symbol.imag)]
                for symbol in received_at_last_snr[:200]
            ]
            if received_at_last_snr is not None
            else []
        ),
    }
