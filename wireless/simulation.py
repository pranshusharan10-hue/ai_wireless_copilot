"""BPSK/QPSK simulation over AWGN, Rayleigh, and Rician channels."""

from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

Modulation = Literal["BPSK", "QPSK"]
Channel = Literal["AWGN", "Rayleigh", "Rician"]


def modulate(bits: NDArray[np.int_], modulation: Modulation) -> NDArray[np.complex128]:
    """Map bits to unit-energy BPSK or Gray-coded QPSK symbols."""
    bits = np.asarray(bits, dtype=np.int_)

    if bits.ndim != 1 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be a one-dimensional binary array")

    if modulation == "BPSK":
        return (1 - 2 * bits).astype(np.complex128)

    if modulation == "QPSK":
        if len(bits) % 2:
            raise ValueError("QPSK requires an even number of bits")

        pairs = bits.reshape(-1, 2)
        i_component = 1 - 2 * pairs[:, 0]
        q_component = 1 - 2 * pairs[:, 1]
        return ((i_component + 1j * q_component) / sqrt(2)).astype(np.complex128)

    raise ValueError(f"Unsupported modulation: {modulation}")


def add_awgn(
    symbols: NDArray[np.complex128],
    ebn0_db: float,
    bits_per_symbol: int,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """Add complex AWGN for a specified Eb/N0 in dB."""
    if bits_per_symbol not in (1, 2):
        raise ValueError("bits_per_symbol must be 1 or 2")
    if not np.isfinite(ebn0_db):
        raise ValueError("Eb/N0 must be finite")

    ebn0_linear = 10 ** (ebn0_db / 10)
    noise_variance_per_dimension = 1 / (2 * bits_per_symbol * ebn0_linear)

    noise = np.sqrt(noise_variance_per_dimension) * (
        rng.standard_normal(len(symbols)) + 1j * rng.standard_normal(len(symbols))
    )
    return symbols + noise


def apply_fading(
    symbols: NDArray[np.complex128],
    channel: Channel,
    rng: np.random.Generator,
    rician_k_db: float = 6.0,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Apply a normalized flat-fading channel and return (signal, channel)."""
    if channel == "AWGN":
        h = np.ones(len(symbols), dtype=np.complex128)
    elif channel == "Rayleigh":
        h = (
            rng.standard_normal(len(symbols)) + 1j * rng.standard_normal(len(symbols))
        ) / sqrt(2)
    elif channel == "Rician":
        if not np.isfinite(rician_k_db):
            raise ValueError("Rician K-factor must be finite")

        k_linear = 10 ** (rician_k_db / 10)
        scattered = (
            rng.standard_normal(len(symbols)) + 1j * rng.standard_normal(len(symbols))
        ) / sqrt(2)
        h = sqrt(k_linear / (k_linear + 1)) + sqrt(1 / (k_linear + 1)) * scattered
    else:
        raise ValueError(f"Unsupported channel: {channel}")

    return h * symbols, h.astype(np.complex128)


def demodulate(
    received: NDArray[np.complex128],
    modulation: Modulation,
) -> NDArray[np.int_]:
    """Recover bits using hard-decision detection."""
    if modulation == "BPSK":
        return (received.real < 0).astype(np.int_)

    if modulation == "QPSK":
        i_bits = (received.real < 0).astype(np.int_)
        q_bits = (received.imag < 0).astype(np.int_)
        return np.column_stack((i_bits, q_bits)).ravel()

    raise ValueError(f"Unsupported modulation: {modulation}")


def theoretical_ber(
    modulation: Modulation,
    ebn0_db: NDArray[np.float64] | list[float],
    channel: Channel = "AWGN",
    rician_k_db: float = 6.0,
) -> NDArray[np.float64]:
    """Calculate theoretical coherent BER for BPSK/QPSK."""
    if modulation not in ("BPSK", "QPSK"):
        raise ValueError(f"Unsupported modulation: {modulation}")
    if channel not in ("AWGN", "Rayleigh", "Rician"):
        raise ValueError(f"Unsupported channel: {channel}")

    snr_db = np.asarray(ebn0_db, dtype=float)
    if not np.all(np.isfinite(snr_db)):
        raise ValueError("Eb/N0 values must be finite")
    if channel == "Rician" and not np.isfinite(rician_k_db):
        raise ValueError("Rician K-factor must be finite")

    gamma_values = 10 ** (snr_db / 10)
    results: list[float] = []

    for gamma in gamma_values:
        if channel == "AWGN":
            ber = 0.5 * erfc(sqrt(gamma))
        elif channel == "Rayleigh":
            ber = 0.5 * (1 - sqrt(gamma / (1 + gamma)))
        else:
            # MGF method for coherent BPSK/QPSK over Rician fading.
            k_linear = 10 ** (rician_k_db / 10)

            def integrand(theta: float) -> float:
                sine = np.sin(theta)
                if sine < 1e-12:
                    return 0.0

                s = -gamma / (sine * sine)
                denominator = 1 + k_linear - s
                mgf = (1 + k_linear) / denominator * np.exp(k_linear * s / denominator)
                return float(mgf)

            ber = quad(integrand, 0, np.pi / 2, limit=150)[0] / np.pi

        results.append(float(np.clip(ber, 0.0, 0.5)))

    return np.asarray(results, dtype=float)


def simulate_ber(
    modulation: Modulation = "BPSK",
    ebn0_db: NDArray[np.float64] | list[float] | tuple[float, ...] = (
        -2.0,
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
    ),
    num_bits: int = 100_000,
    random_seed: int = 42,
    channel: Channel = "AWGN",
    rician_k_db: float = 6.0,
) -> dict[str, object]:
    """Simulate BER versus Eb/N0 with perfect receiver channel knowledge."""
    if modulation not in ("BPSK", "QPSK"):
        raise ValueError(f"Unsupported modulation: {modulation}")
    if channel not in ("AWGN", "Rayleigh", "Rician"):
        raise ValueError(f"Unsupported channel: {channel}")
    if num_bits <= 0:
        raise ValueError("num_bits must be positive")
    if channel == "Rician" and not np.isfinite(rician_k_db):
        raise ValueError("Rician K-factor must be finite")

    bits_per_symbol = 1 if modulation == "BPSK" else 2
    usable_bits = (num_bits // bits_per_symbol) * bits_per_symbol
    if usable_bits == 0:
        raise ValueError(f"num_bits must be at least {bits_per_symbol}")

    snr_values = np.asarray(ebn0_db, dtype=float)
    if snr_values.ndim != 1 or len(snr_values) == 0:
        raise ValueError("ebn0_db must be a non-empty one-dimensional list")
    if not np.all(np.isfinite(snr_values)):
        raise ValueError("Eb/N0 values must be finite")

    rng = np.random.default_rng(random_seed)
    transmitted_bits = rng.integers(0, 2, size=usable_bits, dtype=np.int_)
    transmitted_symbols = modulate(transmitted_bits, modulation)

    ber_values: list[float] = []
    error_counts: list[int] = []
    received_at_last_snr: NDArray[np.complex128] | None = None

    for snr_db in snr_values:
        faded_signal, h = apply_fading(transmitted_symbols, channel, rng, rician_k_db)
        received = add_awgn(faded_signal, float(snr_db), bits_per_symbol, rng)

        # Perfect channel knowledge: compensate for the channel before decoding.
        equalized = received * np.conj(h) / np.maximum(np.abs(h) ** 2, 1e-12)

        detected_bits = demodulate(equalized, modulation)
        errors = int(np.count_nonzero(transmitted_bits != detected_bits))
        error_counts.append(errors)
        ber_values.append(errors / usable_bits)
        received_at_last_snr = equalized

    return {
        "modulation": modulation,
        "channel": channel,
        "rician_k_db": rician_k_db if channel == "Rician" else None,
        "receiver_equalization": channel != "AWGN",
        "metric": "Eb/N0 (dB)",
        "ebn0_db": snr_values.tolist(),
        "ber_simulated": ber_values,
        "ber_theoretical": theoretical_ber(
            modulation, snr_values, channel, rician_k_db
        ).tolist(),
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


def plot_ber(
    result: dict[str, object],
    output_path: str | Path = "experiments/ber_curve.png",
) -> Path:
    """Save simulated and theoretical BER curves to a PNG file."""
    snr_values = np.asarray(result["ebn0_db"], dtype=float)
    simulated = np.asarray(result["ber_simulated"], dtype=float)
    theoretical = np.asarray(result["ber_theoretical"], dtype=float)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_simulated = np.where(simulated == 0, np.nan, simulated)

    plt.figure(figsize=(8, 5))
    plt.semilogy(snr_values, plot_simulated, "o-", label="Simulated BER")
    plt.semilogy(snr_values, theoretical, "s--", label="Theoretical BER")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"{result['modulation']} over {result['channel']}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    return output
