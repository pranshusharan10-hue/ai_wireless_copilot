# phase 1 simulation engine

from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

Modulation = Literal["BPSK", "QPSK"]


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
) -> NDArray[np.float64]:
    """Theoretical BER for coherent BPSK/QPSK over AWGN."""
    if modulation not in ("BPSK", "QPSK"):
        raise ValueError(f"Unsupported modulation: {modulation}")

    ebn0_db_array = np.asarray(ebn0_db, dtype=float)

    if not np.all(np.isfinite(ebn0_db_array)):
        raise ValueError("Eb/N0 values must be finite")

    ebn0_linear = 10 ** (ebn0_db_array / 10)
    return np.array(
        [0.5 * erfc(sqrt(value)) for value in ebn0_linear],
        dtype=float,
    )


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
) -> dict[str, object]:
    """Simulate BER versus Eb/N0 for BPSK/QPSK over AWGN."""
    if modulation not in ("BPSK", "QPSK"):
        raise ValueError(f"Unsupported modulation: {modulation}")

    if num_bits <= 0:
        raise ValueError("num_bits must be positive")

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
        received = add_awgn(
            transmitted_symbols,
            float(snr_db),
            bits_per_symbol,
            rng,
        )
        detected_bits = demodulate(received, modulation)
        errors = int(np.count_nonzero(transmitted_bits != detected_bits))

        error_counts.append(errors)
        ber_values.append(errors / usable_bits)
        received_at_last_snr = received

    return {
        "modulation": modulation,
        "channel": "AWGN",
        "metric": "Eb/N0 (dB)",
        "ebn0_db": snr_values.tolist(),
        "ber_simulated": ber_values,
        "ber_theoretical": theoretical_ber(modulation, snr_values).tolist(),
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

    # BER can be zero in finite simulations, which cannot be shown on
    # a logarithmic axis. Replace zeros only for plotting, not in results.
    plot_simulated = np.where(simulated == 0, np.nan, simulated)

    plt.figure(figsize=(8, 5))
    plt.semilogy(
        snr_values,
        plot_simulated,
        "o-",
        label="Simulated BER",
    )
    plt.semilogy(
        snr_values,
        theoretical,
        "s--",
        label="Theoretical BER",
    )
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"{result['modulation']} over AWGN")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

    return output
