"""Gray-coded, unit-average-energy 16-QAM modulation and demodulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Two-bit Gray-coded mapping:
# 00 -> +3, 01 -> +1, 11 -> -1, 10 -> -3
_GRAY_LEVELS = np.array([3.0, 1.0, -3.0, -1.0])
_GRAY_BITS = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    dtype=np.int_,
)

_NORMALIZATION = np.sqrt(10.0)


def _validate_bits(bits: NDArray[np.int_] | list[int]) -> NDArray[np.int_]:
    """Validate a one-dimensional binary input."""
    values = np.asarray(bits)

    if values.ndim != 1:
        raise ValueError("bits must be a one-dimensional array")

    if not np.all((values == 0) | (values == 1)):
        raise ValueError("bits must contain only 0 and 1")

    return values.astype(np.int_)


def modulate_16qam(
    bits: NDArray[np.int_] | list[int],
) -> NDArray[np.complex128]:
    """Map groups of four bits to normalized Gray-coded 16-QAM symbols."""
    binary_bits = _validate_bits(bits)

    if len(binary_bits) % 4 != 0:
        raise ValueError("16-QAM requires a number of bits divisible by 4")

    if len(binary_bits) == 0:
        return np.array([], dtype=np.complex128)

    bit_groups = binary_bits.reshape(-1, 4)

    # Each axis uses two bits.
    i_indices = bit_groups[:, 0] * 2 + bit_groups[:, 1]
    q_indices = bit_groups[:, 2] * 2 + bit_groups[:, 3]

    i_values = _GRAY_LEVELS[i_indices]
    q_values = _GRAY_LEVELS[q_indices]

    symbols = (i_values + 1j * q_values) / _NORMALIZATION

    return symbols.astype(np.complex128)


def demodulate_16qam(
    symbols: NDArray[np.complex128] | list[complex],
) -> NDArray[np.int_]:
    """Recover bits using nearest-level Gray-coded 16-QAM detection."""
    received = np.asarray(symbols)

    if received.ndim != 1:
        raise ValueError("symbols must be a one-dimensional array")

    if not np.issubdtype(received.dtype, np.number):
        raise ValueError("symbols must contain numeric values")

    if not np.all(np.isfinite(received)):
        raise ValueError("symbols must contain only finite values")

    if len(received) == 0:
        return np.array([], dtype=np.int_)

    received = received.astype(np.complex128)

    # Undo average-energy normalization.
    i_values = received.real * _NORMALIZATION
    q_values = received.imag * _NORMALIZATION

    # Find the nearest permitted amplitude on each axis.
    i_indices = np.argmin(
        np.abs(i_values[:, np.newaxis] - _GRAY_LEVELS),
        axis=1,
    )
    q_indices = np.argmin(
        np.abs(q_values[:, np.newaxis] - _GRAY_LEVELS),
        axis=1,
    )

    i_bits = _GRAY_BITS[i_indices]
    q_bits = _GRAY_BITS[q_indices]

    # Restore original order: I1, I2, Q1, Q2.
    recovered = np.column_stack((i_bits, q_bits)).reshape(-1)

    return recovered.astype(np.int_)
