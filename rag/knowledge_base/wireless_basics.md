#first knowledge doc.,starter content

# BPSK and QPSK Modulation

BPSK (Binary Phase Shift Keying) transmits one bit per symbol.
QPSK (Quadrature Phase Shift Keying) transmits two bits per symbol.

For coherent BPSK over an AWGN channel, the theoretical bit error
rate is:

BER = Q(sqrt(2 * Eb/N0))

Equivalently:

BER = 0.5 * erfc(sqrt(Eb/N0))

Here Eb/N0 is the energy per bit to noise power spectral density ratio
expressed as a linear ratio, not in decibels.

# AWGN Channel

AWGN stands for Additive White Gaussian Noise.
It models noise added to the transmitted signal.

In an AWGN channel, the received signal is the transmitted signal
plus Gaussian noise. There is no fading coefficient in this model.

# Rayleigh Fading Channel

Rayleigh fading models a wireless channel with multiple propagation
paths and no dominant line-of-sight component under the usual
Rayleigh model assumptions.

For coherent BPSK with ideal channel knowledge, the average BER is:

BER = 0.5 * (1 - sqrt(gamma / (1 + gamma)))

Here gamma is Eb/N0 in linear units.

# Rician Fading Channel

Rician fading models a channel with a dominant component, such as
a line-of-sight path, together with scattered components.

The Rician K-factor describes the power ratio of the dominant
component to the scattered components.

A larger K-factor indicates a stronger dominant component relative
to scattered power.

# Signal-to-Noise Ratio

SNR measures signal power relative to noise power.
In decibels:

SNR_dB = 10 * log10(SNR_linear)

For BER calculations, Eb/N0 must be distinguished from symbol SNR
when the modulation and energy conventions require a conversion.
