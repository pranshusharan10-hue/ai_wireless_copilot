# Test that invalid configurations are rejected
import pytest

from llm.validator import validate_simulation_config

VALID_PROMPT = "Simulate BPSK over Rayleigh fading."


VALID_CONFIG = {
    "modulation": "BPSK",
    "channel": "Rayleigh",
    "snr_start_db": 0,
    "snr_stop_db": 20,
    "snr_step_db": 2,
    "num_bits": 10000,
}


def test_valid_configuration_is_accepted():
    result = validate_simulation_config(
        VALID_PROMPT,
        VALID_CONFIG,
    )

    assert result.modulation == "BPSK"
    assert result.channel == "Rayleigh"
    assert result.num_bits == 10000


def test_unsupported_modulation_is_rejected():
    config = {
        **VALID_CONFIG,
        "modulation": "16QAM",
    }

    with pytest.raises(ValueError, match="not supported"):
        validate_simulation_config(VALID_PROMPT, config)


def test_invalid_snr_range_is_rejected():
    config = {
        **VALID_CONFIG,
        "snr_start_db": 20,
        "snr_stop_db": 0,
    }

    with pytest.raises(ValueError, match="Invalid simulation"):
        validate_simulation_config(VALID_PROMPT, config)


def test_invalid_channel_is_rejected():
    config = {
        **VALID_CONFIG,
        "channel": "UnknownChannel",
    }

    with pytest.raises(ValueError, match="Invalid simulation"):
        validate_simulation_config(VALID_PROMPT, config)
