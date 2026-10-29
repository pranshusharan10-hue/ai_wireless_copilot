# the validation module
"""Validate LLM-generated wireless simulation configurations."""

from pydantic import ValidationError

from backend.schemas import SimulationRequest

SUPPORTED_MODULATIONS = {"BPSK", "QPSK"}


def validate_simulation_config(
    prompt: str,
    config: dict,
) -> SimulationRequest:
    """Validate extracted settings against our API schema."""

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")

    # Keep the original user prompt and validate all settings
    # using the same schema as the API.
    request_data = {
        "prompt": prompt,
        **config,
    }

    try:
        request = SimulationRequest.model_validate(request_data)
    except ValidationError as exc:
        raise ValueError(f"Invalid simulation configuration:\n{exc}") from exc

    # Our simulator currently implements BPSK and QPSK only.
    if request.modulation not in SUPPORTED_MODULATIONS:
        raise ValueError(
            f"{request.modulation} is not supported by the simulator. "
            "Currently supported: BPSK and QPSK."
        )

    return request
