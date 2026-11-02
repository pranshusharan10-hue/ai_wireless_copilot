"""Validate LLM-generated wireless simulation configurations."""

from pydantic import ValidationError

from backend.schemas import SimulationRequest

SUPPORTED_MODULATIONS = {"BPSK", "QPSK"}

# Only fields belonging to the simulation configuration are accepted.
# The original user prompt is supplied separately by the application.
ALLOWED_CONFIG_FIELDS = set(SimulationRequest.model_fields) - {"prompt"}


def validate_simulation_config(
    prompt: str,
    config: dict,
) -> SimulationRequest:
    """Validate extracted settings against our API schema."""

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")

    unexpected_fields = set(config) - ALLOWED_CONFIG_FIELDS

    if unexpected_fields:
        fields = ", ".join(sorted(str(field) for field in unexpected_fields))
        raise ValueError(f"Unexpected configuration field(s): {fields}")

    # Reject unsupported modulation before Pydantic schema validation.
    # This ensures users receive a clear, consistent error message.
    modulation = config.get("modulation", "BPSK")

    if modulation not in SUPPORTED_MODULATIONS:
        raise ValueError(
            f"{modulation} is not supported by the simulator. "
            "Currently supported: BPSK and QPSK."
        )

    # The application always controls the original user prompt.
    # LLM-generated configuration cannot replace it.
    request_data = {
        **config,
        "prompt": prompt,
    }

    try:
        request = SimulationRequest.model_validate(request_data)
    except ValidationError as exc:
        raise ValueError(f"Invalid simulation configuration:\n{exc}") from exc

    return request
