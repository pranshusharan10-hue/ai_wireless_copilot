# Test the LLM connection

from llm.client import extract_simulation_config
from llm.validator import validate_simulation_config

prompt = (
    "Simulate BPSK over Rayleigh fading from 0 to 20 dB "
    "in steps of 2 dB using 10000 bits."
)

# Step A: Ask the LLM to extract settings.
config = extract_simulation_config(prompt)
print("LLM output:")
print(config)

# Step B: Validate the extracted settings.
validated_request = validate_simulation_config(prompt, config)

print("\nValidated configuration:")
print(validated_request.model_dump())
print("\nValidation successful!")
