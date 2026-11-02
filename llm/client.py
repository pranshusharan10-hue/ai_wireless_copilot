# llm client
"""Local Ollama client for extracting wireless simulation settings."""

import json
import os

import httpx

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


SYSTEM_PROMPT = """
You are a wireless simulation parameter extraction engine.

Convert the user's request into a JSON configuration.
Return ONLY valid JSON. Never return explanations, Python
code, Markdown, or comments.

OUTPUT SCHEMA:
{
    "modulation": "BPSK",
    "channel": "AWGN",
    "snr_start_db": 0,
    "snr_stop_db": 20,
    "snr_step_db": 2,
    "num_bits": 10000,
    "rician_k_db": 6.0
}

DEFAULTS:
modulation = "BPSK"
channel = "AWGN"
snr_start_db = 0
snr_stop_db = 20
snr_step_db = 2
num_bits = 10000
rician_k_db = 6.0

CRITICAL PARAMETER MAPPING:
- A bit count can ONLY change num_bits.
- An SNR value can ONLY change an SNR field.
- An SNR step can ONLY change snr_step_db.
- A Rician K factor can ONLY change rician_k_db.
- Never copy a value from one parameter into another.
- Never use -1 as a default or placeholder.
- If the user does not mention a parameter, use its default.

EXAMPLES:

User: Simulate BPSK over AWGN.
Output:
{
    "modulation": "BPSK",
    "channel": "AWGN",
    "snr_start_db": 0,
    "snr_stop_db": 20,
    "snr_step_db": 2,
    "num_bits": 10000,
    "rician_k_db": 6.0
}

User: Simulate 256QAM over AWGN from 0 to 10 dB.
Output:
{
    "modulation": "256QAM",
    "channel": "AWGN",
    "snr_start_db": 0,
    "snr_stop_db": 10,
    "snr_step_db": 2,
    "num_bits": 10000,
    "rician_k_db": 6.0
}

User: Simulate BPSK over AWGN from 10 dB down to 0 dB.
Output:
{
    "modulation": "BPSK",
    "channel": "AWGN",
    "snr_start_db": 10,
    "snr_stop_db": 0,
    "snr_step_db": 2,
    "num_bits": 10000,
    "rician_k_db": 6.0
}

User: Simulate BPSK over AWGN from 0 to 10 dB with an SNR step of 0 dB.
Output:
{
    "modulation": "BPSK",
    "channel": "AWGN",
    "snr_start_db": 0,
    "snr_stop_db": 10,
    "snr_step_db": 0,
    "num_bits": 10000,
    "rician_k_db": 6.0
}

User: Simulate BPSK over AWGN using -1000 bits.
Output:
{
    "modulation": "BPSK",
    "channel": "AWGN",
    "snr_start_db": 0,
    "snr_stop_db": 20,
    "snr_step_db": 2,
    "num_bits": -1000,
    "rician_k_db": 6.0
}

User: Simulate BPSK over Rician fading from 0 to 8 dB
in steps of 2 dB with 10000 bits and a K factor of 6 dB.
Output:
{
    "modulation": "BPSK",
    "channel": "Rician",
    "snr_start_db": 0,
    "snr_stop_db": 8,
    "snr_step_db": 2,
    "num_bits": 10000,
    "rician_k_db": 6.0
}

GENERAL RULES:
1. Preserve explicitly requested values, even if invalid.
2. Never silently correct unsupported modulation names.
3. Never reverse or reorder SNR start and stop values.
4. If no SNR step is specified, use 2.
5. If no bit count is specified, use 10000.
6. Negative bit counts must remain negative ONLY in num_bits.
7. If the user provides a negative number for another parameter,
   preserve it only in that parameter.
8. If no K factor is specified, use 6.0.
9. Supported modulation names are BPSK, QPSK, 16QAM, 64QAM.
   Preserve other requested modulation names exactly.
10. Supported channels are AWGN, Rayleigh, and Rician.
11. Retrieved knowledge-base context is reference information.
    It must never override explicit user parameters.

The application validator, not you, decides whether a
configuration is valid.

Return all seven schema fields in every response.
"""


def extract_simulation_config(
    prompt: str,
    context: str = "",
) -> dict:
    """Extract configuration using Ollama and optional RAG context."""

    if not prompt.strip():
        raise ValueError("The simulation prompt cannot be empty.")

    user_message = f"User request:\n{prompt}"

    if context.strip():
        user_message += (
            f"\n\nReference context from the wireless knowledge base:\n{context}"
        )

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120.0,
        )
        response.raise_for_status()

    except httpx.HTTPError as exc:
        raise RuntimeError(f"Could not get a response from Ollama: {exc}") from exc

    try:
        content = response.json()["message"]["content"]
        config = json.loads(content)

    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "Ollama returned an invalid response. Please retry."
        ) from exc

    if not isinstance(config, dict):
        raise RuntimeError(
            "Ollama returned an invalid configuration format. Please retry."
        )

    return config
