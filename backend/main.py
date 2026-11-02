# It preserves your three existing endpoints and adds the new copilot endpoint.
"""FastAPI endpoints for wireless simulation and AI copilot."""

import logging

import numpy as np
from fastapi import FastAPI, HTTPException

from backend.schemas import CopilotRequest, SimulationRequest
from llm.client import (
    LLMResponseError,
    LLMServiceError,
    extract_simulation_config,
)
from llm.validator import validate_simulation_config
from rag.retriever import retrieve
from wireless.simulation import Modulation as WirelessModulation
from wireless.simulation import simulate_ber

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-Wireless Copilot API",
    version="0.3.0",
    description="LLM-assisted wireless simulation and optimization",
)


def execute_simulation(request: SimulationRequest) -> dict:
    """Run a validated simulation using the trusted Python engine."""

    if request.modulation == "BPSK":
        modulation: WirelessModulation = "BPSK"
    elif request.modulation == "QPSK":
        modulation = "QPSK"
    else:
        raise ValueError("Supported modulation schemes are BPSK and QPSK.")

    count = (
        int(
            np.floor(
                (request.snr_stop_db - request.snr_start_db) / request.snr_step_db
                + 1e-9
            )
        )
        + 1
    )

    snr_values = np.round(
        request.snr_start_db + np.arange(count) * request.snr_step_db,
        6,
    )

    return simulate_ber(
        modulation=modulation,
        channel=request.channel,
        ebn0_db=snr_values.tolist(),
        num_bits=request.num_bits,
        random_seed=request.random_seed,
        rician_k_db=request.rician_k_db,
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "phase": 3}


@app.post("/api/v1/simulations/validate")
def validate_simulation_request(request: SimulationRequest):
    return {
        "status": "request_validated",
        "simulation": request.model_dump(),
        "message": "Simulation request validated successfully.",
    }


@app.post("/api/v1/simulations/run")
def run_simulation(request: SimulationRequest):
    if request.modulation not in ("BPSK", "QPSK"):
        raise HTTPException(
            status_code=422,
            detail="Supported modulation schemes are BPSK and QPSK.",
        )

    return execute_simulation(request)


@app.post("/api/v1/copilot/run")
def run_copilot(request: CopilotRequest):
    """Interpret a prompt, retrieve knowledge, validate, and simulate."""

    # 1. Retrieve wireless knowledge.
    try:
        knowledge = retrieve(request.prompt, top_k=3)

    except RuntimeError as exc:
        logger.exception("RAG retrieval failed")
        raise HTTPException(
            status_code=503,
            detail="Wireless knowledge retrieval is currently unavailable.",
        ) from exc

    # 2. Extract configuration using the LLM.
    context = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}" for item in knowledge
    )

    try:
        try:
            config = extract_simulation_config(
                request.prompt,
                context=context,
            )

        except LLMResponseError as first_error:
            logger.warning(
                "Malformed LLM response. Attempting one bounded retry: %s",
                first_error,
            )

            config = extract_simulation_config(
                request.prompt,
                context=context,
                retry_feedback=(
                    "Your previous response was malformed or incomplete. "
                    "Return valid JSON with all seven required configuration "
                    "fields. Preserve every explicitly requested parameter "
                    "exactly, including invalid values."
                ),
            )

    except LLMResponseError as exc:
        logger.exception("LLM response remained invalid after retry")
        raise HTTPException(
            status_code=502,
            detail="The AI service returned an invalid configuration response.",
        ) from exc

    except LLMServiceError as exc:
        logger.exception("LLM configuration service failed")
        raise HTTPException(
            status_code=503,
            detail="The AI configuration service is currently unavailable.",
        ) from exc

    # 3. Validate the extracted configuration.
    try:
        validated = validate_simulation_config(
            request.prompt,
            config,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # 4. Execute only the trusted simulation implementation.
    try:
        simulation = execute_simulation(validated)

        if not request.include_constellation:
            simulation.pop("constellation_received", None)

    except Exception as exc:
        logger.exception("Wireless simulation execution failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while running the simulation.",
        ) from exc

    # 5. Return configuration, results, and retrieved references.
    return {
        "status": "success",
        "configuration": validated.model_dump(),
        "simulation": simulation,
        "retrieved_knowledge": knowledge,
        "explanation": (
            f"Ran {validated.modulation} over {validated.channel} "
            f"from {validated.snr_start_db} to "
            f"{validated.snr_stop_db} dB."
        ),
    }
