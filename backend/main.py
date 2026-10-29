# It preserves your three existing endpoints and adds the new copilot endpoint.
"""FastAPI endpoints for wireless simulation and AI copilot."""

import numpy as np
from fastapi import FastAPI, HTTPException

from backend.schemas import CopilotRequest, SimulationRequest
from llm.client import extract_simulation_config
from llm.validator import validate_simulation_config
from rag.retriever import retrieve
from wireless.simulation import Modulation as WirelessModulation
from wireless.simulation import simulate_ber

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

    try:
        # 1. Retrieve wireless knowledge relevant to the user's prompt.
        knowledge = retrieve(request.prompt, top_k=3)

        context = "\n\n".join(
            f"Source: {item['source']}\n{item['text']}" for item in knowledge
        )

        # 2. Ask the LLM to extract the simulation settings.
        config = extract_simulation_config(
            request.prompt,
            context=context,
        )

        # 3. Validate the model's output before execution.
        validated = validate_simulation_config(
            request.prompt,
            config,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # 4. Run only the trusted, existing simulation implementation.
    try:
        simulation = execute_simulation(validated)

        if not request.include_constellation:
            simulation.pop("constellation_received", None)

    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {exc}",
        ) from exc

    # 5. Return the configuration, results, and retrieved references.
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
