import numpy as np
from fastapi import FastAPI, HTTPException

from backend.schemas import SimulationRequest
from wireless.simulation import simulate_ber

app = FastAPI(
    title="AI-Wireless Copilot API",
    version="0.1.0",
    description="LLM-assisted wireless simulation and optimization",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "phase": 0}


@app.post("/api/v1/simulations/validate")
def validate_simulation_request(request: SimulationRequest):
    return {
        "status": "request_validated",
        "simulation": request.model_dump(),
        "message": "Simulation execution will be added in a later phase.",
    }


@app.post("/api/v1/simulations/run")
def run_simulation(request: SimulationRequest):
    # The current engine supports BPSK/QPSK over AWGN only.
    if request.modulation not in ("BPSK", "QPSK"):
        raise HTTPException(
            status_code=422,
            detail="Phase 1 currently supports BPSK and QPSK only.",
        )

    if request.channel != "AWGN":
        raise HTTPException(
            status_code=422,
            detail="Phase 1 currently supports the AWGN channel only.",
        )

    # Generate an inclusive SNR range.
    snr_values = np.arange(
        request.snr_start_db,
        request.snr_stop_db + request.snr_step_db * 0.5,
        request.snr_step_db,
    )
    snr_values = np.round(snr_values, 6)

    result = simulate_ber(
        modulation=request.modulation,
        ebn0_db=snr_values.tolist(),
        num_bits=request.num_bits,
        random_seed=request.random_seed,
    )

    return result
