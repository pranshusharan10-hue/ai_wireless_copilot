import numpy as np
from fastapi import FastAPI, HTTPException

from backend.schemas import SimulationRequest
from wireless.simulation import simulate_ber

app = FastAPI(
    title="AI-Wireless Copilot API",
    version="0.2.0",  # earlier 0.1.0
    description="LLM-assisted wireless simulation and optimization",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "phase": 2}


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
            detail="Supported modulation schemes are BPSK and QPSK.",
        )

    # Include SNR points up to the requested stop value without overshooting.
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
        modulation=request.modulation,
        channel=request.channel,
        ebn0_db=snr_values.tolist(),
        num_bits=request.num_bits,
        random_seed=request.random_seed,
        rician_k_db=request.rician_k_db,
    )


# after phase 2
# endpoint now accepts all three channels.
# 16QAM and 64QAM remain unsupported and will still return HTTP 422.
