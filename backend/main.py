from fastapi import FastAPI

from backend.schemas import SimulationRequest

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
