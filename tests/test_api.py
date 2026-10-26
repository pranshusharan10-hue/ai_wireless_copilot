# existing phase 0 tests
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_valid_simulation_request():
    response = client.post(
        "/api/v1/simulations/validate",
        json={
            "prompt": "Simulate BPSK over AWGN",
            "modulation": "BPSK",
            "channel": "AWGN",
            "snr_start_db": 0,
            "snr_stop_db": 20,
            "snr_step_db": 2,
            "num_bits": 10000,
            "random_seed": 42,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "request_validated"


def test_invalid_snr_range():
    response = client.post(
        "/api/v1/simulations/validate",
        json={
            "prompt": "Invalid simulation",
            "snr_start_db": 20,
            "snr_stop_db": 0,
        },
    )

    assert response.status_code == 422
