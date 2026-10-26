# existing phase 0 tests
import pytest
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


# tests below:-
# A successful BPSK simulation returning HTTP 200.
# A successful QPSK simulation returning HTTP 200.
# Unsupported modulation returning HTTP 422.
# Unsupported channel returning HTTP 422.
# The response containing JSON-serializable constellation samples.


@pytest.mark.parametrize("modulation", ["BPSK", "QPSK"])
def test_run_simulation_success(modulation):
    payload = {
        "prompt": f"Simulate {modulation} over AWGN",
        "modulation": modulation,
        "channel": "AWGN",
        "snr_start_db": 0,
        "snr_stop_db": 4,
        "snr_step_db": 2,
        "num_bits": 10000,
        "random_seed": 42,
    }

    response = client.post("/api/v1/simulations/run", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["modulation"] == modulation
    assert data["channel"] == "AWGN"
    assert len(data["ebn0_db"]) == 3
    assert len(data["ber_simulated"]) == 3
    assert len(data["ber_theoretical"]) == 3
    assert len(data["constellation_received"]) > 0

    # Every complex sample must be returned as [real, imaginary].
    for symbol in data["constellation_received"]:
        assert len(symbol) == 2
        assert all(isinstance(value, (int, float)) for value in symbol)


def test_run_simulation_rejects_unsupported_modulation():
    payload = {
        "prompt": "Simulate 16QAM over AWGN",
        "modulation": "16QAM",
        "channel": "AWGN",
        "snr_start_db": 0,
        "snr_stop_db": 4,
        "snr_step_db": 2,
        "num_bits": 10000,
        "random_seed": 42,
    }

    response = client.post("/api/v1/simulations/run", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("channel", ["AWGN", "Rayleigh", "Rician"])
def test_run_simulation_accepts_supported_channels(channel):
    payload = {
        "prompt": f"Simulate BPSK over {channel}",
        "modulation": "BPSK",
        "channel": channel,
        "snr_start_db": 0,
        "snr_stop_db": 4,
        "snr_step_db": 2,
        "num_bits": 10_000,
        "random_seed": 42,
    }

    response = client.post("/api/v1/simulations/run", json=payload)

    assert response.status_code == 200
    assert response.json()["channel"] == channel


# phase 2 tests


@pytest.mark.parametrize("channel", ["AWGN", "Rayleigh", "Rician"])
def test_run_simulation_supports_phase2_channels(channel):
    response = client.post(
        "/api/v1/simulations/run",
        json={
            "prompt": f"Simulate BPSK over {channel}",
            "modulation": "BPSK",
            "channel": channel,
            "snr_start_db": 0,
            "snr_stop_db": 4,
            "snr_step_db": 2,
            "num_bits": 10_000,
            "random_seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["channel"] == channel
    assert len(data["ber_simulated"]) == 3
    assert len(data["ber_theoretical"]) == 3
    assert all(
        isinstance(point, list) and len(point) == 2
        for point in data["constellation_received"]
    )


def test_run_simulation_rejects_16qam():
    response = client.post(
        "/api/v1/simulations/run",
        json={
            "prompt": "Test 16QAM",
            "modulation": "16QAM",
            "channel": "AWGN",
            "num_bits": 10_000,
        },
    )

    assert response.status_code == 422
