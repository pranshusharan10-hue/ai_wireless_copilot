# integration tests
import pytest
from fastapi.testclient import TestClient

import backend.main as main
from backend.schemas import SimulationRequest

client = TestClient(main.app)


def fake_config(prompt, context=""):
    return {
        "modulation": "BPSK",
        "channel": "Rayleigh",
        "snr_start_db": 0,
        "snr_stop_db": 4,
        "snr_step_db": 2,
        "num_bits": 1000,
    }


def fake_knowledge(prompt, top_k=3):
    return [
        {
            "source": "wireless_basics.md",
            "text": "# Rayleigh Fading Channel",
            "score": 0.9,
        }
    ]


def test_copilot_runs_validated_simulation(monkeypatch):
    monkeypatch.setattr(
        main,
        "retrieve",
        fake_knowledge,
    )
    monkeypatch.setattr(
        main,
        "extract_simulation_config",
        fake_config,
    )

    def fake_simulation(request):
        assert isinstance(request, SimulationRequest)
        return {"modulation": request.modulation, "ber": [0.1, 0.01, 0.001]}

    monkeypatch.setattr(main, "execute_simulation", fake_simulation)

    response = client.post(
        "/api/v1/copilot/run",
        json={"prompt": "Simulate BPSK over Rayleigh fading"},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "success"
    assert body["configuration"]["channel"] == "Rayleigh"
    assert body["simulation"]["modulation"] == "BPSK"
    assert body["retrieved_knowledge"][0]["source"] == "wireless_basics.md"


def test_copilot_rejects_unsupported_modulation(monkeypatch):
    monkeypatch.setattr(main, "retrieve", fake_knowledge)

    monkeypatch.setattr(
        main,
        "extract_simulation_config",
        lambda prompt, context="": {
            **fake_config(prompt),
            "modulation": "16QAM",
        },
    )

    response = client.post(
        "/api/v1/copilot/run",
        json={"prompt": "Simulate 16QAM over Rayleigh fading"},
    )

    assert response.status_code == 422
    assert "not supported" in response.json()["detail"]


def test_copilot_handles_ollama_failure(monkeypatch):
    monkeypatch.setattr(main, "retrieve", fake_knowledge)

    def fail_llm(prompt, context=""):
        raise RuntimeError("Ollama is unavailable")

    monkeypatch.setattr(main, "extract_simulation_config", fail_llm)

    response = client.post(
        "/api/v1/copilot/run",
        json={"prompt": "Simulate BPSK over AWGN"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The AI configuration service is currently unavailable."
    )


def test_copilot_rejects_empty_prompt():
    response = client.post(
        "/api/v1/copilot/run",
        json={"prompt": ""},
    )

    assert response.status_code == 422


def test_copilot_omits_constellation_by_default(monkeypatch):
    monkeypatch.setattr(main, "retrieve", fake_knowledge)
    monkeypatch.setattr(main, "extract_simulation_config", fake_config)

    def fake_simulation(request):
        return {
            "modulation": request.modulation,
            "ber_simulated": [0.1, 0.02],
            "constellation_received": [[1.0, 0.0], [-1.0, 0.0]],
        }

    monkeypatch.setattr(main, "execute_simulation", fake_simulation)

    response = client.post(
        "/api/v1/copilot/run",
        json={"prompt": "Simulate BPSK over AWGN"},
    )

    assert response.status_code == 200

    simulation = response.json()["simulation"]
    assert "constellation_received" not in simulation
    assert simulation["ber_simulated"] == [0.1, 0.02]


def test_copilot_includes_constellation_when_requested(monkeypatch):
    monkeypatch.setattr(main, "retrieve", fake_knowledge)
    monkeypatch.setattr(main, "extract_simulation_config", fake_config)

    constellation = [[1.0, 0.0], [-1.0, 0.0]]

    def fake_simulation(request):
        return {
            "modulation": request.modulation,
            "ber_simulated": [0.1, 0.02],
            "constellation_received": constellation,
        }

    monkeypatch.setattr(main, "execute_simulation", fake_simulation)

    response = client.post(
        "/api/v1/copilot/run",
        json={
            "prompt": "Simulate BPSK over AWGN",
            "include_constellation": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["simulation"]["constellation_received"] == constellation


def test_copilot_handles_rag_failure(monkeypatch):
    def fail_retrieval(prompt, top_k=3):
        raise RuntimeError("Embedding service unavailable")

    monkeypatch.setattr(main, "retrieve", fail_retrieval)

    response = client.post(
        "/api/v1/copilot/run",
        json={"prompt": "Simulate BPSK over AWGN"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Wireless knowledge retrieval is currently unavailable."
    )


def test_copilot_does_not_run_invalid_configuration(monkeypatch):
    monkeypatch.setattr(main, "retrieve", fake_knowledge)

    monkeypatch.setattr(
        main,
        "extract_simulation_config",
        lambda prompt, context="": {
            **fake_config(prompt),
            "modulation": "16QAM",
        },
    )

    def simulation_must_not_run(request):
        pytest.fail("Simulator must not run with an invalid configuration")

    monkeypatch.setattr(main, "execute_simulation", simulation_must_not_run)

    response = client.post(
        "/api/v1/copilot/run",
        json={"prompt": "Simulate 16QAM over Rayleigh fading"},
    )

    assert response.status_code == 422
