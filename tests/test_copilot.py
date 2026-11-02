# integration tests
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import backend.main as main
from backend.schemas import SimulationRequest

client = TestClient(main.app)

VALID_CONFIG = {
    "modulation": "BPSK",
    "channel": "AWGN",
    "snr_start_db": 0,
    "snr_stop_db": 20,
    "snr_step_db": 2,
    "num_bits": 10000,
    "rician_k_db": 6.0,
}


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


@pytest.fixture(autouse=True)
def mock_rag_retrieval(monkeypatch):
    monkeypatch.setattr(main, "retrieve", fake_knowledge)


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

    def fail_llm(prompt, context="", retry_feedback=None):
        raise main.LLMServiceError("Ollama is unavailable")

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


def test_copilot_retries_malformed_response_then_succeeds(monkeypatch):
    llm = Mock(
        side_effect=[
            main.LLMResponseError("Malformed response"),
            VALID_CONFIG,
        ]
    )

    simulator = Mock(return_value={"status": "success"})

    monkeypatch.setattr(main, "extract_simulation_config", llm)
    monkeypatch.setattr(main, "execute_simulation", simulator)

    response = client.post(
        "/api/v1/copilot/run",
        json={"prompt": "Simulate BPSK over AWGN"},
    )

    assert response.status_code == 200
    assert llm.call_count == 2
    assert llm.call_args_list[1].kwargs["retry_feedback"]
    simulator.assert_called_once()


def test_copilot_returns_502_after_second_malformed_response(monkeypatch):
    llm = Mock(side_effect=main.LLMResponseError("Malformed response"))
    simulator = Mock()

    monkeypatch.setattr(main, "extract_simulation_config", llm)
    monkeypatch.setattr(main, "execute_simulation", simulator)

    response = client.post(
        "/api/v1/copilot/run",
        json={"prompt": "Simulate BPSK over AWGN"},
    )

    assert response.status_code == 502
    assert llm.call_count == 2
    simulator.assert_not_called()


def test_copilot_does_not_retry_service_error(monkeypatch):
    llm = Mock(side_effect=main.LLMServiceError("Ollama unavailable"))
    simulator = Mock()

    monkeypatch.setattr(main, "extract_simulation_config", llm)
    monkeypatch.setattr(main, "execute_simulation", simulator)

    response = client.post(
        "/api/v1/copilot/run",
        json={"prompt": "Simulate BPSK over AWGN"},
    )

    assert response.status_code == 503
    llm.assert_called_once()
    simulator.assert_not_called()


def test_copilot_does_not_retry_valid_response(monkeypatch):
    llm = Mock(return_value=VALID_CONFIG)
    simulator = Mock(return_value={"status": "success"})

    monkeypatch.setattr(main, "extract_simulation_config", llm)
    monkeypatch.setattr(main, "execute_simulation", simulator)

    response = client.post(
        "/api/v1/copilot/run",
        json={"prompt": "Simulate BPSK over AWGN"},
    )

    assert response.status_code == 200
    llm.assert_called_once()
    simulator.assert_called_once()


@pytest.mark.parametrize(
    "invalid_config",
    [
        {**VALID_CONFIG, "modulation": "INVALID"},
        {**VALID_CONFIG, "num_bits": -100},
        {**VALID_CONFIG, "snr_step_db": 0},
        {**VALID_CONFIG, "snr_start_db": 20, "snr_stop_db": 0},
    ],
)
def test_copilot_does_not_retry_invalid_parameters(
    monkeypatch,
    invalid_config,
):
    llm = Mock(return_value=invalid_config)
    simulator = Mock()

    monkeypatch.setattr(main, "extract_simulation_config", llm)
    monkeypatch.setattr(main, "execute_simulation", simulator)

    response = client.post(
        "/api/v1/copilot/run",
        json={"prompt": "Run the requested simulation"},
    )

    assert response.status_code == 422
    llm.assert_called_once()
    simulator.assert_not_called()
