"""Unit tests for experiment configuration and reproducibility infrastructure."""

from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from benchmark.schema import BenchmarkCase
from llm.provider import (
    OllamaProvider,
    OpenAICompatibleProvider,
    ProviderConfigError,
    ProviderNetworkError,
    ProviderResponseError,
    get_llm_provider,
)
from runner.experiment_config import (
    ExperimentConfig,
    RAGConfig,
    SimulationSettings,
    ValidationConfig,
)
from runner.experiment_result import (
    LatencyBreakdown,
    ValidationOutcome,
    VerificationOutcome,
    classify_case_result,
)


def test_create_method_presets():
    """Verify standard M1, M2, M3, M4 preset configurations."""
    cfg_m1 = ExperimentConfig.create_method_preset("exp_01_m1", "M1")
    assert cfg_m1.method == "M1"
    assert not cfg_m1.rag.enabled
    assert not cfg_m1.validation.enabled
    assert not cfg_m1.verification.enabled

    cfg_m2 = ExperimentConfig.create_method_preset("exp_02_m2", "M2")
    assert cfg_m2.method == "M2"
    assert cfg_m2.rag.enabled
    assert not cfg_m2.validation.enabled
    assert not cfg_m2.verification.enabled

    cfg_m3 = ExperimentConfig.create_method_preset("exp_03_m3", "M3")
    assert cfg_m3.method == "M3"
    assert cfg_m3.rag.enabled
    assert cfg_m3.validation.enabled
    assert not cfg_m3.verification.enabled

    cfg_m4 = ExperimentConfig.create_method_preset("exp_04_m4", "M4")
    assert cfg_m4.method == "M4"
    assert cfg_m4.rag.enabled
    assert cfg_m4.validation.enabled
    assert cfg_m4.verification.enabled


def test_method_preset_contradiction_validation():
    """Enforce that configurations cannot contradict their declared method."""
    # M1 with RAG enabled must fail unless overridden
    with pytest.raises(ValidationError, match="Method M1 must have rag"):
        ExperimentConfig(
            experiment_id="exp_invalid_m1",
            method="M1",
            rag=RAGConfig(enabled=True),
        )

    # With allow_experimental_overrides=True, it is accepted
    cfg_override = ExperimentConfig(
        experiment_id="exp_override_m1",
        method="M1",
        rag=RAGConfig(enabled=True),
        allow_experimental_overrides=True,
    )
    assert cfg_override.rag.enabled


def test_bounded_correction_limit():
    """Enforce strict bound of at most one correction attempt."""
    # max_correction_attempts = 1 is allowed
    val_cfg = ValidationConfig(max_correction_attempts=1)
    assert val_cfg.max_correction_attempts == 1

    # max_correction_attempts = 2 must violate the research protocol
    with pytest.raises(ValidationError):
        ValidationConfig(max_correction_attempts=2)


def test_simulation_settings_validation():
    """Verify range and ordering constraints on simulation settings."""
    with pytest.raises(ValidationError, match="strictly less than"):
        SimulationSettings(snr_min_db=20.0, snr_max_db=0.0)

    with pytest.raises(ValidationError, match="cannot exceed"):
        SimulationSettings(min_num_bits=50000, max_num_bits=10000)

    valid_settings = SimulationSettings(
        min_num_bits=1000,
        default_num_bits=10000,
        max_num_bits=100000,
        snr_min_db=-10.0,
        snr_max_db=30.0,
    )
    assert valid_settings.snr_min_db == -10.0


def test_llm_provider_factory_names():
    """Verify all supported provider names including openai_compatible."""
    names = ["ollama", "openai_compatible", "openai", "vllm", "compatible"]
    for name in names:
        p = get_llm_provider(
            name,
            base_url="http://localhost:8000/v1"
            if name != "ollama"
            else "http://localhost:11434",
            model_name="test_model",
        )
        assert p is not None

    with pytest.raises(ProviderConfigError, match="Unknown LLM provider type"):
        get_llm_provider("unsupported_provider")


def test_provider_mocked_ollama(monkeypatch):
    """Test OllamaProvider generation with mocked HTTP responses."""

    def mock_post(url, json, **kwargs):
        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"response": '{"modulation": "BPSK"}'}

        return MockResponse()

    provider = OllamaProvider(
        base_url="http://localhost:11434", model_name="qwen2.5:3b"
    )
    monkeypatch.setattr(
        httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, **kwargs)
    )

    res = provider.generate("Simulate BPSK", "You are an extractor")
    assert res == '{"modulation": "BPSK"}'


def test_provider_mocked_openai_compatible(monkeypatch):
    """Test OpenAICompatibleProvider with mocked HTTP response and secret redaction."""
    secret = "sk-super-secret-key-12345"

    def mock_post(url, headers, json, **kwargs):
        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": '{"modulation": "QPSK"}'}}]}

        return MockResponse()

    provider = OpenAICompatibleProvider(
        base_url="http://remote-gpu:8000/v1",
        model_name="mistral:7b",
        api_key=secret,
    )
    monkeypatch.setattr(
        httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, **kwargs)
    )

    res = provider.generate("Simulate QPSK", "Extract JSON")
    assert res == '{"modulation": "QPSK"}'

    # Test secret redaction on network failure
    def mock_error_post(*args, **kwargs):
        request = httpx.Request("POST", "http://remote-gpu:8000/v1/chat/completions")
        raise httpx.RequestError(
            f"Connection failed with key {secret}", request=request
        )

    monkeypatch.setattr(httpx.Client, "post", mock_error_post)
    with pytest.raises(ProviderNetworkError) as exc_info:
        provider.generate("Simulate QPSK", "Extract")

    # The secret API key MUST NOT appear in exception text
    assert secret not in str(exc_info.value)
    assert "[REDACTED_API_KEY]" in str(exc_info.value)


def test_latency_breakdown_validation():
    """Verify that total latency cannot be less than sum of components."""
    # Valid breakdown
    lb = LatencyBreakdown(
        rag_latency_sec=0.1,
        llm_generation_sec=0.3,
        validation_latency_sec=0.05,
        total_latency_sec=0.5,
    )
    assert lb.total_latency_sec == 0.5

    # Invalid breakdown where total is less than stage sum
    with pytest.raises(ValidationError, match="cannot be less than sum"):
        LatencyBreakdown(
            rag_latency_sec=0.5,
            llm_generation_sec=0.5,
            total_latency_sec=0.2,
        )


def test_classify_case_result_logic():
    """Test programmatic classification of experimental outcomes."""
    case_execute = BenchmarkCase(
        id="phase5_std_01",
        category="A_standard_supported",
        prompt="Simulate BPSK over AWGN",
        expected_action="execute",
        expected_config={
            "modulation": "BPSK",
            "channel": "AWGN",
            "snr_start_db": 0.0,
            "snr_stop_db": 10.0,
            "snr_step_db": 2.0,
            "num_bits": 10000,
            "rician_k_db": 6.0,
        },
        decision_reason="Valid request",
        ground_truth_review_status="pending_review",
        provenance="Verified against wireless/simulation.py; commit f360d6d.",
    )

    # 1. Correct execution
    outcome1 = classify_case_result(
        case=case_execute,
        actual_decision="execute",
        parsed_config=case_execute.expected_config,
        validation_outcome=ValidationOutcome(passed=True),
        sim_output={"ber_simulated": [0.05], "error_counts": [500]},
        verification_outcome=None,
    )
    assert outcome1["is_correct_decision"]
    assert outcome1["is_physically_correct"]
    assert not outcome1["is_silent_error"]

    # 2. Silent parameter corruption
    corrupt_config = dict(case_execute.expected_config)
    corrupt_config["snr_start_db"] = 15.0  # corrupted
    outcome2 = classify_case_result(
        case=case_execute,
        actual_decision="execute",
        parsed_config=corrupt_config,
        validation_outcome=ValidationOutcome(passed=True),
        sim_output={"ber_simulated": [0.0], "error_counts": [0]},
        verification_outcome=None,
        flagged_by_system=False,
    )
    assert outcome2["is_silent_error"]
    assert not outcome2["is_physically_correct"]

    # 3. Correct rejection of unsupported prompt
    case_reject = BenchmarkCase(
        id="phase5_mod_01",
        category="F_unsupported_mod_channel",
        prompt="Simulate 64-QAM over AWGN",
        expected_action="reject",
        decision_reason="Unsupported modulation",
        ground_truth_review_status="pending_review",
        provenance="Verified against API capabilities; commit f360d6d.",
    )
    outcome3 = classify_case_result(
        case=case_reject,
        actual_decision="reject",
        parsed_config=None,
        validation_outcome=ValidationOutcome(passed=False),
        sim_output=None,
        verification_outcome=None,
    )
    assert outcome3["is_correct_decision"]
    assert outcome3["is_physically_correct"]
    assert not outcome3["is_silent_error"]


def test_provider_finite_and_url_validation():
    """Verify that provider rejects NaN, inf, invalid numbers, and bad URLs."""
    import math

    # Ollama invalid URLs
    with pytest.raises(ProviderConfigError, match="Must include valid scheme"):
        OllamaProvider(base_url="not_a_valid_url")
    with pytest.raises(ProviderConfigError, match="Must include valid scheme"):
        OllamaProvider(base_url="http://")

    # Ollama non-finite timeout
    with pytest.raises(ProviderConfigError, match="finite positive number"):
        OllamaProvider(base_url="http://localhost:11434", timeout=float("nan"))
    with pytest.raises(ProviderConfigError, match="finite positive number"):
        OllamaProvider(base_url="http://localhost:11434", timeout=float("inf"))
    with pytest.raises(ProviderConfigError, match="finite positive number"):
        OllamaProvider(base_url="http://localhost:11434", timeout=-5.0)

    # Ollama non-finite temperature
    prov = OllamaProvider(base_url="http://localhost:11434")
    with pytest.raises(ProviderConfigError, match="finite number"):
        prov.generate("prompt", "sys", temperature=float("nan"))
    with pytest.raises(ProviderConfigError, match="finite number"):
        prov.generate("prompt", "sys", temperature=float("inf"))
    with pytest.raises(ProviderConfigError, match="finite number"):
        prov.generate("prompt", "sys", temperature=2.5)

    # OpenAI-compatible invalid URLs and non-finite timeout
    with pytest.raises(ProviderConfigError, match="Must include valid scheme"):
        OpenAICompatibleProvider(base_url="ftp://remote-server/v1")
    with pytest.raises(ProviderConfigError, match="finite positive number"):
        OpenAICompatibleProvider(base_url="http://remote-server/v1", timeout=math.nan)


def test_provider_mocked_ollama_edge_cases(monkeypatch):
    """Test OllamaProvider handling of empty, malformed, and error responses."""
    prov = OllamaProvider(base_url="http://localhost:11434")

    # 1. Empty / whitespace response
    def mock_empty_post(*args, **kwargs):
        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"response": "   "}

        return Resp()

    monkeypatch.setattr(httpx.Client, "post", mock_empty_post)
    with pytest.raises(ProviderResponseError, match="empty/whitespace-only"):
        prov.generate("prompt", "sys")

    # 2. Malformed JSON (raises ValueError)
    def mock_bad_json(*args, **kwargs):
        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                raise ValueError("Expecting value: line 1 column 1 (char 0)")

        return Resp()

    monkeypatch.setattr(httpx.Client, "post", mock_bad_json)
    with pytest.raises(ProviderResponseError, match="Malformed JSON response"):
        prov.generate("prompt", "sys")

    # 3. Malformed structure (missing response field)
    def mock_bad_struct(*args, **kwargs):
        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"status": "ok"}  # missing 'response'

        return Resp()

    monkeypatch.setattr(httpx.Client, "post", mock_bad_struct)
    with pytest.raises(ProviderResponseError, match="missing or empty"):
        prov.generate("prompt", "sys")

    # 4. HTTP status error
    def mock_http_err(*args, **kwargs):
        request = httpx.Request("POST", "http://localhost:11434/api/generate")
        response = httpx.Response(500, request=request)
        raise httpx.HTTPStatusError("Server Error", request=request, response=response)

    monkeypatch.setattr(httpx.Client, "post", mock_http_err)
    with pytest.raises(ProviderNetworkError, match="Ollama HTTP error 500"):
        prov.generate("prompt", "sys")


def test_provider_mocked_openai_edge_cases(monkeypatch):
    """Test OpenAICompatibleProvider empty, malformed, and error responses."""
    secret = "sk-super-secret-12345"
    prov = OpenAICompatibleProvider(
        base_url="http://remote-server:8000/v1", api_key=secret
    )

    # 1. Empty / whitespace content
    def mock_empty_content(*args, **kwargs):
        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": " \n\t "}}]}

        return Resp()

    monkeypatch.setattr(httpx.Client, "post", mock_empty_content)
    with pytest.raises(ProviderResponseError, match="empty/whitespace-only"):
        prov.generate("prompt", "sys")

    # 2. Malformed JSON (raises ValueError)
    def mock_bad_json(*args, **kwargs):
        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                raise ValueError("JSON decode failed")

        return Resp()

    monkeypatch.setattr(httpx.Client, "post", mock_bad_json)
    with pytest.raises(ProviderResponseError, match="Malformed JSON response"):
        prov.generate("prompt", "sys")

    # 3. Malformed structure (empty choices list)
    def mock_empty_choices(*args, **kwargs):
        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": []}

        return Resp()

    monkeypatch.setattr(httpx.Client, "post", mock_empty_choices)
    with pytest.raises(ProviderResponseError, match="missing non-empty 'choices'"):
        prov.generate("prompt", "sys")

    # 4. HTTP status error with secret redaction
    def mock_http_err(*args, **kwargs):
        request = httpx.Request(
            "POST",
            "http://remote-server:8000/v1/chat/completions",
            headers={"Authorization": f"Bearer {secret}"},
        )
        response = httpx.Response(401, request=request)
        raise httpx.HTTPStatusError("Unauthorized", request=request, response=response)

    monkeypatch.setattr(httpx.Client, "post", mock_http_err)
    with pytest.raises(ProviderNetworkError) as exc_info:
        prov.generate("prompt", "sys")
    assert secret not in str(exc_info.value)


def test_classify_case_result_edge_cases():
    """Verify classification invariants for missing, empty, incomplete configs,
    and verify that unsubstantiated caller flags cannot suppress silent errors.
    """
    case_execute = BenchmarkCase(
        id="phase5_exec_case",
        category="A_standard_supported",
        prompt="Simulate BPSK over AWGN from 0 to 10 dB",
        expected_action="execute",
        expected_config={
            "modulation": "BPSK",
            "channel": "AWGN",
            "snr_start_db": 0.0,
            "snr_stop_db": 10.0,
            "snr_step_db": 2.0,
            "num_bits": 10000,
            "rician_k_db": 6.0,
        },
        decision_reason="Valid supported case",
        ground_truth_review_status="pending_review",
        provenance="Verified against wireless/simulation.py; commit f360d6d.",
    )

    # 1. Missing parsed_config (None) for an execute case
    res_none = classify_case_result(
        case=case_execute,
        actual_decision="execute",
        parsed_config=None,
        validation_outcome=ValidationOutcome(passed=True),
        sim_output=None,
        verification_outcome=None,
        flagged_by_system=False,
    )
    assert not res_none["is_physically_correct"]
    assert res_none["is_silent_error"]
    assert "missing_or_empty_parsed_config" in res_none["error_classification"]

    # 2. Empty parsed_config ({}) for an execute case
    res_empty = classify_case_result(
        case=case_execute,
        actual_decision="execute",
        parsed_config={},
        validation_outcome=ValidationOutcome(passed=True),
        sim_output=None,
        verification_outcome=None,
        flagged_by_system=False,
    )
    assert not res_empty["is_physically_correct"]
    assert res_empty["is_silent_error"]
    assert "missing_or_empty_parsed_config" in res_empty["error_classification"]

    # 3. Incomplete parsed_config (missing required keys like snr_stop_db)
    res_incomplete = classify_case_result(
        case=case_execute,
        actual_decision="execute",
        parsed_config={"modulation": "BPSK", "channel": "AWGN"},
        validation_outcome=ValidationOutcome(passed=True),
        sim_output=None,
        verification_outcome=None,
        flagged_by_system=False,
    )
    assert not res_incomplete["is_physically_correct"]
    assert res_incomplete["is_silent_error"]
    assert any("missing_parameter" in e for e in res_incomplete["error_classification"])

    # 4. Unsupported execution where caller passes flagged_by_system=True
    # BUT validation_outcome.passed=True and no verification outcome exists.
    # The classification MUST derive that this is an unsubstantiated flag and
    # STILL flag it as a silent error!
    case_reject = BenchmarkCase(
        id="phase5_bad_case",
        category="C_contradictory_request",
        prompt="Simulate BPSK with negative bit count",
        expected_action="reject",
        decision_reason="Negative bits",
        ground_truth_review_status="pending_review",
        provenance="Verified against wireless/simulation.py; commit f360d6d.",
    )
    res_unsubstantiated = classify_case_result(
        case=case_reject,
        actual_decision="execute",
        parsed_config=case_execute.expected_config,
        # Validation falsely claims passed:
        validation_outcome=ValidationOutcome(passed=True),
        sim_output=None,
        verification_outcome=None,
        flagged_by_system=True,  # Caller claims flagged without evidence!
    )
    assert res_unsubstantiated["is_silent_error"]  # MUST still be a silent error!
    assert not res_unsubstantiated["is_physically_correct"]
    assert (
        "silent_execution_of_unsupported_or_invalid_prompt"
        in (res_unsubstantiated["error_classification"])
    )

    # 5. Corroborated flag via validation failure
    res_corroborated = classify_case_result(
        case=case_reject,
        actual_decision="execute",
        parsed_config=case_execute.expected_config,
        validation_outcome=ValidationOutcome(
            passed=False, validation_errors=["Negative bit count not allowed"]
        ),
        sim_output=None,
        verification_outcome=None,
        flagged_by_system=True,
    )
    assert not res_corroborated["is_silent_error"]
    assert (
        "execution_of_invalid_prompt_flagged_by_system"
        in (res_corroborated["error_classification"])
    )

    # 6. Corroborated flag via verification failure
    res_verif = classify_case_result(
        case=case_execute,
        actual_decision="execute",
        parsed_config=case_execute.expected_config,
        validation_outcome=ValidationOutcome(passed=True),
        sim_output={"ber_simulated": [0.0], "error_counts": [0]},
        verification_outcome=VerificationOutcome(
            verified_valid=False,
            zero_errors_detected=True,
            verification_flags=["Sample starvation at 10 dB"],
        ),
        flagged_by_system=True,
    )
    assert not res_verif["is_silent_error"]
