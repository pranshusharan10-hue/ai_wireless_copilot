"""Tests for the LLM benchmark evaluator."""

import httpx

from scripts.evaluate_llm import (
    calculate_metrics,
    evaluate_case,
    is_timeout_error,
)


def make_result(
    *,
    extraction_success: bool,
    latency: float,
    timeout: bool = False,
    expected_status: str = "valid",
    validation_correct: bool = True,
    field_matches: dict[str, bool] | None = None,
    all_required_fields_present: bool = True,
    unexpected_fields: list[str] | None = None,
) -> dict:
    """Create a benchmark result for metric tests."""
    field_matches = field_matches or {}
    unexpected_fields = unexpected_fields or []

    return {
        "extraction_success": extraction_success,
        "extraction_latency_seconds": latency,
        "extraction_timeout": timeout,
        "expected_status": expected_status,
        "validation_correct": validation_correct,
        "all_required_fields_present": all_required_fields_present,
        "missing_fields": ([] if all_required_fields_present else ["rician_k_db"]),
        "unexpected_fields": unexpected_fields,
        "expected_config_match": (
            all(field_matches.values()) and not unexpected_fields
            if field_matches
            else False
        ),
        "field_matches": field_matches,
    }


def test_timeout_detection_handles_wrapped_httpx_timeout():
    """Detect timeouts wrapped by the Ollama client."""
    original_error = httpx.ReadTimeout("Request timed out")
    wrapped_error = RuntimeError("Could not get a response from Ollama")
    wrapped_error.__cause__ = original_error

    assert is_timeout_error(wrapped_error) is True


def test_timeout_detection_returns_false_for_other_errors():
    """Do not classify ordinary errors as timeouts."""
    assert is_timeout_error(RuntimeError("Invalid JSON")) is False


def test_evaluate_case_records_latency_and_timeout(monkeypatch):
    """Record latency when extraction fails with a timeout."""

    def fake_extraction(prompt):
        raise RuntimeError("Ollama timeout") from httpx.ReadTimeout("Request timed out")

    monkeypatch.setattr(
        "scripts.evaluate_llm.extract_simulation_config",
        fake_extraction,
    )

    case = {
        "id": "timeout_test",
        "prompt": "Simulate BPSK over AWGN",
        "expected_status": "valid",
        "expected_config": {"modulation": "BPSK"},
    }

    result = evaluate_case(case)

    assert result["extraction_success"] is False
    assert result["extraction_timeout"] is True
    assert result["extraction_latency_seconds"] is not None
    assert result["validation_status"] == "not_run"


def test_calculate_metrics_includes_latency_and_timeouts():
    """Calculate performance metrics across successful and failed cases."""
    results = [
        make_result(
            extraction_success=True,
            latency=1.0,
            field_matches={"modulation": True, "channel": True},
        ),
        make_result(
            extraction_success=True,
            latency=2.0,
            field_matches={"modulation": True, "channel": False},
        ),
        make_result(
            extraction_success=False,
            latency=3.0,
            timeout=True,
            validation_correct=False,
        ),
    ]

    metrics = calculate_metrics(
        results,
        total_benchmark_duration_seconds=7.5,
    )

    assert metrics["total_cases"] == 3
    assert metrics["successful_extractions"] == 2
    assert metrics["average_latency_seconds"] == 2.0
    assert metrics["median_latency_seconds"] == 2.0
    assert metrics["maximum_latency_seconds"] == 3.0
    assert metrics["timeout_count"] == 1
    assert metrics["total_benchmark_duration_seconds"] == 7.5
    assert metrics["field_level_accuracy_percent"] == 75.0


def test_calculate_metrics_handles_empty_results():
    """Handle an empty benchmark without division errors."""
    metrics = calculate_metrics([])

    assert metrics["total_cases"] == 0
    assert metrics["average_latency_seconds"] is None
    assert metrics["median_latency_seconds"] is None
    assert metrics["maximum_latency_seconds"] is None
    assert metrics["timeout_count"] == 0
