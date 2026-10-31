"""Tests for the copilot benchmark dataset."""

import json
from pathlib import Path

import pytest

from llm.validator import validate_simulation_config

BENCHMARK_PATH = Path(__file__).parent / "fixtures" / "copilot_benchmark.json"


@pytest.fixture(scope="module")
def benchmark_cases():
    with BENCHMARK_PATH.open(encoding="utf-8") as file:
        cases = json.load(file)

    assert isinstance(cases, list)
    assert cases, "The benchmark dataset must not be empty."
    return cases


def test_benchmark_cases_have_required_fields(benchmark_cases):
    required_fields = {
        "id",
        "prompt",
        "expected_status",
        "expected_config",
    }

    ids = []

    for case in benchmark_cases:
        assert required_fields.issubset(case)
        assert case["expected_status"] in {"valid", "invalid"}
        assert isinstance(case["prompt"], str)
        assert case["prompt"].strip()
        assert isinstance(case["expected_config"], dict)
        ids.append(case["id"])

    assert len(ids) == len(set(ids)), "Benchmark IDs must be unique."


@pytest.mark.parametrize(
    "expected_status",
    ["valid", "invalid"],
)
def test_benchmark_expected_statuses_are_consistent(
    benchmark_cases,
    expected_status,
):
    selected_cases = [
        case for case in benchmark_cases if case["expected_status"] == expected_status
    ]

    assert selected_cases, f"No {expected_status} benchmark cases found."

    for case in selected_cases:
        if expected_status == "valid":
            validate_simulation_config(
                case["prompt"],
                case["expected_config"],
            )
        else:
            with pytest.raises(ValueError):
                validate_simulation_config(
                    case["prompt"],
                    case["expected_config"],
                )
