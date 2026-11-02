"""Evaluate local Ollama extraction against wireless copilot benchmarks.

This script evaluates extraction and validation only.
It does not execute wireless simulations.
"""

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from llm.client import OLLAMA_MODEL, extract_simulation_config
from llm.validator import validate_simulation_config

ROOT_DIR = Path(__file__).resolve().parent.parent

FIXTURE_DIR = ROOT_DIR / "tests" / "fixtures"
REPORT_ROOT = ROOT_DIR / "evaluation" / "reports"

REQUIRED_CONFIG_FIELDS = {
    "modulation",
    "channel",
    "snr_start_db",
    "snr_stop_db",
    "snr_step_db",
    "num_bits",
    "rician_k_db",
}


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """Load and validate a benchmark dataset."""

    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("Benchmark dataset must be a JSON list.")

    if not dataset:
        raise ValueError("Benchmark dataset cannot be empty.")

    for case in dataset:
        if not isinstance(case, dict):
            raise ValueError("Every benchmark case must be a JSON object.")

        required_case_fields = {
            "id",
            "prompt",
            "expected_status",
            "expected_config",
        }

        missing_case_fields = required_case_fields - case.keys()

        if missing_case_fields:
            raise ValueError(
                f"Benchmark case is missing fields: {sorted(missing_case_fields)}"
            )

        if case["expected_status"] not in {"valid", "invalid"}:
            raise ValueError(f"Invalid expected_status in case: {case['id']}")

        expected_config = case["expected_config"]

        if not isinstance(expected_config, dict):
            raise ValueError(f"expected_config must be an object in case: {case['id']}")

        missing_config_fields = REQUIRED_CONFIG_FIELDS - expected_config.keys()
        unexpected_config_fields = expected_config.keys() - REQUIRED_CONFIG_FIELDS

        if missing_config_fields or unexpected_config_fields:
            raise ValueError(
                f"Incorrect expected_config fields in case {case['id']}. "
                f"Missing: {sorted(missing_config_fields)}. "
                f"Unexpected: {sorted(unexpected_config_fields)}."
            )

    return dataset


def compare_expected_fields(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, bool]:
    """Compare all seven required configuration fields."""

    return {
        key: key in actual and actual[key] == expected[key]
        for key in REQUIRED_CONFIG_FIELDS
    }


def is_timeout_error(exc: BaseException) -> bool:
    """Check whether an exception or its cause is an HTTP timeout."""

    current: BaseException | None = exc

    while current is not None:
        if isinstance(current, httpx.TimeoutException):
            return True

        current = current.__cause__ or current.__context__

    return False


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one prompt using the actual local LLM and validator."""

    case_id = case["id"]
    prompt = case["prompt"]
    expected_status = case["expected_status"]
    expected_config = case["expected_config"]

    result: dict[str, Any] = {
        "id": case_id,
        "prompt": prompt,
        "expected_status": expected_status,
        "expected_config": expected_config,
        "extraction_success": False,
        "extraction_latency_seconds": None,
        "extraction_timeout": False,
        "extracted_config": None,
        "field_matches": {},
        "missing_fields": [],
        "unexpected_fields": [],
        "all_required_fields_present": False,
        "expected_config_match": False,
        "validation_status": "not_run",
        "validation_error": None,
        "validation_correct": False,
    }

    # Stage 1: Extract configuration using Ollama.
    extraction_started = time.perf_counter()

    try:
        extracted_config = extract_simulation_config(prompt)

        result["extraction_success"] = True
        result["extracted_config"] = extracted_config

    except Exception as exc:
        result["extraction_timeout"] = is_timeout_error(exc)
        result["validation_error"] = f"Extraction failed: {type(exc).__name__}: {exc}"

        return result

    finally:
        result["extraction_latency_seconds"] = round(
            time.perf_counter() - extraction_started,
            4,
        )

    # Stage 2: Check output fields.
    actual_fields = set(extracted_config.keys())

    result["missing_fields"] = sorted(REQUIRED_CONFIG_FIELDS - actual_fields)

    result["unexpected_fields"] = sorted(actual_fields - REQUIRED_CONFIG_FIELDS)

    result["all_required_fields_present"] = not result["missing_fields"]

    # Stage 3: Compare all seven expected configuration fields.
    field_matches = compare_expected_fields(
        expected_config,
        extracted_config,
    )

    result["field_matches"] = field_matches

    result["expected_config_match"] = (
        all(field_matches.values()) and not result["unexpected_fields"]
    )

    # Stage 4: Validate the extracted configuration.
    try:
        validate_simulation_config(prompt, extracted_config)
        result["validation_status"] = "valid"

    except ValueError as exc:
        result["validation_status"] = "invalid"
        result["validation_error"] = str(exc)

    # Stage 5: Check whether the validation outcome was correct.
    result["validation_correct"] = expected_status == result["validation_status"]

    return result


def calculate_metrics(
    results: list[dict[str, Any]],
    total_benchmark_duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Calculate benchmark accuracy and performance metrics."""

    total_cases = len(results)

    successful_extractions = [
        result for result in results if result["extraction_success"]
    ]

    valid_cases = [result for result in results if result["expected_status"] == "valid"]

    invalid_cases = [
        result for result in results if result["expected_status"] == "invalid"
    ]

    all_field_matches = [
        match
        for result in successful_extractions
        for match in result["field_matches"].values()
    ]

    latencies = [
        result["extraction_latency_seconds"]
        for result in results
        if result["extraction_latency_seconds"] is not None
    ]

    def percentage(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0

        return round((numerator / denominator) * 100, 2)

    if total_cases == 0:
        return {
            "total_cases": 0,
            "average_latency_seconds": None,
            "median_latency_seconds": None,
            "maximum_latency_seconds": None,
            "timeout_count": 0,
            "total_benchmark_duration_seconds": (total_benchmark_duration_seconds),
        }

    return {
        "total_cases": total_cases,
        "successful_extractions": len(successful_extractions),
        "extraction_failures": (total_cases - len(successful_extractions)),
        "extraction_success_rate_percent": percentage(
            len(successful_extractions),
            total_cases,
        ),
        "complete_configuration_count": sum(
            result["all_required_fields_present"] for result in successful_extractions
        ),
        "expected_config_match_count": sum(
            result["expected_config_match"] for result in successful_extractions
        ),
        "expected_config_match_rate_percent": percentage(
            sum(result["expected_config_match"] for result in successful_extractions),
            len(successful_extractions),
        ),
        "field_level_accuracy_percent": percentage(
            sum(all_field_matches),
            len(all_field_matches),
        ),
        "validation_correct_count": sum(
            result["validation_correct"] for result in results
        ),
        "validation_accuracy_percent": percentage(
            sum(result["validation_correct"] for result in results),
            total_cases,
        ),
        "valid_case_acceptance_rate_percent": percentage(
            sum(result["validation_correct"] for result in valid_cases),
            len(valid_cases),
        ),
        "invalid_case_rejection_rate_percent": percentage(
            sum(result["validation_correct"] for result in invalid_cases),
            len(invalid_cases),
        ),
        "average_latency_seconds": (
            round(statistics.mean(latencies), 4) if latencies else None
        ),
        "median_latency_seconds": (
            round(statistics.median(latencies), 4) if latencies else None
        ),
        "maximum_latency_seconds": (round(max(latencies), 4) if latencies else None),
        "timeout_count": sum(result["extraction_timeout"] for result in results),
        "total_benchmark_duration_seconds": (
            round(total_benchmark_duration_seconds, 4)
            if total_benchmark_duration_seconds is not None
            else None
        ),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse the selected benchmark dataset."""

    parser = argparse.ArgumentParser(description="Evaluate the wireless copilot LLM.")

    parser.add_argument(
        "--dataset",
        choices=["development", "heldout"],
        default="development",
        help="Select the development or held-out benchmark.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the selected benchmark and save a JSON report."""

    args = parse_arguments()

    dataset_filename = (
        "copilot_benchmark.json"
        if args.dataset == "development"
        else "copilot_heldout.json"
    )

    dataset_path = FIXTURE_DIR / dataset_filename

    report_dir = REPORT_ROOT / args.dataset
    report_dir.mkdir(parents=True, exist_ok=True)

    benchmark_started = time.perf_counter()
    evaluation_timestamp = datetime.now(UTC)

    dataset = load_dataset(dataset_path)

    print("=" * 65)
    print("AI-WIRELESS COPILOT: LLM BENCHMARK")
    print("=" * 65)
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Dataset: {args.dataset}")
    print(f"Total benchmark cases: {len(dataset)}")
    print("Simulation execution: DISABLED")
    print("=" * 65)

    results = []

    for index, case in enumerate(dataset, start=1):
        print(f"\n[{index}/{len(dataset)}] Evaluating: {case['id']}")

        result = evaluate_case(case)
        results.append(result)

        print(f"  Extraction: {result['extraction_success']}")
        print(f"  Extraction latency: {result['extraction_latency_seconds']} seconds")
        print(f"  Extraction timeout: {result['extraction_timeout']}")
        print(f"  Validation: {result['validation_status']}")
        print(f"  Complete configuration: {result['all_required_fields_present']}")
        print(f"  Expected configuration match: {result['expected_config_match']}")
        print(f"  Validation outcome correct: {result['validation_correct']}")

        if result["missing_fields"]:
            print(f"  Missing fields: {result['missing_fields']}")

        if result["unexpected_fields"]:
            print(f"  Unexpected fields: {result['unexpected_fields']}")

        if result["validation_error"]:
            print(f"  Error: {result['validation_error']}")

    total_duration = time.perf_counter() - benchmark_started

    metrics = calculate_metrics(
        results,
        total_benchmark_duration_seconds=total_duration,
    )

    timestamp = evaluation_timestamp.strftime("%Y%m%d_%H%M%S")

    report_path = report_dir / (f"llm_{args.dataset}_{timestamp}.json")

    report = {
        "model": OLLAMA_MODEL,
        "dataset_type": args.dataset,
        "evaluation_timestamp_utc": evaluation_timestamp.isoformat(),
        "dataset": str(dataset_path.relative_to(ROOT_DIR)),
        "simulation_execution": False,
        "metrics": metrics,
        "results": results,
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print("\nReport saved to:")
    print(report_path.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
