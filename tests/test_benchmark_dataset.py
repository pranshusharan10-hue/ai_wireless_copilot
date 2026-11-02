import json
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"

DATASETS = [
    "copilot_benchmark.json",
    "copilot_heldout.json",
]

REQUIRED_CONFIG_FIELDS = {
    "modulation",
    "channel",
    "snr_start_db",
    "snr_stop_db",
    "snr_step_db",
    "num_bits",
    "rician_k_db",
}


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_benchmark_cases_have_required_fields(dataset_name):
    dataset_path = FIXTURE_DIR / dataset_name

    with dataset_path.open(encoding="utf-8") as file:
        cases = json.load(file)

    assert isinstance(cases, list)
    assert len(cases) > 0

    case_ids = set()

    for case in cases:
        assert {"id", "prompt", "expected_status", "expected_config"} <= set(case)

        assert case["id"] not in case_ids
        case_ids.add(case["id"])

        assert isinstance(case["prompt"], str)
        assert case["prompt"].strip()

        assert case["expected_status"] in {"valid", "invalid"}

        assert set(case["expected_config"]) == REQUIRED_CONFIG_FIELDS


@pytest.mark.parametrize("dataset_name", DATASETS)
@pytest.mark.parametrize("expected_status", ["valid", "invalid"])
def test_benchmark_expected_statuses_are_consistent(dataset_name, expected_status):
    dataset_path = FIXTURE_DIR / dataset_name

    with dataset_path.open(encoding="utf-8") as file:
        cases = json.load(file)

    matching_cases = [
        case for case in cases if case["expected_status"] == expected_status
    ]

    assert matching_cases, f"No {expected_status} cases found in {dataset_name}"
