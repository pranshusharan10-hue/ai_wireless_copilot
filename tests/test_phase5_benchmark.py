"""Automated validation tests for the Phase 5 benchmark dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.schema import (
    ALL_BENCHMARK_CATEGORIES,
    BenchmarkCase,
    BenchmarkDataset,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "phase5_benchmark.json"
)

CANONICAL_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "benchmark"
    / "phase5_benchmark.json"
)


@pytest.fixture(scope="module")
def benchmark_data() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_benchmark_files_exist():
    assert FIXTURE_PATH.is_file(), f"Fixture file not found at {FIXTURE_PATH}"
    assert CANONICAL_PATH.is_file(), f"Canonical file not found at {CANONICAL_PATH}"


def test_benchmark_dataset_pydantic_validation(benchmark_data):
    dataset = BenchmarkDataset.model_validate(benchmark_data)
    assert dataset.total_cases >= 50
    assert dataset.total_cases <= 100
    assert len(dataset.cases) == dataset.total_cases


def test_benchmark_unique_case_ids(benchmark_data):
    cases = benchmark_data["cases"]
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), "Duplicate case IDs detected!"


def test_benchmark_category_coverage(benchmark_data):
    cases = benchmark_data["cases"]
    categories = {case["category"] for case in cases}
    assert categories == ALL_BENCHMARK_CATEGORIES, (
        f"Missing categories: {ALL_BENCHMARK_CATEGORIES - categories}"
    )

    # Every category must contain at least 5 cases
    for cat in ALL_BENCHMARK_CATEGORIES:
        count = sum(1 for c in cases if c["category"] == cat)
        assert count >= 5, f"Category {cat} has only {count} cases (expected >= 5)"


def test_benchmark_schema_invariants():
    """Verify schema-level validation invariants for categories and actions."""
    # 1. Dataset requires all 8 categories
    with pytest.raises(ValidationError, match="Dataset must declare all 8 categories"):
        BenchmarkDataset(
            version="v1.0",
            description="Test dataset",
            created_at="2026-11-07T00:00:00Z",
            total_cases=0,
            categories=["A_standard_supported"],  # missing 7 categories
            cases=[],
        )

    # 2. Case category must be in declared categories
    with pytest.raises(ValidationError):
        BenchmarkCase(
            id="phase5_bad_cat",
            category="NonexistentCategory",  # type: ignore
            prompt="Simulate BPSK",
            expected_action="reject",
            decision_reason="Testing bad category",
            ground_truth_review_status="independently_reviewed",
            provenance="Verified against wireless/simulation.py; commit f360d6d.",
        )

    # 3. Provenance must be at least 15 chars (auditable)
    with pytest.raises(ValidationError):
        BenchmarkCase(
            id="phase5_short_prov",
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
            ground_truth_review_status="independently_reviewed",
            provenance="short",  # too short (< 15 chars)
        )


def test_benchmark_expected_action_validity(benchmark_data):
    for case_dict in benchmark_data["cases"]:
        case = BenchmarkCase.model_validate(case_dict)
        if case.expected_action == "execute":
            assert case.expected_config is not None
            assert case.expected_config["modulation"] in ("BPSK", "QPSK")
            assert case.expected_config["channel"] in ("AWGN", "Rayleigh", "Rician")
            assert case.expected_config["num_bits"] >= 1000
            assert (
                case.expected_config["snr_stop_db"]
                > case.expected_config["snr_start_db"]
            )
            assert case.expected_config["snr_step_db"] > 0
        elif case.expected_action in ("reject", "clarify"):
            # For reject or clarify, expected_config must be None
            err_msg = (
                f"Case {case.id} has action '{case.expected_action}' "
                "but includes an executable config"
            )
            assert case.expected_config is None, err_msg


def test_benchmark_semantic_distinctions(benchmark_data):
    """Verify distinction between convertible, invalid, and standalone cases."""
    case_map = {c["id"]: c for c in benchmark_data["cases"]}

    # Convertible Es/N0 for QPSK -> execute with converted Eb/N0 range
    snr_01 = case_map["phase5_snr_01"]
    assert snr_01["expected_action"] == "execute"
    assert (
        snr_01["expected_config"]["snr_start_db"] < 0
    )  # converted from 0 dB Es/N0 to ~ -3.01 dB Eb/N0

    # Dimensionally invalid unit -> reject
    snr_05 = case_map["phase5_snr_05"]
    assert snr_05["expected_action"] == "reject"
    assert "Watts" in snr_05["prompt"]

    # Contradictory SNR relationship -> reject
    snr_07 = case_map["phase5_snr_07"]
    assert snr_07["expected_action"] == "reject"

    # Standalone module requested through primary API -> reject with explanation
    mod_01 = case_map["phase5_mod_01"]
    assert mod_01["expected_action"] == "reject"
    assert "standalone" in mod_01["decision_reason"].lower()

    mod_10 = case_map["phase5_mod_10"]
    assert mod_10["expected_action"] == "reject"
    assert "standalone" in mod_10["decision_reason"].lower()


def test_benchmark_provenance_and_review_status(benchmark_data):
    for case in benchmark_data["cases"]:
        assert case["ground_truth_review_status"] == "pending_review"
        assert len(case["decision_reason"]) >= 10
        assert len(case["provenance"]) >= 15
        assert (
            "commit" in case["provenance"].lower()
            or "audit" in case["provenance"].lower()
        )


def test_benchmark_exact_action_and_category_counts(benchmark_data):
    """Verify exact counts of categories and actions across the canonical dataset."""
    cases = benchmark_data["cases"]
    assert len(cases) == 78

    action_counts = {"execute": 0, "clarify": 0, "reject": 0}
    cat_action_counts = {
        cat: {"execute": 0, "clarify": 0, "reject": 0, "total": 0}
        for cat in ALL_BENCHMARK_CATEGORIES
    }
    for c in cases:
        act = c["expected_action"]
        cat = c["category"]
        action_counts[act] += 1
        cat_action_counts[cat][act] += 1
        cat_action_counts[cat]["total"] += 1

    # Exact aggregate counts: 20 execute, 10 clarify, 48 reject
    assert action_counts == {"execute": 20, "clarify": 10, "reject": 48}

    # Exact per-category distribution
    assert cat_action_counts["A_standard_supported"] == {
        "execute": 10,
        "clarify": 0,
        "reject": 0,
        "total": 10,
    }
    assert cat_action_counts["B_incomplete_request"] == {
        "execute": 8,
        "clarify": 2,
        "reject": 0,
        "total": 10,
    }
    assert cat_action_counts["C_contradictory_request"] == {
        "execute": 0,
        "clarify": 0,
        "reject": 10,
        "total": 10,
    }
    assert cat_action_counts["D_invalid_parameters"] == {
        "execute": 0,
        "clarify": 0,
        "reject": 10,
        "total": 10,
    }
    assert cat_action_counts["E_snr_unit_confusion"] == {
        "execute": 2,
        "clarify": 4,
        "reject": 2,
        "total": 8,
    }
    assert cat_action_counts["F_unsupported_mod_channel"] == {
        "execute": 0,
        "clarify": 0,
        "reject": 10,
        "total": 10,
    }
    assert cat_action_counts["G_unsupported_assumptions"] == {
        "execute": 0,
        "clarify": 0,
        "reject": 10,
        "total": 10,
    }
    assert cat_action_counts["H_clarify_or_reject"] == {
        "execute": 0,
        "clarify": 4,
        "reject": 6,
        "total": 10,
    }


def test_review_integrity_schema_enforcement():
    """Verify that author-curated cases cannot falsely claim independent review."""
    with pytest.raises(ValidationError, match="Cannot claim 'independently_reviewed'"):
        BenchmarkCase(
            id="phase5_fake_review",
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
            decision_reason="Valid supported case",
            ground_truth_review_status="independently_reviewed",
            provenance="Author-curated against wireless/simulation.py; commit f360d6d.",
        )
