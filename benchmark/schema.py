"""Pydantic schema for the Phase 5 wireless simulation benchmark dataset."""

from __future__ import annotations

import typing
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

BenchmarkCategory = Literal[
    "A_standard_supported",
    "B_incomplete_request",
    "C_contradictory_request",
    "D_invalid_parameters",
    "E_snr_unit_confusion",
    "F_unsupported_mod_channel",
    "G_unsupported_assumptions",
    "H_clarify_or_reject",
]

ALL_BENCHMARK_CATEGORIES: set[BenchmarkCategory] = set(
    typing.get_args(BenchmarkCategory)
)

ExpectedAction = Literal["execute", "clarify", "reject"]
ReviewStatus = Literal["independently_reviewed", "pending_review"]


class BenchmarkCase(BaseModel):
    """Schema for a single natural language benchmark case."""

    id: str = Field(..., pattern=r"^phase5_[a-z0-9_]+$")
    category: BenchmarkCategory
    prompt: str = Field(..., min_length=5, max_length=2000)
    expected_action: ExpectedAction
    expected_config: dict[str, Any] | None = None
    expected_modulation: str | None = None
    expected_channel: str | None = None
    expected_snr_convention: str | None = None
    expected_parameter_values: dict[str, Any] | None = None
    expected_physical_assumptions: str | None = None
    decision_reason: str = Field(..., min_length=10)
    reference_result: dict[str, Any] | None = None
    # No default value: review status must be explicitly declared and auditable
    ground_truth_review_status: ReviewStatus
    provenance: str = Field(..., min_length=15)

    @model_validator(mode="after")
    def validate_review_provenance_integrity(self) -> BenchmarkCase:
        """Enforce that independently_reviewed claims require independent evidence."""
        if self.ground_truth_review_status == "independently_reviewed":
            prov_lower = self.provenance.lower()
            if (
                "author-curated" in prov_lower
                and "independent reviewer" not in prov_lower
            ):
                msg = (
                    f"Case {self.id}: Cannot claim 'independently_reviewed' with "
                    "author-curated provenance without independent reviewer evidence."
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_action_consistency(self) -> BenchmarkCase:
        """Enforce consistency between action and expected config."""
        if self.expected_action == "execute":
            if self.expected_config is None:
                msg = (
                    f"Case {self.id}: expected_action='execute' "
                    "requires expected_config"
                )
                raise ValueError(msg)
            required_fields = {
                "modulation",
                "channel",
                "snr_start_db",
                "snr_stop_db",
                "snr_step_db",
                "num_bits",
                "rician_k_db",
            }
            missing = required_fields - set(self.expected_config.keys())
            if missing:
                missing_str = sorted(missing)
                raise ValueError(
                    f"Case {self.id}: expected_config missing fields: {missing_str}"
                )
            if self.expected_config["modulation"] not in ("BPSK", "QPSK"):
                raise ValueError(
                    f"Case {self.id}: executable modulation must be BPSK or QPSK"
                )
            if self.expected_config["channel"] not in ("AWGN", "Rayleigh", "Rician"):
                msg = (
                    f"Case {self.id}: executable channel must be "
                    "AWGN, Rayleigh, or Rician"
                )
                raise ValueError(msg)
            if (
                self.expected_config["snr_stop_db"]
                <= self.expected_config["snr_start_db"]
            ):
                msg = (
                    f"Case {self.id}: snr_stop_db must be greater than "
                    f"snr_start_db ({self.expected_config['snr_start_db']})"
                )
                raise ValueError(msg)
            if self.expected_config["snr_step_db"] <= 0:
                raise ValueError(f"Case {self.id}: snr_step_db must be positive")
            if self.expected_config["num_bits"] < 1000:
                raise ValueError(f"Case {self.id}: num_bits must be >= 1000")

        if (
            self.expected_action in ("clarify", "reject")
            and self.expected_config is not None
        ):
            msg = (
                f"Case {self.id}: expected_action='{self.expected_action}' "
                "must not contain an executable config"
            )
            raise ValueError(msg)

        return self


class BenchmarkDataset(BaseModel):
    """Schema for the entire versioned benchmark dataset."""

    version: str = Field(..., pattern=r"^v\d+\.\d+(\.\d+)?$")
    description: str = Field(..., min_length=10)
    created_at: str
    total_cases: int
    categories: list[BenchmarkCategory]
    cases: list[BenchmarkCase]

    @model_validator(mode="after")
    def validate_dataset_integrity(self) -> BenchmarkDataset:
        """Verify unique case IDs, exact category coverage, and consistency."""
        if len(self.cases) != self.total_cases:
            msg = (
                f"total_cases ({self.total_cases}) does not match "
                f"len(cases) ({len(self.cases)})"
            )
            raise ValueError(msg)

        # Enforce uniqueness of declared categories
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("Declared categories list contains duplicate entries")

        # Require all 8 benchmark categories to be declared
        declared_set = set(self.categories)
        if declared_set != ALL_BENCHMARK_CATEGORIES:
            missing = ALL_BENCHMARK_CATEGORIES - declared_set
            extra = declared_set - ALL_BENCHMARK_CATEGORIES
            msg = (
                f"Dataset must declare all 8 categories. Missing: {sorted(missing)}, "
                f"Unexpected: {sorted(extra)}"
            )
            raise ValueError(msg)

        seen_ids: set[str] = set()
        present_categories: set[BenchmarkCategory] = set()

        for case in self.cases:
            if case.id in seen_ids:
                raise ValueError(f"Duplicate case id: {case.id}")
            seen_ids.add(case.id)

            if case.category not in declared_set:
                msg = (
                    f"Case {case.id} category '{case.category}' "
                    "is not in declared categories"
                )
                raise ValueError(msg)
            present_categories.add(case.category)

        # Every declared category must have at least one case
        uncovered = declared_set - present_categories
        if uncovered:
            raise ValueError(
                f"Benchmark missing cases for categories: {sorted(uncovered)}"
            )

        return self
