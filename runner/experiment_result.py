"""Structured machine-readable result schemas and automated evaluation logic."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from benchmark.schema import BenchmarkCase
from runner.experiment_config import ExperimentConfig


class LatencyBreakdown(BaseModel):
    """Execution latency measured across pipeline components in seconds."""

    rag_latency_sec: float = Field(default=0.0, ge=0.0)
    llm_generation_sec: float = Field(default=0.0, ge=0.0)
    validation_latency_sec: float = Field(default=0.0, ge=0.0)
    simulation_latency_sec: float = Field(default=0.0, ge=0.0)
    verification_latency_sec: float = Field(default=0.0, ge=0.0)
    total_latency_sec: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_latency_consistency(self) -> LatencyBreakdown:
        sum_stages = (
            self.rag_latency_sec
            + self.llm_generation_sec
            + self.validation_latency_sec
            + self.simulation_latency_sec
            + self.verification_latency_sec
        )
        # Total latency should be at least sum of measured stages
        # (allowing minor float precision)
        if self.total_latency_sec < sum_stages - 1e-4:
            raise ValueError(
                f"total_latency_sec ({self.total_latency_sec:.4f}) cannot be less than "
                f"sum of stages ({sum_stages:.4f})"
            )
        return self


class ValidationOutcome(BaseModel):
    """Detailed record of schema and physical validation."""

    passed: bool
    validation_errors: list[str] = Field(default_factory=list)
    correction_attempted: bool = False
    corrected_successfully: bool = False

    @model_validator(mode="after")
    def validate_correction_consistency(self) -> ValidationOutcome:
        if not self.passed and self.corrected_successfully:
            raise ValueError("corrected_successfully cannot be True if passed is False")
        if self.corrected_successfully and not self.correction_attempted:
            raise ValueError(
                "corrected_successfully cannot be True without correction_attempted"
            )
        return self


class VerificationOutcome(BaseModel):
    """Detailed record of statistical and analytical BER verification."""

    verified_valid: bool
    confidence_level: float = Field(default=0.95, ge=0.50, le=0.999)
    zero_errors_detected: bool = False
    rule_of_three_upper_bound: float | None = None
    all_points_in_ci: bool = True
    max_log_deviation: float | None = Field(default=None, ge=0.0)
    verification_flags: list[str] = Field(default_factory=list)


class CaseEvaluationResult(BaseModel):
    """Evaluation result for an individual benchmark prompt."""

    case_id: str
    category: str
    prompt: str
    expected_action: str
    actual_decision: str  # "execute", "clarify", "reject"
    raw_model_response: str
    parsed_config: dict[str, Any] | None = None
    expected_config: dict[str, Any] | None = None
    validation_outcome: ValidationOutcome
    simulation_output: dict[str, Any] | None = None
    verification_outcome: VerificationOutcome | None = None
    latency_breakdown: LatencyBreakdown
    error_classification: list[str] = Field(default_factory=list)
    is_silent_error: bool = False
    is_physically_correct: bool = False
    is_correct_decision: bool = False


class BenchmarkRunSummary(BaseModel):
    """Master record of an executed experiment."""

    experiment_id: str
    method: str
    timestamp: str
    config: ExperimentConfig
    total_cases_evaluated: int = Field(..., ge=0)
    metrics: dict[str, float]  # CVR, PCR, SER, TCR, BVE, Latency median/p95
    case_results: list[CaseEvaluationResult]


def classify_case_result(
    case: BenchmarkCase,
    actual_decision: str,
    parsed_config: dict[str, Any] | None,
    validation_outcome: ValidationOutcome,
    sim_output: dict[str, Any] | None,
    verification_outcome: VerificationOutcome | None,
    flagged_by_system: bool = False,
) -> dict[str, Any]:
    """Programmatically classify a case result against benchmark ground truth.

    Derives is_correct_decision, is_physically_correct, is_silent_error, and
    error_classification strictly from actual pipeline outcomes and ground truth.
    A caller-provided flagged_by_system=True is only respected if corroborated
    by actual validation or verification evidence. Missing or empty parsed_config
    is never treated as a matching executable configuration.
    """
    errors: list[str] = []
    is_correct_decision = actual_decision == case.expected_action
    is_physically_correct = False
    is_silent_error = False

    # Derive whether the pipeline actually produced a genuine flag
    has_validation_flag = not validation_outcome.passed or bool(
        validation_outcome.validation_errors
    )
    has_verification_flag = verification_outcome is not None and (
        not verification_outcome.verified_valid
        or verification_outcome.zero_errors_detected
        or not verification_outcome.all_points_in_ci
        or bool(verification_outcome.verification_flags)
    )
    # Supporting validation/verification evidence is mandatory to suppress silent errors
    is_substantiated_flag = has_validation_flag or has_verification_flag

    if case.expected_action in ("reject", "clarify"):
        if actual_decision == case.expected_action:
            is_physically_correct = True
        else:
            # Case should have been rejected or clarified,
            # but was executed or mishandled
            if actual_decision == "execute":
                if not is_substantiated_flag:
                    is_silent_error = True
                    errors.append("silent_execution_of_unsupported_or_invalid_prompt")
                else:
                    errors.append("execution_of_invalid_prompt_flagged_by_system")
            else:
                errors.append(
                    f"decision_mismatch_expected_{case.expected_action}_got_{actual_decision}"
                )

    elif case.expected_action == "execute":
        if actual_decision != "execute":
            errors.append(f"false_refusal_expected_execute_got_{actual_decision}")
        else:
            # Executed: check if configuration matches physical expectations.
            # Missing or empty parsed_config must never match!
            if not parsed_config or not isinstance(parsed_config, dict):
                config_matches = False
                errors.append("missing_or_empty_parsed_config")
            elif case.expected_config is None:
                config_matches = False
                errors.append("missing_expected_config_in_benchmark_case")
            else:
                config_matches = True
                # Require ALL expected configuration fields to be present and matched
                for k, exp_val in case.expected_config.items():
                    if k not in parsed_config:
                        config_matches = False
                        errors.append(f"missing_parameter_{k}")
                        continue
                    act_val = parsed_config[k]
                    if isinstance(exp_val, float) and isinstance(act_val, (int, float)):
                        if abs(exp_val - float(act_val)) > 0.05:
                            config_matches = False
                            errors.append(
                                f"parameter_mismatch_{k}_expected_{exp_val}_got_{act_val}"
                            )
                    elif exp_val != act_val:
                        config_matches = False
                        errors.append(
                            f"parameter_mismatch_{k}_expected_{exp_val}_got_{act_val}"
                        )

            if config_matches:
                is_physically_correct = True
            else:
                if not is_substantiated_flag:
                    is_silent_error = True
                    errors.append("silent_parameter_corruption")

            # Check zero-error starvation at high SNR without verification flagging
            if sim_output and not is_substantiated_flag:
                err_counts = sim_output.get("error_counts", [])
                if any(err == 0 for err in err_counts):
                    if (
                        verification_outcome is None
                        or not verification_outcome.zero_errors_detected
                    ):
                        is_silent_error = True
                        errors.append("silent_zero_error_starvation")

    if not errors:
        errors.append("none")

    return {
        "is_correct_decision": is_correct_decision,
        "is_physically_correct": is_physically_correct,
        "is_silent_error": is_silent_error,
        "error_classification": errors,
    }
