"""Machine-readable experiment configuration schemas for Phase 5 & 6 reproducibility."""

from __future__ import annotations

import datetime
import platform
import shutil
import subprocess
import sys
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ExperimentMethod = Literal["M1", "M2", "M3", "M4"]
ProviderType = Literal["ollama", "openai_compatible", "openai", "vllm", "compatible"]


class ModelConfig(BaseModel):
    """Specification of the LLM inference target."""

    provider: ProviderType = "ollama"
    model_name: str = "qwen2.5:3b"
    exact_version: str | None = None
    quantization: str | None = "q4_k_m"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=64, le=4096)
    context_window: int = Field(default=4096, ge=1024, le=32768)
    base_url: str = "http://localhost:11434"


class RAGConfig(BaseModel):
    """Configuration for retrieval augmentation."""

    enabled: bool = False
    embed_model: str = "nomic-embed-text"
    top_k: int = Field(default=3, ge=1, le=10)
    similarity_metric: str = "cosine"
    knowledge_corpus: str = "rag/knowledge_base/wireless_basics.md"


class ValidationConfig(BaseModel):
    """Configuration for physics-aware and schema validation."""

    enabled: bool = False
    allow_bounded_correction: bool = True
    # Research protocol strictly enforces at most one bounded correction attempt
    max_correction_attempts: int = Field(default=1, ge=0, le=1)
    enforce_physical_consistency: bool = True
    enforce_snr_disambiguation: bool = True
    enforce_sample_size_adequacy: bool = True


class VerificationConfig(BaseModel):
    """Configuration for independent post-simulation BER verification."""

    enabled: bool = False
    confidence_level: float = Field(default=0.95, ge=0.80, le=0.999)
    min_error_count_threshold: int = Field(default=10, ge=1, le=100)
    flag_zero_errors: bool = True
    apply_rule_of_three: bool = True
    max_ber_log_deviation: float = Field(default=0.5, ge=0.05, le=2.0)


class SimulationSettings(BaseModel):
    """Execution bounds and defaults for the trusted simulation engine."""

    default_num_bits: int = Field(default=10000, ge=1000)
    max_num_bits: int = Field(default=2000000, le=10000000)
    min_num_bits: int = Field(default=1000, ge=1000)
    random_seed: int = 42
    snr_min_db: float = Field(default=-20.0, ge=-50.0)
    snr_max_db: float = Field(default=60.0, le=100.0)

    @model_validator(mode="after")
    def validate_simulation_ranges(self) -> SimulationSettings:
        if self.min_num_bits > self.max_num_bits:
            raise ValueError(
                f"min_num_bits ({self.min_num_bits}) cannot exceed "
                f"max_num_bits ({self.max_num_bits})"
            )
        if not (self.min_num_bits <= self.default_num_bits <= self.max_num_bits):
            raise ValueError(
                f"default_num_bits ({self.default_num_bits}) must be between "
                f"min ({self.min_num_bits}) and max ({self.max_num_bits})"
            )
        if self.snr_min_db >= self.snr_max_db:
            raise ValueError(
                f"snr_min_db ({self.snr_min_db}) must be strictly less than "
                f"snr_max_db ({self.snr_max_db})"
            )
        return self


def detect_environment_metadata() -> EnvironmentMetadata:
    """Dynamically audit the host environment without reporting false data."""
    # Detect Git commit
    commit_hash = "unavailable"
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            commit_hash = res.stdout.strip()
    except Exception:
        commit_hash = "unavailable"

    # Detect GPU realistically
    gpu_available = False
    gpu_model: str | None = None

    if shutil.which("nvidia-smi"):
        try:
            gpu_res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if gpu_res.returncode == 0 and gpu_res.stdout.strip():
                gpu_available = True
                gpu_model = gpu_res.stdout.strip().split("\n")[0].strip()
        except Exception:
            gpu_available = False

    return EnvironmentMetadata(
        python_version=sys.version.split()[0],
        os_platform=f"{platform.system()} {platform.release()}",
        cpu_architecture=platform.machine(),
        gpu_available=gpu_available,
        gpu_model=gpu_model,
        git_commit=commit_hash,
    )


class EnvironmentMetadata(BaseModel):
    """Snapshot of host execution environment for audit and reproducibility."""

    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    os_platform: str = Field(
        default_factory=lambda: f"{platform.system()} {platform.release()}"
    )
    cpu_architecture: str = Field(default_factory=platform.machine)
    gpu_available: bool = False
    gpu_model: str | None = None
    git_commit: str = "unavailable"
    execution_timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )


class ExperimentConfig(BaseModel):
    """Master configuration for an experimental run comparing M1-M4."""

    experiment_id: str = Field(..., pattern=r"^exp_[a-zA-Z0-9_\-]+$")
    benchmark_version: str = "v1.0"
    benchmark_path: str = "data/benchmark/phase5_benchmark.json"
    method: ExperimentMethod
    model: ModelConfig = Field(default_factory=ModelConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    environment: EnvironmentMetadata = Field(
        default_factory=detect_environment_metadata
    )
    allow_experimental_overrides: bool = False
    custom_parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_method_preset_consistency(self) -> ExperimentConfig:
        """Enforce standard M1-M4 component combinations unless overridden."""
        if self.allow_experimental_overrides:
            return self

        if self.method == "M1":
            if self.rag.enabled or self.validation.enabled or self.verification.enabled:
                raise ValueError(
                    "Method M1 must have rag, validation, and verification disabled "
                    "(or set allow_experimental_overrides=True)"
                )
        elif self.method == "M2":
            if (
                not self.rag.enabled
                or self.validation.enabled
                or self.verification.enabled
            ):
                raise ValueError(
                    "Method M2 must have rag enabled and "
                    "validation/verification disabled "
                    "(or set allow_experimental_overrides=True)"
                )
        elif self.method == "M3":
            if (
                not self.rag.enabled
                or not self.validation.enabled
                or self.verification.enabled
            ):
                raise ValueError(
                    "Method M3 must have rag and validation enabled, "
                    "and verification disabled "
                    "(or set allow_experimental_overrides=True)"
                )
        elif self.method == "M4":
            if (
                not self.rag.enabled
                or not self.validation.enabled
                or not self.verification.enabled
            ):
                raise ValueError(
                    "Method M4 must have rag, validation, and verification enabled "
                    "(or set allow_experimental_overrides=True)"
                )

        return self

    @classmethod
    def create_method_preset(
        cls,
        experiment_id: str,
        method: ExperimentMethod,
        model_name: str = "qwen2.5:3b",
        provider: ProviderType = "ollama",
        base_url: str = "http://localhost:11434",
    ) -> ExperimentConfig:
        """Create a standard preset configuration for M1, M2, M3, or M4."""
        model_cfg = ModelConfig(
            provider=provider,
            model_name=model_name,
            base_url=base_url,
        )

        rag_cfg = RAGConfig(enabled=(method in ("M2", "M3", "M4")))
        val_cfg = ValidationConfig(enabled=(method in ("M3", "M4")))
        ver_cfg = VerificationConfig(enabled=(method == "M4"))

        return cls(
            experiment_id=experiment_id,
            method=method,
            model=model_cfg,
            rag=rag_cfg,
            validation=val_cfg,
            verification=ver_cfg,
        )
