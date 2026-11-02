from typing import Literal

from pydantic import BaseModel, Field, model_validator

Modulation = Literal["BPSK", "QPSK", "16QAM", "64QAM"]
Channel = Literal["AWGN", "Rayleigh", "Rician"]


class SimulationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    modulation: Modulation = "BPSK"
    channel: Channel = "AWGN"
    snr_start_db: float = 0
    snr_stop_db: float = 20
    snr_step_db: float = 2
    num_bits: int = Field(default=10_000, ge=1000, le=2_000_000)
    random_seed: int = 42
    rician_k_db: float = Field(default=6.0, ge=0, le=30)

    @model_validator(mode="after")
    def validate_snr_range(self):
        if self.snr_step_db <= 0:
            raise ValueError("snr_step_db must be positive")

        if self.snr_stop_db <= self.snr_start_db:
            raise ValueError("snr_stop_db must be greater than snr_start_db")

        if self.snr_start_db < -20 or self.snr_stop_db > 60:
            raise ValueError("SNR range must remain between -20 and 60 dB")

        return self


class CopilotRequest(BaseModel):
    """Natural-language request submitted to the AI copilot."""

    prompt: str = Field(min_length=1, max_length=2000)
    include_constellation: bool = False


# Configure the dominant-path strength for Rician fading.
# Higher K means a stronger dominant path relative to scattered paths.
