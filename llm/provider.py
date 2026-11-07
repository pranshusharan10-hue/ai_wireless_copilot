"""Provider abstraction for model-independent LLM inference.

Supports local Ollama endpoints and OpenAI-compatible remote endpoints (e.g.,
vLLM, Ollama, or DGX/Kaggle hosted endpoints) without modifying wireless simulation
or benchmark code.
"""

from __future__ import annotations

import abc
import math
import os
from typing import Any
from urllib.parse import urlparse

import httpx

VALID_PROVIDER_NAMES = {"ollama", "openai_compatible", "openai", "vllm", "compatible"}


class ProviderError(RuntimeError):
    """Base exception for LLM provider errors."""


class ProviderConfigError(ProviderError):
    """Raised when provider configuration or parameters are invalid."""


class ProviderNetworkError(ProviderError):
    """Raised when network, connectivity, or timeout failures occur."""


class ProviderResponseError(ProviderError):
    """Raised when an LLM returns an empty, incomplete, or malformed response."""


def _sanitize_secret(message: str, secret: str | None) -> str:
    """Ensure secret keys do not leak in error messages."""
    if not secret or secret == "EMPTY" or len(secret) < 3:
        return message
    return message.replace(secret, "[REDACTED_API_KEY]")


def _validate_base_url(url: str, provider_name: str) -> str:
    """Validate that base_url has a valid http/https scheme and a host."""
    if not url or not isinstance(url, str) or not url.strip():
        raise ProviderConfigError(
            f"{provider_name} base_url must be a non-empty string"
        )
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ProviderConfigError(
            f"Invalid {provider_name} base_url: '{url}'. "
            "Must include valid scheme (http:// or https://) and host."
        )
    return url.strip().rstrip("/")


def _validate_timeout(timeout: Any, provider_name: str) -> float:
    """Validate that timeout is a finite, positive number (not NaN or inf)."""
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ProviderConfigError(
            f"{provider_name} timeout must be a finite positive number, got {timeout}"
        )
    return float(timeout)


def _validate_temperature(temperature: Any) -> float:
    """Validate that temperature is a finite number in [0.0, 2.0] (not NaN or inf)."""
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or not (0.0 <= temperature <= 2.0)
    ):
        raise ProviderConfigError(
            f"temperature must be a finite number between 0.0 and 2.0, "
            f"got {temperature}"
        )
    return float(temperature)


class LLMProvider(abc.ABC):
    """Abstract base class for LLM inference providers."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Generate raw text response from the model."""
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Inference provider for local or remote Ollama instances."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        raw_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = _validate_base_url(raw_url, "Ollama")

        raw_model = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        if not raw_model or not isinstance(raw_model, str) or not raw_model.strip():
            raise ProviderConfigError("Ollama model_name must be a non-empty string")
        self.model_name = raw_model.strip()

        self.timeout = _validate_timeout(timeout, "Ollama")

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise ProviderConfigError("prompt must be a non-empty string")
        valid_temp = _validate_temperature(temperature)
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens <= 0
        ):
            raise ProviderConfigError(f"max_tokens must be positive, got {max_tokens}")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": valid_temp,
                "num_predict": max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderNetworkError(
                f"Ollama request timed out after {self.timeout}s at {self.base_url}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderNetworkError(
                f"Ollama HTTP error {exc.response.status_code} from {self.base_url}"
            ) from exc
        except ValueError as exc:
            raise ProviderResponseError(
                f"Malformed JSON response from Ollama at {self.base_url}: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderNetworkError(
                f"Failed to communicate with Ollama at {self.base_url}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ProviderResponseError("Ollama response must be a JSON object")

        text = data.get("response")
        if text is None or not isinstance(text, str) or not text.strip():
            raise ProviderResponseError(
                "Ollama response missing or empty/whitespace-only 'response' field"
            )

        return text.strip()


class OpenAICompatibleProvider(LLMProvider):
    """Provider for OpenAI-compatible endpoints (vLLM, TGI, Kaggle/DGX servers)."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        raw_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "http://localhost:8000/v1"
        )
        self.base_url = _validate_base_url(raw_url, "OpenAI-compatible")

        raw_model = (
            model_name
            or os.getenv("OPENAI_MODEL")
            or os.getenv("LLM_MODEL")
            or "mistralai/Mistral-7B-Instruct-v0.3"
        )
        if not raw_model or not isinstance(raw_model, str) or not raw_model.strip():
            raise ProviderConfigError(
                "OpenAI-compatible model_name must be a non-empty string"
            )
        self.model_name = raw_model.strip()

        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("LLM_API_KEY")
            or "EMPTY"
        )

        self.timeout = _validate_timeout(timeout, "OpenAI-compatible")

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise ProviderConfigError("prompt must be a non-empty string")
        valid_temp = _validate_temperature(temperature)
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens <= 0
        ):
            raise ProviderConfigError(f"max_tokens must be positive, got {max_tokens}")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": valid_temp,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            msg = f"Remote provider timed out after {self.timeout}s at {self.base_url}"
            raise ProviderNetworkError(_sanitize_secret(msg, self.api_key)) from exc
        except httpx.HTTPStatusError as exc:
            msg = f"Remote provider HTTP {exc.response.status_code} at {self.base_url}"
            raise ProviderNetworkError(_sanitize_secret(msg, self.api_key)) from exc
        except ValueError as exc:
            msg = (
                f"Malformed JSON response from remote provider at {self.base_url}: "
                f"{exc}"
            )
            raise ProviderResponseError(_sanitize_secret(msg, self.api_key)) from exc
        except httpx.RequestError as exc:
            msg = f"Remote provider error at {self.base_url}: {exc}"
            raise ProviderNetworkError(_sanitize_secret(msg, self.api_key)) from exc

        if not isinstance(data, dict):
            raise ProviderResponseError("Response payload must be a JSON object")

        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise ProviderResponseError(
                "Response payload missing non-empty 'choices' list"
            )

        first_choice = choices[0]
        if not isinstance(first_choice, dict) or "message" not in first_choice:
            raise ProviderResponseError(
                "Choice entry missing required 'message' object"
            )

        message = first_choice["message"]
        if not isinstance(message, dict) or "content" not in message:
            raise ProviderResponseError(
                "Message object missing required 'content' field"
            )

        content = message["content"]
        if not isinstance(content, str) or not content.strip():
            raise ProviderResponseError(
                "OpenAI-compatible message content is missing or empty/whitespace-only"
            )

        return content.strip()


def get_llm_provider(
    provider_type: str = "ollama",
    **kwargs: Any,
) -> LLMProvider:
    """Factory for instantiating an LLM inference provider."""
    norm_type = provider_type.lower().strip()
    if norm_type not in VALID_PROVIDER_NAMES:
        raise ProviderConfigError(
            f"Unknown LLM provider type: '{provider_type}'. "
            f"Supported: {sorted(VALID_PROVIDER_NAMES)}"
        )

    if norm_type == "ollama":
        return OllamaProvider(
            base_url=kwargs.get("base_url"),
            model_name=kwargs.get("model_name"),
            timeout=kwargs.get("timeout", 120.0),
        )

    return OpenAICompatibleProvider(
        base_url=kwargs.get("base_url"),
        model_name=kwargs.get("model_name"),
        api_key=kwargs.get("api_key"),
        timeout=kwargs.get("timeout", 120.0),
    )
