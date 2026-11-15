"""Explicit configuration loading, independent of terminal/editor settings."""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from financial_research_agent.providers.llm import LLMProvider


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    provider: Literal["ollama"] = "ollama"
    model: str = Field(default="qwen3:8b", min_length=1, pattern=r"^\S+$")
    temperature: float = Field(default=0, ge=0, le=2, allow_inf_nan=False)
    num_ctx: int = Field(default=4096, ge=512, le=131072)
    base_url: str = "http://localhost:11434"
    thinking: bool = False
    timeout_seconds: float = Field(default=60, gt=0, le=600, allow_inf_nan=False)
    max_retries: int = Field(default=1, ge=0, le=2)
    num_predict: int = Field(default=512, ge=64, le=4096)

    @model_validator(mode="after")
    def local_endpoint_only(self) -> Self:
        """Remote research transmission is outside this milestone's authorization."""
        parsed = urlsplit(self.base_url)
        host = parsed.hostname or ""
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.lower() == "localhost"
        if (
            parsed.scheme not in {"http", "https"}
            or not loopback
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("OLLAMA_BASE_URL must be a loopback HTTP(S) origin")
        # Accessing .port also validates malformed/out-of-range ports.
        parsed.port
        return self


_ENV_FIELDS = {
    "LLM_PROVIDER": "provider",
    "LLM_MODEL": "model",
    "LLM_TEMPERATURE": "temperature",
    "LLM_NUM_CTX": "num_ctx",
    "OLLAMA_BASE_URL": "base_url",
    "LLM_THINKING": "thinking",
    "LLM_TIMEOUT_SECONDS": "timeout_seconds",
    "LLM_MAX_RETRIES": "max_retries",
    "LLM_NUM_PREDICT": "num_predict",
}


def load_llm_settings(env_file: Path | str | None = None) -> LLMSettings:
    """Load the source checkout's .env (or an explicit file); shell values win.

    Loading is explicit, not an import side effect. Unrelated settings, including
    the SEC User-Agent, are preserved. No configuration values are logged.
    """
    path = Path(env_file) if env_file is not None else Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(path, override=False, interpolate=False, encoding="utf-8-sig")
    return LLMSettings.model_validate(
        {field: os.environ[key] for key, field in _ENV_FIELDS.items() if key in os.environ}
    )


@contextmanager
def open_llm_provider(settings: LLMSettings) -> Iterator[LLMProvider]:
    """Open the configured provider at the application boundary."""
    from financial_research_agent.providers.ollama import OllamaProvider

    with OllamaProvider(LLMSettings.model_validate(settings.model_dump())) as provider:
        yield provider
