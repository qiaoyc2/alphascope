"""Local Ollama structured output; no tool execution or state mutations."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests
from pydantic import ValidationError

from financial_research_agent.config import LLMSettings
from financial_research_agent.providers.llm import (
    DecisionT, LLMError, LLMErrorCategory, LLMPerformance, StructuredLLMResult,
)


logger = logging.getLogger(__name__)


def _count(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    return value if type(value) is int and value >= 0 else None


class OllamaProvider:
    def __init__(self, settings: LLMSettings, *, session: requests.Session | None = None):
        self.settings = settings
        self.session = session if session is not None else requests.Session()
        # Local traffic must not follow environment-configured HTTP proxies.
        self.session.trust_env = False

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> OllamaProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def generate(self, prompt: str, schema: type[DecisionT]) -> StructuredLLMResult[DecisionT]:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": "Return a single JSON object matching this output schema:\n"
                 + json.dumps(schema.model_json_schema())},
                {"role": "user", "content": prompt},
            ],
            "format": schema.model_json_schema(),
            "stream": False,
            "think": self.settings.thinking,
            "options": {
                "temperature": self.settings.temperature,
                "num_ctx": self.settings.num_ctx,
                "num_predict": self.settings.num_predict,
            },
        }
        overall_start = time.perf_counter()
        for attempt in range(1, self.settings.max_retries + 2):
            start = time.perf_counter()
            data: dict[str, Any] = {}
            category: LLMErrorCategory | None = None
            retryable = False
            message = "Ollama response rejected"
            decision = None
            try:
                with self.session.post(
                    self.settings.base_url.rstrip("/") + "/api/chat",
                    json=payload,
                    timeout=self.settings.timeout_seconds,
                    allow_redirects=False,
                ) as response:
                    status = response.status_code
                    if status != 200:
                        if status == 404:
                            category = LLMErrorCategory.MODEL_NOT_FOUND
                            message = f"Configured model unavailable; run: ollama pull {self.settings.model}"
                        else:
                            category = LLMErrorCategory.HTTP_ERROR
                            retryable = status == 429 or status >= 500
                            message = f"Ollama HTTP {status}"
                    else:
                        try:
                            body = response.json()
                        except ValueError:
                            category = LLMErrorCategory.INVALID_JSON
                            message = "Ollama returned an invalid JSON envelope"
                        else:
                            if not isinstance(body, dict):
                                category = LLMErrorCategory.INVALID_RESPONSE
                            else:
                                data = body
                                category, message, decision = self._validate(data, schema)
            except requests.Timeout:
                category = LLMErrorCategory.TIMEOUT
                message = "Ollama request timed out"
                retryable = True
            except requests.ConnectionError:
                category = LLMErrorCategory.CONNECTION
                message = "Ollama unavailable; start it with: ollama serve"
                retryable = True
            except requests.RequestException:
                category = LLMErrorCategory.HTTP_ERROR
                message = "Ollama request failed"

            output_tokens = _count(data, "eval_count")
            duration_ns = _count(data, "eval_duration")
            generation_seconds = duration_ns / 1e9 if duration_ns is not None else None
            performance = LLMPerformance(
                model=self.settings.model,
                request_duration_seconds=time.perf_counter() - start,
                total_duration_seconds=time.perf_counter() - overall_start,
                attempts=attempt,
                input_tokens=_count(data, "prompt_eval_count"),
                output_tokens=output_tokens,
                generation_duration_seconds=generation_seconds,
                tokens_per_second=(output_tokens / generation_seconds
                                   if output_tokens is not None and generation_seconds else None),
                validation_passed=category is None,
                error_category=category,
            )
            # No prompt, response text, reasoning, exception body, or secrets.
            logger.info("llm_request %s", performance.model_dump_json())
            if category is None and decision is not None:
                return StructuredLLMResult[schema](decision=decision, performance=performance)
            if not retryable or attempt > self.settings.max_retries:
                raise LLMError(message, performance) from None
            time.sleep(0.25 * attempt)
        raise AssertionError("unreachable")

    @staticmethod
    def _validate(data: dict[str, Any], schema: type[DecisionT]) -> tuple[LLMErrorCategory | None, str, DecisionT | None]:
        if "error" in data:
            return LLMErrorCategory.MODEL_ERROR, "Ollama reported a model error", None
        if data.get("done") is not True or data.get("done_reason") not in (None, "stop"):
            return LLMErrorCategory.INVALID_RESPONSE, "Ollama generation was incomplete", None
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if content is None or (isinstance(content, str) and not content.strip()):
            return LLMErrorCategory.EMPTY_RESPONSE, "Ollama returned no decision content", None
        if not isinstance(content, str):
            return LLMErrorCategory.INVALID_RESPONSE, "Ollama content must be a string", None
        try:
            json.loads(content)
        except ValueError:
            return LLMErrorCategory.INVALID_JSON, "Decision content is not valid JSON", None
        try:
            decision = schema.model_validate_json(content)
        except ValidationError:
            return LLMErrorCategory.SCHEMA_VALIDATION, "Decision violates the expected schema", None
        return None, "", decision
