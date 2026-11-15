"""Provider-independent structured generation contract and safe telemetry."""

from __future__ import annotations

from enum import StrEnum
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field


DecisionT = TypeVar("DecisionT", bound=BaseModel)


class LLMErrorCategory(StrEnum):
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_ERROR = "model_error"
    HTTP_ERROR = "http_error"
    INVALID_JSON = "invalid_json"
    SCHEMA_VALIDATION = "schema_validation"
    EMPTY_RESPONSE = "empty_response"
    INVALID_RESPONSE = "invalid_response"


class LLMPerformance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    request_duration_seconds: float = Field(ge=0)
    total_duration_seconds: float = Field(ge=0)
    attempts: int = Field(ge=1)
    input_tokens: int | None = None
    output_tokens: int | None = None
    generation_duration_seconds: float | None = None
    tokens_per_second: float | None = None
    validation_passed: bool
    error_category: LLMErrorCategory | None = None


class StructuredLLMResult(BaseModel, Generic[DecisionT]):
    decision: DecisionT
    performance: LLMPerformance


class LLMError(Exception):
    def __init__(self, message: str, performance: LLMPerformance):
        super().__init__(message)
        self.category = performance.error_category
        self.performance = performance


class LLMProvider(Protocol):
    def generate(self, prompt: str, schema: type[DecisionT]) -> StructuredLLMResult[DecisionT]: ...
