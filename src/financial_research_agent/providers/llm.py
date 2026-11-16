"""Provider-independent structured generation contract and safe telemetry."""

from __future__ import annotations

from enum import StrEnum
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from financial_research_agent.llm_decision import RecommendedAction
from financial_research_agent.tools import ToolName


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
    attempt_durations_seconds: tuple[float, ...] = ()


class ProposalMetadata(BaseModel):
    """Allowlisted projection of a planning response, including a schema-invalid one."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    action: RecommendedAction | None = None
    tool: ToolName | None = None
    unknown_tool: bool = False
    requests_satisfaction: bool = False

    @classmethod
    def from_payload(cls, payload: object) -> ProposalMetadata | None:
        if not isinstance(payload, dict) or "recommended_action" not in payload:
            return None
        raw_action, raw_tool = payload.get("recommended_action"), payload.get("next_tool")
        action = next((item for item in RecommendedAction if item.value == raw_action), None)
        tool = next((item for item in ToolName if item.value == raw_tool), None)
        return cls(
            action=action, tool=tool, unknown_tool=raw_tool is not None and tool is None,
            requests_satisfaction=action is RecommendedAction.STOP and raw_tool is None
            and "information_gap" in payload and payload["information_gap"] is None,
        )


class StructuredLLMResult(BaseModel, Generic[DecisionT]):
    decision: DecisionT
    performance: LLMPerformance


class LLMError(Exception):
    def __init__(
        self, message: str, performance: LLMPerformance, *, proposal: ProposalMetadata | None = None,
    ):
        super().__init__(message)
        self.category = performance.error_category
        self.performance = performance
        self.proposal = proposal


class LLMProvider(Protocol):
    def generate(self, prompt: str, schema: type[DecisionT]) -> StructuredLLMResult[DecisionT]: ...
