"""Data contracts for research; no model output or tool is executed here."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    JsonValue,
    StringConstraints,
    model_validator,
)


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResearchModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, revalidate_instances="always"
    )


class ActionStatus(StrEnum):
    PLANNED = "planned"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class HypothesisStatus(StrEnum):
    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    WEAKENED = "weakened"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class ResearchStatus(StrEnum):
    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class ResearchGoal(ResearchModel):
    query: NonEmptyStr
    objective: NonEmptyStr
    company: NonEmptyStr | None = None
    ticker: NonEmptyStr | None = None
    start_date: date | None = None
    end_date: date | None = None
    assumptions: list[NonEmptyStr] = Field(default_factory=list)
    unresolved_details: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_date_range(self) -> Self:
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        return self


class ResearchAction(ResearchModel):
    action_id: NonEmptyStr
    tool_name: NonEmptyStr
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    purpose: NonEmptyStr
    related_hypothesis_ids: list[NonEmptyStr] = Field(default_factory=list)
    status: ActionStatus = ActionStatus.PLANNED
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class SourceReference(ResearchModel):
    url: HttpUrl
    title: NonEmptyStr | None = None
    published_at: datetime | None = None


class ToolError(ResearchModel):
    code: NonEmptyStr
    message: NonEmptyStr
    retryable: bool = False


class ExecutionMetadata(ResearchModel):
    api_call_count: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0, allow_inf_nan=False)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ToolResult(ResearchModel):
    result_id: NonEmptyStr
    action_id: NonEmptyStr
    tool_name: NonEmptyStr
    success: bool
    data: dict[str, JsonValue] = Field(default_factory=dict)
    sources: list[SourceReference] = Field(default_factory=list)
    error: ToolError | None = None
    metadata: ExecutionMetadata = Field(default_factory=ExecutionMetadata)
    recorded_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def check_error(self) -> Self:
        if self.success and self.error is not None:
            raise ValueError("successful results cannot contain an error")
        if not self.success and self.error is None:
            raise ValueError("failed results require an error")
        return self


class Observation(ResearchModel):
    observation_id: NonEmptyStr
    action_id: NonEmptyStr
    result_id: NonEmptyStr
    summary: NonEmptyStr
    recorded_at: datetime = Field(default_factory=utc_now)


class Evidence(ResearchModel):
    evidence_id: NonEmptyStr
    claim: NonEmptyStr
    evidence_type: NonEmptyStr
    source_url: HttpUrl | None = None
    calculation_method: NonEmptyStr | None = None
    calculation_inputs: dict[str, JsonValue] = Field(default_factory=dict)
    observation_ids: list[NonEmptyStr] = Field(min_length=1)
    supports_hypothesis_ids: list[NonEmptyStr] = Field(default_factory=list)
    contradicts_hypothesis_ids: list[NonEmptyStr] = Field(default_factory=list)
    retrieved_at: datetime | None = None
    event_at: datetime | None = None

    @model_validator(mode="after")
    def check_provenance(self) -> Self:
        if self.source_url is None and not self.calculation_inputs:
            raise ValueError("evidence requires a source URL or calculation inputs")
        if self.calculation_inputs and self.calculation_method is None:
            raise ValueError("calculation inputs require a calculation method")
        if set(self.supports_hypothesis_ids) & set(self.contradicts_hypothesis_ids):
            raise ValueError("evidence cannot both support and contradict a hypothesis")
        return self


class Hypothesis(ResearchModel):
    hypothesis_id: NonEmptyStr
    statement: NonEmptyStr
    status: HypothesisStatus = HypothesisStatus.UNVERIFIED
    # This is a heuristic assessment, not a calibrated probability.
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    supporting_evidence_ids: list[NonEmptyStr] = Field(default_factory=list)
    contradicting_evidence_ids: list[NonEmptyStr] = Field(default_factory=list)
    research_gaps: list[NonEmptyStr] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def check_evidence_roles(self) -> Self:
        if set(self.supporting_evidence_ids) & set(self.contradicting_evidence_ids):
            raise ValueError("evidence cannot be both supporting and contradicting")
        return self


class InformationGainRecord(ResearchModel):
    action_id: NonEmptyStr
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    recorded_at: datetime = Field(default_factory=utc_now)


def _unique_ids(items: list[ResearchModel], field: str) -> set[str]:
    values = [getattr(item, field) for item in items]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {field}")
    return set(values)


def _check_refs(refs: list[str], known: set[str], label: str) -> None:
    if len(refs) != len(set(refs)):
        raise ValueError(f"duplicate {label} reference")
    missing = set(refs) - known
    if missing:
        raise ValueError(f"unknown {label} reference: {', '.join(sorted(missing))}")


class ResearchState(ResearchModel):
    goal: ResearchGoal
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    planned_actions: list[ResearchAction] = Field(default_factory=list)
    completed_actions: list[ResearchAction] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    research_gaps: list[NonEmptyStr] = Field(default_factory=list)
    tool_call_count: int = Field(default=0, ge=0)
    api_call_count: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0, allow_inf_nan=False)
    started_at: datetime = Field(default_factory=utc_now)
    information_gain_history: list[InformationGainRecord] = Field(default_factory=list)
    status: ResearchStatus = ResearchStatus.INITIALIZED
    stop_reason: NonEmptyStr | None = None

    @model_validator(mode="after")
    def check_references(self) -> Self:
        hypothesis_ids = _unique_ids(self.hypotheses, "hypothesis_id")
        actions = self.planned_actions + self.completed_actions
        action_ids = _unique_ids(actions, "action_id")
        result_ids = _unique_ids(self.tool_results, "result_id")
        observation_ids = _unique_ids(self.observations, "observation_id")
        evidence_ids = _unique_ids(self.evidence, "evidence_id")

        results_by_id = {result.result_id: result for result in self.tool_results}
        results_by_action = {result.action_id: result for result in self.tool_results}
        observations_by_id = {item.observation_id: item for item in self.observations}
        if len(results_by_action) != len(self.tool_results):
            raise ValueError("multiple results for one action")
        completed_by_id = {action.action_id: action for action in self.completed_actions}

        for action in self.planned_actions:
            if action.status is not ActionStatus.PLANNED:
                raise ValueError("planned action must have planned status")
        for action in self.completed_actions:
            if action.status not in (ActionStatus.SUCCEEDED, ActionStatus.FAILED):
                raise ValueError("completed action must have a terminal status")
            if action.action_id not in results_by_action:
                raise ValueError("completed action requires a tool result")
        for action in actions:
            _check_refs(action.related_hypothesis_ids, hypothesis_ids, "hypothesis")

        for result in self.tool_results:
            action = completed_by_id.get(result.action_id)
            if action is None:
                raise ValueError("tool result refers to a non-completed action")
            if result.tool_name != action.tool_name:
                raise ValueError("tool result name differs from action tool name")
            expected = ActionStatus.SUCCEEDED if result.success else ActionStatus.FAILED
            if action.status is not expected:
                raise ValueError("tool result success differs from action status")

        for observation in self.observations:
            _check_refs([observation.action_id], action_ids, "action")
            _check_refs([observation.result_id], result_ids, "tool result")
            if results_by_id[observation.result_id].action_id != observation.action_id:
                raise ValueError("observation action differs from its tool result")

        for item in self.evidence:
            _check_refs(item.observation_ids, observation_ids, "observation")
            _check_refs(item.supports_hypothesis_ids, hypothesis_ids, "hypothesis")
            _check_refs(item.contradicts_hypothesis_ids, hypothesis_ids, "hypothesis")
            if item.source_url is not None:
                observed_urls = {
                    str(source.url)
                    for observation_id in item.observation_ids
                    for source in results_by_id[
                        observations_by_id[observation_id].result_id
                    ].sources
                }
                if str(item.source_url) not in observed_urls:
                    raise ValueError("evidence source URL is absent from linked tool results")
        for hypothesis in self.hypotheses:
            _check_refs(hypothesis.supporting_evidence_ids, evidence_ids, "evidence")
            _check_refs(hypothesis.contradicting_evidence_ids, evidence_ids, "evidence")
        gain_action_ids = [entry.action_id for entry in self.information_gain_history]
        _check_refs(gain_action_ids, set(completed_by_id), "information gain action")
        return self
