"""Validated recommendations, separate from executable ResearchAction/state."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, StrictBool, StringConstraints, model_validator

from financial_research_agent.models import ResearchModel
from financial_research_agent.tools import ToolName


ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1200)]


class RecommendedAction(StrEnum):
    CONTINUE = "continue"
    REVISE = "revise"
    CLARIFY = "clarify"
    STOP = "stop"


class LLMDecision(ResearchModel):
    # Nullable fields are required: omitted and explicitly unknown differ.
    premise_valid: StrictBool | None = Field(
        description="Do supplied observations support the factual event assumed by the original question? False if the assumed event is contradicted; null if not assessable. Not a judgment of observation consistency or authenticity."
    )
    recommended_action: RecommendedAction = Field(
        description="Continue with a tool-resolvable gap; clarify for necessary user intent; revise proposes a change for approval; stop ends research."
    )
    next_tool: ToolName | None = Field(description="Available tool addressing the next information gap, or null when awaiting the user or stopping")
    reason: ShortText = Field(description="Short rationale: state the observed movement and numerical values when available, compare with the original question's assumed event, and justify the next step")
    information_gap: ShortText | None
    should_stop: StrictBool = Field(description="True only for the terminal stop action")

    @model_validator(mode="after")
    def consistent_recommendation(self) -> Self:
        if self.should_stop != (self.recommended_action == RecommendedAction.STOP):
            raise ValueError("should_stop must agree with the stop action")
        if self.recommended_action in {RecommendedAction.STOP, RecommendedAction.CLARIFY} and self.next_tool is not None:
            raise ValueError("stop/clarify cannot request a tool")
        if self.recommended_action == RecommendedAction.CONTINUE and self.next_tool is None:
            raise ValueError("continue requires a next tool")
        if self.recommended_action == RecommendedAction.CLARIFY and self.information_gap is None:
            raise ValueError("clarify requires a specific information gap")
        return self
