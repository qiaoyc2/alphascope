from datetime import date

import pytest
from pydantic import ValidationError

from financial_research_agent.models import (
    Evidence,
    Hypothesis,
    ResearchAction,
    ResearchGoal,
    ResearchState,
    ToolError,
    ToolResult,
)


def test_goal_allows_unresolved_company_ticker_and_dates() -> None:
    goal = ResearchGoal(query="Why did the stock fall?", objective="Explain the move")
    state = ResearchState(goal=goal)

    assert goal.company is None
    assert goal.ticker is None
    assert goal.start_date is None
    assert state.planned_actions == []
    assert state.tool_call_count == 0


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (ResearchGoal, {"query": " ", "objective": "Investigate"}),
        (ResearchGoal, {"query": "Why?", "objective": " "}),
        (
            ResearchGoal,
            {
                "query": "Why?",
                "objective": "Investigate",
                "start_date": date(2025, 2, 1),
                "end_date": date(2025, 1, 1),
            },
        ),
        (
            ResearchAction,
            {"action_id": "A1", "tool_name": "market", "purpose": "Check", "status": "bogus"},
        ),
        (Hypothesis, {"hypothesis_id": "H1", "statement": "A cause", "confidence": 1.1}),
        (Hypothesis, {"hypothesis_id": "H1", "statement": "A cause", "confidence": -0.1}),
        (Hypothesis, {"hypothesis_id": "H1", "statement": "A cause", "status": "bogus"}),
    ],
)
def test_invalid_model_fields_are_rejected(model: type, values: dict) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tool_call_count", -1),
        ("api_call_count", -1),
        ("estimated_cost_usd", -0.01),
        ("status", "bogus"),
    ],
)
def test_invalid_state_fields_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ResearchState.model_validate(
            {"goal": {"query": "Why?", "objective": "Investigate"}, field: value}
        )


def test_evidence_requires_traceable_provenance() -> None:
    common = {
        "evidence_id": "E1",
        "claim": "A reported event occurred",
        "evidence_type": "news",
        "observation_ids": ["O1"],
    }
    with pytest.raises(ValidationError, match="source URL or calculation inputs"):
        Evidence.model_validate(common)
    with pytest.raises(ValidationError, match="calculation method"):
        Evidence.model_validate({**common, "calculation_inputs": {"start": 10, "end": 11}})

    evidence = Evidence.model_validate({**common, "source_url": "https://example.org/report"})
    assert str(evidence.source_url) == "https://example.org/report"


def test_tool_result_failure_requires_error_and_success_excludes_it() -> None:
    base = {"result_id": "R1", "action_id": "A1", "tool_name": "market"}
    with pytest.raises(ValidationError, match="failed results require an error"):
        ToolResult.model_validate({**base, "success": False})
    with pytest.raises(ValidationError, match="successful results cannot contain an error"):
        ToolResult.model_validate(
            {**base, "success": True, "error": ToolError(code="timeout", message="Timed out")}
        )


def test_action_arguments_must_be_json_values() -> None:
    with pytest.raises(ValidationError):
        ResearchAction(
            action_id="A1", tool_name="market", purpose="Check", arguments={"bad": object()}
        )
