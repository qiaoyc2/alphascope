import json

import pytest
from pydantic import ValidationError

from financial_research_agent.agent.state import (
    add_action,
    add_evidence,
    add_hypothesis,
    add_observation,
    add_research_gap,
    record_information_gain,
    record_tool_result,
    update_hypothesis,
)
from financial_research_agent.models import (
    ActionStatus,
    Evidence,
    ExecutionMetadata,
    Hypothesis,
    HypothesisStatus,
    InformationGainRecord,
    Observation,
    ResearchAction,
    ResearchGoal,
    ResearchState,
    ResearchStatus,
    SourceReference,
    ToolError,
    ToolResult,
)


def initial_state() -> ResearchState:
    return ResearchState(goal=ResearchGoal(query="Why did ACME fall?", objective="Investigate"))


def researched_state() -> ResearchState:
    state = add_hypothesis(initial_state(), Hypothesis(hypothesis_id="H1", statement="Company news"))
    state = add_action(
        state,
        ResearchAction(
            action_id="A1",
            tool_name="market_data",
            arguments={"ticker": "ACME"},
            purpose="Verify the movement",
            related_hypothesis_ids=["H1"],
        ),
    )
    state = record_tool_result(
        state,
        ToolResult(
            result_id="R1",
            action_id="A1",
            tool_name="market_data",
            success=True,
            data={"start": 100, "end": 90},
            sources=[SourceReference(url="https://example.org/story")],
            metadata=ExecutionMetadata(api_call_count=1),
        ),
    )
    return add_observation(
        state,
        Observation(
            observation_id="O1", action_id="A1", result_id="R1", summary="Price declined"
        ),
    )


def test_state_operations_are_explicit_and_preserve_previous_state() -> None:
    original = initial_state()
    with_hypothesis = add_hypothesis(
        original, Hypothesis(hypothesis_id="H1", statement="Company news")
    )
    with_gap = add_research_gap(with_hypothesis, "Check filings")

    assert original.hypotheses == []
    assert original.research_gaps == []
    assert len(with_hypothesis.hypotheses) == 1
    assert with_gap.research_gaps == ["Check filings"]
    assert add_research_gap(with_gap, "Check filings").research_gaps == ["Check filings"]
    with pytest.raises(ValidationError):
        add_research_gap(with_gap, " ")


def test_result_moves_action_and_updates_usage() -> None:
    state = researched_state()

    assert state.planned_actions == []
    assert state.completed_actions[0].status is ActionStatus.SUCCEEDED
    assert state.status is ResearchStatus.IN_PROGRESS
    assert state.tool_call_count == 1
    assert state.api_call_count == 1
    assert state.observations[0].result_id == state.tool_results[0].result_id


def test_failure_result_is_recorded_without_becoming_evidence() -> None:
    state = add_action(
        initial_state(), ResearchAction(action_id="A1", tool_name="market", purpose="Check")
    )
    state = record_tool_result(
        state,
        ToolResult(
            result_id="R1",
            action_id="A1",
            tool_name="market",
            success=False,
            error=ToolError(code="timeout", message="Provider timed out", retryable=True),
        ),
    )

    assert state.completed_actions[0].status is ActionStatus.FAILED
    assert state.evidence == []
    assert state.tool_call_count == 1


def test_duplicate_ids_and_invalid_action_hypothesis_references() -> None:
    state = add_action(
        initial_state(), ResearchAction(action_id="A1", tool_name="market", purpose="Check")
    )
    with pytest.raises(ValidationError, match="duplicate action_id"):
        add_action(state, ResearchAction(action_id="A1", tool_name="news", purpose="Search"))
    with pytest.raises(ValidationError, match="unknown hypothesis"):
        add_action(
            state,
            ResearchAction(
                action_id="A2",
                tool_name="news",
                purpose="Search",
                related_hypothesis_ids=["H404"],
            ),
        )
    with pytest.raises(ValidationError, match="duplicate hypothesis_id"):
        add_hypothesis(
            add_hypothesis(initial_state(), Hypothesis(hypothesis_id="H1", statement="A")),
            Hypothesis(hypothesis_id="H1", statement="B"),
        )


def test_observation_must_link_to_matching_action_and_result() -> None:
    state = researched_state()
    with pytest.raises(ValidationError, match="unknown action"):
        add_observation(
            state,
            Observation(observation_id="O2", action_id="A404", result_id="R1", summary="Bad"),
        )
    with pytest.raises(ValidationError, match="unknown tool result"):
        add_observation(
            state,
            Observation(observation_id="O2", action_id="A1", result_id="R404", summary="Bad"),
        )
    with pytest.raises(ValidationError, match="duplicate observation_id"):
        add_observation(
            state,
            Observation(observation_id="O1", action_id="A1", result_id="R1", summary="Again"),
        )


def test_evidence_and_hypothesis_references_are_checked() -> None:
    state = researched_state()
    evidence = Evidence(
        evidence_id="E1",
        claim="A source reports an event",
        evidence_type="news",
        source_url="https://example.org/story",
        observation_ids=["O1"],
        supports_hypothesis_ids=["H1"],
    )
    state = add_evidence(state, evidence)
    hypothesis = state.hypotheses[0].model_copy(
        update={
            "status": HypothesisStatus.SUPPORTED,
            "confidence": 0.6,
            "supporting_evidence_ids": ["E1"],
        }
    )
    state = update_hypothesis(state, hypothesis)

    assert state.hypotheses[0].supporting_evidence_ids == ["E1"]
    with pytest.raises(ValidationError, match="unknown observation"):
        add_evidence(state, evidence.model_copy(update={"evidence_id": "E2", "observation_ids": ["O404"]}))
    with pytest.raises(ValidationError, match="unknown hypothesis"):
        add_evidence(
            state,
            evidence.model_copy(update={"evidence_id": "E2", "supports_hypothesis_ids": ["H404"]}),
        )
    with pytest.raises(ValidationError, match="duplicate evidence_id"):
        add_evidence(state, evidence)
    with pytest.raises(ValidationError, match="evidence source URL is absent"):
        add_evidence(
            state,
            evidence.model_copy(
                update={"evidence_id": "E2", "source_url": "https://example.org/unseen"}
            ),
        )
    with pytest.raises(ValidationError, match="unknown evidence"):
        update_hypothesis(
            state,
            hypothesis.model_copy(update={"supporting_evidence_ids": ["E404"]}),
        )
    with pytest.raises(ValidationError):
        update_hypothesis(state, hypothesis.model_copy(update={"confidence": 1.5}))


def test_information_gain_references_completed_action() -> None:
    with pytest.raises(ValidationError, match="unknown information gain action"):
        record_information_gain(initial_state(), InformationGainRecord(action_id="A404", score=0.5))

    state = record_information_gain(researched_state(), InformationGainRecord(action_id="A1", score=0.5))
    assert state.information_gain_history[0].score == 0.5


def test_state_json_round_trip_preserves_references_and_values() -> None:
    state = researched_state()
    state = add_evidence(
        state,
        Evidence(
            evidence_id="E1",
            claim="The price declined",
            evidence_type="calculation",
            calculation_method="(end - start) / start",
            calculation_inputs={"start": 100, "end": 90},
            observation_ids=["O1"],
        ),
    )

    encoded = state.model_dump_json()
    restored = ResearchState.model_validate_json(encoded)

    assert json.loads(encoded) == json.loads(restored.model_dump_json())
    assert restored.evidence[0].calculation_inputs == {"start": 100, "end": 90}
    assert restored.observations[0].result_id == "R1"
