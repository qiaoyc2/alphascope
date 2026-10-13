"""Explicit, validated research-state transitions.

Each operation returns a new state. Callers retain the previous state for tracing
or debugging without relying on in-place mutations.
"""

from __future__ import annotations

from financial_research_agent.models import (
    ActionStatus,
    Evidence,
    Hypothesis,
    InformationGainRecord,
    Observation,
    ResearchAction,
    ResearchState,
    ResearchStatus,
    ToolResult,
)


def _updated(state: ResearchState, **changes: object) -> ResearchState:
    data = state.model_dump(mode="python")
    data.update(changes)
    return ResearchState.model_validate(data)


def add_action(state: ResearchState, action: ResearchAction) -> ResearchState:
    return _updated(
        state,
        planned_actions=[*state.planned_actions, action],
        status=ResearchStatus.IN_PROGRESS,
    )


def record_tool_result(state: ResearchState, result: ToolResult) -> ResearchState:
    action = next(
        (item for item in state.planned_actions if item.action_id == result.action_id),
        None,
    )
    if action is None:
        raise ValueError("tool result requires a planned action")
    completed = action.model_copy(
        update={
            "status": ActionStatus.SUCCEEDED if result.success else ActionStatus.FAILED,
            "started_at": action.started_at or result.metadata.started_at,
            "finished_at": result.metadata.finished_at or result.recorded_at,
        }
    )
    return _updated(
        state,
        planned_actions=[item for item in state.planned_actions if item.action_id != result.action_id],
        completed_actions=[*state.completed_actions, completed],
        tool_results=[*state.tool_results, result],
        tool_call_count=state.tool_call_count + 1,
        api_call_count=state.api_call_count + result.metadata.api_call_count,
        estimated_cost_usd=state.estimated_cost_usd + result.metadata.estimated_cost_usd,
    )


def add_observation(state: ResearchState, observation: Observation) -> ResearchState:
    return _updated(state, observations=[*state.observations, observation])


def add_evidence(state: ResearchState, evidence: Evidence) -> ResearchState:
    return _updated(state, evidence=[*state.evidence, evidence])


def add_hypothesis(state: ResearchState, hypothesis: Hypothesis) -> ResearchState:
    return _updated(state, hypotheses=[*state.hypotheses, hypothesis])


def update_hypothesis(state: ResearchState, hypothesis: Hypothesis) -> ResearchState:
    if not any(item.hypothesis_id == hypothesis.hypothesis_id for item in state.hypotheses):
        raise ValueError(f"unknown hypothesis: {hypothesis.hypothesis_id}")
    return _updated(
        state,
        hypotheses=[
            hypothesis if item.hypothesis_id == hypothesis.hypothesis_id else item
            for item in state.hypotheses
        ],
    )


def add_research_gap(state: ResearchState, gap: str) -> ResearchState:
    gap = gap.strip()
    if gap in state.research_gaps:
        return state
    return _updated(state, research_gaps=[*state.research_gaps, gap])


def record_information_gain(
    state: ResearchState, record: InformationGainRecord
) -> ResearchState:
    return _updated(
        state, information_gain_history=[*state.information_gain_history, record]
    )
