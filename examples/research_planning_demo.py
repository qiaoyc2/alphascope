"""Read-only M3B planning validation with the configured local Ollama model."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date

from pydantic import ValidationError

from financial_research_agent.agent.goal import interpret_goal
from financial_research_agent.agent.hypotheses import initialize_research
from financial_research_agent.agent.planning import PlanDecision, ResearchPlanner, record_observation, record_plan
from financial_research_agent.agent.state import add_action
from financial_research_agent.config import load_llm_settings, open_llm_provider
from financial_research_agent.models import (
    MarketMovement, Observation, ObservationProvenance, ResearchAction, ResearchGoal,
    ResearchState, ToolResult,
)
from financial_research_agent.providers.llm import LLMError, LLMProvider
from financial_research_agent.tools import ToolName


FIXTURE_START = date(2026, 8, 28)
FIXTURE_END = date(2026, 10, 12)


def start_scenario(label: str, question: str, provider: LLMProvider, *, fixture: bool = False) -> ResearchState:
    goal, goal_perf = interpret_goal(question, provider)
    if fixture:
        # The supplied evaluation period deliberately replaces the ordinary "recently" convention.
        data = goal.model_dump(mode="python")
        data.update(
            start_date=FIXTURE_START, end_date=FIXTURE_END,
            assumptions=["Scope selected explicitly for the supplied historical fixture."],
        )
        goal = ResearchGoal.model_validate(data)
    state, hypothesis_perf = initialize_research(goal, provider)
    print(json.dumps({
        "scenario": label, "goal": goal.model_dump(mode="json"),
        "hypotheses": [item.model_dump(mode="json") for item in state.hypotheses],
        "research_gaps": state.research_gaps,
        "goal_performance": goal_perf.model_dump(mode="json"),
        "hypothesis_performance": hypothesis_perf.model_dump(mode="json") if hypothesis_perf else None,
    }, ensure_ascii=True))
    return state


def inject_historical_fixture(state: ResearchState) -> ResearchState:
    action = ResearchAction(
        action_id="fixture-market", tool_name=ToolName.MARKET_DATA,
        arguments={"ticker": "GOOGL", "benchmark": "QQQ",
                   "start_date": FIXTURE_START.isoformat(), "end_date": FIXTURE_END.isoformat()},
        purpose="Register the historical comparison supplied for this evaluation.",
    )
    result = ToolResult(result_id="fixture-result", action_id=action.action_id,
                        tool_name=action.tool_name, success=True)
    observation = Observation(
        observation_id="fixture-observation", action_id=action.action_id, result_id=result.result_id,
        summary="Historical evaluation fixture: GOOGL +1.83%, QQQ +4.24%, relative −2.41 percentage points.",
        provenance=ObservationProvenance.HISTORICAL_FIXTURE,
        market_movement=MarketMovement(
            ticker="GOOGL", start_date=FIXTURE_START, end_date=FIXTURE_END, return_pct=1.83,
            benchmark="QQQ", benchmark_return_pct=4.24,
        ),
    )
    return record_observation(add_action(state, action), result, observation)


def report_plan(
    label: str, planner: ResearchPlanner, state: ResearchState, expected: Callable[[PlanDecision], bool],
) -> tuple[ResearchState, bool]:
    result = planner.plan(state)
    revised = record_plan(state, result.decision)
    passed = expected(result.decision)
    print(json.dumps({
        "scenario": label, "matches_expectation": passed, **result.model_dump(mode="json"),
        "hypotheses_after": [
            {"id": item.hypothesis_id, "status": item.status, "research_gaps": item.research_gaps}
            for item in revised.hypotheses
        ],
        "research_gaps_after": revised.research_gaps,
    }, ensure_ascii=True))
    return revised, passed


def stock_scenarios(provider: LLMProvider, planner: ResearchPlanner) -> bool:
    state = start_scenario("stock_goal", "Why did GOOGL stock decline recently?", provider, fixture=True)
    try:
        state, initial_passed = report_plan(
            "initial_stock_plan", planner, state,
            lambda decision: decision.next_tool is ToolName.MARKET_DATA and decision.premise_valid is None,
        )
    except (ValueError, LLMError) as exc:
        report_error("initial_stock_plan", exc)
        initial_passed = False
    _, revised_passed = report_plan(
        "historical_fixture_replan", planner, inject_historical_fixture(state),
        lambda decision: decision.next_tool is None and decision.premise_valid is False
        and decision.clarification_required and not decision.premise_assessment.independently_verified,
    )
    return initial_passed and revised_passed


def regulatory_scenario(provider: LLMProvider, planner: ResearchPlanner) -> bool:
    state = start_scenario(
        "regulatory_goal", "What did the DOJ announce about Google in its antitrust press releases last week?", provider,
    )
    _, passed = report_plan(
        "regulatory_plan", planner, state, lambda decision: decision.next_tool is ToolName.NEWS,
    )
    return passed


def unsupported_scenario(provider: LLMProvider, planner: ResearchPlanner) -> bool:
    state = start_scenario(
        "unsupported_goal", "Can the available public sources establish what two investors said in a private phone call? "
        "I have no recording or transcript.", provider,
    )
    _, passed = report_plan(
        "unsupported_source_plan", planner, state,
        lambda decision: decision.next_tool is None and decision.should_stop and decision.information_gap is not None,
    )
    return passed


def report_error(label: str, exc: Exception) -> None:
    entry = {"scenario": label, "error": type(exc).__name__}
    if isinstance(exc, LLMError):
        entry["performance"] = exc.performance.model_dump(mode="json")
    elif not isinstance(exc, ValidationError):
        entry["message"] = str(exc)
    print(json.dumps(entry, ensure_ascii=True))


def main() -> int:
    try:
        settings = load_llm_settings()
    except ValueError:
        print("Invalid local LLM configuration.")
        return 1
    if (settings.provider, settings.model) != ("ollama", "qwen3:8b"):
        print("The live validation requires the configured local Ollama qwen3:8b model.")
        return 1

    passed = []
    with open_llm_provider(settings) as provider:
        planner = ResearchPlanner(provider)
        for scenario in (stock_scenarios, regulatory_scenario, unsupported_scenario):
            try:
                passed.append(scenario(provider, planner))
            except (ValueError, LLMError) as exc:
                report_error(scenario.__name__, exc)
                passed.append(False)
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
