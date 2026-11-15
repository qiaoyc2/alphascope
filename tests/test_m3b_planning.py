"""Deterministic tests for goal interpretation and observation-driven planning."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from financial_research_agent.agent.goal import GoalDraft, interpret_goal
from financial_research_agent.agent.hypotheses import (
    HypothesisUpdate, InitialHypotheses, apply_hypothesis_update, initialize_research, unresolved_hypotheses,
)
from financial_research_agent.agent.planning import (
    GoalStatus, PlanProposal, PlanValidationError, ResearchPlanner, ScopedPriceProposal, record_observation, record_plan,
)
from financial_research_agent.agent.premise import assess_premise
from financial_research_agent.agent.state import add_action
from financial_research_agent.config import LLMSettings, open_llm_provider
from financial_research_agent.models import (
    Evidence, HypothesisStatus, MarketMovement, Observation, ObservationProvenance, PricePremise,
    ResearchAction, ResearchGoal, ResearchState, ResearchType, ToolResult,
)
from financial_research_agent.providers.llm import (
    LLMError, LLMErrorCategory, LLMPerformance, StructuredLLMResult,
)
from financial_research_agent.providers.ollama import OllamaProvider
from financial_research_agent.tools import ToolName
from financial_research_agent.tools.planning import validate_tool_arguments


TODAY = date(2026, 10, 12)
START = date(2026, 8, 28)
END = date(2026, 10, 12)
TICKER_DATES = {"ticker": "GOOGL", "start_date": START.isoformat(), "end_date": END.isoformat()}
PERF = LLMPerformance(
    model="test", request_duration_seconds=0, total_duration_seconds=0, attempts=1, validation_passed=True,
)


class MockProvider:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.prompts = []

    def generate(self, prompt, schema):
        self.prompts.append((schema, prompt))
        payload = next(self.responses)
        if isinstance(payload, LLMError):
            raise payload
        return StructuredLLMResult[schema](decision=schema.model_validate(payload), performance=PERF)


def draft(**changes):
    return {
        "objective": "Investigate the reported move.",
        "research_type": "causal_price_move", "company": "Alphabet", "ticker": "GOOGL",
        "benchmark": None, "temporal_expression": None, "start_date": None, "end_date": None,
        "price_premise": "decline", "clarification": None,
        **changes,
    }


def proposal(**changes):
    return {
        "recommended_action": "continue", "next_tool": "market_data",
        "arguments": TICKER_DATES, "purpose": "Test the initial premise.",
        "reason": "Scoped prices can establish the movement.",
        "information_gap": "Price behavior during the specified period.",
        "hypothesis_ids": [], "hypothesis_updates": [],
        **changes,
    }


def scoped_proposal(**changes):
    payload = proposal(**changes)
    payload.pop("arguments")
    return payload


def price_goal(**changes):
    return ResearchGoal(
        query="Why did GOOGL decline recently?", objective="Investigate possible causes.",
        **{
            "ticker": "GOOGL", "start_date": START, "end_date": END,
            "research_type": ResearchType.CAUSAL_PRICE_MOVE, "price_premise": PricePremise.DECLINE,
            **changes,
        },
    )


def stock_observation(state, stock_return=1.83, *, name="fixture", **changes):
    settings = {
        "ticker": "GOOGL", "start_date": START, "end_date": END, "return_pct": stock_return,
        "benchmark": "QQQ", "benchmark_return_pct": 4.24, **changes,
    }
    action = ResearchAction(
        action_id=f"action-{name}", tool_name=ToolName.MARKET_DATA, arguments=TICKER_DATES,
        purpose="Register a supplied comparison",
    )
    result = ToolResult(result_id=f"result-{name}", action_id=action.action_id,
                        tool_name=action.tool_name, success=True)
    observation = Observation(
        observation_id=name, action_id=action.action_id, result_id=result.result_id,
        summary="Supplied price comparison.", provenance=ObservationProvenance.HISTORICAL_FIXTURE,
        market_movement=MarketMovement(**settings),
    )
    return record_observation(add_action(state, action), result, observation)


@pytest.mark.parametrize("text,expected", [
    ("recently", (date(2026, 9, 13), TODAY)),
    ("last week", (date(2026, 10, 5), date(2026, 10, 11))),
    ("this month", (date(2026, 10, 1), TODAY)),
    ("past 10 days", (date(2026, 10, 3), TODAY)),
])
def test_goal_relative_periods_bind_to_injected_clock(text, expected):
    provider = MockProvider(draft(temporal_expression=text))
    question = f"Why did GOOGL fall {text}?"
    goal, perf = interpret_goal(question, provider, clock=lambda: TODAY)
    assert goal.query == question and goal.temporal_expression == text
    assert goal.ticker == "GOOGL" and goal.price_premise is PricePremise.DECLINE
    assert (goal.start_date, goal.end_date) == expected
    assert goal.assumptions and not goal.needs_clarification
    assert perf == PERF and provider.prompts[0][0] is GoalDraft
    assert TODAY.isoformat() in provider.prompts[0][1]


def test_marked_lowercase_ticker_is_kept_and_non_date_phrases_are_discarded():
    goal, _ = interpret_goal(
        "Check ticker googl about cloud revenue.",
        MockProvider(draft(research_type="filings", temporal_expression="cloud revenue", price_premise=None)),
    )
    assert goal.ticker == "GOOGL"
    assert goal.temporal_expression is None


def test_source_availability_statement_is_not_a_goal_clarification():
    provider = MockProvider(draft(
        research_type="other", ticker=None, company=None, price_premise=None,
        clarification="No recording is present in the supplied context.",
    ))
    goal, _ = interpret_goal("Can public sources establish what the investors said in a private call?", provider)
    assert not goal.needs_clarification
    assert goal.unresolved_details == []


def test_unambiguous_iso_dates_are_preserved():
    span = f"{START} to {END}"
    goal, _ = interpret_goal(
        f"Why did GOOGL fall from {span}?",
        MockProvider(draft(temporal_expression=span, start_date=START, end_date=END)),
    )
    assert (goal.start_date, goal.end_date, goal.temporal_expression) == (START, END, span)


@pytest.mark.parametrize("text,changes", [
    ("Why did Google fall recently?", {"temporal_expression": "recently"}),
    ("Why did GOOGL decline from 03/04 to 04/05?",
     {"temporal_expression": "03/04 to 04/05", "start_date": START, "end_date": END}),
    ("Did GOOGL fall recently or last week?", {"temporal_expression": "recently"}),
    ("Why did GOOGL underperform last week?", {"temporal_expression": "last week", "price_premise": "underperform"}),
])
def test_ambiguous_goals_have_explicit_resolution_questions(text, changes):
    goal, _ = interpret_goal(text, MockProvider(draft(**changes)), clock=lambda: TODAY)
    provider = MockProvider()
    assert goal.needs_clarification and goal.unresolved_details
    if "Google" in text:
        assert goal.ticker is None
    result = ResearchPlanner(provider).plan(ResearchState(goal=goal))
    assert result.decision.clarification_required and result.decision.next_tool is None
    assert result.performance is None and not provider.prompts


def test_initial_hypotheses_have_stable_ids_and_distinct_statements():
    provider = MockProvider({"hypotheses": [
        {"statement": "Changes in industry demand could affect revenue.", "research_gaps": ["Sector demand"]},
        {"statement": "Changes in industry demand could affect revenue!", "research_gaps": ["Industry demand"]},
        {"statement": "A change in financing costs may explain the move.", "research_gaps": ["Financing costs"]},
    ]})
    state, performance = initialize_research(price_goal(), provider)
    assert provider.prompts[0][0] is InitialHypotheses and performance == PERF
    assert len(state.hypotheses) == 2
    assert len({item.hypothesis_id for item in state.hypotheses}) == 2
    assert state.hypotheses == unresolved_hypotheses(state)
    restored = ResearchState.model_validate_json(state.model_dump_json())
    assert restored.hypotheses == state.hypotheses
    assert all(h.status is HypothesisStatus.UNVERIFIED and h.confidence is None for h in state.hypotheses)


def test_factual_goals_skip_causal_hypothesis_generation():
    provider = MockProvider()
    state, performance = initialize_research(
        price_goal(research_type=ResearchType.PRICE_COMPARISON, price_premise=None), provider,
    )
    assert state.hypotheses == [] and performance is None and not provider.prompts


def test_initial_market_plan_and_recording_never_execute_tools():
    state = ResearchState(goal=price_goal())
    provider = MockProvider(scoped_proposal())
    result = ResearchPlanner(provider).plan(state)
    assert provider.prompts[0][0] is ScopedPriceProposal
    assert result.decision.next_tool is ToolName.MARKET_DATA
    assert result.decision.premise_valid is None
    updated = record_plan(state, result.decision)
    assert state.planned_actions == []
    assert updated.planned_actions[0].arguments == TICKER_DATES
    assert updated.tool_call_count == 0 and not updated.tool_results
    with pytest.raises(PlanValidationError, match="equivalent"):
        record_plan(updated, result.decision)


def test_scoped_market_args_are_grounded_and_cannot_be_overridden_by_the_model():
    state = ResearchState(goal=price_goal(benchmark="QQQ"))
    decision = ResearchPlanner(MockProvider(scoped_proposal())).plan(state).decision
    assert decision.tool_arguments == {**TICKER_DATES, "benchmark": "QQQ", "price_basis": "adjusted_close"}
    with pytest.raises(ValidationError):
        ScopedPriceProposal.model_validate(proposal(arguments={**TICKER_DATES, "ticker": "WRONG"}))


def test_positive_fixture_replans_without_claiming_real_world_verification():
    state = ResearchState(goal=price_goal())
    planner = ResearchPlanner(MockProvider(scoped_proposal(), proposal()))
    initial = planner.plan(state).decision
    state = stock_observation(record_plan(state, initial))
    result = planner.plan(state)
    decision = result.decision
    assert result.adjustment
    assert decision.premise_valid is False and decision.clarification_required
    assert decision.goal_status is GoalStatus.PREMISE_CONFLICT and decision.next_tool is None
    assert "+1.83%" in decision.reason and "+4.24%" in decision.reason and "-2.41" in decision.reason
    assert decision.premise_assessment.provenance == [ObservationProvenance.HISTORICAL_FIXTURE]
    assert not decision.premise_assessment.independently_verified
    replanned = record_plan(state, decision)
    assert initial.information_gap not in replanned.research_gaps
    assert decision.information_gap in replanned.research_gaps
    with pytest.raises(PlanValidationError, match="fresh plan"):
        record_plan(state, initial)


def test_supported_price_premise_can_advance_to_filings():
    state = stock_observation(ResearchState(goal=price_goal()), -1.2)
    result = ResearchPlanner(MockProvider(proposal(
        next_tool="sec_filings", arguments={"ticker": "GOOGL", "forms": ["10-Q"]},
        purpose="Look for company disclosures after observing a decline.",
    ))).plan(state)
    assert result.decision.premise_valid is True
    assert result.decision.next_tool is ToolName.SEC_FILINGS
    assert not result.decision.premise_assessment.independently_verified
    recorded = record_plan(state, result.decision)
    assert recorded.planned_actions[-1].tool_name == "sec_filings"


def test_relative_underperformance_is_true_while_absolute_decline_is_false():
    absolute = stock_observation(ResearchState(goal=price_goal()))
    relative = stock_observation(ResearchState(goal=price_goal(price_premise=PricePremise.UNDERPERFORM, benchmark="QQQ")))
    assert assess_premise(absolute).conditional_valid is False
    assert assess_premise(relative).conditional_valid is True


def test_incompatible_dates_do_not_establish_premise():
    state = stock_observation(ResearchState(goal=price_goal()), start_date=date(2026, 9, 1))
    assert assess_premise(state).conditional_valid is None


def test_conflicting_observations_leave_premise_unresolved():
    state = stock_observation(ResearchState(goal=price_goal()))
    state = stock_observation(state, -1.83, name="conflict")
    decision = ResearchPlanner(MockProvider(proposal())).plan(state).decision
    assert decision.premise_valid is None and decision.clarification_required
    assert decision.premise_assessment.observations_conflict
    assert decision.premise_assessment.observation_ids == ["fixture", "conflict"]


def test_regulatory_question_uses_doj_coverage_first():
    goal = ResearchGoal(query="What did DOJ announce about Alphabet?", objective="Find official announcements.",
                        research_type=ResearchType.REGULATORY)
    provider = MockProvider(proposal(next_tool="news", arguments={"query": "Google antitrust"}))
    result = ResearchPlanner(provider).plan(ResearchState(goal=goal))
    assert result.decision.next_tool is ToolName.NEWS
    assert result.decision.tool_arguments["query"] == "Google antitrust"
    assert "Official DOJ announcements only" in provider.prompts[0][1]


def test_equivalent_completed_request_with_normalized_arguments_is_not_proposed_again():
    goal = ResearchGoal(query="Find a DOJ release.", objective="Check the antitrust record.", research_type="regulatory")
    action = ResearchAction(action_id="old", tool_name="news", purpose="Check DOJ", arguments={"query": "Google Antitrust"})
    result = ToolResult(result_id="old-result", action_id="old", tool_name="news", success=True)
    observation = Observation(observation_id="o", action_id="old", result_id="old-result", summary="Press release.")
    state = record_observation(add_action(ResearchState(goal=goal), action), result, observation)
    provider = MockProvider(proposal(next_tool="news", arguments={"query": " google  antitrust ", "limit": 10}))
    plan = ResearchPlanner(provider).plan(state)
    assert plan.decision.next_tool is None and plan.adjustment
    assert record_plan(state, plan.decision).planned_actions == []
    assert len(state.completed_actions) == 1


def test_missing_source_coverage_is_not_replaced_by_general_news():
    goal = ResearchGoal(query="What was said in a private call?", objective="Obtain private conversation contents.")
    plan = ResearchPlanner(MockProvider(proposal(
        recommended_action="stop", next_tool=None, purpose=None, arguments={},
        information_gap="No authorized source covers the private call.",
        reason="Available tools only cover public disclosures, official releases, market data and calculations.",
    ))).plan(ResearchState(goal=goal)).decision
    assert plan.goal_status is GoalStatus.UNSUPPORTED_GAP and plan.should_stop
    assert plan.next_tool is None and "private call" in plan.information_gap


def test_no_tool_recommendation_discarding_extraneous_details_is_recordable():
    state = ResearchState(goal=ResearchGoal(query="Review private records.", objective="Check source coverage."))
    provider = MockProvider(proposal(
        recommended_action="stop", next_tool=None, purpose="Explain the source limitation.",
        arguments={"unsupported_source": "private records"},
        information_gap="The available tools do not cover these records.",
    ))
    result = ResearchPlanner(provider).plan(state)
    assert result.adjustment and result.decision.goal_status is GoalStatus.UNSUPPORTED_GAP
    assert result.decision.purpose is None and result.decision.tool_arguments == {}
    assert record_plan(state, result.decision).planned_actions == []


@pytest.mark.parametrize("tool,args", [
    ("market_data", {"ticker": "GOOGL"}),
    ("market_data", {**TICKER_DATES, "extra": True}),
    ("market_data", {**TICKER_DATES, "benchmark": "googl"}),
    ("news", {"query": "  "}),
    ("sec_filings", {"ticker": "GOOGL", "forms": ["NOT-A-FORM"]}),
    ("sec_filings", {"accession_number": "0000000000-00-000001"}),
    ("calculator", {"operation": "percentage_return", "starting_price": 0, "ending_price": 5}),
    ("calculator", {"operation": "percentage_return", "starting_price": True, "ending_price": 5}),
    ("calculator", {"operation": "relative_performance", "stock_return_pct": 3, "benchmark_return_pct": 5,
                    "source_result_ids": ["nonexistent"]}),
])
def test_rejects_malformed_tool_arguments(tool, args):
    state = ResearchState(goal=price_goal(research_type=ResearchType.OTHER))
    with pytest.raises(PlanValidationError, match="contract"):
        ResearchPlanner(MockProvider(proposal(next_tool=tool, arguments=args))).plan(state)


def test_unknown_tool_is_schema_invalid_and_premature_causal_action_is_rejected():
    with pytest.raises(ValidationError):
        PlanProposal.model_validate(proposal(next_tool="web_search"))
    with pytest.raises(PlanValidationError, match="scoped ticker and dates"):
        ResearchPlanner(MockProvider(scoped_proposal(next_tool="news"))).plan(
            ResearchState(goal=price_goal())
        )


def test_calculator_uses_existing_argument_operations():
    state = ResearchState(goal=price_goal())
    validated = validate_tool_arguments(ToolName.CALCULATOR, {
        "operation": "relative_performance", "stock_return_pct": 1.83, "benchmark_return_pct": 4.24,
    }, state)
    assert validated["operation"] == "relative_performance"
    assert validated["stock_return_pct"] == 1.83


def test_hypothesis_assessments_reference_existing_evidence_and_are_serializable():
    state, _ = initialize_research(price_goal(), MockProvider({"hypotheses": [
        {"statement": "The move may reflect financing costs.", "research_gaps": ["Debt cost"]},
        {"statement": "The move may reflect new regulation.", "research_gaps": ["Regulatory statements"]},
    ]}))
    state = stock_observation(state)
    hypothesis = state.hypotheses[0]
    from financial_research_agent.agent.state import add_evidence

    state = add_evidence(state, Evidence(
        evidence_id="verified-move", claim="Reported price change was positive.", evidence_type="comparison",
        calculation_inputs={"absolute_return": 1.83}, calculation_method="recorded comparison",
        observation_ids=["fixture"], contradicts_hypothesis_ids=[hypothesis.hypothesis_id],
    ))
    note = HypothesisUpdate(
        hypothesis_id=hypothesis.hypothesis_id, status=HypothesisStatus.WEAKENED,
        reason="The reported direction was inconsistent with the hypothesized decline.",
        contradicting_evidence_ids=["verified-move"], new_gaps=["Check independent prices."],
    )
    revised = apply_hypothesis_update(state, note)
    current = revised.hypotheses[0]
    assert current.hypothesis_id == hypothesis.hypothesis_id and current.status is HypothesisStatus.WEAKENED
    assert current.contradicting_evidence_ids == ["verified-move"] and current.assessment_observation_ids == ["fixture"]
    assert revised.hypotheses[1] == state.hypotheses[1]
    assert ResearchState.model_validate_json(revised.model_dump_json()) == revised
    assert current in unresolved_hypotheses(revised)
    rejected = apply_hypothesis_update(revised, note.model_copy(update={"status": HypothesisStatus.REJECTED}))
    assert rejected.hypotheses[0].status is HypothesisStatus.REJECTED
    assert rejected.hypotheses[0] not in unresolved_hypotheses(rejected)
    with pytest.raises(ValueError, match="unknown"):
        apply_hypothesis_update(state, note.model_copy(update={"contradicting_evidence_ids": ["fabricated"]}))


def test_partial_sec_excerpt_cannot_be_used_to_reject_a_hypothesis():
    state, _ = initialize_research(price_goal(), MockProvider({"hypotheses": [
        {"statement": "Guidance may explain the move.", "research_gaps": ["Guidance section"]},
        {"statement": "Sector pressures may explain the move.", "research_gaps": ["Sector returns"]},
    ]}))
    action = ResearchAction(action_id="excerpt", tool_name="sec_filings", purpose="Review discussion")
    result = ToolResult(result_id="excerpt-result", action_id=action.action_id, tool_name=action.tool_name,
                        success=True, data={"kind": "sec_document_excerpt", "extraction_incomplete": True,
                                            "truncated": True, "excerpt": "cover-page text"})
    observation = Observation(observation_id="excerpt-observation", action_id=action.action_id,
                              result_id=result.result_id, summary="Only the cover page is visible.")
    state = record_observation(add_action(state, action), result, observation)
    update = HypothesisUpdate(
        hypothesis_id=state.hypotheses[0].hypothesis_id, status=HypothesisStatus.REJECTED,
        reason="The excerpt did not mention guidance.", observation_ids=[observation.observation_id],
    )
    with pytest.raises(ValueError, match="incomplete SEC"):
        apply_hypothesis_update(state, update)
    cautious = apply_hypothesis_update(state, update.model_copy(update={
        "status": HypothesisStatus.INCONCLUSIVE, "new_gaps": ["Relevant section unavailable"],
    }))
    assert cautious.hypotheses[0].status is HypothesisStatus.INCONCLUSIVE
    assert "Relevant section unavailable" in cautious.hypotheses[0].research_gaps


def test_planner_references_and_provider_failures_are_never_recorded():
    state = ResearchState(goal=price_goal())
    with pytest.raises(PlanValidationError, match="references"):
        ResearchPlanner(MockProvider(scoped_proposal(hypothesis_ids=["invented"]))).plan(state)
    error = LLMError("Test failure", PERF.model_copy(update={
        "validation_passed": False, "error_category": LLMErrorCategory.TIMEOUT,
    }))
    with pytest.raises(LLMError) as captured:
        ResearchPlanner(MockProvider(error)).plan(state)
    assert captured.value.category is LLMErrorCategory.TIMEOUT
    assert not state.planned_actions


def test_hosted_provider_is_unavailable_even_if_paid_flag_or_key_is_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-value")
    monkeypatch.setenv("OPENAI_PAID_ENABLED", "true")
    with open_llm_provider(LLMSettings()) as provider:
        assert isinstance(provider, OllamaProvider)
    with pytest.raises(ValidationError) as captured:
        with open_llm_provider(LLMSettings.model_construct(provider="openai")):
            pytest.fail("No hosted provider may be opened")
    assert "synthetic-test-value" not in str(captured.value)
