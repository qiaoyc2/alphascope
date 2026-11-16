from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from evals.cases import CASES, END, FIXTURE, PRICE_ARGS, START, _Prices, conflicting_state
from financial_research_agent.agent.observe import observe_result
from financial_research_agent.agent.quality import (
    ClaimKind, Completeness, Disposition, QualityAssessment, QualityStatus, Relevance,
    SupportStatus, evaluate_quality,
)
from financial_research_agent.agent.state import add_action
from financial_research_agent.models import ObservationProvenance, ResearchAction, ResearchGoal, ResearchState, ResearchType
from financial_research_agent.providers.market_data import PriceBasis
from financial_research_agent.tools.calculator import CalculatorTool
from financial_research_agent.tools.market_data import MarketDataTool


def _record(state, *, start=START, end=END, ticker="GOOGL", ending=106, basis=None, corrupt=False,
            source="prices"):
    action = ResearchAction(
        action_id=f"request-{len(state.observations)}", tool_name="market_data", purpose="Measure price return.",
        arguments={"ticker": ticker, "start_date": str(start), "end_date": str(end)},
    )
    tool = MarketDataTool(_Prices(ending, source))
    if basis is None:
        result = tool.get_history(action_id=action.action_id, ticker=ticker, start_date=start, end_date=end)
    else:
        action.arguments.update({"benchmark": "QQQ", "price_basis": basis})
        result = tool.compare_performance(action_id=action.action_id, ticker=ticker, benchmark="QQQ",
                                          start_date=start, end_date=end, price_basis=basis)
        if corrupt:
            result.data["stock_return_pct"] = 80
    return observe_result(add_action(state, action), action, result, FIXTURE)


def _state(*, benchmark=None, query=None):
    return ResearchState(goal=ResearchGoal(
        query=query or f"What was the GOOGL return from {START} to {END}?",
        objective="Measure the return.", research_type=ResearchType.PRICE_COMPARISON,
        ticker="GOOGL", benchmark=benchmark, start_date=START, end_date=END,
    ))


def test_factual_endpoint_return_is_supported_and_round_trips():
    assessment = evaluate_quality(_record(_state()))
    assert assessment.goal_satisfied and assessment.status is QualityStatus.SUFFICIENT
    finding, = assessment.supported_findings
    assert "+6.00%" in finding.statement
    assert finding.kind is ClaimKind.MARKET_RETURN and finding.evidence_ids and finding.observation_ids
    assert assessment.sources[0].provenance is FIXTURE
    assert QualityAssessment.model_validate_json(assessment.model_dump_json()) == assessment
    with pytest.raises(ValidationError):
        QualityAssessment.model_validate(assessment.model_dump() | {"status": "arbitrary"})
    with pytest.raises(ValidationError):
        QualityAssessment.model_validate(assessment.model_dump() | {"goal_satisfied": False})


def test_equivalent_market_conflict_does_not_prefer_one_provider():
    assessment = evaluate_quality(conflicting_state(CASES[3]))
    assert assessment.status is QualityStatus.CONFLICTED and assessment.recommended_disposition is Disposition.CLARIFY
    assert not assessment.supported_findings and len(assessment.contradictions) == 1
    assert {item.support_status for item in assessment.unverified_claims} == {SupportStatus.CONTRADICTED}


def test_incompatible_periods_are_not_conflicts():
    state = _record(_record(_state()), start=date(2026, 9, 1), ending=84, source="other")
    assessment = evaluate_quality(state)
    assert assessment.goal_satisfied and not assessment.contradictions
    assert len(assessment.supported_findings) == 1
    assert assessment.sources[-1].relevance is Relevance.UNKNOWN


def test_different_price_bases_are_not_conflicting_or_interchangeable():
    state = _state(benchmark="QQQ", query=f"What was the GOOGL return versus QQQ from {START} to {END}?")
    state = _record(_record(state, basis=PriceBasis.RAW_CLOSE), basis=PriceBasis.ADJUSTED_CLOSE, ending=94)
    assessment = evaluate_quality(state)
    assert assessment.status is QualityStatus.PARTIAL
    assert not assessment.contradictions and not assessment.goal_satisfied


def test_reported_benchmark_values_are_checked_against_endpoint_calculations():
    state = _record(_state(benchmark="QQQ"), basis=PriceBasis.ADJUSTED_CLOSE, corrupt=True)
    assessment = evaluate_quality(state)
    assert not assessment.goal_satisfied
    assert assessment.sources[0].completeness is Completeness.UNVALIDATED
    assert assessment.evidence_checks[0].valid is False


def test_declared_raw_return_requires_raw_data_and_comparison_requires_benchmark():
    for question in (
        f"What was the unadjusted GOOGL return from {START} to {END}?",
        f"What was the GOOGL return versus QQQ from {START} to {END}?",
    ):
        assert not evaluate_quality(_record(_state(query=question))).goal_satisfied


def test_forged_claim_on_a_real_source_cannot_be_used():
    state = _record(_state())
    state.evidence[0].claim = "GOOGL fell by 99%."
    result = evaluate_quality(state)
    assert result.status is QualityStatus.INSUFFICIENT
    assert not result.evidence_checks[0].valid
    assert not result.supported_findings


@pytest.mark.parametrize("broken", ["source", "observation", "action", "tool"])
def test_dangling_or_unregistered_provenance_is_rejected(broken):
    state = _record(_state())
    data = state.model_dump(mode="json")
    if broken == "tool":
        data["completed_actions"][0]["tool_name"] = data["tool_results"][0]["tool_name"] = "unregistered"
        assessment = evaluate_quality(ResearchState.model_validate(data))
        assert not assessment.evidence_checks[0].valid
        assert assessment.sources[0].tool is None
        return
    if broken == "source":
        data["tool_results"][0]["sources"] = []
    elif broken == "observation":
        data["evidence"][0]["observation_ids"] = ["missing"]
    else:
        data["observations"][0]["action_id"] = "missing"
    with pytest.raises(ValidationError):
        ResearchState.model_validate(data)


def test_repeated_same_source_deduplicates_without_merging_two_sources():
    state = _record(_record(_record(_state())), source="independent")
    assessment = evaluate_quality(state)
    assert len(assessment.supported_findings) == 2
    assert sum(bool(item.duplicate_of) for item in assessment.sources) == 1
    assert sum(bool(item.duplicate_of) for item in assessment.evidence_checks) == 1


def test_valid_price_for_a_different_ticker_cannot_satisfy_the_goal():
    state = _record(_state(), ticker="QQQ")
    assessment = evaluate_quality(state)
    assert assessment.sources[0].relevance is Relevance.UNRELATED
    assert not assessment.supported_findings and not assessment.goal_satisfied
    assert assessment.evidence_checks[0].valid


def test_explicit_query_dates_prevent_silent_goal_rewrite():
    original = f"What was the GOOGL return from 2026-08-01 to {END}?"
    assert not evaluate_quality(_record(_state(query=original))).goal_satisfied


def test_calculator_proves_arithmetic_without_proving_market_operands():
    state = _state()
    action = ResearchAction(
        action_id="calculation", tool_name="calculator", purpose="Check supplied arithmetic.",
        arguments={"operation": "percentage_return", "starting_price": 100, "ending_price": 120},
    )
    result = CalculatorTool().execute(action)
    state = observe_result(add_action(state, action), action, result, ObservationProvenance.DERIVED)
    assessment = evaluate_quality(state)
    assert assessment.supported_findings[0].kind is ClaimKind.CALCULATION
    assert not assessment.goal_satisfied and assessment.evidence_checks[0].valid
    state.tool_results[0].data["value"] = 80
    assessment = evaluate_quality(state)
    assert not assessment.supported_findings and not assessment.evidence_checks[0].valid


def test_compound_factual_request_is_unresolved_if_one_part_has_evidence():
    state = _record(_state(query=f"What was the GOOGL return from {START} to {END}, and what was its trading volume?"))
    assessment = evaluate_quality(state)
    assert assessment.status is QualityStatus.PARTIAL and assessment.supported_findings
