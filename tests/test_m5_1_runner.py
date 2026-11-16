"""Exercise runner stops and corrective planning with real tool adapters over fixtures."""

from __future__ import annotations

import json
from datetime import date, timedelta
from threading import Event

import pytest

from evals.cases import CASES, CLOCK, FIXTURE, REGULATORY_QUESTION, RETURN_QUESTION
from evals.run_evals import evaluate_case, run_suite
from examples.autonomous_research_demo import FixtureNews, FixturePrices
from financial_research_agent.agent import runner as runner_module
from financial_research_agent.agent.budget import ResearchLimits
from financial_research_agent.agent.dispatch import ToolRegistry
from financial_research_agent.agent.runner import ResearchRunner, StopReason
from financial_research_agent.agent.state import add_action
from financial_research_agent.models import ActionStatus
from financial_research_agent.tools import ToolName
from financial_research_agent.tools.market_data import MarketDataTool
from financial_research_agent.tools.news import NewsTool
from test_m4_runner import DOJ_Q, PRICE_Q, RANGE, SPAN, Model, SEEDS, draft, plan, registry


def test_return_completes_when_the_only_eligible_plan_used_the_last_available_budgets():
    result = evaluate_case(CASES[2])
    assert result["question"] == RETURN_QUESTION
    assert result["passed"] and result["stop_reason"] == "goal_satisfied"
    assert result["statistics"]["llm_requests"] == 2
    assert result["steps"][0]["tool_executed"] and result["steps"][0]["quality_gate_terminated"]
    assert not result["steps"][0]["raw_proposal"]["requests_satisfaction"]

    case = CASES[2]
    goal, first, _ = json.loads(case.offline_proposals_json)
    model = Model(goal, first)
    run = ResearchRunner(model, registry(price=FixturePrices(ending_price=106))).run(
        case.question, ResearchLimits(max_steps=1, max_tool_calls=1, max_llm_requests=2), today=CLOCK,
    )
    assert run.stop_reason is StopReason.GOAL_SATISFIED
    assert len(model.prompts) == 2
    assert run.quality.supported_findings[0].evidence_ids == [run.state.evidence[0].evidence_id]


def test_partial_endpoint_coverage_does_not_complete_the_requested_return():
    class ShortHistory(FixturePrices):
        def get_price_history(self, request):
            history = super().get_price_history(request)
            late_start = history.actual_start_date + timedelta(days=10)
            return history.model_copy(update={
                "bars": [history.bars[0].model_copy(update={"trading_date": late_start}), history.bars[-1]],
                "actual_start_date": late_start,
            })

    goal, first, stop = json.loads(CASES[2].offline_proposals_json)
    run = ResearchRunner(Model(goal, first, stop), registry(price=ShortHistory())).run(RETURN_QUESTION)
    assert run.stop_reason is StopReason.INSUFFICIENT_EVIDENCE
    assert run.quality.status == "partial" and run.quality.supported_findings
    assert run.steps[-1].correction_category == "quality_gate_rejection"


def test_contradicted_premise_keeps_conditional_provenance_without_planning_again():
    model = Model(draft(), SEEDS, plan("market_data"))
    run = ResearchRunner(model, registry(price=FixturePrices(ending_price=103))).run(PRICE_Q)
    assert run.stop_reason is StopReason.NEEDS_CLARIFICATION and run.quality_gate_terminated
    assert len(model.prompts) == 3 and run.statistics.tool_calls == 1
    assert run.quality.premise.conditional_valid is False
    assert run.quality.premise.observation_ids == [run.state.observations[0].observation_id]
    assert run.quality.premise.provenance == ["historical_fixture"]
    assert not run.quality.premise.independently_verified
    assert "+3%" in run.quality.premise.summary


class CappedTitles(FixtureNews):
    def search_news(self, request):
        result = super().search_news(request)
        article = result.articles[0]
        return result.model_copy(update={
            "articles": [article.model_copy(update={
                "article_id": str(number), "title": f"{article.title} ({number})",
                "url": f"https://example.invalid/simulated-doj/release-{number}",
            }) for number in range(request.limit)],
            "limitations": ["Only the first page was inspected."],
        })


def test_bounded_title_lookup_can_stop_with_a_linked_finding_for_every_returned_title():
    model = Model(draft("regulatory"), plan("news", {"query": "Google", **RANGE, "limit": 4}))
    run = ResearchRunner(model, registry(news=CappedTitles())).run(DOJ_Q)
    assert run.stop_reason is StopReason.GOAL_SATISFIED and len(model.prompts) == 2
    assert len(run.quality.supported_findings) == 4
    assert all(finding.evidence_ids and "first page" in finding.limitations[-1]
               for finding in run.quality.supported_findings)
    assert run.quality.sources[0].completeness == "partial"


def test_untrusted_live_title_records_fail_before_evidence_completion():
    model = Model(draft("regulatory"), plan("news", {"query": "Google", **RANGE}))
    tools = ToolRegistry(news=NewsTool(FixtureNews()))
    run = ResearchRunner(model, tools).run(DOJ_Q)
    assert run.stop_reason is StopReason.EXECUTION_ERROR
    assert not run.quality_gate_terminated and not run.quality.supported_findings
    assert run.issues[-1].code == "tool_result_contract"


@pytest.mark.parametrize("question", [
    f"What did the DOJ press releases about Google announce from {SPAN}?",
    f"Which DOJ press-release titles mention Google from {SPAN} and what did they say?",
    f"List comprehensive DOJ press-release titles about Google from {SPAN}",
    f"What DOJ press-release titles mention Google in antitrust cases from {SPAN}?",
])
def test_limited_title_metadata_cannot_satisfy_content_coverage_or_extra_filters(question):
    model = Model(draft("regulatory"), plan("news", {"query": "Google", **RANGE, "limit": 4}), plan(arguments={}))
    run = ResearchRunner(model, registry(news=CappedTitles())).run(question)
    assert run.stop_reason is StopReason.INSUFFICIENT_EVIDENCE and not run.quality_gate_terminated
    assert run.quality.status == "partial" and len(run.quality.supported_findings) == 4
    last = run.steps[-1]
    assert last.schema_valid and last.contract_valid and not last.raw_valid
    assert last.correction_category == "quality_gate_rejection"
    assert not last.tool_executed and last.raw_proposal.requests_satisfaction


def test_confirmed_decline_and_relevant_titles_do_not_complete_causal_research():
    run = evaluate_case(CASES[8])
    assert run["passed"] and run["stop_reason"] == "insufficient_evidence"
    assert any(finding["kind"] == "temporal_association" for finding in run["quality"]["supported_findings"])
    assert not run["quality_gate_terminated"] and not run["steps"][-1]["raw_valid"]


def test_duplicate_is_recorded_and_a_different_scoped_lookup_can_complete():
    narrow = {"query": "Google", **RANGE, "start_date": "2026-09-01"}

    def wider(prompt, schema):
        state = json.loads(prompt.split("\nContext:\n", 1)[1].splitlines()[0])
        prior, = state["attempted_actions"]
        assert prior["arguments"] == {**narrow, "query": "google", "limit": 10}
        assert prior["retrieved"] and prior["limitation"] and len(prior["fingerprint"]) == 64
        assert prior["addressed_lookup"] and state["next_open_question"]
        return plan("news", {"query": "Google", **RANGE})

    model = Model(draft("regulatory"), plan("news", narrow), plan("news", {**narrow, "query": " GOOGLE "}), wider)
    run = ResearchRunner(model, registry(news=FixtureNews())).run(REGULATORY_QUESTION)
    first, duplicate, last = run.steps
    assert duplicate.schema_valid and duplicate.contract_valid and duplicate.duplicate
    assert duplicate.raw_proposal.tool is ToolName.NEWS and duplicate.raw_valid is False
    assert duplicate.correction_category == "duplicate_action" and duplicate.fingerprint == first.fingerprint
    assert duplicate.final_disposition == "retry_planning"
    assert not duplicate.tool_executed and duplicate.action_id is None
    assert last.quality_gate_terminated and run.statistics.tool_calls == 2
    assert run.stop_reason is StopReason.GOAL_SATISFIED
    assert run.quality.supported_findings[0].observation_ids == [last.observation_id]


def test_duplicate_failed_scoped_price_action_ends_when_no_alternative_can_verify_the_premise():
    model = Model(draft(), SEEDS, plan("market_data"), plan("market_data"))
    run = ResearchRunner(model, registry(price=FixturePrices(failed=True))).run(PRICE_Q)
    assert run.stop_reason is StopReason.REPEATED_ACTION
    assert run.statistics.planning_steps == 2 and run.statistics.tool_calls == 1
    assert run.steps[-1].duplicate and not run.steps[-1].tool_executed
    assert run.state.completed_actions[0].status == ActionStatus.FAILED
    assert not run.state.evidence


def test_repeated_contract_invalid_proposals_stop_with_no_dispatch():
    invalid = plan("news", {"query": "Google", "unexpected_argument": True})
    model = Model(draft("regulatory"), invalid, invalid)
    run = ResearchRunner(model, registry(news=FixtureNews())).run(DOJ_Q)
    assert run.stop_reason is StopReason.INVALID_PROPOSAL
    assert run.statistics.planning_steps == 2 and run.statistics.tool_calls == 0
    assert all(step.schema_valid and step.contract_valid is False and not step.duplicate
               and step.correction_category == "invalid_arguments" for step in run.steps)


def test_comparable_captures_reach_clarification_through_the_runner(monkeypatch):
    observe = runner_module.observe_result
    capture = MarketDataTool(FixturePrices(ending_price=94))

    def observe_with_captured_comparison(state, action, result, provenance):
        prior = action.model_copy(update={"action_id": "prior-capture"})
        captured = capture.get_history(
            action_id=prior.action_id, ticker="GOOGL",
            start_date=date.fromisoformat(RANGE["start_date"]), end_date=date.fromisoformat(RANGE["end_date"]),
        )
        # Replay a prior capture, since the registry will never dispatch an equivalent action twice.
        state = observe(add_action(state, prior), prior, captured, FIXTURE)
        return observe(state, action, result, provenance)

    monkeypatch.setattr(runner_module, "observe_result", observe_with_captured_comparison)
    goal, first, _ = json.loads(CASES[2].offline_proposals_json)
    model = Model(goal, first)
    run = ResearchRunner(model, registry(price=FixturePrices(ending_price=106))).run(RETURN_QUESTION)
    assert run.stop_reason is StopReason.NEEDS_CLARIFICATION and run.quality_gate_terminated
    assert len(run.quality.contradictions) == 1 and not run.quality.supported_findings
    assert len(model.prompts) == 2 and len(run.state.observations) == 2 and len(run.steps) == 1


def test_deadline_discards_an_in_flight_title_result_even_if_it_could_complete_the_goal():
    release = Event()

    class SlowTitles(FixtureNews):
        def search_news(self, request):
            release.wait(2)
            return super().search_news(request)

    model = Model(draft("regulatory"), plan("news", {"query": "Google", **RANGE}))
    try:
        run = ResearchRunner(model, registry(news=SlowTitles())).run(DOJ_Q, ResearchLimits(runtime_seconds=.15))
        assert run.stop_reason is StopReason.BUDGET_EXHAUSTED and not run.quality_gate_terminated
        assert run.steps[-1].tool_in_flight_at_exit and run.steps[-1].tool_wait_seconds > 0
        assert run.state.tool_results[-1].error.code == "DEADLINE_EXCEEDED"
        assert not run.quality.supported_findings and not run.state.planned_actions
    finally:
        release.set()


def test_interruption_preserves_earlier_evidence_and_the_interrupted_attempt():
    class InterruptedTitles(FixtureNews):
        def search_news(self, request):
            raise KeyboardInterrupt

    model = Model(draft(), SEEDS, plan("market_data"), plan("news", {"query": "Google", **RANGE}))
    run = ResearchRunner(model, registry(price=FixturePrices(), news=InterruptedTitles())).run(PRICE_Q)
    assert run.stop_reason is StopReason.INTERRUPTED and not run.quality_gate_terminated
    assert run.steps[-1].final_disposition == "interrupted"
    assert run.state.tool_results[-1].error.code == "INTERRUPTED"
    assert len(run.quality.supported_findings) == 1 and not run.state.planned_actions


def test_frozen_expectations_keep_e6_visible_and_separate_from_m5_1():
    frozen = run_suite()
    revised = run_suite(expectations_version="m5.1")
    assert [case["case_id"] for case in frozen["cases"] if not case["passed"]] == ["E6"]
    assert frozen["metrics"]["unexpected_satisfactions"] == 1
    assert revised["metrics"]["cases_passed"]["numerator"] == 11
    for report in (frozen, revised):
        assert report["metrics"]["final_action_validity_rate"]["value"] == 1
        assert report["metrics"]["premature_satisfactions"] == 0
