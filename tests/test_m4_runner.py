"""Exercise M4 using structured model responses and real tool adapters over fixtures."""

from __future__ import annotations

import json
import re
from threading import Event
from time import monotonic

import pytest
from pydantic import ValidationError

from examples.autonomous_research_demo import FixtureFilings, FixtureNews, FixturePrices
from financial_research_agent.agent.budget import ResearchLimits
from financial_research_agent.agent.dispatch import ActionRejected, ToolRegistry
from financial_research_agent.agent.runner import ResearchRun, ResearchRunner, StopReason
from financial_research_agent.agent.state import add_action
from financial_research_agent.config import LLMSettings
from financial_research_agent.models import (
    HypothesisStatus, ObservationOutcome, ObservationProvenance, ResearchAction, ResearchGoal,
    ResearchState, ResearchStatus,
)
from financial_research_agent.providers.http_client import HttpFailure
from financial_research_agent.providers.llm import (
    LLMError, LLMErrorCategory, LLMPerformance, StructuredLLMResult,
)
from financial_research_agent.providers.ollama import OllamaProvider
from financial_research_agent.tools import ToolName
from financial_research_agent.tools.calculator import CalculatorTool
from financial_research_agent.tools.market_data import MarketDataTool
from financial_research_agent.tools.news import NewsTool
from financial_research_agent.tools.sec_filings import SecFilingsTool


START, END = "2026-08-28", "2026-10-12"
RANGE = {"start_date": START, "end_date": END}
SPAN = f"{START} to {END}"
PRICE_Q = f"Why did GOOGL decline from {SPAN}?"
DOJ_Q = f"What DOJ press-release titles mention Google from {SPAN}?"
DOJ_CONTENT_Q = f"What did the DOJ press releases about Google announce from {SPAN}?"
PERF = LLMPerformance(model="mock", request_duration_seconds=.1, total_duration_seconds=.1,
                      attempts=1, input_tokens=20, output_tokens=5, validation_passed=True)


def draft(kind="causal_price_move"):
    return {
        "objective": "Research the specified subject and period.",
        "research_type": kind, "company": "Google",
        "ticker": "GOOGL" if kind == "causal_price_move" else None, "benchmark": None,
        "temporal_expression": SPAN, **RANGE,
        "price_premise": "decline" if kind == "causal_price_move" else None, "clarification": None,
    }


SEEDS = {"hypotheses": [
    {"statement": "Lower advertising demand could contribute if the movement is confirmed.", "research_gaps": ["Ad results"]},
    {"statement": "A regulatory event might be relevant if the movement is confirmed.", "research_gaps": ["Agency releases"]},
]}


def plan(tool=None, arguments=None, *, action=None, gap=None, updates=()):
    base = {
        "recommended_action": action or ("continue" if tool else "stop"), "next_tool": tool,
        "purpose": f"Research via {tool}." if tool else None, "reason": "Use existing evidence.",
        "information_gap": (gap or f"Need evidence from {tool}.") if tool else gap,
        "hypothesis_ids": [], "hypothesis_updates": list(updates),
    }
    if arguments is not None:
        base["arguments"] = arguments
    return base


class Model:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.prompts = []

    def generate(self, prompt, schema):
        self.prompts.append((schema, prompt))
        item = next(self.responses)
        if isinstance(item, LLMError):
            raise item
        if callable(item):
            item = item(prompt, schema)
        try:
            validated = schema.model_validate(item)
        except ValidationError:
            raise model_error(LLMErrorCategory.SCHEMA_VALIDATION) from None
        return StructuredLLMResult[schema](decision=validated, performance=PERF)


def model_error(category):
    return LLMError("Test model failure", PERF.model_copy(update={"error_category": category, "validation_passed": False}))


def registry(*, price=None, news=None, sec=None, calculator=False):
    return ToolRegistry(
        market_data=MarketDataTool(price) if price else None,
        news=NewsTool(news) if news else None,
        sec_filings=SecFilingsTool(sec) if sec else None,
        calculator=CalculatorTool() if calculator else None,
        provenance={name: ObservationProvenance.HISTORICAL_FIXTURE for name in ToolName},
    )


def test_loop_dispatches_three_different_actions_as_the_state_changes():
    def calculate(prompt, schema):
        assert "GOOGL returned -6.00%" in prompt and "adjusted close 100 to 94" in prompt
        assert "Identify the next open causal gap" in prompt
        assert '"tool_calls_remaining":4' in prompt
        result_id = re.search(r'"result_id":"([^"]+)"', prompt).group(1)
        return plan("calculator", {"operation": "percentage_return", "starting_price": 100,
                                   "ending_price": 94, "source_result_ids": [result_id]})

    def find_filings(prompt, schema):
        assert "from supplied inputs" in prompt
        assert '"tool_calls_remaining":3' in prompt
        return plan("sec_filings", {"ticker": "GOOGL", "forms": ["10-Q"], **RANGE})

    def assess(prompt, schema):
        assert "0000000001-26-000001" in prompt and "GOOGL returned" in prompt
        hypothesis = re.search(r'"id":"(h-[^"]+)"', prompt).group(1)
        context = json.loads(prompt.split("\nContext:\n", 1)[1].splitlines()[0])
        evidence = next(row["evidence"][0]["id"] for row in context["attempted_actions"] if row["tool"] == "sec_filings")
        return plan(arguments={}, gap="The filing metadata does not establish a cause.", updates=[{
            "hypothesis_id": hypothesis, "status": "inconclusive", "reason": "A filing record was found, with its cause still unresolved.",
            "supporting_evidence_ids": [evidence],
        }])

    model = Model(draft(), SEEDS, plan("market_data"), calculate, find_filings, assess)
    result = ResearchRunner(model, registry(price=FixturePrices(), sec=FixtureFilings(), calculator=True)).run(PRICE_Q)
    state = result.state
    assert [a.tool_name for a in state.completed_actions] == ["market_data", "calculator", "sec_filings"]
    assert result.statistics.planning_steps == 4 and result.statistics.llm_requests == 6
    assert [o.provenance for o in state.observations] == ["historical_fixture", "derived", "historical_fixture"]
    assert state.hypotheses[0].status is HypothesisStatus.INCONCLUSIVE
    assert result.stop_reason is StopReason.INSUFFICIENT_EVIDENCE
    assert state.hypotheses[0].supporting_evidence_ids[-1] == state.evidence[-1].evidence_id
    for step, action, observation, tool_result in zip(result.steps, state.completed_actions, state.observations, state.tool_results):
        assert step.action_id == action.action_id == observation.action_id == tool_result.action_id
        assert step.result_id == observation.result_id == tool_result.result_id
        assert step.observation_id == observation.observation_id
        assert step.fingerprint
    assert state.tool_results[1].data["value"] == -6
    assert not state.planned_actions
    assert ResearchRun.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("max_steps", [1, 3])
def test_contradiction_stops_even_if_planning_budget_is_exhausted(max_steps):
    model = Model(draft(), SEEDS, plan("market_data"), plan("news", {"query": "Google", **RANGE}))
    result = ResearchRunner(model, registry(price=FixturePrices(ending_price=103), news=FixtureNews())).run(
        PRICE_Q, ResearchLimits(max_steps=max_steps),
    )
    assert result.stop_reason is StopReason.NEEDS_CLARIFICATION and result.statistics.tool_calls == 1
    assert result.state.status is ResearchStatus.STOPPED
    assert "+3.00%" in result.state.observations[0].summary
    assert all(h.status is HypothesisStatus.INCONCLUSIVE for h in result.state.hypotheses)
    assert all(h.assessment_observation_ids == [result.state.observations[0].observation_id] for h in result.state.hypotheses)
    assert not result.state.observations[0].independently_verified
    assert len(result.steps) == 1 and result.steps[0].quality_gate_terminated


def test_positive_regulatory_metadata_can_satisfy_a_scoped_goal():
    model = Model(draft("regulatory"), plan("news", {"query": "Google", **RANGE}), plan(arguments={}))
    result = ResearchRunner(model, registry(news=FixtureNews())).run(DOJ_Q)
    assert result.stop_reason is StopReason.GOAL_SATISFIED
    assert result.state.status is ResearchStatus.COMPLETED
    observation, evidence = result.state.observations[0], result.state.evidence[0]
    assert evidence.observation_ids == [observation.observation_id]
    assert str(evidence.source_url).endswith("/simulated-doj/release")
    stored = result.state.tool_results[0]
    assert stored.data["articles"][0]["publication_date"] == END
    assert stored.data["provider"] == "simulated_doj"


class EmptyNews(FixtureNews):
    def search_news(self, request):
        return super().search_news(request).model_copy(update={"articles": []})


def test_empty_source_is_recorded_and_cannot_satisfy_a_goal():
    model = Model(draft("regulatory"), plan("news", {"query": "Google", **RANGE}), plan(arguments={}))
    result = ResearchRunner(model, registry(news=EmptyNews())).run(DOJ_Q)
    assert result.stop_reason is StopReason.INSUFFICIENT_EVIDENCE
    assert result.state.observations[0].outcome is ObservationOutcome.EMPTY
    assert not result.state.evidence


class OfflineNews(FixtureNews):
    def search_news(self, request):
        if request.query == "Google":
            raise HttpFailure("RATE_LIMITED", "Simulated rate limit", True, 1)
        return super().search_news(request)


def test_failure_is_visible_to_replanner_and_a_different_valid_query_can_recover():
    def recover(prompt, schema):
        assert "RATE_LIMITED" in prompt and '"outcome":"failed"' in prompt
        return plan("news", {"query": "Alphabet", **RANGE})

    model = Model(draft("regulatory"), plan("news", {"query": "Google", **RANGE}), recover, plan(arguments={}))
    result = ResearchRunner(model, registry(news=OfflineNews())).run(DOJ_Q)
    assert result.stop_reason is StopReason.GOAL_SATISFIED
    assert [o.outcome for o in result.state.observations] == ["failed", "data"]
    assert len(result.state.evidence) == 1
    assert result.state.evidence[0].observation_ids == [result.state.observations[1].observation_id]
    assert result.state.tool_results[0].error.code == "RATE_LIMITED"


@pytest.mark.parametrize("provider", [FixtureNews(), OfflineNews()])
def test_equivalent_successful_or_failed_actions_are_blocked_measurably(provider):
    narrower = {**RANGE, "start_date": "2026-09-01"}
    model = Model(draft("regulatory"), plan("news", {"query": "Google", **narrower}),
                  plan("news", {"query": " GOOGLE ", **narrower}), plan("news", {"query": "Google", **narrower}))
    result = ResearchRunner(model, registry(news=provider)).run(DOJ_Q)
    assert result.stop_reason is StopReason.REPEATED_ACTION
    assert result.statistics.tool_calls == 1 and result.statistics.planning_steps == 3
    assert [step.validation for step in result.steps] == ["accepted", "blocked", "blocked"]
    assert all(step.proposed_tool == "news" for step in result.steps)
    assert len({step.fingerprint for step in result.steps}) == 1
    assert result.steps[0].fingerprint is not None
    assert len(result.state.observations) == 1


@pytest.mark.parametrize("tool,arguments", [
    ("shell", {"command": "never execute"}),
    ("news", {"query": "Google", "url": "http://127.0.0.1/"}),
    ("sec_filings", {"filing": {"primary_document_url": "http://127.0.0.1/"}}),
    ("calculator", {"operation": "unknown"}),
])
def test_invalid_or_unauthorized_proposals_do_not_execute(tool, arguments):
    model = Model(draft("regulatory"), plan(tool, arguments), plan(tool, arguments))
    result = ResearchRunner(model, registry(news=FixtureNews(), sec=FixtureFilings())).run(DOJ_Q)
    assert result.stop_reason in {StopReason.INVALID_PROPOSAL, StopReason.LLM_FAILURE}
    assert result.statistics.tool_calls == 0
    assert not result.state.tool_results and not result.state.planned_actions


def test_only_enabled_and_recorded_tools_can_dispatch():
    state = ResearchState(goal=ResearchGoal(query=DOJ_Q, objective="Check regulatory metadata"))
    action = ResearchAction(action_id="unsafe", tool_name="unknown", arguments={}, purpose="invalid request")
    tools = registry(news=FixtureNews())
    with pytest.raises(ActionRejected, match="Record"):
        tools.execute(state, action)
    with pytest.raises(ActionRejected, match="unknown"):
        tools.execute(add_action(state, action), action)
    with pytest.raises(ActionRejected, match="unavailable"):
        tools.authorize(ToolName.MARKET_DATA, {"ticker": "GOOGL", **RANGE}, state)


def test_sec_excerpt_uses_registered_filing_and_preserves_truncation():
    question = f"What did Alphabet disclose from {SPAN}?"
    model = Model(
        draft("filings"), plan("sec_filings", {"ticker": "Alphabet", **RANGE}),
        plan("sec_filings", {"accession_number": "0000000001-26-000001"}),
        plan(arguments={}, gap="The relevant filing section is incomplete."),
    )
    result = ResearchRunner(model, registry(sec=FixtureFilings())).run(question)
    observation = result.state.observations[-1]
    tool_result = result.state.tool_results[-1]
    assert observation.outcome is ObservationOutcome.INCOMPLETE and tool_result.data["truncated"] is True
    assert tool_result.data["characters_returned"] == len(tool_result.data["excerpt"])
    assert result.stop_reason is StopReason.INSUFFICIENT_EVIDENCE
    assert all(observation.observation_id not in item.observation_ids for item in result.state.evidence)
    assert tool_result.data["excerpt"][:100] not in model.prompts[-1][1]


def test_unavailable_source_is_not_confused_with_goal_satisfaction():
    model = Model(draft("other"), plan(arguments={}, gap="The sources do not cover private conversations."))
    result = ResearchRunner(model, ToolRegistry()).run("What was said privately?")
    assert result.stop_reason is StopReason.UNSUPPORTED_QUESTION
    assert result.statistics.tool_calls == 0


@pytest.mark.parametrize("limits,remaining", [
    (ResearchLimits(max_steps=1), "step"),
    (ResearchLimits(max_tool_calls=1), "tool call"),
    (ResearchLimits(max_llm_requests=2), "LLM request"),
])
def test_limits_end_research_without_turning_partial_results_into_success(limits, remaining):
    model = Model(draft("regulatory"), plan("news", {"query": "Google", **RANGE}),
                  plan("news", {"query": "Alphabet", **RANGE}))
    result = ResearchRunner(model, registry(news=FixtureNews())).run(DOJ_CONTENT_Q, limits)
    assert result.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert remaining in result.stop_detail
    assert result.statistics.tool_calls == 1 and result.state.status is ResearchStatus.STOPPED
    assert ResearchRun.model_validate_json(result.model_dump_json()) == result


def test_provider_failure_threshold_preserves_attempted_results():
    model = Model(draft(), SEEDS, plan("market_data"))
    result = ResearchRunner(model, registry(price=FixturePrices(failed=True))).run(
        PRICE_Q, ResearchLimits(max_consecutive_failures=1),
    )
    assert result.stop_reason is StopReason.TOOL_FAILURE
    assert result.state.observations[0].outcome is ObservationOutcome.FAILED
    assert not result.state.evidence


def test_deadline_returns_partial_state_while_a_sync_tool_is_blocked():
    release = Event()
    started = Event()

    class SlowNews(FixtureNews):
        def search_news(self, request):
            started.set()
            release.wait(2)
            return super().search_news(request)

    model = Model(draft("regulatory"), plan("news", {"query": "Google", **RANGE}))
    t0 = monotonic()
    try:
        result = ResearchRunner(model, registry(news=SlowNews())).run(DOJ_Q, ResearchLimits(runtime_seconds=.10))
        assert started.is_set()
        assert monotonic() - t0 < 1
        assert result.stop_reason is StopReason.BUDGET_EXHAUSTED
        assert result.state.tool_results[0].error.code == "DEADLINE_EXCEEDED"
        assert not result.state.evidence and not result.state.planned_actions
    finally:
        release.set()


def test_ollama_retries_cannot_spend_more_than_the_physical_request_budget(monkeypatch):
    class FailResponse:
        status_code = 503
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None

    class Session:
        def __init__(self):
            self.calls = []
        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return FailResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key-value")
    monkeypatch.setattr("financial_research_agent.providers.ollama.time.sleep", lambda sec: None)
    session = Session()
    provider = OllamaProvider(LLMSettings(max_retries=2), session=session)
    result = ResearchRunner(provider, registry(news=FixtureNews())).run(DOJ_Q, ResearchLimits(max_llm_requests=1))
    assert result.stop_reason is StopReason.BUDGET_EXHAUSTED and result.statistics.llm_requests == 1
    assert [url for url, _ in session.calls] == ["http://localhost:11434/api/chat"]
    assert session.calls[0][1]["allow_redirects"] is False
    assert "synthetic-key-value" not in result.model_dump_json()


def test_llm_timeout_preserves_goal_and_partial_execution():
    model = Model(draft("regulatory"), plan("news", {"query": "Google", **RANGE}), model_error(LLMErrorCategory.TIMEOUT))
    result = ResearchRunner(model, registry(news=FixtureNews())).run(DOJ_CONTENT_Q)
    assert result.stop_reason is StopReason.LLM_FAILURE
    assert result.state.goal.query == DOJ_CONTENT_Q and len(result.state.completed_actions) == 1
    assert result.issues[-1].code == "llm_timeout"
    assert result.steps[-1].validation == "llm_failed"
