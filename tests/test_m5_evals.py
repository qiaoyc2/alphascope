from __future__ import annotations

import json

import pytest
from peewee import OperationalError
from pydantic import ValidationError

from evals.cases import CASES, CLOCK, PRICE_ARGS, EvalCase, ScriptedProvider, registry
from evals.run_evals import evaluate_case, main, run_suite
from financial_research_agent.agent.budget import ResearchLimits
from financial_research_agent.agent.runner import ResearchRunner
from financial_research_agent.config import LLMSettings
from financial_research_agent.providers.llm import LLMError
from financial_research_agent.providers.ollama import OllamaProvider
from financial_research_agent.providers.yahoo_finance import YahooFinanceProvider
from financial_research_agent.agent.planning import PlanProposal
from financial_research_agent.tools.market_data import MarketDataTool


def test_offline_invariants_are_reproducible_without_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline suite used a network or paid inference path")

    monkeypatch.setenv("OPENAI_API_KEY", "unused-synthetic-key")
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    suite = run_suite(repeat=2)
    assert len(CASES) == 11
    first, second = suite["cases"][:len(CASES)], suite["cases"][len(CASES):]
    for a, b in zip(first, second, strict=True):
        assert a["passed"] and b["passed"], a["checks"]
        for field in ("stop_reason", "checks", "data_mode"):
            assert a[field] == b[field]
    metrics = suite["metrics"]
    assert metrics["final_action_validity_rate"]["value"] == 1
    assert metrics["raw_action_validity_rate"]["value"] < 1
    assert metrics["reported_tokens"] == {"input": None, "output": None, "input_records": 0, "output_records": 0}
    assert metrics["premature_satisfactions"] == 0
    assert metrics["estimated_paid_inference_usd"] is None


def test_timing_does_not_support_causal_conclusion():
    result = evaluate_case(CASES[8])
    assert result["passed"]
    supported = {finding["kind"] for finding in result["quality"]["supported_findings"]}
    assert supported == {"market_return", "event_record", "temporal_association"}
    assert any(finding["kind"] == "causal_conclusion" and finding["support_status"] == "unsupported"
               for finding in result["quality"]["unverified_claims"])
    step = result["steps"][-1]
    assert step["raw_proposal"]["requests_satisfaction"]
    assert step["raw_valid"] is False and step["correction_category"] == "quality_gate_rejection"
    assert step["final_disposition"] == "insufficient_evidence"


def test_premise_conflict_records_raw_tool_and_safe_clarification():
    result = evaluate_case(CASES[0])
    raw = result["steps"][-1]
    assert raw["raw_proposal"]["action"] == "continue" and raw["raw_proposal"]["tool"] == "news"
    assert raw["correction_category"] == "premise_conflict"
    assert raw["recommended_action"] == "clarify" and raw["final_disposition"] == "needs_clarification"
    assert not raw["tool_executed"] and raw["authorized_tool"] is None
    assert raw["llm_performance"]["validation_passed"] and raw["remaining_budget"]["tool_calls"] == 2


def test_exhausted_budget_preserves_available_findings():
    case = CASES[1]
    runner = ResearchRunner(ScriptedProvider(case), registry(case))
    result = runner.run(case.question, ResearchLimits(max_steps=1), today=CLOCK)
    assert result.stop_reason == "budget_exhausted" and result.state.status == "stopped"
    assert result.quality.status == "partial" and result.quality.supported_findings


def test_secure_local_metadata_projects_invalid_tool_without_persisting_values():
    unsafe = {"recommended_action": "continue", "next_tool": "secret-unsafe-tool",
              "arguments": {"url": "http://127.0.0.1/private-secret"}, "reason": "private-reason"}

    class Response:
        status_code = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def json(self):
            return {"done": True, "message": {"content": json.dumps(unsafe), "thinking": "secret-thinking"}}

    class Session:
        def post(self, url, **kwargs):
            assert url == "http://localhost:11434/api/chat" and kwargs["allow_redirects"] is False
            return Response()

    ollama = OllamaProvider(LLMSettings(), session=Session())
    with pytest.raises(LLMError) as error:
        ollama.generate("Synthetic request", PlanProposal)
    projection = error.value.proposal.model_dump_json()
    assert error.value.proposal.unknown_tool and error.value.proposal.tool is None
    assert "secret" not in projection and "http" not in projection


def test_unauthorized_raw_tool_remains_blocked_and_observable():
    case = CASES[2]
    payload = json.loads(case.offline_proposals_json)
    payload[1] = {**payload[1], "next_tool": "unsafe-shell", "arguments": PRICE_ARGS}
    modified = case.model_copy(update={"offline_proposals_json": json.dumps(payload)})
    result = evaluate_case(modified)
    assert result["steps"][0]["raw_proposal"]["unknown_tool"]
    assert result["steps"][0]["correction_category"] == "unsupported_tool"
    assert not any(step["tool_executed"] for step in result["steps"])
    assert result["statistics"]["tool_calls"] == 0
    assert "unsafe-shell" not in json.dumps(result)


@pytest.mark.parametrize("mutation", ["missing", "invalid", "contradictory", "no_fixture"])
def test_missing_or_invalid_expectations_are_rejected(mutation):
    data = CASES[0].model_dump(mode="json")
    if mutation == "missing":
        data.pop("expectations")
    elif mutation == "invalid":
        data["expectations"]["checks"] = ["typo"]
    elif mutation == "contradictory":
        data["expectations"]["prohibited_tools"] = data["expectations"]["required_tools"]
    else:
        data["offline_proposals_json"] = None
    with pytest.raises(ValidationError):
        EvalCase.model_validate(data)


def test_cli_json_keeps_truncation_metadata_without_excerpt_text(tmp_path):
    path = tmp_path / "report.json"
    assert main(["--mode", "offline", "--case", "E5", "--json", str(path)]) == 0
    report = json.loads(path.read_text(encoding="utf-8"))
    sources = report["cases"][0]["quality"]["sources"]
    assert sources[-1]["truncated"] is True and sources[-1]["truncation_reason"] == "character_limit"
    assert "Management reported" not in path.read_text(encoding="utf-8")


def test_yahoo_cache_failure_becomes_a_provider_failure(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OperationalError("Synthetic cache path that should never appear in the tool result")

    monkeypatch.setattr("yfinance.Ticker.history", unavailable)
    case = CASES[2]
    tools = registry(case)
    tools.market_data = MarketDataTool(YahooFinanceProvider())
    run = ResearchRunner(ScriptedProvider(case), tools).run(case.question, today=CLOCK)
    assert run.stop_reason == "tool_failure"
    assert run.state.tool_results[0].error.code == "PROVIDER_UNAVAILABLE"
    assert run.quality.sources[0].completeness == "failed"
    assert "Synthetic cache path" not in run.model_dump_json()
