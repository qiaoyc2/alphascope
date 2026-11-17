"""Validate M6 drafting, report integrity, metric projection, and replay/export."""

from __future__ import annotations

import json
import re

import pytest
from pydantic import ValidationError

from evals.cases import CASES, PRICE_ARGS, SPAN
from evals.run_report_evals import ScriptedDraft, replay_research, run_suite
from examples import autonomous_research_demo as demo
from financial_research_agent.agent.dispatch import ToolRegistry
from financial_research_agent.agent.report import ResearchReport, build_research_report, render_report_markdown, validate_research_report
from financial_research_agent.agent.report_export import export_research_report
from financial_research_agent.agent.report_metrics import CalculationMetric, MarketMetric
from financial_research_agent.agent.report_narrative import generate_research_report
from financial_research_agent.agent.runner import ResearchRunner
from financial_research_agent.models import ObservationProvenance, ResearchState
from financial_research_agent.providers.llm import LLMError, LLMErrorCategory, StructuredLLMResult
from financial_research_agent.tools import ToolName
from financial_research_agent.tools.market_data import MarketDataTool
from test_m4_runner import Model, PRICE_Q, SEEDS, draft, plan, registry as fixture_registry


class DraftProvider(ScriptedDraft):
    def __init__(self, edit):
        super().__init__()
        self.edit = edit
        self.prompt = None

    def generate(self, prompt, schema):
        self.prompt = prompt
        result = super().generate(prompt, schema)
        changed = self.edit(result.decision)
        return StructuredLLMResult[schema].model_construct(decision=changed, performance=result.performance)


def test_accepted_draft_reorders_grounded_facts_and_keeps_caveats():
    run = replay_research(CASES[8])
    provider = DraftProvider(lambda draft: {**draft, "key_findings": ["f3", "f2"], "hypotheses": ["h2"]})
    report = generate_research_report(run, provider)
    validate_research_report(report, run)
    assert report.narrative.mode == "llm_assisted" and report.narrative.validation_succeeded
    assert report.narrative.requests == 1 and report.narrative.prompt_bytes <= 2800
    assert [item.kind.value for item in report.supported_findings] == [
        "temporal_association", "event_record", "market_return",
    ]
    markdown = render_report_markdown(report)
    assert "Timing alone does not establish a causal link." in markdown
    assert "Its cause remains unresolved." in markdown
    assert report.hypotheses[0].support_status == "not_evaluated"


@pytest.mark.parametrize("edit,category", [
    (lambda draft: {**draft, "key_findings": ["f6"]}, "duplicate_or_unknown_reference"),
    (lambda draft: {**draft, "key_findings": ["f1", "f1"]}, "duplicate_or_unknown_reference"),
    (lambda draft: {**draft, "lead_finding": None}, "invalid_lead_reference"),
    (lambda draft: {**draft, "return_pct": 99}, "invalid_report_schema"),
    (lambda draft: {**draft, "source_url": "https://example.invalid/fabricated"}, "invalid_report_schema"),
    (lambda draft: {**draft, "causal_conclusion": "The release caused the price movement."}, "invalid_report_schema"),
    (lambda draft: {field: value for field, value in draft.items() if field != "hypotheses"}, "invalid_report_schema"),
])
def test_invalid_model_draft_returns_only_validated_facts(edit, category):
    run = replay_research(CASES[2])
    report = generate_research_report(run, DraftProvider(edit))
    assert report.narrative.fallback_used and report.narrative.validation_succeeded is False
    assert report.narrative.failure_category == category and report.narrative.requests == 1
    assert "draft failed validation" in render_report_markdown(report)
    assert report.goal_satisfied and report.metrics[0].stock.return_pct == pytest.approx(6)
    validate_research_report(report, run)


@pytest.mark.parametrize("category", [LLMErrorCategory.CONNECTION, LLMErrorCategory.TIMEOUT])
def test_provider_failure_retains_partial_findings(category):
    class Unavailable(ScriptedDraft):
        def generate(self, prompt, schema):
            performance = super().generate(prompt, schema).performance
            raise LLMError("Simulated provider failure", performance.model_copy(
                update={"error_category": category, "validation_passed": False},
            ))

    run = replay_research(CASES[7])
    report = generate_research_report(run, Unavailable())
    assert report.narrative.validation_succeeded is None and report.narrative.fallback_used
    assert report.narrative.failure_category == category
    assert report.supported_findings and report.coverage[-1].error_code == "PROVIDER_UNAVAILABLE"
    validate_research_report(report, run)


def test_draft_context_omits_raw_results_and_saved_report_omits_document_text():
    run = replay_research(CASES[4])
    provider = DraftProvider(lambda draft: draft)
    report = generate_research_report(run, provider)
    assert provider.prompt and report.narrative.mode == "llm_assisted"
    assert run.state.tool_results[-1].data["excerpt"] not in provider.prompt
    assert run.state.tool_results[-1].data["excerpt"] not in report.model_dump_json()
    assert not report.goal_satisfied and report.coverage[-1].truncated


@pytest.mark.parametrize("field", ["citation_number", "evidence_id", "numeric_value", "requested_date"])
def test_structured_report_rejects_broken_links_and_mismatched_metrics(field):
    report = build_research_report(replay_research(CASES[2])).model_dump()
    if field == "citation_number":
        report["supported_findings"][0]["citation_numbers"] = [99]
    elif field == "evidence_id":
        report["supported_findings"][0]["evidence_ids"] = ["missing-evidence"]
    elif field == "numeric_value":
        report["metrics"][0]["stock"]["return_pct"] = 900.0
    else:
        report["start_date"] = "2026-08-27"
    with pytest.raises(ValidationError):
        ResearchReport.model_validate(report)


def test_realistic_fabricated_url_and_changed_claim_are_rejected_against_state():
    run = replay_research(CASES[2])
    data = build_research_report(run).model_dump(mode="json")
    data["citations"][0]["url"] = "https://example.invalid/not-the-source"
    with pytest.raises(ValueError, match="grounded projection"):
        validate_research_report(ResearchReport.model_validate(data), run)
    data["citations"][0]["url"] = str(run.state.evidence[0].source_url)
    data["supported_findings"][0]["statement"] = "Invented result, 900%."
    with pytest.raises(ValueError, match="grounded projection"):
        validate_research_report(ResearchReport.model_validate(data), run)


def test_raw_benchmark_endpoints_and_percentage_point_difference_are_checked():
    class DifferentAdjustedCloses(demo.FixturePrices):
        def get_price_history(self, request):
            history = super().get_price_history(request)
            last = history.bars[-1].model_copy(update={"adjusted_close": 111.0})
            return history.model_copy(update={"bars": [history.bars[0], last]})

    tools = ToolRegistry(
        market_data=MarketDataTool(DifferentAdjustedCloses()),
        provenance={ToolName.MARKET_DATA: ObservationProvenance.SIMULATED_FIXTURE},
    )
    model = Model(
        {**draft("price_comparison"), "ticker": "GOOGL", "benchmark": "QQQ"},
        plan("market_data", {**PRICE_ARGS, "benchmark": "QQQ", "price_basis": "raw_close"}),
    )
    question = f"What was the raw closing price return for GOOGL versus QQQ from {SPAN}?"
    run = ResearchRunner(model, tools).run(question)
    report = build_research_report(run)
    metric, = report.metrics
    assert isinstance(metric, MarketMetric)
    assert report.goal_satisfied and metric.price_basis == "raw_close"
    assert metric.stock.end_price == 94 and metric.benchmark.end_price == 99
    assert metric.stock.return_pct == pytest.approx(-6)
    assert metric.relative_performance_percentage_points == pytest.approx(-5)
    assert "percentage points" in render_report_markdown(report)
    fabricated = metric.model_dump()
    fabricated["relative_performance_percentage_points"] = 500.0
    with pytest.raises(ValidationError):
        MarketMetric.model_validate(fabricated)


@pytest.mark.parametrize("closing,matched", [(94, True), (95, False)])
def test_calculator_links_underlying_market_result_only_when_operands_match(closing, matched):
    def calculate(prompt, schema):
        result_id = re.search(r'"result_id":"([^"]+)"', prompt).group(1)
        return plan("calculator", {
            "operation": "percentage_return", "starting_price": 100, "ending_price": closing,
            "source_result_ids": [result_id], "price_basis": "adjusted_close",
        })

    model = Model(draft(), SEEDS, plan("market_data"), calculate, plan(arguments={}, gap="The cause is unresolved."))
    tools = fixture_registry(price=demo.FixturePrices(), calculator=True, sec=demo.FixtureFilings())
    run = ResearchRunner(model, tools).run(PRICE_Q)
    report = build_research_report(run)
    calculation = next(metric for metric in report.metrics if isinstance(metric, CalculationMetric))
    finding = next(item for item in report.supported_findings if item.finding_id == calculation.finding_id)
    assert calculation.source_result_ids == ([run.state.tool_results[0].result_id] if matched else [])
    assert len(finding.citation_numbers) == (2 if matched else 1)
    assert finding.provenance[0] == "derived"
    assert calculation.value == pytest.approx(closing - 100)
    validate_research_report(report, run)


def test_repeated_source_keeps_one_bibliography_entry_and_both_evidence_traces():
    captured = replay_research(CASES[3])
    state = captured.state.model_dump(mode="json")
    same_url = state["tool_results"][0]["data"]["source_url"]
    state["tool_results"][1]["data"]["source_url"] = same_url
    state["tool_results"][1]["sources"][0]["url"] = same_url
    state["evidence"][1]["source_url"] = same_url
    report = build_research_report(captured.model_copy(update={"state": ResearchState.model_validate(state)}))
    assert len(report.citations) == 1 and len(report.citations[0].records) == 2
    assert report.quality_status == "conflicted" and len(report.disputed_findings) == 2
    assert {tuple(item.citation_numbers) for item in report.disputed_findings} == {(1,)}


def test_export_and_saved_run_replay_do_not_research_or_overwrite(tmp_path, monkeypatch, capsys):
    run = replay_research(CASES[2])
    input_file = tmp_path / "run.json"
    input_file.write_text(run.model_dump_json(), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("saved deterministic replay must not launch providers")

    monkeypatch.setattr(demo, "open_llm_provider", forbidden)
    monkeypatch.setattr(demo, "build_tools", forbidden)
    output = tmp_path / "reports"
    args = ["--from-run", str(input_file), "--deterministic-report", "--report-json", "--output-dir", str(output)]
    assert demo.main(args) == 0
    content = capsys.readouterr()
    report = ResearchReport.model_validate_json(content.out)
    [path] = list(output.iterdir())
    assert path.name == f"alphascope-{report.report_id.hex}.json"
    assert ResearchReport.model_validate_json(path.read_bytes()) == report
    original = path.read_bytes()
    with pytest.raises(SystemExit):
        demo.main(args)
    assert path.read_bytes() == original
    assert demo.main([*args, "--overwrite"]) == 0
    assert export_research_report(report, output, format="md").is_file()


def test_empty_and_unsupported_outcomes_and_report_evals_work_offline(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("offline evaluations are fixture-only")

    monkeypatch.setattr("requests.sessions.Session.send", denied)
    empty = generate_research_report(replay_research(CASES[1]), ScriptedDraft())
    assert not empty.goal_satisfied and empty.coverage[-1].completeness == "empty"
    assert "DOJ title search result was empty" in render_report_markdown(empty)
    private = build_research_report(replay_research(CASES[6]))
    assert private.quality_status == "unsupported" and private.open_questions
    assert not private.supported_findings and not private.citations
    metrics = run_suite(("E1", "E2", "E3", "E7", "E8", "E9", "E11"), fault_probes=True)
    assert metrics["cases_passed"] == metrics["case_count"]
    assert metrics["fallbacks"] == 3 and metrics["narrative_validation_failures"] == 2
    assert metrics["unsupported_factual_claims"] == metrics["unsupported_causal_claims"] == 0
