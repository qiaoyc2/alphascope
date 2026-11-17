"""Exercise both report formats using evidence produced by the existing runner."""

from __future__ import annotations

import json
from contextlib import contextmanager
from html import unescape

import pytest

from evals.cases import CASES, PRICE_ARGS, RANGE, SPAN, ScriptedProvider, conflicting_state, registry
from examples import autonomous_research_demo as demo
from financial_research_agent.agent.budget import ResearchLimits
from financial_research_agent.agent.dispatch import ToolRegistry
from financial_research_agent.agent.quality import ClaimKind
from financial_research_agent.agent.report import ResearchReport, build_research_report, render_report_markdown
from financial_research_agent.agent.runner import ResearchRunner, StopReason
from financial_research_agent.agent.state import finish_research
from financial_research_agent.models import ObservationProvenance, ResearchStatus
from financial_research_agent.tools import ToolName
from financial_research_agent.tools.market_data import MarketDataTool
from test_m4_runner import Model, draft, plan


def _run(case_number: int, *, limits=None, tools=None):
    case = CASES[case_number - 1]
    return ResearchRunner(ScriptedProvider(case), tools or registry(case)).run(
        case.question, limits or case.limits, today=case.today,
    )


def test_return_report_links_the_summary_to_checked_evidence_and_round_trips():
    run = _run(3)
    snapshot = run.model_dump()
    report = build_research_report(run)
    finding, = report.supported_findings
    source, = report.citations
    assert report.question == run.state.goal.query and report.goal_satisfied
    assert "+6.00%" in report.executive_summary and report.executive_finding_id == finding.finding_id
    assert finding.evidence_ids == [run.state.evidence[0].evidence_id]
    assert finding.citation_numbers == [source.number]
    assert source.observation_id in finding.observation_ids and source.result_id == run.state.tool_results[0].result_id
    assert finding.provenance == ["simulated_fixture"] and not finding.independently_verified
    assert str(source.url) == str(run.state.evidence[0].source_url) and source.retrieved_at is not None
    assert ResearchReport.model_validate_json(report.model_dump_json()) == report
    assert run.model_dump() == snapshot
    markdown = render_report_markdown(report)
    assert markdown.count("+6.00%") == 1
    assert r"\[1\]" in markdown and "example.invalid" not in markdown
    assert f"Simulated fixture — GOOGL price history (evidence {finding.evidence_ids[0]})" in markdown
    assert f"**Requested interval** {SPAN}" in markdown
    assert "**Endpoint return**" in markdown
    assert "two closing-price observations, on 2026-08-28 and 2026-10-12" in markdown
    assert "The intervening price path is unobserved." in markdown
    assert "simulated fixture, independent verification unavailable" in markdown


def test_public_provider_citation_remains_clickable():
    class PricesWithPublicURL(demo.FixturePrices):
        def get_price_history(self, request):
            return super().get_price_history(request).model_copy(
                update={"source_url": "https://finance.yahoo.com/quote/GOOGL/history/"},
            )

    tools = ToolRegistry(market_data=MarketDataTool(PricesWithPublicURL(ending_price=106)))
    report = build_research_report(_run(3, tools=tools))
    markdown = render_report_markdown(report)
    assert "[1](<https://finance.yahoo.com/quote/GOOGL/history/>)" in markdown
    assert "live provider" in markdown


def test_benchmark_comparison_cites_each_price_source():
    class PricesBySymbol(demo.FixturePrices):
        def get_price_history(self, request):
            history = super().get_price_history(request)
            return history.model_copy(update={"source_url": f"https://example.invalid/{request.ticker}"})

    tools = ToolRegistry(
        market_data=MarketDataTool(PricesBySymbol()),
        provenance={ToolName.MARKET_DATA: ObservationProvenance.SIMULATED_FIXTURE},
    )
    model = Model(
        {**draft("price_comparison"), "ticker": "GOOGL", "benchmark": "QQQ"},
        plan("market_data", {**PRICE_ARGS, "benchmark": "QQQ"}),
    )
    question = f"What was the adjusted closing price return for GOOGL versus QQQ from {SPAN}?"
    report = build_research_report(ResearchRunner(model, tools).run(question))
    assert report.goal_satisfied and report.benchmark == "QQQ"
    assert report.supported_findings[0].citation_numbers == [1, 2]
    assert [str(source.url) for source in report.citations] == [
        "https://example.invalid/GOOGL", "https://example.invalid/QQQ",
    ]
    assert all(citation.retrieved_at for citation in report.citations)


def test_price_contradiction_keeps_the_measured_return_and_requests_clarification():
    report = build_research_report(_run(1))
    assert report.stop_reason is StopReason.NEEDS_CLARIFICATION and not report.goal_satisfied
    assert "contradicts" in report.executive_summary
    assert "+3.00%" in report.supported_findings[0].statement
    assert report.hypotheses and all(item.depends_on_contradicted_premise for item in report.hypotheses)
    assert all(not item.evidence_needed for item in report.hypotheses)
    assert report.open_questions == ["Should research follow the observed movement or use a different period or premise?"]
    assert "depends on the contradicted price premise" in render_report_markdown(report)


def test_causal_report_cites_timing_and_keeps_hypotheses_provisional():
    report = build_research_report(_run(9))
    assert not report.goal_satisfied and report.stop_reason is StopReason.INSUFFICIENT_EVIDENCE
    assert report.hypotheses and "cause remains unresolved" in report.executive_summary
    assert {item.kind for item in report.supported_findings} == {
        ClaimKind.MARKET_RETURN, ClaimKind.EVENT_RECORD, ClaimKind.TEMPORAL_ASSOCIATION,
    }
    timing = next(item for item in report.supported_findings if item.kind is ClaimKind.TEMPORAL_ASSOCIATION)
    assert set(timing.citation_numbers) == {1, 2}
    assert "Timing alone does not establish a causal link." in timing.limitations
    markdown = render_report_markdown(report)
    assert "## Hypotheses" in markdown and "remain unverified" in markdown
    assert "Evidence to check" in markdown
    assert "Timing alone does not establish a causal link." in markdown


def test_title_lookup_discloses_inspected_coverage():
    report = build_research_report(_run(11))
    assert report.goal_satisfied and report.supported_findings[0].kind is ClaimKind.EVENT_RECORD
    markdown = render_report_markdown(report)
    assert "one supported DOJ title match" in markdown
    assert r"\[1\]" in markdown and "example.invalid" not in markdown
    assert "Simulated fixture" in markdown
    assert "inspected results" in markdown


def test_truncated_filing_preserves_the_gap_without_exporting_the_excerpt():
    run = _run(5)
    report = build_research_report(run)
    assert not report.goal_satisfied and report.quality_status == "partial"
    assert report.open_questions and report.coverage[-1].truncated
    assert report.coverage[-1].truncation_reason == "character_limit"
    for output in (report.model_dump_json(), render_report_markdown(report)):
        assert run.state.tool_results[-1].data["excerpt"] not in output
        assert "character_limit" in output or r"character\_limit" in output


def test_tool_failure_and_budget_stop_keep_partial_facts():
    failure = build_research_report(_run(8))
    budget = build_research_report(_run(2, limits=ResearchLimits(max_steps=1)))
    assert failure.stop_reason is StopReason.TOOL_FAILURE
    assert failure.coverage[-1].error_code == "PROVIDER_UNAVAILABLE"
    assert failure.supported_findings[0].kind is ClaimKind.MARKET_RETURN and len(failure.citations) == 1
    assert budget.stop_reason is StopReason.BUDGET_EXHAUSTED and not budget.goal_satisfied
    assert budget.stop_explanation == "The planning step limit was reached."
    assert budget.supported_findings and budget.open_questions


def test_comparable_conflicts_are_disputed_with_both_source_links():
    state = finish_research(
        conflicting_state(CASES[3]), ResearchStatus.STOPPED, StopReason.NEEDS_CLARIFICATION.value,
    )
    run = _run(3).model_copy(update={"state": state, "stop_reason": StopReason.NEEDS_CLARIFICATION, "quality": None})
    report = build_research_report(run)
    assert report.quality_status == "conflicted" and not report.goal_satisfied and not report.supported_findings
    assert len(report.disputed_findings) == len(report.citations) == 2
    assert {finding.citation_numbers[0] for finding in report.disputed_findings} == {1, 2}
    assert "Disputed measurements" in render_report_markdown(report)


def test_report_rechecks_cached_findings_and_unverifiable_completion():
    run = _run(3)
    bogus = run.quality.supported_findings[0].model_copy(update={"statement": "Falsified return: +900%."})
    cached = run.quality.model_copy(update={"supported_findings": [bogus]})
    rebuilt = build_research_report(run.model_copy(update={"quality": cached}))
    assert "+6.00%" in rebuilt.executive_summary and "900%" not in rebuilt.model_dump_json()

    partial = _run(9)
    invalid_stop = finish_research(partial.state, ResearchStatus.COMPLETED, StopReason.GOAL_SATISFIED.value)
    report = build_research_report(partial.model_copy(update={
        "state": invalid_stop, "stop_reason": StopReason.GOAL_SATISFIED, "quality": cached,
    }))
    assert report.stop_reason is StopReason.GOAL_SATISFIED and not report.goal_satisfied
    markdown = render_report_markdown(report)
    assert "**Completion unverified**" in markdown and "**Goal satisfied**" not in markdown


def test_report_omits_credentials_in_source_urls_and_escapes_inline_markup():
    class PricesWithCredentials(demo.FixturePrices):
        def get_price_history(self, request):
            return super().get_price_history(request).model_copy(
                update={"source_url": "https://example.invalid/prices?access_token=sample-private-value"},
            )

    tools = ToolRegistry(market_data=MarketDataTool(PricesWithCredentials(ending_price=106)))
    report = build_research_report(_run(3, tools=tools))
    assert report.goal_satisfied and report.citations[0].url is None
    assert "sample-private-value" not in report.model_dump_json()
    report.question = "[see details](https://example.invalid/other)\n# Unexpected heading <script>alert(1)</script>"
    markdown = render_report_markdown(report)
    assert r"\[see details\]\(" in markdown and "<script>" not in markdown
    assert "\n# Unexpected" not in markdown and "<script>" in unescape(markdown)
    assert "sample-private-value" not in markdown


@pytest.mark.parametrize("option", ["--report", "--report-json", "--json"])
def test_existing_demo_cli_can_render_a_report_or_the_existing_trace(option, monkeypatch, capsys):
    case = CASES[2]

    @contextmanager
    def provider(settings):
        yield ScriptedProvider(case)

    monkeypatch.setattr(demo, "load_llm_settings", lambda: None)
    monkeypatch.setattr(demo, "open_llm_provider", provider)
    monkeypatch.setattr(demo, "build_tools", lambda scenario: registry(case))
    assert demo.main(["--question", case.question, option]) == 0
    output = capsys.readouterr().out
    if option == "--report":
        assert "# AlphaScope research report" in output and "**Goal satisfied**" in output
    elif option == "--report-json":
        report = ResearchReport.model_validate_json(output)
        assert report.goal_satisfied and report.citations
    else:
        trace = json.loads(output)
        assert trace["state"]["goal"]["query"] == case.question and trace["tool_results"]
        assert "executive_summary" not in trace
