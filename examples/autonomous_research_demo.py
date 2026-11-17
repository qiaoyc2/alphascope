"""Run bounded research and display a trace or an evidence-grounded report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from financial_research_agent.agent.budget import ResearchLimits
from financial_research_agent.agent.dispatch import ToolRegistry
from financial_research_agent.agent.report import render_report_markdown
from financial_research_agent.agent.report_export import export_research_report
from financial_research_agent.agent.report_narrative import generate_research_report
from financial_research_agent.agent.runner import ResearchRun, ResearchRunner
from financial_research_agent.config import load_llm_settings, open_llm_provider
from financial_research_agent.models import ObservationProvenance, utc_now
from financial_research_agent.providers.doj_press_releases import DoJPressReleaseProvider
from financial_research_agent.providers.market_data import (
    MarketDataError, MarketDataErrorCode, PriceBar, PriceHistory, PriceHistoryRequest,
)
from financial_research_agent.providers.news import NewsArticle, NewsRequest, NewsSearchResult
from financial_research_agent.providers.sec_edgar import (
    SecCompany, SecEdgarProvider, SecExcerpt, SecFiling, SecFilingRequest, SecFilingsResult,
)
from financial_research_agent.providers.yahoo_finance import YahooFinanceProvider
from financial_research_agent.tools import ToolName
from financial_research_agent.tools.calculator import CalculatorTool
from financial_research_agent.tools.market_data import MarketDataTool
from financial_research_agent.tools.news import NewsTool
from financial_research_agent.tools.sec_filings import SecFilingsTool


SCENARIOS = {
    "live-market": "Why did GOOGL decline between August 28 and October 12, 2026?",
    "live-doj": "What official DOJ press-release titles mention Google from 2024-01-01 to 2026-10-12?",
    "fixture": "Why did GOOGL decline from 2026-08-28 to 2026-10-12?",
    "fixture-rise": "Why did GOOGL decline from 2026-08-28 to 2026-10-12?",
    "fixture-return": "What was the historical adjusted closing price return for GOOGL from 2026-08-28 to 2026-10-12?",
    "fixture-doj": "What DOJ press-release titles mention Google from 2024-01-01 to 2026-10-12?",
    "failure": "Why did GOOGL decline from 2026-08-28 to 2026-10-12?",
}


class FixturePrices:
    def __init__(self, *, failed: bool = False, ending_price: float = 94):
        self.failed = failed
        self.ending_price = ending_price

    def get_price_history(self, request: PriceHistoryRequest) -> PriceHistory:
        if self.failed:
            raise MarketDataError(MarketDataErrorCode.PROVIDER_UNAVAILABLE, "Simulated provider unavailable", retryable=True)
        if request.ticker not in {"GOOGL", "QQQ"}:
            raise MarketDataError(MarketDataErrorCode.NO_DATA, "The simulation has no data for this ticker")
        ending_price = self.ending_price if request.ticker == "GOOGL" else 99
        bars = [
            PriceBar(trading_date=day, open=price, high=price, low=price, close=price,
                     adjusted_close=price, volume=1000)
            for day, price in ((request.start_date, 100.0), (request.end_date, ending_price))
        ]
        return PriceHistory(
            ticker=request.ticker, requested_start_date=request.start_date, requested_end_date=request.end_date,
            actual_start_date=request.start_date, actual_end_date=request.end_date, bars=bars,
            currency="USD", provider="simulated_prices", source_url="https://example.invalid/simulated-prices",
            retrieved_at=utc_now(), adjustment_policy="simulated adjusted closes",
        )


class FixtureNews:
    def search_news(self, request: NewsRequest) -> NewsSearchResult:
        now = utc_now()
        feed = "https://example.invalid/simulated-doj"
        title = "[Simulated] Agency opens a competition review involving Google parent Alphabet"
        return NewsSearchResult(
            query=request.query, feed_url=feed, retrieved_at=now, api_call_count=0, provider="simulated_doj",
            articles=[NewsArticle(
                article_id="simulated-release", title=title,
                publisher="Simulated DOJ fixture", publication_date=request.end_date or date(2026, 10, 12),
                publication_date_status="simulated", retrieved_at=now,
                url="https://example.invalid/simulated-doj/release", feed_url=feed, query=request.query,
                provider="simulated_doj", source_type="simulated_fixture",
            )] if request.query.casefold() in title.casefold() else [],
        )


class FixtureFilings:
    def get_company_filings(self, request: SecFilingRequest) -> SecFilingsResult:
        now = utc_now()
        company = SecCompany(company_name="Alphabet (simulated)", cik="0000000001", tickers=["GOOGL"])
        return SecFilingsResult(
            company=company, submissions_url="https://example.invalid/simulated-sec", retrieved_at=now,
            api_call_count=0, provider="simulated_sec",
            filings=[SecFiling(
                company_name=company.company_name, cik=company.cik, form=request.forms[0],
                filing_date=request.end_date or date(2026, 10, 12), acceptance_timezone_status="unavailable",
                accession_number="0000000001-26-000001", primary_document="example.htm",
                filing_url="https://example.invalid/simulated-sec/index",
                primary_document_url="https://example.invalid/simulated-sec/example.htm", retrieved_at=now,
            )],
        )

    def get_document_excerpt(self, filing: SecFiling) -> SecExcerpt:
        text = "[Simulated] Management reported lower advertising demand. The document continues beyond this excerpt."
        return SecExcerpt(
            company_name=filing.company_name, cik=filing.cik, accession_number=filing.accession_number,
            form=filing.form, filing_date=filing.filing_date, document_url=filing.primary_document_url,
            excerpt=text, retrieved_at=utc_now(), extraction_incomplete=True, truncated=True,
            truncation_reason="character_limit", characters_returned=len(text), max_excerpt_characters=len(text),
            provider="simulated_sec", api_call_count=0,
        )


def build_tools(scenario: str) -> ToolRegistry:
    calculator = CalculatorTool()
    fixture = ObservationProvenance.SIMULATED_FIXTURE
    if scenario == "fixture-return":
        return ToolRegistry(
            market_data=MarketDataTool(FixturePrices(ending_price=106)), provenance={ToolName.MARKET_DATA: fixture},
        )
    if scenario == "fixture-doj":
        return ToolRegistry(news=NewsTool(FixtureNews()), provenance={ToolName.NEWS: fixture})
    if scenario in {"fixture", "fixture-rise", "failure"}:
        return ToolRegistry(
            market_data=MarketDataTool(FixturePrices(failed=scenario == "failure", ending_price=103 if scenario == "fixture-rise" else 94)),
            calculator=calculator,
            news=NewsTool(FixtureNews()) if scenario == "fixture" else None,
            sec_filings=SecFilingsTool(FixtureFilings()) if scenario == "fixture" else None,
            provenance={tool: fixture for tool in (ToolName.MARKET_DATA, ToolName.NEWS, ToolName.SEC_FILINGS)},
        )
    if scenario == "live-doj":
        return ToolRegistry(news=NewsTool(DoJPressReleaseProvider()), calculator=calculator)
    try:
        sec = SecFilingsTool(SecEdgarProvider(user_agent=os.environ.get("ALPHASCOPE_SEC_USER_AGENT", "")))
    except ValueError:
        sec = None  # No approved SEC contact is configured.
    return ToolRegistry(
        market_data=MarketDataTool(YahooFinanceProvider()), calculator=calculator,
        news=NewsTool(DoJPressReleaseProvider()), sec_filings=sec,
    )


def report(run: ResearchRun, scenario: str, machine_readable: bool) -> None:
    observations = {item.observation_id: item for item in run.state.observations}
    if machine_readable:
        result = run.model_dump(mode="json", exclude={"state": {"tool_results"}})
        result["scenario"] = scenario
        result["tool_results"] = [
            {"result_id": item.result_id, "action_id": item.action_id, "tool": item.tool_name,
             "success": item.success, "error_code": item.error.code if item.error else None,
             "sources": [str(source.url) for source in item.sources[:5]]}
            for item in run.state.tool_results
        ]
        print(json.dumps(result, ensure_ascii=True))
        return
    print(f"{scenario} | {run.state.goal.query}")
    print(f"Goal: {run.state.goal.objective}")
    for step in run.steps:
        print(f"{step.number}. {step.proposed_tool or step.recommended_action or 'no tool'} | {step.validation}")
        if step.note:
            print(f"   {step.note}")
        if step.observation_id:
            observation = observations[step.observation_id]
            print(f"   {observation.provenance} / {observation.outcome}: {observation.summary}")
        if step.hypothesis_changes:
            print(f"   Hypothesis changes: {step.hypothesis_changes}")
    for item in run.state.hypotheses:
        print(f"Hypothesis {item.hypothesis_id} [{item.status}]: {item.statement}")
    for gap in run.state.research_gaps:
        print(f"Gap: {gap}")
    for issue in run.issues:
        print(f"{issue.code}: {issue.message}")
    stats = run.statistics
    print(f"Stop: {run.stop_reason} | {run.stop_detail}")
    print(
        f"{stats.tool_calls} tool calls, {stats.llm_requests} LLM requests, "
        f"{stats.input_tokens_reported}/{stats.output_tokens_reported} reported tokens, {stats.elapsed_seconds:.2f}s"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="fixture")
    parser.add_argument("--question", help="Override the scenario's research question")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--tool-calls", type=int, default=5)
    parser.add_argument("--llm-requests", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=180)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="Print the existing machine-readable research trace.")
    output.add_argument("--report", action="store_true", help="Print a cited Markdown research report.")
    output.add_argument("--report-json", action="store_true", help="Print the structured research report as JSON.")
    parser.add_argument("--from-run", type=Path, help="Load a full ResearchRun JSON file and reuse its research.")
    parser.add_argument("--output-dir", type=Path, help="Also export the selected report format to this directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing report file for this run.")
    parser.add_argument("--deterministic-report", action="store_true", help="Render without requesting a narrative draft.")
    args = parser.parse_args(argv)
    wants_report = args.report or args.report_json
    if not wants_report and (args.from_run or args.output_dir or args.overwrite or args.deterministic_report):
        parser.error("Replay, export, and report options require --report or --report-json.")
    if args.overwrite and not args.output_dir:
        parser.error("--overwrite requires --output-dir.")
    if args.from_run and args.question:
        parser.error("--from-run uses the saved research question.")

    if args.from_run:
        try:
            run = ResearchRun.model_validate_json(args.from_run.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            parser.error("The input must be a readable full ResearchRun JSON file.")
        if args.deterministic_report:
            research_report = generate_research_report(run)
        else:
            try:
                settings = load_llm_settings()
            except ValueError:
                research_report = generate_research_report(run, narrative_requested=True)
            else:
                with open_llm_provider(settings) as provider:
                    research_report = generate_research_report(run, provider)
    else:
        try:
            limits = ResearchLimits(
                max_steps=args.steps, max_tool_calls=args.tool_calls, max_llm_requests=args.llm_requests,
                runtime_seconds=args.seconds,
            )
            settings = load_llm_settings()
        except ValueError:
            parser.error("Invalid research limits or local LLM configuration.")
        with open_llm_provider(settings) as provider:
            run = ResearchRunner(provider, build_tools(args.scenario)).run(
                args.question or SCENARIOS[args.scenario], limits,
            )
            if wants_report:
                research_report = generate_research_report(run, None if args.deterministic_report else provider)
    if wants_report:
        if args.output_dir:
            try:
                destination = export_research_report(
                    research_report, args.output_dir, format="json" if args.report_json else "md", overwrite=args.overwrite,
                )
            except FileExistsError:
                parser.error("A report file already exists. Use --overwrite to replace it.")
            except OSError:
                parser.error("The report could not be written to the requested directory.")
            print(f"Saved {destination}", file=sys.stderr)
        if args.report_json:
            print(json.dumps(research_report.model_dump(mode="json"), ensure_ascii=True))
        else:
            print(render_report_markdown(research_report), end="")
    else:
        report(run, args.scenario, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
