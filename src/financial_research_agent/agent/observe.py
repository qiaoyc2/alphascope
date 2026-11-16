"""Record tool results and grounded observations using the shared state contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from financial_research_agent.agent.hypotheses import incomplete_sec_excerpt
from financial_research_agent.agent.planning import record_observation
from financial_research_agent.models import (
    Evidence, Observation, ObservationOutcome, ObservationProvenance,
    ResearchAction, ResearchState, ToolResult,
)
from financial_research_agent.providers.market_data import PriceHistory
from financial_research_agent.providers.news import NewsSearchResult
from financial_research_agent.providers.sec_edgar import SecFilingsResult
from financial_research_agent.tools import ToolName
from financial_research_agent.tools.calculator import percentage_return
from financial_research_agent.tools.market_data import BenchmarkComparison


@dataclass(frozen=True)
class SourceFact:
    claim: str
    url: str
    retrieved_at: datetime | None = None
    event_at: datetime | None = None


@dataclass(frozen=True)
class Finding:
    summary: str
    outcome: ObservationOutcome = ObservationOutcome.DATA
    facts: tuple[SourceFact, ...] = ()


def _market_finding(result: ToolResult) -> Finding:
    data = {key: value for key, value in result.data.items() if key != "kind"}
    if result.data.get("kind") == "benchmark_comparison":
        item = BenchmarkComparison.model_validate(data)
        summary = (
            f"{item.stock_ticker} {item.stock_return_pct:+.2f}% vs {item.benchmark_ticker} "
            f"{item.benchmark_return_pct:+.2f}% from {item.effective_start_date} to {item.effective_end_date} "
            f"({item.price_basis}, {item.relative_performance_percentage_points:+.2f} percentage points relative)."
        )
        return Finding(summary, facts=(SourceFact(summary, str(item.stock_source_url), item.stock_retrieved_at),))
    history = PriceHistory.model_validate(data)
    if len(history.bars) < 2:
        return Finding(f"{history.ticker} has only one price observation in the requested range.", ObservationOutcome.INCOMPLETE)
    first, last = history.bars[0], history.bars[-1]
    change = percentage_return(first.adjusted_close, last.adjusted_close)
    summary = (
        f"{history.ticker} returned {change:+.2f}% from {first.trading_date} to {last.trading_date} "
        f"(adjusted close {first.adjusted_close:g} to {last.adjusted_close:g}, {len(history.bars)} sessions)."
    )
    return Finding(summary, facts=(SourceFact(summary, str(history.source_url), history.retrieved_at),))


def _news_finding(result: ToolResult, provenance: ObservationProvenance) -> Finding:
    data = {key: value for key, value in result.data.items() if key != "kind"}
    news = NewsSearchResult.model_validate(data)
    if not news.articles:
        return Finding(
            "No matching DOJ titles were returned by this search. Coverage is limited to inspected API results.",
            ObservationOutcome.EMPTY,
        )
    items = news.articles[:3]
    label = "official DOJ" if provenance is ObservationProvenance.LIVE_PROVIDER else "simulated DOJ"
    summary = f"{len(news.articles)} {label} title matches. " + " | ".join(
        f"{item.publication_date or 'date unavailable'} {item.title[:115]}" for item in items
    )
    facts = tuple(SourceFact(
        f"DOJ release title: {item.title[:240]} ({item.publication_date or 'date unavailable'}).",
        str(item.url), item.retrieved_at, item.published_at,
    ) for item in items)
    return Finding(summary, facts=facts)


def _filing_finding(result: ToolResult) -> Finding:
    if result.data.get("kind") == "sec_document_excerpt":
        incomplete = incomplete_sec_excerpt(result)
        filing = result.data.get("accession_number", "unknown filing")
        text = str(result.data.get("excerpt", ""))[:130]
        qualifier = "Incomplete bounded SEC excerpt" if incomplete else "Bounded SEC excerpt"
        return Finding(
            f"{qualifier} for {filing}. Returned text begins: {text}",
            ObservationOutcome.INCOMPLETE if incomplete else ObservationOutcome.DATA,
        )
    data = {key: value for key, value in result.data.items() if key != "kind"}
    filings = SecFilingsResult.model_validate(data)
    if not filings.filings:
        return Finding("SEC recent submissions returned no matching filings.", ObservationOutcome.EMPTY)
    records = filings.filings[:3]
    summary = f"{len(filings.filings)} SEC filing records for {filings.company.company_name}. " + " | ".join(
        f"{item.form} {item.filing_date} ({item.accession_number})" for item in records
    )
    facts = tuple(SourceFact(
        f"{item.company_name} filed {item.form} on {item.filing_date} ({item.accession_number}).",
        str(item.filing_url), item.retrieved_at, item.acceptance_at,
    ) for item in records)
    return Finding(summary, facts=facts)


def describe_result(result: ToolResult, provenance: ObservationProvenance) -> Finding:
    if not result.success:
        assert result.error is not None
        return Finding(f"{result.tool_name} did not provide evidence ({result.error.code}).", ObservationOutcome.FAILED)
    if result.tool_name == ToolName.MARKET_DATA:
        return _market_finding(result)
    if result.tool_name == ToolName.NEWS:
        return _news_finding(result, provenance)
    if result.tool_name == ToolName.SEC_FILINGS:
        return _filing_finding(result)
    data = result.data
    value = data.get("value", data.get("correlation"))
    output = f"{value:g} {data.get('unit', '')}" if isinstance(value, int | float) else "a calculated series"
    origin = ", ".join(data.get("inputs", {}).get("source_result_ids", [])[:3])
    return Finding(
        f"Calculator {data.get('operation')} returned {output} from supplied inputs. "
        f"Claimed input result IDs: {origin or 'none'}. Input values are not independently verified."
    )


def observe_result(
    state: ResearchState, action: ResearchAction, result: ToolResult, provenance: ObservationProvenance,
) -> ResearchState:
    result = ToolResult.model_validate(result)
    if result.action_id != action.action_id or result.tool_name != action.tool_name:
        raise ValueError("tool result does not match the dispatched action")
    finding = describe_result(result, provenance)
    observation = Observation(
        observation_id="o-" + uuid4().hex[:12], action_id=action.action_id, result_id=result.result_id,
        summary=finding.summary, provenance=provenance, outcome=finding.outcome,
        recorded_at=result.recorded_at,
    )
    known = {(str(item.source_url), item.claim) for item in state.evidence}
    evidence = []
    for fact in finding.facts:
        key = (fact.url, fact.claim)
        if key in known and result.tool_name != ToolName.MARKET_DATA:
            continue
        known.add(key)
        evidence.append(Evidence(
            evidence_id="e-" + uuid4().hex[:12], claim=fact.claim,
            evidence_type=f"{result.tool_name}_source", source_url=fact.url,
            observation_ids=[observation.observation_id],
            retrieved_at=fact.retrieved_at or result.recorded_at, event_at=fact.event_at,
        ))
    if result.success and result.tool_name == ToolName.CALCULATOR:
        evidence.append(Evidence(
            evidence_id="e-" + uuid4().hex[:12], claim=finding.summary,
            evidence_type="calculation", observation_ids=[observation.observation_id],
            calculation_method=result.data["operation"], calculation_inputs=result.data["inputs"],
        ))

    # A successful, non-empty observation can answer a narrowly scoped measurement
    # or metadata lookup. Coverage and causal claims remain open for replanning.
    gap = action.target_information_gap
    resolved = (
        (gap,) if gap in state.research_gaps and finding.outcome is ObservationOutcome.DATA
        and (result.tool_name == ToolName.MARKET_DATA
             or result.data.get("kind") in {"news_search", "sec_filings"}) else ()
    )
    return record_observation(state, result, observation, tuple(evidence), resolved_gaps=resolved)
