"""Validate proposed arguments without constructing providers or executing tools."""

from __future__ import annotations

import json
from collections.abc import Mapping

from pydantic import Field, JsonValue

from financial_research_agent.models import ResearchState
from financial_research_agent.providers.market_data import PriceBasis, PriceHistoryRequest
from financial_research_agent.providers.news import NewsRequest
from financial_research_agent.providers.sec_edgar import SecFiling, SecFilingRequest
from financial_research_agent.tools import ToolName
from financial_research_agent.tools.calculator import validate_calculation_arguments


TOOL_CAPABILITIES = {
    ToolName.MARKET_DATA: "Price or aligned benchmark returns. Args: ticker,start_date,end_date, optionally benchmark and price_basis.",
    ToolName.CALCULATOR: "Calculates from supplied inputs. Args: operation and its numeric inputs. For percentage_return use starting_price,ending_price; for relative_performance use stock_return_pct,benchmark_return_pct.",
    ToolName.SEC_FILINGS: "Company filings and excerpts. Args: ticker (or exact company name), optional forms,start_date,end_date,limit. For a known filing excerpt use accession_number from context.",
    ToolName.NEWS: "Official DOJ announcements only. Args: query, optional start_date,end_date,limit. Search title metadata with a short literal phrase such as a company name. No article bodies, general financial news or private sources.",
}


class ComparisonRequest(PriceHistoryRequest):
    benchmark: str
    price_basis: PriceBasis = PriceBasis.ADJUSTED_CLOSE


class FilingSearch(SecFilingRequest):
    identifier: str = Field(alias="ticker", min_length=1, max_length=120)


def validate_tool_arguments(
    tool: ToolName, arguments: Mapping[str, JsonValue], state: ResearchState,
) -> dict[str, JsonValue]:
    if tool is ToolName.MARKET_DATA:
        if "benchmark" not in arguments:
            return PriceHistoryRequest.model_validate(arguments).model_dump(mode="json")
        request = ComparisonRequest.model_validate(arguments)
        benchmark = PriceHistoryRequest(
            ticker=request.benchmark, start_date=request.start_date, end_date=request.end_date,
        ).ticker
        if request.ticker == benchmark:
            raise ValueError("benchmark must differ from the stock ticker")
        return {**request.model_dump(mode="json"), "benchmark": benchmark}
    if tool is ToolName.NEWS:
        return NewsRequest.model_validate(arguments).model_dump(mode="json")
    if tool is ToolName.CALCULATOR:
        normalized = validate_calculation_arguments(dict(arguments))
        result_ids = {result.result_id for result in state.tool_results if result.success}
        if set(normalized["source_result_ids"]) - result_ids:
            raise ValueError("calculator source references must identify existing successful results")
        return normalized
    if tool is not ToolName.SEC_FILINGS:
        raise ValueError("unknown research tool")
    if arguments.keys() in ({"accession_number"}, {"filing"}):
        supplied = SecFiling.model_validate(arguments["filing"]) if "filing" in arguments else None
        accession = supplied.accession_number if supplied else arguments["accession_number"]
        for result in reversed(state.tool_results):
            if not result.success or result.tool_name != tool or result.data.get("kind") != "sec_filings":
                continue
            records = result.data.get("filings")
            if not isinstance(records, list):
                continue
            for record in records:
                if isinstance(record, dict) and record.get("accession_number") == accession:
                    filing = SecFiling.model_validate(record)
                    if (supplied is None or filing == supplied) and filing.primary_document_url:
                        return {"filing": filing.model_dump(mode="json")}
        raise ValueError("SEC excerpt needs a known filing with a primary document")
    return FilingSearch.model_validate(arguments).model_dump(mode="json", by_alias=True)


def action_signature(tool: ToolName, arguments: dict[str, JsonValue]) -> str:
    """Compare canonical intent independently of argument ordering and casing."""
    key = arguments.copy()
    if tool is ToolName.NEWS:
        key["query"] = " ".join(str(key["query"]).casefold().split())
    if tool is ToolName.SEC_FILINGS:
        if "filing" in key:
            key = {"accession_number": key["filing"]["accession_number"]}
        else:
            key["ticker"] = str(key["ticker"]).casefold()
            key["forms"] = sorted(set(key["forms"]))
    if tool is ToolName.CALCULATOR:
        # Provenance metadata does not change a calculation's numerical input.
        for field in ("source_result_ids", "price_basis", "effective_start_date", "effective_end_date"):
            key.pop(field, None)
    return json.dumps(key, sort_keys=True, separators=(",", ":"))
