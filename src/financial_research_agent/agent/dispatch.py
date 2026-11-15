"""Map vetted research actions to the four read-only tool interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from pydantic import JsonValue

from financial_research_agent.models import (
    ActionStatus, ObservationProvenance, ResearchAction, ResearchState, ToolResult,
)
from financial_research_agent.providers.market_data import PriceBasis
from financial_research_agent.tools import ToolName
from financial_research_agent.tools.calculator import CalculatorTool
from financial_research_agent.tools.market_data import MarketDataTool
from financial_research_agent.tools.news import NewsTool
from financial_research_agent.tools.planning import action_fingerprint, validate_tool_arguments
from financial_research_agent.tools.sec_filings import SecFilingsTool


class ActionRejected(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ToolRegistry:
    """Only these tool interfaces may be dispatched. Bind providers in the caller."""

    def __init__(
        self, *, market_data: MarketDataTool | None = None, calculator: CalculatorTool | None = None,
        sec_filings: SecFilingsTool | None = None, news: NewsTool | None = None,
        provenance: Mapping[ToolName, ObservationProvenance] | None = None,
    ):
        self.market_data = market_data
        self.calculator = calculator
        self.sec_filings = sec_filings
        self.news = news
        self._enabled = tuple(
            tool for tool, handler in (
                (ToolName.MARKET_DATA, market_data), (ToolName.CALCULATOR, calculator),
                (ToolName.SEC_FILINGS, sec_filings), (ToolName.NEWS, news),
            ) if handler is not None
        )
        self._provenance = dict(provenance or {})

    @property
    def enabled(self) -> tuple[ToolName, ...]:
        return self._enabled

    def provenance(self, tool: ToolName) -> ObservationProvenance:
        if tool is ToolName.CALCULATOR:
            return ObservationProvenance.DERIVED
        return self._provenance.get(tool, ObservationProvenance.LIVE_PROVIDER)

    def authorize(
        self, tool: ToolName, arguments: dict[str, JsonValue], state: ResearchState,
    ) -> dict[str, JsonValue]:
        if tool not in self.enabled:
            raise ActionRejected("tool_unavailable", "The proposed tool is unavailable in this run.")
        try:
            args = validate_tool_arguments(tool, arguments, state)
        except (ValueError, TypeError):
            raise ActionRejected("invalid_arguments", "Arguments failed the registered tool contract.") from None
        current = action_fingerprint(tool, args)
        for prior in state.completed_actions:
            if prior.tool_name != tool:
                continue
            try:
                previous = validate_tool_arguments(tool, prior.arguments, state)
            except (ValueError, TypeError):
                continue
            if action_fingerprint(tool, previous) == current:
                raise ActionRejected("duplicate_action", "Equivalent tool arguments were already attempted.")

        return args

    def execute(self, state: ResearchState, action: ResearchAction) -> ToolResult:
        if action.status is not ActionStatus.PLANNED or action not in state.planned_actions:
            raise ActionRejected("unregistered_action", "Record the validated action before dispatch.")
        try:
            tool = ToolName(action.tool_name)
        except ValueError:
            raise ActionRejected("unknown_tool", "The action names an unknown research tool.") from None
        args = self.authorize(tool, action.arguments, state)

        if tool is ToolName.MARKET_DATA:
            assert self.market_data is not None
            start, end = date.fromisoformat(args["start_date"]), date.fromisoformat(args["end_date"])
            if "benchmark" in args:
                return self.market_data.compare_performance(
                    action_id=action.action_id, ticker=args["ticker"], benchmark=args["benchmark"],
                    start_date=start, end_date=end, price_basis=PriceBasis(args["price_basis"]),
                )
            return self.market_data.get_history(
                action_id=action.action_id, ticker=args["ticker"], start_date=start, end_date=end,
            )
        if tool is ToolName.CALCULATOR:
            assert self.calculator is not None
            return self.calculator.execute(action.model_copy(update={"arguments": args}))
        if tool is ToolName.SEC_FILINGS and "filing" in args:
            assert self.sec_filings is not None
            return self.sec_filings.get_document_excerpt(action_id=action.action_id, filing=args["filing"])

        start = date.fromisoformat(args["start_date"]) if args["start_date"] is not None else None
        end = date.fromisoformat(args["end_date"]) if args["end_date"] is not None else None
        if tool is ToolName.SEC_FILINGS:
            assert self.sec_filings is not None
            return self.sec_filings.get_company_filings(
                action_id=action.action_id, ticker=args["ticker"], forms=args["forms"],
                start_date=start, end_date=end, limit=args["limit"],
            )
        assert self.news is not None
        return self.news.search_news(
            action_id=action.action_id, query=args["query"], start_date=start, end_date=end, limit=args["limit"],
        )
