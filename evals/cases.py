"""Frozen scenarios for the quality gate and the existing research runner."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Self

from pydantic import Field, ValidationError, model_validator

from examples.autonomous_research_demo import FixtureFilings, FixtureNews, FixturePrices
from financial_research_agent.agent.budget import ResearchLimits
from financial_research_agent.agent.dispatch import ToolRegistry
from financial_research_agent.agent.observe import observe_result
from financial_research_agent.agent.quality import ClaimKind, Completeness, QualityStatus
from financial_research_agent.agent.runner import StopReason
from financial_research_agent.agent.state import add_action
from financial_research_agent.models import (
    ObservationProvenance, ResearchAction, ResearchGoal, ResearchModel, ResearchState, ResearchType,
)
from financial_research_agent.providers.doj_press_releases import DoJPressReleaseProvider
from financial_research_agent.providers.http_client import HttpFailure
from financial_research_agent.providers.llm import LLMError, LLMErrorCategory, LLMPerformance, ProposalMetadata, StructuredLLMResult
from financial_research_agent.providers.yahoo_finance import YahooFinanceProvider
from financial_research_agent.tools import ToolName
from financial_research_agent.tools.market_data import MarketDataTool
from financial_research_agent.tools.news import NewsTool
from financial_research_agent.tools.sec_filings import SecFilingsTool


CLOCK = date(2026, 10, 15)
RETRIEVED = datetime(2026, 10, 15, tzinfo=timezone.utc)
START, END = date(2026, 8, 28), date(2026, 10, 12)
SPAN = f"{START} to {END}"
RANGE = {"start_date": str(START), "end_date": str(END)}
PRICE_ARGS = {"ticker": "GOOGL", **RANGE}
PRICE_QUESTION = f"Why did GOOGL decline from {SPAN}?"
RETURN_QUESTION = f"What was the historical adjusted closing price return for GOOGL from {SPAN}?"
REGULATORY_QUESTION = f"What DOJ press-release titles mention Google from {SPAN}?"
REGULATORY_RANGE = {"start_date": "2024-01-01", "end_date": str(END)}
NEWS_ARGS = {"query": "Google", **RANGE}
FIXTURE = ObservationProvenance.SIMULATED_FIXTURE


class Check(StrEnum):
    ORIGINAL_GOAL = "original_goal"
    PROVENANCE = "provenance"
    EXECUTION_LIMITS = "execution_limits"
    PRICE_FIRST = "price_first"
    PREMISE_CONTRADICTED = "premise_contradicted"
    CONFLICT_RECORDED = "conflict_recorded"
    DUPLICATE_SOURCE = "duplicate_source"
    DUPLICATE_PROPOSAL = "duplicate_proposal"
    FAILED_WITHOUT_EVIDENCE = "failed_without_evidence"
    QUALITY_VETO = "quality_veto"
    CORRECTED_ACTION = "corrected_action"
    SOURCE_LIMITATIONS = "source_limitations"


class Expectations(ResearchModel):
    stops: set[StopReason] = Field(min_length=1)
    quality: set[QualityStatus] = Field(min_length=1)
    checks: set[Check] = Field(min_length=1)
    findings: set[ClaimKind] = Field(default_factory=set)
    prohibited_findings: set[ClaimKind] = Field(default_factory=lambda: {ClaimKind.CAUSAL_CONCLUSION})
    required_tools: set[ToolName] = Field(default_factory=set)
    prohibited_tools: set[ToolName] = Field(default_factory=set)
    completeness: set[Completeness] = Field(default_factory=set)
    max_tool_calls: int = Field(ge=0)

    @model_validator(mode="after")
    def check_expectations(self) -> Self:
        if self.required_tools & self.prohibited_tools or self.findings & self.prohibited_findings:
            raise ValueError("evaluation requirements must not also be prohibited")
        if len(self.required_tools) > self.max_tool_calls:
            raise ValueError("required tools exceed the expected tool call cap")
        if StopReason.GOAL_SATISFIED in self.stops and QualityStatus.SUFFICIENT not in self.quality:
            raise ValueError("successful stopping requires sufficient quality")
        return self


class EvalCase(ResearchModel):
    case_id: str = Field(pattern=r"^E[0-9]+$")
    label: str
    question: str
    today: date = CLOCK
    tools: tuple[ToolName, ...]
    expectations: Expectations
    offline_proposals_json: str | None
    quality_only: bool = False
    ending_price: float = 94
    empty_news: bool = False
    failed_news: bool = False
    live_source: ToolName | None = None
    limits: ResearchLimits = Field(default_factory=lambda: ResearchLimits(
        max_steps=4, max_tool_calls=3, max_llm_requests=7, runtime_seconds=105,
    ))

    @model_validator(mode="after")
    def check_case(self) -> Self:
        if self.quality_only != (self.offline_proposals_json is None):
            raise ValueError("runner cases require a frozen structured response fixture")
        if self.offline_proposals_json is not None:
            proposals = json.loads(self.offline_proposals_json)
            if not isinstance(proposals, list) or not proposals or not all(isinstance(item, dict) for item in proposals):
                raise ValueError("runner fixtures require a nonempty array of response objects")
        if not self.expectations.required_tools <= set(self.tools) or len(set(self.tools)) != len(self.tools):
            raise ValueError("available tools must be unique and include expected tools")
        if self.live_source is not None and (
            self.live_source not in self.tools or self.live_source not in {ToolName.MARKET_DATA, ToolName.NEWS}
        ):
            raise ValueError("live source must be an available approved evaluation provider")
        return self


def _plan(tool: str | None = None, arguments: dict | None = None, *, gap: str | None = None, action: str | None = None) -> dict:
    item = {
        "recommended_action": action or ("continue" if tool else "stop"),
        "next_tool": tool, "purpose": f"Retrieve evidence with {tool}." if tool else None,
        "information_gap": (gap or "Evidence from the selected source.") if tool else gap,
        "reason": "Assess the evidence needed for the original question.",
        "hypothesis_ids": [], "hypothesis_updates": [],
    }
    if arguments is not None:
        item["arguments"] = arguments
    return item


SEEDS = {"hypotheses": [
    {"statement": "A business development could matter if the decline is verified.", "research_gaps": ["Business disclosures"]},
    {"statement": "An agency action could matter if the decline is verified.", "research_gaps": ["Agency releases"]},
]}
COMMON = {Check.ORIGINAL_GOAL, Check.PROVENANCE, Check.EXECUTION_LIMITS}
PRICE = ToolName.MARKET_DATA
NEWS = ToolName.NEWS
SEC = ToolName.SEC_FILINGS
MARKET = ClaimKind.MARKET_RETURN
EVENT = ClaimKind.EVENT_RECORD
PARTIAL = QualityStatus.PARTIAL
SUFFICIENT = QualityStatus.SUFFICIENT
UNRESOLVED = StopReason.INSUFFICIENT_EVIDENCE
CLARIFY = StopReason.NEEDS_CLARIFICATION
SATISFIED = StopReason.GOAL_SATISFIED


def _case(case_id: str, label: str, question: str, kind: ResearchType, tools: tuple[ToolName, ...],
          stops: set[StopReason], quality: QualityStatus, proposals: tuple[dict, ...] = (), *,
          checks: set[Check] | None = None, findings: set[ClaimKind] | None = None,
          required: set[ToolName] | None = None, prohibited: set[ToolName] | None = None,
          completeness: set[Completeness] | None = None, max_calls: int = 3,
          period: dict[str, str] | None = None, **options) -> EvalCase:
    causal = kind is ResearchType.CAUSAL_PRICE_MOVE
    priced = causal or kind is ResearchType.PRICE_COMPARISON
    period = (period or RANGE) if kind is not ResearchType.OTHER else {}
    dates = {"start_date": None, "end_date": None, **period}
    draft = {
        "objective": question, "research_type": kind, "company": "Google" if kind is not ResearchType.FILINGS else "Alphabet",
        "ticker": "GOOGL" if priced else None, "benchmark": None, "price_premise": "decline" if causal else None,
        "temporal_expression": " to ".join(period.values()) or None, "clarification": None, **dates,
    }
    return EvalCase(
        case_id=case_id, label=label, question=question, tools=tools,
        offline_proposals_json=json.dumps([draft, *([SEEDS] if causal else []), *proposals]) if proposals else None,
        expectations=Expectations(
            stops=stops, quality={quality}, checks=COMMON | (checks or set()), findings=findings or set(),
            required_tools=required or set(), prohibited_tools=prohibited or set(),
            completeness=completeness or set(), max_tool_calls=max_calls,
        ),
        **options,
    )


CASES = (
    _case("E1", "Unsupported decline premise", PRICE_QUESTION, ResearchType.CAUSAL_PRICE_MOVE, (PRICE, NEWS),
          {CLARIFY}, PARTIAL, (_plan("market_data"), _plan("news", NEWS_ARGS)),
          checks={Check.PRICE_FIRST, Check.PREMISE_CONTRADICTED}, findings={MARKET},
          required={PRICE}, prohibited={NEWS}, max_calls=1, ending_price=103, live_source=PRICE),
    _case("E2", "Confirmed decline, unknown cause", PRICE_QUESTION, ResearchType.CAUSAL_PRICE_MOVE, (PRICE, NEWS),
          {UNRESOLVED, StopReason.REPEATED_ACTION, StopReason.BUDGET_EXHAUSTED}, PARTIAL,
          (_plan("market_data"), _plan("news", NEWS_ARGS), _plan(arguments={}, gap="Causal evidence is unavailable.")),
          checks={Check.PRICE_FIRST}, findings={MARKET}, required={PRICE, NEWS},
          completeness={Completeness.EMPTY}, empty_news=True),
    _case("E3", "Sufficient historical return", RETURN_QUESTION, ResearchType.PRICE_COMPARISON, (PRICE,),
          {SATISFIED}, SUFFICIENT, (_plan("market_data", PRICE_ARGS), _plan(arguments={})),
          findings={MARKET}, required={PRICE}, max_calls=1, ending_price=106, live_source=PRICE),
    _case("E4", "Conflicting comparable returns", RETURN_QUESTION, ResearchType.PRICE_COMPARISON, (PRICE,),
          {CLARIFY}, QualityStatus.CONFLICTED, checks={Check.CONFLICT_RECORDED}, quality_only=True),
    _case("E5", "Incomplete SEC excerpt", f"What cybersecurity incident did Alphabet disclose from {SPAN}?",
          ResearchType.FILINGS, (SEC,), {UNRESOLVED}, PARTIAL,
          (_plan("sec_filings", {"ticker": "Alphabet", **RANGE}),
           _plan("sec_filings", {"accession_number": "0000000001-26-000001"}), _plan(arguments={})),
          checks={Check.QUALITY_VETO, Check.SOURCE_LIMITATIONS}, required={SEC},
          completeness={Completeness.TRUNCATED}, findings={ClaimKind.FILING_RECORD}),
    _case("E6", "Duplicate source and duplicate proposal", REGULATORY_QUESTION, ResearchType.REGULATORY, (NEWS,),
          {StopReason.REPEATED_ACTION}, SUFFICIENT,
          (_plan("news", NEWS_ARGS), _plan("news", {**NEWS_ARGS, "query": "Alphabet"}),
           _plan("news", {**NEWS_ARGS, "query": " GOOGLE "}), _plan("news", NEWS_ARGS)),
          checks={Check.DUPLICATE_SOURCE, Check.DUPLICATE_PROPOSAL}, findings={EVENT}, required={NEWS}, max_calls=2),
    _case("E7", "Private communications", "What was said in Google's private communications?", ResearchType.OTHER, (),
          {StopReason.UNSUPPORTED_QUESTION}, QualityStatus.UNSUPPORTED,
          (_plan(arguments={}, gap="No authorized private records."),), max_calls=0),
    _case("E8", "Provider failure preserves price evidence", PRICE_QUESTION, ResearchType.CAUSAL_PRICE_MOVE, (PRICE, NEWS),
          {StopReason.TOOL_FAILURE}, PARTIAL, (_plan("market_data"), _plan("news", NEWS_ARGS)),
          checks={Check.PRICE_FIRST, Check.FAILED_WITHOUT_EVIDENCE}, findings={MARKET}, required={PRICE, NEWS},
          completeness={Completeness.FAILED}, failed_news=True,
          limits=ResearchLimits(max_consecutive_failures=1, max_llm_requests=7, max_steps=4, max_tool_calls=3)),
    _case("E9", "Timing versus causation", f"Why did GOOGL (Google) decline from {SPAN}?", ResearchType.CAUSAL_PRICE_MOVE, (PRICE, NEWS),
          {UNRESOLVED}, PARTIAL, (_plan("market_data"), _plan("news", NEWS_ARGS), _plan(arguments={})),
          checks={Check.PRICE_FIRST, Check.QUALITY_VETO}, findings={MARKET, EVENT, ClaimKind.TEMPORAL_ASSOCIATION},
          required={PRICE, NEWS}),
    _case("E10", "Corrected inappropriate raw action", PRICE_QUESTION, ResearchType.CAUSAL_PRICE_MOVE, (PRICE, NEWS),
          {UNRESOLVED}, PARTIAL, (_plan("news"), _plan("market_data"), _plan(arguments={}, gap="Missing causal evidence.")),
          checks={Check.PRICE_FIRST, Check.CORRECTED_ACTION}, findings={MARKET}, required={PRICE}, max_calls=1),
    _case("E11", "Scoped DOJ titles and coverage", f"What DOJ press-release titles mention Google from 2024-01-01 to {END}?",
          ResearchType.REGULATORY, (NEWS,), {SATISFIED}, SUFFICIENT,
          (_plan("news", {"query": "Google", **REGULATORY_RANGE}), _plan(arguments={})),
          checks={Check.SOURCE_LIMITATIONS}, findings={EVENT}, required={NEWS}, period=REGULATORY_RANGE,
          live_source=NEWS),
)
LOCAL_CASE_IDS = ("E1", "E2", "E3", "E11")


class ScriptedProvider:
    def __init__(self, case: EvalCase):
        if case.offline_proposals_json is None:
            raise ValueError("quality-only cases do not need a model")
        self.responses = iter(json.loads(case.offline_proposals_json))

    def generate(self, prompt, schema):
        item = next(self.responses, None)
        performance = LLMPerformance(
            model="scripted", request_duration_seconds=0, total_duration_seconds=0, attempts=1,
            validation_passed=True,
        )
        try:
            decision = schema.model_validate(item)
        except ValidationError:
            raise LLMError(
                "Offline response did not validate.", performance.model_copy(update={
                    "error_category": LLMErrorCategory.SCHEMA_VALIDATION, "validation_passed": False,
                }), proposal=ProposalMetadata.from_payload(item),
            ) from None
        return StructuredLLMResult[schema](decision=decision, performance=performance)


class _Prices(FixturePrices):
    def __init__(self, ending_price: float, source: str = "prices"):
        super().__init__(ending_price=ending_price)
        self.source = source

    def get_price_history(self, request):
        history = super().get_price_history(request)
        return history.model_copy(update={
            "source_url": f"https://example.invalid/{self.source}", "retrieved_at": RETRIEVED,
        })


class _News(FixtureNews):
    def __init__(self, case: EvalCase):
        self.case = case

    def search_news(self, request):
        if self.case.failed_news:
            raise HttpFailure("PROVIDER_UNAVAILABLE", "Simulated provider unavailable", True, 0)
        result = super().search_news(request)
        return result.model_copy(update={
            "retrieved_at": RETRIEVED, "articles": [] if self.case.empty_news else [
                item.model_copy(update={"retrieved_at": RETRIEVED, "publication_date": END}) for item in result.articles
            ],
        })


class _Filings(FixtureFilings):
    def get_company_filings(self, request):
        result = super().get_company_filings(request)
        return result.model_copy(update={
            "retrieved_at": RETRIEVED, "filings": [
                item.model_copy(update={"retrieved_at": RETRIEVED}) for item in result.filings
            ],
        })

    def get_document_excerpt(self, filing):
        return super().get_document_excerpt(filing).model_copy(update={"retrieved_at": RETRIEVED})


def registry(case: EvalCase, *, live_sources: bool = False) -> ToolRegistry:
    live = case.live_source if live_sources else None
    return ToolRegistry(
        market_data=MarketDataTool(YahooFinanceProvider() if live is PRICE else _Prices(case.ending_price))
        if PRICE in case.tools else None,
        news=NewsTool(DoJPressReleaseProvider() if live is NEWS else _News(case)) if NEWS in case.tools else None,
        sec_filings=SecFilingsTool(_Filings()) if SEC in case.tools else None,
        provenance={tool: FIXTURE for tool in case.tools if tool != live},
    )


def conflicting_state(case: EvalCase) -> ResearchState:
    """Replay two separately captured equivalent requests through M1 state operations."""
    state = ResearchState(goal=ResearchGoal(
        query=case.question, objective=case.question, research_type=ResearchType.PRICE_COMPARISON,
        ticker="GOOGL", start_date=START, end_date=END,
    ))
    for number, close in enumerate((94, 106), start=1):
        action = ResearchAction(
            action_id=f"capture-{number}", tool_name=PRICE, arguments=PRICE_ARGS, purpose="Compare captured prices.",
        )
        result = MarketDataTool(_Prices(close, f"prices-{number}")).get_history(
            action_id=action.action_id, ticker="GOOGL", start_date=START, end_date=END,
        )
        state = observe_result(add_action(state, action), action, result, FIXTURE)
    return state
