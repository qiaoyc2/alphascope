"""Assess the evidence in an existing research state without executing tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from financial_research_agent.agent.observe import SourceFact, describe_result
from financial_research_agent.agent.premise import (
    MovementReading, assess_premise, comparable_conflicts, market_reading, scoped_readings,
)
from financial_research_agent.models import (
    Evidence, Observation, ObservationOutcome, ObservationProvenance, ResearchAction, ResearchModel,
    ResearchState, ResearchType, ToolResult,
)
from financial_research_agent.providers.market_data import PriceBasis
from financial_research_agent.providers.sec_edgar import SecExcerpt
from financial_research_agent.tools import ToolName
from financial_research_agent.tools.calculator import calculate_values
from financial_research_agent.tools.planning import validate_tool_arguments


class QualityStatus(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    CONFLICTED = "conflicted"
    UNSUPPORTED = "unsupported"


class Disposition(StrEnum):
    COMPLETE = "complete"
    CONTINUE = "continue"
    CLARIFY = "clarify"
    STOP_UNRESOLVED = "stop_unresolved"


class SupportStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    NOT_EVALUATED = "not_evaluated"


class Relevance(StrEnum):
    RELEVANT = "relevant"
    UNRELATED = "unrelated"
    UNKNOWN = "unknown"


class Completeness(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    TRUNCATED = "truncated"
    EMPTY = "empty"
    FAILED = "failed"
    UNVALIDATED = "unvalidated"


class ClaimKind(StrEnum):
    MARKET_RETURN = "market_return"
    EVENT_RECORD = "event_record"
    FILING_RECORD = "filing_record"
    CALCULATION = "calculation"
    TEMPORAL_ASSOCIATION = "temporal_association"
    HYPOTHESIS = "hypothesis"
    CAUSAL_CONCLUSION = "causal_conclusion"


class ResearchFinding(ResearchModel):
    finding_id: str
    statement: str
    support_status: SupportStatus
    kind: ClaimKind
    evidence_ids: list[str] = Field(default_factory=list)
    observation_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class EvidenceCheck(ResearchModel):
    evidence_id: str
    valid: bool
    relevance: Relevance = Relevance.UNKNOWN
    duplicate_of: str | None = None
    issue: str | None = None


class SourceAssessment(ResearchModel):
    observation_id: str
    result_id: str
    tool: ToolName | None
    provenance: ObservationProvenance
    completeness: Completeness
    relevance: Relevance
    limitations: list[str] = Field(default_factory=list)
    truncated: bool | None = None
    truncation_reason: str | None = None
    duplicate_of: str | None = None


class EvidenceConflict(ResearchModel):
    observation_ids: tuple[str, str]
    reason: str


class QualityAssessment(ResearchModel):
    status: QualityStatus
    goal_satisfied: bool
    recommended_disposition: Disposition
    supported_findings: list[ResearchFinding]
    unverified_claims: list[ResearchFinding]
    evidence_checks: list[EvidenceCheck]
    sources: list[SourceAssessment]
    contradictions: list[EvidenceConflict]
    gaps: list[str]
    remaining_needs: list[str]
    reasons: list[str]
    premise_contradicted: bool = False

    @model_validator(mode="after")
    def check_satisfaction(self) -> Self:
        if self.goal_satisfied != (self.status is QualityStatus.SUFFICIENT) or (
            self.goal_satisfied != (self.recommended_disposition is Disposition.COMPLETE)
        ):
            raise ValueError("sufficient quality, satisfaction and completion must agree")
        if self.goal_satisfied and (not self.supported_findings or self.contradictions or self.premise_contradicted):
            raise ValueError("a satisfied goal requires support and no material contradiction")
        return self


@dataclass
class _Retrieved:
    result: ToolResult
    assessment: SourceAssessment
    facts: tuple[SourceFact, ...] = ()
    movement: MovementReading | None = None
    calculation: str | None = None


_PRIVATE = re.compile(
    r"\b(?:private|confidential|internal|nonpublic|non-public)\s+"
    r"(?:communications?|conversations?|calls?|emails?|messages?|chats?|transcripts?|meetings?)\b"
    r"|\b(?:said|discussed|told|wrote)\b.{0,60}\bprivately\b", re.I,
)
_RETURN = re.compile(r"\b(?:returns?|performance|outperform\w*|underperform\w*|rose|fell|rise|declin\w*|gain\w*|drop\w*|change\w*|up|down)\b|%", re.I)
_CAUSAL = re.compile(r"\b(?:why|cause[ds]?|reasons?|explain|drive|drove|due to)\b", re.I)
_COMPARATIVE = re.compile(r"\b(?:against|versus|vs\.?|relative to|compared (?:with|to)|outperform\w*|underperform\w*)\b", re.I)
_RAW_PRICE = re.compile(r"\b(?:unadjusted|raw clos(?:e|ing))\b", re.I)
_ADJUSTED_PRICE = re.compile(r"\badjusted\b", re.I)
_FORM = re.compile(r"\b(?:10-[KQ]|8-K|20-F|6-K)(?:/A)?\b", re.I)
_COMPOUND = re.compile(r"\b(?:and|also)\s+(?:what|which|when|how|why|whether|explain|identify|describe)\b", re.I)


def _mentions(text: str, terms: set[str]) -> bool:
    return any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I) for term in terms)


def _retrieve(
    state: ResearchState, observation: Observation, result: ToolResult, action: ResearchAction,
    terms: set[str],
) -> _Retrieved:
    try:
        tool = ToolName(action.tool_name)
    except ValueError:
        tool = None
    assess = SourceAssessment(
        observation_id=observation.observation_id, result_id=result.result_id, tool=tool,
        provenance=observation.provenance, completeness=Completeness.UNVALIDATED, relevance=Relevance.UNKNOWN,
    )
    item = _Retrieved(result, assess)
    if tool is None:
        assess.limitations.append("The action does not reference a registered tool.")
        return item
    if not result.success:
        assess.completeness = Completeness.FAILED
        assess.limitations.append("The retrieval failed. It provides no evidence about the requested facts.")
        return item
    if observation.provenance is ObservationProvenance.USER_INPUT or (
        (observation.provenance is ObservationProvenance.DERIVED) != (tool is ToolName.CALCULATOR)
    ):
        assess.limitations.append("The observation lacks compatible retrieval or calculation provenance.")
        return item
    try:
        args = validate_tool_arguments(tool, action.arguments, state)
        if tool is ToolName.CALCULATOR:
            if calculate_values(args) != result.data:
                raise ValueError("calculation differs from replay")
        description = describe_result(result, observation.provenance)
        if observation.outcome is not None and observation.outcome != description.outcome:
            raise ValueError("inconsistent observation outcome")
        kind = result.data.get("kind")
        urls = {source.url for source in result.sources}
        if tool is ToolName.MARKET_DATA:
            benchmark = "benchmark" in args
            if kind != ("benchmark_comparison" if benchmark else "price_history"):
                raise ValueError("unexpected market result kind")
            key = "stock_ticker" if benchmark else "ticker"
            if result.data.get("stock_source_url" if benchmark else "source_url") not in {str(url) for url in urls}:
                raise ValueError("price history lacks its source reference")
            if any(result.data.get(field) != expected for field, expected in (
                (key, args["ticker"]), ("requested_start_date", args["start_date"]),
                ("requested_end_date", args["end_date"]),
            )):
                raise ValueError("market result scope differs from the action")
            if benchmark and (result.data.get("benchmark_ticker") != args["benchmark"]
                              or result.data.get("price_basis") != args["price_basis"]
                              or result.data.get("benchmark_source_url") not in {str(url) for url in urls}):
                raise ValueError("benchmark scope or provenance differs")
            item.movement = market_reading(observation, result)
            goal = state.goal
            if goal.ticker and goal.ticker.upper() != args["ticker"]:
                assess.relevance = Relevance.UNRELATED
            elif goal.ticker and all(
                value is None or args[field] == value.isoformat()
                for field, value in (("start_date", goal.start_date), ("end_date", goal.end_date))
            ):
                assess.relevance = Relevance.RELEVANT
            assess.limitations.append("Returns use available endpoint sessions within the requested interval.")
        elif tool is ToolName.NEWS:
            if kind != "news_search" or result.data.get("query") != args["query"]:
                raise ValueError("news result differs from the action")
            assess.limitations.append("DOJ title metadata covers inspected results within the search scope.")
            if result.data["invalid_items_skipped"] or len(result.data["articles"]) >= args["limit"]:
                assess.limitations.append("Invalid items or a result cap may limit search coverage.")
            if any(_event_in_scope(record, state, terms) for record in result.data["articles"]):
                assess.relevance = Relevance.RELEVANT
        elif tool is ToolName.SEC_FILINGS:
            if "filing" in args:
                if kind != "sec_document_excerpt":
                    raise ValueError("expected a filing excerpt")
                excerpt = SecExcerpt.model_validate({key: value for key, value in result.data.items() if key != "kind"})
                filing = args["filing"]
                if (excerpt.accession_number != filing["accession_number"] or excerpt.cik != filing["cik"]
                    or str(excerpt.document_url) != filing["primary_document_url"] or excerpt.document_url not in urls):
                    raise ValueError("excerpt source differs from the registered filing")
                assess.truncated = excerpt.truncated
                assess.truncation_reason = excerpt.truncation_reason
                assess.limitations.append("A bounded excerpt cannot establish absence from the full filing.")
                if _mentions(excerpt.company_name, terms) or (state.goal.ticker or "").upper() in {
                    ticker.upper() for ticker in _filing_tickers(state, excerpt.accession_number)
                }:
                    assess.relevance = Relevance.RELEVANT
            else:
                if kind != "sec_filings":
                    raise ValueError("expected filing metadata")
                company = result.data["company"]
                if len(result.data["filings"]) > args["limit"] or any(
                    record["cik"] != company["cik"] or record["form"] not in args["forms"]
                    or (args["start_date"] and record["filing_date"] < args["start_date"])
                    or (args["end_date"] and record["filing_date"] > args["end_date"])
                    for record in result.data["filings"]
                ):
                    raise ValueError("filing metadata differs from the lookup scope")
                if (state.goal.ticker or "").upper() in company["tickers"] or _mentions(company["company_name"], terms):
                    assess.relevance = Relevance.RELEVANT
                elif state.goal.ticker:
                    assess.relevance = Relevance.UNRELATED
                assess.limitations.append("Recent SEC submissions return filing metadata within the requested filters.")
        else:
            item.calculation = description.summary
            assess.limitations.append("Replayed arithmetic validates the output from supplied operands only.")

        if any(fact.url not in {str(url) for url in urls} for fact in description.facts):
            raise ValueError("a result fact has no source reference")
        item.facts = description.facts
        assess.completeness = {
            ObservationOutcome.DATA: Completeness.COMPLETE,
            ObservationOutcome.EMPTY: Completeness.EMPTY,
            ObservationOutcome.INCOMPLETE: Completeness.PARTIAL,
        }[description.outcome]
        if assess.truncated:
            assess.completeness = Completeness.TRUNCATED
        elif tool is ToolName.NEWS and assess.completeness is Completeness.COMPLETE and (
            result.data["invalid_items_skipped"] or len(result.data["articles"]) >= args["limit"]
        ):
            assess.completeness = Completeness.PARTIAL
        if assess.completeness is Completeness.EMPTY:
            assess.limitations.append("An empty result applies only to this retrieval and its coverage.")
        if tool in {ToolName.NEWS, ToolName.SEC_FILINGS}:
            assess.limitations.extend(result.data.get("limitations", []))
    except (ValueError, TypeError, KeyError):
        assess.completeness = Completeness.UNVALIDATED
        assess.limitations.append("The observation does not match the expected tool result, action, or sources.")
        item.facts, item.movement, item.calculation = (), None, None
    return item


def _filing_tickers(state: ResearchState, accession: str) -> set[str]:
    return {
        ticker
        for result in state.tool_results
        if result.success and result.tool_name == ToolName.SEC_FILINGS and result.data.get("kind") == "sec_filings"
        and any(record.get("accession_number") == accession for record in result.data.get("filings", []))
        for ticker in result.data.get("company", {}).get("tickers", [])
    }


def _event_in_scope(record: dict, state: ResearchState, terms: set[str]) -> bool:
    if not _mentions(record["title"], terms):
        return False
    published = record.get("publication_date")
    if published is None:
        return state.goal.start_date is None and state.goal.end_date is None
    day = date.fromisoformat(published)
    return (state.goal.start_date is None or state.goal.start_date <= day) and (
        state.goal.end_date is None or day <= state.goal.end_date
    )


def _assess_evidence(
    item: Evidence, retrievals: dict[str, _Retrieved], state: ResearchState, terms: set[str],
) -> EvidenceCheck:
    check = EvidenceCheck(evidence_id=item.evidence_id, valid=False)
    for oid in item.observation_ids:
        retrieved = retrievals[oid]
        source, result = retrieved.assessment, retrieved.result
        if item.source_url is None:
            if (item.evidence_type != "calculation" or not retrieved.calculation or item.claim != retrieved.calculation
                or item.calculation_method != result.data["operation"]
                or item.calculation_inputs != result.data["inputs"]):
                check.issue = "The evidence does not match the replayed calculation."
                return check
            continue
        matching = [
            fact for fact in retrieved.facts
            if fact.url == str(item.source_url) and fact.claim == item.claim
            and item.retrieved_at in {None, fact.retrieved_at or result.recorded_at}
            and item.event_at in {None, fact.event_at}
        ]
        if not matching or item.evidence_type != f"{source.tool}_source" or item.calculation_inputs:
            check.issue = "The claim or source metadata does not match a fact from the linked observation."
            return check
        relevance = source.relevance
        if source.tool is ToolName.NEWS and not any(
            record["url"] == str(item.source_url) and _event_in_scope(record, state, terms)
            for record in result.data["articles"]
        ):
            relevance = Relevance.UNKNOWN
        if check.relevance is not Relevance.RELEVANT:
            check.relevance = relevance
    check.valid = True
    return check


def evaluate_quality(state: ResearchState, *, available_tools: tuple[ToolName, ...] | None = None) -> QualityAssessment:
    """Evaluate defensible claims and recommend a disposition without authorizing execution."""
    state = ResearchState.model_validate(state)
    goal = state.goal
    query = goal.query
    terms = {name.casefold() for name in (goal.company, goal.ticker) if name and _mentions(query, {name})}
    results = {item.result_id: item for item in state.tool_results}
    actions = {item.action_id: item for item in state.completed_actions}
    retrievals = {
        item.observation_id: _retrieve(state, item, results[item.result_id], actions[item.action_id], terms)
        for item in state.observations
    }
    seen_results: dict[tuple, str] = {}
    for oid, retrieved in retrievals.items():
        if not retrieved.facts:
            continue
        movement = retrieved.movement
        key = (
            retrieved.assessment.tool, tuple(sorted((fact.url, fact.claim) for fact in retrieved.facts)),
            (movement.scope, movement.measurement.return_pct, movement.measurement.benchmark_return_pct) if movement else None,
        )
        retrieved.assessment.duplicate_of = seen_results.get(key)
        seen_results.setdefault(key, oid)
    checks = []
    distinct: dict[tuple, str] = {}
    for item in state.evidence:
        check = _assess_evidence(item, retrievals, state, terms)
        if check.valid:
            key = (str(item.source_url), item.claim)
            if item.source_url is None:
                key = ("calculation", tuple(retrievals[oid].calculation for oid in item.observation_ids))
            # Scope is material for market results, even when the rendered values round alike.
            scopes = frozenset(
                (reading.scope, reading.measurement.return_pct, reading.measurement.benchmark_return_pct)
                for oid in item.observation_ids if (reading := retrievals[oid].movement)
            )
            check.duplicate_of = distinct.setdefault((*key, scopes), item.evidence_id)
            if check.duplicate_of == item.evidence_id:
                check.duplicate_of = None
        checks.append(check)
    evidence = {item.evidence_id: item for item in state.evidence}
    usable = [check for check in checks if check.valid and not check.duplicate_of]
    evidenced = {oid for check in checks if check.valid for oid in evidence[check.evidence_id].observation_ids}
    readings = [
        reading for retrieved in retrievals.values()
        if (reading := retrieved.movement) and reading.observation.observation_id in evidenced
    ]
    scoped = scoped_readings(state, readings)
    pairs = comparable_conflicts(scoped)
    contradictions = [EvidenceConflict(
        observation_ids=(a.observation.observation_id, b.observation.observation_id),
        reason="Comparable endpoint returns disagree for the same ticker, period, basis and benchmark convention.",
    ) for a, b in pairs]
    disputed = {oid for conflict in contradictions for oid in conflict.observation_ids}
    premise = assess_premise(state, readings=readings)
    findings = []
    events = []
    for check in usable:
        item = evidence[check.evidence_id]
        source = retrievals[item.observation_ids[0]].assessment
        kind = {
            ToolName.MARKET_DATA: ClaimKind.MARKET_RETURN, ToolName.NEWS: ClaimKind.EVENT_RECORD,
            ToolName.SEC_FILINGS: ClaimKind.FILING_RECORD, ToolName.CALCULATOR: ClaimKind.CALCULATION,
        }[source.tool]
        support = SupportStatus.NOT_EVALUATED
        if disputed.intersection(item.observation_ids):
            support = SupportStatus.CONTRADICTED
        elif check.relevance is Relevance.RELEVANT or kind is ClaimKind.CALCULATION:
            support = SupportStatus.SUPPORTED
        finding = ResearchFinding(
            finding_id=f"finding-{item.evidence_id}", statement=item.claim, kind=kind,
            support_status=support, evidence_ids=[item.evidence_id], observation_ids=item.observation_ids,
            limitations=source.limitations,
        )
        findings.append(finding)
        if kind is ClaimKind.EVENT_RECORD and support is SupportStatus.SUPPORTED:
            retrieved = retrievals[item.observation_ids[0]]
            events.extend((item, date.fromisoformat(record["publication_date"]))
                          for record in retrieved.result.data["articles"]
                          if record["url"] == str(item.source_url) and record["publication_date"])

    causal = goal.research_type is ResearchType.CAUSAL_PRICE_MOVE or (
        goal.research_type in {ResearchType.PRICE_COMPARISON, ResearchType.REGULATORY, ResearchType.FILINGS}
        and bool(_CAUSAL.search(query))
    )
    if causal and premise.conditional_valid is True and scoped and scoped[0].effective_dates:
        first, last = scoped[0].effective_dates
        price = next((item for item in findings if item.kind is ClaimKind.MARKET_RETURN
                      and scoped[0].observation.observation_id in item.observation_ids), None)
        for event, day in events:
            if price and first <= day <= last:
                findings.append(ResearchFinding(
                    finding_id=f"timing-{event.evidence_id}", kind=ClaimKind.TEMPORAL_ASSOCIATION,
                    statement=f"The retrieved DOJ release is dated {day}, within the measured price interval.",
                    support_status=SupportStatus.SUPPORTED,
                    evidence_ids=[*price.evidence_ids, event.evidence_id],
                    observation_ids=list(dict.fromkeys([*price.observation_ids, *event.observation_ids])),
                    limitations=["Timing alone does not establish a causal link."],
                ))
    if causal:
        findings.append(ResearchFinding(
            finding_id="causal-conclusion", kind=ClaimKind.CAUSAL_CONCLUSION,
            statement="A causal explanation for the movement is unverified.", support_status=SupportStatus.UNSUPPORTED,
        ))
    usable_ids = {check.evidence_id for check in checks if check.valid}
    findings.extend(ResearchFinding(
        finding_id=hypothesis.hypothesis_id, kind=ClaimKind.HYPOTHESIS, statement=hypothesis.statement,
        support_status=SupportStatus.NOT_EVALUATED,
        evidence_ids=[eid for eid in hypothesis.supporting_evidence_ids if eid in usable_ids],
        limitations=["Planner hypothesis and proposed evidence links require causal review."],
    ) for hypothesis in state.hypotheses)
    supported = [item for item in findings if item.support_status is SupportStatus.SUPPORTED]
    status, disposition, needs = _outcome(
        state, available_tools, scoped, contradictions, premise.conditional_valid is False, causal, supported,
    )
    rejected = sum(not check.valid for check in checks)
    reasons = list(dict.fromkeys(needs or ["The scoped factual finding is backed by validated source records."]))
    if rejected:
        reasons.append(f"{rejected} evidence record(s) were excluded from support after validation.")
    return QualityAssessment(
        status=status, goal_satisfied=status is QualityStatus.SUFFICIENT, recommended_disposition=disposition,
        supported_findings=supported, unverified_claims=[item for item in findings if item not in supported],
        evidence_checks=checks, sources=[item.assessment for item in retrievals.values()],
        contradictions=contradictions, gaps=needs, remaining_needs=needs, reasons=reasons,
        premise_contradicted=premise.conditional_valid is False,
    )


def _outcome(
    state: ResearchState, tools: tuple[ToolName, ...] | None, readings: list[MovementReading],
    contradictions: list[EvidenceConflict], premise_contradicted: bool, causal: bool,
    supported: list[ResearchFinding],
) -> tuple[QualityStatus, Disposition, list[str]]:
    goal = state.goal
    query = goal.query
    if goal.needs_clarification:
        return QualityStatus.INSUFFICIENT, Disposition.CLARIFY, goal.unresolved_details or ["Clarify the intended scope."]
    if contradictions:
        return QualityStatus.CONFLICTED, Disposition.CLARIFY, ["Reconcile conflicting price observations."]
    if premise_contradicted:
        return QualityStatus.PARTIAL, Disposition.CLARIFY, ["Confirm the intended price premise or period."]
    if _PRIVATE.search(query):
        return QualityStatus.UNSUPPORTED, Disposition.STOP_UNRESOLVED, ["Private communications require an authorized source or supplied record."]

    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query)
    dates_match = not dates or (
        len(dates) == 2 and goal.start_date and goal.end_date
        and dates == [goal.start_date.isoformat(), goal.end_date.isoformat()]
    )
    simple_request = query.count("?") <= 1 and not _COMPOUND.search(query)
    actions_by_observation = {
        obs.observation_id: action.arguments
        for action in state.completed_actions for obs in state.observations if obs.action_id == action.action_id
    }
    scoped_findings = [
        finding for finding in supported if any(all(
            value is None or actions_by_observation[oid].get(field) == value.isoformat()
            for field, value in (("start_date", goal.start_date), ("end_date", goal.end_date))
        ) for oid in finding.observation_ids)
    ]
    if causal:
        needs = [] if readings else ["Verify the movement for the requested symbol and period."]
        needs.append("Establish a causal link and assess competing explanations.")
        preferred = {ToolName.NEWS, ToolName.SEC_FILINGS} if readings else {ToolName.MARKET_DATA}
    elif goal.research_type is ResearchType.PRICE_COMPARISON:
        preferred = {ToolName.MARKET_DATA}
        comparative = bool(_COMPARATIVE.search(query))
        basis = PriceBasis.RAW_CLOSE if _RAW_PRICE.search(query) else (
            PriceBasis.ADJUSTED_CLOSE if _ADJUSTED_PRICE.search(query) else None
        )
        if (readings and len({item.scope for item in readings}) == 1 and dates_match and simple_request and _RETURN.search(query)
            and goal.ticker and _mentions(query, {goal.ticker})
            and (basis is None or readings[0].price_basis is basis)
            and (not comparative or (goal.benchmark and _mentions(query, {goal.benchmark})))):
            return QualityStatus.SUFFICIENT, Disposition.COMPLETE, []
        needs = ["Verify the requested ticker, period and calculation basis with matching price data."]
    elif goal.research_type is ResearchType.REGULATORY:
        preferred = {ToolName.NEWS}
        if (dates_match and simple_request and any(item.kind is ClaimKind.EVENT_RECORD for item in scoped_findings)
            and re.search(r"\b(?:DOJ|Department of Justice)\b", query, re.I)
            and re.search(r"\b(?:titles?|headlines?)\b", query, re.I)
            and not re.search(r"\b(?:all|every|complete|exhaustive)\b", query, re.I)):
            return QualityStatus.SUFFICIENT, Disposition.COMPLETE, []
        needs = ["Obtain in-scope agency evidence that answers the requested question. Title search has limited coverage."]
    elif goal.research_type is ResearchType.FILINGS:
        preferred = {ToolName.SEC_FILINGS}
        requested_forms = {form.upper() for form in _FORM.findall(query)}
        reported_forms = {
            form.upper()
            for item in scoped_findings if item.kind is ClaimKind.FILING_RECORD
            for form in _FORM.findall(item.statement)
        }
        if (dates_match and simple_request and reported_forms and requested_forms <= reported_forms
            and re.search(r"\b(?:which|list|when)\b", query, re.I)
            and re.search(r"\b(?:filings?|filed|forms?)\b", query, re.I)
            and not re.search(r"\b(?:all|every|complete|exhaustive|disclos\w+|contents?)\b", query, re.I)):
            return QualityStatus.SUFFICIENT, Disposition.COMPLETE, []
        needs = ["Verify the requested disclosure from the relevant filing section with adequate coverage."]
    else:
        preferred = set(tools or ())
        needs = ["Determine whether an approved source can verify the requested information."]
    if tools is not None and not preferred.intersection(tools):
        needs.append("Available tools do not cover the remaining research need.")
        return (QualityStatus.PARTIAL if supported else QualityStatus.INSUFFICIENT), Disposition.STOP_UNRESOLVED, needs
    return (QualityStatus.PARTIAL if supported else QualityStatus.INSUFFICIENT), Disposition.CONTINUE, needs
