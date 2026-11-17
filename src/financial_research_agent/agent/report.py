"""Build and render evidence-grounded reports from completed research runs."""

from __future__ import annotations

import re
from datetime import date, datetime
from html import escape
from typing import Literal, Self
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, FiniteFloat, HttpUrl, model_validator

from financial_research_agent.agent.quality import (
    ClaimKind, Completeness, QualityAssessment, QualityStatus, ResearchFinding, SourceAssessment,
    SupportStatus, evaluate_quality,
)
from financial_research_agent.agent.report_metrics import (
    CalculationMetric, MarketMetric, ReportMetric, market_metric, matched_calculation_source,
)
from financial_research_agent.agent.runner import ResearchRun, StopReason
from financial_research_agent.models import (
    Evidence, HypothesisStatus, ObservationProvenance, ResearchModel, ResearchStatus, ResearchType, utc_now,
)
from financial_research_agent.tools import ToolName


class CitationRecord(ResearchModel):
    evidence_ids: list[str] = Field(min_length=1)
    observation_id: str
    result_id: str
    retrieved_at: datetime | None = None
    published_at: datetime | None = None


class ReportCitation(ResearchModel):
    number: int = Field(ge=1)
    url: HttpUrl | None = None
    title: str | None = None
    provenance: ObservationProvenance
    observation_id: str
    result_id: str
    retrieved_at: datetime | None = None
    published_at: datetime | None = None
    records: list[CitationRecord] = Field(min_length=1)


class ReportFinding(ResearchFinding):
    citation_numbers: list[int] = Field(min_length=1)
    provenance: list[ObservationProvenance]
    independently_verified: bool


class ReportHypothesis(ResearchModel):
    hypothesis_id: str
    statement: str
    depends_on_contradicted_premise: bool
    evidence_needed: list[str]
    planner_status: HypothesisStatus
    support_status: Literal[SupportStatus.NOT_EVALUATED] = SupportStatus.NOT_EVALUATED
    supporting_observation_ids: list[str]
    contradicting_observation_ids: list[str]


class ReportCoverage(SourceAssessment):
    error_code: str | None = None


class NarrativeMetadata(ResearchModel):
    mode: Literal["deterministic", "llm_assisted", "fallback"] = "deterministic"
    validation_succeeded: bool | None = None
    fallback_used: bool = False
    failure_category: str | None = None
    prompt_bytes: int = Field(default=0, ge=0)
    model: str | None = None
    requests: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    elapsed_seconds: FiniteFloat = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_outcome(self) -> Self:
        if self.mode == "fallback":
            if not self.fallback_used or not self.failure_category or self.validation_succeeded is True:
                raise ValueError("a fallback must record why the draft was unavailable or rejected")
        elif self.fallback_used or self.failure_category is not None or (
            self.mode == "llm_assisted" and (self.validation_succeeded is not True or not self.requests)
        ) or (self.mode == "deterministic" and (self.validation_succeeded is not None or self.requests)):
            raise ValueError("narrative metadata is inconsistent with generation mode")
        return self


class ResearchReport(ResearchModel):
    schema_version: Literal["1.0"] = "1.0"
    report_id: UUID
    generated_at: datetime
    question: str
    ticker: str | None = None
    company: str | None = None
    benchmark: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    executive_summary: str
    executive_finding_id: str | None = None
    stop_reason: StopReason
    stop_explanation: str
    quality_status: QualityStatus
    research_status: ResearchStatus
    goal_satisfied: bool
    supported_findings: list[ReportFinding]
    disputed_findings: list[ReportFinding]
    hypotheses: list[ReportHypothesis]
    open_questions: list[str]
    coverage: list[ReportCoverage]
    citations: list[ReportCitation]
    metrics: list[ReportMetric]
    methodology: list[str]
    narrative: NarrativeMetadata = Field(default_factory=NarrativeMetadata)

    @model_validator(mode="after")
    def check_report_references(self) -> Self:
        citations = {item.number: item for item in self.citations}
        findings = {item.finding_id: item for item in [*self.supported_findings, *self.disputed_findings]}
        if len(citations) != len(self.citations) or len(findings) != len(self.supported_findings) + len(self.disputed_findings):
            raise ValueError("report identifiers must be unique")
        if len({item.hypothesis_id for item in self.hypotheses}) != len(self.hypotheses):
            raise ValueError("report hypotheses must be unique")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("report period is reversed")
        if self.goal_satisfied and (
            self.quality_status is not QualityStatus.SUFFICIENT or self.stop_reason is not StopReason.GOAL_SATISFIED
            or self.research_status is not ResearchStatus.COMPLETED or not self.supported_findings
        ):
            raise ValueError("report completion requires sufficient evidence and a completed run")
        if self.executive_finding_id and self.executive_finding_id not in {item.finding_id for item in self.supported_findings}:
            raise ValueError("executive finding requires a supported observation")
        if any(item.support_status is not SupportStatus.SUPPORTED for item in self.supported_findings) or any(
            item.support_status is not SupportStatus.CONTRADICTED for item in self.disputed_findings
        ):
            raise ValueError("finding status differs from its report section")
        for finding in findings.values():
            numbers = set(finding.citation_numbers)
            if numbers - citations.keys():
                raise ValueError("finding references an unknown citation")
            trace = [record for n in numbers for record in citations[n].records if record.observation_id in finding.observation_ids]
            if not finding.evidence_ids or set(finding.evidence_ids) - {eid for record in trace for eid in record.evidence_ids}:
                raise ValueError("finding references evidence absent from its source traces")
        for metric in self.metrics:
            finding = findings.get(metric.finding_id)
            if finding is None or metric.kind != finding.kind:
                raise ValueError("metric requires a matching finding")
            linked = {record.result_id for n in finding.citation_numbers for record in citations[n].records
                      if record.observation_id in finding.observation_ids}
            if metric.result_id not in linked:
                raise ValueError("metric result is absent from its source traces")
            if isinstance(metric, MarketMetric) and (
                self.start_date not in (None, metric.requested_start_date)
                or self.end_date not in (None, metric.requested_end_date)
                or self.ticker is not None and self.ticker.casefold() != metric.stock.ticker.casefold()
            ):
                raise ValueError("market metric is outside the original scope")
            if isinstance(metric, CalculationMetric) and set(metric.source_result_ids) - linked:
                raise ValueError("calculation references unmatched input sources")
        return self


_STOP_EXPLANATIONS = {
    StopReason.GOAL_SATISFIED: "The available evidence met the original research objective within the recorded scope.",
    StopReason.NEEDS_CLARIFICATION: "The run paused for clarification before continuing with the research.",
    StopReason.INSUFFICIENT_EVIDENCE: "The run ended with gaps in the evidence needed to answer the question.",
    StopReason.UNSUPPORTED_QUESTION: "The requested information requires a source outside the run's coverage.",
    StopReason.BUDGET_EXHAUSTED: "The run reached an execution limit and retained the findings recorded so far.",
    StopReason.REPEATED_ACTION: "Equivalent proposals were blocked and the run ended without further progress.",
    StopReason.INVALID_PROPOSAL: "Invalid proposals prevented the run from making further progress.",
    StopReason.TOOL_FAILURE: "A tool failure ended the run.",
    StopReason.LLM_FAILURE: "The model failed to produce usable structured output.",
    StopReason.EXECUTION_ERROR: "A state or execution error ended the run.",
    StopReason.INTERRUPTED: "The run was interrupted.",
}
_BUDGET_DETAILS = {
    "The planning step limit was reached.",
    "The tool call limit was reached.",
    "The LLM request limit was reached.",
    "The elapsed-time limit was reached.",
    "The elapsed-time limit was reached during a tool call.",
}
_CREDENTIAL_FIELDS = {
    "apikey", "accesstoken", "authtoken", "authorization", "clientsecret",
    "key", "password", "signature", "sig", "token", "xamzcredential", "xamzsignature",
}
_TOOL_LABELS = {
    ToolName.MARKET_DATA: "Market data",
    ToolName.NEWS: "DOJ title search",
    ToolName.SEC_FILINGS: "SEC filings",
    ToolName.CALCULATOR: "Calculator",
}


def _citation_url(url: HttpUrl | None) -> HttpUrl | None:
    if url is None or url.username is not None or url.password is not None:
        return None
    if any(re.sub(r"[^a-z0-9]", "", name.casefold()) in _CREDENTIAL_FIELDS for name, _ in url.query_params()):
        return None
    return url


def _summary(
    run: ResearchRun, quality: QualityAssessment, completed: bool, executive_finding: ReportFinding | None,
) -> str:
    if completed:
        if executive_finding:
            return executive_finding.statement
        if run.state.goal.research_type is ResearchType.REGULATORY:
            titles = sum(finding.kind is ClaimKind.EVENT_RECORD for finding in quality.supported_findings)
            if titles == 1:
                return "The search returned one supported DOJ title match within the recorded coverage."
            return f"The search returned {titles} supported DOJ title matches within the recorded coverage."
        return "The evidence supports the original factual question within the scope described below."
    if run.stop_reason is StopReason.GOAL_SATISFIED:
        return "The recorded completion could not be verified from the evidence in this run."
    if quality.status is QualityStatus.CONFLICTED:
        return "Comparable price observations disagree. The report retains the competing measurements for review."
    if quality.premise_contradicted:
        return "The recorded price movement contradicts the direction assumed in the question."
    if run.stop_reason is StopReason.NEEDS_CLARIFICATION:
        return "The intended research scope needs clarification."
    if quality.status is QualityStatus.UNSUPPORTED:
        return "The available sources cannot answer the original question."
    if any(finding.kind is ClaimKind.CAUSAL_CONCLUSION for finding in quality.unverified_claims):
        if any(finding.kind is ClaimKind.MARKET_RETURN for finding in quality.supported_findings):
            return "The recorded price movement is supported. Its cause remains unresolved."
        return "The available evidence leaves the price movement or its cause unresolved."
    if quality.supported_findings:
        return "The run produced supported findings and left the original question unresolved."
    return "The run produced no supported findings for the original question."


def build_research_report(run: ResearchRun) -> ResearchReport:
    """Recheck a full run using the existing quality gate and project reportable evidence."""
    run = ResearchRun.model_validate(run)
    state = run.state
    quality = evaluate_quality(state)
    observations = {item.observation_id: item for item in state.observations}
    results = {item.result_id: item for item in state.tool_results}
    valid = {check.evidence_id for check in quality.evidence_checks if check.valid}
    evidence = {item.evidence_id: item for item in state.evidence if item.evidence_id in valid}
    disputed_ids = {oid for conflict in quality.contradictions for oid in conflict.observation_ids}
    market_inputs = {
        observations[oid].result_id: (item, oid)
        for item in evidence.values() for oid in item.observation_ids
        if oid not in disputed_ids and results[observations[oid].result_id].tool_name == ToolName.MARKET_DATA
    }
    citations: dict[tuple[str, ObservationProvenance], ReportCitation] = {}
    metrics: list[ReportMetric] = []

    def cite(item: Evidence, observation_id: str) -> list[int]:
        observation = observations[observation_id]
        result = results[observation.result_id]
        urls = [item.source_url]
        if result.data.get("kind") == "benchmark_comparison":
            urls.extend(ref.url for ref in result.sources if str(ref.url) == result.data["benchmark_source_url"])
        numbers: list[int] = []
        for url in dict.fromkeys(urls):
            key = (str(url) if url is not None else result.result_id, observation.provenance)
            reference = next((ref for ref in result.sources if ref.url == url), None)
            retrieved = item.retrieved_at if url == item.source_url else result.data["benchmark_retrieved_at"]
            record = CitationRecord(
                evidence_ids=[item.evidence_id], observation_id=observation_id, result_id=result.result_id,
                retrieved_at=retrieved, published_at=reference.published_at if reference else None,
            )
            if key not in citations:
                safe_url = _citation_url(url)
                title = "Replayed calculation" if url is None else (reference.title if safe_url and reference else None)
                citations[key] = ReportCitation(
                    number=len(citations) + 1, url=safe_url, title=title, provenance=observation.provenance,
                    observation_id=record.observation_id, result_id=record.result_id,
                    retrieved_at=record.retrieved_at, published_at=record.published_at, records=[record],
                )
            elif record not in citations[key].records:
                citations[key].records.append(record)
            numbers.append(citations[key].number)
        return numbers

    def project(finding: ResearchFinding) -> ReportFinding:
        numbers: list[int] = []
        evidence_ids = list(finding.evidence_ids)
        observation_ids = list(finding.observation_ids)
        result_ids: list[str] = []
        for evidence_id in finding.evidence_ids:
            item = evidence[evidence_id]
            for oid in item.observation_ids:
                numbers.extend(cite(item, oid))
                result_ids.append(observations[oid].result_id)
        for result_id in dict.fromkeys(result_ids):
            result = results[result_id]
            if finding.kind is ClaimKind.MARKET_RETURN:
                metrics.append(market_metric(finding.finding_id, result))
            if finding.kind is ClaimKind.CALCULATION:
                inputs = result.data["inputs"]
                matched = []
                for source_id in inputs.get("source_result_ids", ()):
                    if source_id in market_inputs and matched_calculation_source(result, results[source_id]):
                        source, oid = market_inputs[source_id]
                        matched.append(source_id)
                        numbers.extend(cite(source, oid))
                        evidence_ids.append(source.evidence_id)
                        observation_ids.append(oid)
                metrics.append(CalculationMetric(
                    finding_id=finding.finding_id, result_id=result_id, operation=result.data["operation"],
                    unit=result.data["unit"], value=result.data.get("value", result.data.get("correlation")),
                    formula=result.data["formula"], source_result_ids=matched,
                ))
        linked_observations = [observations[oid] for oid in dict.fromkeys(observation_ids)]
        return ReportFinding(
            **finding.model_dump(exclude={"evidence_ids", "observation_ids"}),
            evidence_ids=list(dict.fromkeys(evidence_ids)), observation_ids=list(dict.fromkeys(observation_ids)),
            citation_numbers=list(dict.fromkeys(numbers)),
            provenance=list(dict.fromkeys(item.provenance for item in linked_observations)),
            independently_verified=bool(linked_observations) and all(
                item.independently_verified for item in linked_observations
            ),
        )

    supported = [project(finding) for finding in quality.supported_findings]
    disputed = [project(finding) for finding in quality.unverified_claims
                if finding.support_status is SupportStatus.CONTRADICTED]
    executive_finding = None
    if state.goal.research_type in {ResearchType.PRICE_COMPARISON, ResearchType.CAUSAL_PRICE_MOVE}:
        executive_finding = next((finding for finding in supported if finding.kind is ClaimKind.MARKET_RETURN), None)
    completed = (run.stop_reason is StopReason.GOAL_SATISFIED and state.status is ResearchStatus.COMPLETED
                 and quality.goal_satisfied)
    stop_explanation = _STOP_EXPLANATIONS[run.stop_reason]
    if run.stop_reason is StopReason.GOAL_SATISFIED and not completed:
        stop_explanation = "The recorded stop indicates completion, but the evidence does not verify it."
    elif completed and executive_finding and any(
        isinstance(metric, MarketMetric) and metric.finding_id == executive_finding.finding_id
        and metric.session_count == 2 for metric in metrics
    ):
        stop_explanation = "The recorded prices support the requested endpoint return."
    elif run.stop_reason is StopReason.BUDGET_EXHAUSTED and run.stop_detail in _BUDGET_DETAILS:
        stop_explanation = run.stop_detail
    questions = [*quality.remaining_needs]
    if quality.premise_contradicted:
        questions = ["Should research follow the observed movement or use a different period or premise?"]
    elif run.stop_reason is StopReason.NEEDS_CLARIFICATION and not questions:
        questions = ["What scope should the research use?"]
    # Preserve the runner's tool-availability finding when it was recorded. Source
    # coverage and factual support are independently rebuilt above.
    unavailable = "Available tools do not cover the remaining research need."
    if run.quality and unavailable in run.quality.remaining_needs and questions:
        questions.append(unavailable)
    hypotheses = []
    for item in state.hypotheses:
        inapplicable = quality.premise_contradicted and item.depends_on_price_premise
        hypotheses.append(ReportHypothesis(
            hypothesis_id=item.hypothesis_id, statement=item.statement,
            depends_on_contradicted_premise=inapplicable, evidence_needed=[] if inapplicable else item.research_gaps,
            planner_status=item.status,
            supporting_observation_ids=list(dict.fromkeys(
                oid for eid in item.supporting_evidence_ids if eid in evidence for oid in evidence[eid].observation_ids
            )),
            contradicting_observation_ids=list(dict.fromkeys(
                oid for eid in item.contradicting_evidence_ids if eid in evidence for oid in evidence[eid].observation_ids
            )),
        ))
    return ResearchReport(
        report_id=uuid5(NAMESPACE_URL, f"alphascope:{state.started_at.isoformat()}:{state.goal.query}"),
        generated_at=utc_now(),
        question=state.goal.query, ticker=state.goal.ticker, company=state.goal.company,
        benchmark=state.goal.benchmark, start_date=state.goal.start_date, end_date=state.goal.end_date,
        executive_summary=_summary(run, quality, completed, executive_finding),
        executive_finding_id=executive_finding.finding_id if executive_finding else None,
        stop_reason=run.stop_reason, stop_explanation=stop_explanation, quality_status=quality.status,
        research_status=state.status,
        goal_satisfied=completed, supported_findings=supported, disputed_findings=disputed,
        hypotheses=hypotheses,
        open_questions=list(dict.fromkeys(questions)),
        coverage=[ReportCoverage(
            **source.model_dump(),
            error_code=results[source.result_id].error.code if results[source.result_id].error else None,
        ) for source in quality.sources],
        citations=list(citations.values()),
        metrics=metrics,
        methodology=[
            "Findings were checked against their recorded observations and sources using the research quality gate.",
            *(
                ["Price returns use the recorded endpoint sessions. Benchmark comparisons use aligned sessions."]
                if any(isinstance(metric, MarketMetric) for metric in metrics) else []
            ),
            *(
                ["Calculator outputs were replayed from their recorded operands."]
                if any(isinstance(metric, CalculationMetric) for metric in metrics) else []
            ),
        ],
    )


def check_report_projection(report: ResearchReport, expected: ResearchReport) -> None:
    """Check that report presentation choices preserve the grounded projection."""
    presentation = {"generated_at", "narrative", "executive_finding_id", "supported_findings", "hypotheses", "open_questions"}
    if (
        report.model_dump(exclude=presentation) != expected.model_dump(exclude=presentation)
        or {item.finding_id: item for item in report.supported_findings}
        != {item.finding_id: item for item in expected.supported_findings}
        or {item.hypothesis_id: item for item in report.hypotheses} != {item.hypothesis_id: item for item in expected.hypotheses}
        or sorted(report.open_questions) != sorted(expected.open_questions)
    ):
        raise ValueError("report content differs from the grounded projection")
    if expected.executive_finding_id and report.executive_finding_id != expected.executive_finding_id:
        raise ValueError("report changed the required executive finding")
    if report.executive_finding_id and not expected.executive_finding_id:
        if not expected.goal_satisfied or not any(
            item.finding_id == report.executive_finding_id and item.kind in {ClaimKind.EVENT_RECORD, ClaimKind.FILING_RECORD}
            for item in expected.supported_findings
        ):
            raise ValueError("report executive finding is outside the factual objective")


def validate_research_report(report: ResearchReport, run: ResearchRun) -> None:
    """Verify a structured report against the original validated research run."""
    check_report_projection(ResearchReport.model_validate(report), build_research_report(run))


def _markdown_text(text: str) -> str:
    inline = escape(" ".join(text.split()), quote=False)
    return re.sub(r"([\\`*_{}\[\]()#!|~])", r"\\\1", inline)


def _public_citation_url(citation: ReportCitation) -> HttpUrl | None:
    url = _citation_url(citation.url)
    if url is None:
        return None
    host = (url.host or "").lower().rstrip(".")
    placeholders = ("example.com", "example.net", "example.org", "localhost")
    if host.endswith((".invalid", ".test", ".example")) or any(
        host == domain or host.endswith(f".{domain}") for domain in placeholders
    ):
        return None
    return url


def _source_link(url: HttpUrl | None, label: str) -> str:
    return f"[{label}](<{url}>)" if url else f"\\[{label}\\]"


def render_report_markdown(report: ResearchReport) -> str:
    """Render a report without model calls or raw tool output."""
    source_urls = {citation.number: _public_citation_url(citation) for citation in report.citations}
    lines = ["# AlphaScope research report", "", f"**Question** {_markdown_text(report.question)}"]
    entities = [value for value in (report.company, report.ticker) if value]
    period = " to ".join(day.isoformat() for day in (report.start_date, report.end_date) if day)
    if report.start_date and not report.end_date:
        period = f"From {period}"
    if report.end_date and not report.start_date:
        period = f"Through {period}"
    scope = [" / ".join(dict.fromkeys(entities)), f"Benchmark {report.benchmark}" if report.benchmark else ""]
    if any(scope):
        lines.extend(["", f"**Scope** {_markdown_text(', '.join(value for value in scope if value))}"])
    if period:
        lines.extend(["", f"**Requested interval** {_markdown_text(period)}"])
    lines.extend(["", f"**Evidence quality** {report.quality_status.value.capitalize()}"])
    lines.extend(["", "## Executive summary"])

    def finding_description(finding: ReportFinding) -> str:
        links = " ".join(_source_link(source_urls[n], str(n)) for n in finding.citation_numbers)
        label = ("Endpoint return" if finding.kind is ClaimKind.MARKET_RETURN
                 else finding.kind.value.replace("_", " ").capitalize())
        provenance = ", ".join(item.value.replace("_", " ") for item in finding.provenance)
        verification = "independently verified" if finding.independently_verified else "independent verification unavailable"
        statement = finding.statement
        if finding.kind is ClaimKind.CALCULATION:
            statement = statement.replace("percentage_points", "percentage points").replace("annualized_percent", "% annualized")
        return f"**{label}** {_markdown_text(statement)} {links} ({provenance}, {verification})"

    executive_finding = next(
        (finding for finding in report.supported_findings if finding.finding_id == report.executive_finding_id), None,
    )
    if not executive_finding or report.executive_summary != executive_finding.statement:
        lines.extend(["", _markdown_text(report.executive_summary)])
    if executive_finding:
        lines.extend(["", finding_description(executive_finding)])

    def findings_section(heading: str, findings: list[ReportFinding]) -> None:
        if not findings:
            return
        lines.extend(["", f"## {heading}", ""])
        lines.extend(f"- {finding_description(finding)}" for finding in findings)

    findings_section("Supported findings", [finding for finding in report.supported_findings if finding != executive_finding])
    findings_section("Disputed measurements", report.disputed_findings)
    benchmarks = [metric for metric in report.metrics if isinstance(metric, MarketMetric) and metric.benchmark]
    if benchmarks:
        lines.extend(["", "## Benchmark metrics", ""])
        disputed_ids = {item.finding_id for item in report.disputed_findings}
        for metric in benchmarks:
            assert metric.benchmark is not None and metric.relative_performance_percentage_points is not None
            status = "Disputed measurement, " if metric.finding_id in disputed_ids else ""
            start, end = metric.effective_start_date, metric.effective_end_date
            basis = metric.price_basis.value.replace("_", " ")
            lines.append(f"- {status}{metric.session_count} aligned sessions from {start} to {end} using {basis}.")
            for endpoint in (metric.stock, metric.benchmark):
                currency = f" {endpoint.currency}" if endpoint.currency else ""
                lines.append(
                    f"  - {_markdown_text(endpoint.ticker + currency)} {endpoint.start_price:g} to {endpoint.end_price:g}"
                )
    if report.hypotheses:
        lines.extend(["", "## Hypotheses", "", "These proposed explanations remain unverified.", ""])
        for hypothesis in report.hypotheses:
            qualifier = " (depends on the contradicted price premise)" if hypothesis.depends_on_contradicted_premise else ""
            lines.append(f"- {_markdown_text(hypothesis.statement)}{qualifier}")
            if hypothesis.planner_status is not HypothesisStatus.UNVERIFIED:
                lines.append(f"  - Planner assessment {hypothesis.planner_status.value}.")
            if hypothesis.evidence_needed:
                lines.append(f"  - Evidence to check — {_markdown_text(', '.join(hypothesis.evidence_needed))}")
    if report.open_questions:
        lines.extend(["", "## Open questions", ""])
        lines.extend(f"- {_markdown_text(question)}" for question in report.open_questions)

    coverage = [limitation for finding in [*report.supported_findings, *report.disputed_findings]
                for limitation in finding.limitations]
    for item in report.coverage:
        tool = _TOOL_LABELS.get(item.tool, "Data source")
        if item.completeness is Completeness.FAILED:
            coverage.append(f"{tool} failed ({item.error_code or 'unknown error'}).")
        elif item.completeness in {Completeness.EMPTY, Completeness.TRUNCATED, Completeness.PARTIAL, Completeness.UNVALIDATED}:
            status = item.completeness.value
            detail = f" ({item.truncation_reason})" if item.truncation_reason else ""
            coverage.append(f"{tool} result was {status}{detail}.")
        coverage.extend(item.limitations)
    if coverage:
        lines.extend(["", "## Source coverage", ""])
        lines.extend(f"- {_markdown_text(item)}" for item in dict.fromkeys(coverage))
    outcome = report.stop_reason.value.replace("_", " ").capitalize()
    if report.stop_reason is StopReason.GOAL_SATISFIED and not report.goal_satisfied:
        outcome = "Completion unverified"
    lines.extend([
        "", "## Why research stopped", "",
        f"**{outcome}**. {_markdown_text(report.stop_explanation)}",
    ])
    if report.narrative.fallback_used:
        lines.extend([
            "",
            ("The narrative draft failed validation." if report.narrative.validation_succeeded is False
             else "Narrative generation was unavailable.") + " This report uses the validated research records.",
        ])
    if report.citations:
        lines.extend(["", "## Sources", ""])
        for citation in report.citations:
            url = source_urls[citation.number]
            title = citation.title or ("Source record" if url else "Source reference without a public link")
            retrieved = f", retrieved {citation.retrieved_at.date().isoformat()}" if citation.retrieved_at else ""
            label = _source_link(url, _markdown_text(title)) if url else _markdown_text(title)
            if url is None and citation.provenance in {
                ObservationProvenance.SIMULATED_FIXTURE, ObservationProvenance.HISTORICAL_FIXTURE,
            }:
                provenance = citation.provenance.value.replace("_", " ").capitalize()
                evidence_ids = dict.fromkeys(eid for record in citation.records for eid in record.evidence_ids)
                label = f"{provenance} — {label} (evidence {_markdown_text(', '.join(evidence_ids))})"
            lines.append(f"{citation.number}. {label}{retrieved}")
    return "\n".join(lines) + "\n"
