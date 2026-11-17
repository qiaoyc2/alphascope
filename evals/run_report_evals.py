"""Evaluate M6 reports over frozen research fixtures and optional loopback drafting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import monotonic

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.cases import CASES, EvalCase, ScriptedProvider, conflicting_state, registry
from financial_research_agent.agent.quality import ClaimKind, evaluate_quality
from financial_research_agent.agent.report import ResearchReport, validate_research_report
from financial_research_agent.agent.report_export import export_research_report
from financial_research_agent.agent.report_metrics import MarketMetric
from financial_research_agent.agent.report_narrative import generate_research_report
from financial_research_agent.agent.runner import ResearchRun, ResearchRunner, RunStatistics, StopReason
from financial_research_agent.agent.state import finish_research
from financial_research_agent.config import load_llm_settings, open_llm_provider
from financial_research_agent.models import ResearchStatus
from financial_research_agent.providers.llm import (
    LLMError, LLMErrorCategory, LLMPerformance, LLMProvider, StructuredLLMResult,
)


CASES_BY_ID = {case.case_id: case for case in CASES if case.case_id != "E6"}
LOCAL_CASES = ("E3", "E1", "E11", "E2")
EXPECTED_RETURNS = {"E1": [3.0], "E2": [-6.0], "E3": [6.0], "E4": [-6.0, 6.0],
                    "E8": [-6.0], "E9": [-6.0], "E10": [-6.0]}


class ScriptedDraft:
    def __init__(self, fault: str | None = None):
        self.fault = fault

    def generate(self, prompt, schema):
        context = json.loads(prompt.rsplit("\n", 1)[-1])
        performance = LLMPerformance(
            model="scripted", request_duration_seconds=0, total_duration_seconds=0, attempts=1, validation_passed=True,
        )
        if self.fault == "connection":
            raise LLMError("Simulated unavailability.", performance.model_copy(update={
                "validation_passed": False, "error_category": LLMErrorCategory.CONNECTION,
            }))
        payload = {
            "lead_finding": context["required_lead"] or next(iter(context["optional_leads"]), None),
            "key_findings": list(context["findings"])[:4],
            "hypotheses": list(context["unverified_hypotheses"]),
            "evidence_gaps": list(context["evidence_gaps"]),
        }
        if self.fault == "reference":
            payload["key_findings"] = ["f6"]
        if self.fault == "schema":
            payload["conclusion"] = {"return_pct": 900, "source": "https://example.invalid/fabricated"}
        return StructuredLLMResult[schema].model_construct(decision=payload, performance=performance)


def replay_research(case: EvalCase) -> ResearchRun:
    if not case.quality_only:
        return ResearchRunner(ScriptedProvider(case), registry(case)).run(case.question, case.limits, today=case.today)
    state = finish_research(conflicting_state(case), ResearchStatus.STOPPED, StopReason.NEEDS_CLARIFICATION.value)
    return ResearchRun(
        state=state, stop_reason=StopReason.NEEDS_CLARIFICATION, stop_detail="Comparable observations conflict.",
        quality=evaluate_quality(state, available_tools=case.tools), steps=[], issues=[],
        statistics=RunStatistics(
            planning_steps=0, tool_calls=state.tool_call_count, provider_calls_reported=0, llm_requests=0,
            elapsed_seconds=0, input_tokens_reported=0, output_tokens_reported=0, llm_performance=[],
        ),
    )


def _evaluate(case: EvalCase, run: ResearchRun, report: ResearchReport, *, fault: str | None, elapsed: float) -> dict:
    try:
        validate_research_report(report, run)
        projection_matches = True
    except ValueError:
        projection_matches = False
    quality = evaluate_quality(run.state)
    results = {item.result_id: item for item in run.state.tool_results}
    observations = {item.observation_id: item for item in run.state.observations}
    valid_ids = {item.evidence_id for item in quality.evidence_checks if item.valid}
    evidence = {item.evidence_id: item for item in run.state.evidence if item.evidence_id in valid_ids}
    citation_checks = [
        all(
            record.observation_id in observations
            and observations[record.observation_id].result_id == record.result_id
            and (citation.url is None or any(ref.url == citation.url for ref in results[record.result_id].sources))
            for record in citation.records
        ) for citation in report.citations
    ]
    evidence_checks = [
        eid in evidence and record.observation_id in evidence[eid].observation_ids
        for citation in report.citations for record in citation.records for eid in record.evidence_ids
    ]
    qualified_facts = {(item.kind, item.statement) for item in quality.supported_findings}
    unsupported_facts = sum(
        (item.kind, item.statement) not in qualified_facts for item in report.supported_findings
    )
    unsupported_causal = sum(item.kind is ClaimKind.CAUSAL_CONCLUSION for item in report.supported_findings)
    measured_returns = sorted(round(item.stock.return_pct, 9) for item in report.metrics if isinstance(item, MarketMetric))
    outcome_matches = (
        report.quality_status in case.expectations.quality and report.stop_reason in case.expectations.stops
        and report.goal_satisfied == (report.stop_reason is StopReason.GOAL_SATISFIED)
    )
    checks = {
        "citation_validity": all(citation_checks),
        "evidence_integrity": all(evidence_checks),
        "financial_metrics": measured_returns == EXPECTED_RETURNS.get(case.case_id, []),
        "supported_claims": unsupported_facts == 0,
        "causal_grounding": unsupported_causal == 0,
        "report_completeness": projection_matches,
        "quality_and_stop": outcome_matches,
        "fallback_behavior": report.narrative.fallback_used if fault else True,
        "authorized_research_dispatches": all(step.authorized_tool in case.tools for step in run.steps if step.tool_executed),
    }
    return {
        "case": case.case_id, "data_mode": "fixtures", "fault": fault,
        "passed": all(checks.values()), "checks": checks, "quality": report.quality_status.value,
        "stop_reason": report.stop_reason.value,
        "valid_citations": sum(citation_checks), "citations": len(citation_checks),
        "valid_evidence_references": sum(evidence_checks), "evidence_references": len(evidence_checks),
        "unsupported_factual_claims": unsupported_facts, "unsupported_causal_claims": unsupported_causal,
        "deterministic_corrections": int(report.narrative.validation_succeeded is False),
        "report_seconds": elapsed, "narrative": report.narrative.model_dump(mode="json"),
    }


def run_suite(
    case_ids: tuple[str, ...], *, provider: LLMProvider | None = None, fault_probes: bool = False,
    report_dir: Path | None = None,
) -> dict:
    rows = []
    planned = [(cid, None) for cid in case_ids]
    if fault_probes:
        planned.extend([("E3", "reference"), ("E9", "schema"), ("E8", "connection")])
    for cid, fault in planned:
        case = CASES_BY_ID[cid]
        run = replay_research(case)
        started = monotonic()
        report = generate_research_report(run, provider if provider is not None else ScriptedDraft(fault))
        elapsed = monotonic() - started
        row = _evaluate(case, run, report, fault=fault, elapsed=elapsed)
        if report_dir is not None:
            row["reports"] = [
                export_research_report(report, report_dir, format=format).name for format in ("md", "json")
            ]
        rows.append(row)
    return {
        "cases": rows,
        "cases_passed": sum(row["passed"] for row in rows), "case_count": len(rows),
        "valid_citations": sum(row["valid_citations"] for row in rows),
        "citations": sum(row["citations"] for row in rows),
        "valid_evidence_references": sum(row["valid_evidence_references"] for row in rows),
        "evidence_references": sum(row["evidence_references"] for row in rows),
        "unsupported_factual_claims": sum(row["unsupported_factual_claims"] for row in rows),
        "unsupported_causal_claims": sum(row["unsupported_causal_claims"] for row in rows),
        "narrative_validation_failures": sum(row["deterministic_corrections"] for row in rows),
        "fallbacks": sum(row["narrative"]["fallback_used"] for row in rows),
        "report_model_requests": sum(row["narrative"]["requests"] for row in rows),
        "report_seconds": sum(row["report_seconds"] for row in rows),
        "reported_tokens": {
            field: {
                "total": sum(counts) if (counts := [row["narrative"][field] for row in rows
                                                   if row["narrative"][field] is not None]) else None,
                "records": len(counts),
            } for field in ("input_tokens", "output_tokens")
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "local"), default="offline")
    parser.add_argument("--case", action="append", choices=CASES_BY_ID)
    parser.add_argument("--env-file", type=Path, help="Existing loopback Ollama settings file.")
    parser.add_argument("--report-dir", type=Path, help="Export Markdown and JSON reports.")
    parser.add_argument("--json", type=Path, help="Create a JSON evaluation summary.")
    args = parser.parse_args(argv)
    if args.env_file and args.mode != "local":
        parser.error("--env-file requires local mode.")
    if args.mode == "offline":
        summary = run_suite(tuple(args.case or CASES_BY_ID), fault_probes=not args.case, report_dir=args.report_dir)
    else:
        with open_llm_provider(load_llm_settings(args.env_file)) as provider:
            summary = run_suite(tuple(args.case or LOCAL_CASES), provider=provider, report_dir=args.report_dir)
    for row in summary["cases"]:
        print(f'{row["case"]} | {row["quality"]}/{row["stop_reason"]} | {row["narrative"]["mode"]} | '
              + ("PASS" if row["passed"] else "FAIL"))
    print(f'{summary["cases_passed"]}/{summary["case_count"]} passed')
    if args.json:
        with args.json.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
            stream.write("\n")
    return 0 if summary["cases_passed"] == summary["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
