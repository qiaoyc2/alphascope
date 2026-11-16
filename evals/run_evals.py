"""Run fixed offline evaluations or bounded loopback Ollama trials."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.cases import CASES, LOCAL_CASE_IDS, Check, EvalCase, Expectations, ScriptedProvider, conflicting_state, registry
from financial_research_agent.agent.quality import (
    Completeness, Disposition, QualityAssessment, SupportStatus, evaluate_quality,
)
from financial_research_agent.agent.runner import ResearchRun, ResearchRunner, StopReason
from financial_research_agent.config import load_llm_settings, open_llm_provider
from financial_research_agent.models import ResearchState
from financial_research_agent.providers.llm import LLMProvider
from financial_research_agent.providers.ollama import OllamaProvider
from financial_research_agent.tools import ToolName


METRIC_DEFINITIONS = {
    "structured_output_rate": "Schema-valid generations / generations returning structured output or a response validation error. Transport and model availability failures are excluded.",
    "raw_action_validity_rate": "Raw tool proposals accepted without deterministic correction / assessed raw tool proposals, including decoded schema-invalid tool proposals.",
    "final_action_validity_rate": "Dispatched tool actions matching the recorded authorized tool / dispatched tool actions.",
    "duplicate_proposal_rate": "Duplicate raw tool proposals / assessed raw tool proposals.",
    "tool_execution_success_rate": "Successful dispatched tool results / dispatched tool actions.",
    "correct_stopping_rate": "Cases meeting expected terminal stop reason and quality / cases evaluated. Includes E4's quality-only stop recommendation.",
    "quality_gate_rejections": "Planning steps whose correction category is quality_gate_rejection.",
    "guardrail_interventions": "Planning steps with a structured correction or rejection category.",
    "quality_gate_terminations": "Runner outcomes selected independently of the planner after assessing state.",
    "quality_gate_completions": "Independent quality-gate terminations with goal_satisfied.",
    "unsupported_claims": "Findings assessed as unsupported. Provisional hypotheses marked not_evaluated are reported separately.",
    "premature_satisfactions": "Final goal_satisfied outcomes without gate support, required findings, or valid provenance.",
    "unexpected_satisfactions": "Final goal_satisfied outcomes that fail the versioned stop, quality, satisfaction or required findings checks (the M5 premature_satisfactions formula).",
    "raw_premature_satisfaction_proposals": "Raw satisfaction requests vetoed by the quality gate.",
    "provenance": "Valid / examined evidence records.",
    "retries_reported": "Additional attempts reported by completed LLM generations. Interrupted generations can have unreported retry details.",
    "reported_tokens": "Sum of provider-supplied token counts, with contributing generation counts. Unreported totals are null.",
    "reported_request_latency_seconds": "Per-attempt durations from the Ollama adapter, including failed attempts where reported.",
    "estimated_paid_inference_usd": "Local Ollama makes no paid API requests. Hardware and electricity costs are outside this measure.",
}


def _ratio(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def _expectations(case: EvalCase, version: str) -> Expectations:
    if version == "m5":
        return case.expectations
    if version != "m5.1":
        raise ValueError("unknown evaluation expectations version")
    if case.case_id != "E6":
        return case.expectations
    # E6's historical expectations deliberately consume duplicate proposals after
    # a sufficient title lookup. The same fixture now stops at the first retrieval.
    return Expectations.model_validate({
        **case.expectations.model_dump(),
        "stops": {StopReason.GOAL_SATISFIED},
        "checks": case.expectations.checks - {Check.DUPLICATE_SOURCE, Check.DUPLICATE_PROPOSAL} | {Check.SOURCE_LIMITATIONS},
        "max_tool_calls": 1,
    })


def _checks(
    case: EvalCase, state: ResearchState, quality: QualityAssessment, stop: StopReason, run: ResearchRun | None,
    *, expectations_version: str,
) -> dict[str, bool]:
    expected = _expectations(case, expectations_version)
    steps = run.steps if run else []
    limits = case.limits
    executed = [step.authorized_tool for step in steps if step.tool_executed]
    kinds = {item.kind for item in quality.supported_findings}
    duplicates = [item for item in quality.sources if item.duplicate_of]
    failed = {item.observation_id for item in quality.sources if item.completeness is Completeness.FAILED}
    # An empty set of evidence references is expected on failed observations.
    evidenced = {oid for evidence in state.evidence for oid in evidence.observation_ids}
    checks = {
        Check.ORIGINAL_GOAL: state.goal.query == case.question,
        Check.PROVENANCE: all(check.valid for check in quality.evidence_checks),
        Check.EXECUTION_LIMITS: run is None or (
            run.statistics.llm_requests <= limits.max_llm_requests
            and run.statistics.planning_steps <= limits.max_steps
            and run.statistics.tool_calls <= limits.max_tool_calls
        ),
        Check.PRICE_FIRST: bool(executed and executed[0] is ToolName.MARKET_DATA),
        Check.PREMISE_CONTRADICTED: quality.premise_contradicted,
        Check.CONFLICT_RECORDED: bool(quality.contradictions) and not quality.goal_satisfied,
        Check.DUPLICATE_SOURCE: bool(duplicates) and len(quality.supported_findings) == 1,
        Check.DUPLICATE_PROPOSAL: any(step.correction_category == "duplicate_action" and not step.tool_executed for step in steps),
        Check.FAILED_WITHOUT_EVIDENCE: bool(failed) and failed.isdisjoint(evidenced),
        Check.QUALITY_VETO: any(step.correction_category == "quality_gate_rejection"
                               and step.raw_proposal and step.raw_proposal.requests_satisfaction
                               and step.final_disposition != StopReason.GOAL_SATISFIED for step in steps),
        Check.CORRECTED_ACTION: any(
            step.raw_proposal and step.raw_proposal.tool and step.raw_valid is False
            and step.correction_category and not step.tool_executed for step in steps
        ) and any(step.raw_valid and step.tool_executed for step in steps),
        Check.SOURCE_LIMITATIONS: any(item.limitations for item in quality.sources),
    }
    return {
        "stop": stop in expected.stops,
        "quality": quality.status in expected.quality,
        "satisfaction_consistent": (stop != StopReason.GOAL_SATISFIED or quality.goal_satisfied),
        "required_findings": expected.findings <= kinds,
        "prohibited_findings": not expected.prohibited_findings & kinds,
        "required_tools": expected.required_tools <= set(executed),
        "prohibited_tools": not expected.prohibited_tools & set(executed),
        "observed_completeness": expected.completeness <= {item.completeness for item in quality.sources},
        "tool_call_cap": len(executed) <= expected.max_tool_calls,
        **{check.value: checks[check] for check in sorted(expected.checks)},
    }


def _collect(
    case: EvalCase, state: ResearchState, quality: QualityAssessment, stop: StopReason, run: ResearchRun | None,
    *, mode: str, trial: int, live_sources: bool, expectations_version: str,
) -> dict:
    checks = _checks(case, state, quality, stop, run, expectations_version=expectations_version)
    statistics = run.statistics if run else None
    # Artifact includes evidence links and structured telemetry. Model prose, prompts and raw result bodies stay in memory.
    assessment = quality.model_dump(mode="json", exclude={
        "unverified_claims": {"__all__": {"statement"}}, "premise": {"summary"},
    })
    result_by_action = {item.action_id: item for item in state.tool_results}
    actions = {item.action_id: item for item in state.completed_actions}
    steps = [
        step.model_dump(mode="json", exclude={"note", "fingerprint"})
        | {"tool_success": result_by_action[step.action_id].success if step.action_id in result_by_action else None}
        | {"tool_authorization_matches": bool(
            step.action_id in actions and step.action_id in result_by_action
            and step.authorized_tool == actions[step.action_id].tool_name == result_by_action[step.action_id].tool_name
        ) if step.tool_executed else None}
        for step in run.steps
    ] if run else []
    return {
        "case_id": case.case_id, "label": case.label, "trial": trial, "mode": mode,
        "clock_date": case.today.isoformat(), "question": case.question,
        "data_mode": "live_sources_enabled" if live_sources and case.live_source else "fixtures",
        "quality_only": case.quality_only,
        "passed": all(checks.values()), "checks": checks, "stop_reason": stop.value,
        "quality_gate_terminated": run.quality_gate_terminated if run else False,
        "quality": assessment, "steps": steps,
        "issues": [item.model_dump(mode="json", exclude={"message"}) for item in run.issues] if run else [],
        "statistics": statistics.model_dump(mode="json") if statistics else None,
        "tool_results": [{
            "result_id": item.result_id, "tool": item.tool_name, "success": item.success,
            "error_code": item.error.code if item.error else None, "sources": [str(source.url) for source in item.sources],
            "provider": item.data.get("provider", item.data.get("stock_provider")),
        } for item in state.tool_results],
    }


def evaluate_case(
    case: EvalCase, *, provider: LLMProvider | None = None, trial: int = 1, live_sources: bool = False,
    expectations_version: str = "m5",
) -> dict:
    case = EvalCase.model_validate(case)
    if live_sources and provider is None:
        raise ValueError("offline evaluations only use fixtures")
    if case.quality_only:
        state = conflicting_state(case)
        quality = evaluate_quality(state, available_tools=case.tools)
        stop = {
            Disposition.CLARIFY: StopReason.NEEDS_CLARIFICATION,
            Disposition.COMPLETE: StopReason.GOAL_SATISFIED,
        }.get(quality.recommended_disposition, StopReason.INSUFFICIENT_EVIDENCE)
        run = None
    else:
        run = ResearchRunner(
            provider if provider is not None else ScriptedProvider(case), registry(case, live_sources=live_sources),
        ).run(case.question, case.limits, today=case.today)
        state, quality, stop = run.state, run.quality, run.stop_reason
        assert quality is not None
    return _collect(case, state, quality, stop, run, mode="offline" if provider is None else "local",
                    trial=trial, live_sources=live_sources, expectations_version=expectations_version)


def summarize(results: list[dict]) -> dict[str, Any]:
    steps = [step for result in results for step in result["steps"]]
    executions = [step for step in steps if step["tool_executed"]]
    raw = [step for step in steps if step["raw_proposal"] and (
        step["raw_proposal"]["tool"] or step["raw_proposal"]["unknown_tool"]
    ) and step["raw_valid"] is not None]
    stats = [result["statistics"] for result in results if result["statistics"] is not None]
    generations = [item for stat in stats for item in stat["llm_performance"]]
    schema_errors = {"invalid_json", "invalid_response", "schema_validation", "empty_response"}
    assessed_generations = [item for item in generations if item["validation_passed"] or item["error_category"] in schema_errors]
    evidence = [item for result in results for item in result["quality"]["evidence_checks"]]
    sources = [item for result in results for item in result["quality"]["sources"]]
    claims = [item for result in results for field in ("supported_findings", "unverified_claims")
              for item in result["quality"][field]]
    input_counts = [item["input_tokens"] for item in generations if item["input_tokens"] is not None]
    output_counts = [item["output_tokens"] for item in generations if item["output_tokens"] is not None]
    variants = {
        case_id: dict(Counter((result["stop_reason"] + "/" + result["quality"]["status"])
                             for result in results if result["case_id"] == case_id))
        for case_id in sorted({item["case_id"] for item in results})
    }
    return {
        "cases_passed": _ratio(sum(item["passed"] for item in results), len(results)),
        "structured_output_rate": _ratio(sum(item["validation_passed"] for item in assessed_generations), len(assessed_generations)),
        "raw_action_validity_rate": _ratio(sum(item["raw_valid"] for item in raw), len(raw)),
        "final_action_validity_rate": _ratio(sum(item["tool_authorization_matches"] for item in executions), len(executions)),
        "duplicate_proposal_rate": _ratio(sum(item["correction_category"] == "duplicate_action" for item in raw), len(raw)),
        "tool_execution_success_rate": _ratio(sum(item["tool_success"] is True for item in executions), len(executions)),
        "guardrail_interventions": sum(bool(item["correction_category"]) for item in steps),
        "quality_gate_terminations": sum(item["quality_gate_terminated"] for item in results),
        "quality_gate_completions": sum(
            item["quality_gate_terminated"] and item["stop_reason"] == StopReason.GOAL_SATISFIED for item in results
        ),
        "correct_stopping_rate": _ratio(sum(item["checks"]["stop"] and item["checks"]["quality"]
                                           and item["checks"]["satisfaction_consistent"] for item in results), len(results)),
        "quality_gate_rejections": sum(item["correction_category"] == "quality_gate_rejection" for item in steps),
        "raw_premature_satisfaction_proposals": sum(
            item["correction_category"] == "quality_gate_rejection" and bool(item["raw_proposal"])
            and item["raw_proposal"]["requests_satisfaction"] for item in steps
        ),
        "unsupported_claims": sum(item["support_status"] == SupportStatus.UNSUPPORTED for item in claims),
        "premature_satisfactions": sum(
            item["stop_reason"] == StopReason.GOAL_SATISFIED and (
                not all(item["checks"][check] for check in (
                    "satisfaction_consistent", "required_findings", "prohibited_findings",
                )) or not all(check["valid"] for check in item["quality"]["evidence_checks"])
            ) for item in results
        ),
        "unexpected_satisfactions": sum(
            item["stop_reason"] == StopReason.GOAL_SATISFIED and not all(
                item["checks"][check] for check in ("stop", "quality", "satisfaction_consistent", "required_findings")
            ) for item in results
        ),
        "provenance": _ratio(sum(item["valid"] for item in evidence), len(evidence)),
        "source_completeness": dict(Counter(item["completeness"] for item in sources)),
        "duplicate_evidence": sum(item["duplicate_of"] is not None for item in evidence),
        "duplicate_observations": sum(item["duplicate_of"] is not None for item in sources),
        "claim_support": dict(Counter(item["support_status"] for item in claims)),
        "contradiction_pairs": sum(len(item["quality"]["contradictions"]) for item in results),
        "llm_request_attempts": sum(item["llm_requests"] for item in stats),
        "llm_generation_records": len(generations),
        "reported_tokens": {
            "input": sum(input_counts) if input_counts else None, "input_records": len(input_counts),
            "output": sum(output_counts) if output_counts else None, "output_records": len(output_counts),
        },
        "reported_request_latency_seconds": [
            value for generation in generations for value in generation["attempt_durations_seconds"]
        ],
        "retries_reported": sum(item["attempts"] - 1 for item in generations),
        "tool_calls": sum(item["tool_calls"] for item in stats),
        "tool_wait_seconds": [item["tool_wait_seconds"] for item in executions if item["tool_wait_seconds"] is not None],
        "tool_calls_in_flight_at_exit": sum(item["tool_in_flight_at_exit"] for item in steps),
        "stop_reasons": dict(Counter(item["stop_reason"] for item in results)),
        "corrections": dict(Counter(item["correction_category"] for item in steps if item["correction_category"])),
        "observed_stop_quality_by_case": variants,
    }


def run_suite(cases: tuple[EvalCase, ...] = CASES, *, provider: LLMProvider | None = None,
              repeat: int = 1, live_sources: bool = False, expectations_version: str = "m5") -> dict:
    if not cases or repeat < 1 or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("a suite requires distinct cases and at least one trial")
    results = [evaluate_case(case, provider=provider, trial=trial, live_sources=live_sources,
                             expectations_version=expectations_version)
               for trial in range(1, repeat + 1) for case in cases]
    metrics = summarize(results)
    metrics["estimated_paid_inference_usd"] = 0 if isinstance(provider, OllamaProvider) else None
    return {"expectations_version": expectations_version, "metric_definitions": METRIC_DEFINITIONS,
            "metrics": metrics, "cases": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "local"), default="offline")
    parser.add_argument("--case", action="append", choices=[case.case_id for case in CASES])
    parser.add_argument("--repeat", type=int, choices=range(1, 6))
    parser.add_argument("--expectations", choices=("m5", "m5.1"), default="m5")
    parser.add_argument("--live-sources", action="store_true", help="Use approved Yahoo/DOJ providers for applicable local cases.")
    parser.add_argument("--env-file", type=Path, help="Use an existing Ollama settings file in local mode.")
    parser.add_argument("--json", type=Path, metavar="PATH", help="Save metrics and safe structured traces.")
    args = parser.parse_args(argv)
    if args.mode == "offline" and (args.live_sources or args.env_file):
        parser.error("offline mode only uses frozen fixtures and requires no settings")
    ids = args.case or (LOCAL_CASE_IDS if args.mode == "local" else [case.case_id for case in CASES])
    cases = tuple(case for case in CASES if case.case_id in ids)
    if args.mode == "local":
        try:
            settings = load_llm_settings(args.env_file)
        except ValueError:
            parser.error("local Ollama settings were invalid")
        with open_llm_provider(settings) as provider:
            report = run_suite(cases, provider=provider, repeat=args.repeat or 2, live_sources=args.live_sources,
                               expectations_version=args.expectations)
        report["local_model"] = settings.model
    else:
        report = run_suite(cases, repeat=args.repeat or 1, expectations_version=args.expectations)
    for item in report["cases"]:
        failures = [check for check, passed in item["checks"].items() if not passed]
        line = (f"{item['case_id']} trial {item['trial']} | {item['data_mode']} | "
                f"{item['quality']['status']} / {item['stop_reason']}")
        print(line + (f" | FAIL ({', '.join(failures)})" if failures else " | PASS"))
    metrics = report["metrics"]
    passed = metrics["cases_passed"]
    print(f"{passed['numerator']}/{passed['denominator']} passed | {metrics['tool_calls']} tools | "
          f"{metrics['llm_request_attempts']} LLM attempts | {metrics['quality_gate_rejections']} quality vetoes")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return 0 if passed["numerator"] == passed["denominator"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
