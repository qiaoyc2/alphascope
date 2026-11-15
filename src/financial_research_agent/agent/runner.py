"""Run bounded financial research against the existing planner and tool contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import Field, ValidationError

from financial_research_agent.agent.budget import (
    BudgetedLLM, DeadlineExceeded, LLMRequestBudgetExceeded, ResearchLimits, RunBudget,
)
from financial_research_agent.agent.dispatch import ActionRejected, ToolRegistry
from financial_research_agent.agent.goal import interpret_goal
from financial_research_agent.agent.hypotheses import (
    HypothesisProposalError, HypothesisUpdate, apply_hypothesis_update, initialize_research,
)
from financial_research_agent.agent.observe import observe_result
from financial_research_agent.agent.planning import (
    GoalStatus, PlanDecision, PlanningContext, PlanValidationError, ResearchPlanner, record_plan,
)
from financial_research_agent.agent.premise import assess_premise
from financial_research_agent.agent.state import add_research_gap, finish_research
from financial_research_agent.llm_decision import RecommendedAction
from financial_research_agent.models import (
    ExecutionMetadata, HypothesisStatus, ObservationOutcome, ResearchAction, ResearchGoal,
    ResearchModel, ResearchState, ResearchStatus, ResearchType, ToolError, ToolResult, utc_now,
)
from financial_research_agent.providers.llm import LLMError, LLMErrorCategory, LLMPerformance, LLMProvider
from financial_research_agent.tools import ToolName
from financial_research_agent.tools.planning import action_fingerprint


class StopReason(StrEnum):
    GOAL_SATISFIED = "goal_satisfied"
    NEEDS_CLARIFICATION = "needs_clarification"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED_QUESTION = "unsupported_question"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REPEATED_ACTION = "repeated_action"
    INVALID_PROPOSAL = "invalid_proposal"
    TOOL_FAILURE = "tool_failure"
    LLM_FAILURE = "llm_failure"
    EXECUTION_ERROR = "execution_error"


class RunIssue(ResearchModel):
    code: str
    message: str
    step: int | None = None


class RunStep(ResearchModel):
    number: int
    recommended_action: RecommendedAction | None = None
    proposed_tool: ToolName | None = None
    validation: str = "accepted"
    note: str | None = None
    fingerprint: str | None = None
    action_id: str | None = None
    result_id: str | None = None
    observation_id: str | None = None
    hypothesis_changes: dict[str, HypothesisStatus] = Field(default_factory=dict)


class RunStatistics(ResearchModel):
    planning_steps: int
    tool_calls: int
    provider_calls_reported: int
    llm_requests: int
    elapsed_seconds: float
    input_tokens_reported: int
    output_tokens_reported: int
    llm_performance: list[LLMPerformance]


class ResearchRun(ResearchModel):
    state: ResearchState
    stop_reason: StopReason
    stop_detail: str
    statistics: RunStatistics
    steps: list[RunStep]
    issues: list[RunIssue]


@dataclass
class _Progress:
    state: ResearchState
    steps: list[RunStep] = field(default_factory=list)
    issues: list[RunIssue] = field(default_factory=list)
    failed_actions: int = 0
    rejected_proposals: int = 0

    def flag(self, code: str, message: str, *, step: int | None = None) -> None:
        self.issues.append(RunIssue(code=code, message=message, step=step))

    def block(self, step: RunStep, code: str, reason: str) -> None:
        step.validation = "blocked"
        step.note = reason
        self.rejected_proposals += 1
        self.flag(code, reason, step=step.number)


_MALFORMED = {
    LLMErrorCategory.INVALID_JSON, LLMErrorCategory.INVALID_RESPONSE,
    LLMErrorCategory.SCHEMA_VALIDATION, LLMErrorCategory.EMPTY_RESPONSE,
}


def _changes(before: ResearchState, after: ResearchState) -> dict[str, HypothesisStatus]:
    previous = {item.hypothesis_id: item.status for item in before.hypotheses}
    return {
        item.hypothesis_id: item.status for item in after.hypotheses
        if previous.get(item.hypothesis_id) != item.status
    }


def _reassess_price_hypotheses(state: ResearchState) -> ResearchState:
    assessment = assess_premise(state)
    if assessment.conditional_valid is not False:
        return state
    for item in state.hypotheses:
        if not item.depends_on_price_premise:
            continue
        state = apply_hypothesis_update(state, HypothesisUpdate(
            hypothesis_id=item.hypothesis_id,
            status=HypothesisStatus.INCONCLUSIVE,
            observation_ids=assessment.observation_ids[:8],
            reason="Inapplicable to the stated price premise because the observed direction contradicts it.",
        ))
    return state


def _sufficient(state: ResearchState) -> bool:
    """Conservative completion checks for the supported goal categories."""
    premise = assess_premise(state)
    if state.goal.needs_clarification or premise.observations_conflict or premise.conditional_valid is False:
        return False
    if state.goal.research_type is ResearchType.PRICE_COMPARISON:
        return bool(premise.observation_ids)
    source = {
        ResearchType.REGULATORY: "news_source",
        ResearchType.FILINGS: "sec_filings_source",
    }.get(state.goal.research_type)
    if source is None:
        return False
    action_map = {item.action_id: item for item in state.completed_actions}
    observations = {
        item.observation_id: item for item in state.observations if item.outcome is ObservationOutcome.DATA
    }
    for evidence in state.evidence:
        if evidence.evidence_type != source:
            continue
        for observation_id in evidence.observation_ids:
            observation = observations.get(observation_id)
            if observation is None:
                continue
            args = action_map[observation.action_id].arguments
            start, end = state.goal.start_date, state.goal.end_date
            if (not start or args.get("start_date") == start.isoformat()) and (
                not end or args.get("end_date") == end.isoformat()
            ):
                return True
    return False


def _decision_stop(state: ResearchState, decision: PlanDecision) -> tuple[StopReason, str] | None:
    if decision.clarification_required or decision.recommended_action is RecommendedAction.REVISE:
        return StopReason.NEEDS_CLARIFICATION, decision.information_gap or decision.reason
    if not decision.should_stop:
        return None
    if not decision.information_gap and _sufficient(state):
        return StopReason.GOAL_SATISFIED, decision.reason
    if state.observations and all(item.outcome is ObservationOutcome.FAILED for item in state.observations):
        return StopReason.TOOL_FAILURE, decision.information_gap or decision.reason
    if decision.goal_status is GoalStatus.UNSUPPORTED_GAP and not state.completed_actions:
        return StopReason.UNSUPPORTED_QUESTION, decision.information_gap or decision.reason
    return StopReason.INSUFFICIENT_EVIDENCE, decision.information_gap or "The recorded evidence does not establish the goal."


def _make_result(
    progress: _Progress, budget: RunBudget, llm: BudgetedLLM, stop: tuple[StopReason, str],
) -> ResearchRun:
    reason, detail = stop
    assessment = assess_premise(progress.state)
    if reason is not StopReason.NEEDS_CLARIFICATION and (
        assessment.observations_conflict or assessment.conditional_valid is False
    ):
        reason = StopReason.NEEDS_CLARIFICATION
        detail = (assessment.summary or "The price measurements disagree.") + " Confirm the intended price premise or scope."
        progress.state = add_research_gap(progress.state, detail)
    status = ResearchStatus.STOPPED
    if reason is StopReason.GOAL_SATISFIED:
        status = ResearchStatus.COMPLETED
    elif reason in {StopReason.EXECUTION_ERROR, StopReason.LLM_FAILURE, StopReason.TOOL_FAILURE}:
        status = ResearchStatus.FAILED
    state = finish_research(progress.state, status, reason.value)
    performance = llm.performance
    return ResearchRun(
        state=state, stop_reason=reason, stop_detail=detail, steps=progress.steps, issues=progress.issues,
        statistics=RunStatistics(
            planning_steps=len(progress.steps), tool_calls=state.tool_call_count,
            provider_calls_reported=state.api_call_count, llm_requests=budget.llm_requests,
            elapsed_seconds=budget.elapsed, input_tokens_reported=sum(p.input_tokens or 0 for p in performance),
            output_tokens_reported=sum(p.output_tokens or 0 for p in performance),
            llm_performance=performance,
        ),
    )


class ResearchRunner:
    def __init__(self, provider: LLMProvider, tools: ToolRegistry):
        self.provider = provider
        self.tools = tools

    def run(
        self, question: str, limits: ResearchLimits | None = None, *, today: date | None = None,
    ) -> ResearchRun:
        if not question.strip() or len(question) > 2000:
            raise ValueError("research question must contain 1–2000 characters")
        limits = ResearchLimits.model_validate(limits) if limits is not None else ResearchLimits()
        budget = RunBudget(limits)
        llm = BudgetedLLM(self.provider, budget)
        progress = _Progress(ResearchState(goal=ResearchGoal(query=question, objective=question)))
        try:
            goal, _ = interpret_goal(question, llm, clock=lambda: today or date.today())
            progress.state = ResearchState(goal=goal, research_gaps=goal.unresolved_details)
            progress.state, _ = initialize_research(goal, llm)
            stop = self._loop(progress, budget, llm)
        except DeadlineExceeded:
            stop = (StopReason.BUDGET_EXHAUSTED, "The elapsed-time limit was reached.")
        except LLMRequestBudgetExceeded:
            stop = (StopReason.BUDGET_EXHAUSTED, "The LLM request limit was reached.")
        except LLMError as exc:
            progress.flag(f"llm_{exc.category}", "Structured model generation failed.")
            stop = (StopReason.LLM_FAILURE, "The language model did not return a usable response.")
        except HypothesisProposalError as exc:
            progress.flag("invalid_initial_hypotheses", str(exc))
            stop = (StopReason.LLM_FAILURE, "The language model did not propose distinct initial hypotheses.")
        except ValidationError:
            progress.flag("state_validation", "Research state failed validation.")
            stop = (StopReason.EXECUTION_ERROR, "The last valid research state was preserved.")
        return _make_result(progress, budget, llm, stop)

    def _loop(
        self, progress: _Progress, budget: RunBudget, llm: BudgetedLLM,
    ) -> tuple[StopReason, str]:
        planner = ResearchPlanner(llm)
        limits = budget.limits
        for number in range(1, limits.max_steps + 1):
            remaining = budget.remaining_time()
            step = RunStep(number=number)
            progress.steps.append(step)
            context = PlanningContext(
                available_tools=list(self.tools.enabled),
                steps_remaining=limits.max_steps - number,
                tool_calls_remaining=limits.max_tool_calls - progress.state.tool_call_count,
                llm_requests_remaining=max(0, limits.max_llm_requests - budget.llm_requests - 1),
                seconds_remaining=round(remaining),
                feedback=[item.message[:150] for item in progress.issues[-3:]],
            )
            try:
                planned = planner.plan(progress.state, execution=context)
            except PlanValidationError as exc:
                progress.block(step, "invalid_proposal", str(exc))
            except LLMError as exc:
                if exc.category not in _MALFORMED:
                    step.validation = "llm_failed"
                    raise
                progress.block(step, f"llm_{exc.category}", "The model produced malformed structured output.")
                if progress.rejected_proposals >= limits.max_repeated_proposals:
                    return StopReason.LLM_FAILURE, "The model repeatedly produced malformed output."
            except (DeadlineExceeded, LLMRequestBudgetExceeded):
                step.validation = "budget_exhausted"
                raise
            else:
                step.recommended_action = planned.decision.recommended_action
                step.proposed_tool = planned.blocked_tool or planned.decision.next_tool
                step.note = planned.adjustment
                if planned.blocked_tool is not None:
                    step.fingerprint = planned.blocked_fingerprint
                    progress.block(step, "duplicate_action", "An equivalent tool action was already covered.")
                else:
                    stop = self._accept(progress, step, planned.decision, budget)
                    if stop is not None:
                        return stop
                    if step.validation != "blocked":
                        progress.rejected_proposals = 0
            if progress.rejected_proposals >= limits.max_repeated_proposals:
                code = progress.issues[-1].code
                reason = StopReason.REPEATED_ACTION if code == "duplicate_action" else StopReason.INVALID_PROPOSAL
                return reason, "Repeated redundant or invalid proposals prevented progress."
        return StopReason.BUDGET_EXHAUSTED, "The planning step limit was reached."

    def _accept(
        self, progress: _Progress, step: RunStep, decision: PlanDecision, budget: RunBudget,
    ) -> tuple[StopReason, str] | None:
        before = progress.state
        terminal = _decision_stop(before, decision)
        if terminal is not None:
            # Apply any final assessment before deciding evidence sufficiency.
            progress.state = record_plan(before, decision)
            step.hypothesis_changes = _changes(before, progress.state)
            return _decision_stop(progress.state, decision)

        tool = decision.next_tool
        if tool is None:
            progress.block(step, "invalid_proposal", "The recommendation did not specify a useful next action.")
            return None
        if before.tool_call_count >= budget.limits.max_tool_calls:
            step.validation = "budget_exhausted"
            return StopReason.BUDGET_EXHAUSTED, "The tool call limit was reached."
        try:
            args = self.tools.authorize(tool, decision.tool_arguments, before)
        except ActionRejected as exc:
            step.fingerprint = action_fingerprint(tool, decision.tool_arguments) if exc.code == "duplicate_action" else None
            progress.block(step, exc.code, str(exc))
            return None
        budget.remaining_time()
        progress.state = record_plan(before, decision)
        action = progress.state.planned_actions[-1]
        step.action_id = action.action_id
        step.fingerprint = action_fingerprint(tool, args)
        return self._execute(progress, step, action, budget, before)

    def _execute(
        self, progress: _Progress, step: RunStep, action: ResearchAction, budget: RunBudget,
        before: ResearchState,
    ) -> tuple[StopReason, str] | None:
        started = utc_now()
        timed_out = False
        try:
            pending = progress.state
            result = budget.call(lambda: self.tools.execute(pending, action))
        except DeadlineExceeded:
            timed_out = True
            result = _failure(action, "DEADLINE_EXCEEDED", started)
        except ActionRejected as exc:
            progress.flag(exc.code, str(exc), step=step.number)
            result = _failure(action, exc.code, started)
        try:
            progress.state = observe_result(progress.state, action, result, self.tools.provenance(ToolName(action.tool_name)))
        except (ValidationError, ValueError):
            progress.flag("tool_result_validation", "The result failed validation for the dispatched action.", step=step.number)
            result = _failure(action, "INVALID_TOOL_RESULT", started)
            progress.state = observe_result(progress.state, action, result, self.tools.provenance(ToolName(action.tool_name)))
            step.result_id = result.result_id
            step.observation_id = progress.state.observations[-1].observation_id
            return StopReason.EXECUTION_ERROR, "The invalid result was discarded and the last valid evidence was preserved."
        observation = progress.state.observations[-1]
        step.result_id = result.result_id
        step.observation_id = observation.observation_id
        progress.state = _reassess_price_hypotheses(progress.state)
        step.hypothesis_changes = _changes(before, progress.state)
        if result.success:
            progress.failed_actions = 0
            if observation.outcome in {ObservationOutcome.EMPTY, ObservationOutcome.INCOMPLETE}:
                progress.flag(f"observation_{observation.outcome}", observation.summary[:230], step=step.number)
            return None
        assert result.error is not None
        progress.failed_actions += 1
        progress.flag(f"tool_{result.error.code}", observation.summary, step=step.number)
        if timed_out:
            return StopReason.BUDGET_EXHAUSTED, "The elapsed-time limit was reached during a tool call."
        if progress.failed_actions >= budget.limits.max_consecutive_failures:
            return StopReason.TOOL_FAILURE, "The consecutive tool failure limit was reached."
        return None


def _failure(action: ResearchAction, code: str, started: datetime) -> ToolResult:
    return ToolResult(
        result_id="r-" + uuid4().hex, action_id=action.action_id, tool_name=action.tool_name, success=False,
        error=ToolError(code=code, message="The dispatched action did not return a usable result."),
        metadata=ExecutionMetadata(started_at=started, finished_at=utc_now()),
    )
