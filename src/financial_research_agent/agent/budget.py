"""Per-run limits and bounded calls for synchronous research providers."""

from __future__ import annotations

from collections.abc import Callable
from queue import Empty, Queue
from threading import Lock, Thread
from time import monotonic
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import Field

from financial_research_agent.models import ResearchModel
from financial_research_agent.providers.llm import (
    DecisionT, LLMError, LLMPerformance, LLMProvider, StructuredLLMResult,
)


T = TypeVar("T")


class ResearchLimits(ResearchModel):
    max_steps: int = Field(default=5, ge=1, le=100, strict=True)
    max_tool_calls: int = Field(default=5, ge=0, le=100, strict=True)
    max_consecutive_failures: int = Field(default=2, ge=1, le=100, strict=True)
    max_repeated_proposals: int = Field(default=2, ge=1, le=100, strict=True)
    max_llm_requests: int = Field(default=8, ge=1, le=200, strict=True)
    runtime_seconds: float = Field(default=180, gt=0, le=3600, allow_inf_nan=False, strict=True)


class DeadlineExceeded(Exception):
    """The run's elapsed-time limit was reached."""


class LLMRequestBudgetExceeded(Exception):
    """No further requests may be sent to the language model."""


class RunBudget:
    def __init__(self, limits: ResearchLimits, clock: Callable[[], float] = monotonic):
        self.limits = limits
        self.clock = clock
        self.started = clock()
        self._llm_requests = 0
        self._lock = Lock()

    @property
    def elapsed(self) -> float:
        return self.clock() - self.started

    @property
    def llm_requests(self) -> int:
        with self._lock:
            return self._llm_requests

    def remaining_time(self) -> float:
        remaining = self.limits.runtime_seconds - self.elapsed
        if remaining <= 0:
            raise DeadlineExceeded
        return remaining

    def reserve_llm_request(self) -> float:
        with self._lock:
            remaining = self.remaining_time()
            if self._llm_requests >= self.limits.max_llm_requests:
                raise LLMRequestBudgetExceeded
            self._llm_requests += 1
            return remaining

    def call(self, operation: Callable[[], T]) -> T:
        """Bound the wait and ignore any late result from a synchronous provider."""
        self.remaining_time()
        replies: Queue[tuple[bool, T | BaseException]] = Queue(maxsize=1)

        def invoke() -> None:
            try:
                self.remaining_time()
                replies.put((True, operation()))
            except BaseException as exc:
                replies.put((False, exc))

        Thread(target=invoke, daemon=True).start()
        try:
            succeeded, value = replies.get(timeout=self.remaining_time())
        except Empty:
            raise DeadlineExceeded from None
        self.remaining_time()
        if not succeeded:
            assert isinstance(value, BaseException)
            raise value
        # The queue carries a T whenever the worker reports success.
        return value  # type: ignore[return-value]


@runtime_checkable
class _AttemptAwareProvider(Protocol):
    def generate_with_budget(
        self, prompt: str, schema: type[DecisionT], *, before_attempt: Callable[[], float],
    ) -> StructuredLLMResult[DecisionT]: ...


class BudgetedLLM:
    def __init__(self, provider: LLMProvider, budget: RunBudget):
        self.provider = provider
        self.budget = budget
        self.performance: list[LLMPerformance] = []

    def generate(self, prompt: str, schema: type[DecisionT]) -> StructuredLLMResult[DecisionT]:
        provider = self.provider

        def request() -> StructuredLLMResult[DecisionT]:
            if isinstance(provider, _AttemptAwareProvider):
                return provider.generate_with_budget(prompt, schema, before_attempt=self.budget.reserve_llm_request)
            self.budget.reserve_llm_request()
            return provider.generate(prompt, schema)

        try:
            result = self.budget.call(request)
        except LLMError as exc:
            self.performance.append(exc.performance)
            raise
        self.performance.append(result.performance)
        return result
