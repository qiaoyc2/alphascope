"""Assess price premises conditionally on matching observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import combinations
from math import isclose

from pydantic import Field

from financial_research_agent.models import (
    MarketMovement, Observation, ObservationProvenance, PricePremise, ResearchModel, ResearchState, ToolResult,
)
from financial_research_agent.providers.market_data import PriceBasis, PriceHistory
from financial_research_agent.tools import ToolName
from financial_research_agent.tools.calculator import CalculationError, percentage_return, relative_performance
from financial_research_agent.tools.market_data import BenchmarkComparison


class PremiseAssessment(ResearchModel):
    conditional_valid: bool | None = None
    observations_conflict: bool = False
    observation_ids: list[str] = Field(default_factory=list)
    provenance: list[ObservationProvenance] = Field(default_factory=list)
    independently_verified: bool = False
    summary: str | None = None


@dataclass(frozen=True)
class MovementReading:
    observation: Observation
    measurement: MarketMovement
    price_basis: PriceBasis | None = None
    effective_dates: tuple[date, date] | None = None
    adjustment_policy: str | None = None

    @property
    def scope(self) -> tuple:
        item = self.measurement
        return (
            item.ticker.casefold(), item.start_date, item.end_date, self.effective_dates,
            self.price_basis, self.adjustment_policy, (item.benchmark or "").casefold(),
        )


def market_reading(observation: Observation, result: ToolResult) -> MovementReading | None:
    """Recompute endpoint returns using the registered market result contract."""
    if not result.success or result.tool_name != ToolName.MARKET_DATA:
        return None
    kind = result.data.get("kind")
    data = {key: value for key, value in result.data.items() if key != "kind"}
    if kind == "benchmark_comparison":
        item = BenchmarkComparison.model_validate(data)
        stock = percentage_return(item.stock_start_price, item.stock_end_price)
        benchmark = percentage_return(item.benchmark_start_price, item.benchmark_end_price)
        expected = (stock, benchmark, relative_performance(stock, benchmark))
        reported = (item.stock_return_pct, item.benchmark_return_pct, item.relative_performance_percentage_points)
        if not (item.requested_start_date <= item.effective_start_date
                < item.effective_end_date <= item.requested_end_date
                and all(isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(expected, reported))):
            raise ValueError("inconsistent benchmark endpoints or returns")
        movement = MarketMovement(
            ticker=item.stock_ticker, start_date=item.requested_start_date, end_date=item.requested_end_date,
            return_pct=stock, benchmark=item.benchmark_ticker, benchmark_return_pct=benchmark,
        )
        return MovementReading(
            observation, movement, item.price_basis, (item.effective_start_date, item.effective_end_date),
            item.adjustment_policy if item.price_basis is PriceBasis.ADJUSTED_CLOSE else None,
        )
    if kind == "price_history":
        history = PriceHistory.model_validate(data)
        if len(history.bars) >= 2:
            movement = MarketMovement(
                ticker=history.ticker, start_date=history.requested_start_date, end_date=history.requested_end_date,
                return_pct=percentage_return(history.bars[0].adjusted_close, history.bars[-1].adjusted_close),
            )
            return MovementReading(
                observation, movement, PriceBasis.ADJUSTED_CLOSE,
                (history.actual_start_date, history.actual_end_date), history.adjustment_policy,
            )
    return None


def recorded_movements(state: ResearchState) -> list[MovementReading]:
    """Use typed supplied measurements or adapt the existing market tool contracts."""
    result_map = {result.result_id: result for result in state.tool_results}
    readings = []
    for observation in state.observations:
        result = result_map[observation.result_id]
        if not result.success:
            continue
        if observation.market_movement is not None:
            readings.append(MovementReading(observation, observation.market_movement))
            continue
        try:
            reading = market_reading(observation, result)
            if reading is not None:
                readings.append(reading)
        except ValueError:
            # Legacy or unstructured results are still available as text in the planner context.
            continue
    return readings


def comparable_conflicts(readings: list[MovementReading]) -> list[tuple[MovementReading, MovementReading]]:
    def disagree(left: float, right: float) -> bool:
        return (left > 0) != (right > 0) or (left < 0) != (right < 0) or not isclose(
            left, right, rel_tol=0, abs_tol=0.01,
        )

    conflicts = []
    for left, right in combinations(readings, 2):
        if left.scope != right.scope:
            continue
        a, b = left.measurement, right.measurement
        if disagree(a.return_pct, b.return_pct) or (
            a.benchmark_return_pct is not None and b.benchmark_return_pct is not None
            and (disagree(a.benchmark_return_pct, b.benchmark_return_pct)
                 or disagree(a.return_pct - a.benchmark_return_pct, b.return_pct - b.benchmark_return_pct))
        ):
            conflicts.append((left, right))
    return conflicts


def scoped_readings(state: ResearchState, readings: list[MovementReading]) -> list[MovementReading]:
    goal = state.goal
    if not (goal.ticker and goal.start_date and goal.end_date):
        return []
    relative = goal.price_premise in {PricePremise.OUTPERFORM, PricePremise.UNDERPERFORM}
    requires_benchmark = relative or (goal.price_premise is None and goal.benchmark is not None)
    return [
        reading for reading in readings
        if reading.measurement.ticker.casefold() == goal.ticker.casefold()
        and (reading.measurement.start_date, reading.measurement.end_date) == (goal.start_date, goal.end_date)
        and (not requires_benchmark or (
            (reading.measurement.benchmark or "").casefold() == (goal.benchmark or "").casefold()
            and reading.measurement.benchmark_return_pct is not None
        ))
    ]


def assess_premise(state: ResearchState, *, readings: list[MovementReading] | None = None) -> PremiseAssessment:
    goal = state.goal
    relevant = scoped_readings(state, recorded_movements(state) if readings is None else readings)
    if not relevant:
        return PremiseAssessment()
    first = relevant[0].measurement
    conflict = bool(comparable_conflicts(relevant))
    relative = goal.price_premise in {PricePremise.OUTPERFORM, PricePremise.UNDERPERFORM}
    summary = f"{first.ticker} returned {first.return_pct:+g}% from {first.start_date} to {first.end_date} (supplied period)."
    comparative = next((item.measurement for item in relevant if item.measurement.benchmark is not None
                        and (not goal.benchmark or item.measurement.benchmark.casefold() == goal.benchmark.casefold())), None)
    relative_return = None
    if comparative is not None and comparative.benchmark_return_pct is not None:
        try:
            relative_return = relative_performance(comparative.return_pct, comparative.benchmark_return_pct)
        except CalculationError:
            comparative = None
        if comparative is not None:
            summary += (
                f" Against {comparative.benchmark} ({comparative.benchmark_return_pct:+g}%), "
                f"{comparative.ticker} returned {comparative.return_pct:+g}% and its relative return was "
                f"{relative_return:+.2f} percentage points."
            )
    if conflict:
        summary = f"Supplied measurements for {goal.ticker} and the same period disagree."
    conditional = None
    if not conflict and len({reading.scope for reading in relevant}) == 1 and goal.price_premise is not None:
        measured = relative_return if relative else first.return_pct
        if measured is not None:
            increasing = goal.price_premise in {PricePremise.RISE, PricePremise.OUTPERFORM}
            conditional = measured > 0 if increasing else measured < 0
    return PremiseAssessment(
        conditional_valid=conditional,
        observations_conflict=conflict,
        observation_ids=[item.observation.observation_id for item in relevant],
        provenance=list(dict.fromkeys(item.observation.provenance for item in relevant)),
        independently_verified=not conflict and any(item.observation.independently_verified for item in relevant),
        summary=summary,
    )
