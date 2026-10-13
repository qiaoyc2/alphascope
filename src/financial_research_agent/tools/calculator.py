"""Deterministic calculations over caller-supplied financial observations."""

from __future__ import annotations

import math
import statistics
from datetime import date, datetime
from enum import StrEnum
from numbers import Real
from typing import Annotated, Literal, Mapping, Self, Sequence
from uuid import uuid4

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from financial_research_agent.models import (
    ExecutionMetadata,
    ResearchAction,
    ResearchModel,
    ToolError,
    ToolResult,
    utc_now,
)


class CalculationErrorCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NO_OVERLAP = "NO_OVERLAP"
    CONSTANT_SERIES = "CONSTANT_SERIES"
    INVALID_RESULT = "INVALID_RESULT"


class CalculationError(ValueError):
    def __init__(self, code: CalculationErrorCode, message: str):
        super().__init__(message)
        self.code = code


def _finite_number(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CalculationError(CalculationErrorCode.INVALID_INPUT, f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalculationError(CalculationErrorCode.INVALID_INPUT, f"{name} must be numeric") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise CalculationError(CalculationErrorCode.INVALID_INPUT, f"{name} must be finite and valid")
    return number


def _finite_result(value: float) -> float:
    if not math.isfinite(value):
        raise CalculationError(CalculationErrorCode.INVALID_RESULT, "Calculation is not finite")
    return value


def _prices(prices: Sequence[float], minimum: int) -> list[float]:
    if len(prices) < minimum:
        raise CalculationError(
            CalculationErrorCode.INSUFFICIENT_DATA, f"At least {minimum} prices are required"
        )
    return [_finite_number(value, "price", positive=True) for value in prices]


def percentage_return(starting_price: float, ending_price: float) -> float:
    """Simple endpoint return, in percent, from prices on the same basis."""
    start = _finite_number(starting_price, "starting_price", positive=True)
    end = _finite_number(ending_price, "ending_price", positive=True)
    return _finite_result((end - start) / start * 100.0)


def relative_performance(stock_return_pct: float, benchmark_return_pct: float) -> float:
    """Stock minus benchmark return, in percentage points."""
    stock = _finite_number(stock_return_pct, "stock_return_pct")
    benchmark = _finite_number(benchmark_return_pct, "benchmark_return_pct")
    return _finite_result(stock - benchmark)


def moving_average(prices: Sequence[float], window: int) -> list[float]:
    """Rolling arithmetic means, one value per complete window."""
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise CalculationError(CalculationErrorCode.INVALID_INPUT, "window must be a positive integer")
    values = _prices(prices, window)
    try:
        return [
            _finite_result(math.fsum(values[index : index + window]) / window)
            for index in range(len(values) - window + 1)
        ]
    except OverflowError as exc:
        raise CalculationError(CalculationErrorCode.INVALID_RESULT, "Moving average overflowed") from exc


def historical_volatility(
    prices: Sequence[float], *, annualize: bool = True, trading_periods_per_year: int = 252
) -> float:
    """Sample SD of adjacent simple daily returns, in percent.

    Annualized result multiplies the daily sample SD by sqrt(trading_periods_per_year).
    """
    if (
        isinstance(trading_periods_per_year, bool)
        or not isinstance(trading_periods_per_year, int)
        or trading_periods_per_year < 1
    ):
        raise CalculationError(
            CalculationErrorCode.INVALID_INPUT, "trading_periods_per_year must be positive"
        )
    values = _prices(prices, 3)
    returns = [values[index] / values[index - 1] - 1.0 for index in range(1, len(values))]
    try:
        daily_sd = statistics.stdev(returns)
    except OverflowError as exc:
        raise CalculationError(CalculationErrorCode.INVALID_RESULT, "Volatility overflowed") from exc
    return _finite_result(
        daily_sd * (math.sqrt(trading_periods_per_year) if annualize else 1.0) * 100.0
    )


def volume_change(current_volume: int, baseline_volumes: Sequence[int]) -> float:
    """Current volume versus the arithmetic mean of explicit prior volumes, in percent."""
    if isinstance(current_volume, bool) or not isinstance(current_volume, int) or current_volume < 0:
        raise CalculationError(CalculationErrorCode.INVALID_INPUT, "current_volume must be nonnegative")
    if not baseline_volumes:
        raise CalculationError(CalculationErrorCode.INSUFFICIENT_DATA, "Baseline volumes are required")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in baseline_volumes
    ):
        raise CalculationError(
            CalculationErrorCode.INVALID_INPUT, "Baseline volumes must be nonnegative integers"
        )
    try:
        baseline = statistics.fmean(baseline_volumes)
    except OverflowError as exc:
        raise CalculationError(CalculationErrorCode.INVALID_RESULT, "Volume baseline overflowed") from exc
    if not math.isfinite(baseline) or baseline <= 0:
        raise CalculationError(CalculationErrorCode.INVALID_INPUT, "Baseline mean must be positive")
    return _finite_result((current_volume / baseline - 1.0) * 100.0)


class CorrelationResult(ResearchModel):
    correlation: float = Field(ge=-1, le=1, allow_inf_nan=False)
    effective_start_date: date
    effective_end_date: date
    common_session_count: int = Field(ge=3)
    paired_return_count: int = Field(ge=2)


def return_correlation(
    stock_prices: Mapping[date, float], benchmark_prices: Mapping[date, float]
) -> CorrelationResult:
    """Pearson correlation of simple returns over consecutive common dates."""
    stock = {
        day: _finite_number(price, "stock price", positive=True)
        for day, price in stock_prices.items()
    }
    benchmark = {
        day: _finite_number(price, "benchmark price", positive=True)
        for day, price in benchmark_prices.items()
    }
    if any(type(day) is not date for day in stock.keys() | benchmark.keys()):
        raise CalculationError(CalculationErrorCode.INVALID_INPUT, "Price keys must be dates")
    common_dates = sorted(stock.keys() & benchmark.keys())
    if len(common_dates) < 3:
        raise CalculationError(
            CalculationErrorCode.INSUFFICIENT_DATA, "At least three common dates are required"
        )
    stock_returns = [
        stock[common_dates[index]] / stock[common_dates[index - 1]] - 1.0
        for index in range(1, len(common_dates))
    ]
    benchmark_returns = [
        benchmark[common_dates[index]] / benchmark[common_dates[index - 1]] - 1.0
        for index in range(1, len(common_dates))
    ]
    if statistics.pstdev(stock_returns) == 0 or statistics.pstdev(benchmark_returns) == 0:
        raise CalculationError(CalculationErrorCode.CONSTANT_SERIES, "Return series is constant")
    coefficient = _finite_result(statistics.correlation(stock_returns, benchmark_returns))
    return CorrelationResult(
        correlation=max(-1.0, min(1.0, coefficient)),
        effective_start_date=common_dates[0],
        effective_end_date=common_dates[-1],
        common_session_count=len(common_dates),
        paired_return_count=len(stock_returns),
    )


class _CalculationArgs(ResearchModel):
    """Optional provenance supplied when inputs came from a market-data result."""

    source_result_ids: list[str] = Field(default_factory=list)
    price_basis: Literal["raw_close", "adjusted_close"] | None = None
    effective_start_date: date | None = None
    effective_end_date: date | None = None

    @model_validator(mode="after")
    def check_context_dates(self) -> Self:
        if (self.effective_start_date is None) != (self.effective_end_date is None):
            raise ValueError("Both effective dates must be supplied together")
        if (
            self.effective_start_date is not None
            and self.effective_start_date > self.effective_end_date
        ):
            raise ValueError("Effective start date is after end date")
        return self


class _PercentageReturnArgs(_CalculationArgs):
    operation: Literal["percentage_return"]
    starting_price: float
    ending_price: float


class _RelativePerformanceArgs(_CalculationArgs):
    operation: Literal["relative_performance"]
    stock_return_pct: float
    benchmark_return_pct: float


class _MovingAverageArgs(_CalculationArgs):
    operation: Literal["moving_average"]
    prices: list[float]
    window: int


class _HistoricalVolatilityArgs(_CalculationArgs):
    operation: Literal["historical_volatility"]
    prices: list[float]
    annualize: bool = True
    trading_periods_per_year: int = 252


class _VolumeChangeArgs(_CalculationArgs):
    operation: Literal["volume_change"]
    current_volume: int
    baseline_volumes: list[int]


class _ReturnCorrelationArgs(_CalculationArgs):
    operation: Literal["return_correlation"]
    stock_prices: dict[date, float]
    benchmark_prices: dict[date, float]


_CalculatorArgs = Annotated[
    _PercentageReturnArgs
    | _RelativePerformanceArgs
    | _MovingAverageArgs
    | _HistoricalVolatilityArgs
    | _VolumeChangeArgs
    | _ReturnCorrelationArgs,
    Field(discriminator="operation"),
]
_ARGUMENT_ADAPTER = TypeAdapter(_CalculatorArgs)


class CalculatorTool:
    name = "calculator"

    def execute(self, action: ResearchAction) -> ToolResult:
        if action.tool_name != self.name:
            raise ValueError("action is not addressed to the calculator")
        started_at = utc_now()
        try:
            args = _ARGUMENT_ADAPTER.validate_python(action.arguments)
        except ValidationError:
            return self._failure(
                action, "INVALID_ARGUMENTS", "Invalid calculator operation or arguments", started_at
            )

        try:
            if isinstance(args, _PercentageReturnArgs):
                value = percentage_return(args.starting_price, args.ending_price)
                output = {
                    "value": value,
                    "unit": "percent",
                    "formula": "(end - start) / start * 100",
                }
            elif isinstance(args, _RelativePerformanceArgs):
                value = relative_performance(args.stock_return_pct, args.benchmark_return_pct)
                output = {
                    "value": value,
                    "unit": "percentage_points",
                    "formula": "stock_return_pct - benchmark_return_pct",
                }
            elif isinstance(args, _MovingAverageArgs):
                output = {
                    "values": moving_average(args.prices, args.window),
                    "unit": "input_price_units",
                    "formula": "arithmetic mean of each complete window",
                }
            elif isinstance(args, _HistoricalVolatilityArgs):
                value = historical_volatility(
                    args.prices,
                    annualize=args.annualize,
                    trading_periods_per_year=args.trading_periods_per_year,
                )
                output = {
                    "value": value,
                    "unit": "annualized_percent" if args.annualize else "periodic_percent",
                    "formula": (
                        "sample_sd(adjacent_simple_returns) * sqrt(periods_per_year) * 100"
                        if args.annualize
                        else "sample_sd(adjacent_simple_returns) * 100"
                    ),
                }
            elif isinstance(args, _VolumeChangeArgs):
                output = {
                    "value": volume_change(args.current_volume, args.baseline_volumes),
                    "unit": "percent",
                    "formula": "(current / mean(baseline) - 1) * 100",
                }
            else:
                correlation = return_correlation(args.stock_prices, args.benchmark_prices)
                output = correlation.model_dump(mode="json")
                output.update(
                    {
                        "unit": "correlation_coefficient",
                        "formula": "Pearson correlation of aligned simple returns",
                    }
                )
        except CalculationError as exc:
            return self._failure(action, exc.code.value, str(exc), started_at)

        return ToolResult(
            result_id=str(uuid4()),
            action_id=action.action_id,
            tool_name=self.name,
            success=True,
            data={
                "operation": args.operation,
                "inputs": args.model_dump(mode="json", exclude_none=True),
                **output,
            },
            metadata=ExecutionMetadata(started_at=started_at, finished_at=utc_now()),
        )

    def _failure(self, action: ResearchAction, code: str, message: str, started_at: datetime) -> ToolResult:
        return ToolResult(
            result_id=str(uuid4()),
            action_id=action.action_id,
            tool_name=self.name,
            success=False,
            error=ToolError(code=code, message=message),
            metadata=ExecutionMetadata(started_at=started_at, finished_at=utc_now()),
        )
