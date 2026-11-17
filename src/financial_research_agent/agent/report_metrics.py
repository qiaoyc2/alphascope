"""Typed presentation of checked market observations and replayed arithmetic."""

from __future__ import annotations

from datetime import date
from math import isclose
from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, model_validator

from financial_research_agent.models import ResearchModel, ToolResult
from financial_research_agent.providers.market_data import PriceBasis, PriceHistory
from financial_research_agent.tools.calculator import percentage_return, relative_performance
from financial_research_agent.tools.market_data import BenchmarkComparison


class PriceEndpoints(ResearchModel):
    ticker: str
    currency: str | None
    start_price: FiniteFloat = Field(gt=0)
    end_price: FiniteFloat = Field(gt=0)
    return_pct: FiniteFloat

    @model_validator(mode="after")
    def check_return(self) -> Self:
        if not isclose(self.return_pct, percentage_return(self.start_price, self.end_price), abs_tol=1e-9, rel_tol=1e-9):
            raise ValueError("return does not match recorded endpoints")
        return self


class MarketMetric(ResearchModel):
    kind: Literal["market_return"] = "market_return"
    finding_id: str
    result_id: str
    requested_start_date: date
    requested_end_date: date
    effective_start_date: date
    effective_end_date: date
    session_count: int = Field(ge=2)
    price_basis: PriceBasis
    stock: PriceEndpoints
    benchmark: PriceEndpoints | None = None
    relative_performance_percentage_points: FiniteFloat | None = None

    @model_validator(mode="after")
    def check_scope_and_benchmark(self) -> Self:
        if not self.requested_start_date <= self.effective_start_date < self.effective_end_date <= self.requested_end_date:
            raise ValueError("effective sessions are outside the requested period")
        if self.benchmark is None:
            if self.relative_performance_percentage_points is not None:
                raise ValueError("relative performance requires a benchmark")
        elif self.relative_performance_percentage_points is None or not isclose(
            self.relative_performance_percentage_points,
            relative_performance(self.stock.return_pct, self.benchmark.return_pct), abs_tol=1e-9, rel_tol=1e-9,
        ):
            raise ValueError("relative performance must be the difference in percentage points")
        return self


class CalculationMetric(ResearchModel):
    kind: Literal["calculation"] = "calculation"
    finding_id: str
    result_id: str
    operation: str
    unit: Literal["percent", "percentage_points", "annualized_percent", "periodic_percent",
                  "correlation_coefficient", "input_price_units"]
    value: FiniteFloat | None
    formula: str
    source_result_ids: list[str]


ReportMetric = Annotated[MarketMetric | CalculationMetric, Field(discriminator="kind")]


def market_metric(finding_id: str, result: ToolResult) -> MarketMetric:
    """Project an observation already checked against the M5 market tool contract."""
    data = {key: value for key, value in result.data.items() if key != "kind"}
    if result.data["kind"] == "benchmark_comparison":
        comparison = BenchmarkComparison.model_validate(data)
        return MarketMetric(
            finding_id=finding_id, result_id=result.result_id,
            requested_start_date=comparison.requested_start_date, requested_end_date=comparison.requested_end_date,
            effective_start_date=comparison.effective_start_date, effective_end_date=comparison.effective_end_date,
            session_count=comparison.common_session_count, price_basis=comparison.price_basis,
            stock=PriceEndpoints(
                ticker=comparison.stock_ticker, currency=comparison.stock_currency,
                start_price=comparison.stock_start_price, end_price=comparison.stock_end_price,
                return_pct=comparison.stock_return_pct,
            ),
            benchmark=PriceEndpoints(
                ticker=comparison.benchmark_ticker, currency=comparison.benchmark_currency,
                start_price=comparison.benchmark_start_price, end_price=comparison.benchmark_end_price,
                return_pct=comparison.benchmark_return_pct,
            ),
            relative_performance_percentage_points=comparison.relative_performance_percentage_points,
        )
    history = PriceHistory.model_validate(data)
    first, last = history.bars[0].adjusted_close, history.bars[-1].adjusted_close
    return MarketMetric(
        finding_id=finding_id, result_id=result.result_id,
        requested_start_date=history.requested_start_date, requested_end_date=history.requested_end_date,
        effective_start_date=history.actual_start_date, effective_end_date=history.actual_end_date,
        session_count=len(history.bars), price_basis=PriceBasis.ADJUSTED_CLOSE,
        stock=PriceEndpoints(
            ticker=history.ticker, currency=history.currency, start_price=first, end_price=last,
            return_pct=percentage_return(first, last),
        ),
    )


def matched_calculation_source(calculation: ToolResult, market: ToolResult) -> bool:
    """Verify claimed operands against an already checked market result."""
    inputs = calculation.data["inputs"]
    if inputs.get("source_result_ids") != [market.result_id]:
        return False
    metric = market_metric("input", market)
    scope = {
        "price_basis": metric.price_basis.value,
        "effective_start_date": metric.effective_start_date.isoformat(),
        "effective_end_date": metric.effective_end_date.isoformat(),
    }
    if any(inputs.get(key) not in (None, value) for key, value in scope.items()):
        return False
    operation = calculation.data["operation"]
    if operation == "percentage_return":
        return (inputs["starting_price"], inputs["ending_price"]) == (metric.stock.start_price, metric.stock.end_price)
    if operation == "relative_performance" and metric.benchmark:
        return (inputs["stock_return_pct"], inputs["benchmark_return_pct"]) == (
            metric.stock.return_pct, metric.benchmark.return_pct,
        )
    if operation == "historical_volatility" and market.data["kind"] == "price_history":
        return inputs["prices"] == [bar["adjusted_close"] for bar in market.data["bars"]]
    return False
