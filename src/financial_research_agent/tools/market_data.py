"""Validated market-data requests and aligned benchmark comparisons."""

from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4

from pydantic import Field, HttpUrl, ValidationError

from financial_research_agent.models import (
    ExecutionMetadata,
    ResearchModel,
    SourceReference,
    ToolError,
    ToolResult,
    utc_now,
)
from financial_research_agent.providers.market_data import (
    MarketDataError,
    MarketDataErrorCode,
    MarketDataProvider,
    PriceBasis,
    PriceHistory,
    PriceHistoryRequest,
)
from financial_research_agent.tools.calculator import (
    CalculationError,
    percentage_return,
    relative_performance,
)


class BenchmarkComparison(ResearchModel):
    stock_ticker: str
    benchmark_ticker: str
    requested_start_date: date
    requested_end_date: date
    effective_start_date: date
    effective_end_date: date
    common_session_count: int = Field(ge=2)
    price_basis: PriceBasis
    stock_start_price: float = Field(gt=0, allow_inf_nan=False)
    stock_end_price: float = Field(gt=0, allow_inf_nan=False)
    benchmark_start_price: float = Field(gt=0, allow_inf_nan=False)
    benchmark_end_price: float = Field(gt=0, allow_inf_nan=False)
    stock_return_pct: float = Field(allow_inf_nan=False)
    benchmark_return_pct: float = Field(allow_inf_nan=False)
    relative_performance_percentage_points: float = Field(allow_inf_nan=False)
    stock_currency: str | None
    benchmark_currency: str | None
    stock_provider: str
    benchmark_provider: str
    stock_source_url: HttpUrl
    benchmark_source_url: HttpUrl
    stock_retrieved_at: datetime
    benchmark_retrieved_at: datetime


def compare_histories(
    stock: PriceHistory, benchmark: PriceHistory, basis: PriceBasis = PriceBasis.ADJUSTED_CLOSE
) -> BenchmarkComparison:
    """Compare endpoint returns on identical available sessions and price basis."""
    if (stock.requested_start_date, stock.requested_end_date) != (
        benchmark.requested_start_date,
        benchmark.requested_end_date,
    ):
        raise MarketDataError(
            MarketDataErrorCode.INVALID_PROVIDER_DATA, "Histories have different requested periods"
        )
    if stock.adjustment_policy != benchmark.adjustment_policy:
        raise MarketDataError(
            MarketDataErrorCode.INVALID_PROVIDER_DATA, "Histories use different adjustment policies"
        )
    stock_prices = stock.closes(basis)
    benchmark_prices = benchmark.closes(basis)
    common_dates = sorted(stock_prices.keys() & benchmark_prices.keys())
    if len(common_dates) < 2:
        raise MarketDataError(
            MarketDataErrorCode.NO_OVERLAP, "At least two common trading dates are required"
        )
    start, end = common_dates[0], common_dates[-1]
    try:
        stock_return = percentage_return(stock_prices[start], stock_prices[end])
        benchmark_return = percentage_return(benchmark_prices[start], benchmark_prices[end])
        relative_return = relative_performance(stock_return, benchmark_return)
    except CalculationError as exc:
        raise MarketDataError(
            MarketDataErrorCode.INVALID_PROVIDER_DATA, "Price return calculation is not finite"
        ) from exc
    return BenchmarkComparison(
        stock_ticker=stock.ticker,
        benchmark_ticker=benchmark.ticker,
        requested_start_date=stock.requested_start_date,
        requested_end_date=stock.requested_end_date,
        effective_start_date=start,
        effective_end_date=end,
        common_session_count=len(common_dates),
        price_basis=basis,
        stock_start_price=stock_prices[start],
        stock_end_price=stock_prices[end],
        benchmark_start_price=benchmark_prices[start],
        benchmark_end_price=benchmark_prices[end],
        stock_return_pct=stock_return,
        benchmark_return_pct=benchmark_return,
        relative_performance_percentage_points=relative_return,
        stock_currency=stock.currency,
        benchmark_currency=benchmark.currency,
        stock_provider=stock.provider,
        benchmark_provider=benchmark.provider,
        stock_source_url=stock.source_url,
        benchmark_source_url=benchmark.source_url,
        stock_retrieved_at=stock.retrieved_at,
        benchmark_retrieved_at=benchmark.retrieved_at,
    )


class MarketDataTool:
    name = "market_data"

    def __init__(self, provider: MarketDataProvider):
        self.provider = provider

    @staticmethod
    def _request(ticker: str, start_date: date, end_date: date) -> PriceHistoryRequest:
        try:
            return PriceHistoryRequest(
                ticker=ticker, start_date=start_date, end_date=end_date
            )
        except ValidationError as exc:
            code = (
                MarketDataErrorCode.INVALID_TICKER
                if any(error["loc"] == ("ticker",) for error in exc.errors())
                else MarketDataErrorCode.INVALID_DATE_RANGE
            )
            message = (
                "Ticker syntax is invalid"
                if code is MarketDataErrorCode.INVALID_TICKER
                else "Date range is invalid"
            )
            raise MarketDataError(code, message) from exc

    @staticmethod
    def _validate_response(history: object, request: PriceHistoryRequest) -> PriceHistory:
        try:
            validated = PriceHistory.model_validate(history)
        except ValidationError as exc:
            raise MarketDataError(
                MarketDataErrorCode.INVALID_PROVIDER_DATA, "Provider returned invalid history"
            ) from exc
        if (
            validated.ticker != request.ticker
            or validated.requested_start_date != request.start_date
            or validated.requested_end_date != request.end_date
        ):
            raise MarketDataError(
                MarketDataErrorCode.INVALID_PROVIDER_DATA, "Provider history does not match request"
            )
        return validated

    def get_history(
        self, *, action_id: str, ticker: str, start_date: date, end_date: date
    ) -> ToolResult:
        started_at = utc_now()
        api_calls = 0
        try:
            request = self._request(ticker, start_date, end_date)
            api_calls = 1
            history = self._validate_response(self.provider.get_price_history(request), request)
        except MarketDataError as exc:
            return self._failure(action_id, exc, started_at, api_calls)
        return ToolResult(
            result_id=str(uuid4()),
            action_id=action_id,
            tool_name=self.name,
            success=True,
            data={"kind": "price_history", **history.model_dump(mode="json")},
            sources=[SourceReference(url=history.source_url, title=f"{history.ticker} price history")],
            metadata=ExecutionMetadata(
                api_call_count=api_calls, started_at=started_at, finished_at=utc_now()
            ),
        )

    def compare_performance(
        self,
        *,
        action_id: str,
        ticker: str,
        benchmark: str,
        start_date: date,
        end_date: date,
        price_basis: PriceBasis = PriceBasis.ADJUSTED_CLOSE,
    ) -> ToolResult:
        started_at = utc_now()
        api_calls = 0
        try:
            stock_request = self._request(ticker, start_date, end_date)
            benchmark_request = self._request(benchmark, start_date, end_date)
            try:
                basis = PriceBasis(price_basis)
            except ValueError as exc:
                raise MarketDataError(
                    MarketDataErrorCode.INVALID_PRICE_BASIS, "Price basis is invalid"
                ) from exc
            api_calls = 1
            stock = self._validate_response(
                self.provider.get_price_history(stock_request), stock_request
            )
            api_calls = 2
            benchmark_history = self._validate_response(
                self.provider.get_price_history(benchmark_request), benchmark_request
            )
            comparison = compare_histories(stock, benchmark_history, basis)
        except MarketDataError as exc:
            return self._failure(action_id, exc, started_at, api_calls)
        return ToolResult(
            result_id=str(uuid4()),
            action_id=action_id,
            tool_name=self.name,
            success=True,
            data={"kind": "benchmark_comparison", **comparison.model_dump(mode="json")},
            sources=[
                SourceReference(url=stock.source_url, title=f"{stock.ticker} price history"),
                SourceReference(
                    url=benchmark_history.source_url,
                    title=f"{benchmark_history.ticker} price history",
                ),
            ],
            metadata=ExecutionMetadata(
                api_call_count=api_calls, started_at=started_at, finished_at=utc_now()
            ),
        )

    def _failure(
        self, action_id: str, error: MarketDataError, started_at: datetime, api_calls: int
    ) -> ToolResult:
        return ToolResult(
            result_id=str(uuid4()),
            action_id=action_id,
            tool_name=self.name,
            success=False,
            error=ToolError(
                code=error.code.value, message=str(error), retryable=error.retryable
            ),
            metadata=ExecutionMetadata(
                api_call_count=api_calls, started_at=started_at, finished_at=utc_now()
            ),
        )
