from datetime import date, datetime, timezone

import pytest

from financial_research_agent.providers.market_data import (
    MarketDataError,
    MarketDataErrorCode,
    PriceBar,
    PriceBasis,
    PriceHistory,
    PriceHistoryRequest,
)
from financial_research_agent.tools.market_data import MarketDataTool, compare_histories


START = date(2025, 1, 2)
END = date(2025, 1, 8)


def history(ticker: str, values: list[tuple[date, float, float]]) -> PriceHistory:
    return PriceHistory(
        ticker=ticker,
        requested_start_date=START,
        requested_end_date=END,
        actual_start_date=values[0][0],
        actual_end_date=values[-1][0],
        bars=[
            PriceBar(
                trading_date=day,
                open=raw,
                high=raw + 1,
                low=raw - 1,
                close=raw,
                adjusted_close=adjusted,
                volume=1000,
            )
            for day, raw, adjusted in values
        ],
        currency="USD",
        provider="fixture_provider",
        source_url=f"https://example.org/history/{ticker}",
        retrieved_at=datetime(2025, 1, 9, tzinfo=timezone.utc),
        adjustment_policy="raw_ohlc_with_separate_split_dividend_adjusted_close",
    )


class FakeProvider:
    def __init__(self, histories: dict[str, PriceHistory | MarketDataError]):
        self.histories = histories
        self.calls: list[str] = []

    def get_price_history(self, request: PriceHistoryRequest) -> PriceHistory:
        self.calls.append(request.ticker)
        result = self.histories[request.ticker]
        if isinstance(result, MarketDataError):
            raise result
        return result


def test_comparison_uses_common_dates_and_percentage_points() -> None:
    stock = history(
        "GOOGL",
        [(START, 100, 100), (date(2025, 1, 3), 110, 110), (date(2025, 1, 6), 121, 121)],
    )
    benchmark = history(
        "QQQ", [(date(2025, 1, 3), 200, 200), (date(2025, 1, 6), 210, 210)],
    )

    result = MarketDataTool(FakeProvider({"GOOGL": stock, "QQQ": benchmark})).compare_performance(
        action_id="A1", ticker="GOOGL", benchmark="QQQ", start_date=START, end_date=END
    )

    assert result.success
    assert result.metadata.api_call_count == 2
    assert len(result.sources) == 2
    assert result.data["effective_start_date"] == "2025-01-03"
    assert result.data["effective_end_date"] == "2025-01-06"
    assert result.data["common_session_count"] == 2
    assert result.data["stock_return_pct"] == pytest.approx(10)
    assert result.data["benchmark_return_pct"] == pytest.approx(5)
    assert result.data["relative_performance_percentage_points"] == pytest.approx(5)


def test_adjusted_and_raw_returns_are_not_mixed() -> None:
    stock = history("GOOGL", [(START, 100, 50), (date(2025, 1, 6), 50, 50)])
    benchmark = history("QQQ", [(START, 100, 100), (date(2025, 1, 6), 100, 100)])

    adjusted = compare_histories(stock, benchmark)
    raw = compare_histories(stock, benchmark, PriceBasis.RAW_CLOSE)

    assert adjusted.price_basis is PriceBasis.ADJUSTED_CLOSE
    assert adjusted.stock_return_pct == 0
    assert raw.stock_return_pct == -50
    assert raw.relative_performance_percentage_points == -50


def test_no_common_endpoints_fails_without_substituting_benchmark() -> None:
    stock = history("GOOGL", [(START, 100, 100), (date(2025, 1, 3), 110, 110)])
    benchmark = history("QQQ", [(date(2025, 1, 3), 200, 200), (date(2025, 1, 6), 210, 210)])
    with pytest.raises(MarketDataError) as error:
        compare_histories(stock, benchmark)
    assert error.value.code is MarketDataErrorCode.NO_OVERLAP


def test_unavailable_requested_benchmark_is_reported() -> None:
    stock = history("GOOGL", [(START, 100, 100), (date(2025, 1, 6), 110, 110)])
    provider = FakeProvider(
        {
            "GOOGL": stock,
            "QQQ": MarketDataError(MarketDataErrorCode.NO_DATA, "Benchmark unavailable"),
        }
    )
    result = MarketDataTool(provider).compare_performance(
        action_id="A1", ticker="GOOGL", benchmark="QQQ", start_date=START, end_date=END
    )

    assert provider.calls == ["GOOGL", "QQQ"]
    assert not result.success
    assert result.error.code == "NO_DATA"
    assert result.metadata.api_call_count == 2


def test_provider_must_return_requested_ticker() -> None:
    wrong = history("MSFT", [(START, 100, 100), (date(2025, 1, 6), 110, 110)])
    result = MarketDataTool(FakeProvider({"GOOGL": wrong})).get_history(
        action_id="A1", ticker="GOOGL", start_date=START, end_date=END
    )
    assert result.error.code == "INVALID_PROVIDER_DATA"


def test_comparison_rejects_inconsistent_adjustment_policies() -> None:
    stock = history("GOOGL", [(START, 100, 100), (date(2025, 1, 6), 110, 110)])
    benchmark = history("QQQ", [(START, 100, 100), (date(2025, 1, 6), 105, 105)])
    benchmark.adjustment_policy = "raw_close_only"

    with pytest.raises(MarketDataError) as error:
        compare_histories(stock, benchmark)
    assert error.value.code is MarketDataErrorCode.INVALID_PROVIDER_DATA
