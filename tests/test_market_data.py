from datetime import date

import pandas as pd
import pytest
from curl_cffi.requests.exceptions import Timeout
from yfinance.exceptions import YFRateLimitError, YFTzMissingError

from financial_research_agent.agent.state import add_action, add_observation, record_tool_result
from financial_research_agent.models import Observation, ResearchAction, ResearchGoal, ResearchState
from financial_research_agent.providers.market_data import PriceHistoryRequest
from financial_research_agent.providers.yahoo_finance import YahooFinanceProvider
from financial_research_agent.tools.market_data import MarketDataTool


START = date(2025, 1, 3)  # Friday
END = date(2025, 1, 6)  # Monday; the weekend has no fabricated rows


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [102.0, 100.0],
            "High": [104.0, 102.0],
            "Low": [99.0, 98.0],
            "Close": [103.0, 101.0],
            "Adj Close": [101.5, 99.5],
            "Volume": [1200, 1000],
        },
        index=pd.to_datetime(["2025-01-06", "2025-01-03"]),
    )


class FakeTicker:
    def __init__(self, data: pd.DataFrame | Exception):
        self.data = data
        self.history_metadata = {"currency": "USD"}
        self.kwargs: dict[str, object] = {}

    def history(self, **kwargs: object) -> pd.DataFrame:
        self.kwargs = kwargs
        if isinstance(self.data, Exception):
            raise self.data
        return self.data


def patch_ticker(monkeypatch: pytest.MonkeyPatch, ticker: FakeTicker) -> None:
    monkeypatch.setattr(
        "financial_research_agent.providers.yahoo_finance.yf.Ticker", lambda symbol: ticker
    )


def test_yahoo_normalizes_unsorted_sessions_and_uses_exclusive_provider_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticker = FakeTicker(frame())
    patch_ticker(monkeypatch, ticker)
    history = YahooFinanceProvider().get_price_history(
        PriceHistoryRequest(ticker=" googl ", start_date=START, end_date=END)
    )

    assert ticker.kwargs["start"] == "2025-01-03"
    assert ticker.kwargs["end"] == "2025-01-07"
    assert ticker.kwargs["auto_adjust"] is False
    assert ticker.kwargs["timeout"] == 10.0
    assert [bar.trading_date for bar in history.bars] == [START, END]
    assert history.bars[0].close == 101.0
    assert history.bars[0].adjusted_close == 99.5
    assert history.currency == "USD"
    assert history.actual_start_date == START
    assert history.actual_end_date == END


def test_market_tool_result_is_m1_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_ticker(monkeypatch, FakeTicker(frame()))
    result = MarketDataTool(YahooFinanceProvider()).get_history(
        action_id="A1", ticker="GOOGL", start_date=START, end_date=END
    )

    assert result.success
    assert result.action_id == "A1"
    assert result.tool_name == "market_data"
    assert result.data["kind"] == "price_history"
    assert result.data["bars"][0]["trading_date"] == "2025-01-03"
    assert result.metadata.api_call_count == 1
    assert str(result.sources[0].url).startswith("https://finance.yahoo.com/")
    assert result.model_validate_json(result.model_dump_json()) == result

    state = add_action(
        ResearchState(goal=ResearchGoal(query="Did GOOGL decline?", objective="Verify movement")),
        ResearchAction(action_id="A1", tool_name="market_data", purpose="Fetch prices"),
    )
    state = record_tool_result(state, result)
    state = add_observation(
        state,
        Observation(
            observation_id="O1", action_id="A1", result_id=result.result_id, summary="Prices retrieved"
        ),
    )
    assert state.api_call_count == 1
    assert state.observations[0].result_id == result.result_id


@pytest.mark.parametrize(
    "ticker,start,end,code",
    [
        ("bad/ticker", START, END, "INVALID_TICKER"),
        ("GOOGL", END, START, "INVALID_DATE_RANGE"),
    ],
)
def test_invalid_requests_do_not_call_provider(
    ticker: str, start: date, end: date, code: str
) -> None:
    class NeverCalled:
        def get_price_history(self, request: PriceHistoryRequest):
            raise AssertionError("Provider should not be called")

    result = MarketDataTool(NeverCalled()).get_history(
        action_id="A1", ticker=ticker, start_date=start, end_date=end
    )
    assert not result.success
    assert result.error.code == code
    assert result.metadata.api_call_count == 0


def test_empty_history_is_not_a_success(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_ticker(monkeypatch, FakeTicker(pd.DataFrame()))
    result = MarketDataTool(YahooFinanceProvider()).get_history(
        action_id="A1", ticker="GOOGL", start_date=START, end_date=END
    )
    assert not result.success
    assert result.error.code == "NO_DATA"


@pytest.mark.parametrize(
    "exc,code,retryable",
    [
        (YFTzMissingError("DELISTED"), "NO_DATA", False),
        (Timeout("Timed out"), "PROVIDER_UNAVAILABLE", True),
        (YFRateLimitError(), "RATE_LIMITED", True),
    ],
)
def test_provider_failures_are_structured(
    monkeypatch: pytest.MonkeyPatch, exc: Exception, code: str, retryable: bool
) -> None:
    patch_ticker(monkeypatch, FakeTicker(exc))
    result = MarketDataTool(YahooFinanceProvider()).get_history(
        action_id="A1", ticker="GOOGL", start_date=START, end_date=END
    )
    assert not result.success
    assert result.error.code == code
    assert result.error.retryable is retryable
    assert result.metadata.api_call_count == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.drop(columns="Adj Close"),
        lambda data: data.assign(Close=float("nan")),
        lambda data: data.assign(Close=0.0),
        lambda data: data.assign(Volume=-1),
        lambda data: data.set_axis(pd.to_datetime(["2025-01-03", "2025-01-03"])),
    ],
)
def test_invalid_provider_rows_are_rejected(
    monkeypatch: pytest.MonkeyPatch, change
) -> None:
    patch_ticker(monkeypatch, FakeTicker(change(frame())))
    result = MarketDataTool(YahooFinanceProvider()).get_history(
        action_id="A1", ticker="GOOGL", start_date=START, end_date=END
    )
    assert not result.success
    assert result.error.code == "INVALID_PROVIDER_DATA"


def test_no_rows_on_weekend_only_request(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_ticker(monkeypatch, FakeTicker(pd.DataFrame()))
    result = MarketDataTool(YahooFinanceProvider()).get_history(
        action_id="A1",
        ticker="GOOGL",
        start_date=date(2025, 1, 4),
        end_date=date(2025, 1, 5),
    )
    assert result.error.code == "NO_DATA"


def test_holiday_gap_does_not_create_a_trading_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    data = frame().set_axis(pd.to_datetime(["2025-01-21", "2025-01-17"]))
    patch_ticker(monkeypatch, FakeTicker(data))
    result = MarketDataTool(YahooFinanceProvider()).get_history(
        action_id="A1",
        ticker="GOOGL",
        start_date=date(2025, 1, 17),
        end_date=date(2025, 1, 21),
    )
    assert [bar["trading_date"] for bar in result.data["bars"]] == [
        "2025-01-17",
        "2025-01-21",
    ]


def test_unexpected_provider_bug_stays_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_ticker(monkeypatch, FakeTicker(RuntimeError("programming bug")))
    with pytest.raises(RuntimeError, match="programming bug"):
        YahooFinanceProvider().get_price_history(
            PriceHistoryRequest(ticker="GOOGL", start_date=START, end_date=END)
        )
