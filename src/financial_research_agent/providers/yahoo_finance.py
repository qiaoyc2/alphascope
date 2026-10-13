"""Yahoo Finance daily history normalized behind the market-data protocol."""

from __future__ import annotations

import math
from datetime import timedelta
from numbers import Real
from urllib.parse import quote

import pandas as pd
import yfinance as yf
from curl_cffi.requests.exceptions import RequestException as CurlRequestException
from pydantic import ValidationError
from yfinance.exceptions import (
    YFDataException,
    YFPricesMissingError,
    YFRateLimitError,
    YFTzMissingError,
)

from financial_research_agent.models import utc_now
from financial_research_agent.providers.market_data import (
    MarketDataError,
    MarketDataErrorCode,
    PriceBar,
    PriceHistory,
    PriceHistoryRequest,
)


def _numeric(value: object, field: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MarketDataError(MarketDataErrorCode.INVALID_PROVIDER_DATA, f"Invalid {field} value")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0) or (not positive and number < 0):
        raise MarketDataError(MarketDataErrorCode.INVALID_PROVIDER_DATA, f"Invalid {field} value")
    return number


class YahooFinanceProvider:
    name = "yahoo_finance_via_yfinance"
    adjustment_policy = "raw_ohlc_with_separate_split_dividend_adjusted_close"

    def __init__(self, *, timeout_seconds: float = 10.0):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self.timeout_seconds = timeout_seconds

    def get_price_history(self, request: PriceHistoryRequest) -> PriceHistory:
        # Revalidate because the provider can also be called without the tool.
        request = PriceHistoryRequest.model_validate(request)
        ticker = yf.Ticker(request.ticker)
        try:
            frame = ticker.history(
                start=request.start_date.isoformat(),
                end=(request.end_date + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=False,
                keepna=True,
                rounding=False,
                timeout=self.timeout_seconds,
                raise_errors=True,
            )
            if not isinstance(frame, pd.DataFrame):
                raise MarketDataError(
                    MarketDataErrorCode.INVALID_PROVIDER_DATA, "Provider returned a non-tabular history"
                )
            if frame.empty:
                raise MarketDataError(
                    MarketDataErrorCode.NO_DATA,
                    "No daily prices were returned for this ticker and date range",
                )
            metadata = ticker.history_metadata
        except YFRateLimitError as exc:
            raise MarketDataError(
                MarketDataErrorCode.RATE_LIMITED, "Yahoo Finance rate limited the request", retryable=True
            ) from exc
        except (YFPricesMissingError, YFTzMissingError) as exc:
            raise MarketDataError(
                MarketDataErrorCode.NO_DATA, "Ticker or requested history is unavailable"
            ) from exc
        except (CurlRequestException, TimeoutError, ConnectionError) as exc:
            raise MarketDataError(
                MarketDataErrorCode.PROVIDER_UNAVAILABLE,
                "Yahoo Finance request failed or timed out",
                retryable=True,
            ) from exc
        except YFDataException as exc:
            raise MarketDataError(
                MarketDataErrorCode.INVALID_PROVIDER_DATA, "Yahoo Finance returned invalid data"
            ) from exc

        required = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if isinstance(frame.columns, pd.MultiIndex) or not required.issubset(frame.columns):
            raise MarketDataError(
                MarketDataErrorCode.INVALID_PROVIDER_DATA, "Daily history is missing required columns"
            )
        bars: list[PriceBar] = []
        for index_value, row in frame.iterrows():
            try:
                trading_date = pd.Timestamp(index_value).date()
                volume = _numeric(row["Volume"], "Volume", positive=False)
                if not volume.is_integer():
                    raise MarketDataError(
                        MarketDataErrorCode.INVALID_PROVIDER_DATA, "Volume is not an integer"
                    )
                bars.append(
                    PriceBar(
                        trading_date=trading_date,
                        open=_numeric(row["Open"], "Open"),
                        high=_numeric(row["High"], "High"),
                        low=_numeric(row["Low"], "Low"),
                        close=_numeric(row["Close"], "Close"),
                        adjusted_close=_numeric(row["Adj Close"], "Adj Close"),
                        volume=int(volume),
                    )
                )
            except (ValueError, TypeError, ValidationError) as exc:
                raise MarketDataError(
                    MarketDataErrorCode.INVALID_PROVIDER_DATA, "Daily history has an invalid row"
                ) from exc

        bars.sort(key=lambda bar: bar.trading_date)
        currency = metadata.get("currency") if isinstance(metadata, dict) else None
        if not isinstance(currency, str) or not currency.strip():
            currency = None
        source_url = f"https://finance.yahoo.com/quote/{quote(request.ticker, safe='')}/history/"
        try:
            return PriceHistory(
                ticker=request.ticker,
                requested_start_date=request.start_date,
                requested_end_date=request.end_date,
                actual_start_date=bars[0].trading_date,
                actual_end_date=bars[-1].trading_date,
                bars=bars,
                currency=currency,
                provider=self.name,
                source_url=source_url,
                retrieved_at=utc_now(),
                adjustment_policy=self.adjustment_policy,
            )
        except ValidationError as exc:
            raise MarketDataError(
                MarketDataErrorCode.INVALID_PROVIDER_DATA, "Daily history failed validation"
            ) from exc
