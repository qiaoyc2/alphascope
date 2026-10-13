"""Provider-neutral daily market data contracts."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Protocol, Self

from pydantic import Field, HttpUrl, field_validator, model_validator

from financial_research_agent.models import NonEmptyStr, ResearchModel


class MarketDataErrorCode(StrEnum):
    INVALID_TICKER = "INVALID_TICKER"
    INVALID_DATE_RANGE = "INVALID_DATE_RANGE"
    NO_DATA = "NO_DATA"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    INVALID_PROVIDER_DATA = "INVALID_PROVIDER_DATA"
    NO_OVERLAP = "NO_OVERLAP"
    INVALID_PRICE_BASIS = "INVALID_PRICE_BASIS"


class MarketDataError(Exception):
    def __init__(self, code: MarketDataErrorCode, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class PriceBasis(StrEnum):
    RAW_CLOSE = "raw_close"
    ADJUSTED_CLOSE = "adjusted_close"


class PriceHistoryRequest(ResearchModel):
    """Both dates are inclusive at this boundary."""

    ticker: str = Field(min_length=1, max_length=20, pattern=r"^\^?[A-Z0-9][A-Z0-9.^=-]*$")
    start_date: date
    end_date: date

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def check_dates(self) -> Self:
        if self.start_date > self.end_date or self.end_date == date.max:
            raise ValueError("invalid inclusive date range")
        return self


class PriceBar(ResearchModel):
    trading_date: date
    open: float = Field(gt=0, allow_inf_nan=False)
    high: float = Field(gt=0, allow_inf_nan=False)
    low: float = Field(gt=0, allow_inf_nan=False)
    close: float = Field(gt=0, allow_inf_nan=False)
    adjusted_close: float = Field(gt=0, allow_inf_nan=False)
    volume: int = Field(ge=0)

    @model_validator(mode="after")
    def check_ohlc(self) -> Self:
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("OHLC values are inconsistent")
        if self.high < self.low:
            raise ValueError("high is below low")
        return self


class PriceHistory(ResearchModel):
    ticker: NonEmptyStr
    requested_start_date: date
    requested_end_date: date
    actual_start_date: date
    actual_end_date: date
    bars: list[PriceBar] = Field(min_length=1)
    currency: str | None = None
    provider: NonEmptyStr
    source_url: HttpUrl
    retrieved_at: datetime
    adjustment_policy: NonEmptyStr

    @model_validator(mode="after")
    def check_sessions(self) -> Self:
        dates = [bar.trading_date for bar in self.bars]
        if dates != sorted(set(dates)):
            raise ValueError("trading dates must be unique and chronological")
        if dates[0] != self.actual_start_date or dates[-1] != self.actual_end_date:
            raise ValueError("actual dates must match the first and last bars")
        if not (self.requested_start_date <= dates[0] <= dates[-1] <= self.requested_end_date):
            raise ValueError("trading dates fall outside the requested period")
        return self

    def closes(self, basis: PriceBasis) -> dict[date, float]:
        field = "adjusted_close" if basis is PriceBasis.ADJUSTED_CLOSE else "close"
        return {bar.trading_date: getattr(bar, field) for bar in self.bars}


class MarketDataProvider(Protocol):
    def get_price_history(self, request: PriceHistoryRequest) -> PriceHistory: ...
