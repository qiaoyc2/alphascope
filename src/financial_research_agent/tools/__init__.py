"""Callable research tools."""

from enum import StrEnum


class ToolName(StrEnum):
    MARKET_DATA = "market_data"
    CALCULATOR = "calculator"
    SEC_FILINGS = "sec_filings"
    NEWS = "news"
