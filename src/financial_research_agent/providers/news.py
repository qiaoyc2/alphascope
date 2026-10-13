"""Provider-neutral news metadata contracts."""

from __future__ import annotations

from datetime import date
from typing import Protocol, Self

from pydantic import AwareDatetime, Field, HttpUrl, field_validator, model_validator

from financial_research_agent.models import NonEmptyStr, ResearchModel


class NewsRequest(ResearchModel):
    query: str = Field(min_length=1, max_length=200)
    start_date: date | None = None
    end_date: date | None = None
    limit: int = Field(default=10, ge=1, le=50)

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def check_dates(self) -> Self:
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("invalid date range")
        if not self.query.strip():
            raise ValueError("query cannot be blank")
        return self


class NewsArticle(ResearchModel):
    article_id: NonEmptyStr
    title: NonEmptyStr
    publisher: NonEmptyStr | None = None
    published_at: AwareDatetime | None = None
    publication_date_status: NonEmptyStr
    retrieved_at: AwareDatetime
    url: HttpUrl  # Provider-reported item URL; source_type identifies its role.
    original_url: HttpUrl | None = None
    feed_url: HttpUrl  # RSS feed or provider query endpoint, retained for compatibility.
    publication_date: date | None = None
    description: str | None = None
    query: NonEmptyStr
    language: str | None = None
    provider: NonEmptyStr
    source_type: str = "rss_headline"


class NewsSearchResult(ResearchModel):
    query: NonEmptyStr
    feed_url: HttpUrl  # RSS feed or provider query endpoint.
    retrieved_at: AwareDatetime
    articles: list[NewsArticle]
    limitations: list[NonEmptyStr] = Field(default_factory=list)
    duplicates_removed: int = Field(default=0, ge=0)
    invalid_items_skipped: int = Field(default=0, ge=0)
    api_call_count: int = Field(ge=0)
    provider: NonEmptyStr


class NewsProvider(Protocol):
    def search_news(self, request: NewsRequest) -> NewsSearchResult: ...
