"""Validated news search wrapped in the existing ToolResult contract."""

from __future__ import annotations

from datetime import date
from uuid import uuid4

from pydantic import ValidationError

from financial_research_agent.models import ExecutionMetadata, SourceReference, ToolError, ToolResult, utc_now
from financial_research_agent.providers.http_client import HttpFailure
from financial_research_agent.providers.news import NewsProvider, NewsRequest, NewsSearchResult


class NewsTool:
    name = "news"

    def __init__(self, provider: NewsProvider):
        self.provider = provider

    def search_news(
        self, *, action_id: str, query: str, start_date: date | None = None,
        end_date: date | None = None, limit: int = 10,
    ) -> ToolResult:
        started_at = utc_now()
        try:
            request = NewsRequest(query=query, start_date=start_date, end_date=end_date, limit=limit)
        except ValidationError:
            return self._failure(action_id, "INVALID_REQUEST", "Invalid news query, date range, or limit", started_at)
        try:
            result = NewsSearchResult.model_validate(self.provider.search_news(request))
            if (result.query != request.query or len(result.articles) > request.limit or
                any(item.query != request.query or item.feed_url != result.feed_url or
                    item.provider != result.provider for item in result.articles)):
                raise HttpFailure("INVALID_PROVIDER_DATA", "News provider result does not match request")
        except HttpFailure as exc:
            return self._failure(action_id, exc.code, exc.message, started_at, exc.retryable, exc.request_count)
        except ValidationError:
            return self._failure(action_id, "INVALID_PROVIDER_DATA", "News provider returned invalid metadata", started_at)
        sources = [SourceReference(url=result.feed_url, title=f"{result.provider} query source")]
        sources.extend(SourceReference(url=item.url, title=item.title, published_at=item.published_at)
                       for item in result.articles)
        return ToolResult(
            result_id=str(uuid4()), action_id=action_id, tool_name=self.name, success=True,
            data={"kind": "news_search", **result.model_dump(mode="json")}, sources=sources,
            metadata=ExecutionMetadata(api_call_count=result.api_call_count, started_at=started_at, finished_at=utc_now()),
        )

    def _failure(self, action_id: str, code: str, message: str, started_at, retryable: bool = False, calls: int = 0) -> ToolResult:
        return ToolResult(result_id=str(uuid4()), action_id=action_id, tool_name=self.name, success=False,
                          error=ToolError(code=code, message=message, retryable=retryable),
                          metadata=ExecutionMetadata(api_call_count=calls, started_at=started_at, finished_at=utc_now()))
