"""Title search over the DOJ's documented public press-release API."""

from __future__ import annotations

import hashlib
import html
import json
import re
import threading
import time
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlencode, urlsplit

import requests
from pydantic import ValidationError

from financial_research_agent.models import utc_now
from financial_research_agent.providers.http_client import HttpFailure, fetch_bytes
from financial_research_agent.providers.news import NewsArticle, NewsRequest, NewsSearchResult


_ENDPOINT = "https://www.justice.gov/api/v1/press_releases.json"
_ALLOWED_HOSTS = frozenset({"www.justice.gov", "justice.gov"})
_PACE_LOCK = threading.Lock()
_NEXT_REQUEST_AT = 0.0


def _pace_request() -> None:
    """One DOJ request per second across provider instances in this process."""
    global _NEXT_REQUEST_AT
    with _PACE_LOCK:
        delay = max(0.0, _NEXT_REQUEST_AT - time.monotonic())
        if delay:
            time.sleep(delay)
        _NEXT_REQUEST_AT = time.monotonic() + 1.0


def _official_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        return (parsed.scheme == "https" and parsed.hostname in _ALLOWED_HOSTS
                and parsed.port is None and parsed.username is None)
    except ValueError:
        return False


class DoJPressReleaseProvider:
    """Only DOJ-issued release metadata; no publisher pages or article body."""

    name = "doj_press_releases"

    def __init__(
        self, *, session: requests.Session | None = None,
        timeout_seconds: float = 10.0,
        before_request: Callable[[], None] = _pace_request,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds
        self.before_request = before_request

    def search_news(self, request: NewsRequest) -> NewsSearchResult:
        request = NewsRequest.model_validate(request)
        query_url = _ENDPOINT + "?" + urlencode({
            "parameters[title]": request.query,
            "fields": "date,title,url,uuid",
            "sort": "date",
            "direction": "DESC",
            "pagesize": 50,
            "page": 0,
        })
        raw, calls = fetch_bytes(
            self.session,
            query_url,
            allowed_hosts=frozenset({"www.justice.gov"}),
            headers={"User-Agent": "AlphaScope/0.1 (personal research; DOJ release metadata)",
                     "Accept": "application/json"},
            max_bytes=1_000_000,
            timeout_seconds=self.timeout_seconds,
            before_request=self.before_request,
        )
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise HttpFailure("INVALID_PROVIDER_DATA", "DOJ returned invalid JSON", False, calls) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise HttpFailure("INVALID_PROVIDER_DATA", "DOJ response lacks press-release results", False, calls)
        retrieved_at = utc_now()
        articles: list[NewsArticle] = []
        seen_ids: set[str] = set()
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        duplicates = invalid = undated = 0
        for entry in payload["results"]:
            if not isinstance(entry, dict):
                invalid += 1
                continue
            title = html.unescape(entry.get("title", "")).strip() if isinstance(entry.get("title"), str) else ""
            url = entry.get("url")
            if not title or not _official_url(url):
                invalid += 1
                continue
            date_value = entry.get("date")
            publication_date = None
            date_status = "unavailable"
            if date_value not in (None, ""):
                try:
                    publication_date = datetime.fromtimestamp(int(date_value), timezone.utc).date()
                    date_status = "date_only_time_unverified"
                except (ValueError, TypeError, OverflowError, OSError):
                    date_status = "unparseable"
            if publication_date is None:
                undated += 1
                if request.start_date or request.end_date:
                    continue
            elif (request.start_date and publication_date < request.start_date) or (
                request.end_date and publication_date > request.end_date
            ):
                continue
            identifier = entry.get("uuid")
            article_id = identifier if isinstance(identifier, str) and identifier.strip() else hashlib.sha256(url.encode()).hexdigest()[:24]
            title_key = re.sub(r"\W+", "", title.casefold())
            if article_id in seen_ids or url in seen_urls or title_key in seen_titles:
                duplicates += 1
                continue
            seen_ids.add(article_id)
            seen_urls.add(url)
            seen_titles.add(title_key)
            try:
                articles.append(NewsArticle(
                    article_id=article_id,
                    title=title,
                    publisher="U.S. Department of Justice",
                    published_at=None,
                    publication_date=publication_date,
                    publication_date_status=date_status,
                    retrieved_at=retrieved_at,
                    url=url,
                    original_url=url,
                    feed_url=query_url,
                    description=None,
                    query=request.query,
                    language=None,
                    provider=self.name,
                    source_type="official_government_press_release",
                ))
            except ValidationError:
                invalid += 1
                continue
            if len(articles) == request.limit:
                break
        limitations = [
            "DOJ title-filtered press releases only; not general financial or company news",
            "Only the first 50 matching API records are inspected; historical completeness is not guaranteed",
            "The DOJ API supplies a release date, not a verified publication time",
            "Only API metadata is used; release pages are not downloaded",
        ]
        if undated:
            limitations.append(f"{undated} item(s) had no usable release date; excluded when filtering by date")
        if invalid:
            limitations.append(f"{invalid} item(s) had invalid metadata and were skipped")
        if not articles:
            limitations.append("No matching press releases in the inspected API results")
        return NewsSearchResult(
            query=request.query,
            feed_url=query_url,
            retrieved_at=retrieved_at,
            articles=articles,
            limitations=limitations,
            duplicates_removed=duplicates,
            invalid_items_skipped=invalid,
            api_call_count=calls,
            provider=self.name,
        )
