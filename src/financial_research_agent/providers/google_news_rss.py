"""Headline-level Google News RSS search; publisher pages are never fetched."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urlencode, urlsplit

import requests
from pydantic import ValidationError

from financial_research_agent.models import utc_now
from financial_research_agent.providers.http_client import HttpFailure, fetch_bytes
from financial_research_agent.providers.news import NewsArticle, NewsRequest, NewsSearchResult


class _TextOnly(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _TextOnly()
    parser.feed(value)
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


class GoogleNewsRssProvider:
    """Manual personal feed-reader display only; not approved for agent analysis."""

    name = "google_news_rss"
    allowed_hosts = frozenset({"news.google.com"})

    def __init__(self, *, session: requests.Session | None = None, timeout_seconds: float = 10.0):
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def search_news(self, request: NewsRequest) -> NewsSearchResult:
        request = NewsRequest.model_validate(request)
        feed_url = "https://news.google.com/rss/search?" + urlencode(
            {"q": request.query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        )
        raw, calls = fetch_bytes(
            self.session,
            feed_url,
            allowed_hosts=self.allowed_hosts,
            headers={"User-Agent": "AlphaScope/0.1 (RSS research; headline metadata only)", "Accept": "application/rss+xml, application/xml"},
            max_bytes=1_000_000,
            timeout_seconds=self.timeout_seconds,
        )
        if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
            raise HttpFailure("INVALID_FEED", "RSS document contains unsupported declarations", False, calls)
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            raise HttpFailure("INVALID_FEED", "Provider returned malformed RSS", False, calls) from exc
        channel = root.find("channel") if root.tag == "rss" else None
        if channel is None:
            raise HttpFailure("INVALID_FEED", "Provider did not return an RSS channel", False, calls)
        language = channel.findtext("language")
        articles: list[NewsArticle] = []
        seen_ids: set[str] = set()
        seen_links: set[str] = set()
        seen_headlines: set[str] = set()
        duplicates = invalid = undated = 0
        retrieved_at = utc_now()
        for item in channel.findall("item"):
            title = _plain_text(item.findtext("title") or "")
            link = (item.findtext("link") or "").strip()
            try:
                parsed_link = urlsplit(link)
                valid_link = (parsed_link.scheme == "https" and parsed_link.hostname == "news.google.com"
                              and parsed_link.username is None and parsed_link.port is None)
            except ValueError:
                valid_link = False
            if not title or not valid_link:
                invalid += 1
                continue
            publisher = _plain_text(item.findtext("source") or "") or None
            raw_date = (item.findtext("pubDate") or "").strip()
            published_at = None
            date_status = "unavailable"
            if raw_date:
                try:
                    published_at = parsedate_to_datetime(raw_date)
                    if published_at.tzinfo is None:
                        published_at = published_at.replace(tzinfo=timezone.utc)
                        date_status = "timezone_inferred_utc"
                    else:
                        date_status = "provided"
                    published_at = published_at.astimezone(timezone.utc)
                except (TypeError, ValueError, IndexError):
                    date_status = "unparseable"
            if published_at is None:
                undated += 1
                if request.start_date or request.end_date:
                    continue
            elif (request.start_date and published_at.date() < request.start_date) or (
                request.end_date and published_at.date() > request.end_date
            ):
                continue
            guid = (item.findtext("guid") or "").strip()
            headline_key = re.sub(r"\W+", "", title.casefold())
            if (guid and guid in seen_ids) or link in seen_links or headline_key in seen_headlines:
                duplicates += 1
                continue
            seen_ids.add(guid)
            seen_links.add(link)
            seen_headlines.add(headline_key)
            description = _plain_text(item.findtext("description") or "")[:1000] or None
            try:
                articles.append(
                    NewsArticle(
                        article_id=hashlib.sha256((guid or link).encode()).hexdigest()[:24],
                        title=title,
                        publisher=publisher,
                        published_at=published_at,
                        publication_date_status=date_status,
                        publication_date=published_at.date() if published_at else None,
                        retrieved_at=retrieved_at,
                        url=link,
                        original_url=None,
                        feed_url=feed_url,
                        description=description,
                        query=request.query,
                        language=language,
                        provider=self.name,
                    )
                )
            except ValidationError:
                invalid += 1
                continue
            if len(articles) == request.limit:
                break
        limitations = [
            "Headline and feed description only; article body and original publisher URL unverified",
            "RSS search coverage, ranking, and historical completeness are not guaranteed",
            "Google News feed terms restrict use to rendering results in a personal, non-commercial feed reader",
        ]
        if undated:
            limitations.append(f"{undated} item(s) lacked a usable publication date; excluded when date filtering was requested")
        if invalid:
            limitations.append(f"{invalid} item(s) had invalid or incomplete metadata and were skipped")
        if not articles:
            limitations.append("No matching items were available in the returned feed")
        return NewsSearchResult(
            query=request.query,
            feed_url=feed_url,
            retrieved_at=retrieved_at,
            articles=articles,
            limitations=limitations,
            duplicates_removed=duplicates,
            invalid_items_skipped=invalid,
            api_call_count=calls,
            provider=self.name,
        )
