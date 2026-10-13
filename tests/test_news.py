from __future__ import annotations

from datetime import date
from io import BytesIO
from unittest.mock import Mock

import requests

from financial_research_agent.providers.google_news_rss import GoogleNewsRssProvider
from financial_research_agent.providers.news import NewsRequest
from financial_research_agent.tools.news import NewsTool


def response(content: bytes, status: int = 200, headers: dict | None = None) -> requests.Response:
    item = requests.Response()
    item.status_code = status
    item.raw = BytesIO(content)
    item.headers.update(headers or {})
    return item


def feed(items: str) -> bytes:
    return f'<rss version="2.0"><channel><language>en-US</language>{items}</channel></rss>'.encode()


def article(title: str, link: str, pub_date: str = "Mon, 12 Oct 2026 10:00:00 GMT", guid: str = "one") -> str:
    date_xml = f"<pubDate>{pub_date}</pubDate>" if pub_date else ""
    return f"<item><title>{title}</title><link>{link}</link><guid>{guid}</guid>{date_xml}<source>Example Press</source><description>Feed excerpt</description></item>"


def provider(*responses: requests.Response | Exception) -> GoogleNewsRssProvider:
    session = Mock()
    session.get.side_effect = responses
    return GoogleNewsRssProvider(session=session)


def test_rss_multiple_articles_and_provenance():
    raw = feed(article("Alphabet earnings", "https://news.google.com/rss/articles/a") +
               article("Google regulation", "https://news.google.com/rss/articles/b", guid="two"))
    result = NewsTool(provider(response(raw))).search_news(action_id="n1", query="Alphabet GOOGL earnings")
    assert result.success
    assert len(result.data["articles"]) == 2
    first = result.data["articles"][0]
    assert first["publisher"] == "Example Press"
    assert first["published_at"].endswith("Z")
    assert first["original_url"] is None
    assert first["source_type"] == "rss_headline"
    assert str(result.sources[1].url) == first["url"]
    assert result.metadata.api_call_count == 1


def test_undated_items_and_date_filter():
    raw = feed(article("Undated", "https://news.google.com/rss/articles/a", pub_date="") +
               article("Dated", "https://news.google.com/rss/articles/b", guid="two"))
    unfiltered = provider(response(raw)).search_news(NewsRequest(query="Alphabet"))
    assert unfiltered.articles[0].published_at is None
    assert unfiltered.articles[0].publication_date_status == "unavailable"
    filtered = provider(response(raw)).search_news(NewsRequest(query="Alphabet", start_date=date(2026, 10, 12), end_date=date(2026, 10, 12)))
    assert [x.title for x in filtered.articles] == ["Dated"]
    assert any("lacked a usable" in x for x in filtered.limitations)


def test_dedupes_headlines_and_urls():
    raw = feed(article("Same Story", "https://news.google.com/rss/articles/a") +
               article("Same Story", "https://news.google.com/rss/articles/b", guid="two") +
               article("Different", "https://news.google.com/rss/articles/a", guid="three"))
    result = provider(response(raw)).search_news(NewsRequest(query="Alphabet"))
    assert len(result.articles) == 1
    assert result.duplicates_removed == 2


def test_invalid_xml_and_empty_feed():
    invalid = NewsTool(provider(response(b"<rss>"))).search_news(action_id="n", query="Google")
    assert invalid.error.code == "INVALID_FEED"
    empty = NewsTool(provider(response(feed("")))).search_news(action_id="n", query="Google")
    assert empty.success and empty.data["articles"] == []
    assert "No matching items" in empty.data["limitations"][-1]


def test_timeout_http_rate_limit_and_redirect(monkeypatch):
    monkeypatch.setattr("financial_research_agent.providers.http_client.time.sleep", lambda _: None)
    for replies, code in [
        ((requests.Timeout(), requests.Timeout()), "PROVIDER_UNAVAILABLE"),
        ((response(b"", 404),), "HTTP_ERROR"),
        ((response(b"", 429), response(b"", 429)), "RATE_LIMITED"),
        ((response(b"", 302, {"Location": "https://evil.example"}),), "UNEXPECTED_REDIRECT"),
    ]:
        tool = NewsTool(provider(*replies))
        result = tool.search_news(action_id="n", query="Alphabet")
        assert not result.success and result.error.code == code


def test_invalid_item_and_response_size():
    raw = feed(article("Unsafe", "file:///etc/passwd"))
    result = provider(response(raw)).search_news(NewsRequest(query="Alphabet"))
    assert result.invalid_items_skipped == 1
    assert not result.articles
    huge = NewsTool(provider(response(b"x", headers={"Content-Length": "2000000"}))).search_news(action_id="n", query="Alphabet")
    assert huge.error.code == "RESPONSE_TOO_LARGE"


def test_malformed_link_is_skipped():
    raw = feed(article("Broken link", "https://news.google.com:bad/rss/articles/a"))
    result = provider(response(raw)).search_news(NewsRequest(query="Alphabet"))
    assert result.invalid_items_skipped == 1


def test_invalid_query_is_tool_error():
    result = NewsTool(provider()).search_news(action_id="n", query=" ")
    assert result.error.code == "INVALID_REQUEST"
