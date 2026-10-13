from __future__ import annotations

import json
from datetime import date, datetime, timezone
from io import BytesIO
from unittest.mock import Mock

import requests

from financial_research_agent.providers.doj_press_releases import DoJPressReleaseProvider
from financial_research_agent.providers.news import NewsRequest
from financial_research_agent.tools.news import NewsTool


def _response(data: object, status: int = 200, *, raw: bool = False) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.raw = BytesIO(data if raw else json.dumps(data).encode())
    return response


def _entry(title: str, suffix: str, day: int = 1, uuid: str | None = None) -> dict:
    return {
        "title": title,
        "url": f"https://www.justice.gov/opa/pr/{suffix}",
        "uuid": uuid or suffix,
        "date": str(int(datetime(2026, 10, day, tzinfo=timezone.utc).timestamp())),
    }


def _provider(*replies: requests.Response | Exception) -> DoJPressReleaseProvider:
    session = Mock()
    session.get.side_effect = replies
    return DoJPressReleaseProvider(session=session, before_request=lambda: None)


def test_doj_tool_preserves_official_provenance_and_date_only():
    entries = [_entry("DOJ announces Google remedy", "google-remedy"),
               _entry("DOJ sues a technology company", "tech-case", 2)]
    provider = _provider(_response({"results": entries}))
    result = NewsTool(provider).search_news(action_id="doj", query="Google", limit=5)
    assert result.success and len(result.data["articles"]) == 2
    first = result.data["articles"][0]
    assert first["publisher"] == "U.S. Department of Justice"
    assert first["published_at"] is None
    assert first["publication_date"] == "2026-10-01"
    assert first["publication_date_status"] == "date_only_time_unverified"
    assert first["original_url"] == first["url"]
    assert first["source_type"] == "official_government_press_release"
    assert str(result.sources[1].url) == first["url"]
    assert "parameters%5Btitle%5D=Google" in str(result.sources[0].url)
    assert result.metadata.api_call_count == 1
    provider.session.get.assert_called_once()
    assert provider.session.get.call_args.kwargs["allow_redirects"] is False


def test_doj_date_filter_and_missing_date():
    dated = _entry("Google announcement", "a", 1)
    undated = {**_entry("Undated Google announcement", "b"), "date": ""}
    payload = {"results": [dated, undated]}
    filtered = _provider(_response(payload)).search_news(
        NewsRequest(query="Google", start_date=date(2026, 10, 1), end_date=date(2026, 10, 1)))
    assert [item.title for item in filtered.articles] == ["Google announcement"]
    assert any("no usable release date" in note for note in filtered.limitations)
    unfiltered = _provider(_response(payload)).search_news(NewsRequest(query="Google"))
    assert unfiltered.articles[1].publication_date is None
    assert unfiltered.articles[1].publication_date_status == "unavailable"


def test_doj_dedupes_syndicated_headline_and_skips_unsafe_url():
    entries = [_entry("Same DOJ Action", "a"), _entry("Same DOJ Action", "b"),
               {**_entry("Unsafe", "c"), "url": "https://evil.example/story"}]
    result = _provider(_response({"results": entries})).search_news(NewsRequest(query="DOJ"))
    assert len(result.articles) == 1
    assert result.duplicates_removed == 1
    assert result.invalid_items_skipped == 1


def test_doj_empty_invalid_json_and_http_failures(monkeypatch):
    monkeypatch.setattr("financial_research_agent.providers.http_client.time.sleep", lambda _: None)
    empty = NewsTool(_provider(_response({"results": []}))).search_news(action_id="d", query="Alphabet")
    assert empty.success and not empty.data["articles"]
    for replies, code in [
        ((_response(b"bad json", raw=True),), "INVALID_PROVIDER_DATA"),
        ((_response({}, 403),), "ACCESS_DENIED"),
        ((_response({}, 429), _response({}, 429)), "RATE_LIMITED"),
        ((requests.Timeout(), requests.Timeout()), "PROVIDER_UNAVAILABLE"),
    ]:
        result = NewsTool(_provider(*replies)).search_news(action_id="d", query="Google")
        assert not result.success and result.error.code == code
