from __future__ import annotations

import json
from io import BytesIO
from datetime import date
from unittest.mock import Mock

import requests

from financial_research_agent.providers.sec_edgar import SecEdgarProvider, SecRateLimiter
from financial_research_agent.tools.sec_filings import SecFilingsTool


def response(data: object, status: int = 200, *, raw: bool = False, headers: dict | None = None) -> requests.Response:
    result = requests.Response()
    result.status_code = status
    result.raw = BytesIO(data if raw else json.dumps(data).encode())
    result.headers.update(headers or {})
    return result


MAPPING = {"0": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."}}
SUBMISSIONS = {
    "cik": 1652044, "name": "Alphabet Inc.", "tickers": ["GOOGL", "GOOG"],
    "filings": {"recent": {
        "form": ["10-K", "8-K", "4"],
        "filingDate": ["2026-10-10", "2026-09-01", "2026-08-01"],
        "reportDate": ["2026-09-30", "", ""],
        "acceptanceDateTime": ["2026-10-10T16:00:00", "", ""],
        "accessionNumber": ["0001652044-26-000001", "0001652044-26-000002", "0001652044-26-000003"],
        "primaryDocument": ["googl-10k.htm", "", "other.htm"],
    }},
}


class NoWait:
    def wait(self):
        pass


def provider(*replies, max_document_bytes=2_000_000, excerpt_length=2000):
    session = Mock()
    session.get.side_effect = replies
    return SecEdgarProvider(user_agent="AlphaScope/0.1 research contact@example.com", session=session,
                            rate_limiter=NoWait(), max_document_bytes=max_document_bytes,
                            excerpt_length=excerpt_length)


def test_ticker_resolution_metadata_filtering_and_urls():
    tool = SecFilingsTool(provider(response(MAPPING), response(SUBMISSIONS)))
    result = tool.get_company_filings(action_id="s", ticker="GOOGL", forms=["10-K"],
                                      start_date=date(2026, 10, 1), end_date=date(2026, 10, 31))
    assert result.success
    assert result.data["company"]["cik"] == "0001652044"
    assert result.metadata.api_call_count == 2
    assert len(result.data["filings"]) == 1
    filing = result.data["filings"][0]
    assert filing["report_date"] == "2026-09-30"
    assert filing["acceptance_timezone_status"] == "sec_eastern_inferred"
    assert filing["accession_number"] == "0001652044-26-000001"
    assert filing["filing_url"].endswith("0001652044-26-000001-index.htm")
    assert filing["primary_document_url"].endswith("000165204426000001/googl-10k.htm")
    assert str(result.sources[1].url) == filing["filing_url"]


def test_missing_ticker_and_ambiguous_name():
    missing = SecFilingsTool(provider(response(MAPPING))).get_company_filings(action_id="s", ticker="MISSING")
    assert missing.error.code == "COMPANY_NOT_FOUND"
    ambiguous_mapping = {**MAPPING, "1": {"cik_str": 123456, "ticker": "GOOG", "title": "Alphabet Inc."}}
    ambiguous = SecFilingsTool(provider(response(ambiguous_mapping))).get_company_filings(action_id="s", ticker="Alphabet Inc.")
    assert ambiguous.error.code == "AMBIGUOUS_COMPANY"


def test_cik_lookup_and_missing_primary_document():
    result = SecFilingsTool(provider(response(SUBMISSIONS))).get_company_filings(action_id="s", ticker="1652044", forms=["8-K"])
    assert result.success and result.metadata.api_call_count == 1
    filing = result.data["filings"][0]
    assert filing["primary_document"] is None
    assert filing["primary_document_url"] is None
    excerpt = SecFilingsTool(provider()).get_document_excerpt(action_id="e", filing=filing)
    assert excerpt.error.code == "MISSING_DOCUMENT"


def test_sec_403_429_invalid_json_and_redirect(monkeypatch):
    monkeypatch.setattr("financial_research_agent.providers.http_client.time.sleep", lambda _: None)
    for replies, code in [
        ((response({}, 403),), "ACCESS_DENIED"),
        ((response({}, 429), response({}, 429)), "RATE_LIMITED"),
        ((response(b"broken", raw=True),), "INVALID_PROVIDER_DATA"),
        ((response({}, 302, headers={"Location": "https://example.com"}),), "UNEXPECTED_REDIRECT"),
    ]:
        result = SecFilingsTool(provider(*replies)).get_company_filings(action_id="s", ticker="GOOGL")
        assert result.error.code == code


def test_document_extraction_and_size_limit():
    filing = SecFilingsTool(provider(response(MAPPING), response(SUBMISSIONS))).get_company_filings(action_id="s", ticker="GOOGL", forms=["10-K"]).data["filings"][0]
    html = b"<html><head><title>hidden</title></head><body><ix:header><ix:resources><xbrli:context>hidden</xbrli:context></ix:resources></ix:header><div style='display:none'>hidden</div><h1>Business</h1><p>Alphabet filing text for investors.</p><script>bad()</script></body></html>"
    excerpt = SecFilingsTool(provider(response(html, raw=True), excerpt_length=100)).get_document_excerpt(action_id="e", filing=filing)
    assert excerpt.success
    assert "Alphabet filing text" in excerpt.data["excerpt"]
    assert "hidden" not in excerpt.data["excerpt"] and "bad()" not in excerpt.data["excerpt"]
    assert excerpt.data["source_type"] == "official_sec_primary_document_excerpt"
    assert str(excerpt.sources[0].url) == filing["primary_document_url"]
    oversized = SecFilingsTool(provider(response(b"x", raw=True, headers={"Content-Length": "3000"}), max_document_bytes=1024)).get_document_excerpt(action_id="e", filing=filing)
    assert oversized.error.code == "RESPONSE_TOO_LARGE"


def test_unsafe_document_name_and_invalid_request():
    malicious = json.loads(json.dumps(SUBMISSIONS))
    malicious["filings"]["recent"]["primaryDocument"][0] = "../../evil"
    result = SecFilingsTool(provider(response(MAPPING), response(malicious))).get_company_filings(action_id="s", ticker="GOOGL", forms=["10-K"])
    assert result.error.code == "INVALID_PROVIDER_DATA"
    invalid = SecFilingsTool(provider()).get_company_filings(action_id="s", ticker="", forms=["10-X"])
    assert invalid.error.code == "INVALID_REQUEST"


def test_shared_limiter_paces_calls(monkeypatch):
    clock = [10.0]
    delays = []
    monkeypatch.setattr("financial_research_agent.providers.sec_edgar.time.monotonic", lambda: clock[0])
    def advance(seconds):
        delays.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr("financial_research_agent.providers.sec_edgar.time.sleep", advance)
    limiter = SecRateLimiter(interval_seconds=1.0)
    limiter.wait()
    limiter.wait()
    assert delays == [1.0]
