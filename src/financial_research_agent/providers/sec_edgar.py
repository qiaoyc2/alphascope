"""Official SEC ticker mapping, recent submissions, and bounded primary-document text."""

from __future__ import annotations

import re
import threading
import time
from datetime import date, datetime, timezone
from enum import StrEnum
from html.parser import HTMLParser
from typing import Protocol, Self
from zoneinfo import ZoneInfo

import requests
from pydantic import AwareDatetime, Field, HttpUrl, ValidationError, field_validator, model_validator

from financial_research_agent.models import NonEmptyStr, ResearchModel, utc_now
from financial_research_agent.providers.http_client import HttpFailure, fetch_bytes


SUPPORTED_FORMS = frozenset({"10-K", "10-Q", "8-K"})
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
DOCUMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")


class SecErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    COMPANY_NOT_FOUND = "COMPANY_NOT_FOUND"
    AMBIGUOUS_COMPANY = "AMBIGUOUS_COMPANY"
    INVALID_PROVIDER_DATA = "INVALID_PROVIDER_DATA"
    MISSING_DOCUMENT = "MISSING_DOCUMENT"
    UNSUPPORTED_DOCUMENT = "UNSUPPORTED_DOCUMENT"


class SecError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False, api_call_count: int = 0):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.api_call_count = api_call_count


class SecFilingRequest(ResearchModel):
    identifier: str = Field(min_length=1, max_length=120)
    forms: list[str] = Field(default_factory=lambda: ["10-K", "10-Q", "8-K"])
    start_date: date | None = None
    end_date: date | None = None
    limit: int = Field(default=10, ge=1, le=50)

    @field_validator("identifier", mode="before")
    @classmethod
    def strip_identifier(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def check_request(self) -> Self:
        if not self.identifier or (self.start_date and self.end_date and self.start_date > self.end_date):
            raise ValueError("invalid company or date range")
        if not self.forms or not set(self.forms) <= SUPPORTED_FORMS:
            raise ValueError("forms must be 10-K, 10-Q, or 8-K")
        return self


class SecCompany(ResearchModel):
    company_name: NonEmptyStr
    cik: str = Field(pattern=r"^\d{10}$")
    tickers: list[str] = Field(default_factory=list)


class SecFiling(ResearchModel):
    company_name: NonEmptyStr
    cik: str = Field(pattern=r"^\d{10}$")
    form: str
    filing_date: date
    report_date: date | None = None
    acceptance_at: AwareDatetime | None = None
    acceptance_timezone_status: NonEmptyStr
    accession_number: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    primary_document: str | None = None
    filing_url: HttpUrl
    primary_document_url: HttpUrl | None = None
    retrieved_at: AwareDatetime
    source_type: str = "official_sec_filing_record"


class SecFilingsResult(ResearchModel):
    company: SecCompany
    submissions_url: HttpUrl
    retrieved_at: AwareDatetime
    filings: list[SecFiling]
    limitations: list[NonEmptyStr] = Field(default_factory=list)
    api_call_count: int = Field(ge=0)
    provider: str = "sec_edgar"


class SecExcerpt(ResearchModel):
    company_name: NonEmptyStr
    cik: str
    accession_number: str
    form: str
    filing_date: date
    document_url: HttpUrl
    excerpt: str
    retrieved_at: AwareDatetime
    extraction_incomplete: bool
    limitations: list[NonEmptyStr] = Field(default_factory=list)
    api_call_count: int = Field(ge=0)
    provider: str = "sec_edgar"
    source_type: str = "official_sec_primary_document_excerpt"


class SecProvider(Protocol):
    def get_company_filings(self, request: SecFilingRequest) -> SecFilingsResult: ...
    def get_document_excerpt(self, filing: SecFiling) -> SecExcerpt: ...


class SecRateLimiter:
    """Shared in-process pacing across SEC provider instances."""

    def __init__(self, interval_seconds: float = 1.0):
        if interval_seconds < 0.5:
            raise ValueError("SEC interval must be at least 0.5 seconds")
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_at - now)
            if delay:
                time.sleep(delay)
            self._next_at = time.monotonic() + self.interval_seconds


_SHARED_SEC_LIMITER = SecRateLimiter()


class _FilingText(HTMLParser):
    _SKIP_TAGS = {"head", "script", "style", "noscript", "ix:header", "ix:hidden", "ix:resources"}
    _VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        hidden = any(name == "hidden" or (name == "style" and value and
                      re.search(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", value, re.I))
                     for name, value in attrs)
        if self.skip_depth or tag in self._SKIP_TAGS or hidden:
            if tag not in self._VOID_TAGS:
                self.skip_depth += 1
        elif tag in {"p", "div", "br", "tr", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth:
            if tag not in self._VOID_TAGS:
                self.skip_depth -= 1
        elif tag in {"p", "div", "tr", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)


class SecEdgarProvider:
    name = "sec_edgar"
    allowed_hosts = frozenset({"www.sec.gov", "data.sec.gov"})

    def __init__(
        self,
        *,
        user_agent: str,
        session: requests.Session | None = None,
        rate_limiter: SecRateLimiter | None = None,
        timeout_seconds: float = 10.0,
        max_document_bytes: int = 2_000_000,
        excerpt_length: int = 2_000,
    ):
        if (not user_agent or len(user_agent) > 200 or "\r" in user_agent or "\n" in user_agent or
            not re.search(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b", user_agent) or "AlphaScope" not in user_agent):
            raise ValueError("SEC User-Agent must identify AlphaScope and a contact email")
        if max_document_bytes < 1024 or not 100 <= excerpt_length <= 10_000:
            raise ValueError("Invalid document or excerpt bounds")
        self.user_agent = user_agent
        self.session = session or requests.Session()
        self.rate_limiter = rate_limiter or _SHARED_SEC_LIMITER
        self.timeout_seconds = timeout_seconds
        self.max_document_bytes = max_document_bytes
        self.excerpt_length = excerpt_length

    def _get(self, url: str, max_bytes: int) -> tuple[bytes, int]:
        try:
            return fetch_bytes(
                self.session,
                url,
                allowed_hosts=self.allowed_hosts,
                headers={"User-Agent": self.user_agent, "Accept": "application/json, text/html, text/plain"},
                max_bytes=max_bytes,
                timeout_seconds=self.timeout_seconds,
                before_request=self.rate_limiter.wait,
            )
        except HttpFailure as exc:
            raise SecError(exc.code, exc.message, retryable=exc.retryable, api_call_count=exc.request_count) from exc

    @staticmethod
    def _json(raw: bytes, count: int) -> dict:
        import json
        try:
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("top-level JSON is not an object")
            return value
        except (ValueError, UnicodeDecodeError) as exc:
            raise SecError(SecErrorCode.INVALID_PROVIDER_DATA, "SEC returned invalid JSON", api_call_count=count) from exc

    def _resolve_company(self, identifier: str) -> tuple[SecCompany, int]:
        if identifier.isdecimal() and len(identifier) <= 10:
            return SecCompany(company_name="CIK lookup", cik=identifier.zfill(10)), 0
        raw, count = self._get(TICKERS_URL, 2_000_000)
        mapping = self._json(raw, count)
        matches: dict[str, dict] = {}
        for value in mapping.values():
            if not isinstance(value, dict):
                continue
            ticker, title, cik = value.get("ticker"), value.get("title"), value.get("cik_str")
            if not isinstance(ticker, str) or not isinstance(title, str) or not isinstance(cik, int):
                continue
            if identifier.casefold() in {ticker.casefold(), title.casefold()}:
                matches[str(cik).zfill(10)] = value
        if not matches:
            raise SecError(SecErrorCode.COMPANY_NOT_FOUND, "Company or ticker was not found in SEC mapping", api_call_count=count)
        if len(matches) != 1:
            raise SecError(SecErrorCode.AMBIGUOUS_COMPANY, "Company or ticker matched multiple SEC CIKs", api_call_count=count)
        cik, entry = next(iter(matches.items()))
        return SecCompany(company_name=entry["title"], cik=cik, tickers=[entry["ticker"]]), count

    @staticmethod
    def _document_url(cik: str, accession: str, document: str | None) -> str | None:
        if not document:
            return None
        if not DOCUMENT_RE.fullmatch(document) or document in {".", ".."}:
            raise SecError(SecErrorCode.INVALID_PROVIDER_DATA, "SEC primary document name is unsafe")
        return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{document}"

    def get_company_filings(self, request: SecFilingRequest) -> SecFilingsResult:
        request = SecFilingRequest.model_validate(request)
        company, mapping_calls = self._resolve_company(request.identifier)
        url = SUBMISSIONS_URL.format(cik=company.cik)
        try:
            raw, submission_calls = self._get(url, 5_000_000)
        except SecError as exc:
            exc.api_call_count += mapping_calls
            raise
        total_calls = mapping_calls + submission_calls
        data = self._json(raw, total_calls)
        if str(data.get("cik", "")).zfill(10) != company.cik:
            raise SecError(SecErrorCode.INVALID_PROVIDER_DATA, "SEC submissions CIK did not match lookup", api_call_count=total_calls)
        official_name = data.get("name")
        if isinstance(official_name, str) and official_name.strip():
            company.company_name = official_name.strip()
        tickers = data.get("tickers")
        if isinstance(tickers, list):
            company.tickers = [x for x in tickers if isinstance(x, str)]
        filing_section = data.get("filings")
        recent = filing_section.get("recent") if isinstance(filing_section, dict) else None
        if not isinstance(recent, dict):
            raise SecError(SecErrorCode.INVALID_PROVIDER_DATA, "SEC submissions omit recent filings", api_call_count=total_calls)
        required = ("form", "filingDate", "accessionNumber", "primaryDocument")
        if any(not isinstance(recent.get(key), list) for key in required):
            raise SecError(SecErrorCode.INVALID_PROVIDER_DATA, "SEC recent filings have invalid columns", api_call_count=total_calls)
        count = len(recent["form"])
        if any(len(recent[key]) != count for key in required):
            raise SecError(SecErrorCode.INVALID_PROVIDER_DATA, "SEC recent filing columns are misaligned", api_call_count=total_calls)
        retrieved_at = utc_now()
        filings: list[SecFiling] = []
        for index in range(count):
            form = recent["form"][index]
            if form not in request.forms:
                continue
            try:
                filing_date = date.fromisoformat(recent["filingDate"][index])
                if request.start_date and filing_date < request.start_date:
                    continue
                if request.end_date and filing_date > request.end_date:
                    continue
                accession = recent["accessionNumber"][index]
                if not isinstance(accession, str) or not ACCESSION_RE.fullmatch(accession):
                    raise ValueError("invalid accession")
                primary = recent["primaryDocument"][index] or None
                document_url = self._document_url(company.cik, accession, primary)
                report_raw = _column(recent, "reportDate", index)
                report_date = date.fromisoformat(report_raw) if report_raw else None
                acceptance_raw = _column(recent, "acceptanceDateTime", index)
                acceptance_at = datetime.fromisoformat(acceptance_raw) if acceptance_raw else None
                timezone_status = "unavailable"
                if acceptance_at is not None:
                    if acceptance_at.tzinfo is None:
                        acceptance_at = acceptance_at.replace(tzinfo=ZoneInfo("America/New_York"))
                        timezone_status = "sec_eastern_inferred"
                    else:
                        timezone_status = "provided"
                    acceptance_at = acceptance_at.astimezone(timezone.utc)
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(company.cik)}/{accession}-index.htm"
                filings.append(SecFiling(
                    company_name=company.company_name,
                    cik=company.cik,
                    form=form,
                    filing_date=filing_date,
                    report_date=report_date,
                    acceptance_at=acceptance_at,
                    acceptance_timezone_status=timezone_status,
                    accession_number=accession,
                    primary_document=primary,
                    filing_url=filing_url,
                    primary_document_url=document_url,
                    retrieved_at=retrieved_at,
                ))
            except (ValueError, TypeError, ValidationError) as exc:
                raise SecError(SecErrorCode.INVALID_PROVIDER_DATA, "SEC returned invalid filing metadata", api_call_count=total_calls) from exc
            if len(filings) == request.limit:
                break
        limitations = ["Only SEC recent submissions are searched; older filing batches are not loaded"]
        if not filings:
            limitations.append("No matching filings in the recent submissions response")
        if any(item.primary_document is None for item in filings):
            limitations.append("Some filings lack a primary document in SEC metadata")
        return SecFilingsResult(company=company, submissions_url=url, retrieved_at=retrieved_at,
                                filings=filings, limitations=limitations, api_call_count=total_calls)

    def get_document_excerpt(self, filing: SecFiling) -> SecExcerpt:
        filing = SecFiling.model_validate(filing)
        if not filing.primary_document:
            raise SecError(SecErrorCode.MISSING_DOCUMENT, "Filing has no primary document")
        if not ACCESSION_RE.fullmatch(filing.accession_number):
            raise SecError(SecErrorCode.INVALID_REQUEST, "Invalid accession number")
        url = self._document_url(filing.cik, filing.accession_number, filing.primary_document)
        if str(filing.primary_document_url) != url:
            raise SecError(SecErrorCode.INVALID_REQUEST, "Filing document URL does not match SEC metadata")
        if not filing.primary_document.lower().endswith((".htm", ".html", ".txt")):
            raise SecError(SecErrorCode.UNSUPPORTED_DOCUMENT, "Only HTML and text primary documents are supported")
        raw, calls = self._get(url, self.max_document_bytes)
        if raw.startswith(b"%PDF-"):
            raise SecError(SecErrorCode.UNSUPPORTED_DOCUMENT, "Primary document is a PDF", api_call_count=calls)
        content = raw.decode("utf-8", errors="replace")
        decoding_incomplete = "\ufffd" in content
        if filing.primary_document.lower().endswith((".htm", ".html")):
            parser = _FilingText()
            parser.feed(content)
            content = " ".join(parser.parts)
        normalized = re.sub(r"\s+", " ", content).strip()
        if not normalized:
            raise SecError(SecErrorCode.INVALID_PROVIDER_DATA, "Primary document has no readable text", api_call_count=calls)
        excerpt = normalized[: self.excerpt_length]
        truncated = len(normalized) > self.excerpt_length
        incomplete = truncated or decoding_incomplete
        limitations = ["Only a bounded plain-text excerpt of the primary document was extracted"]
        if truncated:
            limitations.append("Excerpt truncated before the end of the document")
        if decoding_incomplete:
            limitations.append("Some document bytes could not be decoded as UTF-8")
        return SecExcerpt(company_name=filing.company_name, cik=filing.cik, accession_number=filing.accession_number,
                          form=filing.form, filing_date=filing.filing_date, document_url=url, excerpt=excerpt,
                          retrieved_at=utc_now(), extraction_incomplete=incomplete, limitations=limitations,
                          api_call_count=calls)


def _column(recent: dict, key: str, index: int) -> str | None:
    values = recent.get(key)
    if not isinstance(values, list) or len(values) <= index:
        return None
    value = values[index]
    return value if isinstance(value, str) and value else None
