"""Validated SEC lookup and on-demand excerpt tools."""

from __future__ import annotations

from datetime import date
from uuid import uuid4

from pydantic import ValidationError

from financial_research_agent.models import ExecutionMetadata, SourceReference, ToolError, ToolResult, utc_now
from financial_research_agent.providers.sec_edgar import SecError, SecExcerpt, SecFiling, SecFilingRequest, SecFilingsResult, SecProvider


class SecFilingsTool:
    name = "sec_filings"

    def __init__(self, provider: SecProvider):
        self.provider = provider

    def get_company_filings(
        self, *, action_id: str, ticker: str, forms: list[str] | None = None,
        start_date: date | None = None, end_date: date | None = None, limit: int = 10,
    ) -> ToolResult:
        started_at = utc_now()
        try:
            request = SecFilingRequest(identifier=ticker, forms=forms if forms is not None else ["10-K", "10-Q", "8-K"],
                                       start_date=start_date, end_date=end_date, limit=limit)
        except ValidationError:
            return self._failure(action_id, "INVALID_REQUEST", "Invalid company, forms, date range, or limit", started_at)
        try:
            result = SecFilingsResult.model_validate(self.provider.get_company_filings(request))
            if (len(result.filings) > request.limit or
                any(item.cik != result.company.cik or item.form not in request.forms or
                    (request.start_date and item.filing_date < request.start_date) or
                    (request.end_date and item.filing_date > request.end_date)
                    for item in result.filings)):
                raise SecError("INVALID_PROVIDER_DATA", "SEC provider result does not match request",
                               api_call_count=result.api_call_count)
        except SecError as exc:
            return self._failure(action_id, exc.code, str(exc), started_at, exc.retryable, exc.api_call_count)
        except ValidationError:
            return self._failure(action_id, "INVALID_PROVIDER_DATA", "SEC provider returned invalid filings", started_at)
        sources = [SourceReference(url=result.submissions_url, title=f"{result.company.company_name} SEC submissions")]
        sources.extend(SourceReference(url=item.filing_url, title=f"{item.form} {item.accession_number}") for item in result.filings)
        return ToolResult(result_id=str(uuid4()), action_id=action_id, tool_name=self.name, success=True,
                          data={"kind": "sec_filings", **result.model_dump(mode="json")}, sources=sources,
                          metadata=ExecutionMetadata(api_call_count=result.api_call_count, started_at=started_at, finished_at=utc_now()))

    def get_document_excerpt(self, *, action_id: str, filing: SecFiling | dict) -> ToolResult:
        started_at = utc_now()
        try:
            validated = SecFiling.model_validate(filing)
        except ValidationError:
            return self._failure(action_id, "INVALID_REQUEST", "Invalid SEC filing record", started_at)
        try:
            result = SecExcerpt.model_validate(self.provider.get_document_excerpt(validated))
            if (result.cik != validated.cik or result.accession_number != validated.accession_number or
                result.document_url != validated.primary_document_url):
                raise SecError("INVALID_PROVIDER_DATA", "SEC excerpt does not match filing",
                               api_call_count=result.api_call_count)
        except SecError as exc:
            return self._failure(action_id, exc.code, str(exc), started_at, exc.retryable, exc.api_call_count)
        except ValidationError:
            return self._failure(action_id, "INVALID_PROVIDER_DATA", "SEC provider returned invalid excerpt", started_at)
        return ToolResult(result_id=str(uuid4()), action_id=action_id, tool_name=self.name, success=True,
                          data={"kind": "sec_document_excerpt", **result.model_dump(mode="json")},
                          sources=[SourceReference(url=result.document_url, title=f"SEC {result.form} primary document")],
                          metadata=ExecutionMetadata(api_call_count=result.api_call_count, started_at=started_at, finished_at=utc_now()))

    def _failure(self, action_id: str, code: str, message: str, started_at, retryable: bool = False, calls: int = 0) -> ToolResult:
        return ToolResult(result_id=str(uuid4()), action_id=action_id, tool_name=self.name, success=False,
                          error=ToolError(code=code, message=message, retryable=retryable),
                          metadata=ExecutionMetadata(api_call_count=calls, started_at=started_at, finished_at=utc_now()))
