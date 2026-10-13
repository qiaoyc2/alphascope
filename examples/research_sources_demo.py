"""Read-only news and SEC source demonstration; no causal research claims."""

from __future__ import annotations

import argparse
import os

from financial_research_agent.providers.doj_press_releases import DoJPressReleaseProvider
from financial_research_agent.providers.sec_edgar import SecEdgarProvider
from financial_research_agent.tools.news import NewsTool
from financial_research_agent.tools.sec_filings import SecFilingsTool


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect news headlines and official SEC filings")
    parser.add_argument("--company", default="Alphabet")
    parser.add_argument("--ticker", default="GOOGL")
    parser.add_argument("--query", help="DOJ press-release title search (default: company name)")
    parser.add_argument("--excerpt", action="store_true", help="Fetch one primary SEC document excerpt")
    args = parser.parse_args()

    query = args.query or args.company
    news = NewsTool(DoJPressReleaseProvider()).search_news(action_id="demo-news", query=query, limit=5)
    if news.success:
        print(f"DOJ press releases with {query!r} in the title (API metadata only):")
        for item in news.data["articles"]:
            print(f"- {item['title']} | {item['publisher'] or 'publisher unavailable'} | "
                  f"{item['publication_date'] or 'release date unavailable'} | {item['url']}")
        if not news.data["articles"]:
            print("  No matching releases in the inspected API results")
        print(f"DOJ API query: {news.data['feed_url']}")
    else:
        print(f"News search failed: {news.error.code}: {news.error.message}")

    user_agent = os.environ.get("ALPHASCOPE_SEC_USER_AGENT", "")
    if not user_agent:
        print("SEC lookup skipped: set ALPHASCOPE_SEC_USER_AGENT with an AlphaScope identifier and contact email")
        return 1
    try:
        sec = SecFilingsTool(SecEdgarProvider(user_agent=user_agent))
    except ValueError as exc:
        print(f"SEC configuration invalid: {exc}")
        return 1
    filings = sec.get_company_filings(action_id="demo-sec", ticker=args.ticker, limit=5)
    if not filings.success:
        print(f"SEC lookup failed: {filings.error.code}: {filings.error.message}")
        return 1
    print(f"SEC filings for {filings.data['company']['company_name']} (CIK {filings.data['company']['cik']}):")
    for filing in filings.data["filings"]:
        print(f"- {filing['form']} | filed {filing['filing_date']} | "
              f"{filing['accession_number']} | {filing['filing_url']}")
    if not filings.data["filings"]:
        print("  No matching recent filings")
    print(f"Submissions: {filings.data['submissions_url']}")
    if args.excerpt:
        filing = next((item for item in filings.data["filings"] if item["primary_document"]), None)
        if filing is None:
            print("No primary document available for an excerpt")
        else:
            excerpt = sec.get_document_excerpt(action_id="demo-excerpt", filing=filing)
            if excerpt.success:
                print(f"Excerpt from {excerpt.data['document_url']}:")
                print(excerpt.data["excerpt"])
                print(f"Extraction incomplete: {excerpt.data['extraction_incomplete']}")
            else:
                print(f"Excerpt unavailable: {excerpt.error.code}: {excerpt.error.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
