# AlphaScope

AlphaScope is a Python project for an autonomous financial research agent that will investigate stock questions using traceable evidence. The current milestone provides validated research state, a free historical market-data adapter, and deterministic calculations; agent planning and causal research will follow later.

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
```

Milestone 2A has no required environment variables. To try the live market-data tools:

```bash
.venv/bin/python examples/market_data_demo.py --ticker GOOGL --benchmark QQQ
```

Dates passed with `--start` and `--end` are inclusive. Returns use adjusted close by default; raw close remains separately available. Yahoo data accessed through yfinance is subject to Yahoo's usage terms and may be unavailable or incomplete.

Milestone 2B adds read-only U.S. Department of Justice press-release title search and SEC filing lookup. Set a descriptive SEC User-Agent with a real contact email, then run the source demo:

```bash
export ALPHASCOPE_SEC_USER_AGENT="AlphaScope/0.1 contact=you@example.com"
.venv/bin/python examples/research_sources_demo.py --company Alphabet --ticker GOOGL --query Google
# Add --excerpt to request one bounded primary-document excerpt.
```

The default news source is the DOJ's public press-release API. It covers official U.S. government announcements with a title match, not general financial news or company earnings. It returns release dates without verified publication times. The Google News RSS adapter remains available only for personal, non-commercial feed-reader display; its feed terms do not support use in the later automated agent loop. See [news source suitability](docs/news_source_suitability.md) for the source review and remaining limits. SEC lookup uses the official ticker mapping and recent submissions only; older submission batches are not searched. DOJ and SEC requests are each paced to one per second within this process. Downloads are bounded, and filing documents are not fetched until `--excerpt` is requested.
