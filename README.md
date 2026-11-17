# AlphaScope

AlphaScope runs bounded financial research with a local language model, read-only data tools, validated state, and source-linked observations.

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

Milestone 3A adds local structured LLM recommendations without tool execution. The demo automatically loads the repository `.env`; existing shell variables take precedence. See `.env.example` for the baseline: `LLM_PROVIDER=ollama`, `LLM_MODEL=qwen3:8b`, `LLM_TEMPERATURE=0`, `LLM_NUM_CTX=4096`, `OLLAMA_BASE_URL=http://localhost:11434`, and `LLM_THINKING=false`. Only loopback endpoints are accepted; redirects and environment HTTP proxies are disabled. `.env` stays ignored by Git.

```powershell
# Windows (POSIX: use .venv/bin/python instead)
.venv/Scripts/python.exe -m pip install -e '.[dev]'
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe examples/llm_decision_demo.py --replan
```

Ollama must already be installed. If unavailable, run `ollama serve`; if the configured model is missing, run `ollama pull qwen3:8b`. The demo uses Ollama's [structured output API](https://docs.ollama.com/api/chat) and Pydantic validation. `--replan` adds the labeled historical fixture, without fetching financial data or changing the research objective. Both decisions require human review: schema validity does not establish factual correctness. `LLM_TIMEOUT_SECONDS` defaults to 60 per HTTP attempt, `LLM_MAX_RETRIES` to 1 (maximum 2), and `LLM_NUM_PREDICT` to 512. Only transient transport/server errors are retried; malformed decisions are rejected. Performance is returned with each result/error and available through the provider's INFO logger; prompts and hidden reasoning are excluded. Missing token/timing metadata remains null. Thinking can be enabled with `LLM_THINKING=true`, but the baseline is disabled and two examples cannot establish reliability.

## M4 autonomous research

Run the default simulated scenario or opt into a live, read-only provider scenario.

```powershell
.\.venv\Scripts\python.exe examples\autonomous_research_demo.py
.\.venv\Scripts\python.exe examples\autonomous_research_demo.py --scenario live-market --json
.\.venv\Scripts\python.exe examples\autonomous_research_demo.py --scenario live-doj --steps 4 --tool-calls 3
```

AlphaScope plans one action, validates its arguments, runs it through an enabled
tool, records a source-linked observation, and replans. The allowlist covers
historical market data, calculations, SEC filings, and DOJ press-release titles.
The DOJ source returns title metadata and SEC document excerpts can be incomplete.
The simulated scenarios label their observations.

`ResearchLimits` defaults to five planning steps, five tool calls, eight Ollama
requests, two consecutive tool failures, two redundant or invalid proposals, and
180 seconds. A budget stop preserves the partial state. A goal-satisfied stop
requires scoped evidence. A contradicted price premise stops for clarification.
See [M4 validation](docs/m4_validation.md) for the run results and stop conditions.

## M5 research quality and evaluation

`ResearchRun.quality` records whether an answer is sufficient, partial,
insufficient, conflicted, or unsupported. The gate validates source and
observation links, recomputes endpoint returns, and checks comparability across
periods, price bases, and benchmarks. Findings retain evidence IDs and source
limitations. An event date within a price interval supports a timing observation.
Causal conclusions require evidence beyond timing and remain unresolved with
the current sources.

The runner keeps the raw action and registered tool proposal alongside any
correction, the final disposition, and dispatch status. A failed or empty search
does not establish factual absence. A truncated SEC excerpt cannot establish
absence from the complete filing.

```powershell
.\.venv\Scripts\python.exe -B evals\run_evals.py --mode offline
.\.venv\Scripts\python.exe -B evals\run_evals.py --mode local --repeat 2 --json m5-local.json
.\.venv\Scripts\python.exe -B evals\run_evals.py --mode local --live-sources --repeat 1
```

Offline mode uses fixed cases and fixtures. Local mode uses the configured
loopback Ollama model and fixture sources by default. The live-source option
enables approved Yahoo and DOJ adapters for applicable cases. Other cases retain
their labeled fixtures. See [M5 validation](docs/m5_validation.md) for the
architecture, metric denominators, and measured results.

## M6 evidence-grounded research reports

Use the existing demo to render a cited Markdown or structured JSON report.
The fixture scenarios use simulated data and local Ollama.

```powershell
.\.venv\Scripts\python.exe -X utf8 examples\autonomous_research_demo.py --scenario fixture-return --report
.\.venv\Scripts\python.exe -X utf8 examples\autonomous_research_demo.py --scenario fixture-doj --report-json --output-dir reports
```

Ollama can prioritize validated findings. The report retains the original
evidence, scope and caveats when drafting fails. Use `--from-run` with a full
`ResearchRun` JSON file to reuse research and `--deterministic-report` to skip
drafting. File export requires `--overwrite` to replace an existing report.
See [M6 validation](docs/m6_validation.md) for the API, checks and results.
