# Milestone 3A validation

The local structured-decision integration is implemented. The final live pair was
structurally valid and selected sensible next steps, with a remaining conditional
premise-assessment limitation described below. No tool execution or research-state
changes are connected to model output.

## Foundation and configuration

- Inspected repository `AGENTS.md`, configuration, M1 models/state operations,
  M2 tool/provider contracts, and existing tests before editing.
- Starting worktree was clean. `main`, cached `origin/main`, and a read-only
  `git ls-remote origin refs/heads/main` check matched M2B commit
  `a614e14716f694f13371f986906463003659c485`.
- Existing `.env` was updated only for the requested Ollama baseline and disabled
  thinking; unrelated variables, including SEC configuration, were preserved.
  `.env` is ignored and untracked. `.gitignore` already had the needed exclusions.
- Confirmed `http://localhost:11434/api/tags` and `/api/version` were accessible.
  Ollama reported version `0.40.11` and installed model `qwen3:8b`, Q4_K_M,
  approximately 5.23 GB on disk. No model download, system configuration, or public
  API binding changes were needed.
- Created the missing Windows `.venv` using available Python 3.14.2. The repository
  requires Python 3.11+. Windows uses `.venv/Scripts/python.exe` in place of the
  documented POSIX `.venv/bin/python`.

## Architecture

`LLMProvider.generate(prompt, schema)` accepts any Pydantic output model and returns
`StructuredLLMResult` with a validated decision and `LLMPerformance`. The Ollama
adapter sends the schema using the supported [`format` API parameter](https://docs.ollama.com/api/chat),
also includes it in a system message, and validates returned content independently.
The provider contract has no dependency on Ollama or research-state operations.

`LLMDecision` is separate from `ResearchAction` and `ResearchState`. Its tool enum
is shared with the existing four tool classes; the DOJ-backed tool remains `news`.
Required nullable fields distinguish absent fields from explicit uncertainty.
Cross-field checks reject contradictory stop flags, tool requests during clarify/
stop, and continue without a tool. These checks do not establish factual truth.

`load_llm_settings()` loads the checkout's `.env` regardless of working directory,
without replacing shell variables. An explicit environment file is also supported.
Only loopback origins are accepted. Redirects and environment HTTP proxies are
disabled. Timeouts, output-token budgets, and retries are bounded. Only transient
transport failures, rate limits, and server HTTP errors are retried. Model errors,
invalid JSON, and schema failures are returned as categorized `LLMError` instances.

Metadata is returned and logged at INFO for each attempt, including failed attempts.
It contains request/total elapsed time, model, token counts when supplied, generation
time, calculable generation tokens/second, validation status, and error category.
Missing or malformed optional counters remain null. Prompts, response content,
hidden reasoning, and raw server error bodies are excluded from logs.

Added dependency: `python-dotenv>=1.0,<2` (installed version 1.2.4). HTTP calls reuse
the existing `requests` dependency; no Ollama SDK or hosted LLM dependency was added.

## Offline verification

```powershell
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m pip check
```

160 tests passed: 82 existing M1/M2 tests and 78 new tests. New tests use mocked
Ollama responses and cover valid generation, Pydantic validation, invalid tool
names, missing fields, invalid JSON/envelopes, incomplete/empty responses, connection
and timeout failures, missing models, redirects, bounded retries, configuration
loading, optional metadata, telemetry privacy, and alternate output schemas.
Dependency consistency also passed. No live financial data is used by the suite.

## Final live pair

```powershell
.venv/Scripts/python.exe examples/llm_decision_demo.py --replan
```

Settings: `qwen3:8b`, temperature 0, context 4096, thinking disabled, output limit
512. Each final request succeeded on its first attempt with Pydantic validation.

| Scenario | Premise | Action | Next tool | Assessment |
| --- | --- | --- | --- | --- |
| Initial question, no observations | null | continue | market_data | Reasonable: identifies missing price/benchmark observations before drawing a conclusion. |
| Historical fixture | null | clarify | null | Recognizes positive absolute return and relative underperformance; requests user clarification without duplicate retrieval. Premise field remains overly cautious. |

The historical fixture uses exactly the supplied numerical observations for
2026-08-28 through 2026-10-12: GOOGL +1.83%, QQQ +4.24%, relative performance
-2.41 percentage points. The demo explicitly scopes "recently" to this supplied
period for the conditional test, labels fixture provenance, and makes no live
price request. The model's final rationale recognizes that the assumed decline
conflicts with the positive return. It invents no event or causal explanation and
does not change the objective. However, it leaves `premise_valid=null` because the
observations are fixtures; the desired conditional assessment would be `false`.

| Scenario | Request seconds | Input tokens | Output tokens | Generation seconds | Generation tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 5.041 | 797 | 114 | 2.555 | 44.61 |
| Replanning fixture | 4.024 | 980 | 173 | 3.912 | 44.22 |

Request time includes more than generation and can include loading, prompt
evaluation, and local overhead. GPU utilization was not independently measured
in this milestone. These timings do not guarantee longer-request performance.

Several exploratory prompt revisions preceded this final pair. Earlier responses
were schema-valid but sometimes requested duplicate data or mishandled premise
assessment. General prompt/schema descriptions were refined; no expected answer
is substituted into model output. These developmental examples are not an
independent reliability benchmark. Human review remains necessary, and the
conditional-premise issue remains unresolved even in the final pair.

## Files and remaining scope

Created:

- `src/financial_research_agent/config.py`
- `src/financial_research_agent/llm_decision.py`
- `src/financial_research_agent/providers/llm.py`
- `src/financial_research_agent/providers/ollama.py`
- `src/financial_research_agent/prompts/__init__.py`
- `src/financial_research_agent/prompts/tool_decision.py`
- `examples/llm_decision_demo.py`
- `tests/test_llm_config.py`
- `tests/test_llm_decision.py`
- `tests/test_ollama.py`
- `tests/test_tool_decision_prompt.py`
- `docs/m3a_validation.md`

Modified: `.env.example`, `pyproject.toml`, `README.md`, tool package `__init__.py`,
and the four existing tool modules (`market_data.py`, `calculator.py`, `news.py`,
`sec_filings.py`) to share tool identifiers. Local-only `.env` was also updated.

Runtime defaults target a source checkout; packaged deployments should supply an
explicit `.env` path. The HTTP timeout bounds individual network operations rather
than imposing a strict total wall-clock deadline; retries can increase total time.
Thinking is configurable but was not validated live. Live service/model errors
were not induced; their behavior is covered by offline tests. No long-context,
concurrency, broad factual reliability, or performance benchmark was performed.

Git changes remain unstaged: eight modified tracked files and twelve new files;
`.env` and `.venv` are ignored. No commits, pushes, rebases, history edits, or remote
changes were made.

Suggested commit message: `feat: add local Ollama structured research decisions (M3A)`.

M3B orchestration, model-driven tool execution, and all later milestones remain
outside this change and require approval.
