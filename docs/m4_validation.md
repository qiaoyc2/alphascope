# AlphaScope M4 validation

M4 adds a bounded research runner over the M3B planner. Each step validates the
proposal against the enabled tool contracts, records a unique action, dispatches
the read-only request, and records the result and a source-linked observation.
The next planning prompt includes the observations, evidence references, failures,
and remaining budget. The full structured result is available through the Python API.
The CLI JSON view limits raw tool output and omits raw SEC excerpts.

## Run and limits

```powershell
.\.venv\Scripts\python.exe -B examples\autonomous_research_demo.py --scenario fixture
.\.venv\Scripts\python.exe -B examples\autonomous_research_demo.py --scenario live-market --json
.\.venv\Scripts\python.exe -B examples\autonomous_research_demo.py --scenario live-doj --json
```

The defaults are five planning steps, five tool calls, eight Ollama HTTP requests
including retries, two consecutive tool failures, two consecutive redundant or
invalid proposals, and a 180-second deadline. The live scenarios below used four
steps, three tool calls, seven Ollama requests, and 120 seconds each. Fixture runs
use real local Qwen inference and simulated financial sources.

The deadline bounds how long the runner waits. An already-started synchronous
read-only provider call may finish in the background. The runner discards a late
result and schedules no further actions. Provider-call counts reflect the tool
adapters' reported counters, which can include fixture invocations or exclude
internal HTTP retries. Token totals use the counters reported by Ollama.

## Safeguards and stop reasons

- Dispatch accepts only market data, calculator, SEC EDGAR, and DOJ title search.
  Provider interfaces construct the URLs. SEC excerpts require a filing already
  present in state. Argument fingerprints suppress equivalent actions, including
  failed attempts. A genuinely different query or date range has a different fingerprint.
- The source, action, result, and observation IDs stay connected. Live retrieval,
  simulated input, derived calculations, errors, and incomplete or empty results
  have separate provenance or outcome fields. Single-source retrieval leaves
  independent verification unset. A failed retrieval creates no evidence.
- Price-dependent hypotheses start unverified. A conflicting price premise leaves
  the dependent hypotheses inconclusive with supporting observation IDs for that
  reassessment. Partial SEC excerpts cannot establish absence.
- “Goal satisfied” requires a tool-free stop with scoped provider evidence for the
  supported factual goal types. Causal explanations remain provisional. Missing
  intent or a conflicting premise stops for clarification. Missing coverage,
  insufficient evidence, repeated proposals, exhausted budgets, tool failure,
  LLM failure, and execution errors have distinct stop reasons.
- The existing ignored `.env` remained in use. Ollama was available on
  `127.0.0.1:11434`, and the installed `qwen3:8b` was used without a separate warm-up request. There is no
  paid-provider fallback. No dependencies were added.

## Offline verification

`.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider -q` and
`.venv/Scripts/python.exe -B -m pip check` passed. All 221 tests passed, including the original 199. The M4 suite includes a
four-decision test that dispatches price history, a calculation, and SEC filings in
response to changing observations. It checks premise contradiction, provenance,
invalid references, empty and truncated results, duplicate suppression, provider
recovery, LLM attempt limits, and serialization after interruption.

## Live Qwen3 8B validation

| Scenario | Result | Stop reason | Steps / tools / LLM attempts | Input / output tokens | Seconds |
| --- | --- | --- | ---: | ---: | ---: |
| Yahoo GOOGL price history | Yahoo reported +1.83%. The runner requested clarification. | `needs_clarification` | 2 / 1 / 4 | 5231 / 891 | 59.26 |
| DOJ historical title search | Eight titles retrieved. Two equivalent proposals were blocked. | `repeated_action` | 3 / 1 / 4 | 5964 / 657 | 40.76 |
| Simulated decline and SEC lookup | Price history and SEC fixture both executed. The step limit ended research. | `budget_exhausted` | 4 / 2 / 6 | 10093 / 1315 | 46.28 |
| Simulated provider failure | One failed tool attempt retained. Two equivalent proposals were blocked. | `repeated_action` | 3 / 1 / 5 | 6852 / 1024 | 33.24 |


### Action, observation, and replanning traces

- **Price contradiction.** Qwen selected scoped market data. Yahoo returned
  31 sessions from August 28 to October 12, 2026. Adjusted closes rose from
  346.367 to 352.72. The observation linked the action and source with the
  calculated +1.83% return. Qwen then proposed another market call. Deterministic
  validation blocked it, asked for clarification, and reassessed four dependent
  hypotheses as inconclusive. [Yahoo Finance GOOGL price history](https://finance.yahoo.com/quote/GOOGL/history/)
- **Regulatory evidence.** Qwen selected the DOJ tool and the literal title term
  `Google` for January 1, 2024 through October 12, 2026. The API returned eight
  titles in the inspected results. State retained their metadata and source URLs,
  with explicit evidence references for the leading three records. Qwen repeated
  the query twice, so the runner stopped with the retrieved results intact.
  [Example DOJ record](https://www.justice.gov/opa/pr/department-justice-again-wins-substantial-relief-against-google)
- **Confirmed decline fixture.** A simulated GOOGL close from 100 to 94 supported
  the premise within the fixture. Qwen first proposed malformed market-data
  arguments. The runner rejected them, then executed Qwen’s valid SEC lookup and
  observed one simulated 10-K record. Qwen’s next market request was a duplicate.
  The run reached its step limit with both observations and the hypotheses still
  unverified. Filing metadata alone did not establish a cause.
- **Failure fixture.** The simulated price provider returned
  `PROVIDER_UNAVAILABLE`. The runner recorded the failed observation with no
  evidence and blocked both repeated proposals.

Initial network-restricted runs recorded provider-unavailable errors for Yahoo
and DOJ. Both succeeded on the bounded network-enabled retry. An exploratory
decline fixture produced a benchmark comparison and then repeated it. The final
fixture restricts simulated symbols to its supported data and prompts the planner
to examine the next causal gap after price verification. These runs demonstrate
integration behavior on selected scenarios. Some generated hypotheses still
phrased the price premise as if it had already happened. State kept them
provisional, and none of these runs establishes general model reliability.

## Repository changes

New modules `agent/budget.py`, `agent/dispatch.py`, `agent/observe.py`, and
`agent/runner.py` under `src/financial_research_agent` separate per-run limits,
provider dispatch, observations, and orchestration. Updated `agent/planning.py`,
`agent/hypotheses.py`, `agent/state.py`, `models.py`, `providers/ollama.py`, and
`tools/planning.py` in the same package to reuse the existing contracts. Added
`examples/autonomous_research_demo.py`, `tests/test_m4_runner.py`, and
`docs/m4_validation.md`. Updated `README.md` with M4 usage.

M4 changes are unstaged on `main`. No commits, pushes, or paid API requests were
made. Suggested commit message is `feat: add bounded evidence-driven research execution (M4)`.
