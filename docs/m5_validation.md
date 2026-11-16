# AlphaScope M5 validation

## Architecture

`agent.quality.evaluate_quality(state, available_tools=...)` consumes the existing
`ResearchState` and returns a validated `QualityAssessment`. Useful observations
and results remain in the state on every exit. Source references use the URLs
in `ToolResult.sources`, including title and publication metadata where available.
The gate checks the observation, result and completed action links, the registered
tool contract, and correspondence between each evidence claim and its structured
result. Invalid records do not contribute to support.

- Market returns are recomputed from available endpoint prices. Contradiction
  checks require matching ticker, requested and effective dates, price basis,
  adjustment convention, and benchmark where applicable. Comparable values
  conflict when their difference exceeds 0.01 percentage points or their direction
  differs. Incompatible scopes can leave a goal ambiguous.
- Equivalent observations and evidence are marked separately. The same source
  and fact contribute once. Distinct sources remain visible. Contradictory
  values from a repeated source remain visible, too.
- The completeness model distinguishes complete structured results, partial
  results, truncated excerpts, empty results, retrieval failures, and unvalidated
  results. Relevant price and filing metadata can support narrowly scoped facts.
  DOJ results support the publication of retrieved titles in scope. A release
  date can support overlap with a price interval. An asserted event detail or
  causal attribution requires further evidence.
- A calculation can support the output obtained from the supplied operands.
  Replaying it does not establish that the operands represent market data.
  Hypothesis statements and proposed links stay provisional. Semantic relevance
  that cannot be established by a scope or entity check remains unknown.

The runner retains the M4 stop reasons and the existing tool authorization and
budget checks. The gate can block a satisfaction request, identify a false
price premise that requires clarification, or stop when the approved tools do
not cover the remaining need. A factual price return with sufficient evidence
can stop after a redundant proposal. A positive title or filing metadata lookup
has a narrower completion rule and can still end with a repeated-action stop
if the planner keeps asking for the same lookup. A tool failure or a budget
stop retains validated partial findings.

Each planning step records the allowlisted raw action and registered tool,
validation and correction, guarded disposition, authorized tool, dispatch
status, LLM performance, and remaining budget. A schema-invalid response can
contribute allowlisted metadata when it contains parseable JSON. The evaluator's
JSON excludes prompts, model rationale, hidden reasoning, and raw tool result
bodies. It includes bounded findings, source URLs, error categories, provenance
and truncation metadata.

## Offline verification

All **249 tests** passed, including the **221 M1–M4 tests** and 28 M5 tests.
Dependency verification also passed. Eleven evaluation cases passed.
The E4 conflict check replays captured results directly into the same state
operations because the running agent deliberately suppresses equivalent dispatch.
The remaining cases exercise the existing runner.

| Case | Expected behavior observed |
| --- | --- |
| E1 | Positive fixture return contradicted the decline premise and stopped for clarification |
| E2 | Decline and empty news response produced an unresolved causal result |
| E3 | Recomputed historical return and stopped with sufficient evidence |
| E4 | Comparable incompatible returns were retained and flagged |
| E5 | Truncation and the disclosure gap were preserved after a completion veto |
| E6 | Same source counted once and equivalent proposals were blocked |
| E7 | Private communications required an authorized source or supplied record |
| E8 | Provider failure preserved the earlier price evidence |
| E9 | Release timing was recorded and a causal completion request was vetoed |
| E10 | A news proposal before premise verification was blocked and market data was dispatched |
| E11 | Scoped title findings retained the DOJ coverage limitation |

The offline run had 40 schema-valid scripted generations, 14 of 18 assessed raw
tool proposals valid, and 14 of 14 final dispatches authorized. Thirteen
dispatches succeeded. Two duplicate proposals and one duplicate observation
were detected. All 12 evidence records passed provenance checks. One comparable
contradiction was recorded. Two raw requests for premature satisfaction were
vetoed, and there were zero final premature satisfactions. Offline generation
counts describe script invocations, and token usage and inference latency are
unavailable. Repeated offline runs produce the same behavioral checks. Runtime
timestamps, generated IDs, and elapsed times can differ.

## Local Qwen3 8B with fixture data

Two trials per case used the same fixed questions, limits, and expectations.
Qwen3 `qwen3:8b` was called through loopback Ollama. All resulting prices and
titles in this table came from simulated sources. The expectations and prompts
were held fixed during validation.

| Case | Trial | Quality / stop | Tools / LLM attempts | Input / output tokens reported | Run seconds |
| --- | ---: | --- | ---: | ---: | ---: |
| E1 premise conflict | 1 | Insufficient / budget exhausted | 0 / 1 | Unavailable | 500.74 |
| E2 decline, unknown cause | 1 | Partial / budget exhausted | 3 / 6 | 10,065 / 1,467 | 48.28 |
| E3 historical return | 1 | Sufficient / goal satisfied | 1 / 3 | 3,942 / 610 | 16.20 |
| E11 DOJ title scope | 1 | Sufficient / repeated action | 1 / 4 | 5,493 / 665 | 17.44 |
| E1 premise conflict | 2 | Partial / needs clarification | 1 / 4 | 5,245 / 958 | 25.17 |
| E2 decline, unknown cause | 2 | Partial / repeated action | 2 / 6 | 10,173 / 1,416 | 37.46 |
| E3 historical return | 2 | Sufficient / goal satisfied | 1 / 3 | 3,942 / 604 | 15.96 |
| E11 DOJ title scope | 2 | Sufficient / repeated action | 1 / 4 | 5,503 / 682 | 17.92 |

Five of eight trials met expectations. Both E11 trials found sufficient title
metadata, then proposed equivalent searches twice and failed the expected stop
check. Both E3 trials verified the +6.00% fixture return and blocked the model's
second market request before completing. E1 trial 2 verified the +3.00% fixture
return. The model proposed another market call and the guard converted it to
`needs_clarification` without dispatch. E2 found a −6.00% fixture return and
empty release searches in both trials. One trial also tried a market comparison
with a symbol outside fixture coverage, and the failed result remained in state.
Neither trial established a cause.

E1 trial 1 overlapped Windows Modern Standby, as recorded in the system power
event log. The process returned after resume, exceeded the configured 105-second
wall budget, and retained no generation or tool result. It is included in the
denominator. The other seven runs took 15.96–48.28 seconds. Thirty generations
had recorded metrics and passed schema validation. Those generations reported
44,363 input and 6,402 output tokens. The 30 measured HTTP attempts took
3.43–11.39 seconds each (mean 5.94). The 31st attempt had no returned metrics.
There were zero retries in completed generation records.

Raw tool proposals were valid in **10/20** assessed cases. Nine were duplicates
and one conflicted with the verified premise. Final dispatches were authorized
in **10/10**, and **9/10** returned successful structured results. Zero trial
outcomes prematurely claimed completion. Local inference incurred **$0 in paid
API charges**. The sample describes observed variation and does not estimate
general model reliability.

## Local Qwen3 8B with live Yahoo and DOJ sources

The second cohort used the same E3 and E11 questions and expectations, with one
trial per case and live sources enabled. Yahoo's default local database path
was unavailable on the initial validation attempt. The Yahoo adapter now
records that failure as `PROVIDER_UNAVAILABLE`. The measured retry used a
writable yfinance cache. A mocked regression test covers the failure path.

| Case and source | Retrieved evidence | Quality / stop | Tools / LLM attempts | Input / output tokens | Seconds |
| --- | --- | --- | ---: | ---: | ---: |
| E3, Yahoo Finance | GOOGL +1.83% from August 28 to October 12, 2026, using adjusted closes over 31 sessions | Sufficient / goal satisfied | 1 / 3 | 3,963 / 601 | 21.00 |
| E11, DOJ press-release API | Eight title matches retained in the result, with evidence records for the leading three | Sufficient / repeated action | 1 / 4 | 5,849 / 674 | 20.76 |

Yahoo and DOJ both returned live provider data. The Yahoo probe met all
expectations and blocked a redundant market request. The DOJ probe failed the
stopping expectation after two duplicate proposals. Seven of seven generations
passed schema validation. Two of five raw tool proposals were valid and two of
two dispatched actions were authorized and succeeded. All four evidence records
passed provenance checks. Across both trials, seven measured HTTP attempts took
3.75–6.18 seconds each and reported 9,812 input and 1,275 output tokens, with
zero reported retries. No trial claimed premature satisfaction. Paid inference
charges were $0.

- [Yahoo Finance GOOGL historical prices](https://finance.yahoo.com/quote/GOOGL/history/)
- [Example DOJ title result](https://www.justice.gov/opa/pr/department-justice-again-wins-substantial-relief-against-google)

## Metric definitions

| Metric | Numerator / denominator or count |
| --- | --- |
| Structured output rate | Schema-valid generations / schema-valid generations and response-validation failures. Connection, provider availability, and unreturned generations are excluded |
| Raw action validity | Raw tool proposals accepted without correction / raw tool proposals with a validation outcome, including allowlisted metadata from decoded schema-invalid responses |
| Final action validity | Dispatched actions that match the authorized registered tool and recorded result / dispatched tool actions |
| Duplicate proposal rate | Assessed raw tool proposals corrected as duplicates / assessed raw tool proposals |
| Tool execution success rate | Successful tool results / dispatched tool actions. Successful empty searches count as executions |
| Correct stopping rate | Cases whose stop and quality match expectations / evaluated cases, including E4's gate-only stop recommendation |
| Gate rejections | Planning steps tagged `quality_gate_rejection` |
| Unsupported claims | Findings tagged `unsupported`. Provisional hypotheses tagged `not_evaluated` have their own count |
| Premature satisfactions | Final satisfaction stops without sufficient gate support or without expected stop, quality, or required findings |
| Provenance and completeness | Valid evidence checks / all evidence records checked. Completeness counts are grouped by observation outcome |
| Duplication and contradiction | Duplicate evidence records, duplicate observations, and comparable contradiction pairs counted separately |
| Attempts, time, and tokens | Reserved LLM attempts, measured HTTP attempt durations, per-run wall time, and provider token totals with contributing record counts. Retry counts represent completed generation records |
| Cost | Paid inference charges for the known local Ollama adapter are zero. Electricity and hardware are outside this metric |

A rate with no applicable denominator is null. A missing provider counter
does not contribute zero tokens. The JSON reports include counts and the
per-case distribution of stopping outcomes, including failures.

## Limits and reproduction

- Market coverage uses available endpoint sessions and has no independent
  exchange-calendar completeness check. Source linkage proves what was retrieved.
  Fixtures remain conditional on their inputs. Legacy observations without
  verifiable result contracts remain available in state for inspection.
- DOJ title search covers inspected official API results and generally has
  dates rather than intraday timestamps. Recent SEC metadata and bounded
  excerpts have partial coverage. The gate does not perform semantic
  verification of arbitrary filing disclosures or prove financial causation.
- Duplicate proposals can consume all planning steps. Open-ended or compound
  questions may remain unresolved even when some facts are verified.
- An already-started synchronous provider request may finish in the background.
  Its late result is discarded. System suspension can delay the runner's
  response until scheduling resumes. Physical counters can therefore contain
  attempts for which no token or latency record returned.

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q
.\.venv\Scripts\python.exe -B -m pip check
.\.venv\Scripts\python.exe -B evals\run_evals.py --mode offline --json m5-offline.json
.\.venv\Scripts\python.exe -B evals\run_evals.py --mode local --repeat 2 --json m5-local.json
.\.venv\Scripts\python.exe -B evals\run_evals.py --mode local --live-sources --repeat 1 --case E3 --case E11 --json m5-live-sources.json
```

The local command exits nonzero when a fixed expectation fails. The report
preserves the results and failed checks for review.
