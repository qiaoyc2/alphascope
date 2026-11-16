# AlphaScope M5.1 validation

## Stopping and recovery

The runner assesses the original goal after intent extraction and after recording
each usable tool result. It applies the following priority.

1. Terminal dispatch or state-integrity errors, elapsed deadlines, and
   interruptions retain their stop reason and the last usable evidence. Late
   provider results are discarded. A provider failure at the existing failure
   limit stops the run.
2. Unclear intent, incompatible comparable observations, or a contradicted
   price premise require clarification. A contradicted premise carries the
   recorded observation IDs, provenance, and independent-verification status.
3. The gate can complete a scoped endpoint return or a bounded DOJ title
   lookup using validated evidence from the requested period and subject.
   This check runs before another model call or a stop for consumed step,
   tool, or LLM-request budgets. Filing metadata retains its existing
   proposal-driven stopping path. Causal findings still require causal support.
4. Unresolved work proceeds within the existing execution limits. A duplicate
   replan requires an enabled source that can address a remaining gap, and a
   known scoped market request must still pass authorization. A blocked
   duplicate with no such alternative stops as `repeated_action`.

Price findings identify the effective sessions and calculation basis. A price
series with materially missing endpoints can still support a partial finding.
The completion rule allows up to four calendar days between a requested endpoint
and its available session. There is no exchange-calendar or session continuity
verification.

The gate checks that returned DOJ titles and source references match the lookup.
Live title records must identify official DOJ press releases and use DOJ source
URLs. Every returned title can carry a separate linked finding. Repeated titles
count once, and the gate prefers the evidence from a retrieval that matches the
requested period. The findings and sources disclose search terms, result caps,
and the provider's inspected-page limitations. Title metadata leaves questions
about release content, comprehensive coverage, and causation unresolved.

## Planner context and decision telemetry

The state summary includes the original question, resolved scope, conditional
premise, provenance, a prioritized remaining need, and bounded hypotheses.
Attempted actions include canonical arguments and the fingerprint when
validated, a compact retrieval outcome, useful evidence and observation IDs,
source limitations, and failure codes. The summary prioritizes premise and
conflict records, then failures and recent actions. It reports omitted attempts.
Large argument payloads can be represented by their fingerprint. SEC excerpts
contribute coverage and truncation metadata, and known filings contribute
accession numbers. Excerpt text stays out of planning prompts.

The action registry still validates and authorizes every dispatch. The runner
records each raw recommendation with independent schema, contract, and
duplicate fields. A duplicate can pass the schema and tool contract while
failing raw executable-action validity. Structured corrections and rejected
proposals remain distinct from executed tools. An independent gate stop is
recorded at the run level and, when applicable, on the preceding tool step.
An accepted retrieval contributes to raw validity. The resulting automatic
clarification or completion does not count as a raw model recommendation to stop.

## Offline verification

The M5 checkout at `8bbcb68` passed all 249 original tests and reproduced the
11/11 frozen offline baseline. M5.1 passed 266 tests and dependency checks.
Changes to M4/M5 test inputs and assertions keep the existing duplicate, failure,
budget, and clarification checks meaningful under proactive stopping. Seventeen
new tests exercise the actual runner, including replays of comparable conflicting
captures at the observation boundary. The registry still blocks duplicate
dispatch, so the frozen E4 quality-only case remains unchanged.

`evals/cases.py` and `docs/m5_validation.md` retain their M5 contents.
Frozen E6 expects two equivalent source results, duplicate proposals, and
`repeated_action` after a sufficient lookup. With M5.1, the first scoped
retrieval completes the question. The frozen checks therefore fail E6.
Explicit `--expectations m5.1` expects E6 to complete after one call with linked
findings and disclosed coverage. The other ten cases use their original
expectations. Separate runner tests still take the duplicate path under an
unresolved goal and show a different lookup can make progress.

| Offline case | M5 reproduced | M5.1 with frozen M5 expectations | M5.1 with versioned expectations |
| --- | --- | --- | --- |
| E1 contradicted decline | Pass | Pass | Pass |
| E2 unknown cause | Pass | Pass | Pass |
| E3 historical return | Pass | Pass | Pass |
| E4 comparable conflicts | Pass | Pass | Pass |
| E5 truncated filing excerpt | Pass | Pass | Pass |
| E6 duplicate source and action | Pass | Fail, stopped on sufficient titles | Pass |
| E7 private communications | Pass | Pass | Pass |
| E8 failed provider | Pass | Pass | Pass |
| E9 timing versus causation | Pass | Pass | Pass |
| E10 inappropriate initial tool | Pass | Pass | Pass |
| E11 bounded title retrieval | Pass | Pass | Pass |

| Offline metric | M5 reproduced | M5.1, frozen | M5.1, versioned |
| --- | ---: | ---: | ---: |
| Cases passed and correct stopping | 11/11 | 10/11 | 11/11 |
| Schema-valid generations | 40/40 | 34/34 | 34/34 |
| Raw valid tool proposals | 14/18 | 13/14 | 13/14 |
| Duplicate proposals | 2/18 | 0/14 | 0/14 |
| Authorized dispatches | 14/14 | 13/13 | 13/13 |
| Successful dispatches | 13/14 | 12/13 | 12/13 |
| Structured interventions, including gate vetoes | 6 (2 vetoes) | 3 (2 vetoes) | 3 (2 vetoes) |
| Independent gate stops, including completions | Unavailable | 4 (3 completions) | 4 (3 completions) |
| Evidence-unsupported completions | 0/11 | 0/11 | 0/11 |
| LLM attempts | 40 | 34 | 34 |

The harness exposes the M5 expectation-linked premature-satisfaction formula
as `unexpected_satisfactions`. It counts E6 once with the frozen expectations.
`premature_satisfactions` measures completion that lacks gate support,
provenance, or the required findings, or includes prohibited findings.

## Local Qwen3 8B evaluation

The M5 local numbers are report-derived from the historical M5 validation.
Original local traces were unavailable. The M5 and M5.1 cases, fixture data,
and limits match. An intermediate M5.1 fixture run passed 6/8. Both E3 trials
failed before dispatch because the planner generated invalid market arguments.
The final prompt gives the exact stock-history argument contract without
relaxing validation. A targeted E3 probe passed before the full rerun.

| Fixture case and trial | M5 | Final M5.1 | M5.1 tools / attempts | M5.1 input / output tokens | M5.1 seconds |
| --- | --- | --- | ---: | ---: | ---: |
| E1 / 1 | Fail | Pass, clarification | 1 / 3 | 2760 / 672 | 24.27 |
| E2 / 1 | Pass | Pass, repeated action | 2 / 5 | 6840 / 1132 | 37.72 |
| E3 / 1 | Pass | Pass, goal satisfied | 1 / 2 | 2045 / 359 | 12.01 |
| E11 / 1 | Fail | Pass, goal satisfied | 1 / 2 | 2024 / 331 | 11.03 |
| E1 / 2 | Pass | Pass, clarification | 1 / 3 | 2760 / 673 | 21.81 |
| E2 / 2 | Pass | Fail, invalid proposal | 2 / 6 | 8995 / 1354 | 45.19 |
| E3 / 2 | Pass | Pass, goal satisfied | 1 / 2 | 2045 / 359 | 12.03 |
| E11 / 2 | Fail | Pass, goal satisfied | 1 / 2 | 2024 / 331 | 11.00 |

Both E1 trials stopped on the +3% simulated observation. Both E3 trials
completed on the +6% simulated endpoint return. E11 preserved scoped title
metadata with coverage limitations. E2 trial 2 retained the confirmed simulated
decline and an empty DOJ search. Two invalid tool proposals ended the run
with `invalid_proposal`, outside the frozen expected stop set.

| Fixture metric | M5 report-derived | Final M5.1 |
| --- | ---: | ---: |
| Cases passed and correct stopping | 5/8 | 7/8 |
| Schema-valid generations | 30/30 | 25/25 |
| Raw valid tool proposals | 10/20 | 10/13 |
| Duplicate proposals | 9/20 | 1/13 |
| Authorized dispatches | 10/10 | 10/10 |
| Successful dispatches | 9/10 | 10/10 |
| Structured interventions | At least 10 from reported tool corrections | 3 |
| Independent gate stops and completions | Unavailable | 6 stops, 4 completions |
| Evidence-unsupported completions | 0/8 | 0/8 |
| LLM attempts | 31 | 25 |
| Input / output tokens | 44,363 / 6,402 across 30 records | 29,493 / 5,211 across 25 records |
| Model request seconds, mean (range) | 5.94 (3.43–11.39), 30 attempts | 7.00 (4.31–11.61), 25 attempts |

Raw validity and duplicate rates measure the proposals that actually reached
assessment. Earlier evidence stops also remove opportunities to propose
redundant actions. These samples describe the measured runs and do not
establish statistical significance.

### Live Yahoo Finance and DOJ

The first run under restricted networking passed 0/2. Both providers
returned `PROVIDER_UNAVAILABLE`; Yahoo recorded a DNS timeout. That run
made two authorized dispatches, both failed, and made seven Ollama requests.
The Yahoo and DOJ tool waits were 60.06 s and 1.04 s respectively. Neither
case completed. The table compares the separately measured network-enabled
retry with the report-derived M5 live cohort.

| Live case | M5 | M5.1 retry | M5.1 tools / attempts | M5.1 input / output tokens | M5.1 seconds |
| --- | --- | --- | ---: | ---: | ---: |
| E3 Yahoo | Pass | Pass, goal satisfied | 1 / 2 | 2044 / 362 | 14.95 |
| E11 DOJ | Fail, repeated action | Pass, goal satisfied | 1 / 2 | 2024 / 331 | 11.44 |

Yahoo reported a +1.83% adjusted-close return across 31 sessions. The DOJ
API returned eight matching titles and the gate retained eight linked
findings with the inspected-page limitation. Both runs completed on the
first recorded result. The live retry had no in-flight tool workers at exit.
The Yahoo dependency emitted pandas deprecation warnings and returned
usable data.

| Live metric | M5 report-derived | M5.1 retry |
| --- | ---: | ---: |
| Cases passed and correct stopping | 1/2 | 2/2 |
| Schema-valid generations | 7/7 | 4/4 |
| Raw valid tool proposals | 2/5 | 2/2 |
| Duplicate proposals | 3/5 | 0/2 |
| Authorized and successful dispatches | 2/2 | 2/2 |
| Structured interventions | At least 3 from reported duplicate rejections | 0 |
| Independent gate stops and completions | Unavailable | 2 stops, 2 completions |
| Evidence-unsupported completions | 0/2 | 0/2 |
| LLM attempts | 7 | 4 |
| Input / output tokens | 9,812 / 1,275 across 7 records | 4,068 / 693 across 4 records |
| Model request seconds, range | 3.75–6.18 across 7 attempts | 4.39–7.59 across 4 attempts |
| Tool wait seconds, range | Unavailable | 0.37–0.65 across 2 dispatches |

## Prompt size and runtime limits

The frozen offline scripts sent 25 planning prompts on M5 and 19 on M5.1.
Over 19 matched case and planning-step positions, mean rendered planner prompt
size fell from 3,657 to 2,952 characters (19.3%). The maximum on those positions
fell from 4,920 to 4,323 characters. Mean JSON state summary size fell from
1,233 to 1,175 characters (4.7%), with maxima of 2,284 and 2,261. These counts
exclude the separate schema sent to the model. The local token totals above
come from Ollama and include goal extraction, hypothesis generation, and
planning. Context and model configuration stayed fixed.

Model-request latency comes from Ollama adapter HTTP attempts, and
`tool_wait_seconds` measures how long the runner waited on the tool adapter.
An in-flight tool flag records whether the worker was still running when the
runner stopped waiting. All ten completed fixture tool waits were under
1 ms. Wall time also includes local processing, retries, and scheduling.
The M5 report identified one Windows Modern Standby overlap. The new fixture
cohort returned in 11.00–45.19 seconds per trial, with no comparable wall-time
overrun. Scheduler delays cannot be isolated from the elapsed counters alone.
The existing synchronous provider workers may continue after a deadline or
interrupt. The runner discards late results. The deadline regression exercises
that path without rewriting the worker architecture.

## Reproduction

Run in PowerShell from the repository with its existing Windows virtual
environment. The frozen M5 comparison under the updated runner returns exit
code 1 for E6. Run the same default command on the historical M5 checkout for
the independently reproduced offline baseline.

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q
.\.venv\Scripts\python.exe -B -m pip check
.\.venv\Scripts\python.exe -B evals\run_evals.py --mode offline --json m5_1-frozen.json
.\.venv\Scripts\python.exe -B evals\run_evals.py --mode offline --expectations m5.1 --json m5_1-offline.json
.\.venv\Scripts\python.exe -B evals\run_evals.py --mode local --repeat 2 --json m5_1-local.json
.\.venv\Scripts\python.exe -B evals\run_evals.py --mode local --live-sources --repeat 1 --case E3 --case E11 --json m5_1-live.json
```

If yfinance needs a writable cache, call `yfinance.set_tz_cache_location(path)`
before calling `evals.run_evals.main`. The local settings used the existing
ignored `.env`, loopback Ollama, `qwen3:8b`, temperature 0, and a 4,096-token
context. The three-tool-call, four-step, seven-request, 105-second case
limits are unchanged for the local comparison. No model was downloaded and
there were no paid inference calls.
