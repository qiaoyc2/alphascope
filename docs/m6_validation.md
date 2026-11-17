# AlphaScope M6 validation

## Report architecture

`build_research_report(run)` validates the full `ResearchRun` and reassesses
its state with the M5 quality gate. The report keeps the original question,
subject, period, quality status, research status, and recorded stop reason.
Completion requires both a completed run and scoped evidence sufficient for
the original objective. Partial facts survive provider, planning, and budget
stops. A contradictory premise leads the report with the observed direction
and requests clarification. Comparable conflicts retain both measurements.

`generate_research_report(run, provider)` first builds that deterministic
report. It then asks the existing loopback provider for a Pydantic
`NarrativeDraft`. The draft can select an eligible lead and order findings,
hypotheses, and gaps using short references. Python renders the selected
sections from the validated records and retains every remaining fact and
caveat. The model's schema has no fields for prose, prices, dates, URLs,
evidence records, or causal conclusions. The prompt uses a bounded reference
list with short previews and omits raw tool payloads and SEC excerpt text.

Python checks the draft's roles, references, and section membership, then
compares the resulting report with the original grounded projection. It
checks that the order changes preserve all conclusions, figures, citations,
scope, and limitations. `validate_research_report(report, run)` provides the
same comparison for a previously serialized report. A valid citation alone
does not establish semantic support. The M5 quality gate and its source
contracts determine which facts may enter the report. Hypotheses keep the
planner's assessment and linked observation IDs while remaining explicitly
unverified in the report.

Drafting allows up to two HTTP attempts and 25 seconds. The existing Ollama
adapter retries transient transport failures within that budget. A malformed
draft, unknown reference, unavailable model, or expired budget returns the
deterministic report with a fallback explanation. A late synchronous call can
continue in the existing provider worker. Rendering itself makes no model or
provider calls.

## JSON schema and source trace

The `1.0` schema includes a stable report ID, generation timestamp, scope,
research outcome, findings, disputed measurements, hypotheses, open questions,
coverage, bibliography, methodology, typed metrics, and narrative metadata.
Narrative metadata records validation outcome, fallback reason, request
count, prompt bytes, elapsed time, and tokens when Ollama reports them.

Each finding links to checked evidence and numbered bibliography entries.
Each entry retains the original observation, result, and evidence IDs.
Repeated URLs within the same provenance class share a bibliography entry
and retain their individual traces. URLs come from the validated tool records.
Source URLs carrying user information or recognized credential query fields
are omitted. A calculation without a public URL retains its replayed result
and evidence trace.

Market metrics carry requested and effective dates, adjusted or raw closing
price basis, endpoints, return, currency when known, and session count.
Benchmark metrics preserve both source references, aligned dates, and the
return difference in percentage points. The report reuses the deterministic
return formulas for consistency checks. The calculator's scalar result and
unit come from the gate's replayed arithmetic. Calculator source IDs attach
as market provenance only when the reported operands and scope match a checked
market result. Matching covers endpoint returns, relative performance from a
benchmark comparison, and volatility computed from a recorded adjusted-close
series. Other calculations can still report their arithmetic result and unit.

Live retrieval, simulated and historical fixtures, user-supplied material, and
derived arithmetic keep separate provenance. Single-source retrieval leaves
independent verification unrecorded. An unretrieved document, empty title
search, or truncated filing leaves its evidence gap open. The current report
does not establish general causal validity or complete regulatory coverage.

## Validation results

The prior M5.1 baseline had **266** offline tests. The completed M6 work passes
**301** tests, including **35** reporting tests, and the dependency check.
The M5.1 versioned suite passes **11/11**. The original frozen expectations
still score **10/11** on the updated runner because E6 expects continued
planning after a sufficient title lookup. Earlier evaluation cases and
milestone documentation retain their original contents.

The report harness uses the existing runner and frozen fixtures. The E4
conflict replays captured state, since equivalent live dispatch is blocked
by the registry. Offline draft responses are scripted. The local cohort uses
scripted fixture research followed by real Qwen3 8B drafting, so the measured
model time and token usage belong to reporting. The fixture values and titles
below are simulated.

| Report metric | Offline | Local Qwen3 8B |
| --- | ---: | ---: |
| Passed cases | 13/13 | 4/4 |
| Valid bibliography entries | 15/15 | 4/4 |
| Valid evidence references | 15/15 | 4/4 |
| Correct fixture return checks | 13/13 | 4/4 |
| Complete grounded projections and correct outcomes | 13/13 | 4/4 |
| Unsupported emitted factual claims detected | 0 | 0 |
| Unsupported emitted causal conclusions detected | 0 | 0 |
| Rejected narrative drafts | 2/13 | 0/4 |
| Deterministic fallbacks | 3/13 | 0/4 |
| Report requests | 13 scripted | 4 loopback Ollama |
| Report generation wall time | 0.013 s | 8.244 s |
| Input / output tokens | Unavailable | 2,099 / 126 |

The reference check validates each citation URL against the linked result and
each evidence reference against a successful M5 check and its observation.
The financial check compares report returns with the frozen fixture values
or expects an empty market-metric list when the case has none. The
completeness check compares the structured report to the rebuilt projection.
Unsupported factual counts measure emitted findings absent from the quality
gate's supported set. Unsupported causal counts measure causal conclusions
emitted as supported. These checks evaluate the implemented rules and are
insufficient to establish general semantic correctness.

The deliberate offline failures were an unknown finding reference, extra
fields containing a fabricated number and URL, and a simulated connection
failure. All three fell back with their original evidence and stopping
outcome. Tests also cover missing fields, duplicate references, invented
causal claims, timeout, wrong dates, raw and adjusted price bases,
percentage-point errors, and attempted export over an existing file.
No unauthorized tool dispatches or paid API requests occurred.

| Local case | Outcome | Citations | Report seconds | Input / output tokens | Prompt bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| E3 historical return | Sufficient, goal satisfied | 1 | 5.695 | 514 / 27 | 872 |
| E1 contradicted premise | Partial, clarification required | 1 | 0.884 | 551 / 34 | 1056 |
| E11 DOJ titles | Sufficient, goal satisfied | 1 | 0.793 | 489 / 31 | 876 |
| E2 decline, cause unknown | Partial, insufficient evidence | 1 | 0.871 | 545 / 34 | 1033 |

Report wall time covers projection, drafting, and validation. The token
counters cover the Ollama request, including schema overhead. The configured
model remained `qwen3:8b` with a 4,096-token context and temperature 0.

## Representative report excerpts

- **E3 historical return.** The report shows a +6.00% simulated adjusted-close
  return from 100 to 106 over 2026-08-28 through 2026-10-12. The cited price
  record carries its fixture provenance and two endpoint sessions.
- **E1 contradicted premise.** The +3.00% simulated return contradicts the
  assumed decline. The report asks whether to follow the observed movement or
  use a different period or premise.
- **E11 DOJ titles.** The report cites the one simulated returned title and
  discloses the inspected-result limit. Its claim is scoped to title metadata.
- **E2 unresolved cause.** The -6.00% simulated movement remains supported.
  Candidate explanations remain unverified, and the DOJ search returned no
  records within the inspected coverage.

## M6 report-quality follow-up

Markdown renders reserved placeholder domains as plain citation references.
Fixture bibliography entries identify the fixture and its evidence IDs.
The structured report retains the recorded source URL and provenance.

Price reports show the requested interval separately from the dates and
sessions returned by the provider. Two valid closing observations can
satisfy a narrowly phrased endpoint-return question, such as E3. They
leave the intervening price path unobserved and broader historical questions
unresolved. The report assesses endpoint alignment without checking full
trading-calendar coverage.

The E3 and E2 fixture reports were regenerated from the existing offline
cases. A separate Yahoo Finance retrieval through the existing adapter
returned 31 GOOGL sessions from 2026-08-28 through 2026-10-12. Its adjusted
close moved from 346.367 to 352.72, a +1.83% endpoint return. The Yahoo report
records live-provider provenance and a public source link. Regeneration used
no model or paid API calls. All 305 offline tests passed.

To replay the fixture cases and export fresh reports, run the report
evaluation with `--mode offline --case E3 --case E2 --report-dir outputs/reports`.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q
.\.venv\Scripts\python.exe -B -m pip check
.\.venv\Scripts\python.exe -B evals\run_evals.py --mode offline --expectations m5.1
.\.venv\Scripts\python.exe -B evals\run_report_evals.py --mode offline --json m6-offline.json
.\.venv\Scripts\python.exe -B evals\run_report_evals.py --mode local --report-dir reports --json m6-local.json
.\.venv\Scripts\python.exe -X utf8 examples\autonomous_research_demo.py --scenario fixture-return --report
.\.venv\Scripts\python.exe -X utf8 examples\autonomous_research_demo.py --scenario fixture-doj --report-json --output-dir reports
.\.venv\Scripts\python.exe -X utf8 examples\autonomous_research_demo.py --from-run saved-run.json --deterministic-report --report --output-dir reports
```

Python callers can pass an in-memory `ResearchRun` to the builder and use
`render_report_markdown(report)` or `report.model_dump_json()`. To replay a
run through the CLI, use a full `ResearchRun.model_dump_json()` capture as
`--from-run` input. The existing `--json` diagnostic trace elides tool
payloads. File exports use a generated name and an exclusive create by
default. Use `--overwrite` to explicitly replace a report for the same run.
