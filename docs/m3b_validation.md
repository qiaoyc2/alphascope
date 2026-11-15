# AlphaScope M3B validation

M3B adds goal interpretation, competing hypotheses, and observation-driven planning for human review. The planner can propose and record a tool action, accept a supplied observation, validate provenance and evidence references, and replan. The demo leaves the research tools idle.

## Local configuration

The existing ignored and untracked `.env` uses Ollama on `http://localhost:11434` with `qwen3:8b`, temperature 0, and context 4096. Duplicate active assignments were removed. Unrelated settings and the existing key were preserved. Git diffs, untracked deliverables, and captured live output were checked for configured secrets without printing them.

Ollama 0.40.16 was bound to `127.0.0.1:11434`. The installed model loaded on the first inference request. That request timed out during the initial load and succeeded on the configured retry.

## Implementation

- Goal interpretation preserves the question and temporal phrase. The injected date resolves supported relative periods. “Recently” uses 30 inclusive calendar days, “last week” uses the preceding Monday–Sunday week, and “this month” starts on the first day of the month. Ambiguous symbols and dates remain open.
- Causal goals request two to four competing hypotheses. Hypotheses retain their IDs, and updates require existing observation or evidence IDs. The existing M1 state operations persist validated changes and preserve previous states.
- The planner extends the M3A decision model. Resolved ticker and dates supply the arguments for a scoped initial price check. Other recommendations use the registered tool contracts. Equivalent actions and covered measurements are suppressed. A partial SEC excerpt cannot by itself close a gap or justify rejecting a hypothesis.
- Conditional premise validity, observation provenance, and independent verification are distinct. Same-period price readings that disagree are flagged. An absolute return and a benchmark-relative return are assessed separately.
- The existing structured provider interface and telemetry remain in use. Ollama is the sole runtime provider. Hosted OpenAI inference and per-run cost accounting are deferred. There is no paid network path or automatic fallback. No dependencies were added.

## Offline validation

`.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider -q` passed all 199 tests, including the original 160. `.venv/Scripts/python.exe -B -m pip check` passed. The suite covers ambiguous goals, stable hypothesis IDs, invalid references, different tools, conflicting observations, duplicate actions, unavailable sources, SEC metadata, state serialization, and provider failures.

## Local Qwen3 live validation

Run `.venv/Scripts/python.exe -B examples/research_planning_demo.py` from the checkout. It validates initial price planning, replanning against the supplied historical fixture, DOJ title search selection, and missing source coverage. The fixture is labeled as historical input and has no independent real-world verification. The demo uses no live market, SEC, or DOJ calls.

| Scenario | Final decision | Request seconds | Input / output tokens |
| --- | --- | ---: | ---: |
| Initial stock question | Proposed scoped `market_data`; the premise remained unknown. Three hypotheses remained unverified. | 4.61 | 1675 / 173 |
| Historical fixture replan | Set `premise_valid=false`, cleared the price-data gap, and requested clarification without another retrieval. | 5.91 | 1947 / 212 |
| DOJ regulatory question | Proposed `news` for DOJ title metadata with scoped dates. | 4.73 | 1426 / 178 |
| Private call, no transcript | Stopped with an unsupported source gap and no tool. | 4.92 | 1379 / 187 |


GOOGL returned +1.83% versus QQQ’s +4.24% from August 28 through October 12, 2026, a relative return of −2.41 percentage points. The supplied positive absolute return contradicts the decline premise while supporting relative underperformance. Independent verification stayed false.

### Observed model errors

- In the final run, Qwen proposed an additional market-data action after the fixture contradicted the premise. The deterministic check overrode that proposal. The unsupported-source response included tool details with a null tool. Validation removed those details and recorded the stop decision.
- The DOJ selection was in scope. Its literal title query was `DOJ Google antitrust`, which may be too restrictive for the existing title filter. Provider recall was outside this planning-only run. Some hypothesis text also assumed a decline in its wording, so the hypotheses stayed unverified and the planner sought price evidence first.
- Earlier development runs exposed a stale measured gap, a claim that price history would also return benchmark results, malformed market arguments, and confusion between missing sources and ambiguous intent. Those runs are retained as development findings. The final run follows changes to argument grounding, gap resolution, and decision validation. These samples do not establish a general model success rate.

## Changes and Git status

New files:

- `src/financial_research_agent/agent/goal.py`
- `src/financial_research_agent/agent/hypotheses.py`
- `src/financial_research_agent/agent/planning.py`
- `src/financial_research_agent/agent/premise.py`
- `src/financial_research_agent/tools/planning.py`
- `examples/research_planning_demo.py`
- `tests/test_m3b_planning.py`
- `docs/m3b_validation.md`

Updated `src/financial_research_agent/models.py`, `src/financial_research_agent/agent/state.py`, `src/financial_research_agent/config.py`, `src/financial_research_agent/providers/sec_edgar.py`, `src/financial_research_agent/tools/calculator.py`, `tests/test_sec_edgar.py`, `.env.example`, and the ignored local `.env`.

The branch is `main`. The pre-existing M3A changes are staged and M3B changes are unstaged. No commits or pushes were made.

Suggested commit message after reviewing the staging split is `feat: add observation-driven research planning for M3B`.
