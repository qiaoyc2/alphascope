# AlphaScope repository instructions

- Build a free-first, evidence-backed financial research agent with a manually controlled decision loop.
- Keep research state and tool contracts independent of LLM and data provider implementations.
- Use Python 3.11+ type hints and Pydantic v2 validation. Prefer small, explicit state operations.
- Preserve source provenance; never treat unverified tool output as established evidence.
- Run offline unit tests with `.venv/bin/python -m pytest`.
- Work only on the approved milestone; add data providers, the LLM, the agent loop, and interfaces when their milestones are approved.
- Do not commit or push unless the user requests it.
