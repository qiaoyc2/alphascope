"""Local structured decisions only; no financial tools or research-state changes."""

from __future__ import annotations

import argparse
import json

from pydantic import ValidationError

from financial_research_agent.config import load_llm_settings
from financial_research_agent.llm_decision import LLMDecision
from financial_research_agent.prompts.tool_decision import build_tool_decision_prompt
from financial_research_agent.providers.llm import LLMError
from financial_research_agent.providers.ollama import OllamaProvider


QUESTION = "Why did GOOGL stock decline recently?"
FIXTURE_OBSERVATIONS = [
    "Historical test fixture from the prior demonstration, evaluated conditionally. "
    "For this scenario, 'recently' refers to the research period below; no live retrieval is requested. "
    "Research period: 2026-08-28 to 2026-10-12. "
    "GOOGL return: +1.83%. QQQ return: +4.24%. "
    "Relative performance: -2.41 percentage points."
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replan", action="store_true", help="Also test contradictory historical fixtures")
    args = parser.parse_args()
    try:
        settings = load_llm_settings()
    except (ValueError, ValidationError):
        print("Invalid LLM configuration; check the documented environment variables.")
        return 1
    scenarios = [("initial", [])]
    if args.replan:
        scenarios.append(("replanning_fixture", FIXTURE_OBSERVATIONS))
    with OllamaProvider(settings) as provider:
        for name, observations in scenarios:
            try:
                result = provider.generate(build_tool_decision_prompt(QUESTION, observations), LLMDecision)
            except LLMError as exc:
                print(json.dumps({"scenario": name, "error": str(exc),
                                  "performance": exc.performance.model_dump(mode="json")}, indent=2))
                return 1
            print(json.dumps({"scenario": name,
                              "observation_source": "user_supplied_historical_test_fixture" if observations else None,
                              **result.model_dump(mode="json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
