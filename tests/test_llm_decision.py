from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from financial_research_agent.llm_decision import LLMDecision
from financial_research_agent.tools import ToolName
from financial_research_agent.tools.calculator import CalculatorTool
from financial_research_agent.tools.market_data import MarketDataTool
from financial_research_agent.tools.news import NewsTool
from financial_research_agent.tools.sec_filings import SecFilingsTool


def decision_data(**changes: object) -> dict:
    return {
        "premise_valid": None,
        "recommended_action": "continue",
        "next_tool": "market_data",
        "reason": "No price observations have been supplied to verify the premise.",
        "information_gap": "Price movement over the intended research period.",
        "should_stop": False,
        **changes,
    }


def test_valid_decision_and_tool_contract_names():
    result = LLMDecision.model_validate(decision_data())
    assert result.premise_valid is None
    assert result.next_tool == MarketDataTool.name
    assert set(ToolName) == {tool.name for tool in (MarketDataTool, CalculatorTool, SecFilingsTool, NewsTool)}


@pytest.mark.parametrize("changes", [
    {"next_tool": "shell"}, {"next_tool": "doj_press_releases"},
    {"recommended_action": "execute"}, {"premise_valid": "false"},
    {"should_stop": 0}, {"reason": " "}, {"reason": "x" * 1201},
    {"extra_command": "python"}, {"recommended_action": "stop"},
    {"next_tool": None}, {"recommended_action": "clarify"},
    {"recommended_action": "clarify", "next_tool": None, "information_gap": None},
])
def test_invalid_decisions_are_rejected(changes):
    with pytest.raises(ValidationError):
        LLMDecision.model_validate_json(json.dumps(decision_data(**changes)))


@pytest.mark.parametrize("field", list(decision_data()))
def test_every_decision_field_is_required(field):
    data = decision_data()
    del data[field]
    with pytest.raises(ValidationError):
        LLMDecision.model_validate(data)


@pytest.mark.parametrize("action,stopped", [("clarify", False), ("stop", True), ("revise", False)])
def test_pausing_or_revising_without_tool_is_valid(action, stopped):
    decision = LLMDecision.model_validate(decision_data(
        recommended_action=action, next_tool=None, should_stop=stopped,
        premise_valid=False, reason="Supplied fixture shows an increase, with relative underperformance.",
        information_gap="Does the user mean relative performance or a different period?",
    ))
    assert decision.next_tool is None
