import json

from financial_research_agent.prompts.tool_decision import build_tool_decision_prompt


def test_prompt_preserves_question_and_observation_provenance_as_data():
    question = "Why did another stock move?"
    observations = ["Unverified TEST FIXTURE: +2% over a specified period."]
    prompt = build_tool_decision_prompt(question, observations)
    context = json.loads(prompt.split("Research context:\n", 1)[1].split("\nDecision task:\n", 1)[0])
    assert context == {"question": question, "observations": observations}
    assert "Never change" in prompt and "not established evidence" in prompt
    assert "2026-08-28" not in prompt and "GOOGL" not in prompt
    assert '"news"' in prompt and "DOJ" in prompt
