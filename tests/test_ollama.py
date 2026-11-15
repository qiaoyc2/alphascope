from __future__ import annotations

import json
import logging
from unittest.mock import Mock

import pytest
import requests
from pydantic import BaseModel

from financial_research_agent.config import LLMSettings
from financial_research_agent.llm_decision import LLMDecision
from financial_research_agent.providers.llm import LLMError, LLMErrorCategory
from financial_research_agent.providers.ollama import OllamaProvider


VALID = {
    "premise_valid": None, "recommended_action": "continue", "next_tool": "market_data",
    "reason": "No price observations are available.", "information_gap": "Price history.", "should_stop": False,
}


def response(content=None, *, status=200, envelope=None):
    reply = requests.Response()
    reply.status_code = status
    body = envelope if envelope is not None else {
        "model": "qwen3:8b", "done": True, "done_reason": "stop",
        "message": {"role": "assistant", "content": json.dumps(VALID) if content is None else content},
        "prompt_eval_count": 100, "eval_count": 40, "eval_duration": 2_000_000_000,
    }
    reply._content = json.dumps(body).encode()
    reply._content_consumed = True
    return reply


def provider(*replies, retries=0):
    session = Mock(spec=requests.Session)
    session.post.side_effect = replies
    return OllamaProvider(LLMSettings(max_retries=retries), session=session)


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr("financial_research_agent.providers.ollama.time.sleep", lambda _: None)


def test_valid_structured_generation_sends_schema_and_options():
    client = provider(response())
    result = client.generate("Research question", LLMDecision)
    assert isinstance(result.decision, LLMDecision)
    assert result.decision.next_tool == "market_data"
    args, kwargs = client.session.post.call_args
    assert args == ("http://localhost:11434/api/chat",)
    assert kwargs["json"]["format"] == LLMDecision.model_json_schema()
    assert kwargs["json"]["think"] is False and kwargs["json"]["stream"] is False
    assert kwargs["json"]["options"] == {"temperature": 0, "num_ctx": 4096, "num_predict": 512}
    assert kwargs["timeout"] == 60 and kwargs["allow_redirects"] is False
    assert client.session.trust_env is False


def test_performance_extraction():
    meta = provider(response()).generate("Question", LLMDecision).performance
    assert meta.model == "qwen3:8b" and meta.input_tokens == 100
    assert meta.output_tokens == 40 and meta.generation_duration_seconds == 2
    assert meta.tokens_per_second == 20 and meta.validation_passed
    assert meta.total_duration_seconds >= meta.request_duration_seconds >= 0
    assert meta.error_category is None and meta.attempts == 1


@pytest.mark.parametrize("envelope", [
    {"done": True, "message": {"content": json.dumps(VALID)}},
    {"done": True, "message": {"content": json.dumps(VALID)},
     "eval_count": "unknown", "eval_duration": -1, "prompt_eval_count": True},
])
def test_missing_or_invalid_optional_metadata_is_unknown(envelope):
    meta = provider(response(envelope=envelope)).generate("Question", LLMDecision).performance
    assert meta.validation_passed
    assert meta.input_tokens is None and meta.output_tokens is None
    assert meta.generation_duration_seconds is None and meta.tokens_per_second is None


def test_zero_generation_duration_avoids_division_by_zero():
    reply = response()
    body = reply.json()
    body["eval_duration"] = 0
    assert provider(response(envelope=body)).generate("Question", LLMDecision).performance.tokens_per_second is None


@pytest.mark.parametrize("content,category", [
    ("not JSON", "invalid_json"), ("{", "invalid_json"),
    ("{}", "schema_validation"), ("[]", "schema_validation"),
    (json.dumps({**VALID, "next_tool": "shell"}), "schema_validation"),
    (json.dumps({**VALID, "should_stop": "false"}), "schema_validation"),
    ("", "empty_response"), ("  ", "empty_response"),
])
def test_invalid_content_is_rejected_without_retry(content, category):
    client = provider(response(content), retries=2)
    with pytest.raises(LLMError) as caught:
        client.generate("Question", LLMDecision)
    assert caught.value.category == category
    assert not caught.value.performance.validation_passed
    assert caught.value.performance.output_tokens == 40
    assert client.session.post.call_count == 1


@pytest.mark.parametrize("envelope,category", [
    ([], "invalid_response"), ({"error": "private server failure"}, "model_error"),
    ({"done": False, "message": {"content": json.dumps(VALID)}}, "invalid_response"),
    ({"done": True, "done_reason": "length", "message": {"content": json.dumps(VALID)}}, "invalid_response"),
    ({"done": True, "done_reason": [], "message": {"content": json.dumps(VALID)}}, "invalid_response"),
    ({"done": True}, "empty_response"),
    ({"done": True, "message": {"content": 5}}, "invalid_response"),
])
def test_invalid_envelopes(envelope, category):
    with pytest.raises(LLMError) as caught:
        provider(response(envelope=envelope)).generate("Question", LLMDecision)
    assert caught.value.category == category
    assert "private server failure" not in str(caught.value)


def test_invalid_envelope_json():
    reply = response()
    reply._content = b"not json"
    with pytest.raises(LLMError) as caught:
        provider(reply).generate("Question", LLMDecision)
    assert caught.value.category == "invalid_json"


@pytest.mark.parametrize("failure,category", [
    (requests.ConnectionError("secret"), "connection"),
    (requests.Timeout("secret"), "timeout"),
    (requests.RequestException("secret"), "http_error"),
])
def test_network_errors_are_categorized_and_sanitized(failure, category):
    client = provider(failure, failure, retries=1)
    with pytest.raises(LLMError) as caught:
        client.generate("Question", LLMDecision)
    assert caught.value.category == category
    assert "secret" not in str(caught.value)
    assert not caught.value.performance.validation_passed
    assert client.session.post.call_count == (1 if category == "http_error" else 2)


def test_connection_error_explains_how_to_start_ollama():
    with pytest.raises(LLMError, match="ollama serve"):
        provider(requests.ConnectionError()).generate("Question", LLMDecision)


def test_transient_failure_can_recover_with_bounded_retry(caplog):
    client = provider(requests.Timeout(), response(), retries=1)
    with caplog.at_level(logging.INFO):
        result = client.generate("Question", LLMDecision)
    assert result.performance.validation_passed and result.performance.attempts == 2
    assert len(caplog.records) == 2
    assert '"error_category":"timeout"' in caplog.records[0].getMessage()


@pytest.mark.parametrize("status,category,calls", [(404, "model_not_found", 1), (302, "http_error", 1),
                                                 (400, "http_error", 1), (500, "http_error", 2), (429, "http_error", 2)])
def test_http_errors(status, category, calls):
    client = provider(response(status=status), response(status=status), retries=1)
    with pytest.raises(LLMError) as caught:
        client.generate("Question", LLMDecision)
    assert caught.value.category == category and client.session.post.call_count == calls
    if status == 404:
        assert "ollama pull qwen3:8b" in str(caught.value)


def test_telemetry_does_not_log_prompt_response_or_hidden_reasoning(caplog):
    body = response().json()
    body["message"]["thinking"] = "PRIVATE REASONING"
    body["message"]["content"] = json.dumps({**VALID, "reason": "PRIVATE RESPONSE"})
    with caplog.at_level(logging.INFO):
        provider(response(envelope=body)).generate("PRIVATE PROMPT API_KEY=secret", LLMDecision)
    assert "PRIVATE" not in caplog.text and "secret" not in caplog.text
    assert "validation_passed" in caplog.text


def test_provider_supports_other_pydantic_output_schemas():
    class SimpleOutput(BaseModel):
        ok: bool

    result = provider(response('{"ok":true}')).generate("Question", SimpleOutput)
    assert isinstance(result.decision, SimpleOutput) and result.decision.ok


def test_thinking_can_be_enabled_and_endpoint_configured():
    client = provider(response())
    client.settings = LLMSettings(thinking=True, base_url="http://127.0.0.1:11435/")
    client.generate("Question", LLMDecision)
    assert client.session.post.call_args.args[0] == "http://127.0.0.1:11435/api/chat"
    assert client.session.post.call_args.kwargs["json"]["think"] is True


def test_context_manager_closes_session():
    client = provider(response())
    with client:
        client.generate("Question", LLMDecision)
    client.session.close.assert_called_once()
