from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from financial_research_agent.config import LLMSettings, load_llm_settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in list(os.environ):
        if key.startswith("LLM_") or key in {"OLLAMA_BASE_URL", "ALPHASCOPE_SEC_USER_AGENT", "UNCHANGED_SETTING"}:
            monkeypatch.delenv(key)
    # Undo all mutations from load_dotenv, including keys absent at test start.
    for key in ("LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_NUM_CTX", "OLLAMA_BASE_URL",
                "LLM_THINKING", "LLM_TIMEOUT_SECONDS", "LLM_MAX_RETRIES", "LLM_NUM_PREDICT",
                "ALPHASCOPE_SEC_USER_AGENT", "UNCHANGED_SETTING"):
        monkeypatch.setenv(key, "test-placeholder")
        monkeypatch.delenv(key)


def test_env_loading_from_explicit_file_preserves_shell_and_sec(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text('LLM_MODEL=qwen3:8b\nLLM_NUM_CTX=4096\nLLM_TEMPERATURE=0\n'
                    'LLM_THINKING=false\nLLM_TIMEOUT_SECONDS=12\nLLM_MAX_RETRIES=0\n'
                    'ALPHASCOPE_SEC_USER_AGENT="AlphaScope contact@example.com"\n'
                    'UNCHANGED_SETTING="literal-${VALUE}"\n', encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "shell-model")
    monkeypatch.setenv("ALPHASCOPE_SEC_USER_AGENT", "Existing SEC contact")
    settings = load_llm_settings(path)
    assert settings.model == "shell-model"
    assert settings.num_ctx == 4096 and settings.temperature == 0
    assert settings.timeout_seconds == 12 and settings.max_retries == 0
    assert settings.thinking is False
    assert os.environ["ALPHASCOPE_SEC_USER_AGENT"] == "Existing SEC contact"
    assert os.environ["UNCHANGED_SETTING"] == "literal-${VALUE}"


def test_default_env_path_does_not_depend_on_working_directory(tmp_path, monkeypatch):
    import financial_research_agent.config as config

    checkout = tmp_path / "checkout"
    module = checkout / "src" / "financial_research_agent" / "config.py"
    module.parent.mkdir(parents=True)
    (checkout / ".env").write_text("LLM_NUM_CTX=2048\n", encoding="utf-8")
    monkeypatch.setattr(config, "__file__", str(module))
    monkeypatch.chdir(tmp_path)
    assert load_llm_settings().num_ctx == 2048


def test_missing_env_file_uses_baseline_defaults(tmp_path):
    assert load_llm_settings(tmp_path / "missing") == LLMSettings()


@pytest.mark.parametrize("changes", [
    {"provider": "paid"}, {"timeout_seconds": 0}, {"max_retries": 3},
    {"temperature": float("nan")}, {"num_ctx": 0}, {"model": " "},
    {"base_url": "https://example.com"}, {"base_url": "http://0.0.0.0:11434"},
    {"base_url": "http://localhost.evil.com"}, {"base_url": "http://localhost/path"},
    {"base_url": "http://user:secret@localhost"}, {"base_url": "http://localhost?secret=1"},
    {"base_url": "http://localhost:99999"}, {"base_url": "file://localhost"},
])
def test_invalid_or_unapproved_settings_are_rejected(changes):
    with pytest.raises(ValidationError):
        LLMSettings(**changes)


@pytest.mark.parametrize("url", ["http://localhost:11434/", "http://127.0.0.1:11435", "http://[::1]:11434"])
def test_configurable_loopback_endpoints(url):
    assert LLMSettings(base_url=url).base_url == url
