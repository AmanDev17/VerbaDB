from __future__ import annotations

import pytest

from verbadb.config import DEFAULT_MODEL, DEFAULT_REDACT, ConfigError, Settings, load_settings


def test_defaults():
    s = load_settings({})
    assert s.api_key is None and s.model == DEFAULT_MODEL and s.databases == {}
    assert (s.max_steps, s.max_rows, s.preview_rows, s.query_timeout) == (8, 500, 20, 15)
    assert s.redact_columns == DEFAULT_REDACT and s.host == "127.0.0.1" and s.port == 8000


def test_everything_can_be_configured():
    s = load_settings({
        "GEMINI_API_KEY": "k", "VERBADB_MODEL": " gemini-x ", "VERBADB_DATABASES": '{"a": "sqlite:///a.db", "b": "mongodb://h/db"}',
        "VERBADB_MAX_STEPS": "3", "VERBADB_MAX_ROWS": "50", "VERBADB_PREVIEW_ROWS": "0", "VERBADB_QUERY_TIMEOUT": "5",
        "VERBADB_REDACT_COLUMNS": "Email, IBAN ,", "VERBADB_LLM_RETRIES": "1", "VERBADB_HOST": "0.0.0.0", "VERBADB_PORT": "9000",
    })
    assert s.api_key == "k" and s.model == "gemini-x"
    assert s.databases == {"a": "sqlite:///a.db", "b": "mongodb://h/db"}
    assert (s.max_steps, s.max_rows, s.preview_rows, s.query_timeout, s.llm_retries) == (3, 50, 0, 5, 1)
    assert s.redact_columns == ("email", "iban") and (s.host, s.port) == ("0.0.0.0", 9000)


def test_the_old_nlq_names_still_work_and_the_new_names_win():
    old = load_settings({"NLQ_MODEL": "legacy-model", "NLQ_MAX_ROWS": "42", "NLQ_DATABASES": '{"a": "sqlite:///a.db"}'})
    assert (old.model, old.max_rows, old.databases) == ("legacy-model", 42, {"a": "sqlite:///a.db"})
    both = load_settings({"NLQ_MODEL": "legacy-model", "VERBADB_MODEL": "new-model"})
    assert both.model == "new-model"


def test_gemini_key_wins_over_google_key():
    assert load_settings({"GOOGLE_API_KEY": "g"}).api_key == "g"
    assert load_settings({"GOOGLE_API_KEY": "g", "GEMINI_API_KEY": "k"}).api_key == "k"
    assert load_settings({"GEMINI_API_KEY": ""}).api_key is None


@pytest.mark.parametrize("env", [
    {"VERBADB_MAX_STEPS": "many"}, {"VERBADB_MAX_STEPS": "0"}, {"VERBADB_MAX_ROWS": "-5"}, {"VERBADB_PREVIEW_ROWS": "-1"},
    {"VERBADB_DATABASES": "not json"}, {"VERBADB_DATABASES": '["a"]'}, {"VERBADB_DATABASES": '{"a": 5}'},
])
def test_bad_values_raise_a_clear_error(env):
    with pytest.raises(ConfigError):
        load_settings(env)


def test_blank_values_fall_back_to_defaults():
    s = load_settings({"VERBADB_MAX_ROWS": "  ", "VERBADB_MODEL": "", "VERBADB_DATABASES": " "})
    assert s.max_rows == 500 and s.model == DEFAULT_MODEL and s.databases == {}


def test_redaction_can_be_switched_off_and_matches_substrings():
    assert load_settings({"VERBADB_REDACT_COLUMNS": ""}).redact_columns == ()
    assert not load_settings({"VERBADB_REDACT_COLUMNS": ""}).is_redacted("email")
    s = Settings()
    assert s.is_redacted("Customer_Email") and s.is_redacted("api_token") and s.is_redacted("PASSWORD_hash")
    assert not s.is_redacted("country")


def test_dotenv_is_found_from_the_current_directory_and_never_overrides_the_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("VERBADB_MODEL=from-dotenv\nVERBADB_MAX_ROWS=77\nGEMINI_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for name in ("VERBADB_MODEL", "VERBADB_MAX_ROWS"):          # register for cleanup: load_dotenv writes to os.environ
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    monkeypatch.setenv("GEMINI_API_KEY", "real-env-key")

    s = load_settings()
    assert s.model == "from-dotenv" and s.max_rows == 77
    assert s.api_key == "real-env-key"
