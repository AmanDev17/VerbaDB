from __future__ import annotations

import httpx
import pytest
from google.genai import errors, types

from verbadb.config import Settings
from verbadb.llm import (
    Blocked, FunctionCall, GeminiLLM, LLMError, LLMUnavailable, ModelNotFound, NotConfigured, RateLimited,
    is_daily_quota, parse_retry_after,
)
from verbadb.tools import tool_specs


def response(*parts, finish="STOP", usage=None):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)), finish_reason=finish)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5, total_token_count=15) if usage is None else usage,
    )


def api_error(code, status, message, details=None, cls=errors.ClientError):
    body = {"error": {"code": code, "status": status, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return cls(code, body)


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.got = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out

    def get(self, *, model):
        self.got.append(model)
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


class FakeClient:
    def __init__(self, *outcomes):
        self.models = FakeModels(outcomes)


def llm_with(*outcomes):
    client = FakeClient(*outcomes)
    return GeminiLLM("key", "gemini-test", client=client), client


SPECS = tool_specs("sqlite", Settings())


# ── turns ─────────────────────────────────────────────────────────────────

def test_text_and_function_calls_are_extracted_and_content_is_kept_verbatim():
    reply = response(
        types.Part(text="thinking about it", thought=True),
        types.Part(text="Let me check. "),
        types.Part(function_call=types.FunctionCall(id="c1", name="run_sql", args={"sql": "SELECT 1"})),
        types.Part(function_call=types.FunctionCall(name="list_tables", args=None)),
    )
    llm, client = llm_with(reply)
    turn = llm.generate("sys", SPECS, [llm.user_message("hi")])
    assert turn.text == "Let me check. "                     # thought summaries are not shown as the answer
    assert turn.calls == [FunctionCall("run_sql", {"sql": "SELECT 1"}, "c1"), FunctionCall("list_tables", {}, None)]
    assert turn.raw is reply.candidates[0].content            # appended to history exactly as returned
    assert turn.usage == {"prompt_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert turn.finish_reason == "STOP"


def test_request_shape():
    llm, client = llm_with(response(types.Part(text="ok")))
    history = [llm.user_message("hello")]
    llm.generate("SYSTEM TEXT", SPECS, history)
    sent = client.models.calls[0]
    assert sent["model"] == "gemini-test" and sent["contents"] == history and sent["contents"] is not history
    config = sent["config"]
    assert config.system_instruction == "SYSTEM TEXT"
    assert config.automatic_function_calling.disable is True
    declared = {d.name: d for d in config.tools[0].function_declarations}
    assert set(declared) == {s.name for s in SPECS}
    assert declared["list_tables"].parameters_json_schema is None            # no-argument tools declare no schema
    assert declared["run_sql"].parameters_json_schema["required"] == ["sql"]
    assert declared["describe_tables"].parameters_json_schema["properties"]["tables"]["items"] == {"type": "string"}


def test_history_items():
    llm, _ = llm_with()
    user = llm.user_message("hi")
    assert user.role == "user" and user.parts[0].text == "hi"
    results = llm.tool_results([
        (FunctionCall("run_sql", {}, "c1"), {"rows": [1]}),
        (FunctionCall("list_tables", {}, None), {"error": "x"}),
    ])
    assert results.role == "user"
    first, second = (p.function_response for p in results.parts)
    assert (first.id, first.name, first.response) == ("c1", "run_sql", {"rows": [1]})
    assert (second.id, second.name, second.response) == (None, "list_tables", {"error": "x"})


# ── unusable responses ────────────────────────────────────────────────────

def test_blocked_prompt():
    blocked = types.GenerateContentResponse(
        candidates=[], prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY"))
    llm, _ = llm_with(blocked)
    with pytest.raises(Blocked, match="SAFETY"):
        llm.generate("s", SPECS, [])


def test_safety_stop_is_blocked_without_retry():
    llm, client = llm_with(types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[]), finish_reason="SAFETY")]))
    with pytest.raises(Blocked):
        llm.generate("s", SPECS, [])
    assert len(client.models.calls) == 1


def test_empty_response_is_retried_once_then_reported():
    empty = lambda: types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[]), finish_reason="STOP")])
    llm, client = llm_with(empty(), response(types.Part(text="recovered")))
    assert llm.generate("s", SPECS, []).text == "recovered"
    assert len(client.models.calls) == 2

    llm, client = llm_with(empty(), empty())
    with pytest.raises(LLMUnavailable):
        llm.generate("s", SPECS, [])
    assert len(client.models.calls) == 2


def test_max_tokens_without_content_is_a_plain_error():
    llm, _ = llm_with(types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[]), finish_reason="MAX_TOKENS")]))
    with pytest.raises(LLMError, match="output tokens"):
        llm.generate("s", SPECS, [])


# ── API errors ────────────────────────────────────────────────────────────

RETRY_INFO = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"}
PER_MINUTE = {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
              "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}
PER_DAY = {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
           "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}


def test_rate_limit_carries_the_advised_delay():
    exc = api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", [PER_MINUTE, RETRY_INFO])
    assert parse_retry_after(exc) == 34.0 and not is_daily_quota(exc)
    llm, _ = llm_with(exc)
    with pytest.raises(RateLimited) as err:
        llm.generate("s", SPECS, [])
    assert err.value.retry_after == 34.0 and not err.value.daily


def test_retry_delay_can_come_from_the_message():
    exc = api_error(429, "RESOURCE_EXHAUSTED", "You exceeded your quota. Please retry in 12.5s.")
    assert parse_retry_after(exc) == 12.5


def test_daily_quota_is_flagged():
    exc = api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", [PER_DAY])
    assert is_daily_quota(exc)
    llm, _ = llm_with(exc)
    with pytest.raises(RateLimited) as err:
        llm.generate("s", SPECS, [])
    assert err.value.daily and "midnight Pacific" in str(err.value)


@pytest.mark.parametrize("exc,expected", [
    (api_error(503, "UNAVAILABLE", "overloaded", cls=errors.ServerError), LLMUnavailable),
    (api_error(500, "INTERNAL", "oops", cls=errors.ServerError), LLMUnavailable),
    (api_error(404, "NOT_FOUND", "models/x is not found"), ModelNotFound),
    (api_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."), NotConfigured),
    (api_error(403, "PERMISSION_DENIED", "denied"), NotConfigured),
    (api_error(400, "INVALID_ARGUMENT", "bad request"), LLMError),
])
def test_error_classification(exc, expected):
    llm, _ = llm_with(exc)
    with pytest.raises(expected) as err:
        llm.generate("s", SPECS, [])
    assert type(err.value) is expected


def test_http_client_errors_count_as_transient_network_problems():
    llm, _ = llm_with(httpx.ConnectError("boom"))
    with pytest.raises(LLMUnavailable, match="Could not reach"):
        llm.generate("s", SPECS, [])


def test_programming_errors_are_not_disguised_as_network_errors():
    llm, _ = llm_with(ValueError("bad config"))
    with pytest.raises(LLMError, match="Unexpected error") as err:
        llm.generate("s", SPECS, [])
    assert not isinstance(err.value, LLMUnavailable)


# ── health ────────────────────────────────────────────────────────────────

def test_check_model():
    llm, client = llm_with(object())
    assert llm.check_model() == (True, "ok") and client.models.got == ["gemini-test"]
    llm, _ = llm_with(api_error(404, "NOT_FOUND", "nope"))
    ok, detail = llm.check_model()
    assert ok is False and "VERBADB_MODEL" in detail
    llm, _ = llm_with(api_error(429, "RESOURCE_EXHAUSTED", "busy"))
    assert llm.check_model()[0] is None
