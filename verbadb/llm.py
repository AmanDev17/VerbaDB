"""
The only module that talks to Gemini. The rest of the app sees ``Turn`` / ``FunctionCall`` and opaque history
items, so changing SDK surface (``generate_content`` -> Interactions API) or provider touches this file only.

Uses ``client.models.generate_content`` with automatic function calling disabled: we run the tool loop
ourselves and own the conversation history (nothing is stored server-side). History items are the SDK's own
``Content`` objects; model turns are appended exactly as returned (they carry thought signatures).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from .tools import ToolSpec

log = logging.getLogger("verbadb.llm")


# ── Errors ────────────────────────────────────────────────────────────────

class LLMError(Exception):
    kind = "internal"


class NotConfigured(LLMError):
    kind = "config"


class RateLimited(LLMError):
    kind = "quota"

    def __init__(self, message: str, retry_after: Optional[float] = None, daily: bool = False) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.daily = daily


class LLMUnavailable(LLMError):
    """Transient failure (5xx, network, empty/malformed response): worth retrying."""

    kind = "unavailable"


class Blocked(LLMError):
    kind = "blocked"


class ModelNotFound(LLMError):
    kind = "model"


# ── Neutral types ─────────────────────────────────────────────────────────

@dataclass
class FunctionCall:
    name: str
    args: Dict[str, Any]
    id: Optional[str] = None


@dataclass
class Turn:
    text: str
    calls: List[FunctionCall]
    raw: Any                                  # appended to history verbatim
    finish_reason: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)


class LLM(Protocol):
    model: str

    def user_message(self, text: str) -> Any: ...

    def tool_results(self, results: Sequence[Tuple[FunctionCall, Dict[str, Any]]]) -> Any: ...

    def generate(self, system: str, tools: Sequence[ToolSpec], history: Sequence[Any]) -> Turn: ...


# ── Gemini ────────────────────────────────────────────────────────────────

_RETRY_IN = re.compile(r"retry in ([0-9.]+)\s*s", re.IGNORECASE)
_DURATION = re.compile(r"^([0-9.]+)s$")


def _error_details(exc: Any) -> List[Dict[str, Any]]:
    body = getattr(exc, "details", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        details = inner.get("details") if isinstance(inner, dict) else None
        if isinstance(details, list):
            return [d for d in details if isinstance(d, dict)]
    return []


def parse_retry_after(exc: Any) -> Optional[float]:
    """Seconds Gemini asks us to wait (RetryInfo.retryDelay, or 'Please retry in 34.9s')."""
    for detail in _error_details(exc):
        match = _DURATION.match(str(detail.get("retryDelay", "")))
        if match:
            return float(match.group(1))
    match = _RETRY_IN.search(str(getattr(exc, "message", "") or exc))
    return float(match.group(1)) if match else None


def is_daily_quota(exc: Any) -> bool:
    """True when the exhausted quota is a per-day one (waiting seconds will not help)."""
    for detail in _error_details(exc):
        for violation in detail.get("violations", []) or []:
            if "perday" in str(violation.get("quotaId", "")).lower().replace("_", ""):
                return True
    return "perday" in str(getattr(exc, "message", "")).lower().replace(" ", "").replace("_", "")


_NETWORK_MODULES = frozenset({"httpx", "httpx2", "httpcore", "requests", "urllib3", "ssl", "socket", "aiohttp"})


def _is_network_error(exc: BaseException) -> bool:
    """HTTP-client errors (httpx/httpx2 do not derive from OSError) and plain socket errors."""
    return isinstance(exc, (OSError, TimeoutError)) or type(exc).__module__.split(".")[0] in _NETWORK_MODULES


def _short(message: Any, limit: int = 300) -> str:
    text = " ".join(str(message or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


class GeminiLLM:
    def __init__(self, api_key: str, model: str, client: Any = None) -> None:
        from google import genai

        self.model = model
        self.client = client or genai.Client(api_key=api_key)

    # history items --------------------------------------------------------

    def user_message(self, text: str) -> Any:
        from google.genai import types

        return types.Content(role="user", parts=[types.Part.from_text(text=text)])

    def tool_results(self, results: Sequence[Tuple[FunctionCall, Dict[str, Any]]]) -> Any:
        from google.genai import types

        parts = [
            types.Part(function_response=types.FunctionResponse(id=call.id, name=call.name, response=response))
            for call, response in results
        ]
        return types.Content(role="user", parts=parts)

    # generation -----------------------------------------------------------

    def _config(self, system: str, tools: Sequence[ToolSpec]) -> Any:
        from google.genai import types

        declarations = []
        for spec in tools:
            kwargs: Dict[str, Any] = {"name": spec.name, "description": spec.description}
            if spec.parameters.get("properties"):          # Gemini rejects an OBJECT schema with no properties
                kwargs["parameters_json_schema"] = spec.parameters
            declarations.append(types.FunctionDeclaration(**kwargs))
        return types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def generate(self, system: str, tools: Sequence[ToolSpec], history: Sequence[Any]) -> Turn:
        config = self._config(system, tools)
        for attempt in range(2):     # one immediate retry for an empty / malformed-call response
            response = self._call(history, config)
            turn, problem = self._to_turn(response)
            if turn is not None:
                return turn
            if attempt == 0 and problem.kind == "unavailable":
                log.warning("Retrying after: %s", problem)
                continue
            raise problem
        raise LLMUnavailable("Gemini returned no usable response.")  # pragma: no cover

    def _call(self, history: Sequence[Any], config: Any) -> Any:
        from google.genai import errors

        try:
            return self.client.models.generate_content(model=self.model, contents=list(history), config=config)
        except errors.APIError as exc:
            raise self._translate(exc) from exc
        except Exception as exc:
            if _is_network_error(exc):
                raise LLMUnavailable(f"Could not reach the Gemini API: {_short(exc)}") from exc
            raise LLMError(f"Unexpected error calling Gemini ({type(exc).__name__}): {_short(exc)}") from exc

    def _translate(self, exc: Any) -> LLMError:
        code = getattr(exc, "code", None)
        status = str(getattr(exc, "status", "") or "")
        message = _short(getattr(exc, "message", None) or exc)
        if code == 429 or status == "RESOURCE_EXHAUSTED":
            daily = is_daily_quota(exc)
            retry_after = parse_retry_after(exc)
            if daily:
                return RateLimited(
                    "The free-tier daily request quota for this model is used up. It resets at midnight Pacific "
                    "time; you can also try another model via VERBADB_MODEL.", retry_after, daily=True,
                )
            return RateLimited(f"Gemini rate limit reached. {message}", retry_after)
        if code in (500, 502, 503, 504) or status in ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"):
            return LLMUnavailable(f"Gemini is temporarily unavailable ({code}). {message}")
        if code == 404 or status == "NOT_FOUND":
            return ModelNotFound(f"Model '{self.model}' was not found or is not available to this key. Set VERBADB_MODEL.")
        if code in (401, 403) or status in ("PERMISSION_DENIED", "UNAUTHENTICATED") or "API key" in message:
            return NotConfigured(f"Gemini rejected the API key or its permissions. {message}")
        return LLMError(f"Gemini API error ({code}): {message}")

    @staticmethod
    def _to_turn(response: Any) -> Tuple[Optional[Turn], Optional[LLMError]]:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            feedback = getattr(response, "prompt_feedback", None)
            reason = getattr(feedback, "block_reason", None)
            if reason:
                return None, Blocked(f"Gemini blocked the request ({getattr(reason, 'name', reason)}).")
            return None, LLMUnavailable("Gemini returned no candidates.")
        candidate = candidates[0]
        finish = getattr(candidate, "finish_reason", None)
        finish_name = getattr(finish, "name", None) or (str(finish) if finish else None)
        parts = list(getattr(getattr(candidate, "content", None), "parts", None) or [])
        if not parts:
            if finish_name in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY", "RECITATION"):
                return None, Blocked(f"Gemini stopped the answer ({finish_name}).")
            if finish_name == "MAX_TOKENS":
                return None, LLMError("Gemini ran out of output tokens before answering.")
            return None, LLMUnavailable(f"Gemini returned an empty response ({finish_name or 'unknown'}).")

        text = "".join(p.text for p in parts if isinstance(getattr(p, "text", None), str) and not getattr(p, "thought", False))
        calls = [
            FunctionCall(name=p.function_call.name, args=dict(p.function_call.args or {}), id=p.function_call.id)
            for p in parts if getattr(p, "function_call", None) is not None
        ]
        usage_meta = getattr(response, "usage_metadata", None)
        usage = {
            "prompt_tokens": int(getattr(usage_meta, "prompt_token_count", 0) or 0),
            "output_tokens": int(getattr(usage_meta, "candidates_token_count", 0) or 0),
            "total_tokens": int(getattr(usage_meta, "total_token_count", 0) or 0),
        }
        return Turn(text=text, calls=calls, raw=candidate.content, finish_reason=finish_name, usage=usage), None

    # health ---------------------------------------------------------------

    def check_model(self) -> Tuple[Optional[bool], str]:
        """``(True|False|None, detail)``: whether the configured model id exists (None = could not tell)."""
        from google.genai import errors

        try:
            self.client.models.get(model=self.model)
            return True, "ok"
        except errors.APIError as exc:
            if getattr(exc, "code", None) == 404:
                return False, f"Model '{self.model}' was not found. Set VERBADB_MODEL to a current model id."
            return None, _short(getattr(exc, "message", None) or exc)
        except Exception as exc:
            return None, _short(exc)
