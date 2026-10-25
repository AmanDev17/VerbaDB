"""
The agent loop. ``Agent.ask_stream`` is a generator of plain-dict events (so the web layer can stream them as
they happen): ask the model -> run the tools it requests -> feed results back -> repeat until it answers.

A session belongs to a *workspace* and has an *active database* (``None`` until one is created). Read-only
databases get the read-only toolset; sandbox databases — and the "no database yet" state, where the agent can
create one — get the sandbox toolset with ``create_database`` / ``modify_database``.

History is append-only and stored as the LLM layer's own opaque items. If a question does not finish
(error, quota, step limit, disconnect) the partial exchange is rolled back so the session stays valid.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional

from .config import Settings
from .llm import LLM, LLMError, LLMUnavailable, RateLimited, Turn
from .manager import DatabaseManager
from .prompts import build_sandbox_prompt, build_system_prompt
from .sandbox_tools import SandboxToolBox
from .tools import ToolBox, ToolSpec

log = logging.getLogger("verbadb.agent")

MAX_QUESTION_CHARS = 2000
Event = Dict[str, Any]


@dataclass
class Session:
    id: str
    workspace: str
    database_id: Optional[str] = None
    history: List[Any] = field(default_factory=list)
    marks: List[int] = field(default_factory=list)          # history index where each finished exchange began
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = field(default_factory=time.monotonic)


def _error(kind: str, message: str, **extra: Any) -> Event:
    return {"type": "error", "kind": kind, "message": message, **extra}


class Agent:
    def __init__(
        self,
        llm: LLM,
        manager: DatabaseManager,
        settings: Settings,
        *,
        sleep: Callable[[float], None] = time.sleep,
        today: Callable[[], dt.date] = dt.date.today,
    ) -> None:
        self.llm = llm
        self.manager = manager
        self.settings = settings
        self._sleep = sleep
        self._today = today

    # ── public ────────────────────────────────────────────────────────────

    def ask_stream(self, session: Session, question: str) -> Iterator[Event]:
        question = (question or "").strip()
        if not question:
            yield _error("input", "Please type a question.")
            return
        if len(question) > MAX_QUESTION_CHARS:
            yield _error("input", f"That question is too long (max {MAX_QUESTION_CHARS} characters).")
            return
        if not session.lock.acquire(blocking=False):
            yield _error("busy", "The previous question in this chat is still running.")
            return
        try:
            session.last_used = time.monotonic()
            yield from self._run(session, question)
        finally:
            session.lock.release()

    # ── internals ─────────────────────────────────────────────────────────

    def _run(self, session: Session, question: str) -> Iterator[Event]:
        entry = None
        if session.database_id:
            try:
                entry = self.manager.resolve(session.workspace, session.database_id)
            except KeyError:
                yield _error("connection", "That database no longer exists. Pick another one from the list.")
                return
            except Exception as exc:
                log.warning("opening database %s failed: %s", session.database_id, exc)
                yield _error("connection", f"Could not open that database: {str(exc).splitlines()[0]}")
                return

        sandbox_mode = entry is None or entry.writable
        if sandbox_mode:
            toolbox: ToolBox = SandboxToolBox(self.manager, session.workspace, entry, self.settings)
            system = build_sandbox_prompt(self.settings)
        else:
            toolbox = ToolBox(entry.flash, self.settings)
            system = build_system_prompt(entry.flash.backend, self.settings)
        specs: List[ToolSpec] = toolbox.specs()

        today = self._today()
        mark = len(session.history)
        steps = requests = total_tokens = 0
        finished = False
        try:
            # Everything that touches the history lives inside the try, so a client that disconnects at any
            # point (even right after "start") still gets the partial exchange rolled back.
            session.history.append(
                self.llm.user_message(f"Today's date: {today.isoformat()} ({today:%A})\n\nQuestion: {question}")
            )
            yield {
                "type": "start", "mode": "sandbox" if sandbox_mode else "readonly",
                "backend": "sqlite" if entry is None else entry.backend, "model": getattr(self.llm, "model", ""),
                "database": entry.info() if entry is not None else None,
            }

            for _ in range(self.settings.max_steps):
                turn = yield from self._generate(system, specs, session.history)
                requests += 1
                total_tokens += turn.usage.get("total_tokens", 0)
                session.history.append(turn.raw)

                if not turn.calls:
                    if not turn.text.strip():
                        raise LLMUnavailable("The model returned an empty answer.")
                    finished = True
                    yield {
                        "type": "final", "text": turn.text.strip(), "steps": steps,
                        "requests": requests, "total_tokens": total_tokens,
                    }
                    return

                if turn.text.strip():
                    yield {"type": "text", "text": turn.text.strip()}
                answers = []
                for call in turn.calls:
                    steps += 1
                    yield {"type": "step", "id": steps, "tool": call.name, "args": call.args}
                    outcome = toolbox.call(call.name, call.args)
                    yield {"type": "observation", "id": steps, "ok": outcome.ok, "summary": outcome.summary}
                    if outcome.result is not None:
                        yield outcome.result.to_event()
                    for extra in getattr(outcome, "events", ()):
                        if extra["type"] == "database":
                            session.database_id = extra["database"]["id"]       # the chat now works on it
                        yield extra
                    answers.append((call, outcome.response))
                session.history.append(self.llm.tool_results(answers))

            yield _error(
                "limit",
                f"I stopped after {self.settings.max_steps} model calls without a final answer. "
                "Try a more specific request, or raise VERBADB_MAX_STEPS.",
            )
        except LLMError as exc:
            extra = {"retry_after": exc.retry_after, "daily": exc.daily} if isinstance(exc, RateLimited) else {}
            yield _error(exc.kind, str(exc), **extra)
        except Exception as exc:  # a bug must not leave the stream hanging or the history corrupt
            log.exception("agent failed")
            yield _error("internal", f"Something went wrong: {exc.__class__.__name__}: {exc}")
        finally:
            if finished:
                session.marks.append(mark)
                self._trim(session)
            else:
                del session.history[mark:]       # drop the partial exchange; the session stays valid

    def _generate(self, system: str, specs: List[ToolSpec], history: List[Any]) -> Generator[Event, None, Turn]:
        """Call the model, retrying rate limits and transient failures (yielding status events while waiting)."""
        attempt = 0
        while True:
            try:
                return self.llm.generate(system, specs, history)
            except RateLimited as exc:
                too_long = (exc.retry_after or 0) > self.settings.max_retry_wait
                if exc.daily or too_long or attempt >= self.settings.llm_retries:
                    raise
                wait = max(1.0, exc.retry_after if exc.retry_after is not None else 5.0 * 2 ** attempt)
                reason = "rate limit reached"
            except LLMUnavailable as exc:
                if attempt >= self.settings.llm_retries:
                    raise
                wait = min(2.0 * 2 ** attempt, 20.0)
                reason = "temporarily unavailable"
            attempt += 1
            yield {"type": "status", "message": f"Gemini {reason}; retrying in {wait:.0f}s (attempt {attempt}).",
                   "retry_after": wait}
            self._sleep(wait)

    def _trim(self, session: Session) -> None:
        """Drop the oldest finished exchanges while the history is too long (always at exchange boundaries)."""
        limit = self.settings.max_history_items
        while len(session.history) > limit and len(session.marks) > 1:
            cut = session.marks[1]
            del session.history[:cut]
            session.marks = [m - cut for m in session.marks[1:]]
