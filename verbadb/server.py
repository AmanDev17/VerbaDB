"""
FastAPI app: a small JSON API plus a static single-page UI. ``POST /api/ask`` streams the agent's events as
Server-Sent Events.

There are no accounts. Each browser is given an anonymous *workspace* (an HttpOnly cookie); the sandbox databases,
databases connected through the UI, and prompt history of a workspace are invisible to every other workspace.
That separates people who share one running app, but it is not authentication: it binds to 127.0.0.1 by default,
rejects requests whose Host/Origin is not the address it is served on (DNS-rebinding / cross-site protection),
and should not be exposed on a network without adding real authentication in front of it.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Tuple
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, introspect
from .agent import Agent, Session
from .config import Settings, load_settings
from .llm import LLM, GeminiLLM
from .manager import DatabaseManager, ManagerError

log = logging.getLogger("verbadb.server")

STATIC_DIR = Path(__file__).parent / "static"
LOOPBACK = {"localhost", "127.0.0.1", "[::1]"}
MODEL_CHECK_TTL = 600.0
COOKIE = "verbadb_ws"
_WORKSPACE_RE = re.compile(r"^[0-9a-f]{32}$")


class AskBody(BaseModel):
    session_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    database_id: Optional[str] = Field(None, max_length=64)
    question: str = Field(max_length=4000)


class CreateDatabaseBody(BaseModel):
    kind: Literal["sandbox", "connection"] = "sandbox"
    name: str = Field(max_length=100)
    description: str = Field("", max_length=500)
    sample: bool = False                              # sandbox only: start with the demo shop data
    url: Optional[str] = Field(None, max_length=2100)  # connection only


class SessionStore:
    """In-memory chat sessions per workspace, with a TTL and a size cap."""

    def __init__(self, ttl: float = 3600.0, max_sessions: int = 200) -> None:
        self._ttl = ttl
        self._max = max_sessions
        self._lock = threading.Lock()
        self._sessions: Dict[Tuple[str, str], Session] = {}

    def get(self, workspace: str, sid: str, database_id: Optional[str]) -> Session:
        with self._lock:
            self._evict()
            key = (workspace, sid)
            session = self._sessions.get(key)
            if session is None or session.database_id != database_id:
                session = Session(id=sid, workspace=workspace, database_id=database_id)
                self._sessions[key] = session
                self._evict()               # enforce the size cap right away, not on the next request
            return session

    def reset(self, workspace: str, sid: str) -> None:
        with self._lock:
            self._sessions.pop((workspace, sid), None)

    def _evict(self) -> None:
        now = time.monotonic()
        for key, session in list(self._sessions.items()):
            if not session.lock.locked() and now - session.last_used > self._ttl:
                del self._sessions[key]
        while len(self._sessions) > self._max:
            idle = [(s.last_used, key) for key, s in self._sessions.items() if not s.lock.locked()]
            if not idle:
                break
            del self._sessions[min(idle)[1]]


def _host_name(header: str) -> str:
    header = header.strip().lower()
    return header.split("]")[0] + "]" if header.startswith("[") else header.split(":")[0]


def _sse(events: Iterator[Dict[str, Any]]) -> Iterator[bytes]:
    try:
        for event in events:
            yield f"data: {json.dumps(event, default=str)}\n\n".encode()
    except Exception as exc:  # the agent already turns failures into events; this is the last resort
        log.exception("stream failed")
        error = {"type": "error", "kind": "internal", "message": f"Something went wrong: {exc.__class__.__name__}"}
        yield f"data: {json.dumps(error)}\n\n".encode()
    finally:
        close = getattr(events, "close", None)
        if close:
            close()          # releases the session lock and rolls back a half-finished exchange
    yield b'data: {"type": "done"}\n\n'


def _recorded(
    events: Iterator[Dict[str, Any]], *, manager: DatabaseManager, workspace: str, question: str,
    database_id: Optional[str], database_name: Optional[str],
) -> Iterator[Dict[str, Any]]:
    """Pass events through unchanged and save the prompt (with its outcome) to the workspace's history."""
    status, summary, skip = "stopped", "", False
    try:
        for event in events:
            kind = event["type"]
            if kind == "database":                                   # the chat moved to a newly created database
                database_id, database_name = event["database"]["id"], event["database"]["name"]
            elif kind == "final":
                status, summary = "ok", event["text"]
            elif kind == "error":
                skip = event["kind"] in ("input", "busy")
                status, summary = "error", event["message"]
            yield event
    finally:
        close = getattr(events, "close", None)
        if close:
            close()
        if not skip:
            try:
                manager.store.add_prompt(workspace, question, status, summary, database_id, database_name)
            except Exception:  # history is a convenience; never break a request because of it
                log.exception("could not save the prompt")


def create_app(
    settings: Optional[Settings] = None,
    *,
    llm: Optional[LLM] = None,
    manager: Optional[DatabaseManager] = None,
) -> FastAPI:
    settings = settings or load_settings()
    manager = manager or DatabaseManager(settings)
    if llm is None and settings.api_key:
        llm = GeminiLLM(settings.api_key, settings.model)
    agent = Agent(llm, manager, settings) if llm is not None else None
    sessions = SessionStore()
    model_check: Dict[str, Any] = {"at": 0.0, "value": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        manager.close_all()

    app = FastAPI(
        title="VerbaDB", version=__version__, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )

    allowed_hosts = set(LOOPBACK) | {"testserver"}
    check_hosts = settings.host not in ("0.0.0.0", "::")
    if not check_hosts:
        log.warning("Listening on all interfaces with NO authentication: anyone who can reach this port can use "
                    "the configured databases and create sandboxes.")
    elif settings.host not in LOOPBACK:
        allowed_hosts.add(settings.host.lower())

    @app.middleware("http")
    async def origin_guard(request: Request, call_next):
        if check_hosts:
            host = request.headers.get("host", "")
            if _host_name(host) not in allowed_hosts:
                return JSONResponse({"detail": "Unexpected Host header."}, status_code=403)
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc.lower() != host.strip().lower():
                return JSONResponse({"detail": "Cross-origin requests are not allowed."}, status_code=403)

        workspace = request.cookies.get(COOKIE, "")
        fresh = not _WORKSPACE_RE.match(workspace)
        if fresh:
            workspace = uuid.uuid4().hex
        request.state.workspace = workspace

        response = await call_next(request)
        if fresh:
            response.set_cookie(
                COOKIE, workspace, max_age=365 * 24 * 3600, httponly=True, samesite="lax",
                secure=request.url.scheme == "https", path="/",
            )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )
        return response

    @app.exception_handler(ManagerError)
    async def manager_error(_: Request, exc: ManagerError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    def ws(request: Request) -> str:
        return request.state.workspace

    def owned(workspace: str, db_id: str) -> Dict[str, Any]:
        try:
            return manager.get_info(workspace, db_id)
        except KeyError:
            raise HTTPException(404, "Unknown database.") from None

    # ── pages ─────────────────────────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    def index() -> Response:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ── API: status ───────────────────────────────────────────────────────

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "version": __version__, "llm_ready": llm is not None, "model": settings.model,
            "max_rows": settings.max_rows, "max_steps": settings.max_steps,
            "preview_rows": settings.preview_rows, "redact_columns": list(settings.redact_columns),
            "allow_user_connections": settings.allow_user_connections,
            "max_databases": settings.max_databases, "sandbox_max_mb": settings.sandbox_max_mb,
        }
        if llm is None:
            info["message"] = "GEMINI_API_KEY is not set. Add it to your environment or a .env file and restart."
        elif hasattr(llm, "check_model"):
            now = time.monotonic()
            if model_check["value"] is None or now - model_check["at"] > MODEL_CHECK_TTL:
                model_check.update(at=now, value=llm.check_model())
            ok, detail = model_check["value"]
            info["model_ok"], info["model_detail"] = ok, detail
        return info

    # ── API: databases ────────────────────────────────────────────────────

    @app.get("/api/databases")
    def list_databases(request: Request) -> List[Dict[str, Any]]:
        return manager.list_databases(ws(request))

    @app.post("/api/databases", status_code=201)
    def create_database(body: CreateDatabaseBody, request: Request) -> Dict[str, Any]:
        workspace = ws(request)
        if body.kind == "connection":
            return manager.add_connection(workspace, body.name, body.url)
        return manager.create_sandbox(workspace, body.name, body.description, sample=body.sample)

    @app.get("/api/databases/{db_id}")
    def database_details(db_id: str, request: Request) -> Dict[str, Any]:
        owned(ws(request), db_id)
        try:
            return manager.describe(ws(request), db_id)
        except Exception as exc:
            raise HTTPException(502, f"Could not open the database: {str(exc).splitlines()[0]}") from exc

    @app.delete("/api/databases/{db_id}", status_code=204)
    def delete_database(db_id: str, request: Request) -> Response:
        owned(ws(request), db_id)
        manager.delete(ws(request), db_id)
        return Response(status_code=204)

    @app.post("/api/databases/{db_id}/undo")
    def undo(db_id: str, request: Request) -> Dict[str, Any]:
        owned(ws(request), db_id)
        return manager.undo(ws(request), db_id)

    @app.post("/api/databases/{db_id}/reset")
    def reset(db_id: str, request: Request) -> Dict[str, Any]:
        owned(ws(request), db_id)
        return manager.reset(ws(request), db_id)

    @app.get("/api/schema/{db_id}")
    def schema(db_id: str, request: Request) -> Dict[str, Any]:
        info = owned(ws(request), db_id)
        try:
            entry = manager.resolve(ws(request), db_id)
            return {"id": db_id, "name": info["name"], "backend": entry.backend, "tables": introspect.schema_overview(entry.flash),
                    "enforcement": entry.enforcement.as_dict()}
        except Exception as exc:
            raise HTTPException(502, f"Could not read the schema: {str(exc).splitlines()[0]}") from exc

    # ── API: prompt history ───────────────────────────────────────────────

    @app.get("/api/history")
    def history(request: Request, database_id: Optional[str] = None, q: Optional[str] = None,
                limit: int = 100) -> List[Dict[str, Any]]:
        rows = manager.store.list_prompts(ws(request), database_id=database_id, q=q, limit=limit)
        return [
            {"id": r["id"], "text": r["text"], "status": r["status"], "summary": r.get("summary") or "",
             "database_id": r.get("database_id"), "database_name": r.get("database_name"),
             "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else None}
            for r in rows
        ]

    @app.delete("/api/history/{prompt_id}", status_code=204)
    def delete_prompt(prompt_id: int, request: Request) -> Response:
        if not manager.store.delete_prompt(ws(request), prompt_id):
            raise HTTPException(404, "Unknown prompt.")
        return Response(status_code=204)

    @app.delete("/api/history")
    def clear_history(request: Request, database_id: Optional[str] = None) -> Dict[str, int]:
        return {"deleted": manager.store.clear_prompts(ws(request), database_id)}

    # ── API: chat ─────────────────────────────────────────────────────────

    @app.post("/api/ask")
    def ask(body: AskBody, request: Request) -> StreamingResponse:
        workspace = ws(request)
        database_name = owned(workspace, body.database_id)["name"] if body.database_id else None
        if agent is None:
            events: Iterator[Dict[str, Any]] = iter([
                {"type": "error", "kind": "config",
                 "message": "GEMINI_API_KEY is not set. Add it to your environment or a .env file and restart."}
            ])
        else:
            events = agent.ask_stream(sessions.get(workspace, body.session_id, body.database_id), body.question)
        events = _recorded(
            events, manager=manager, workspace=workspace, question=body.question.strip(),
            database_id=body.database_id, database_name=database_name,
        )
        return StreamingResponse(
            _sse(events), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/sessions/{session_id}/reset", status_code=204)
    def reset_session(session_id: str, request: Request) -> Response:
        sessions.reset(ws(request), session_id)
        return Response(status_code=204)

    return app


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    import os

    import uvicorn

    parser = argparse.ArgumentParser(
        prog="verbadb",
        description="VerbaDB: talk to your databases. Starts the web app. Settings come from VERBADB_* environment "
                    "variables or a .env file (see the README).",
    )
    parser.add_argument("--host", help="address to listen on (default: VERBADB_HOST, else 127.0.0.1)")
    parser.add_argument("--port", type=int, help="port to listen on (default: VERBADB_PORT, else 8000)")
    parser.add_argument("--version", action="version", version=f"VerbaDB {__version__}")
    args = parser.parse_args(argv)
    # Through the environment, so the host-dependent defaults (e.g. user connections) follow the chosen host.
    if args.host:
        os.environ["VERBADB_HOST"] = args.host
    if args.port:
        os.environ["VERBADB_PORT"] = str(args.port)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    app = create_app(settings)
    print(f"VerbaDB {__version__}: http://{settings.host}:{settings.port}  (model: {settings.model})")
    if not settings.databases:
        print("No shared databases configured (VERBADB_DATABASES): people can still create sandbox databases"
              + (" and connect their own." if settings.allow_user_connections else "."))
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
