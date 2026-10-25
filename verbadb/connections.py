"""Opening a database read-only: session-level protections (applied and verified) plus the query guard."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from pyflashdb import FlashDB
from pyflashdb.urls import parse_url

from . import guard
from .config import Settings

log = logging.getLogger("verbadb.connections")


@dataclass
class Enforcement:
    """What was actually put in place for a connection (shown in the UI so nobody has to guess)."""

    guard: bool = True                  # in-code SQL/pipeline guard + before_query hook
    session_read_only: str = "n/a"      # "verified" | "not enforced" | "n/a"
    statement_timeout: str = "n/a"      # "set" | "not enforced" | "n/a"
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "guard": self.guard, "session_read_only": self.session_read_only,
            "statement_timeout": self.statement_timeout, "notes": self.notes,
        }


def _first_value(rows: Any) -> Any:
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return next(iter(rows[0].values()), None)
    return None


def _hardened_args(backend: str, timeout_s: int) -> Dict[str, Any]:
    ms = int(timeout_s) * 1000
    if backend == "postgres":
        return {"connect_args": {"options": f"-c default_transaction_read_only=on -c statement_timeout={ms}"}}
    if backend == "mysql":
        return {"connect_args": {
            "init_command": f"SET SESSION transaction_read_only = 1, SESSION max_execution_time = {ms}",
        }}
    if backend == "mongodb":
        return {"connect_args": {"timeoutMS": ms}}
    return {}


def _verify(flash: Any, backend: str, enforcement: Enforcement, timeout_s: int) -> None:
    """Confirm the session-level protections really are on; record the truth either way."""
    try:
        if backend == "sqlite":
            flash.raw("PRAGMA query_only = ON")
            on = _first_value(flash.raw("PRAGMA query_only"))
            enforcement.session_read_only = "verified" if str(on) in ("1", "True") else "not enforced"
            if guard.install_sqlite_hardening(flash):
                enforcement.statement_timeout = "set"
            else:
                enforcement.statement_timeout = "not enforced"
                enforcement.notes.append("SQLite has no statement timeout; a very heavy query can run long.")
        elif backend == "postgres":
            on = _first_value(flash.raw("SHOW transaction_read_only"))
            enforcement.session_read_only = "verified" if str(on).lower() == "on" else "not enforced"
            enforcement.statement_timeout = "set"
        elif backend == "mysql":
            on = _first_value(flash.raw("SELECT @@session.transaction_read_only AS ro"))
            enforcement.session_read_only = "verified" if str(on) in ("1", "True") else "not enforced"
            enforcement.statement_timeout = "set" if enforcement.session_read_only == "verified" else "not enforced"
        else:  # mongodb
            enforcement.session_read_only = "n/a"
            enforcement.statement_timeout = "set"
            enforcement.notes.append("MongoDB has no read-only session; use a read-only database role.")
    except Exception as exc:  # verification must never take the connection down
        enforcement.session_read_only = "not enforced"
        enforcement.notes.append(f"Could not verify session settings: {exc}")
    if enforcement.session_read_only == "not enforced":
        enforcement.notes.append("Rely on a read-only database account as the backstop.")


def open_readonly(url: str, settings: Settings) -> Tuple[Any, Enforcement]:
    """Connect, apply and verify session-level protections, then install the query guard."""
    backend, _ = parse_url(url)
    enforcement = Enforcement()
    hardened = _hardened_args(backend, settings.query_timeout)
    try:
        flash = FlashDB.from_url(url, **hardened)
    except Exception:
        if not hardened:
            raise
        # e.g. MariaDB has no max_execution_time; retry plainly so a real error (bad password) still surfaces.
        log.warning("Connecting with session hardening failed for a %s database; retrying without it.", backend)
        flash = FlashDB.from_url(url)
        enforcement.notes.append("Session hardening options were rejected by the server and not applied.")
    try:
        _verify(flash, backend, enforcement, settings.query_timeout)
        guard.install_hook(flash)
    except Exception:
        flash.close()
        raise
    return flash, enforcement
