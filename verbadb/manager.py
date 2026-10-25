"""
Which databases exist and who may use them.

Three kinds, all addressed by an opaque id and always checked against the caller's *workspace*:

* ``shared``     — configured by the administrator in ``VERBADB_DATABASES``; visible to everyone, strictly read-only.
* ``connection`` — a database a person connected themselves through the UI; private to their workspace and
                   strictly read-only (same guard as shared ones). Credentials stay on the server.
* ``sandbox``    — a dummy SQLite database inside the app, private to the workspace, where the agent may create
                   and change things (see ``sandbox.py``), with undo.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

from pyflashdb import FlashDB
from pyflashdb.urls import parse_url

from . import demo_data, introspect
from .config import Settings
from .connections import Enforcement, open_readonly
from .sandbox import Snapshots, open_sandbox
from .store import AppStore

log = logging.getLogger("verbadb.manager")

CONNECT_TIMEOUT = 20.0


class ManagerError(Exception):
    """A problem the user can fix; ``status`` is the HTTP status the API should use."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# ── names and URLs ────────────────────────────────────────────────────────

def clean_name(name: Any) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]", "", str(name or ""))
    text = " ".join(text.split())[:60]
    if not text:
        raise ManagerError("Give the database a name.")
    return text


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "db"


def mask_url(url: str) -> str:
    """The URL without its password (and without query options, which can carry secrets)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "(hidden)"
    netloc = parts.netloc
    if "@" in netloc:
        credentials, host = netloc.rsplit("@", 1)
        netloc = f"{credentials.split(':', 1)[0]}:***@{host}" if ":" in credentials else f"{credentials}@{host}"
    return f"{parts.scheme}://{netloc}{parts.path}"


def redact(message: str, url: str) -> str:
    """Remove the URL's password (and the URL itself) from an error message before showing it."""
    try:
        password = urlsplit(url).password
    except ValueError:
        password = None
    for secret in {url, password, unquote(password) if password else None}:
        if secret:
            message = message.replace(secret, "***")
    return message


def _hosts_of(url: str, base: str) -> List[Tuple[str, Optional[int]]]:
    if base == "mongodb":
        netloc = urlsplit(url).netloc.rsplit("@", 1)[-1]
        hosts = []
        for item in filter(None, netloc.split(",")):
            host, _, port = item.rpartition(":") if not item.endswith("]") and ":" in item else (item, "", "")
            host = (host or item).strip("[]")
            hosts.append((host, int(port) if port.isdigit() else None))
        return hosts
    parts = urlsplit(url)
    return [(parts.hostname, parts.port)] if parts.hostname else []


def _check_host(host: str, port: Optional[int], settings: Settings, resolve: bool) -> None:
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        if not resolve:
            return
        try:
            infos = socket.getaddrinfo(host, port or None, proto=socket.IPPROTO_TCP)
            addresses = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos]
        except (OSError, ValueError) as exc:
            raise ManagerError(f"The host '{host}' could not be found.") from exc
    for ip in addresses:
        ip = getattr(ip, "ipv4_mapped", None) or ip
        local = ip.is_loopback or ip.is_private            # note: Python also calls IPv6 ::1 "reserved"
        if ip.is_link_local or ip.is_unspecified or ip.is_multicast or (ip.is_reserved and not local):
            raise ManagerError(f"Connections to {ip} are not allowed.")
        if local and not settings.allow_private_db_hosts:
            raise ManagerError(
                "Connections to this machine or private network addresses are disabled on this server "
                "(VERBADB_ALLOW_PRIVATE_DB_HOSTS)."
            )


def validate_user_url(url: Any, settings: Settings) -> str:
    """Check a connection URL typed into the UI; returns the backend name. Raises :class:`ManagerError`."""
    if not isinstance(url, str) or not url.strip():
        raise ManagerError("Enter a connection URL.")
    url = url.strip()
    if len(url) > 2000 or re.search(r"[\s\x00-\x1f]", url) or "://" not in url:
        raise ManagerError("That does not look like a valid connection URL, e.g. postgresql://user:password@host:5432/dbname")
    scheme = url.split("://", 1)[0].lower()
    base = scheme.split("+")[0]
    if base == "sqlite":
        raise ManagerError(
            "SQLite files cannot be added here, because the app would be reading files from its own machine. "
            "Create a sandbox database instead, or ask the administrator to add it to VERBADB_DATABASES."
        )
    if base not in ("mysql", "postgres", "postgresql", "mongodb"):
        raise ManagerError("Supported connection URLs start with mysql://, postgresql:// or mongodb://.")
    hosts = _hosts_of(url, base)
    if not hosts:
        raise ManagerError("The URL has no host name.")
    for host, port in hosts:
        _check_host(host, port, settings, resolve=not scheme.endswith("+srv"))
    return parse_url(url)[0]


# ── entries ───────────────────────────────────────────────────────────────

@dataclass
class DbEntry:
    id: str
    name: str
    kind: str                                   # "shared" | "connection" | "sandbox"
    backend: str
    writable: bool
    flash: Any
    enforcement: Enforcement
    description: str = ""
    snapshots: Optional[Snapshots] = None       # sandboxes only
    lock: Any = field(default_factory=threading.RLock)   # serialises changes / undo of one sandbox

    def info(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "kind": self.kind, "backend": self.backend, "writable": self.writable}


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if hasattr(value, "isoformat") else None


class DatabaseManager:
    def __init__(self, settings: Settings, store: Optional[AppStore] = None) -> None:
        self.settings = settings
        self.data_dir = Path(settings.data_dir).resolve()
        (self.data_dir / "sandboxes").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self.store = store or AppStore(self.data_dir, settings.max_saved_prompts)
        self._lock = threading.RLock()
        self._open: Dict[str, DbEntry] = {}
        self._open_locks: Dict[str, threading.Lock] = {}
        self._db_locks: Dict[str, threading.RLock] = {}
        self._shared: Dict[str, Tuple[str, str]] = {}
        for name, url in settings.databases.items():
            key, n = f"shared-{slug(name)}", 2
            while key in self._shared:
                key, n = f"shared-{slug(name)}-{n}", n + 1
            self._shared[key] = (name, url)

    # ── paths and locks ───────────────────────────────────────────────────

    def sandbox_path(self, db_id: str) -> Path:
        return self.data_dir / "sandboxes" / f"{db_id}.db"

    def snapshots_for(self, db_id: str) -> Snapshots:
        return Snapshots(self.data_dir / "snapshots" / db_id, self.sandbox_path(db_id))

    def db_lock(self, db_id: str) -> Any:
        with self._lock:
            return self._db_locks.setdefault(db_id, threading.RLock())

    # ── listing ───────────────────────────────────────────────────────────

    def _row_info(self, row: Dict[str, Any]) -> Dict[str, Any]:
        sandbox = row["kind"] == "sandbox"
        info: Dict[str, Any] = {
            "id": row["public_id"], "name": row["name"], "kind": row["kind"],
            "backend": "sqlite" if sandbox else parse_url(row["url"])[0], "writable": sandbox,
            "description": row.get("description") or "", "created_at": _iso(row.get("created_at")),
        }
        if sandbox:
            info["undo_depth"] = self.snapshots_for(row["public_id"]).depth()
        else:
            info["url"] = mask_url(row["url"])
        return info

    def list_databases(self, workspace: str) -> List[Dict[str, Any]]:
        shared = [
            {"id": key, "name": name, "kind": "shared", "backend": parse_url(url)[0], "writable": False,
             "description": "Configured by the administrator. Read-only.", "created_at": None}
            for key, (name, url) in self._shared.items()
        ]
        return shared + [self._row_info(r) for r in self.store.list_databases(workspace)]

    def get_info(self, workspace: str, db_id: str) -> Dict[str, Any]:
        for info in self.list_databases(workspace):
            if info["id"] == db_id:
                return info
        raise KeyError(db_id)

    # ── opening ───────────────────────────────────────────────────────────

    def resolve(self, workspace: str, db_id: str) -> DbEntry:
        """The open database ``db_id`` if ``workspace`` may use it (``KeyError`` otherwise)."""
        row = None
        if db_id not in self._shared:
            row = self.store.get_database(workspace, db_id)
            if row is None:
                raise KeyError(db_id)
        with self._lock:
            entry = self._open.get(db_id)
            if entry is not None:
                return entry
            opening = self._open_locks.setdefault(db_id, threading.Lock())
        with opening:                                   # a slow remote host must not block everyone else
            with self._lock:
                entry = self._open.get(db_id)
            if entry is None:
                entry = self._open_entry(db_id, row)
                with self._lock:
                    self._open[db_id] = entry
            return entry

    def _open_entry(self, db_id: str, row: Optional[Dict[str, Any]]) -> DbEntry:
        if row is None:
            name, url = self._shared[db_id]
            flash, enforcement = open_readonly(url, self.settings)
            return DbEntry(db_id, name, "shared", flash.backend, False, flash, enforcement)
        if row["kind"] == "sandbox":
            flash, enforcement = open_sandbox(self.sandbox_path(db_id), self.settings)
            return DbEntry(
                db_id, row["name"], "sandbox", "sqlite", True, flash, enforcement,
                description=row.get("description") or "", snapshots=self.snapshots_for(db_id), lock=self.db_lock(db_id),
            )
        flash, enforcement = open_readonly(row["url"], self.settings)
        return DbEntry(db_id, row["name"], "connection", flash.backend, False, flash, enforcement)

    def _close(self, db_id: str) -> None:
        with self._lock:
            entry = self._open.pop(db_id, None)
        if entry is not None:
            try:
                entry.flash.close()
            except Exception:
                pass

    def describe(self, workspace: str, db_id: str) -> Dict[str, Any]:
        """Info plus live details (table count, what protections are in force)."""
        info = self.get_info(workspace, db_id)
        entry = self.resolve(workspace, db_id)
        info["tables"] = len(introspect.known_tables(entry.flash))
        info["enforcement"] = entry.enforcement.as_dict()
        if entry.snapshots is not None:
            info["undo_depth"] = entry.snapshots.depth()
        return info

    # ── creating, adding, deleting ────────────────────────────────────────

    def _owned(self, workspace: str) -> List[Dict[str, Any]]:
        return self.store.list_databases(workspace)

    def _check_quota(self, workspace: str) -> None:
        if len(self._owned(workspace)) >= self.settings.max_databases:
            raise ManagerError(
                f"You can have up to {self.settings.max_databases} databases of your own. Delete one first.", 409
            )

    def _unique_name(self, workspace: str, name: str) -> str:
        taken = {r["name"].lower() for r in self._owned(workspace)} | {n.lower() for n, _ in self._shared.values()}
        candidate, n = name, 2
        while candidate.lower() in taken:
            candidate, n = f"{name} ({n})", n + 1
        return candidate

    def create_sandbox(self, workspace: str, name: Any, description: Any = "", sample: bool = False) -> Dict[str, Any]:
        name = clean_name(name)
        self._check_quota(workspace)
        row = self.store.add_database(
            workspace, self._unique_name(workspace, name), "sandbox", description=str(description or "")[:500]
        )
        if sample:
            with FlashDB.from_url(f"sqlite:///{self.sandbox_path(row['public_id']).as_posix()}") as writer:
                demo_data.build_demo(writer)
        return self._row_info(row)

    def add_connection(self, workspace: str, name: Any, url: Any) -> Dict[str, Any]:
        if not self.settings.allow_user_connections:
            raise ManagerError("Adding your own database connections is disabled on this server.", 403)
        name = clean_name(name)
        validate_user_url(url, self.settings)
        url = url.strip()
        self._check_quota(workspace)

        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(open_readonly, url, self.settings)
        try:
            flash, enforcement = future.result(timeout=CONNECT_TIMEOUT)
        except FutureTimeout:
            future.add_done_callback(lambda f: f.exception() is None and f.result()[0].close())   # no leaked connection
            raise ManagerError(f"Timed out after {CONNECT_TIMEOUT:g}s trying to connect.") from None
        except Exception as exc:
            first_line = (str(exc).splitlines() or [exc.__class__.__name__])[0]
            raise ManagerError(f"Could not connect: {redact(first_line, url)}") from None
        finally:
            pool.shutdown(wait=False)

        row = self.store.add_database(workspace, self._unique_name(workspace, name), "connection", url=url)
        entry = DbEntry(row["public_id"], row["name"], "connection", flash.backend, False, flash, enforcement)
        with self._lock:
            self._open[entry.id] = entry
        return self._row_info(row)

    def delete(self, workspace: str, db_id: str) -> None:
        if db_id in self._shared:
            raise ManagerError("Databases configured by the administrator cannot be removed here.", 403)
        row = self.store.get_database(workspace, db_id)
        if row is None:
            raise KeyError(db_id)
        with self.db_lock(db_id):
            self._close(db_id)
            if row["kind"] == "sandbox":
                self.sandbox_path(db_id).unlink(missing_ok=True)
                self.snapshots_for(db_id).clear()
            self.store.delete_database(workspace, db_id)
        with self._lock:
            self._db_locks.pop(db_id, None)
            self._open_locks.pop(db_id, None)

    # ── sandbox undo / reset ──────────────────────────────────────────────

    def _sandbox_row(self, workspace: str, db_id: str) -> Dict[str, Any]:
        if db_id in self._shared:
            raise ManagerError("Only sandbox databases can be changed.", 403)
        row = self.store.get_database(workspace, db_id)
        if row is None:
            raise KeyError(db_id)
        if row["kind"] != "sandbox":
            raise ManagerError("Only sandbox databases can be changed.", 403)
        return row

    def undo(self, workspace: str, db_id: str) -> Dict[str, Any]:
        self._sandbox_row(workspace, db_id)
        with self.db_lock(db_id):
            self._close(db_id)                              # the file cannot be replaced while it is open
            snapshots = self.snapshots_for(db_id)
            try:
                restored = snapshots.restore_latest()
            except LookupError:
                raise ManagerError("There is nothing to undo.", 409) from None
        return {"undone": restored.get("summary", ""), "undo_depth": snapshots.depth()}

    def reset(self, workspace: str, db_id: str) -> Dict[str, Any]:
        self._sandbox_row(workspace, db_id)
        with self.db_lock(db_id):
            self._close(db_id)
            snapshots = self.snapshots_for(db_id)
            snapshots.take("Reset: removed all tables and data")        # so even a reset can be undone
            self.sandbox_path(db_id).unlink(missing_ok=True)
        return {"undo_depth": snapshots.depth()}

    def close_all(self) -> None:
        with self._lock:
            ids = list(self._open)
        for db_id in ids:
            self._close(db_id)
        self.store.close()
