"""
Sandbox databases: SQLite files owned by the app, where the agent *may* create tables and change data.

Writes are still tightly controlled. Every change goes through :func:`apply_changes`, which

1. validates each statement with the same parser-based approach as the read guard (only INSERT / UPDATE /
   DELETE and CREATE / DROP / ALTER of tables and indexes — no PRAGMA, ATTACH, triggers, views, file or
   extension functions), and executes the regenerated SQL;
2. snapshots the database file first, so the change can be undone;
3. runs the batch in one transaction (all or nothing), under a time limit and the sandbox size cap;
4. compares the database before and after, which is the "simulation": which tables were created or dropped,
   which rows were inserted, deleted or edited (old -> new), including side effects such as FK cascades.

External databases never get any of this: they stay strictly read-only (``guard.py`` / ``connections.py``).
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlglot import exp

from . import executor, guard
from .config import Settings
from .connections import Enforcement
from .guard import GuardError

log = logging.getLogger("verbadb.sandbox")

MAX_STATEMENTS = 30
MAX_STATEMENT_CHARS = 20_000
DIFF_ROW_CAP = 2_000        # tables with more rows than this are compared by count only
DIFF_SHOW = 25              # changed rows shown per kind per table
MAX_SNAPSHOTS = 20

_WRITE_ROOTS = guard._classes("Insert", "Update", "Delete", "Create", "Drop", "Alter")
_WRITE_FORBIDDEN = guard._classes(
    "Command", "Set", "Use", "Transaction", "Commit", "Rollback", "TruncateTable", "LoadData", "Pragma", "Copy",
    "Analyze", "Describe", "Show", "Grant", "Revoke", "Kill", "Attach", "Detach", "Declare", "Execute", "Lock",
)
_ALLOWED_KINDS = frozenset({"TABLE", "INDEX"})
# Virtual tables load SQLite modules; temporary tables live outside the file we snapshot and compare.
_FORBIDDEN_PROPERTIES = guard._classes("VirtualProperty", "ModuleProperty", "TemporaryProperty")


class ChangeFailed(Exception):
    """One statement of a batch failed; the whole batch was rolled back."""

    def __init__(self, index: int, message: str) -> None:
        super().__init__(f"Statement {index} failed and the whole batch was rolled back: {message}")
        self.index = index


# ── Write guard ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PreparedWrite:
    sql: str        # what will actually run (regenerated from the validated tree)
    verb: str       # e.g. "INSERT", "CREATE TABLE"


def _verb(root: exp.Expression) -> str:
    name = type(root).__name__.upper()
    if isinstance(root, (exp.Create, exp.Drop, exp.Alter)):
        return f"{name} {str(root.args.get('kind') or '').upper()}".strip()
    return name


def _check_write_tree(stmt: exp.Expression) -> str:
    if isinstance(stmt, (exp.Select, exp.SetOperation, exp.Subquery)):
        raise GuardError("That is a read: use run_sql for SELECT. modify_database is for changing data or structure.")
    if isinstance(stmt, exp.Command):
        raise GuardError(
            "This statement uses syntax the safety check cannot verify (REPLACE INTO, WITHOUT ROWID, triggers, "
            "VACUUM, ... are not supported). Rewrite it with plain INSERT [OR REPLACE] / UPDATE / DELETE / "
            "CREATE TABLE / ALTER TABLE."
        )
    if not isinstance(stmt, _WRITE_ROOTS):
        raise GuardError(
            f"{type(stmt).__name__.upper()} is not allowed. modify_database accepts INSERT, UPDATE, DELETE and "
            "CREATE / DROP / ALTER of tables and indexes."
        )
    if isinstance(stmt, (exp.Create, exp.Drop, exp.Alter)):
        kind = str(stmt.args.get("kind") or "").upper()
        if kind not in _ALLOWED_KINDS:
            raise GuardError(
                f"{type(stmt).__name__.upper()} {kind or '(unknown)'} is not allowed: only tables and indexes can be "
                "created, altered or dropped here (no views, triggers or virtual tables)."
            )
    for node in stmt.walk():
        if _WRITE_FORBIDDEN and isinstance(node, _WRITE_FORBIDDEN):
            raise GuardError(f"{type(node).__name__.upper()} is not allowed in the sandbox.")
        if _FORBIDDEN_PROPERTIES and isinstance(node, _FORBIDDEN_PROPERTIES):
            raise GuardError("Virtual and temporary tables are not allowed: create a normal table instead.")
        if isinstance(node, exp.Func):
            name = guard.function_name(node)
            if guard.is_forbidden_function(name):
                raise GuardError(f"Function {name}() is not allowed.")
        if isinstance(node, exp.Table) and node.name.lower().startswith("sqlite_"):
            raise GuardError("Names starting with 'sqlite_' are reserved by SQLite and cannot be written to.")
    return _verb(stmt)


def prepare_writes(statements: Any) -> List[PreparedWrite]:
    """Validate a batch of write statements; returns the regenerated SQL to run, or raises :class:`GuardError`."""
    if not isinstance(statements, list) or not statements or not all(isinstance(s, str) for s in statements):
        raise GuardError("'statements' must be a non-empty list of SQL strings.")
    if len(statements) > MAX_STATEMENTS:
        raise GuardError(f"Too many statements in one call (max {MAX_STATEMENTS}). Split the work over several calls.")
    prepared: List[PreparedWrite] = []
    for number, sql in enumerate(statements, start=1):
        try:
            if not sql.strip():
                raise GuardError("The statement is empty.")
            if len(sql) > MAX_STATEMENT_CHARS:
                raise GuardError(f"The statement is too long (max {MAX_STATEMENT_CHARS} characters).")
            guard.check_characters(sql)
            stmt = guard._parse_one(sql, "sqlite")
            verb = _check_write_tree(stmt)
            final = stmt.sql(dialect="sqlite", comments=False)
            _check_write_tree(guard._parse_one(final, "sqlite"))       # what runs must itself pass
        except GuardError as exc:
            raise GuardError(f"Statement {number}: {exc}") from exc
        prepared.append(PreparedWrite(final, verb))
    return prepared


def check_sandbox_statement(info: Dict[str, Any]) -> None:
    """``before_query`` hook for sandbox connections: reads are vetted as reads, everything else as a write."""
    if guard._in_trusted_scope():
        return
    if info.get("backend") != "sqlite":
        raise GuardError("Sandbox databases are SQLite.")
    text = str(info.get("statement")).replace("%s", "?")
    guard.check_characters(text)
    stmt = guard._parse_one(text, "sqlite")
    if isinstance(stmt, (exp.Select, exp.SetOperation, exp.Subquery)):
        guard._check_tree(stmt)
    else:
        _check_write_tree(stmt)


def open_sandbox(path: Path, settings: Settings) -> "tuple[Any, Enforcement]":
    """Open a sandbox file: size-capped, time-limited, and guarded by the sandbox hook."""
    from pyflashdb import FlashDB

    flash = FlashDB.from_url(f"sqlite:///{Path(path).as_posix()}")
    enforcement = Enforcement(session_read_only="n/a", statement_timeout="not enforced")
    try:
        rows = flash.raw("PRAGMA page_size")
        page_size = int(next(iter(rows[0].values()))) if rows else 4096
        flash.raw(f"PRAGMA max_page_count = {max(16, settings.sandbox_max_mb * 1024 * 1024 // page_size)}")
        # Last: once the engine-level authorizer is on, only catalog PRAGMAs are allowed any more.
        if guard.install_sqlite_hardening(flash, writable=True):
            enforcement.statement_timeout = "set"
        flash.before_query(check_sandbox_statement)
    except Exception:
        flash.close()
        raise
    enforcement.notes.append(
        f"Sandbox: lives inside the app, may be changed (undoable), capped at {settings.sandbox_max_mb} MB."
    )
    return flash, enforcement


# ── Snapshots (undo) ──────────────────────────────────────────────────────

class Snapshots:
    """A stack of copies of one sandbox file, newest last. The database must be closed while restoring."""

    def __init__(self, directory: Path, db_path: Path) -> None:
        self.dir = Path(directory)
        self.db_path = Path(db_path)

    def _ids(self) -> List[str]:
        return sorted(p.stem for p in self.dir.glob("*.db")) if self.dir.exists() else []

    def depth(self) -> int:
        return len(self._ids())

    def take(self, summary: str = "") -> str:
        self.dir.mkdir(parents=True, exist_ok=True)
        snap_id = f"{time.time_ns()}"
        if self.db_path.exists():
            shutil.copyfile(self.db_path, self.dir / f"{snap_id}.db")
        else:
            (self.dir / f"{snap_id}.db").write_bytes(b"")
        self._write_meta(snap_id, summary)
        for old in self._ids()[:-MAX_SNAPSHOTS]:
            self.discard(old)
        return snap_id

    def _write_meta(self, snap_id: str, summary: str) -> None:
        (self.dir / f"{snap_id}.json").write_text(json.dumps({"summary": summary}), encoding="utf-8")

    def set_summary(self, snap_id: str, summary: str) -> None:
        if (self.dir / f"{snap_id}.db").exists():
            self._write_meta(snap_id, summary)

    def discard(self, snap_id: str) -> None:
        for suffix in (".db", ".json"):
            try:
                (self.dir / f"{snap_id}{suffix}").unlink()
            except FileNotFoundError:
                pass

    def peek(self) -> Optional[Dict[str, Any]]:
        ids = self._ids()
        if not ids:
            return None
        try:
            meta = json.loads((self.dir / f"{ids[-1]}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        return {"id": ids[-1], "summary": meta.get("summary", "")}

    def restore_latest(self) -> Dict[str, Any]:
        latest = self.peek()
        if latest is None:
            raise LookupError("Nothing to undo.")
        snapshot = self.dir / f"{latest['id']}.db"
        if snapshot.stat().st_size == 0:
            self.db_path.unlink(missing_ok=True)          # the database did not exist yet
        else:
            shutil.copyfile(snapshot, self.db_path)
        self.discard(latest["id"])
        return latest

    def clear(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


# ── Before / after comparison ─────────────────────────────────────────────

def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def capture_state(flash: Any) -> Dict[str, Dict[str, Any]]:
    """Every table's columns, key, row count and (for small tables) rows keyed by primary key / rowid."""
    state: Dict[str, Dict[str, Any]] = {}
    with guard.trusted_scope():
        for name in flash.show_tables():
            columns = flash.describe(name)
            pk = [c["name"] for c in columns if c.get("primary_key")]
            count = int(flash.count(name))
            rows: Optional[Dict[str, Dict[str, Any]]] = None
            if count <= DIFF_ROW_CAP:
                try:
                    raw = flash.raw(f"SELECT rowid AS __rowid__, * FROM {_q(name)}")
                except Exception:                              # WITHOUT ROWID tables
                    raw = flash.raw(f"SELECT * FROM {_q(name)}")
                rows = {}
                for record in raw:
                    row = {k: executor.jsonable(v) for k, v in record.items()}
                    rowid = row.pop("__rowid__", None)
                    key = [row.get(c) for c in pk] if pk else [rowid]
                    rows[json.dumps(key, default=str)] = row
            state[name] = {"columns": [c["name"] for c in columns], "pk": pk, "count": count, "rows": rows}
    return state


def diff_states(before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Describe what changed between two :func:`capture_state` results (unchanged tables are left out)."""
    changes: List[Dict[str, Any]] = []
    for name in sorted(set(before) | set(after)):
        b, a = before.get(name), after.get(name)
        if b is None or a is None:
            present = a if a is not None else b
            kind = "created" if b is None else "dropped"
            rows = list((present["rows"] or {}).values())
            entry: Dict[str, Any] = {
                "name": name, "change": kind, "columns": present["columns"], "row_diff": present["rows"] is not None,
                "before_count": 0 if b is None else b["count"], "after_count": 0 if a is None else a["count"],
                "inserted": rows[:DIFF_SHOW] if kind == "created" else [],
                "deleted": rows[:DIFF_SHOW] if kind == "dropped" else [],
                "updated": [], "inserted_total": present["count"] if kind == "created" else 0,
                "deleted_total": present["count"] if kind == "dropped" else 0, "updated_total": 0,
                "columns_added": [], "columns_removed": [],
            }
            changes.append(entry)
            continue

        added = [c for c in a["columns"] if c not in b["columns"]]
        removed = [c for c in b["columns"] if c not in a["columns"]]
        common = [c for c in a["columns"] if c in b["columns"]]
        inserted: List[Dict[str, Any]] = []
        deleted: List[Dict[str, Any]] = []
        updated: List[Dict[str, Any]] = []
        totals = [0, 0, 0]
        row_diff = a["rows"] is not None and b["rows"] is not None
        if row_diff:
            for key, row in a["rows"].items():
                if key not in b["rows"]:
                    totals[0] += 1
                    if len(inserted) < DIFF_SHOW:
                        inserted.append(row)
                    continue
                old = b["rows"][key]
                changed = [c for c in common if old.get(c) != row.get(c)]
                if changed:
                    totals[2] += 1
                    if len(updated) < DIFF_SHOW:
                        updated.append({"before": old, "after": row, "changed": changed})
            for key, row in b["rows"].items():
                if key not in a["rows"]:
                    totals[1] += 1
                    if len(deleted) < DIFF_SHOW:
                        deleted.append(row)
        elif a["count"] == b["count"] and not added and not removed:
            continue                                     # too big to compare and the count did not move
        if not (added or removed or any(totals)) and row_diff:
            continue
        changes.append({
            "name": name, "change": "altered" if (added or removed) else "modified",
            "columns": a["columns"], "row_diff": row_diff, "before_count": b["count"], "after_count": a["count"],
            "inserted": inserted, "deleted": deleted, "updated": updated,
            "inserted_total": totals[0], "deleted_total": totals[1], "updated_total": totals[2],
            "columns_added": added, "columns_removed": removed,
        })
    return changes


def _summarise(statements: int, tables: List[Dict[str, Any]]) -> str:
    created = sum(1 for t in tables if t["change"] == "created")
    dropped = sum(1 for t in tables if t["change"] == "dropped")
    altered = sum(1 for t in tables if t["change"] == "altered")
    plus = sum(t["inserted_total"] for t in tables if t["change"] in ("modified", "altered"))
    minus = sum(t["deleted_total"] for t in tables if t["change"] in ("modified", "altered"))
    edited = sum(t["updated_total"] for t in tables)
    rows_in_new = sum(t["after_count"] for t in tables if t["change"] == "created")
    parts: List[str] = []
    if created:
        parts.append(f"created {created} table{'s' if created != 1 else ''}")
    if dropped:
        parts.append(f"dropped {dropped} table{'s' if dropped != 1 else ''}")
    if altered:
        parts.append(f"altered {altered} table{'s' if altered != 1 else ''}")
    if plus or rows_in_new:
        parts.append(f"+{plus + rows_in_new} row{'s' if plus + rows_in_new != 1 else ''}")
    if minus:
        parts.append(f"−{minus} row{'s' if minus != 1 else ''}")
    if edited:
        parts.append(f"{edited} row{'s' if edited != 1 else ''} edited")
    head = f"Applied {statements} statement{'s' if statements != 1 else ''}"
    return f"{head}: " + " · ".join(parts) if parts else f"{head}: no visible change"


@dataclass
class ChangeSet:
    statements: List[str]
    verbs: List[str]
    rowcounts: List[int]
    tables: List[Dict[str, Any]]
    summary: str
    snapshot_id: Optional[str]
    elapsed_ms: float = 0.0
    database_id: str = ""

    def to_event(self) -> Dict[str, Any]:
        return {
            "type": "change", "database_id": self.database_id, "summary": self.summary,
            "statements": self.statements, "verbs": self.verbs, "tables": self.tables,
            "can_undo": self.snapshot_id is not None, "snapshot_id": self.snapshot_id,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }

    def model_view(self) -> Dict[str, Any]:
        """Counts only: what the model needs to continue, without sending row contents back to the API."""
        return {
            "ok": True, "summary": self.summary, "statements_run": len(self.statements),
            "changes": [
                {"table": t["name"], "change": t["change"], "rows_before": t["before_count"],
                 "rows_after": t["after_count"], "inserted": t["inserted_total"], "updated": t["updated_total"],
                 "deleted": t["deleted_total"],
                 **({"columns_added": t["columns_added"]} if t["columns_added"] else {})}
                for t in self.tables
            ],
            "note": "The user sees a before/after view of this change and can undo it.",
        }


def apply_changes(flash: Any, statements: Any, settings: Settings, snapshots: Snapshots) -> ChangeSet:
    """Validate, snapshot, apply atomically and compare. On any failure nothing is changed."""
    prepared = prepare_writes(statements)
    before = capture_state(flash)
    snap_id = snapshots.take("(change in progress)")
    started = time.perf_counter()
    rowcounts: List[int] = []
    try:
        with guard.statement_deadline(settings.query_timeout) as deadline:
            with flash.transaction():
                for number, item in enumerate(prepared, start=1):
                    try:
                        result = flash.raw(item.sql)
                    except GuardError:
                        raise
                    except Exception as exc:
                        if deadline.expired:
                            raise ChangeFailed(number, f"stopped after the {settings.query_timeout}s time limit") from exc
                        message = executor._clean_error(exc)
                        if "database or disk is full" in message.lower():
                            message = f"the sandbox is full ({settings.sandbox_max_mb} MB limit)"
                        raise ChangeFailed(number, message) from exc
                    rowcounts.append(int(result) if isinstance(result, int) else 0)
    except BaseException:
        snapshots.discard(snap_id)
        raise
    elapsed = (time.perf_counter() - started) * 1000

    tables = diff_states(before, capture_state(flash))
    summary = _summarise(len(prepared), tables)
    if tables:
        snapshots.set_summary(snap_id, summary)
    else:
        snapshots.discard(snap_id)                  # nothing changed, so there is nothing to undo
        snap_id = None                              # type: ignore[assignment]
    return ChangeSet(
        statements=[p.sql for p in prepared], verbs=[p.verb for p in prepared], rowcounts=rowcounts,
        tables=tables, summary=summary, snapshot_id=snap_id, elapsed_ms=elapsed,
    )
