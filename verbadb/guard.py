"""
Read-only enforcement — in code, not in the prompt.

SQL goes through a real parser (sqlglot): exactly one statement, it must be a SELECT (or a set operation of
SELECTs), and the whole tree is walked for anything that writes, locks, changes settings or calls a dangerous
function. What is finally *executed* is the SQL regenerated from the validated tree (comments dropped, a row
limit injected), never the model's raw text — so MySQL ``/*! ... */`` executable comments and similar parser
differentials cannot smuggle anything past the check.

MongoDB pipelines are walked recursively (including ``$lookup`` / ``$facet`` sub-pipelines) for write stages
and server-side JavaScript.

``install_hook`` registers the same checks as a pyflashdb ``before_query`` hook, so every statement that
reaches the database through the agent's connection is vetted, whichever code path produced it.
"""

from __future__ import annotations

import json
import threading
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Sequence, Set, Tuple

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


class GuardError(Exception):
    """A query was rejected because it is not a safe, read-only query."""


MAX_SQL_CHARS = 20_000
MAX_PIPELINE_STAGES = 40

_DIALECTS = {"mysql": "mysql", "postgres": "postgres", "sqlite": "sqlite"}


def _classes(*names: str) -> tuple:
    return tuple(getattr(exp, name) for name in names if hasattr(exp, name))


# Anything that writes, locks, changes session state or administers the server.
_FORBIDDEN_NODES = _classes(
    "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "Command", "Set", "Use",
    "Transaction", "Commit", "Rollback", "TruncateTable", "Into", "Lock", "LoadData", "Pragma", "Copy",
    "Analyze", "Describe", "Show", "Grant", "Revoke", "Kill", "Attach", "Detach", "Declare", "Execute",
)

# Functions that sleep, touch the filesystem or network, run SQL from a string, take locks or write sequences.
_FORBIDDEN_FUNCTIONS = frozenset({
    # PostgreSQL
    "pg_sleep", "pg_sleep_for", "pg_sleep_until", "pg_read_file", "pg_read_binary_file", "pg_ls_dir",
    "pg_stat_file", "pg_ls_logdir", "pg_ls_waldir", "lo_import", "lo_export", "lo_get", "lo_put",
    "lo_create", "lo_unlink", "dblink", "dblink_exec", "dblink_connect", "dblink_send_query",
    "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf", "pg_rotate_logfile", "set_config",
    "nextval", "setval", "lastval", "query_to_xml", "query_to_xml_and_xmlschema", "table_to_xml",
    "cursor_to_xml", "pg_advisory_lock", "pg_advisory_xact_lock", "pg_try_advisory_lock",
    "pg_create_restore_point", "pg_switch_wal", "pg_start_backup", "pg_stop_backup",
    # MySQL
    "sleep", "benchmark", "load_file", "get_lock", "release_lock", "release_all_locks",
    "master_pos_wait", "sys_exec", "sys_eval", "source_pos_wait",
    # SQLite
    "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer",
})

# Families of such functions (large objects, replication, WAL, advisory locks, file access, string-eval helpers).
_FORBIDDEN_FUNCTION_PREFIXES = (
    "pg_advisory", "pg_try_advisory", "pg_logical", "pg_replication", "pg_drop_replication", "pg_copy_",
    "pg_create_", "pg_wal_", "pg_backup", "pg_stat_reset", "pg_ls_", "pg_read_", "pg_file", "lo_", "dblink",
    "query_to_xml", "database_to_xml", "schema_to_xml", "table_to_xml", "cursor_to_xml", "sys_",
)
_FORBIDDEN_FUNCTIONS = _FORBIDDEN_FUNCTIONS | frozenset({
    "pg_notify", "pg_promote", "pg_stat_statements_reset", "pg_import_system_collations", "loread", "lowrite",
    "pg_logdir_ls", "ts_stat", "xpath_table", "crosstab", "crosstab2", "crosstab3", "crosstab4",
    "wait_for_executed_gtid_set", "wait_until_sql_thread_after_gtids", "pg_switch_xlog",
})

# System tables that expose credentials or other sessions' activity.
_FORBIDDEN_TABLES = frozenset({
    "pg_shadow", "pg_authid", "pg_user_mapping", "pg_user_mappings", "pg_largeobject", "pg_subscription",
    "pg_stat_activity", "pg_stat_statements",
})
_FORBIDDEN_SCHEMAS = frozenset({"mysql"})

_MONGO_FORBIDDEN_KEYS = frozenset({
    "$out", "$merge", "$function", "$accumulator", "$where", "$currentOp", "$listSessions",
    "$listLocalSessions", "$changeStream", "$planCacheStats", "$collStats", "$indexStats", "$queryStats",
    "$listCatalog", "$listClusterCatalog", "$listSampledQueries", "$shardedDataDistribution", "$listSearchIndexes",
})
MAX_PIPELINE_CHARS = 200_000


def function_name(node: exp.Expression) -> str:
    """
    Lower-case name of a function call, looking through quoting: sqlglot keeps a quoted name such as
    ``"set_config"(...)`` as an Identifier, which a plain ``isinstance(name, str)`` test would skip.
    """
    name: Any = node.this if isinstance(node, exp.Anonymous) else node.sql_name()
    if isinstance(name, exp.Expression):
        name = name.name
    return str(name or "").strip().lower()


def is_forbidden_function(name: str) -> bool:
    return name in _FORBIDDEN_FUNCTIONS or name.startswith(_FORBIDDEN_FUNCTION_PREFIXES)


def check_characters(sql: str) -> None:
    """
    Reject control and invisible characters. Databases treat some of them (e.g. U+FEFF) as whitespace while
    the parser treats them as part of a name, which would let a denylisted function name slip past; NUL can make
    a driver truncate the statement.
    """
    for ch in sql:
        if ch in "\t\n\r":
            continue
        if unicodedata.category(ch) in ("Cc", "Cf", "Cs", "Co"):
            raise GuardError(f"The query contains an invisible or control character (U+{ord(ch):04X}); remove it.")
# Operation names exactly as pyflashdb's Mongo adapter reports them in ``before_query`` (anything else is a write).
_MONGO_READ_OPS = frozenset({"find", "count_documents", "aggregate"})


# ── Trusted scope ─────────────────────────────────────────────────────────

_trusted = threading.local()


@contextmanager
def trusted_scope() -> Iterator[None]:
    """
    Statements issued inside this block are built by this package (catalog queries such as
    ``PRAGMA foreign_key_list``), never from model output, so the query hook lets them through.
    """
    _trusted.depth = getattr(_trusted, "depth", 0) + 1
    try:
        yield
    finally:
        _trusted.depth -= 1


def _in_trusted_scope() -> bool:
    return getattr(_trusted, "depth", 0) > 0


# ── Statement deadline (SQLite has no statement timeout of its own) ───────

_deadline = threading.local()


class Deadline:
    """
    Returned by :func:`statement_deadline`; ``expired`` tells whether the time limit was what stopped a query.
    The clock starts when the database first reports progress, i.e. when the statement really runs, so time
    spent waiting for the (single) pooled connection is not counted against it.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.started: Optional[float] = None

    def tick(self) -> bool:
        now = time.monotonic()
        if self.started is None:
            self.started = now
            return False
        return now - self.started > self.seconds

    @property
    def expired(self) -> bool:
        return self.started is not None and time.monotonic() - self.started > self.seconds


@contextmanager
def statement_deadline(seconds: float) -> Iterator[Deadline]:
    """Queries run on this thread inside the block are aborted after ``seconds`` (SQLite connections only)."""
    current = Deadline(seconds)
    _deadline.current = current
    try:
        yield current
    finally:
        _deadline.current = None


def _abort_if_late() -> int:
    current = getattr(_deadline, "current", None)
    return 1 if current is not None and current.tick() else 0       # non-zero aborts the running statement


# ── SQLite engine-level hardening ─────────────────────────────────────────
# The parser-based checks above run *before* SQLite sees the SQL. An authorizer runs *inside* the engine and sees
# what it actually resolved (the real function name, the real action), so quoting and whitespace tricks that fool a
# parser cannot matter to it. It is a second, independent layer, not a replacement.

MAX_SQLITE_VALUE_BYTES = 10_000_000          # longest string / blob a statement may build (bounds memory use)

# Action codes of SQLite's authorizer interface (sqlite3.h); stable across versions.
_CREATE_INDEX, _CREATE_TABLE, _DELETE, _DROP_INDEX, _DROP_TABLE, _INSERT = 1, 2, 9, 10, 11, 18
_PRAGMA, _READ, _SELECT, _TRANSACTION, _UPDATE, _ALTER_TABLE, _FUNCTION, _SAVEPOINT, _RECURSIVE = 19, 20, 21, 22, 23, 26, 31, 32, 33
_REINDEX = 27                                 # the engine checks this while executing CREATE INDEX
_SQLITE_ALWAYS = frozenset({_READ, _SELECT, _RECURSIVE, _TRANSACTION, _SAVEPOINT})
_SQLITE_WRITES = frozenset({
    _INSERT, _UPDATE, _DELETE, _CREATE_TABLE, _CREATE_INDEX, _DROP_TABLE, _DROP_INDEX, _ALTER_TABLE, _REINDEX,
})
_SQLITE_SAFE_PRAGMAS = frozenset({"table_info", "table_xinfo", "foreign_key_list", "index_list", "index_info", "index_xinfo"})
_AUTH_OK, _AUTH_DENY = 0, 1


def _sqlite_authorizer(writable: bool) -> Callable[..., int]:
    def authorize(action: int, arg1: Optional[str], arg2: Optional[str], dbname: Optional[str], source: Optional[str]) -> int:
        if action in _SQLITE_ALWAYS:
            return _AUTH_OK
        if action == _FUNCTION:
            name = (arg2 or "").lower()
            if name.startswith("sqlite_"):                    # the engine's own helpers (used by ALTER TABLE, ...)
                return _AUTH_OK
            return _AUTH_DENY if is_forbidden_function(name) else _AUTH_OK
        if action == _PRAGMA:
            # Only the read-only catalog look-ups pyflashdb and this package perform (their argument is a table or
            # index name). Every setting pragma (query_only, writable_schema, foreign_keys, ...) is not listed.
            return _AUTH_OK if (arg1 or "").lower() in _SQLITE_SAFE_PRAGMAS else _AUTH_DENY
        if writable and action in _SQLITE_WRITES:
            return _AUTH_OK
        return _AUTH_DENY                                     # ATTACH, VACUUM, triggers, views, other PRAGMAs, ...
    return authorize


def install_sqlite_hardening(flash: Any, writable: bool = False) -> bool:
    """
    On the (single) SQLite connection: abort statements past the active :func:`statement_deadline`, install the
    authorizer, and cap the size of any value a statement can build. Uses pyflashdb's pool to reach the raw
    connection; returns False if that is not possible, so callers can report the limitation instead of
    assuming protection.
    """
    try:
        pool = flash._adapter._pool
        conn = pool.acquire()
        try:
            conn.set_progress_handler(_abort_if_late, 10_000)
            conn.set_authorizer(_sqlite_authorizer(writable))
            if hasattr(conn, "setlimit"):                     # Python 3.11+
                import sqlite3

                conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_SQLITE_VALUE_BYTES)
        finally:
            pool.release(conn)
        return True
    except Exception:
        return False


# ── SQL ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PreparedSQL:
    sql: str                 # what will actually be executed
    columns: List[str]       # output column names as far as they can be told statically
    max_rows: int            # rows the caller may keep
    limit_injected: bool     # True if a LIMIT max_rows+1 was added/lowered by the guard
    # Output columns whose values derive from a redacted column, even through an alias or an expression.
    sensitive_positions: FrozenSet[int] = frozenset()    # by position (reliable unless the query uses '*')
    sensitive_names: FrozenSet[str] = frozenset()        # by (lower-cased) name
    projection_count: int = 0                            # how many output columns the query declares


def _check_tree(stmt: exp.Expression) -> exp.Expression:
    root = stmt
    while isinstance(root, exp.Subquery):
        extras = [key for key, value in root.args.items() if value and key not in ("this", "alias")]
        if extras:
            # Unwrapping would silently drop the outer ORDER BY / LIMIT / WITH and return wrong rows.
            raise GuardError(
                "A parenthesised query cannot have its own ORDER BY / LIMIT / WITH outside the parentheses: "
                "put them inside, or drop the outer parentheses."
            )
        root = root.this
    if not isinstance(root, (exp.Select, exp.SetOperation)):
        raise GuardError(
            f"Only SELECT queries are allowed (got {type(stmt).__name__.upper()}). This agent is read-only."
        )
    for node in stmt.walk():
        if _FORBIDDEN_NODES and isinstance(node, _FORBIDDEN_NODES):
            raise GuardError(f"{type(node).__name__.upper()} is not allowed: this agent is read-only.")
        if isinstance(node, exp.Func):
            name = function_name(node)
            if is_forbidden_function(name):
                raise GuardError(f"Function {name}() is not allowed.")
        if isinstance(node, exp.Table):
            if node.name.lower() in _FORBIDDEN_TABLES or node.db.lower() in _FORBIDDEN_SCHEMAS:
                raise GuardError(f"Access to system table {node.sql()} is not allowed.")
    return root


# ── which output columns carry redacted data ──────────────────────────────

def _column_refs(expression: exp.Expression) -> Set[str]:
    """Names of the columns an expression reads (COUNT(x) yields a number, not x, so it is not counted)."""
    return {
        node.name.lower()
        for node in expression.walk(prune=lambda n: isinstance(n, exp.Count))
        if isinstance(node, exp.Column)
    }


def _leaf_selects(node: exp.Expression) -> List[exp.Expression]:
    if isinstance(node, exp.Subquery):
        return _leaf_selects(node.this)
    if isinstance(node, exp.SetOperation):
        return _leaf_selects(node.this) + _leaf_selects(node.expression)
    return [node] if isinstance(node, exp.Select) else []


def _is_star(projection: exp.Expression) -> bool:
    return isinstance(projection, exp.Star) or (isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star))


def analyze_sensitivity(root: exp.Expression, is_redacted: Callable[[str], bool]) -> Tuple[FrozenSet[int], FrozenSet[str], int]:
    """
    Redaction by output name is defeated by ``SELECT email AS contact``. Instead, follow the data: an output is
    sensitive if its expression reads a redacted column, or a column that is itself an alias of such an expression
    (through CTEs and subqueries). Scope-insensitive on purpose: it can over-redact, never under-redact.
    """
    tainted: Set[str] = set()
    selects = list(root.find_all(exp.Select))
    changed = True
    while changed:
        changed = False
        for select in selects:
            for projection in select.expressions:
                if _is_star(projection):
                    continue
                out = (projection.alias_or_name or "").lower()
                if out and out not in tainted and any(is_redacted(r) or r in tainted for r in _column_refs(projection)):
                    tainted.add(out)
                    changed = True

    positions: Set[int] = set()
    star = False
    leaves = _leaf_selects(root)
    for leaf in leaves:
        for index, projection in enumerate(leaf.expressions):
            if _is_star(projection):
                star = True
            elif any(is_redacted(r) or r in tainted for r in _column_refs(projection)):
                positions.add(index)
    count = len(leaves[0].expressions) if leaves else 0
    return (frozenset() if star else frozenset(positions)), frozenset(tainted), count


def _parse_one(sql: str, dialect: str) -> exp.Expression:
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except SqlglotError as exc:
        raise GuardError(f"Could not parse the SQL ({dialect}): {str(exc).splitlines()[0]}") from exc
    if len(statements) != 1:
        raise GuardError("Exactly one SQL statement is allowed per query.")
    return statements[0]


def _literal_limit(limit: Any) -> Optional[int]:
    if isinstance(limit, exp.Limit) and isinstance(limit.expression, exp.Literal) and limit.expression.is_int:
        return int(limit.expression.this)
    return None


def _with_row_limit(root: exp.Expression, max_rows: int, dialect: str) -> tuple:
    """Return ``(tree, injected)`` where the tree returns at most ``max_rows + 1`` rows."""
    existing = root.args.get("limit")
    current = _literal_limit(existing)
    if current is not None and current <= max_rows:
        return root, False
    if existing is None or isinstance(existing, exp.Limit):
        return root.limit(max_rows + 1, copy=True), True
    # FETCH FIRST / TOP and friends: wrap rather than rewrite.
    wrapped = exp.select("*").from_(root.subquery("_verbadb_limited")).limit(max_rows + 1)
    return wrapped, True


def prepare_sql(
    sql: Any, backend: str, max_rows: int, is_redacted: Optional[Callable[[str], bool]] = None
) -> PreparedSQL:
    """
    Validate ``sql`` and return the read-only, row-limited statement to execute. If ``is_redacted`` is given, the
    result also says which output columns derive from redacted columns.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise GuardError("The query must be a non-empty SQL string.")
    if len(sql) > MAX_SQL_CHARS:
        raise GuardError(f"The query is too long (max {MAX_SQL_CHARS} characters).")
    check_characters(sql)
    dialect = _DIALECTS.get(backend)
    if dialect is None:
        raise GuardError(f"SQL queries are not supported on the {backend!r} backend.")

    stmt = _parse_one(sql, dialect)
    root = _check_tree(stmt)
    if hasattr(exp, "Hint"):                    # optimizer hints can override session limits (MySQL SET_VAR, ...)
        for hint in list(stmt.find_all(exp.Hint)):
            hint.pop()
    columns = [c for c in getattr(root, "named_selects", []) if c]
    positions, names, count = analyze_sensitivity(root, is_redacted) if is_redacted else (frozenset(), frozenset(), 0)
    limited, injected = _with_row_limit(root, max_rows, dialect)
    final_sql = limited.sql(dialect=dialect, comments=False)

    # Belt and braces: what we are about to run must itself pass the same checks.
    _check_tree(_parse_one(final_sql, dialect))
    return PreparedSQL(
        sql=final_sql, columns=columns, max_rows=max_rows, limit_injected=injected,
        sensitive_positions=positions, sensitive_names=names, projection_count=count,
    )


# ── MongoDB ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PreparedPipeline:
    collection: str
    pipeline: List[Dict[str, Any]]       # ready for flash.aggregate (extended JSON decoded)
    display: str                         # JSON shown to the user (limit included)
    max_rows: int


def _check_collection_ref(key: str, value: Any) -> None:
    if isinstance(value, dict):
        raise GuardError(f"{key} may only refer to a collection of this database.")
    if isinstance(value, str) and value.lower().startswith("system."):
        raise GuardError("System collections cannot be read.")


def _scan_mongo(value: Any, depth: int = 0) -> None:
    if depth > 30:
        raise GuardError("The pipeline is nested too deeply.")
    if isinstance(value, dict):
        for key, inner in value.items():
            if key in _MONGO_FORBIDDEN_KEYS:
                raise GuardError(f"{key} is not allowed: this agent is read-only and runs no server-side code.")
            if key in ("$lookup", "$graphLookup") and isinstance(inner, dict):
                _check_collection_ref(key, inner.get("from"))
            elif key == "$unionWith":
                _check_collection_ref(key, inner.get("coll") if isinstance(inner, dict) else inner)
            _scan_mongo(inner, depth + 1)
    elif isinstance(value, list):
        for inner in value:
            _scan_mongo(inner, depth + 1)


def _decode_extended_json(text: str) -> Any:
    try:
        from bson import json_util
    except ImportError:  # pymongo not installed: plain JSON only
        return json.loads(text)
    return json_util.loads(text)


def prepare_pipeline(
    collection: Any, pipeline: Any, max_rows: int, collections: Optional[Sequence[str]] = None
) -> PreparedPipeline:
    """Validate an aggregation pipeline (a list or a JSON string) and append a ``$limit``."""
    if not isinstance(collection, str) or not collection:
        raise GuardError("A collection name is required.")
    if collections is not None and collection not in collections:
        raise GuardError(f"Unknown collection {collection!r}. Known collections: {', '.join(sorted(collections))}")

    if isinstance(pipeline, str):
        if len(pipeline) > MAX_PIPELINE_CHARS:
            raise GuardError(f"The pipeline is too long (max {MAX_PIPELINE_CHARS} characters).")
        try:
            plain = json.loads(pipeline)
        except json.JSONDecodeError as exc:
            raise GuardError(f"pipeline_json is not valid JSON: {exc}") from exc
        except (RecursionError, OverflowError) as exc:
            raise GuardError("The pipeline is nested too deeply.") from exc
    else:
        plain = pipeline
        pipeline = json.dumps(plain, default=str)
    if not isinstance(plain, list) or not plain:
        raise GuardError("The pipeline must be a non-empty JSON array of stages.")
    if len(plain) > MAX_PIPELINE_STAGES:
        raise GuardError(f"The pipeline has too many stages (max {MAX_PIPELINE_STAGES}).")
    for index, stage in enumerate(plain):
        if not isinstance(stage, dict) or len(stage) != 1 or not next(iter(stage)).startswith("$"):
            raise GuardError(f"Stage {index} must be an object with exactly one $-operator key.")
    _scan_mongo(plain)

    last = plain[-1].get("$limit")
    shown = list(plain)
    if not (isinstance(last, int) and not isinstance(last, bool) and 0 < last <= max_rows):
        shown.append({"$limit": max_rows + 1})
    try:
        decoded = _decode_extended_json(json.dumps(shown))
    except Exception as exc:                       # bson raises its own error types (e.g. an invalid $oid)
        raise GuardError(f"Could not decode the pipeline: {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}") from exc
    return PreparedPipeline(collection, decoded, json.dumps(shown, indent=2), max_rows)


# ── Query hook ────────────────────────────────────────────────────────────

def check_statement(info: Dict[str, Any]) -> None:
    """Raise :class:`GuardError` unless the statement described by a ``before_query`` payload is read-only."""
    if _in_trusted_scope():
        return
    backend = info.get("backend")
    statement = info.get("statement")
    if backend == "mongodb":
        op = str(statement).rsplit(".", 1)[-1]
        table = info.get("table")
        params = info.get("params") or {}
        if op == "aggregate":
            _scan_mongo(params.get("pipeline"))
        elif op in ("find", "count_documents"):
            _scan_mongo(params.get("filter"))
        elif op == "raw":
            command = params.get("command")
            if table in (None, "<database>") or not isinstance(command, dict):
                raise GuardError("Database commands are not allowed: this agent is read-only.")
            _scan_mongo(command)
        elif op not in _MONGO_READ_OPS:
            raise GuardError(f"MongoDB operation {op!r} is not allowed: this agent is read-only.")
        return
    # pyflashdb's own generated SQL uses %s placeholders; the parser knows ? ones.
    text = str(statement).replace("%s", "?")
    check_characters(text)
    dialect = _DIALECTS.get(str(backend))
    if dialect is None:
        raise GuardError(f"Unsupported backend {backend!r}.")
    _check_tree(_parse_one(text, dialect))


def install_hook(flash: Any) -> None:
    """Vet every statement sent to the database through ``flash`` (raise to abort it)."""
    flash.before_query(check_statement)
