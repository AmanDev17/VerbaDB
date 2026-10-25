"""Run guarded queries through pyflashdb and shape the results for the UI and for the model."""

from __future__ import annotations

import datetime as dt
import decimal
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

from . import guard
from .config import Settings

MODEL_VIEW_MAX_CHARS = 8_000     # keeps free-tier token usage predictable
CELL_MAX_CHARS = 200             # per cell, in what the model sees
UI_CELL_MAX_CHARS = 20_000       # per cell, in what the browser receives (a query can build huge strings)


class QueryFailed(Exception):
    """The database rejected or failed the (already validated) query. The message is safe to show the model."""


@dataclass
class QueryResult:
    columns: List[str]
    rows: List[Dict[str, Any]]          # JSON-safe; at most max_rows
    truncated: bool                     # more rows existed than max_rows
    query: str                          # exactly what was executed (SQL text or pipeline JSON)
    elapsed_ms: float
    language: str                       # "sql" | "mongodb"
    collection: str = ""
    sensitive: frozenset = frozenset()  # output columns that derive from redacted columns (also via aliases)
    row_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.row_count = len(self.rows)

    def to_event(self) -> Dict[str, Any]:
        return {
            "type": "result", "columns": self.columns, "rows": self.rows, "row_count": self.row_count,
            "truncated": self.truncated, "query": self.query, "language": self.language,
            "collection": self.collection, "elapsed_ms": round(self.elapsed_ms, 1),
        }


# ── JSON safety ───────────────────────────────────────────────────────────

def jsonable(value: Any) -> Any:
    """Convert driver values (Decimal, datetime, bytes, ObjectId, ...) into plain JSON types."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, decimal.Decimal):
        if not value.is_finite():
            return str(value)
        if value == value.to_integral_value() and abs(value) < 10 ** 15:
            return int(value)
        return float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    return str(value)


def _ui_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe row with oversized text cut short, so one huge value cannot flood the browser."""
    shaped = jsonable(dict(row))
    for key, value in shaped.items():
        if isinstance(value, str) and len(value) > UI_CELL_MAX_CHARS:
            shaped[key] = f"{value[:UI_CELL_MAX_CHARS]}… [cut: {len(value):,} characters]"
    return shaped


def _clean_error(exc: Exception) -> str:
    text = str(exc)
    # pyflashdb appends "\nSQL: ...\nParams: ..." — the model already knows its own SQL.
    return text.split("\nSQL:", 1)[0].strip() or exc.__class__.__name__


def _columns_from(rows: List[Dict[str, Any]], fallback: List[str]) -> List[str]:
    seen: List[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen or list(fallback)


# ── Execution ─────────────────────────────────────────────────────────────

def run_sql(flash: Any, sql: Any, settings: Settings) -> QueryResult:
    """Validate, limit and execute a SELECT on a SQL backend."""
    prepared = guard.prepare_sql(sql, flash.backend, settings.max_rows, settings.is_redacted)
    started = time.perf_counter()
    with guard.statement_deadline(settings.query_timeout) as deadline:
        try:
            rows = flash.raw(prepared.sql)
        except guard.GuardError:
            raise
        except Exception as exc:  # driver / pyflashdb errors
            if deadline.expired:
                raise QueryFailed(
                    f"The query was stopped after the {settings.query_timeout}s time limit. "
                    "Make it cheaper: filter earlier, join fewer tables, or aggregate before joining."
                ) from exc
            raise QueryFailed(_clean_error(exc)) from exc
    elapsed = (time.perf_counter() - started) * 1000
    if not isinstance(rows, list):
        raise guard.GuardError("The statement did not return rows; only SELECT queries are allowed.")
    truncated = len(rows) > prepared.max_rows
    kept = [_ui_row(r) for r in rows[: prepared.max_rows]]
    columns = _columns_from(kept, prepared.columns)
    sensitive = {c for c in columns if c.lower() in prepared.sensitive_names}
    if prepared.projection_count == len(columns):             # positions are only meaningful if nothing collapsed
        sensitive |= {columns[i] for i in prepared.sensitive_positions if i < len(columns)}
    return QueryResult(
        columns=columns, rows=kept, truncated=truncated, query=prepared.sql, elapsed_ms=elapsed,
        language="sql", sensitive=frozenset(sensitive),
    )


def run_aggregation(flash: Any, collection: Any, pipeline_json: Any, settings: Settings) -> QueryResult:
    """Validate, limit and execute an aggregation pipeline on MongoDB."""
    with guard.trusted_scope():
        known = flash.show_tables()
    prepared = guard.prepare_pipeline(collection, pipeline_json, settings.max_rows, known)
    started = time.perf_counter()
    try:
        rows = flash.aggregate(prepared.collection, prepared.pipeline)
    except guard.GuardError:
        raise
    except Exception as exc:
        raise QueryFailed(_clean_error(exc)) from exc
    elapsed = (time.perf_counter() - started) * 1000
    truncated = len(rows) > prepared.max_rows
    kept = [_ui_row(r) for r in rows[: prepared.max_rows]]
    return QueryResult(
        columns=_columns_from(kept, []), rows=kept, truncated=truncated, query=prepared.display,
        elapsed_ms=elapsed, language="mongodb", collection=prepared.collection,
    )


# ── What the model gets to see ────────────────────────────────────────────

def mask(value: Any, settings: Settings) -> Any:
    """Recursively mask values under redacted keys and clip long strings."""
    if isinstance(value, dict):
        return {
            k: ("[redacted]" if settings.is_redacted(k) else mask(v, settings)) for k, v in value.items()
        }
    if isinstance(value, list):
        return [mask(v, settings) for v in value[:20]]
    if isinstance(value, str) and len(value) > CELL_MAX_CHARS:
        return value[:CELL_MAX_CHARS] + "…"
    return value


def model_view(result: QueryResult, settings: Settings) -> Dict[str, Any]:
    """
    A redacted, size-capped summary of a result for the model. The user always sees the full result in the
    UI; this is only what is sent to the (free-tier) model API.
    """
    redacted = [c for c in result.columns if settings.is_redacted(c) or c in result.sensitive]
    shown = max(0, min(settings.preview_rows, len(result.rows)))
    while True:
        rows = [
            mask({c: ("[redacted]" if c in redacted else r.get(c)) for c in result.columns}, settings)
            for r in result.rows[:shown]
        ]
        if shown == 0 or len(json.dumps(rows, default=str)) <= MODEL_VIEW_MAX_CHARS:
            break
        shown = max(0, shown // 2)
    view: Dict[str, Any] = {
        "columns": result.columns, "row_count": result.row_count, "truncated_at_limit": result.truncated,
        "rows_shown": len(rows), "rows": rows,
    }
    if redacted:
        view["redacted_columns"] = redacted
    if settings.preview_rows == 0:
        view["note"] = "Row values are withheld from you by configuration; only column names and counts are available."
    elif len(rows) < result.row_count:
        view["note"] = (
            f"Showing the first {len(rows)} of {result.row_count} returned rows. "
            "The user sees every returned row in a table next to your answer."
        )
    return view
