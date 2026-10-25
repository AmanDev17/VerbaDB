"""
The tools the model can call. Specs are plain JSON Schema (provider-neutral; ``llm.py`` converts them);
``ToolBox.call`` validates arguments itself because Gemini has no strict-schema mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import executor, guard, introspect
from .config import Settings
from .executor import QueryFailed, QueryResult

log = logging.getLogger("verbadb.tools")

MAX_SAMPLE_ROWS = 5
MAX_TABLES_PER_DESCRIBE = 15


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]      # JSON Schema


@dataclass
class ToolOutcome:
    response: Dict[str, Any]        # returned to the model
    ok: bool
    summary: str                    # one line for the UI's step list
    result: Optional[QueryResult] = None


_NO_ARGS = {"type": "object", "properties": {}}


def tool_specs(backend: str, settings: Settings) -> List[ToolSpec]:
    specs = [
        ToolSpec(
            "list_tables",
            "List every table (or collection) in the database with its approximate row count. "
            "Call this first to see what data exists.",
            _NO_ARGS,
        ),
        ToolSpec(
            "describe_tables",
            "Get columns, types, primary key and foreign keys for one or more tables. "
            "Pass ALL the tables you might need in a single call. Foreign keys marked 'inferred' are guesses "
            "from column names and should be treated as assumptions.",
            {
                "type": "object",
                "properties": {
                    "tables": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Table (or collection) names, e.g. [\"customers\", \"orders\"].",
                    }
                },
                "required": ["tables"],
            },
        ),
    ]
    if settings.preview_rows > 0:
        specs.append(ToolSpec(
            "sample_rows",
            f"Fetch up to {MAX_SAMPLE_ROWS} example rows from a table to see what the data looks like "
            "(formats, nesting, typical values).",
            {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "limit": {"type": "integer", "description": f"1 to {MAX_SAMPLE_ROWS}, default 3."},
                },
                "required": ["table"],
            },
        ))
    if backend == "mongodb":
        specs.append(ToolSpec(
            "run_aggregation",
            "Run a read-only MongoDB aggregation pipeline on one collection and return the rows. "
            "Results are capped; write $sort and $limit yourself for top-N questions. "
            "Join other collections with $lookup.",
            {
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "pipeline_json": {
                        "type": "string",
                        "description": "The pipeline as a JSON array string, e.g. "
                                       "'[{\"$match\": {...}}, {\"$group\": {...}}]'. Use {\"$date\": \"YYYY-MM-DDT00:00:00Z\"} "
                                       "for date literals.",
                    },
                },
                "required": ["collection", "pipeline_json"],
            },
        ))
    else:
        specs.append(ToolSpec(
            "run_sql",
            "Run ONE read-only SELECT statement (CTEs and joins are fine) and return the rows. "
            "Anything that is not a SELECT is rejected. Results are capped; add ORDER BY and LIMIT yourself "
            "for top-N questions.",
            {
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
                "required": ["sql"],
            },
        ))
    return specs


def _need_str(args: Dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' is required and must be a non-empty string")
    return value.strip()


class ToolBox:
    """Executes tool calls against one pyflashdb connection."""

    def __init__(self, flash: Any, settings: Settings) -> None:
        self.flash = flash
        self.settings = settings
        self.backend: str = flash.backend

    def specs(self) -> List[ToolSpec]:
        return tool_specs(self.backend, self.settings)

    # ── dispatch ──────────────────────────────────────────────────────────

    def call(self, name: str, args: Optional[Dict[str, Any]]) -> ToolOutcome:
        args = dict(args or {})
        handler = {
            "list_tables": self._list_tables,
            "describe_tables": self._describe_tables,
            "sample_rows": self._sample_rows,
            "run_sql": self._run_sql,
            "run_aggregation": self._run_aggregation,
        }.get(name)
        allowed = {s.name for s in self.specs()}
        if handler is None or name not in allowed:
            return self._error(f"Unknown tool {name!r}. Available tools: {', '.join(sorted(allowed))}")
        try:
            return handler(args)
        except guard.GuardError as exc:
            return self._error(f"Rejected by the read-only safety guard: {exc}")
        except QueryFailed as exc:
            return self._error(f"The database returned an error: {exc}")
        except ValueError as exc:
            return self._error(f"Invalid arguments: {exc}")
        except Exception as exc:
            log.exception("tool %s failed", name)
            return self._error(f"Tool failed: {executor._clean_error(exc)}")

    @staticmethod
    def _error(message: str) -> ToolOutcome:
        return ToolOutcome({"error": message}, ok=False, summary=message)

    # ── handlers ──────────────────────────────────────────────────────────

    def _list_tables(self, args: Dict[str, Any]) -> ToolOutcome:
        tables = introspect.list_tables(self.flash)
        return ToolOutcome({"tables": tables}, ok=True, summary=f"Found {len(tables)} tables")

    def _describe_tables(self, args: Dict[str, Any]) -> ToolOutcome:
        tables = args.get("tables")
        if isinstance(tables, str):
            tables = [tables]
        if not isinstance(tables, list) or not tables or not all(isinstance(t, str) for t in tables):
            raise ValueError("'tables' must be a non-empty array of table names")
        tables = tables[:MAX_TABLES_PER_DESCRIBE]
        described = introspect.describe_tables(self.flash, tables)
        ok = any("error" not in v for v in described.values())
        return ToolOutcome({"tables": described}, ok=ok, summary=f"Described {', '.join(described)}")

    def _sample_rows(self, args: Dict[str, Any]) -> ToolOutcome:
        table = _need_str(args, "table")
        try:
            limit = int(args.get("limit") or 3)
        except (TypeError, ValueError) as exc:
            raise ValueError("'limit' must be an integer") from exc
        limit = max(1, min(limit, MAX_SAMPLE_ROWS, self.settings.preview_rows))
        known = introspect.known_tables(self.flash)
        lookup = {t.lower(): t for t in known}
        name = table if table in known else lookup.get(table.lower())
        if name is None:
            raise ValueError(f"Unknown table {table!r}. Known tables: {', '.join(known)}")
        rows = self.flash.select(name, limit=limit)
        shaped = [executor.mask(executor.jsonable(dict(r)), self.settings) for r in rows]
        return ToolOutcome({"table": name, "rows": shaped}, ok=True, summary=f"Sampled {len(shaped)} rows from {name}")

    def _run_sql(self, args: Dict[str, Any]) -> ToolOutcome:
        result = executor.run_sql(self.flash, _need_str(args, "sql"), self.settings)
        return self._query_outcome(result)

    def _run_aggregation(self, args: Dict[str, Any]) -> ToolOutcome:
        pipeline = args.get("pipeline_json")
        if not isinstance(pipeline, str) or not pipeline.strip():
            raise ValueError("'pipeline_json' is required and must be a JSON string")
        result = executor.run_aggregation(self.flash, _need_str(args, "collection"), pipeline, self.settings)
        return self._query_outcome(result)

    def _query_outcome(self, result: QueryResult) -> ToolOutcome:
        suffix = " (hit the row limit)" if result.truncated else ""
        return ToolOutcome(
            executor.model_view(result, self.settings), ok=True,
            summary=f"{result.row_count} rows in {result.elapsed_ms:.0f} ms{suffix}", result=result,
        )
