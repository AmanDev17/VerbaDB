"""
Schema inspection on top of pyflashdb.

pyflashdb's ``describe()`` has no foreign keys, so they are read from each backend's catalog (declared) and,
where a column follows a naming convention (``customer_id`` -> ``customers``), inferred. Inferred keys are
labelled as such so the model — and the user — can treat them as assumptions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from . import guard

log = logging.getLogger("verbadb.introspect")


@dataclass(frozen=True)
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str
    declared: bool      # False = inferred from naming conventions

    def as_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column, "references": f"{self.ref_table}.{self.ref_column}",
            "source": "declared" if self.declared else "inferred from column name (unverified)",
        }


def _text(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return "" if value is None else str(value)


# ── Tables ────────────────────────────────────────────────────────────────

def known_tables(flash: Any) -> List[str]:
    with guard.trusted_scope():
        return list(flash.show_tables())


def list_tables(flash: Any, count_limit: int = 60) -> List[Dict[str, Any]]:
    """Table / collection names with row counts (counted only for the first ``count_limit``)."""
    out: List[Dict[str, Any]] = []
    for index, name in enumerate(known_tables(flash)):
        rows: Optional[int] = None
        if index < count_limit:
            try:
                rows = int(flash.count(name))
            except Exception as exc:
                log.info("count(%s) failed: %s", name, exc)
        out.append({"name": name, "rows": rows})
    return out


# ── Foreign keys ──────────────────────────────────────────────────────────

def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _declared_fks(flash: Any, table: str) -> List[ForeignKey]:
    backend = flash.backend
    try:
        with guard.trusted_scope():
            if backend == "sqlite":
                rows = flash.raw(f"PRAGMA foreign_key_list({_quote_ident(table)})")
                return [
                    ForeignKey(_text(r["from"]), _text(r["table"]), _text(r["to"]) or "id", True)
                    for r in rows
                ]
            if backend == "mysql":
                rows = flash.raw(
                    "SELECT COLUMN_NAME AS col, REFERENCED_TABLE_NAME AS ref_table, "
                    "REFERENCED_COLUMN_NAME AS ref_col FROM information_schema.KEY_COLUMN_USAGE "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND REFERENCED_TABLE_NAME IS NOT NULL",
                    [table],
                )
            elif backend == "postgres":
                rows = flash.raw(
                    "SELECT kcu.column_name AS col, ccu.table_name AS ref_table, ccu.column_name AS ref_col "
                    "FROM information_schema.table_constraints tc "
                    "JOIN information_schema.key_column_usage kcu "
                    "ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema "
                    "JOIN information_schema.constraint_column_usage ccu "
                    "ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema "
                    "WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s "
                    "AND tc.table_schema = current_schema()",
                    [table],
                )
            else:
                return []
        return [ForeignKey(_text(r["col"]), _text(r["ref_table"]), _text(r["ref_col"]), True) for r in rows]
    except Exception as exc:  # no catalog access is not fatal: fall back to inference only
        log.info("could not read declared foreign keys of %s: %s", table, exc)
        return []


def _candidate_names(base: str) -> Set[str]:
    names = {base, base + "s", base + "es"}
    if base.endswith("y"):
        names.add(base[:-1] + "ies")
    return names


def _ref_column(flash: Any, ref_table: str, cache: Dict[str, str]) -> str:
    """
    The column an inferred key most likely points at: the referenced table's primary key on SQL backends; on
    MongoDB the collection's own ``id`` field if it has one (typical for imported data), otherwise ``_id``.
    """
    if ref_table not in cache:
        fallback = "_id" if flash.backend == "mongodb" else "id"
        try:
            with guard.trusted_scope():
                columns = flash.describe(ref_table)
            names = [c["name"] for c in columns]
            if flash.backend == "mongodb":
                cache[ref_table] = "id" if "id" in names else "_id"
            else:
                cache[ref_table] = next((c["name"] for c in columns if c.get("primary_key")), fallback)
        except Exception as exc:
            log.info("could not describe %s to find its key: %s", ref_table, exc)
            cache[ref_table] = fallback
    return cache[ref_table]


def _infer_fks(
    columns: Sequence[Dict[str, Any]], table: str, tables: Sequence[str], skip: Set[str], backend: str,
    ref_column_of: Optional[Any] = None,
) -> List[ForeignKey]:
    by_lower = {t.lower(): t for t in tables}
    default_ref = "_id" if backend == "mongodb" else "id"
    found: List[ForeignKey] = []
    for col in columns:
        name = str(col["name"])
        lowered = name.lower()
        if name in skip or col.get("primary_key") or lowered in ("id", "_id"):
            continue
        if lowered.endswith("_id"):
            base = lowered[:-3]
        elif re.search(r"[a-z]Id$", name):
            base = lowered[:-2]
        else:
            continue
        parts = [p for p in base.split("_") if p]
        # billing_customer_id -> try "billing_customer", then "customer"
        for start in range(len(parts)):
            match = next(
                (by_lower[c] for c in _candidate_names("_".join(parts[start:])) if c in by_lower), None
            )
            if match:
                found.append(ForeignKey(name, match, ref_column_of(match) if ref_column_of else default_ref, False))
                break
    return found


def foreign_keys(
    flash: Any, table: str, columns: Sequence[Dict[str, Any]], tables: Sequence[str],
    cache: Optional[Dict[str, str]] = None,
) -> List[ForeignKey]:
    declared = _declared_fks(flash, table)
    skip = {fk.column for fk in declared}
    ref_cache = cache if cache is not None else {}
    inferred = _infer_fks(
        columns, table, tables, skip, flash.backend, lambda ref: _ref_column(flash, ref, ref_cache)
    )
    return declared + inferred


# ── Describe ──────────────────────────────────────────────────────────────

def describe_tables(flash: Any, tables: Sequence[str]) -> Dict[str, Any]:
    """Columns, primary key and foreign keys for each requested table (unknown names get an error entry)."""
    known = known_tables(flash)
    lookup = {t.lower(): t for t in known}
    result: Dict[str, Any] = {}
    ref_cache: Dict[str, str] = {}
    for requested in tables:
        name = requested if requested in known else lookup.get(str(requested).lower())
        if name is None:
            result[str(requested)] = {"error": f"Unknown table. Known tables: {', '.join(known)}"}
            continue
        with guard.trusted_scope():
            columns = flash.describe(name)
        result[name] = {
            "columns": [
                {"name": c["name"], "type": c["type"], "nullable": c["nullable"],
                 **({"primary_key": True} if c.get("primary_key") else {})}
                for c in columns
            ],
            "foreign_keys": [fk.as_dict() for fk in foreign_keys(flash, name, columns, known, ref_cache)],
        }
        if flash.backend == "mongodb":
            result[name]["note"] = (
                "Fields are inferred from a small sample of documents; other fields may exist. "
                "Use sample_rows to see a real document, including nested fields."
            )
    return result


def schema_overview(flash: Any) -> List[Dict[str, Any]]:
    """Compact table -> columns listing for the UI sidebar (no row counts, no foreign-key lookups)."""
    out: List[Dict[str, Any]] = []
    for name in known_tables(flash):
        try:
            with guard.trusted_scope():
                columns = flash.describe(name)
        except Exception as exc:
            out.append({"name": name, "columns": [], "error": str(exc)})
            continue
        out.append({
            "name": name,
            "columns": [
                {"name": c["name"], "type": c["type"], "primary_key": bool(c.get("primary_key"))} for c in columns
            ],
        })
    return out
