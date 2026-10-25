"""
The app's own persistent store (which databases each workspace has, and its saved prompts), kept in a small
SQLite file and — fittingly — accessed through pyflashdb itself.

A *workspace* is an anonymous per-browser identity (see ``server.py``); every query here is scoped to one, so
people sharing the same running app never see each other's databases or prompt history.
"""

from __future__ import annotations

import datetime as dt
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyflashdb import FlashDB


def _escape_like(text: str) -> str:
    """pyflashdb ``like``: ``%`` and ``_`` are wildcards, backslash escapes them."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0)


class AppStore:
    def __init__(self, data_dir: Any, max_saved_prompts: int = 500) -> None:
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_saved_prompts = max_saved_prompts
        self._lock = threading.Lock()
        self.flash = FlashDB.from_url(f"sqlite:///{(self.dir / 'app.db').as_posix()}")
        self._init_schema()

    def close(self) -> None:
        self.flash.close()

    def _init_schema(self) -> None:
        f = self.flash
        if not f.table_exists("databases"):
            f.create_table("databases", {
                "id": "int",
                "public_id": {"type": "str", "length": 32, "unique": True, "nullable": False},
                "workspace": {"type": "str", "length": 64, "nullable": False},
                "name": {"type": "str", "length": 100, "nullable": False},
                "kind": {"type": "str", "length": 16, "nullable": False},      # "sandbox" | "connection"
                "url": {"type": "text", "nullable": True},                     # connections only; never sent to the browser
                "description": {"type": "text", "nullable": True},
                "created_at": "datetime",
            })
            f.create_index("databases", ["workspace"])
        if not f.table_exists("prompts"):
            f.create_table("prompts", {
                "id": "int",
                "workspace": {"type": "str", "length": 64, "nullable": False},
                "database_id": {"type": "str", "length": 64, "nullable": True},
                "database_name": {"type": "str", "length": 100, "nullable": True},
                "text": {"type": "text", "nullable": False},
                "status": {"type": "str", "length": 16, "nullable": False},    # ok | error | blocked | stopped
                "summary": {"type": "text", "nullable": True},
                "created_at": "datetime",
            })
            f.create_index("prompts", ["workspace"])

    # ── databases ─────────────────────────────────────────────────────────

    def add_database(
        self, workspace: str, name: str, kind: str, *, url: Optional[str] = None, description: str = ""
    ) -> Dict[str, Any]:
        row = {
            "public_id": uuid.uuid4().hex, "workspace": workspace, "name": name, "kind": kind,
            "url": url, "description": description or None, "created_at": _now(),
        }
        with self._lock:
            self.flash.add("databases", row)
        return row

    def list_databases(self, workspace: str) -> List[Dict[str, Any]]:
        return self.flash.select("databases", filters={"workspace": workspace}, order_by="id")

    def get_database(self, workspace: str, public_id: str) -> Optional[Dict[str, Any]]:
        return self.flash.find_one("databases", {"workspace": workspace, "public_id": public_id})

    def delete_database(self, workspace: str, public_id: str) -> bool:
        with self._lock:
            return self.flash.delete("databases", {"workspace": workspace, "public_id": public_id}) > 0

    # ── saved prompts ─────────────────────────────────────────────────────

    def add_prompt(
        self, workspace: str, text: str, status: str, summary: str = "",
        database_id: Optional[str] = None, database_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        row = {
            "workspace": workspace, "database_id": database_id, "database_name": database_name,
            "text": text[:2000], "status": status, "summary": (summary or "")[:400] or None, "created_at": _now(),
        }
        with self._lock:
            row["id"] = self.flash.add("prompts", row)
            excess = self.flash.count("prompts", {"workspace": workspace}) - self.max_saved_prompts
            if excess > 0:
                old = self.flash.select(
                    "prompts", fields=["id"], filters={"workspace": workspace}, order_by="id", limit=excess
                )
                self.flash.delete("prompts", {"id": {"in": [r["id"] for r in old]}})
        return row

    def list_prompts(
        self, workspace: str, database_id: Optional[str] = None, q: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"workspace": workspace}
        if database_id:
            filters["database_id"] = database_id
        if q and q.strip():
            filters["text"] = {"ilike": f"%{_escape_like(q.strip())}%"}
        return self.flash.select("prompts", filters=filters, order_by="-id", limit=max(1, min(limit, 500)))

    def delete_prompt(self, workspace: str, prompt_id: int) -> bool:
        with self._lock:
            return self.flash.delete("prompts", {"workspace": workspace, "id": prompt_id}) > 0

    def clear_prompts(self, workspace: str, database_id: Optional[str] = None) -> int:
        filters: Dict[str, Any] = {"workspace": workspace}
        if database_id:
            filters["database_id"] = database_id
        with self._lock:
            return self.flash.delete("prompts", filters)
