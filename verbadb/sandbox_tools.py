"""
The toolset used when the active database is a sandbox (or when none is active yet).

It is the read-only toolset (list / describe / sample / run_sql, unchanged and still SELECT-only) plus two
extra tools: ``create_database`` and ``modify_database``. Changes made by ``modify_database`` are validated,
snapshotted and diffed by ``sandbox.py``; the diff goes to the UI as a ``change`` event, while the model only
gets counts back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import executor, guard
from .config import Settings
from .manager import DatabaseManager, DbEntry, ManagerError
from .sandbox import ChangeFailed, apply_changes
from .tools import ToolBox, ToolOutcome, ToolSpec, _need_str, tool_specs

log = logging.getLogger("verbadb.sandbox_tools")


@dataclass
class SandboxOutcome(ToolOutcome):
    events: List[Dict[str, Any]] = field(default_factory=list)       # extra UI events (change / database)


_WRITE_TOOLS = ("create_database", "modify_database")


def sandbox_tool_specs(settings: Settings) -> List[ToolSpec]:
    return tool_specs("sqlite", settings) + [
        ToolSpec(
            "create_database",
            "Create a new, empty sandbox database and make it the active one. Call this first when no database is "
            "active, or when the user asks for a new / separate database.",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short descriptive name, e.g. \"Bookstore\"."},
                    "description": {"type": "string", "description": "One sentence about what it is for."},
                },
                "required": ["name"],
            },
        ),
        ToolSpec(
            "modify_database",
            "Change the ACTIVE sandbox database: create / alter / drop tables and indexes, insert / update / delete rows. "
            "Pass every statement of one logical change, in order, in a single call: they run atomically (all or "
            "nothing). SQLite syntax, one statement per list item. The user sees a before/after view of the change "
            "and can undo it.",
            {
                "type": "object",
                "properties": {
                    "statements": {
                        "type": "array", "items": {"type": "string"},
                        "description": "SQL statements in execution order, e.g. CREATE TABLE ... then INSERT INTO ...",
                    }
                },
                "required": ["statements"],
            },
        ),
    ]


class SandboxToolBox(ToolBox):
    """Read tools on the active database + create_database / modify_database. ``entry`` may be ``None`` at first."""

    def __init__(self, manager: DatabaseManager, workspace: str, entry: Optional[DbEntry], settings: Settings) -> None:
        self.manager = manager
        self.workspace = workspace
        self.settings = settings
        self.entry = entry
        self.flash = entry.flash if entry else None
        self.backend = "sqlite"

    def specs(self) -> List[ToolSpec]:
        return sandbox_tool_specs(self.settings)

    def _refresh(self) -> None:
        """Re-resolve the active database, because an undo / reset closes and reopens it between calls."""
        if self.entry is not None:
            self.entry = self.manager.resolve(self.workspace, self.entry.id)
            self.flash = self.entry.flash

    def call(self, name: str, args: Optional[Dict[str, Any]]) -> ToolOutcome:
        args = dict(args or {})
        try:
            self._refresh()
            if name == "create_database":
                return self._create(args)
            if name == "modify_database":
                return self._modify(args)
        except KeyError:
            self.entry = self.flash = None
            return self._error("That database no longer exists. Use create_database to make a new one.")
        except guard.GuardError as exc:
            return self._error(f"Rejected by the sandbox safety guard: {exc}")
        except (ChangeFailed, ManagerError) as exc:
            return self._error(str(exc))
        except ValueError as exc:
            return self._error(f"Invalid arguments: {exc}")
        except Exception as exc:
            log.exception("tool %s failed", name)
            return self._error(f"Tool failed: {executor._clean_error(exc)}")

        if self.flash is None and name in {s.name for s in self.specs()}:
            return self._error("No database is active yet. Call create_database first.")
        return super().call(name, args)

    def _create(self, args: Dict[str, Any]) -> ToolOutcome:
        name = _need_str(args, "name")
        description = args.get("description")
        info = self.manager.create_sandbox(
            self.workspace, name, description if isinstance(description, str) else ""
        )
        self.entry = self.manager.resolve(self.workspace, info["id"])
        self.flash = self.entry.flash
        return SandboxOutcome(
            response={"created": True, "name": info["name"],
                      "note": "This database is now active and empty. Create tables with modify_database."},
            ok=True, summary=f"Created database \"{info['name']}\"",
            events=[{"type": "database", "database": info, "created": True}],
        )

    def _modify(self, args: Dict[str, Any]) -> ToolOutcome:
        if self.entry is None or self.entry.snapshots is None:
            return self._error("No database is active yet. Call create_database first.")
        statements = args.get("statements")
        if isinstance(statements, str):
            statements = [statements]
        with self.entry.lock:
            change = apply_changes(self.entry.flash, statements, self.settings, self.entry.snapshots)
        change.database_id = self.entry.id
        return SandboxOutcome(change.model_view(), ok=True, summary=change.summary, events=[change.to_event()])
