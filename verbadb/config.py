"""Settings, read from environment variables (and a local ``.env`` file)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_REDACT = ("email", "phone", "ssn", "password", "token", "secret")


class ConfigError(ValueError):
    """Raised for an unusable configuration value."""


@dataclass(frozen=True)
class Settings:
    api_key: Optional[str] = None
    model: str = DEFAULT_MODEL
    databases: Dict[str, str] = field(default_factory=dict)
    max_steps: int = 8
    max_rows: int = 500
    preview_rows: int = 20
    query_timeout: int = 15
    redact_columns: Tuple[str, ...] = DEFAULT_REDACT
    llm_retries: int = 3
    max_retry_wait: float = 60.0
    max_history_items: int = 60
    host: str = "127.0.0.1"
    port: int = 8000
    # Workspaces, sandboxes and saved prompts
    data_dir: str = "verbadb_data"
    allow_user_connections: bool = True      # may people add their own (read-only) databases from the UI?
    allow_private_db_hosts: bool = True      # ...pointing at localhost / private networks?
    max_databases: int = 8                   # sandboxes + own connections per workspace
    sandbox_max_mb: int = 20                 # size cap of one sandbox database
    max_saved_prompts: int = 500             # prompt history kept per workspace

    def is_redacted(self, column: str) -> bool:
        """True if values of ``column`` must be masked before they are sent to the model."""
        lowered = str(column).lower()
        return any(pattern and pattern in lowered for pattern in self.redact_columns)


def _int(env: Mapping[str, str], name: str, default: int, *, minimum: int = 0) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    lowered = raw.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")


def _databases(raw: Optional[str]) -> Dict[str, str]:
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"VERBADB_DATABASES must be a JSON object of name -> URL: {exc}") from exc
    if not isinstance(parsed, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()):
        raise ConfigError("VERBADB_DATABASES must be a JSON object of name -> URL strings")
    return dict(parsed)


def _with_legacy_names(env: Mapping[str, str]) -> Dict[str, str]:
    """Before the project was called VerbaDB its settings started with ``NLQ_``: still accepted (``VERBADB_*`` wins)."""
    merged = dict(env)
    for key, value in env.items():
        if key.startswith("NLQ_"):
            merged.setdefault("VERBADB_" + key[len("NLQ_"):], value)
    return merged


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Build :class:`Settings`. With ``env=None`` the process environment (plus ``.env``) is used."""
    if env is None:
        try:
            from dotenv import find_dotenv, load_dotenv

            # Search from the current directory upwards (not from this module's location), and never override
            # variables that are already set in the real environment.
            load_dotenv(find_dotenv(usecwd=True))
        except ImportError:  # python-dotenv is optional at runtime
            pass
        env = os.environ
    env = _with_legacy_names(env)

    redact_raw = env.get("VERBADB_REDACT_COLUMNS")
    redact = (
        tuple(p.strip().lower() for p in redact_raw.split(",") if p.strip())
        if redact_raw is not None
        else DEFAULT_REDACT
    )
    host = (env.get("VERBADB_HOST") or "127.0.0.1").strip()
    local = host in ("127.0.0.1", "localhost", "::1")      # a network-facing server must opt in to user connections
    return Settings(
        api_key=(env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or None),
        model=(env.get("VERBADB_MODEL") or DEFAULT_MODEL).strip(),
        databases=_databases(env.get("VERBADB_DATABASES")),
        max_steps=_int(env, "VERBADB_MAX_STEPS", 8, minimum=1),
        max_rows=_int(env, "VERBADB_MAX_ROWS", 500, minimum=1),
        preview_rows=_int(env, "VERBADB_PREVIEW_ROWS", 20, minimum=0),
        query_timeout=_int(env, "VERBADB_QUERY_TIMEOUT", 15, minimum=1),
        redact_columns=redact,
        llm_retries=_int(env, "VERBADB_LLM_RETRIES", 3, minimum=0),
        host=host,
        port=_int(env, "VERBADB_PORT", 8000, minimum=1),
        data_dir=(env.get("VERBADB_DATA_DIR") or "verbadb_data").strip(),
        allow_user_connections=_bool(env, "VERBADB_ALLOW_USER_CONNECTIONS", local),
        allow_private_db_hosts=_bool(env, "VERBADB_ALLOW_PRIVATE_DB_HOSTS", local),
        max_databases=_int(env, "VERBADB_MAX_DATABASES", 8, minimum=1),
        sandbox_max_mb=_int(env, "VERBADB_SANDBOX_MAX_MB", 20, minimum=1),
        max_saved_prompts=_int(env, "VERBADB_MAX_SAVED_PROMPTS", 500, minimum=10),
    )
