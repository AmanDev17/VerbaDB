from __future__ import annotations

import socket

import pytest

from verbadb import manager as manager_module
from verbadb.config import Settings
from verbadb.connections import open_readonly
from verbadb.manager import (
    DatabaseManager, ManagerError, clean_name, mask_url, redact, slug, validate_user_url,
)
from verbadb.sandbox import apply_changes

PUBLIC = "postgresql://reader:s3cret@93.184.216.34:5432/shop"       # a documentation-range public address


def make_settings(tmp_path, demo_url=None, **overrides):
    base = dict(data_dir=str(tmp_path / "data"), databases={"Demo DB": demo_url} if demo_url else {},
                query_timeout=5, max_databases=4)
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def mgr(tmp_path, demo_url):
    m = DatabaseManager(make_settings(tmp_path, demo_url))
    yield m
    m.close_all()


# ── names, masking, URL validation ────────────────────────────────────────

def test_names_are_cleaned():
    assert clean_name("  My   shop\n\t") == "My shop" and clean_name("x" * 200) == "x" * 60
    for bad in ("", "   ", None, "\x00\x01"):
        with pytest.raises(ManagerError, match="name"):
            clean_name(bad)
    assert slug("Demo DB!") == "demo-db" and slug("???") == "db"


def test_urls_are_masked_and_errors_redacted():
    url = "postgresql://reader:p%40ss@db.example.com:5432/shop?sslmode=require&password=oops"
    assert mask_url(url) == "postgresql://reader:***@db.example.com:5432/shop"
    assert "p%40ss" not in mask_url(url) and "oops" not in mask_url(url)
    assert mask_url("mongodb://h1,h2/db") == "mongodb://h1,h2/db"
    assert "p@ss" not in redact("auth failed for p@ss at " + url, url) and "***" in redact("bad p@ss", url)


@pytest.mark.parametrize("url", [
    "postgresql://u:p@93.184.216.34/db", "postgres://u@93.184.216.34:6543/db", "mysql://u:p@93.184.216.34/db",
    "mongodb://u:p@93.184.216.34:27017/db", "mongodb://93.184.216.34,93.184.216.35:27017/db",
    "mongodb+srv://u:p@cluster0.example.net/db", "postgresql+psycopg2://u:p@93.184.216.34/db",
])
def test_ordinary_urls_are_accepted(url):
    assert validate_user_url(url, Settings()) in ("postgres", "mysql", "mongodb")


@pytest.mark.parametrize("url,needle", [
    ("", "Enter a connection URL"), (None, "Enter a connection URL"), ("not a url", "valid connection URL"),
    ("postgresql://u:p@93.184.216.34/db with space", "valid connection URL"),
    ("sqlite:///etc/passwd", "SQLite files cannot be added"), ("sqlite:////C:/Windows/x.db", "SQLite files"),
    ("ftp://host/file", "Supported connection URLs"), ("http://169.254.169.254/latest", "Supported connection URLs"),
    ("postgresql:///dbonly", "no host"),
    ("postgresql://u:p@169.254.169.254/db", "not allowed"),           # cloud metadata endpoint
    ("mysql://u:p@[fe80::1]/db", "not allowed"), ("mysql://u:p@0.0.0.0/db", "not allowed"),
    ("mongodb://93.184.216.34,169.254.1.1/db", "not allowed"),        # every host of a replica set is checked
    ("postgresql://u:p@" + "a" * 2100 + "/db", "valid connection URL"),
])
def test_dangerous_or_malformed_urls_are_rejected(url, needle):
    with pytest.raises(ManagerError, match=needle):
        validate_user_url(url, Settings())


@pytest.mark.parametrize("url", ["postgresql://u:p@127.0.0.1/db", "mysql://u:p@192.168.1.5/db",
                                 "mongodb://u:p@[::1]:27017/db", "postgresql://u:p@10.0.0.7/db"])
def test_private_addresses_follow_the_setting(url):
    assert validate_user_url(url, Settings(allow_private_db_hosts=True))
    with pytest.raises(ManagerError, match="private network"):
        validate_user_url(url, Settings(allow_private_db_hosts=False))


def test_host_names_are_resolved_and_every_address_checked(monkeypatch):
    def fake_getaddrinfo(host, port, **kw):
        table = {"internal.example": [(socket.AF_INET, 0, 0, "", ("10.1.2.3", 0))],
                 "tricky.example": [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0)),
                                    (socket.AF_INET, 0, 0, "", ("169.254.169.254", 0))],
                 "fine.example": [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))]}
        if host not in table:
            raise socket.gaierror("nope")
        return table[host]

    monkeypatch.setattr(manager_module.socket, "getaddrinfo", fake_getaddrinfo)
    strict = Settings(allow_private_db_hosts=False)
    assert validate_user_url("postgresql://u:p@fine.example/db", strict) == "postgres"
    with pytest.raises(ManagerError, match="private network"):
        validate_user_url("postgresql://u:p@internal.example/db", strict)
    with pytest.raises(ManagerError, match="not allowed"):                # one bad address among several is enough
        validate_user_url("postgresql://u:p@tricky.example/db", Settings())
    with pytest.raises(ManagerError, match="could not be found"):
        validate_user_url("postgresql://u:p@missing.example/db", Settings())


# ── listing, ownership, isolation ─────────────────────────────────────────

def test_shared_databases_are_visible_to_everyone_and_read_only(mgr):
    for workspace in ("alice", "bob"):
        (shared,) = mgr.list_databases(workspace)
        assert shared["id"] == "shared-demo-db" and shared["kind"] == "shared" and not shared["writable"]
        assert mgr.resolve(workspace, "shared-demo-db").writable is False
    with pytest.raises(ManagerError, match="administrator"):
        mgr.delete("alice", "shared-demo-db")
    for action in (mgr.undo, mgr.reset):
        with pytest.raises(ManagerError, match="Only sandbox"):
            action("alice", "shared-demo-db")


def test_workspaces_are_isolated(mgr):
    mine = mgr.create_sandbox("alice", "Alice's books")
    assert [d["name"] for d in mgr.list_databases("alice")] == ["Demo DB", "Alice's books"]
    assert [d["name"] for d in mgr.list_databases("bob")] == ["Demo DB"]
    assert mgr.resolve("alice", mine["id"]).writable
    for action in (mgr.resolve, mgr.delete, mgr.undo, mgr.reset, mgr.get_info):
        with pytest.raises(KeyError):
            action("bob", mine["id"])
    with pytest.raises(KeyError):
        mgr.resolve("alice", "does-not-exist")


# ── sandboxes ─────────────────────────────────────────────────────────────

def test_creating_sandboxes(mgr):
    empty = mgr.create_sandbox("w", "  Empty one ", "just a test")
    assert empty["kind"] == "sandbox" and empty["writable"] and empty["backend"] == "sqlite"
    assert empty["description"] == "just a test" and empty["undo_depth"] == 0
    assert mgr.describe("w", empty["id"])["tables"] == 0

    sample = mgr.create_sandbox("w", "Shop", sample=True)
    described = mgr.describe("w", sample["id"])
    assert described["tables"] == 4 and described["enforcement"]["statement_timeout"] == "set"

    again = mgr.create_sandbox("w", "Shop")                       # names stay unique within a workspace
    assert again["name"] == "Shop (2)"
    assert mgr.create_sandbox("w", "demo db")["name"] == "demo db (2)"      # ...and do not clash with shared ones


def test_quota_is_per_workspace(mgr):
    for i in range(4):
        mgr.create_sandbox("w", f"db{i}")
    with pytest.raises(ManagerError, match="up to 4 databases") as err:
        mgr.create_sandbox("w", "one too many")
    assert err.value.status == 409
    assert mgr.create_sandbox("other", "fine")                     # someone else is unaffected


def test_changes_undo_and_reset_through_the_manager(mgr):
    info = mgr.create_sandbox("w", "Lab")
    entry = mgr.resolve("w", info["id"])
    change = apply_changes(entry.flash, ["CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)",
                                         "INSERT INTO t (id, v) VALUES (1, 'a'), (2, 'b')"], mgr.settings, entry.snapshots)
    assert change.snapshot_id and mgr.get_info("w", info["id"])["undo_depth"] == 1

    result = mgr.undo("w", info["id"])                              # back to an empty database
    assert "created 1 table" in result["undone"] and result["undo_depth"] == 0
    assert mgr.describe("w", info["id"])["tables"] == 0
    with pytest.raises(ManagerError, match="nothing to undo") as err:
        mgr.undo("w", info["id"])
    assert err.value.status == 409

    entry = mgr.resolve("w", info["id"])                            # reopened after the restore
    apply_changes(entry.flash, ["CREATE TABLE t (id INTEGER PRIMARY KEY)", "INSERT INTO t (id) VALUES (1)"],
                  mgr.settings, entry.snapshots)
    mgr.reset("w", info["id"])
    assert mgr.describe("w", info["id"])["tables"] == 0
    mgr.undo("w", info["id"])                                       # even a reset can be undone
    assert mgr.describe("w", info["id"])["tables"] == 1


def test_deleting_a_sandbox_removes_its_files(mgr):
    info = mgr.create_sandbox("w", "Temp")
    entry = mgr.resolve("w", info["id"])
    apply_changes(entry.flash, ["CREATE TABLE t (id INTEGER)"], mgr.settings, entry.snapshots)
    path = mgr.sandbox_path(info["id"])
    assert path.exists() and mgr.snapshots_for(info["id"]).depth() == 1
    mgr.delete("w", info["id"])
    assert not path.exists() and mgr.snapshots_for(info["id"]).depth() == 0
    assert [d["kind"] for d in mgr.list_databases("w")] == ["shared"]
    with pytest.raises(KeyError):
        mgr.delete("w", info["id"])


# ── user connections ──────────────────────────────────────────────────────

@pytest.fixture
def fake_connect(monkeypatch, demo_url, settings):
    """Make any 'remote' URL connect to the local demo database instead (strictly read-only, as in real use)."""
    calls = []

    def fake(url, s):
        calls.append(url)
        return open_readonly(demo_url, s)

    monkeypatch.setattr(manager_module, "open_readonly", fake)
    return calls


def test_adding_a_connection(mgr, fake_connect):
    info = mgr.add_connection("w", "Prod replica", PUBLIC)
    assert info["kind"] == "connection" and not info["writable"] and info["backend"] == "postgres"
    assert info["url"] == "postgresql://reader:***@93.184.216.34:5432/shop" and "s3cret" not in str(info)
    assert fake_connect == [PUBLIC]
    entry = mgr.resolve("w", info["id"])
    assert entry.writable is False and entry.kind == "connection" and entry.enforcement.guard
    assert [d["name"] for d in mgr.list_databases("bob")] == ["Demo DB"]            # private to the workspace
    mgr.delete("w", info["id"])
    assert [d["kind"] for d in mgr.list_databases("w")] == ["shared"]


def test_the_url_is_kept_on_the_server_and_survives_a_restart(tmp_path, demo_url, fake_connect):
    first = DatabaseManager(make_settings(tmp_path, None))
    info = first.add_connection("w", "Kept", PUBLIC)
    first.close_all()
    second = DatabaseManager(make_settings(tmp_path, None))
    try:
        assert second.get_info("w", info["id"])["url"].endswith("@93.184.216.34:5432/shop")
        assert second.resolve("w", info["id"]).kind == "connection"
        assert fake_connect.count(PUBLIC) == 2
    finally:
        second.close_all()


def test_connection_failures_are_friendly_and_leak_no_password(tmp_path, monkeypatch):
    m = DatabaseManager(make_settings(tmp_path))
    try:
        def boom(url, s):
            raise RuntimeError(f"FATAL: password authentication failed for user reader (s3cret) via {url}")

        monkeypatch.setattr(manager_module, "open_readonly", boom)
        with pytest.raises(ManagerError, match="Could not connect") as err:
            m.add_connection("w", "Bad", PUBLIC)
        assert "s3cret" not in str(err.value) and PUBLIC not in str(err.value)
        assert m.list_databases("w") == []                                # nothing was saved
    finally:
        m.close_all()


def test_connections_can_be_disabled(tmp_path):
    m = DatabaseManager(make_settings(tmp_path, allow_user_connections=False))
    try:
        with pytest.raises(ManagerError, match="disabled") as err:
            m.add_connection("w", "X", PUBLIC)
        assert err.value.status == 403
    finally:
        m.close_all()


def test_rejected_urls_are_never_saved_or_opened(tmp_path, monkeypatch):
    m = DatabaseManager(make_settings(tmp_path))
    opened = []
    monkeypatch.setattr(manager_module, "open_readonly", lambda *a: opened.append(a))
    try:
        for url in ("sqlite:///secret.db", "postgresql://u:p@169.254.169.254/db", "ftp://x/y"):
            with pytest.raises(ManagerError):
                m.add_connection("w", "Evil", url)
        assert opened == [] and m.list_databases("w") == []
    finally:
        m.close_all()


def test_connection_quota_counts_towards_the_same_limit(tmp_path, fake_connect):
    m = DatabaseManager(make_settings(tmp_path, max_databases=1))
    try:
        m.add_connection("w", "one", PUBLIC)
        with pytest.raises(ManagerError, match="up to 1 databases"):
            m.create_sandbox("w", "two")
    finally:
        m.close_all()
