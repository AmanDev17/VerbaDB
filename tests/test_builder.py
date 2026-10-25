"""Building databases from a prompt, and simulating changes to them, through the real agent loop and tools."""

from __future__ import annotations

import json

import pytest

from verbadb.agent import Agent, Session
from verbadb.config import Settings
from verbadb.manager import DatabaseManager
from verbadb.sandbox_tools import SandboxToolBox

from conftest import TODAY
from fakes import FakeLLM, call, model_turn

BOOKSTORE = [
    "CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT)",
    "CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT NOT NULL, author_id INTEGER NOT NULL "
    "REFERENCES authors(id), price REAL, published TEXT)",
    "INSERT INTO authors (id, name, country) VALUES (1, 'Ada Quill', 'UK'), (2, 'Bo Marsh', 'US'), (3, 'Cy Lund', 'SE')",
    "INSERT INTO books (id, title, author_id, price, published) VALUES "
    "(1, 'The Quiet Engine', 1, 12.5, '2021-03-01'), (2, 'Northern Maps', 3, 18.0, '2019-11-20'), "
    "(3, 'Salt and Signal', 2, 9.99, '2023-06-15'), (4, 'Second Light', 1, 14.0, '2022-01-10')",
]


@pytest.fixture
def manager(tmp_path, settings):
    m = DatabaseManager(Settings(**{**settings.__dict__, "data_dir": str(tmp_path / "data"), "databases": {}}))
    yield m
    m.close_all()


def make(script, manager):
    llm = FakeLLM(script)
    return Agent(llm, manager, manager.settings, sleep=lambda s: None, today=lambda: TODAY), llm


def run(agent, session, question):
    events = list(agent.ask_stream(session, question))
    for event in events:                      # everything streamed to the browser must be JSON
        json.dumps(event)
    return events


def kinds(events):
    return [e["type"] for e in events]


# ── building from a prompt ────────────────────────────────────────────────

def test_a_database_is_built_from_a_prompt_in_a_fresh_chat(manager):
    agent, llm = make([
        model_turn("I'll create a bookstore database.", call("create_database", name="Bookstore", description="Books and authors")),
        model_turn("", call("modify_database", statements=BOOKSTORE)),
        model_turn("Built **Bookstore**: authors and books (4 books by 3 authors)."),
    ], manager)
    session = Session(id="s", workspace="w", database_id=None)

    events = run(agent, session, "Make me a database for a small bookstore with authors and books")

    assert kinds(events) == ["start", "text", "step", "observation", "database", "step", "observation", "change", "final"]
    start = events[0]
    assert start["mode"] == "sandbox" and start["database"] is None
    created = next(e for e in events if e["type"] == "database")
    assert created["created"] and created["database"]["name"] == "Bookstore" and created["database"]["writable"]
    assert session.database_id == created["database"]["id"]                         # the chat now works on it

    change = next(e for e in events if e["type"] == "change")
    assert [t["name"] for t in change["tables"]] == ["authors", "books"]
    assert all(t["change"] == "created" for t in change["tables"])
    assert change["database_id"] == session.database_id and change["can_undo"]
    assert change["summary"] == "Applied 4 statements: created 2 tables · +7 rows"
    assert change["verbs"] == ["CREATE TABLE", "CREATE TABLE", "INSERT", "INSERT"]

    # The sandbox toolset and prompt were used, and the model only got counts back.
    assert "modify_database" in llm.requests[0]["tools"] and "create_database" in llm.requests[0]["tools"]
    assert "SANDBOX" in llm.requests[0]["system"]
    fed_back = llm.requests[2]["history"][-1]["results"][0][1]
    assert fed_back["ok"] and {c["table"] for c in fed_back["changes"]} == {"authors", "books"}
    assert "Quiet Engine" not in json.dumps(fed_back)

    # The data really exists, only for this workspace.
    entry = manager.resolve("w", session.database_id)
    assert entry.flash.count("books") == 4 and entry.flash.count("authors") == 3
    assert [d["name"] for d in manager.list_databases("w")] == ["Bookstore"]
    assert manager.list_databases("someone-else") == []


def test_the_whole_build_can_be_undone(manager):
    agent, _ = make([model_turn("", call("create_database", name="Temp")),
                     model_turn("", call("modify_database", statements=BOOKSTORE)), model_turn("done")], manager)
    session = Session(id="s", workspace="w")
    run(agent, session, "build")
    assert manager.describe("w", session.database_id)["tables"] == 2
    manager.undo("w", session.database_id)
    assert manager.describe("w", session.database_id)["tables"] == 0


def test_follow_ups_continue_on_the_database_that_was_just_created(manager):
    agent, llm = make([
        model_turn("", call("create_database", name="Shop")), model_turn("", call("modify_database", statements=BOOKSTORE)),
        model_turn("built"),
        model_turn("", call("run_sql", sql="SELECT COUNT(*) AS n FROM books")), model_turn("4 books"),
    ], manager)
    session = Session(id="s", workspace="w")
    run(agent, session, "make a bookstore")
    events = run(agent, session, "how many books?")
    assert events[0]["database"]["name"] == "Shop" and next(e for e in events if e["type"] == "result")["rows"] == [{"n": 4}]
    assert events[-1]["text"] == "4 books"


# ── simulating changes ────────────────────────────────────────────────────

@pytest.fixture
def shop(manager):
    info = manager.create_sandbox("w", "Shop", sample=True)
    return Session(id="s", workspace="w", database_id=info["id"]), info


def test_an_update_is_simulated_with_old_and_new_values(manager, shop):
    session, info = shop
    agent, llm = make([
        model_turn("", call("modify_database", statements=["UPDATE products SET price = ROUND(price * 1.1, 2) WHERE category = 'electronics'"])),
        model_turn("Raised electronics prices by 10%."),
    ], manager)
    events = run(agent, session, "raise electronics prices by 10%")

    assert events[0]["mode"] == "sandbox" and events[0]["database"]["id"] == info["id"]
    change = next(e for e in events if e["type"] == "change")
    (products,) = change["tables"]
    assert products["name"] == "products" and products["change"] == "modified"
    assert products["updated_total"] == 4 and products["inserted_total"] == 0 and products["deleted_total"] == 0
    monitor = next(u for u in products["updated"] if u["after"]["name"] == "Monitor")
    assert (monitor["before"]["price"], monitor["after"]["price"], monitor["changed"]) == (219, 240.9, ["price"])
    assert "4 rows edited" in change["summary"]
    fed_back = json.dumps(llm.requests[1]["history"][-1])
    assert "Monitor" not in fed_back and "240.9" not in fed_back                      # counts only go to the model

    entry = manager.resolve("w", info["id"])
    assert entry.snapshots.depth() == 1


def test_a_delete_shows_cascading_side_effects_and_can_be_undone(manager, shop):
    session, info = shop
    agent, _ = make([model_turn("", call("modify_database", statements=["DELETE FROM customers WHERE id <= 3"])),
                     model_turn("Removed three customers.")], manager)
    change = next(e for e in run(agent, session, "remove the first three customers") if e["type"] == "change")
    (customers,) = change["tables"]
    assert customers["deleted_total"] == 3 and {r["id"] for r in customers["deleted"]} == {1, 2, 3}
    assert "−3 rows" in change["summary"]

    manager.undo("w", info["id"])
    assert manager.resolve("w", info["id"]).flash.count("customers") == 40


def test_a_failed_batch_is_reported_changes_nothing_and_the_model_can_retry(manager, shop):
    session, info = shop
    agent, llm = make([
        model_turn("", call("modify_database", statements=[
            "INSERT INTO customers (id, name, email, country, created_at) VALUES (900, 'Temp', 't@x.io', 'US', '2026-01-01')",
            "INSERT INTO orders (id, customer_id, status, ordered_at, total) VALUES (9000, 'oops', 'paid')",   # wrong column count
        ])),
        model_turn("", call("modify_database", statements=[
            "INSERT INTO customers (id, name, email, country, created_at) VALUES (900, 'Temp', 't@x.io', 'US', '2026-01-01')"])),
        model_turn("Added one customer."),
    ], manager)
    events = run(agent, session, "add a customer")
    observations = [e for e in events if e["type"] == "observation"]
    assert [o["ok"] for o in observations] == [False, True]
    assert "rolled back" in observations[0]["summary"]
    assert kinds(events).count("change") == 1                           # only the successful retry produced a diff
    assert manager.resolve("w", info["id"]).flash.count("customers") == 41


def test_the_guard_blocks_unsafe_statements_in_the_sandbox(manager, shop):
    session, info = shop
    bad = [["PRAGMA writable_schema = ON"], ["ATTACH DATABASE 'other.db' AS o"], ["CREATE VIEW v AS SELECT 1"],
           ["INSERT INTO sqlite_master (name) VALUES ('x')"], ["SELECT 1"], ["DROP TABLE customers", "VACUUM"]]
    agent, llm = make([model_turn("", *[call("modify_database", statements=s) for s in bad]), model_turn("Refused.")], manager)
    events = run(agent, session, "try some nasty things")
    observations = [e for e in events if e["type"] == "observation"]
    assert len(observations) == len(bad) and not any(o["ok"] for o in observations)
    assert "change" not in kinds(events)
    entry = manager.resolve("w", info["id"])
    assert len(entry.flash.show_tables()) == 4 and entry.snapshots.depth() == 0


def test_run_sql_stays_read_only_even_in_a_sandbox(manager, shop):
    session, info = shop
    agent, _ = make([model_turn("", call("run_sql", sql="DELETE FROM customers")), model_turn("no")], manager)
    events = run(agent, session, "use the read tool to delete")
    obs = next(e for e in events if e["type"] == "observation")
    assert not obs["ok"] and "read-only" in obs["summary"].lower()
    assert manager.resolve("w", info["id"]).flash.count("customers") == 40


def test_questions_about_a_sandbox_use_the_same_tools(manager, shop):
    session, _ = shop
    agent, _ = make([model_turn("", call("list_tables")), model_turn("", call("describe_tables", tables=["orders"])),
                     model_turn("", call("run_sql", sql="SELECT status, COUNT(*) AS n FROM orders GROUP BY status ORDER BY n DESC")),
                     model_turn("Mostly paid.")], manager)
    events = run(agent, session, "orders by status")
    result = next(e for e in events if e["type"] == "result")
    assert {r["status"] for r in result["rows"]} == {"paid", "shipped", "refunded", "cancelled", "pending"}


# ── toolbox details ───────────────────────────────────────────────────────

def test_read_tools_explain_when_no_database_is_active_yet(manager):
    box = SandboxToolBox(manager, "w", None, manager.settings)
    for name, args in (("list_tables", {}), ("describe_tables", {"tables": ["x"]}), ("run_sql", {"sql": "SELECT 1"}),
                       ("modify_database", {"statements": ["CREATE TABLE t (a int)"]})):
        out = box.call(name, args)
        assert not out.ok and "No database is active yet" in out.response["error"]
    assert not box.call("nonsense", {}).ok


def test_create_database_validates_and_respects_the_quota(manager):
    box = SandboxToolBox(manager, "w", None, manager.settings)
    assert not box.call("create_database", {}).ok and not box.call("create_database", {"name": "   "}).ok
    for i in range(manager.settings.max_databases):
        assert box.call("create_database", {"name": f"db{i}"}).ok
    full = box.call("create_database", {"name": "one more"})
    assert not full.ok and "up to" in full.response["error"]
    assert len(manager.list_databases("w")) == manager.settings.max_databases


def test_modify_database_argument_errors(manager):
    box = SandboxToolBox(manager, "w", None, manager.settings)
    box.call("create_database", {"name": "Lab"})
    for bad in ({}, {"statements": []}, {"statements": [1, 2]}, {"statements": None}):
        out = box.call("modify_database", bad)
        assert not out.ok and "non-empty list" in out.response["error"]
    assert box.call("modify_database", {"statements": "CREATE TABLE t (a INTEGER)"}).ok         # a bare string is tolerated


def test_the_toolbox_survives_an_undo_between_calls(manager):
    box = SandboxToolBox(manager, "w", None, manager.settings)
    info = box.call("create_database", {"name": "Lab"}).events[0]["database"]
    assert box.call("modify_database", {"statements": ["CREATE TABLE t (id INTEGER PRIMARY KEY)"]}).ok
    manager.undo("w", info["id"])                                       # closes and reopens the file behind its back
    tables = box.call("list_tables", {})
    assert tables.ok and tables.response["tables"] == []
    assert box.call("modify_database", {"statements": ["CREATE TABLE u (id INTEGER PRIMARY KEY)"]}).ok


def test_a_deleted_database_is_reported_cleanly(manager):
    box = SandboxToolBox(manager, "w", None, manager.settings)
    info = box.call("create_database", {"name": "Gone"}).events[0]["database"]
    manager.delete("w", info["id"])
    out = box.call("list_tables", {})
    assert not out.ok and "no longer exists" in out.response["error"]
