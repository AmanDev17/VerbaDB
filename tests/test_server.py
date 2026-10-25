from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from verbadb import manager as manager_module
from verbadb.agent import Agent, Session
from verbadb.config import Settings
from verbadb.connections import open_readonly
from verbadb.manager import DatabaseManager
from verbadb.server import SessionStore, _host_name, _recorded, _sse, create_app

from conftest import TODAY
from fakes import FakeLLM, call, model_turn

SID = "a" * 32
BOOKS = [
    "CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
    "CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT NOT NULL, author_id INTEGER REFERENCES authors(id))",
    "INSERT INTO authors (id, name) VALUES (1, 'Ada'), (2, 'Bo')",
    "INSERT INTO books (id, title, author_id) VALUES (1, 'One', 1), (2, 'Two', 1), (3, 'Three', 2)",
]


@pytest.fixture
def server_settings(demo_url, settings, tmp_path):
    return Settings(**{**settings.__dict__, "databases": {"demo": demo_url}, "data_dir": str(tmp_path / "data")})


@pytest.fixture
def app_parts(server_settings):
    llm = FakeLLM([])
    app = create_app(server_settings, llm=llm)
    return app, llm


@pytest.fixture
def client(app_parts):
    with TestClient(app_parts[0]) as c:
        yield c


def events_of(response):
    assert response.headers["content-type"].startswith("text/event-stream")
    return [json.loads(block[5:].strip()) for block in response.text.split("\n\n") if block.startswith("data:")]


def ask(client, question="how many customers?", sid=SID, database_id="shared-demo"):
    return client.post("/api/ask", json={"session_id": sid, "database_id": database_id, "question": question})


# ── static + security ─────────────────────────────────────────────────────

def test_page_and_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "VerbaDB" in page.text
    assert page.headers["cache-control"] == "no-store"
    for asset, marker in (("/static/app.js", "streamAsk"), ("/static/theme.js", "verbadb.theme"), ("/static/style.css", "--accent")):
        r = client.get(asset)
        assert r.status_code == 200 and marker in r.text


def test_security_headers_and_csp(client):
    r = client.get("/api/health")
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp and "frame-ancestors 'none'" in csp
    assert r.headers["x-content-type-options"] == "nosniff"


def test_the_page_uses_no_inline_scripts_or_styles(client):
    html = client.get("/").text
    assert "<style" not in html and " style=" not in html and "onclick=" not in html
    assert html.count("<script") == 2 and 'src="/static/app.js"' in html and 'src="/static/theme.js"' in html


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1.evil.example", "attacker.com:8000"])
def test_unexpected_host_headers_are_rejected(client, host):
    assert client.get("/api/health", headers={"host": host}).status_code == 403


def test_cross_origin_posts_are_rejected(client):
    r = client.post("/api/ask", headers={"origin": "http://evil.example"},
                    json={"session_id": SID, "database_id": None, "question": "hi"})
    assert r.status_code == 403
    assert client.post("/api/databases", headers={"origin": "http://evil.example"}, json={"name": "x"}).status_code == 403
    assert client.post("/api/sessions/xy12345678/reset", headers={"origin": "http://testserver"}).status_code == 204


@pytest.mark.parametrize("header,name", [("localhost:8000", "localhost"), ("[::1]:8000", "[::1]"),
                                         ("127.0.0.1", "127.0.0.1"), ("Example.COM:1", "example.com")])
def test_host_name_parsing(header, name):
    assert _host_name(header) == name


# ── workspaces ────────────────────────────────────────────────────────────

def test_every_browser_gets_a_private_workspace_cookie(app_parts):
    app, _ = app_parts
    with TestClient(app) as alice:
        first = alice.get("/api/health")
        cookie = first.headers["set-cookie"]
        assert "verbadb_ws=" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie.replace("samesite", "SameSite")
        workspace = alice.cookies["verbadb_ws"]
        assert len(workspace) == 32
        assert "set-cookie" not in alice.get("/api/health").headers          # kept afterwards
        assert alice.cookies["verbadb_ws"] == workspace
    # A forged / malformed cookie is replaced with a fresh workspace, never trusted.
    with TestClient(app) as evil:
        reply = evil.get("/api/health", headers={"cookie": "verbadb_ws=../../etc/passwd"})
        issued = reply.headers["set-cookie"].split(";")[0].split("=", 1)[1]
        assert len(issued) == 32 and issued.isalnum() and issued != workspace


def test_workspaces_cannot_see_each_others_databases_or_history(app_parts):
    app, _ = app_parts
    with TestClient(app) as alice, TestClient(app) as bob:
        mine = alice.post("/api/databases", json={"name": "Alice only"}).json()
        ask(alice, "alice question", database_id=mine["id"])
        assert [d["name"] for d in alice.get("/api/databases").json()] == ["demo", "Alice only"]
        assert [d["name"] for d in bob.get("/api/databases").json()] == ["demo"]
        assert bob.get(f"/api/databases/{mine['id']}").status_code == 404
        assert bob.get(f"/api/schema/{mine['id']}").status_code == 404
        assert bob.delete(f"/api/databases/{mine['id']}").status_code == 404
        assert bob.post(f"/api/databases/{mine['id']}/undo").status_code == 404
        assert ask(bob, database_id=mine["id"]).status_code == 404
        assert [h["text"] for h in alice.get("/api/history").json()] == ["alice question"]
        assert bob.get("/api/history").json() == []
        assert alice.get(f"/api/databases/{mine['id']}").status_code == 200


# ── status ────────────────────────────────────────────────────────────────

def test_health_without_a_key_explains_the_setup(server_settings):
    with TestClient(create_app(server_settings, llm=None)) as client:
        info = client.get("/api/health").json()
        assert info["llm_ready"] is False and "GEMINI_API_KEY" in info["message"]
        events = events_of(ask(client))
        assert events[0]["kind"] == "config" and events[-1] == {"type": "done"}


def test_health_reports_privacy_limits_and_features(client):
    info = client.get("/api/health").json()
    assert info["llm_ready"] and info["preview_rows"] == 5 and "email" in info["redact_columns"]
    assert info["allow_user_connections"] is True and info["max_databases"] == 8 and info["sandbox_max_mb"] == 20


def test_health_checks_the_model_id_once(server_settings):
    class Checked(FakeLLM):
        checks = 0

        def check_model(self):
            Checked.checks += 1
            return False, "Model 'x' was not found. Set VERBADB_MODEL to a current model id."

    with TestClient(create_app(server_settings, llm=Checked([]))) as client:
        for _ in range(3):
            info = client.get("/api/health").json()
    assert info["model_ok"] is False and "VERBADB_MODEL" in info["model_detail"] and Checked.checks == 1


# ── databases ─────────────────────────────────────────────────────────────

def test_listing_shared_databases(client):
    (shared,) = client.get("/api/databases").json()
    assert shared["id"] == "shared-demo" and shared["kind"] == "shared" and shared["writable"] is False


def test_creating_sandboxes(client):
    empty = client.post("/api/databases", json={"name": "Lab", "description": "scratch"})
    assert empty.status_code == 201
    info = empty.json()
    assert info["kind"] == "sandbox" and info["writable"] and info["backend"] == "sqlite" and info["undo_depth"] == 0

    sample = client.post("/api/databases", json={"name": "Shop", "sample": True}).json()
    details = client.get(f"/api/databases/{sample['id']}").json()
    assert details["tables"] == 4 and details["enforcement"]["statement_timeout"] == "set"
    schema = client.get(f"/api/schema/{sample['id']}").json()
    assert {t["name"] for t in schema["tables"]} == {"customers", "products", "orders", "order_items"}

    assert client.post("/api/databases", json={"name": "   "}).status_code == 400
    assert client.post("/api/databases", json={"name": "x", "kind": "weird"}).status_code == 422


def test_the_database_quota_is_enforced(client):
    limit = client.get("/api/health").json()["max_databases"]
    for i in range(limit):
        assert client.post("/api/databases", json={"name": f"db{i}"}).status_code == 201
    over = client.post("/api/databases", json={"name": "extra"})
    assert over.status_code == 409 and f"up to {limit}" in over.json()["detail"]


def test_connecting_your_own_database(client, monkeypatch, demo_url):
    bad = client.post("/api/databases", json={"kind": "connection", "name": "x", "url": "sqlite:///secret.db"})
    assert bad.status_code == 400 and "SQLite files cannot be added" in bad.json()["detail"]
    meta = client.post("/api/databases", json={"kind": "connection", "name": "x", "url": "postgresql://u:p@169.254.169.254/db"})
    assert meta.status_code == 400 and "not allowed" in meta.json()["detail"]

    monkeypatch.setattr(manager_module, "open_readonly", lambda url, s: open_readonly(demo_url, s))
    ok = client.post("/api/databases", json={"kind": "connection", "name": "Replica",
                                             "url": "postgresql://reader:hunter2@93.184.216.34:5432/shop"})
    assert ok.status_code == 201
    info = ok.json()
    assert info["kind"] == "connection" and info["writable"] is False and info["url"].endswith("@93.184.216.34:5432/shop")
    assert "hunter2" not in json.dumps(client.get("/api/databases").json()) and "hunter2" not in ok.text
    assert client.get(f"/api/databases/{info['id']}").json()["tables"] == 4
    assert client.delete(f"/api/databases/{info['id']}").status_code == 204


def test_connections_can_be_switched_off(server_settings):
    off = Settings(**{**server_settings.__dict__, "allow_user_connections": False})
    with TestClient(create_app(off, llm=FakeLLM([]))) as client:
        r = client.post("/api/databases", json={"kind": "connection", "name": "x", "url": "postgresql://u:p@93.184.216.34/db"})
        assert r.status_code == 403 and "disabled" in r.json()["detail"]
        assert client.get("/api/health").json()["allow_user_connections"] is False


def test_deleting_and_protecting_databases(client):
    mine = client.post("/api/databases", json={"name": "Temp"}).json()
    assert client.delete(f"/api/databases/{mine['id']}").status_code == 204
    assert client.get(f"/api/databases/{mine['id']}").status_code == 404
    shared = client.delete("/api/databases/shared-demo")
    assert shared.status_code == 403 and "administrator" in shared.json()["detail"]
    assert client.post("/api/databases/shared-demo/undo").status_code == 403
    assert client.post("/api/databases/shared-demo/reset").status_code == 403


# ── building a database over HTTP, then undoing it ────────────────────────

def test_building_and_undoing_a_database_through_the_api(app_parts):
    app, llm = app_parts
    llm.script = [
        model_turn("", call("create_database", name="Library")), model_turn("", call("modify_database", statements=BOOKS)),
        model_turn("Built a small library database."),
        model_turn("", call("run_sql", sql="SELECT COUNT(*) AS n FROM books")), model_turn("Three books."),
    ]
    with TestClient(app) as client:
        events = events_of(ask(client, "make a library database", database_id=None))
        assert [e["type"] for e in events] == [
            "start", "step", "observation", "database", "step", "observation", "change", "final", "done"]
        created = next(e for e in events if e["type"] == "database")["database"]
        change = next(e for e in events if e["type"] == "change")
        assert change["database_id"] == created["id"] and change["can_undo"]
        assert [t["name"] for t in change["tables"]] == ["authors", "books"]

        listed = client.get("/api/databases").json()
        assert [d["name"] for d in listed] == ["demo", "Library"] and listed[1]["undo_depth"] == 1

        follow = events_of(ask(client, "how many books?", database_id=created["id"]))        # same chat, new database id
        assert next(e for e in follow if e["type"] == "result")["rows"] == [{"n": 3}]

        undone = client.post(f"/api/databases/{created['id']}/undo")
        assert undone.status_code == 200 and "created 2 tables" in undone.json()["undone"]
        assert client.get(f"/api/databases/{created['id']}").json()["tables"] == 0
        again = client.post(f"/api/databases/{created['id']}/undo")
        assert again.status_code == 409 and "nothing to undo" in again.json()["detail"]


def test_reset_empties_a_sandbox_and_can_itself_be_undone(app_parts):
    app, _ = app_parts
    with TestClient(app) as client:
        info = client.post("/api/databases", json={"name": "Shop", "sample": True}).json()
        assert client.post(f"/api/databases/{info['id']}/reset").status_code == 200
        assert client.get(f"/api/databases/{info['id']}").json()["tables"] == 0
        assert client.post(f"/api/databases/{info['id']}/undo").status_code == 200
        assert client.get(f"/api/databases/{info['id']}").json()["tables"] == 4


# ── /api/ask ──────────────────────────────────────────────────────────────

def test_ask_streams_events_then_done(app_parts):
    app, llm = app_parts
    llm.script = [model_turn("", call("run_sql", sql="SELECT COUNT(*) AS n FROM customers")), model_turn("There are 40 customers.")]
    with TestClient(app) as client:
        events = events_of(ask(client))
    assert [e["type"] for e in events] == ["start", "step", "observation", "result", "final", "done"]
    assert next(e for e in events if e["type"] == "result")["rows"] == [{"n": 40}]
    assert events[-2]["text"] == "There are 40 customers."


def test_follow_ups_share_history_and_reset_clears_it(app_parts):
    app, llm = app_parts
    llm.script = [model_turn("one"), model_turn("two"), model_turn("three")]
    with TestClient(app) as client:
        ask(client, "first")
        ask(client, "second")
        assert len(llm.requests[1]["history"]) == 3                      # user, model, user
        assert client.post(f"/api/sessions/{SID}/reset").status_code == 204
        ask(client, "third")
        assert len(llm.requests[2]["history"]) == 1                      # fresh session


def test_switching_database_starts_a_fresh_session(app_parts):
    app, llm = app_parts
    llm.script = [model_turn("a"), model_turn("b")]
    with TestClient(app) as client:
        other = client.post("/api/databases", json={"name": "Other"}).json()
        ask(client, "q1", database_id="shared-demo")
        ask(client, "q2", database_id=other["id"])
    assert len(llm.requests[1]["history"]) == 1


def test_sessions_are_scoped_to_the_workspace(app_parts):
    app, llm = app_parts
    llm.script = [model_turn("a"), model_turn("b")]
    with TestClient(app) as alice, TestClient(app) as bob:
        ask(alice, "alice says hi")
        ask(bob, "bob says hi")                                          # same session id, different workspace
    assert len(llm.requests[1]["history"]) == 1


def test_ask_validates_input(client):
    assert client.post("/api/ask", json={"session_id": "short", "database_id": None, "question": "x"}).status_code == 422
    assert client.post("/api/ask", json={"session_id": "bad id!!!!!!", "database_id": None, "question": "x"}).status_code == 422
    assert client.post("/api/ask", json={"session_id": SID, "database_id": None, "question": "x" * 5000}).status_code == 422
    assert ask(client, database_id="nope").status_code == 404


def test_blocked_write_on_a_shared_database_is_reported_as_a_failed_step(app_parts):
    app, llm = app_parts
    llm.script = [model_turn("", call("run_sql", sql="DROP TABLE customers")), model_turn("I can only read data."),
                  model_turn("", call("run_sql", sql="SELECT COUNT(*) AS n FROM customers")), model_turn("40")]
    with TestClient(app) as client:
        events = events_of(ask(client, "drop the customers table"))
        obs = next(e for e in events if e["type"] == "observation")
        assert not obs["ok"] and "read-only" in obs["summary"].lower()
        again = events_of(ask(client, "how many customers?"))              # the table is still there
        assert next(e for e in again if e["type"] == "result")["rows"] == [{"n": 40}]


# ── prompt history ────────────────────────────────────────────────────────

def test_prompts_are_saved_with_their_outcome(app_parts):
    app, llm = app_parts
    llm.script = [model_turn("The answer is 40."), model_turn("", call("run_sql", sql="SELECT 1")), model_turn("   ")]
    with TestClient(app) as client:
        ask(client, "how many customers?")
        ask(client, "this one fails")                                      # blank final answer -> error
        ask(client, "   ")                                                 # validation errors are not saved
        rows = client.get("/api/history").json()
    assert [(r["text"], r["status"]) for r in rows] == [("this one fails", "error"), ("how many customers?", "ok")]
    assert rows[1]["summary"] == "The answer is 40." and rows[1]["database_name"] == "demo"
    assert rows[1]["database_id"] == "shared-demo" and rows[1]["created_at"]


def test_history_follows_the_chat_to_a_database_it_created(app_parts):
    app, llm = app_parts
    llm.script = [model_turn("", call("create_database", name="Zoo")), model_turn("made it")]
    with TestClient(app) as client:
        ask(client, "make me a zoo database", database_id=None)
        (row,) = client.get("/api/history").json()
        assert row["database_name"] == "Zoo" and row["status"] == "ok"
        assert client.get("/api/history", params={"database_id": row["database_id"]}).json()[0]["text"] == "make me a zoo database"


def test_history_search_delete_and_clear(app_parts):
    app, llm = app_parts
    llm.script = [model_turn("ok")] * 4
    with TestClient(app) as client:
        for q in ("revenue by month", "top customers", "orders by status", "revenue by country"):
            ask(client, q)
        assert [r["text"] for r in client.get("/api/history", params={"q": "REVENUE"}).json()] == ["revenue by country", "revenue by month"]
        assert len(client.get("/api/history", params={"limit": 2}).json()) == 2
        first = client.get("/api/history").json()[-1]
        assert client.delete(f"/api/history/{first['id']}").status_code == 204
        assert client.delete(f"/api/history/{first['id']}").status_code == 404
        assert client.delete("/api/history").json() == {"deleted": 3}
        assert client.get("/api/history").json() == []


def test_a_stopped_request_is_recorded_as_stopped(server_settings):
    manager = DatabaseManager(server_settings)
    try:
        agent = Agent(FakeLLM([model_turn("", call("list_tables")), model_turn("never")]), manager, server_settings,
                      today=lambda: TODAY)
        session = Session(id=SID, workspace="w", database_id="shared-demo")
        stream = _sse(_recorded(agent.ask_stream(session, "long question"), manager=manager, workspace="w",
                                question="long question", database_id="shared-demo", database_name="demo"))
        assert json.loads(next(stream)[5:])["type"] == "start"
        stream.close()                                                       # the browser went away
        assert not session.lock.locked() and session.history == []
        (row,) = manager.store.list_prompts("w")
        assert row["text"] == "long question" and row["status"] == "stopped"
    finally:
        manager.close_all()


# ── disconnects and sessions ──────────────────────────────────────────────

def test_closing_the_stream_early_releases_the_session_and_rolls_back(server_settings):
    manager = DatabaseManager(server_settings)
    try:
        llm = FakeLLM([model_turn("", call("list_tables")), model_turn("never reached")])
        agent = Agent(llm, manager, server_settings, today=lambda: TODAY)
        session = Session(id=SID, workspace="w", database_id="shared-demo")
        stream = _sse(agent.ask_stream(session, "hello"))
        assert json.loads(next(stream)[5:])["type"] == "start"      # mid-flight: lock held, history has the question
        assert session.lock.locked() and len(session.history) == 1
        stream.close()                                              # the browser went away
        assert not session.lock.locked() and session.history == []
    finally:
        manager.close_all()


def test_session_store_expires_idle_sessions():
    store = SessionStore(ttl=0.01, max_sessions=10)
    first = store.get("w", "aaaaaaaa", None)
    assert store.get("w", "aaaaaaaa", None) is first                 # still fresh: same session
    time.sleep(0.05)
    assert store.get("w", "aaaaaaaa", None) is not first             # expired: replaced


def test_session_store_keys_by_workspace_and_database():
    store = SessionStore()
    a = store.get("w1", "aaaaaaaa", "db1")
    assert store.get("w1", "aaaaaaaa", "db1") is a
    assert store.get("w2", "aaaaaaaa", "db1") is not a               # another workspace
    assert store.get("w1", "aaaaaaaa", "db2") is not a               # another database: fresh history
    store.reset("w1", "aaaaaaaa")
    assert store.get("w1", "aaaaaaaa", "db2") is not a


def test_session_store_enforces_its_cap_immediately():
    store = SessionStore(max_sessions=2)
    for sid in ("aaaaaaaa", "bbbbbbbb", "cccccccc"):
        store.get("w", sid, None)
    assert len(store._sessions) == 2 and ("w", "cccccccc") in store._sessions


def test_session_store_never_evicts_a_running_session():
    store = SessionStore(max_sessions=2)
    running = store.get("w", "aaaaaaaa", None)
    running.lock.acquire()
    try:
        for sid in ("bbbbbbbb", "cccccccc", "dddddddd"):
            store.get("w", sid, None)
        assert store._sessions[("w", "aaaaaaaa")] is running
    finally:
        running.lock.release()
