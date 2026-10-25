from __future__ import annotations

import threading

import pytest
from pyflashdb import FlashDB

from verbadb import demo_data
from verbadb.agent import Agent, Session
from verbadb.config import Settings
from verbadb.llm import LLMUnavailable, RateLimited
from verbadb.manager import DatabaseManager

from conftest import TODAY
from fakes import FakeLLM, call, model_turn

TOP5_SQL = """
    SELECT c.id, c.name AS customer, ROUND(SUM(o.total), 2) AS revenue
    FROM orders o JOIN customers c ON c.id = o.customer_id
    WHERE o.status IN ('paid', 'shipped') AND o.ordered_at >= '2026-09-01' AND o.ordered_at < '2026-10-01'
    GROUP BY c.id, c.name ORDER BY revenue DESC LIMIT 5
"""


def new_manager(settings, tmp_path, **databases):
    return DatabaseManager(Settings(**{**settings.__dict__, "databases": databases, "data_dir": str(tmp_path / "data")}))


@pytest.fixture
def manager(demo_url, settings, tmp_path):
    m = new_manager(settings, tmp_path, demo=demo_url)
    yield m
    m.close_all()


def make(script, manager, **kw):
    sleeps = []
    llm = FakeLLM(script)
    agent = Agent(llm, manager, manager.settings, sleep=sleeps.append, today=lambda: TODAY, **kw)
    return agent, llm, sleeps


def shared_session(sid="s1"):
    return Session(id=sid, workspace="w", database_id="shared-demo")


def run(agent, question="top 5 customers by revenue last month", session=None):
    session = session or shared_session()
    return session, list(agent.ask_stream(session, question))


def kinds(events):
    return [e["type"] for e in events]


def test_full_exchange_streams_steps_result_and_final(manager, demo_url):
    script = [
        model_turn("Let me look at the tables.", call("list_tables")),
        model_turn("", call("describe_tables", tables=["customers", "orders"])),
        model_turn("", call("run_sql", sql=TOP5_SQL)),
        model_turn("Ethan Kim leads last month."),
    ]
    agent, llm, sleeps = make(script, manager)
    session, events = run(agent)

    assert kinds(events) == [
        "start", "text", "step", "observation", "step", "observation", "step", "observation", "result", "final",
    ]
    start = events[0]
    assert (start["backend"], start["mode"], start["model"]) == ("sqlite", "readonly", "fake-model")
    assert start["database"] == {"id": "shared-demo", "name": "demo", "kind": "shared", "backend": "sqlite", "writable": False}
    assert [e["tool"] for e in events if e["type"] == "step"] == ["list_tables", "describe_tables", "run_sql"]
    assert all(e["ok"] for e in events if e["type"] == "observation")

    result = next(e for e in events if e["type"] == "result")
    with FlashDB.from_url(demo_url) as plain:
        expected = demo_data.top_customers_last_month(plain, TODAY)
    assert [(r["customer"], r["revenue"]) for r in result["rows"]] == expected

    final = events[-1]
    assert final["text"] == "Ethan Kim leads last month." and final["requests"] == 4 and final["steps"] == 3
    assert sleeps == []

    # System prompt names the dialect; today's date rides in the user turn, not the system prompt.
    assert "SQLite" in llm.requests[0]["system"] and "2026" not in llm.requests[0]["system"]
    first_user = llm.requests[0]["history"][0]["text"]
    assert first_user.startswith("Today's date: 2026-10-25") and "top 5 customers" in first_user
    assert llm.requests[0]["tools"] == ["list_tables", "describe_tables", "sample_rows", "run_sql"]    # read-only set

    # History: user + (model, tool) x3 + final model turn, with model turns appended verbatim.
    assert len(session.history) == 8 and session.marks == [0]
    assert session.history[-1]["text"] == "Ethan Kim leads last month."


def test_tool_error_is_fed_back_so_the_model_can_self_correct(manager):
    script = [
        model_turn("", call("run_sql", sql="SELECT nope FROM customers")),
        model_turn("", call("run_sql", sql="SELECT COUNT(*) AS n FROM customers")),
        model_turn("There are 40 customers."),
    ]
    agent, llm, _ = make(script, manager)
    _, events = run(agent, "how many customers?")
    observations = [e for e in events if e["type"] == "observation"]
    assert [o["ok"] for o in observations] == [False, True]
    fed_back = llm.requests[1]["history"][-1]["results"][0][1]
    assert "database returned an error" in fed_back["error"] and "nope" in fed_back["error"]
    assert events[-1]["type"] == "final"


def test_destructive_request_is_blocked_and_nothing_changes(manager, demo_url):
    script = [
        model_turn("", call("run_sql", sql="DELETE FROM customers")),
        model_turn("I can only read data, so I did not delete anything."),
    ]
    agent, llm, _ = make(script, manager)
    _, events = run(agent, "delete all customers")
    obs = next(e for e in events if e["type"] == "observation")
    assert not obs["ok"] and "read-only" in obs["summary"].lower()
    assert "result" not in kinds(events)
    with FlashDB.from_url(demo_url) as plain:
        assert plain.count("customers") == 40


def test_read_only_databases_have_no_write_tools(manager):
    agent, llm, _ = make([model_turn("ok")], manager)
    run(agent)
    assert "modify_database" not in llm.requests[0]["tools"] and "create_database" not in llm.requests[0]["tools"]
    # ...and a model that calls them anyway gets an error, not a change.
    agent, llm, _ = make([model_turn("", call("modify_database", statements=["DROP TABLE customers"])),
                          model_turn("done")], manager)
    _, events = run(agent)
    assert not next(e for e in events if e["type"] == "observation")["ok"]


def test_parallel_calls_get_one_response_each_in_order(manager):
    script = [
        model_turn("", call("list_tables"), call("describe_tables", tables=["products"])),
        model_turn("done"),
    ]
    agent, llm, _ = make(script, manager)
    session, events = run(agent)
    tool_item = llm.requests[1]["history"][-1]
    assert [name for name, _ in tool_item["results"]] == ["list_tables", "describe_tables"]
    assert kinds(events).count("step") == 2


def test_step_limit_ends_with_error_and_rolls_back_history(manager, settings):
    script = [model_turn("", call("list_tables")) for _ in range(settings.max_steps)]
    agent, llm, _ = make(script, manager)
    session, events = run(agent)
    assert events[-1]["type"] == "error" and events[-1]["kind"] == "limit"
    assert session.history == [] and session.marks == []

    # The session is still usable afterwards.
    llm.script.append(model_turn("ok"))
    _, again = run(agent, session=session)
    assert again[-1]["type"] == "final" and len(session.history) == 2


def test_rate_limit_is_retried_after_the_advised_delay(manager):
    script = [RateLimited("slow down", retry_after=7.0), model_turn("fine")]
    agent, _, sleeps = make(script, manager)
    _, events = run(agent)
    status = next(e for e in events if e["type"] == "status")
    assert status["retry_after"] == 7.0 and "rate limit" in status["message"]
    assert sleeps == [7.0] and events[-1]["type"] == "final"


def test_daily_quota_is_not_retried(manager):
    agent, _, sleeps = make([RateLimited("daily used up", daily=True)], manager)
    session, events = run(agent)
    assert events[-1]["kind"] == "quota" and events[-1]["daily"] is True
    assert sleeps == [] and session.history == []


def test_rate_limit_gives_up_after_the_configured_retries(manager):
    script = [RateLimited("again", retry_after=1.0)] * (manager.settings.llm_retries + 1)
    agent, _, sleeps = make(script, manager)
    _, events = run(agent)
    assert len(sleeps) == manager.settings.llm_retries
    assert events[-1]["type"] == "error" and events[-1]["kind"] == "quota"


def test_overlong_retry_delay_is_not_waited_out(manager):
    agent, _, sleeps = make([RateLimited("later", retry_after=3600.0)], manager)
    _, events = run(agent)
    assert sleeps == [] and events[-1]["kind"] == "quota"


def test_transient_failures_back_off_exponentially(manager):
    script = [LLMUnavailable("503"), LLMUnavailable("503"), model_turn("ok")]
    agent, _, sleeps = make(script, manager)
    _, events = run(agent)
    assert sleeps == [2.0, 4.0] and events[-1]["type"] == "final"


def test_empty_model_answer_is_an_error_not_a_blank_reply(manager):
    agent, _, _ = make([model_turn("   ")], manager)
    session, events = run(agent)
    assert events[-1]["type"] == "error" and session.history == []


def test_unexpected_exception_becomes_an_internal_error(manager):
    agent, _, _ = make([RuntimeError("boom")], manager)
    session, events = run(agent)
    assert events[-1]["kind"] == "internal" and "boom" in events[-1]["message"] and session.history == []


def test_input_validation_and_busy_session(manager):
    agent, _, _ = make([], manager)
    assert run(agent, "   ")[1][0]["kind"] == "input"
    assert run(agent, "x" * 5000)[1][0]["kind"] == "input"

    session = shared_session("busy")
    session.lock.acquire()
    try:
        assert run(agent, "hello", session=session)[1][0]["kind"] == "busy"
    finally:
        session.lock.release()


def test_unknown_and_unreachable_databases(manager, settings, tmp_path):
    agent, _, _ = make([], manager)
    events = run(agent, session=Session(id="x", workspace="w", database_id="nope"))[1]
    assert events[0]["kind"] == "connection" and "no longer exists" in events[0]["message"]
    assert run(agent, session=Session(id="y", workspace="other", database_id="s" * 32))[1][0]["kind"] == "connection"

    broken = new_manager(settings, tmp_path / "b", bad="postgresql://u:p@127.0.0.1:1/none")
    try:
        agent2 = Agent(FakeLLM([]), broken, broken.settings)
        events = list(agent2.ask_stream(Session(id="z", workspace="w", database_id="shared-bad"), "hi"))
        assert events[0]["type"] == "error" and events[0]["kind"] == "connection"
        assert "Could not open" in events[0]["message"]
    finally:
        broken.close_all()


def test_one_workspace_cannot_use_anothers_database(manager):
    mine = manager.create_sandbox("alice", "Private")
    agent, _, _ = make([model_turn("never")], manager)
    events = run(agent, session=Session(id="s", workspace="bob", database_id=mine["id"]))[1]
    assert events[0]["type"] == "error" and events[0]["kind"] == "connection"


def test_history_is_trimmed_only_at_exchange_boundaries(manager, demo_url, settings, tmp_path):
    small = new_manager(Settings(**{**settings.__dict__, "max_history_items": 5}), tmp_path / "t", demo=demo_url)
    try:
        script = [model_turn("", call("list_tables")), model_turn("one")] * 4
        agent, _, _ = make(script, small)
        session = shared_session("t")
        for _ in range(4):
            assert run(agent, session=session)[1][-1]["type"] == "final"
        assert session.history[0]["role"] == "user" and session.history[0]["text"].startswith("Today's date")
        assert len(session.history) <= 5 + 4
        assert session.marks[0] == 0 and all(m < len(session.history) for m in session.marks)
    finally:
        small.close_all()


def test_concurrent_asks_on_one_session_do_not_interleave(manager):
    agent, llm, _ = make([model_turn("a"), model_turn("b")], manager)
    session = shared_session("c")
    inside_model = threading.Event()      # the first ask is running and holds the session lock
    release = threading.Event()           # lets the first ask finish
    original = llm.generate

    def slow(system, tools, history):
        inside_model.set()
        assert release.wait(30), "test never released the first ask"
        return original(system, tools, history)

    llm.generate = slow
    first = []
    worker = threading.Thread(target=lambda: first.extend(agent.ask_stream(session, "q1")), daemon=True)
    worker.start()
    try:
        assert inside_model.wait(30), "the first ask never reached the model"
        second = list(agent.ask_stream(session, "q2"))
    finally:
        release.set()
        worker.join(30)
    assert not worker.is_alive()
    assert second[0]["kind"] == "busy" and first[-1]["type"] == "final"
