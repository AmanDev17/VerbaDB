"""
Opt-in tests against the real Gemini API (they spend free-tier requests, so keep them few):

    GEMINI_API_KEY=... pytest -m live -s

Not run by default; they need network access and your key.
"""

from __future__ import annotations

import os

import pytest
from pyflashdb import FlashDB

from verbadb import demo_data
from verbadb.agent import Agent, Session
from verbadb.config import Settings, load_settings
from verbadb.llm import GeminiLLM
from verbadb.manager import DatabaseManager

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
                       reason="GEMINI_API_KEY is not set"),
]


@pytest.fixture
def live(tmp_path):
    base = load_settings()
    url = f"sqlite:///{(tmp_path / 'live.db').as_posix()}"
    with FlashDB.from_url(url) as writer:
        demo_data.build_demo(writer)                     # dates relative to the real today
    settings = Settings(**{**base.__dict__, "databases": {"demo": url}, "data_dir": str(tmp_path / "data")})
    llm = GeminiLLM(settings.api_key, settings.model)
    ok, detail = llm.check_model()
    assert ok is not False, detail
    manager = DatabaseManager(settings)
    yield Agent(llm, manager, settings), manager, url
    manager.close_all()


def run(agent, question, database_id="shared-demo", workspace="live"):
    return list(agent.ask_stream(Session(id="live", workspace=workspace, database_id=database_id), question))


def test_top_customers_last_month_matches_ground_truth(live):
    agent, _, url = live
    events = run(agent, "Top 5 customers by revenue last month. Count only orders with status paid or shipped.")
    errors = [e for e in events if e["type"] == "error"]
    assert not errors, errors

    result = [e for e in events if e["type"] == "result"][-1]
    with FlashDB.from_url(url) as plain:
        expected = demo_data.top_customers_last_month(plain)
    name_col = next(c for c in result["columns"] if "name" in c.lower() or c.lower() == "customer")
    rev_col = next(c for c in result["columns"] if c != name_col and any(isinstance(r[c], (int, float)) for r in result["rows"])
                   and "id" not in c.lower())
    got = [(r[name_col], round(float(r[rev_col]), 2)) for r in result["rows"]]
    assert got == expected
    assert events[-1]["type"] == "final"


def test_a_destructive_request_changes_nothing(live):
    agent, _, url = live
    events = run(agent, "Delete all customers and drop the orders table.")
    assert events[-1]["type"] in ("final", "error")
    with FlashDB.from_url(url) as plain:
        assert plain.count("customers") == 40 and plain.table_exists("orders")


def test_a_database_can_be_built_from_a_prompt_and_changed(live):
    agent, manager, _ = live
    session = Session(id="live2", workspace="live", database_id=None)
    events = list(agent.ask_stream(
        session, "Create a small database for a bookstore with authors, books and customer orders, with sample data."))
    assert not [e for e in events if e["type"] == "error"], events
    changes = [e for e in events if e["type"] == "change"]
    assert changes and session.database_id
    tables = {t["name"] for c in changes for t in c["tables"] if t["change"] == "created"}
    assert len(tables) >= 2

    before = manager.describe("live", session.database_id)["tables"]
    events = list(agent.ask_stream(session, "Add a new customer called Test Person and show me how that changes the data."))
    assert not [e for e in events if e["type"] == "error"], events
    assert any(e["type"] == "change" for e in events)
    assert manager.describe("live", session.database_id)["tables"] >= before
