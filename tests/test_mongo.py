"""
The MongoDB path, exercised against ``mongomock`` (an in-memory stand-in) through pyflashdb's real Mongo adapter:
same agent loop, tools, query hook and introspection as production, just without a server.
"""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pytest

mongomock = pytest.importorskip("mongomock")

from pyflashdb import FlashDB  # noqa: E402

from verbadb import guard, introspect  # noqa: E402
from verbadb.agent import Agent, Session  # noqa: E402
from verbadb.config import Settings  # noqa: E402
from verbadb.executor import run_aggregation  # noqa: E402
from verbadb.guard import GuardError  # noqa: E402
from verbadb.tools import ToolBox, tool_specs  # noqa: E402

from conftest import TODAY  # noqa: E402
from fakes import FakeLLM, call, model_turn  # noqa: E402

D = dt.datetime
ORDERS = [  # (id, customer_id, status, ordered_at, total)
    (1, 1, "paid", D(2026, 9, 3, 10), 100.0), (2, 1, "shipped", D(2026, 9, 15, 12), 50.5),
    (3, 2, "paid", D(2026, 9, 10, 9), 300.0), (4, 2, "refunded", D(2026, 9, 11, 9), 999.0),
    (5, 3, "paid", D(2026, 8, 30, 23), 700.0), (6, 3, "paid", D(2026, 9, 20, 8), 120.0),
    (7, 4, "cancelled", D(2026, 9, 21, 8), 400.0), (8, 5, "shipped", D(2026, 9, 30, 23, 59), 80.0),
    (9, 5, "paid", D(2026, 10, 1, 0, 0), 500.0),
]
EXPECTED = [("Customer 2", 300.0), ("Customer 1", 150.5), ("Customer 3", 120.0)]

PIPELINE = json.dumps([
    {"$match": {"status": {"$in": ["paid", "shipped"]},
                "ordered_at": {"$gte": {"$date": "2026-09-01T00:00:00Z"}, "$lt": {"$date": "2026-10-01T00:00:00Z"}}}},
    {"$group": {"_id": "$customer_id", "revenue": {"$sum": "$total"}}},
    {"$lookup": {"from": "customers", "localField": "_id", "foreignField": "id", "as": "c"}},
    {"$unwind": "$c"},
    {"$project": {"_id": 0, "customer": "$c.name", "revenue": 1}},
    {"$sort": {"revenue": -1}},
    {"$limit": 3},
])


@pytest.fixture(autouse=True)
def _mongomock_lists_collections(monkeypatch):
    """mongomock has no list_collections(); behave like a server whose collections have no validators."""
    from mongomock.database import Database

    def list_collections(self, filter=None, session=None, nameOnly=False):
        wanted = (filter or {}).get("name")
        return iter([{"name": n, "options": {}} for n in self.list_collection_names() if wanted in (None, n)])

    monkeypatch.setattr(Database, "list_collections", list_collections)


@pytest.fixture
def mongo():
    flash = FlashDB("mongodb", {"client": mongomock.MongoClient(), "database": "shop"})
    flash.bulk_insert("customers", [
        {"id": i, "name": f"Customer {i}", "email": f"c{i}@example.com", "country": "US" if i % 2 else "DE"}
        for i in range(1, 6)
    ])
    flash.bulk_insert("orders", [
        {"id": i, "customer_id": c, "status": s, "ordered_at": at, "total": t} for i, c, s, at, t in ORDERS
    ])
    guard.install_hook(flash)
    yield flash
    flash.close()


def pairs(result):
    return [(r["customer"], r["revenue"]) for r in result.rows]


# ── introspection ─────────────────────────────────────────────────────────

def test_tables_and_counts(mongo):
    assert {t["name"]: t["rows"] for t in introspect.list_tables(mongo)} == {"customers": 5, "orders": 9}


def test_describe_infers_foreign_keys_to_the_collections_own_id_field(mongo):
    described = introspect.describe_tables(mongo, ["orders", "customers"])
    assert {fk["column"]: fk["references"] for fk in described["orders"]["foreign_keys"]} == {"customer_id": "customers.id"}
    assert described["customers"]["foreign_keys"] == []
    assert "sample" in described["orders"]["note"].lower()
    names = [c["name"] for c in described["orders"]["columns"]]
    assert {"_id", "customer_id", "status", "ordered_at", "total"} <= set(names)


def test_foreign_keys_fall_back_to_object_id_when_there_is_no_id_field(mongo):
    with pytest.raises(GuardError):                       # the agent's own connection cannot write...
        mongo.bulk_insert("regions", [{"name": "north"}])
    mongo.clear_hooks()                                   # ...so seed through a plain, unguarded setup step
    mongo.bulk_insert("regions", [{"name": "north"}])
    mongo.bulk_insert("stores", [{"name": "s1", "region_id": "x"}])
    guard.install_hook(mongo)
    described = introspect.describe_tables(mongo, ["stores"])
    assert described["stores"]["foreign_keys"][0]["references"] == "regions._id"


# ── aggregation ───────────────────────────────────────────────────────────

def test_top_customers_pipeline_with_lookup_and_extended_json_dates(mongo, settings):
    result = run_aggregation(mongo, "orders", PIPELINE, settings)
    assert pairs(result) == EXPECTED
    assert result.language == "mongodb" and result.collection == "orders" and not result.truncated
    assert result.query.count("$limit") == 1                      # the model's own small limit is kept as-is


def test_a_limit_is_added_and_truncation_reported(mongo):
    result = run_aggregation(mongo, "orders", json.dumps([{"$match": {}}, {"$sort": {"id": 1}}]), Settings(max_rows=4))
    assert result.row_count == 4 and result.truncated and '"$limit": 5' in result.query


def test_object_ids_come_back_as_strings(mongo, settings):
    result = run_aggregation(mongo, "customers", json.dumps([{"$sort": {"id": 1}}, {"$limit": 1}]), settings)
    assert isinstance(result.rows[0]["_id"], str) and len(result.rows[0]["_id"]) == 24


def test_unknown_collections_and_database_errors(mongo, settings):
    with pytest.raises(GuardError, match="Unknown collection"):
        run_aggregation(mongo, "secrets", "[{\"$match\": {}}]", settings)


# ── the guard on the real adapter ─────────────────────────────────────────

def test_write_stages_are_blocked_before_reaching_the_server(mongo, settings):
    with pytest.raises(GuardError, match="not allowed"):
        run_aggregation(mongo, "orders", '[{"$out": "stolen"}]', settings)
    assert "stolen" not in mongo.show_tables()


def test_the_query_hook_blocks_writes_and_database_commands_on_the_connection(mongo):
    with pytest.raises(GuardError):
        mongo.aggregate("orders", [{"$out": "stolen"}])
    with pytest.raises(GuardError):
        mongo.raw("dropDatabase")
    with pytest.raises(GuardError):
        mongo.add("orders", {"id": 99})
    with pytest.raises(GuardError):
        mongo.delete("orders", {"id": 1})
    with pytest.raises(GuardError):
        mongo.update("orders", {"id": 1}, {"total": 0})
    assert mongo.count("orders") == 9 and "stolen" not in mongo.show_tables()


def test_the_hook_still_allows_reads(mongo):
    assert len(mongo.where("orders", {"status": "paid"})) == 5
    assert mongo.count("customers") == 5
    assert mongo.raw({"status": "paid"}, table="orders")


# ── tools + agent ─────────────────────────────────────────────────────────

def test_mongo_tool_specs():
    names = {s.name for s in tool_specs("mongodb", Settings())}
    assert "run_aggregation" in names and "run_sql" not in names


@pytest.fixture
def box(mongo, settings):
    return ToolBox(mongo, settings)


def test_run_aggregation_tool(box):
    out = box.call("run_aggregation", {"collection": "orders", "pipeline_json": PIPELINE})
    assert out.ok and [r["customer"] for r in out.result.rows] == ["Customer 2", "Customer 1", "Customer 3"]
    assert out.response["rows_shown"] == 3 and "3 rows" in out.summary


@pytest.mark.parametrize("args,needle", [
    ({"collection": "orders", "pipeline_json": "not json"}, "not valid JSON"),
    ({"collection": "orders", "pipeline_json": '[{"$merge": {"into": "x"}}]'}, "read-only"),
    ({"collection": "orders", "pipeline_json": ""}, "pipeline_json"),
    ({"collection": "nope", "pipeline_json": '[{"$match": {}}]'}, "Unknown collection"),
    ({"collection": "orders", "pipeline_json": 5}, "pipeline_json"),
])
def test_bad_aggregations_return_errors(box, args, needle):
    out = box.call("run_aggregation", args)
    assert not out.ok and needle in out.response["error"]


def test_sql_tool_is_not_available_on_mongo(box):
    assert not box.call("run_sql", {"sql": "SELECT 1"}).ok


def test_sample_rows_masks_redacted_fields(box):
    out = box.call("sample_rows", {"table": "customers", "limit": 2})
    assert out.ok and all(r["email"] == "[redacted]" and r["name"].startswith("Customer") for r in out.response["rows"])


class StubManager:
    """Just enough of DatabaseManager for the agent: hands out one read-only MongoDB entry."""

    def __init__(self, flash):
        info = {"id": "shop", "name": "shop", "kind": "shared", "backend": "mongodb", "writable": False}
        self.entry = SimpleNamespace(**info, flash=flash, info=lambda: info)

    def resolve(self, workspace, db_id):
        return self.entry


def test_agent_answers_a_question_on_mongodb(mongo, settings):
    llm = FakeLLM([
        model_turn("", call("list_tables")),
        model_turn("", call("describe_tables", tables=["orders", "customers"])),
        model_turn("", call("run_aggregation", collection="orders", pipeline_json=PIPELINE)),
        model_turn("Customer 2 leads."),
    ])
    agent = Agent(llm, StubManager(mongo), settings, today=lambda: TODAY)
    session = Session(id="m", workspace="w", database_id="shop")
    events = list(agent.ask_stream(session, "top 3 customers by revenue last month"))

    assert [e["type"] for e in events] == [
        "start", "step", "observation", "step", "observation", "step", "observation", "result", "final"]
    assert events[0]["backend"] == "mongodb" and events[0]["mode"] == "readonly"
    assert "MongoDB" in llm.requests[0]["system"] and "run_aggregation" in llm.requests[0]["tools"]
    result = next(e for e in events if e["type"] == "result")
    assert [(r["customer"], r["revenue"]) for r in result["rows"]] == EXPECTED
    fed_back = llm.requests[1]["history"][-1]["results"][0][1]
    assert {t["name"] for t in fed_back["tables"]} == {"customers", "orders"}
