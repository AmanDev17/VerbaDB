from __future__ import annotations

import pytest
from pyflashdb import FlashDB

from verbadb import introspect
from verbadb.config import Settings
from verbadb.connections import open_readonly
from verbadb.tools import ToolBox, tool_specs


# ── introspection ─────────────────────────────────────────────────────────

def test_list_tables_with_counts(ro):
    tables = {t["name"]: t["rows"] for t in introspect.list_tables(ro)}
    assert tables == {"customers": 40, "order_items": tables["order_items"], "orders": 480, "products": 12}
    assert tables["order_items"] > 480


def test_describe_infers_foreign_keys_from_names(ro):
    described = introspect.describe_tables(ro, ["orders", "order_items"])
    orders_fk = {fk["column"]: fk for fk in described["orders"]["foreign_keys"]}
    assert orders_fk["customer_id"]["references"] == "customers.id"
    assert "inferred" in orders_fk["customer_id"]["source"]
    items_fk = {fk["column"]: fk["references"] for fk in described["order_items"]["foreign_keys"]}
    assert items_fk == {"order_id": "orders.id", "product_id": "products.id"}
    assert {"name": "id", "type": "INTEGER", "nullable": False, "primary_key": True}.items() <= \
        described["orders"]["columns"][0].items() or described["orders"]["columns"][0]["primary_key"]


def test_describe_reads_declared_foreign_keys(tmp_path, settings):
    url = f"sqlite:///{(tmp_path / 'fk.db').as_posix()}"
    with FlashDB.from_url(url) as writer:
        writer.raw("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT)")
        writer.raw(
            "CREATE TABLE pets (id INTEGER PRIMARY KEY, name TEXT, "
            "keeper INTEGER REFERENCES people(id), vet_id INTEGER)"
        )
    flash, _ = open_readonly(url, settings)
    try:
        fks = {fk["column"]: fk for fk in introspect.describe_tables(flash, ["pets"])["pets"]["foreign_keys"]}
    finally:
        flash.close()
    assert fks["keeper"]["references"] == "people.id" and fks["keeper"]["source"] == "declared"
    assert "vet_id" not in fks                      # no table called vets / vet -> nothing to infer


def test_describe_handles_unknown_and_case_insensitive_names(ro):
    described = introspect.describe_tables(ro, ["Customers", "ghosts"])
    assert "columns" in described["customers"]
    assert "Unknown table" in described["ghosts"]["error"]


@pytest.mark.parametrize("column,tables,expected", [
    ("customer_id", ["customers"], "customers"),
    ("billing_customer_id", ["customers", "billing"], "customers"),
    ("categoryId", ["categories"], "categories"),
    ("category_id", ["categories"], "categories"),
    ("author_id", ["users"], None),
])
def test_inference_rules(column, tables, expected):
    found = introspect._infer_fks([{"name": column}], "x", tables, set(), "sqlite")
    assert (found[0].ref_table if found else None) == expected


def test_schema_overview_lists_columns(ro):
    overview = {t["name"]: t for t in introspect.schema_overview(ro)}
    assert [c["name"] for c in overview["products"]["columns"]] == ["id", "name", "category", "price"]


# ── tools ─────────────────────────────────────────────────────────────────

@pytest.fixture
def box(ro, settings):
    return ToolBox(ro, settings)


def test_tool_specs_depend_on_backend_and_settings():
    names = lambda backend, **kw: {s.name for s in tool_specs(backend, Settings(**kw))}
    assert names("sqlite") == {"list_tables", "describe_tables", "sample_rows", "run_sql"}
    assert names("mongodb") == {"list_tables", "describe_tables", "sample_rows", "run_aggregation"}
    assert "sample_rows" not in names("postgres", preview_rows=0)


def test_tool_specs_are_plain_json_schema():
    for spec in tool_specs("sqlite", Settings()):
        assert spec.parameters["type"] == "object" and "properties" in spec.parameters


def test_list_and_describe_tools(box):
    listed = box.call("list_tables", {})
    assert listed.ok and {t["name"] for t in listed.response["tables"]} >= {"customers", "orders"}
    described = box.call("describe_tables", {"tables": ["customers", "orders"]})
    assert described.ok and set(described.response["tables"]) == {"customers", "orders"}
    assert box.call("describe_tables", {"tables": "customers"}).ok            # a bare string is tolerated


def test_sample_rows_masks_redacted_columns_and_clamps_limit(box):
    out = box.call("sample_rows", {"table": "customers", "limit": 99})
    assert out.ok and len(out.response["rows"]) == 5                          # clamped to preview_rows (5)
    assert all(r["email"] == "[redacted]" for r in out.response["rows"])
    assert all(r["name"] != "[redacted]" for r in out.response["rows"])


def test_run_sql_tool_returns_a_model_view_and_a_result(box):
    out = box.call("run_sql", {"sql": "SELECT name, country FROM customers ORDER BY id LIMIT 12"})
    assert out.ok and out.result is not None and out.result.row_count == 12
    assert out.response["rows_shown"] == 5 and out.response["row_count"] == 12      # preview_rows = 5
    assert "12 rows" in out.summary


@pytest.mark.parametrize("sql", ["DELETE FROM customers", "DROP TABLE orders", "SELECT 1; DROP TABLE orders"])
def test_run_sql_tool_reports_guard_rejections_as_errors(box, sql):
    out = box.call("run_sql", {"sql": sql})
    assert not out.ok and out.result is None
    assert "read-only" in out.response["error"].lower()


def test_run_sql_tool_reports_database_errors(box):
    out = box.call("run_sql", {"sql": "SELECT missing_col FROM customers"})
    assert not out.ok and "database returned an error" in out.response["error"]


@pytest.mark.parametrize("name,args", [
    ("nope", {}), ("run_sql", {}), ("run_sql", {"sql": 5}), ("describe_tables", {"tables": []}),
    ("describe_tables", {"tables": [1]}), ("sample_rows", {"table": "customers", "limit": "many"}),
    ("sample_rows", {"table": "ghosts"}), ("run_aggregation", {"collection": "x", "pipeline_json": "[]"}),
])
def test_bad_calls_return_errors_instead_of_raising(box, name, args):
    out = box.call(name, args)
    assert not out.ok and "error" in out.response


def test_tools_disabled_by_settings_are_not_callable(ro):
    box = ToolBox(ro, Settings(preview_rows=0))
    assert not box.call("sample_rows", {"table": "customers"}).ok
