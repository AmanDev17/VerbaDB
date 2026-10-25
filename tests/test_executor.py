from __future__ import annotations

import datetime as dt
import decimal
import json
import time
import uuid

import pytest
from pyflashdb import FlashDB

from verbadb import demo_data, executor
from verbadb.config import Settings
from verbadb.executor import QueryFailed, QueryResult, jsonable, model_view, run_sql
from verbadb.guard import GuardError

from conftest import TODAY


# ── Connection hardening ──────────────────────────────────────────────────

def test_sqlite_session_is_verified_read_only(ro):
    assert ro.enforcement.guard
    assert ro.enforcement.session_read_only == "verified"
    assert ro.enforcement.statement_timeout == "set"


def test_a_runaway_query_is_stopped_and_the_connection_survives(ro):
    # An unbounded recursive CTE never finishes and SQLite has no timeout of its own.
    runaway = "WITH RECURSIVE t(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM t) SELECT COUNT(*) AS n FROM t"
    started = time.monotonic()
    with pytest.raises(QueryFailed, match="time limit"):
        run_sql(ro, runaway, Settings(query_timeout=1, max_rows=10))
    assert time.monotonic() - started < 6
    assert run_sql(ro, "SELECT COUNT(*) AS n FROM customers", Settings(query_timeout=5)).rows == [{"n": 40}]


def test_the_deadline_does_not_outlive_the_query(ro):
    run_sql(ro, "SELECT 1 AS a", Settings(query_timeout=1))
    time.sleep(1.2)                              # that deadline has now passed...
    assert ro.count("customers") == 40           # ...but later statements on the connection are not aborted


def test_hook_blocks_writes_on_the_agent_connection(ro):
    with pytest.raises(GuardError):
        ro.raw("DELETE FROM customers")
    with pytest.raises(GuardError):
        ro.add("customers", {"name": "Mallory"})
    assert ro.count("customers") == 40


def test_database_itself_refuses_writes_even_if_the_hook_were_bypassed(ro):
    ro.clear_hooks()                              # simulate a bug in / bypass of the guard
    with pytest.raises(Exception, match="not authorized|readonly"):      # engine authorizer first, read-only mode behind it
        ro.raw("DELETE FROM customers")
    assert ro.count("customers") == 40


# ── engine-level layer: works even if the parser-based guard is bypassed ──

def test_the_engine_refuses_dangerous_calls_that_a_parser_might_miss(ro):
    ro.clear_hooks()                              # pretend the parser guard let everything through
    for sql in (
        "SELECT load_extension('x')", "SELECT \"load_extension\"('x')", "SELECT [load_extension]('x')",
        "SELECT `readfile`('/etc/passwd')", "SELECT writefile('x', 'y')", "SELECT fts3_tokenizer('simple')",
    ):
        # Python's SQLite has no readfile/writefile at all ("no such function"); either way it cannot run.
        with pytest.raises(Exception, match="not authorized|no such function"):
            ro.raw(sql)
    for sql in ("PRAGMA query_only = OFF", "PRAGMA writable_schema = ON", "ATTACH DATABASE ':memory:' AS x",
                "CREATE TABLE t (a int)", "CREATE TEMP TABLE t (a int)", "DROP TABLE customers",
                "INSERT INTO customers (id, name) VALUES (999, 'x')", "UPDATE customers SET name = 'x'"):
        with pytest.raises(Exception, match="not authorized"):
            ro.raw(sql)
    assert ro.raw("SELECT ROUND(AVG(total), 2) AS a, strftime('%Y', MIN(ordered_at)) AS y FROM orders")[0]["y"]


def test_a_bom_inside_a_function_name_is_stopped_at_both_layers(ro, settings):
    sql = "SELECT ﻿load_extension('x')"
    with pytest.raises(GuardError, match="invisible"):
        run_sql(ro, sql, settings)                # the parser-level guard
    ro.clear_hooks()
    with pytest.raises(Exception):                # ...and the engine, which would not have run it anyway
        ro.raw(sql)


def test_the_catalog_lookups_the_app_itself_needs_still_work(ro):
    ro.clear_hooks()
    assert ro.describe("customers") and ro.count("customers") == 40 and "orders" in ro.show_tables()
    assert ro.raw('PRAGMA foreign_key_list("orders")') == []


@pytest.mark.skipif(not hasattr(__import__("sqlite3").Connection, "setlimit"), reason="needs Python 3.11+")
def test_a_statement_cannot_build_a_gigantic_value(ro, settings):
    with pytest.raises(QueryFailed, match="too big"):
        run_sql(ro, "SELECT length(zeroblob(50000000)) AS n", settings)
    with pytest.raises(QueryFailed, match="too big"):
        run_sql(ro, "SELECT length(replace(hex(zeroblob(8000000)), '0', 'abc')) AS n", settings)
    assert run_sql(ro, "SELECT length(zeroblob(1000)) AS n", settings).rows == [{"n": 1000}]


def test_huge_cells_are_cut_before_they_reach_the_browser(ro, settings):
    result = run_sql(ro, "SELECT hex(zeroblob(30000)) AS big, 'ok' AS small", settings)
    big = result.rows[0]["big"]
    assert len(big) < 20_100 and "cut: 60,000 characters" in big and result.rows[0]["small"] == "ok"


# ── redaction follows the data, not the column name ───────────────────────

def test_an_alias_does_not_defeat_redaction(ro, settings):
    result = run_sql(ro, "SELECT name, email AS contact FROM customers ORDER BY id LIMIT 3", settings)
    assert all("@" in r["contact"] for r in result.rows)                         # the user still sees real values
    view = model_view(result, settings)
    assert view["redacted_columns"] == ["contact"]
    assert all(r["contact"] == "[redacted]" and r["name"] != "[redacted]" for r in view["rows"])
    assert "@example.com" not in json.dumps(view)


def test_expressions_and_unaliased_columns_are_redacted_by_position(ro, settings):
    result = run_sql(ro, "SELECT id, email || '!' FROM customers ORDER BY id LIMIT 2", settings)
    expression_column = result.columns[1]
    view = model_view(result, settings)
    assert view["redacted_columns"] == [expression_column]
    assert [r[expression_column] for r in view["rows"]] == ["[redacted]", "[redacted]"]
    assert [r["id"] for r in view["rows"]] == [1, 2]


def test_redaction_through_a_cte_and_a_union(ro, settings):
    cte = run_sql(ro, "WITH x AS (SELECT email AS e FROM customers) SELECT e AS who FROM x LIMIT 2", settings)
    assert model_view(cte, settings)["rows"] == [{"who": "[redacted]"}, {"who": "[redacted]"}]
    union = run_sql(ro, "SELECT upper(email) AS a FROM customers WHERE id = 1 UNION ALL SELECT name FROM products WHERE id = 1", settings)
    assert {r["a"] for r in model_view(union, settings)["rows"]} == {"[redacted]"}      # one sensitive branch taints the column


def test_harmless_columns_and_counts_are_not_redacted(ro, settings):
    result = run_sql(ro, "SELECT COUNT(email) AS n, MIN(name) AS first_name FROM customers WHERE email LIKE '%@example.com'", settings)
    view = model_view(result, settings)
    assert "redacted_columns" not in view and view["rows"][0]["n"] == 40 and view["rows"][0]["first_name"]


# ── run_sql ───────────────────────────────────────────────────────────────

def test_run_sql_returns_rows_and_columns(ro, settings):
    result = run_sql(ro, "SELECT id, name FROM customers ORDER BY id LIMIT 3", settings)
    assert result.columns == ["id", "name"]
    assert [r["id"] for r in result.rows] == [1, 2, 3]
    assert not result.truncated and result.language == "sql" and result.elapsed_ms >= 0


def test_run_sql_truncates_at_max_rows(ro):
    small = Settings(max_rows=3)
    result = run_sql(ro, "SELECT id FROM customers ORDER BY id", small)
    assert result.row_count == 3 and result.truncated
    assert "LIMIT 4" in result.query


def test_run_sql_empty_result_still_reports_columns(ro, settings):
    result = run_sql(ro, "SELECT id, name FROM customers WHERE id < 0", settings)
    assert result.rows == [] and result.columns == ["id", "name"]


def test_run_sql_rejects_writes_before_touching_the_database(ro, settings):
    with pytest.raises(GuardError):
        run_sql(ro, "UPDATE customers SET name = 'x'", settings)
    assert ro.count("customers") == 40


def test_run_sql_database_errors_become_query_failed_without_sql_echo(ro, settings):
    with pytest.raises(QueryFailed) as err:
        run_sql(ro, "SELECT nope FROM customers", settings)
    assert "nope" in str(err.value) and "SQL:" not in str(err.value)


def test_percent_signs_round_trip(ro, settings):
    result = run_sql(ro, "SELECT COUNT(*) AS n FROM customers WHERE email LIKE '%@example.com'", settings)
    assert result.rows == [{"n": 40}]


def test_aggregate_query_matches_python_ground_truth(demo_url, ro, settings):
    sql = """
        SELECT c.name AS customer, ROUND(SUM(o.total), 2) AS revenue
        FROM orders o JOIN customers c ON c.id = o.customer_id
        WHERE o.status IN ('paid', 'shipped') AND o.ordered_at >= '2026-09-01' AND o.ordered_at < '2026-10-01'
        GROUP BY c.id, c.name ORDER BY revenue DESC LIMIT 5
    """
    result = run_sql(ro, sql, settings)
    with FlashDB.from_url(demo_url) as plain:
        expected = demo_data.top_customers_last_month(plain, TODAY)
    assert [(r["customer"], r["revenue"]) for r in result.rows] == expected
    assert len(expected) == 5


def test_demo_customer_names_are_unique(demo_url):
    with FlashDB.from_url(demo_url) as plain:
        names = [c["name"] for c in plain.select("customers")]
    assert len(names) == len(set(names)) == 40


# ── JSON safety ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (None, None), (True, True), (3, 3), ("x", "x"),
    (decimal.Decimal("12.50"), 12.5), (decimal.Decimal("7"), 7), (decimal.Decimal("NaN"), "NaN"),
    (float("inf"), "inf"),
    (dt.datetime(2026, 9, 1, 12, 30), "2026-09-01T12:30:00"), (dt.date(2026, 9, 1), "2026-09-01"),
    (b"abc", "<3 bytes>"), (uuid.UUID(int=1), "00000000-0000-0000-0000-000000000001"),
    ({"a": decimal.Decimal("1.5"), 2: [dt.date(2026, 1, 2)]}, {"a": 1.5, "2": ["2026-01-02"]}),
    ((1, 2), [1, 2]), (object, str(object)),
])
def test_jsonable(value, expected):
    assert jsonable(value) == expected


# ── What the model sees ───────────────────────────────────────────────────

def _result(rows, columns=None, truncated=False):
    cols = columns or list(rows[0]) if rows else (columns or [])
    return QueryResult(columns=cols, rows=rows, truncated=truncated, query="SELECT 1", elapsed_ms=1.0, language="sql")


def test_model_view_redacts_columns_and_limits_rows():
    rows = [{"name": f"n{i}", "email": f"n{i}@x.com", "total": i} for i in range(30)]
    view = model_view(_result(rows), Settings(preview_rows=5))
    assert view["rows_shown"] == 5 and view["row_count"] == 30
    assert all(r["email"] == "[redacted]" for r in view["rows"])
    assert view["redacted_columns"] == ["email"]
    assert "first 5 of 30" in view["note"]


def test_model_view_with_zero_preview_withholds_rows():
    view = model_view(_result([{"a": 1}]), Settings(preview_rows=0))
    assert view["rows"] == [] and "withheld" in view["note"]


def test_model_view_redacts_nested_keys_and_clips_long_text():
    rows = [{"user": {"Email": "a@b.c", "city": "Paris"}, "bio": "x" * 500}]
    view = model_view(_result(rows), Settings(preview_rows=5))
    assert view["rows"][0]["user"] == {"Email": "[redacted]", "city": "Paris"}
    assert len(view["rows"][0]["bio"]) <= executor.CELL_MAX_CHARS + 1


def test_model_view_shrinks_to_fit_the_size_cap():
    rows = [{"blob": "y" * 190, "n": i} for i in range(200)]
    view = model_view(_result(rows), Settings(preview_rows=200))
    assert 0 < view["rows_shown"] < 200
