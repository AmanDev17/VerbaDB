from __future__ import annotations

import datetime as dt
import time

import pytest

from verbadb import guard
from verbadb.config import Settings
from verbadb.guard import GuardError, check_statement, prepare_pipeline, prepare_sql, trusted_scope

MAX = 50


# ── SQL: accepted ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("backend", ["sqlite", "mysql", "postgres"])
@pytest.mark.parametrize("sql", [
    "SELECT * FROM orders",
    "select c.name, sum(o.total) as revenue from customers c join orders o on o.customer_id = c.id "
    "group by c.name order by revenue desc limit 5",
    "WITH x AS (SELECT 1 AS a) SELECT * FROM x",
    "SELECT 1 UNION SELECT 2",
    "(SELECT 1)",
    "SELECT 1;",
    "SELECT name FROM customers WHERE name LIKE '%an%'",
])
def test_selects_are_accepted(backend, sql):
    prepared = prepare_sql(sql, backend, MAX)
    assert prepared.sql.upper().startswith(("SELECT", "WITH", "("))


# ── SQL: rejected ─────────────────────────────────────────────────────────

WRITES = [
    "DROP TABLE orders",
    "DELETE FROM orders",
    "UPDATE orders SET total = 0",
    "INSERT INTO orders (id) VALUES (1)",
    "CREATE TABLE x (a int)",
    "ALTER TABLE orders ADD COLUMN c int",
    "TRUNCATE TABLE orders",
    "SELECT 1; DROP TABLE orders",
    "SELECT 1; SELECT 2",
    "SET x = 1",
    "this is not sql at all",
]


@pytest.mark.parametrize("backend", ["sqlite", "mysql", "postgres"])
@pytest.mark.parametrize("sql", WRITES)
def test_writes_and_junk_are_rejected(backend, sql):
    with pytest.raises(GuardError):
        prepare_sql(sql, backend, MAX)


@pytest.mark.parametrize("sql,backend", [
    ("PRAGMA query_only = OFF", "sqlite"),
    ("ATTACH DATABASE 'x.db' AS y", "sqlite"),
    ("VACUUM", "sqlite"),
    ("WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x", "postgres"),
    ("WITH x AS (INSERT INTO orders (id) VALUES (1) RETURNING *) SELECT * FROM x", "postgres"),
    ("SELECT * INTO newt FROM orders", "postgres"),
    ("SELECT * FROM orders FOR UPDATE", "postgres"),
    ("SELECT * FROM orders FOR SHARE", "postgres"),
    ("COPY orders TO '/tmp/x'", "postgres"),
    ("SELECT * FROM orders INTO OUTFILE '/tmp/x'", "mysql"),
    ("REPLACE INTO orders (id) VALUES (1)", "mysql"),
    ("SHOW TABLES", "mysql"),
    ("EXPLAIN SELECT 1", "postgres"),
])
def test_dialect_specific_writes_are_rejected(sql, backend):
    with pytest.raises(GuardError):
        prepare_sql(sql, backend, MAX)


@pytest.mark.parametrize("sql,backend", [
    ("SELECT pg_sleep(10)", "postgres"),
    ("SELECT pg_catalog.pg_sleep(10)", "postgres"),
    ("SELECT * FROM orders WHERE pg_sleep(1) IS NULL", "postgres"),
    ("SELECT pg_read_file('/etc/passwd')", "postgres"),
    ("SELECT nextval('seq')", "postgres"),
    ("SELECT query_to_xml('delete from orders', true, true, '')", "postgres"),
    ("SELECT SLEEP(5)", "mysql"),
    ("SELECT BENCHMARK(100000000, MD5('x'))", "mysql"),
    ("SELECT load_file('/etc/passwd')", "mysql"),
    ("SELECT load_extension('x')", "sqlite"),
    ("SELECT readfile('/etc/passwd')", "sqlite"),
])
def test_dangerous_functions_are_rejected(sql, backend):
    with pytest.raises(GuardError, match="not allowed"):
        prepare_sql(sql, backend, MAX)


@pytest.mark.parametrize("sql,backend", [
    ("SELECT * FROM pg_shadow", "postgres"),
    ("SELECT * FROM pg_catalog.pg_authid", "postgres"),
    ("SELECT * FROM mysql.user", "mysql"),
])
def test_credential_tables_are_rejected(sql, backend):
    with pytest.raises(GuardError, match="system table"):
        prepare_sql(sql, backend, MAX)


@pytest.mark.parametrize("bad", [None, "", "   ", 42, ["SELECT 1"]])
def test_non_string_or_empty_input_is_rejected(bad):
    with pytest.raises(GuardError):
        prepare_sql(bad, "sqlite", MAX)


def test_unsupported_backend_is_rejected():
    with pytest.raises(GuardError):
        prepare_sql("SELECT 1", "mongodb", MAX)


def test_overlong_sql_is_rejected():
    with pytest.raises(GuardError, match="too long"):
        prepare_sql("SELECT " + "1," * guard.MAX_SQL_CHARS + "1", "sqlite", MAX)


def test_mysql_executable_comment_cannot_smuggle_code():
    prepared = prepare_sql("SELECT 1 /*! INTO OUTFILE '/tmp/x' */", "mysql", MAX)
    assert "OUTFILE" not in prepared.sql.upper()
    assert "/*" not in prepared.sql


# ── review findings: quoting, invisible characters, hints, parentheses ────

@pytest.mark.parametrize("sql,backend", [
    # A quoted name is an Identifier, not a string, in the parse tree: it used to skip the denylist entirely.
    ("SELECT pg_catalog.\"set_config\"('default_transaction_read_only', 'off', false)", "postgres"),
    ("SELECT \"set_config\"('statement_timeout', '0', false)", "postgres"),
    ("SELECT public.\"dblink_exec\"('host=x', 'delete from t')", "postgres"),
    ("SELECT pg_catalog.\"pg_read_file\"('/etc/passwd')", "postgres"),
    ("SELECT \"pg_sleep\"(5)", "postgres"),
    ("SELECT pg_catalog.\"nextval\"('seq')", "postgres"),
    ("SELECT `sleep`(5)", "mysql"), ("SELECT `load_file`('/etc/passwd')", "mysql"),
    ("SELECT `benchmark`(100000, md5('x'))", "mysql"), ("SELECT `sys_exec`('id')", "mysql"),
    ("SELECT \"load_extension\"('x')", "sqlite"), ("SELECT [load_extension]('x')", "sqlite"),
    ("SELECT `readfile`('/etc/passwd')", "sqlite"), ("SELECT \"Fts3_Tokenizer\"('simple')", "sqlite"),
])
def test_quoted_function_names_do_not_evade_the_denylist(sql, backend):
    with pytest.raises(GuardError, match="not allowed"):
        prepare_sql(sql, backend, MAX)


@pytest.mark.parametrize("sql,backend", [
    ("SELECT lo_creat(-1)", "postgres"), ("SELECT lo_from_bytea(0, 'x')", "postgres"), ("SELECT lowrite(1, 'x')", "postgres"),
    ("SELECT pg_advisory_lock_shared(1)", "postgres"), ("SELECT pg_try_advisory_xact_lock(1)", "postgres"),
    ("SELECT pg_advisory_unlock_all()", "postgres"), ("SELECT pg_notify('c', 'p')", "postgres"),
    ("SELECT pg_logical_slot_get_changes('s', NULL, NULL)", "postgres"), ("SELECT pg_logical_emit_message(true, 'p', 'm')", "postgres"),
    ("SELECT pg_replication_slot_advance('s', '0/0')", "postgres"), ("SELECT pg_drop_replication_slot('s')", "postgres"),
    ("SELECT pg_copy_logical_replication_slot('a', 'b')", "postgres"), ("SELECT pg_create_physical_replication_slot('s')", "postgres"),
    ("SELECT pg_promote()", "postgres"), ("SELECT pg_wal_replay_pause()", "postgres"), ("SELECT pg_backup_start('l')", "postgres"),
    ("SELECT pg_stat_reset()", "postgres"), ("SELECT pg_stat_statements_reset()", "postgres"),
    ("SELECT pg_ls_tmpdir()", "postgres"), ("SELECT pg_ls_archive_statusdir()", "postgres"), ("SELECT pg_file_write('f', 'x', false)", "postgres"),
    ("SELECT * FROM ts_stat('select * from pg_authid')", "postgres"), ("SELECT query_to_xmlschema('select 1', true, true, '')", "postgres"),
    ("SELECT database_to_xml(true, true, '')", "postgres"), ("SELECT schema_to_xml('public', true, true, '')", "postgres"),
    ("SELECT * FROM crosstab('select 1, 2')", "postgres"), ("SELECT xpath_table('a', 'b', 'c', 'd', 'e')", "postgres"),
    ("SELECT WAIT_FOR_EXECUTED_GTID_SET('x')", "mysql"), ("SELECT WAIT_UNTIL_SQL_THREAD_AFTER_GTIDS('x')", "mysql"),
])
def test_function_families_are_denied(sql, backend):
    with pytest.raises(GuardError, match="not allowed"):
        prepare_sql(sql, backend, MAX)


@pytest.mark.parametrize("sql", [
    "SELECT * FROM pg_stat_activity", "SELECT * FROM pg_catalog.pg_stat_statements", "SELECT * FROM pg_subscription",
    "SELECT * FROM pg_user_mappings",
])
def test_more_sensitive_system_views_are_denied(sql):
    with pytest.raises(GuardError, match="system table"):
        prepare_sql(sql, "postgres", MAX)


@pytest.mark.parametrize("char", ["﻿", "\x00", "​", "‍", "­", "‮", "\x07", "\x1b", "⁠"])
@pytest.mark.parametrize("backend", ["sqlite", "mysql", "postgres"])
def test_invisible_and_control_characters_are_rejected(backend, char):
    for sql in (f"SELECT {char}load_extension('x')", f"SELECT 1{char}", f"SELECT '{char}'"):
        with pytest.raises(GuardError, match="invisible or control character"):
            prepare_sql(sql, backend, MAX)


@pytest.mark.parametrize("sql", [
    "SELECT\n\t*\r\nFROM customers WHERE name = 'José Müller'", "SELECT * FROM t WHERE city = '東京'",
    "SELECT * FROM t WHERE s = 'café ✓ \U0001F600'",
])
def test_ordinary_whitespace_and_unicode_text_are_fine(sql):
    assert prepare_sql(sql, "sqlite", MAX).sql.upper().startswith("SELECT")


def test_the_hook_also_rejects_invisible_characters():
    with pytest.raises(GuardError, match="invisible"):
        check_statement({"backend": "sqlite", "statement": "SELECT ﻿abs(1)", "params": [], "table": None})


@pytest.mark.parametrize("backend", ["mysql", "postgres", "sqlite"])
@pytest.mark.parametrize("hint", ["SET_VAR(max_execution_time=0)", "MAX_EXECUTION_TIME(0)", "SET_VAR(sql_mode='')"])
def test_optimizer_hints_are_removed(backend, hint):
    prepared = prepare_sql(f"SELECT /*+ {hint} */ * FROM t", backend, MAX)
    assert "/*" not in prepared.sql and "SET_VAR" not in prepared.sql.upper() and "EXECUTION_TIME" not in prepared.sql.upper()


@pytest.mark.parametrize("sql", [
    "(SELECT a FROM t UNION SELECT a FROM u) ORDER BY a DESC LIMIT 7",
    "(SELECT a FROM t) ORDER BY a",
    "WITH c AS (SELECT 1 AS a) (SELECT * FROM c) ORDER BY a",
    "(SELECT a FROM t) LIMIT 5 OFFSET 2",
])
@pytest.mark.parametrize("backend", ["sqlite", "mysql", "postgres"])
def test_parenthesised_queries_with_outer_clauses_are_rejected_not_silently_rewritten(backend, sql):
    with pytest.raises(GuardError, match="parenthesised query"):
        prepare_sql(sql, backend, MAX)


@pytest.mark.parametrize("backend", ["sqlite", "mysql", "postgres"])
def test_parentheses_without_outer_clauses_and_inner_clauses_still_work(backend):
    assert prepare_sql("(SELECT 1 AS a)", backend, MAX).sql.upper().startswith("SELECT")
    inner = prepare_sql("(SELECT a FROM t ORDER BY a LIMIT 3)", backend, MAX).sql.upper()
    assert "ORDER BY" in inner and "LIMIT 3" in inner


# ── which outputs carry redacted data ─────────────────────────────────────

REDACT = Settings().is_redacted


def sensitivity(sql, backend="sqlite"):
    prepared = prepare_sql(sql, backend, MAX, REDACT)
    return set(prepared.sensitive_positions), set(prepared.sensitive_names), prepared.projection_count


@pytest.mark.parametrize("sql,positions,names", [
    ("SELECT email AS contact FROM customers", {0}, {"contact"}),                                  # alias
    ("SELECT name, email || '' FROM customers", {1}, set()),                                       # unaliased expression
    ("SELECT lower(email) AS e, name FROM customers", {0}, {"e"}),
    ("SELECT MAX(email) AS m FROM customers", {0}, {"m"}),                                         # still an email
    ("SELECT name, phone AS p FROM customers", {1}, {"p"}),
    ("SELECT name FROM customers WHERE email LIKE '%@x.io'", set(), set()),                        # only used to filter
    ("SELECT COUNT(email) AS n FROM customers", set(), set()),                                     # a number, not an email
    ("SELECT name, country FROM customers", set(), set()),
    ("WITH x AS (SELECT email AS e FROM customers) SELECT e AS who, 1 AS one FROM x", {0}, {"e", "who"}),   # through a CTE
    ("SELECT w FROM (SELECT email AS w FROM customers) AS d", {0}, {"w"}),                         # through a subquery
    ("SELECT upper(email) AS a FROM customers UNION SELECT name FROM products", {0}, {"a"}),       # any branch of a union
    ("SELECT name FROM products UNION ALL SELECT email FROM customers", {0}, {"email"}),          # a plain email column
])
def test_redaction_follows_the_data_not_the_output_name(sql, positions, names):
    got_positions, got_names, _ = sensitivity(sql)
    assert got_positions == positions and got_names == names


def test_select_star_is_covered_by_name_and_has_no_positions():
    positions, names, _ = sensitivity("SELECT * FROM customers")
    assert positions == set() and names == set()
    positions, _, _ = sensitivity("SELECT c.*, email AS e FROM customers c")
    assert positions == set()                                     # positions are unreliable after a star


def test_a_custom_redaction_list_is_used():
    only_iban = Settings(redact_columns=("iban",)).is_redacted
    assert prepare_sql("SELECT iban AS x, email AS y FROM t", "sqlite", MAX, only_iban).sensitive_positions == {0}


# ── SQL: row limits ───────────────────────────────────────────────────────

def test_limit_is_injected():
    prepared = prepare_sql("SELECT * FROM orders", "sqlite", MAX)
    assert prepared.limit_injected and f"LIMIT {MAX + 1}" in prepared.sql


def test_small_limit_is_kept():
    prepared = prepare_sql("SELECT * FROM orders LIMIT 10", "sqlite", MAX)
    assert not prepared.limit_injected and "LIMIT 10" in prepared.sql


def test_large_limit_is_lowered():
    prepared = prepare_sql("SELECT * FROM orders LIMIT 100000", "sqlite", MAX)
    assert prepared.limit_injected and f"LIMIT {MAX + 1}" in prepared.sql and "100000" not in prepared.sql


def test_union_gets_a_limit():
    prepared = prepare_sql("SELECT 1 AS a UNION SELECT 2", "sqlite", MAX)
    assert f"LIMIT {MAX + 1}" in prepared.sql


def test_fetch_first_is_wrapped():
    prepared = prepare_sql("SELECT * FROM orders FETCH FIRST 5 ROWS ONLY", "postgres", MAX)
    assert f"LIMIT {MAX + 1}" in prepared.sql


def test_columns_are_reported_for_empty_results():
    prepared = prepare_sql("SELECT name, country AS c FROM customers WHERE 1 = 0", "sqlite", MAX)
    assert prepared.columns == ["name", "c"]


# ── MongoDB pipelines ─────────────────────────────────────────────────────

COLLS = ["orders", "customers"]


def test_pipeline_is_accepted_and_limited():
    prepared = prepare_pipeline(
        "orders", '[{"$match": {"status": "paid"}}, {"$group": {"_id": "$customer_id", "t": {"$sum": "$total"}}}]',
        MAX, COLLS,
    )
    assert prepared.pipeline[-1] == {"$limit": MAX + 1}
    assert f'"$limit": {MAX + 1}' in prepared.display


def test_existing_small_limit_is_kept():
    prepared = prepare_pipeline("orders", '[{"$sort": {"total": -1}}, {"$limit": 5}]', MAX, COLLS)
    assert prepared.pipeline[-1] == {"$limit": 5}


def test_extended_json_dates_are_decoded():
    prepared = prepare_pipeline(
        "orders", '[{"$match": {"ordered_at": {"$gte": {"$date": "2026-09-01T00:00:00Z"}}}}]', MAX, COLLS
    )
    assert isinstance(prepared.pipeline[0]["$match"]["ordered_at"]["$gte"], dt.datetime)


@pytest.mark.parametrize("pipeline", [
    '[{"$out": "stolen"}]',
    '[{"$merge": {"into": "orders"}}]',
    '[{"$match": {"$where": "sleep(1000)"}}]',
    '[{"$addFields": {"x": {"$function": {"body": "function(){}", "args": [], "lang": "js"}}}}]',
    '[{"$group": {"_id": 1, "x": {"$accumulator": {"init": "function(){}"}}}}]',
    '[{"$lookup": {"from": "customers", "pipeline": [{"$out": "x"}], "as": "c"}}]',
    '[{"$facet": {"a": [{"$match": {}}], "b": [{"$merge": {"into": "x"}}]}}]',
    '[{"$currentOp": {}}]',
])
def test_write_stages_and_javascript_are_rejected_even_when_nested(pipeline):
    with pytest.raises(GuardError, match="not allowed"):
        prepare_pipeline("orders", pipeline, MAX, COLLS)


@pytest.mark.parametrize("pipeline,message", [
    ("not json", "not valid JSON"),
    ("{}", "non-empty JSON array"),
    ("[]", "non-empty JSON array"),
    ('[{"match": {}}]', "one \\$-operator"),
    ('[{"$match": {}, "$sort": {}}]', "one \\$-operator"),
    ('["$match"]', "one \\$-operator"),
])
def test_malformed_pipelines_are_rejected(pipeline, message):
    with pytest.raises(GuardError, match=message):
        prepare_pipeline("orders", pipeline, MAX, COLLS)


def test_unknown_collection_is_rejected():
    with pytest.raises(GuardError, match="Unknown collection"):
        prepare_pipeline("secrets", "[{\"$match\": {}}]", MAX, COLLS)


def test_too_many_stages_are_rejected():
    stages = ",".join(['{"$match": {}}'] * (guard.MAX_PIPELINE_STAGES + 1))
    with pytest.raises(GuardError, match="too many stages"):
        prepare_pipeline("orders", f"[{stages}]", MAX, COLLS)


@pytest.mark.parametrize("pipeline", [
    '[{"$lookup": {"from": "system.users", "localField": "a", "foreignField": "b", "as": "x"}}]',
    '[{"$lookup": {"from": "System.Profile", "pipeline": [], "as": "x"}}]',
    '[{"$lookup": {"from": {"db": "admin", "coll": "users"}, "pipeline": [], "as": "x"}}]',
    '[{"$unionWith": "system.profile"}]',
    '[{"$unionWith": {"coll": "system.views", "pipeline": []}}]',
    '[{"$unionWith": {"coll": {"db": "admin", "coll": "x"}}}]',
    '[{"$graphLookup": {"from": "system.users", "startWith": "$a", "connectFromField": "a", "connectToField": "b", "as": "x"}}]',
    '[{"$facet": {"x": [{"$lookup": {"from": "system.users", "pipeline": [], "as": "y"}}]}}]',
    '[{"$collStats": {"storageStats": {}}}]', '[{"$indexStats": {}}]', '[{"$queryStats": {}}]',
    '[{"$listCatalog": {}}]', '[{"$listSampledQueries": {}}]', '[{"$shardedDataDistribution": {}}]',
])
def test_system_collections_cross_database_lookups_and_introspection_stages_are_rejected(pipeline):
    with pytest.raises(GuardError):
        prepare_pipeline("orders", pipeline, MAX, COLLS)


def test_normal_lookups_and_unions_between_collections_are_fine():
    ok = prepare_pipeline(
        "orders",
        '[{"$lookup": {"from": "customers", "localField": "customer_id", "foreignField": "id", "as": "c"}},'
        ' {"$unionWith": "customers"}, {"$unionWith": {"coll": "orders", "pipeline": [{"$match": {}}]}}]', MAX, COLLS)
    assert ok.pipeline[-1] == {"$limit": MAX + 1}


@pytest.mark.parametrize("pipeline,needle", [
    pytest.param('[{"$match": {"_id": {"$oid": "zzz"}}}]', "Could not decode", id="invalid-oid"),
    pytest.param('[{"$match": {"d": {"$date": "not a date"}}}]', "Could not decode", id="invalid-date"),
    pytest.param("[" * 100_000 + "]" * 100_000, "nested too deeply", id="100k-nesting"),
    pytest.param('[{"$match": {}}]' + " " * guard.MAX_PIPELINE_CHARS, "too long", id="oversized"),
])
def test_bad_extended_json_and_pathological_input_become_guard_errors(pipeline, needle):
    with pytest.raises(GuardError, match=needle):                # never a raw bson / recursion error
        prepare_pipeline("orders", pipeline, MAX, COLLS)


def test_the_hook_scans_find_and_count_filters_too():
    for op in ("find", "count_documents"):
        with pytest.raises(GuardError, match="not allowed"):
            check_statement({"backend": "mongodb", "statement": f"orders.{op}", "table": "orders",
                             "params": {"filter": {"$where": "sleep(1000)"}}})
        check_statement({"backend": "mongodb", "statement": f"orders.{op}", "table": "orders",
                         "params": {"filter": {"status": "paid"}}})


# ── statement deadline ────────────────────────────────────────────────────

def test_the_deadline_clock_starts_when_the_statement_starts_running():
    deadline = guard.Deadline(0.05)
    time.sleep(0.08)                                      # e.g. waiting for the pooled connection
    assert deadline.expired is False                     # not running yet: this wait is not charged to the query
    assert deadline.tick() is False and deadline.started is not None
    time.sleep(0.08)
    assert deadline.tick() is True and deadline.expired is True


# ── The before_query hook ─────────────────────────────────────────────────

@pytest.mark.parametrize("statement", [
    'SELECT COUNT(*) AS "count" FROM "orders"',
    'SELECT * FROM "orders" WHERE "status" = %s LIMIT 3',
    "SELECT * FROM orders WHERE total > ?",
])
def test_hook_allows_reads(statement):
    check_statement({"backend": "sqlite", "statement": statement, "params": [], "table": None})


@pytest.mark.parametrize("statement", [
    "DELETE FROM orders", "INSERT INTO orders (id) VALUES (%s)", "DROP TABLE orders", "SELECT 1; SELECT 2",
])
def test_hook_blocks_writes(statement):
    with pytest.raises(GuardError):
        check_statement({"backend": "postgres", "statement": statement, "params": [], "table": None})


def test_hook_trusted_scope_lets_catalog_statements_through():
    info = {"backend": "sqlite", "statement": 'PRAGMA foreign_key_list("orders")', "params": [], "table": None}
    with pytest.raises(GuardError):
        check_statement(info)
    with trusted_scope():
        check_statement(info)
    with pytest.raises(GuardError):       # and the exemption ends with the block
        check_statement(info)


# Names as pyflashdb's Mongo adapter really reports them (see MongoAdapter._traced calls).
@pytest.mark.parametrize("op", ["find", "count_documents", "aggregate"])
def test_hook_allows_mongo_reads(op):
    check_statement({"backend": "mongodb", "statement": f"orders.{op}", "table": "orders",
                     "params": {"pipeline": [{"$match": {}}]} if op == "aggregate" else {}})


@pytest.mark.parametrize("op", ["insert_one", "insert_many", "update_many", "delete_many", "bulk_write",
                                "drop", "create_index", "count", "insert"])
def test_hook_blocks_mongo_writes_and_unknown_operations(op):
    with pytest.raises(GuardError):
        check_statement({"backend": "mongodb", "statement": f"orders.{op}", "table": "orders", "params": {}})


def test_hook_blocks_mongo_database_commands_and_write_stages():
    with pytest.raises(GuardError):
        check_statement({"backend": "mongodb", "statement": "<database>.raw", "table": "<database>",
                         "params": {"command": "dropDatabase"}})
    with pytest.raises(GuardError):
        check_statement({"backend": "mongodb", "statement": "orders.aggregate", "table": "orders",
                         "params": {"pipeline": [{"$out": "x"}]}})
