from __future__ import annotations

import pytest

from verbadb import sandbox
from verbadb.config import Settings
from verbadb.guard import GuardError
from verbadb.sandbox import (
    MAX_SNAPSHOTS, ChangeFailed, Snapshots, apply_changes, capture_state, diff_states, open_sandbox, prepare_writes,
)

SETUP = [
    "CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
    "CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT NOT NULL, "
    "author_id INTEGER REFERENCES authors(id) ON DELETE CASCADE, price REAL)",
    "INSERT INTO authors (id, name) VALUES (1, 'Ada'), (2, 'Grace')",
    "INSERT INTO books (id, title, author_id, price) VALUES (1, 'A1', 1, 10.0), (2, 'A2', 1, 12.5), (3, 'G1', 2, 20.0)",
]


class Box:
    """A sandbox database plus its snapshot stack, as the manager wires them."""

    def __init__(self, tmp_path, settings):
        self.path = tmp_path / "sb.db"
        self.settings = settings
        self.snaps = Snapshots(tmp_path / "snaps", self.path)
        self.flash, self.enforcement = open_sandbox(self.path, settings)

    def apply(self, *statements):
        return apply_changes(self.flash, list(statements), self.settings, self.snaps)

    def reopen(self):
        self.flash.close()
        self.flash, self.enforcement = open_sandbox(self.path, self.settings)


@pytest.fixture
def box(tmp_path):
    b = Box(tmp_path, Settings(query_timeout=5, sandbox_max_mb=20))
    yield b
    b.flash.close()


@pytest.fixture
def seeded(box):
    box.apply(*SETUP)
    return box


def table(change, name):
    return next(t for t in change.tables if t["name"] == name)


# ── write guard ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("sql", [
    "INSERT INTO t (a, b) VALUES (1, 'x'), (2, 'y')",
    "UPDATE t SET a = a + 1 WHERE b = 'x'",
    "DELETE FROM t",
    "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, owner INTEGER REFERENCES o(id))",
    "CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY)",
    "CREATE INDEX idx_t_name ON t (name)",
    "CREATE UNIQUE INDEX idx_t_name ON t (name)",
    "ALTER TABLE t ADD COLUMN extra TEXT DEFAULT 'n/a'",
    "ALTER TABLE t RENAME TO t2",
    "DROP TABLE t",
    "DROP TABLE IF EXISTS t",
    "DROP INDEX idx_t_name",
    "CREATE TABLE copy AS SELECT * FROM t",
    "INSERT INTO t (a) SELECT a FROM u WHERE a > 3",
    "INSERT INTO t (a) VALUES (1) -- trailing comment",
    "INSERT OR REPLACE INTO t (id, a) VALUES (1, 2)",
    "INSERT OR IGNORE INTO t (id, a) VALUES (1, 2)",
])
def test_normal_writes_are_accepted(sql):
    (prepared,) = prepare_writes([sql])
    assert prepared.verb and "--" not in prepared.sql


@pytest.mark.parametrize("sql,needle", [
    ("SELECT * FROM t", "run_sql"),
    ("WITH x AS (SELECT 1) SELECT * FROM x", "run_sql"),
    ("PRAGMA query_only = OFF", "not allowed"),
    ("PRAGMA writable_schema = ON", "not allowed"),
    ("ATTACH DATABASE 'other.db' AS o", "not allowed"),
    ("DETACH DATABASE o", "not allowed"),
    ("VACUUM", "cannot verify"),
    ("REPLACE INTO t (id, a) VALUES (1, 2)", "cannot verify"),     # not understood by the parser -> refused
    ("CREATE TABLE t (a int) WITHOUT ROWID", "cannot verify"),
    ("BEGIN", "not allowed"),
    ("COMMIT", "not allowed"),
    ("ANALYZE", "not allowed"),
    ("CREATE VIEW v AS SELECT 1", "views, triggers"),
    ("CREATE TRIGGER tr AFTER INSERT ON t BEGIN DELETE FROM t; END", "Exactly one"),   # splits on the inner ';'
    ("CREATE TRIGGER tr AFTER INSERT ON t FOR EACH ROW WHEN 1 BEGIN SELECT 1 END", "cannot verify|not allowed"),
    ("CREATE VIRTUAL TABLE ft USING fts5(x)", "Virtual and temporary"),
    ("CREATE TEMP TABLE scratch (a int)", "Virtual and temporary"),
    ("CREATE TEMPORARY TABLE scratch (a int)", "Virtual and temporary"),
    ("DROP VIEW v", "views, triggers"),
    ("INSERT INTO sqlite_master (name) VALUES ('x')", "reserved"),
    ("UPDATE sqlite_master SET sql = ''", "reserved"),
    ("DELETE FROM sqlite_sequence", "reserved"),
    ("CREATE TABLE sqlite_evil (a int)", "reserved"),
    ("INSERT INTO t (a) VALUES (load_extension('evil'))", "not allowed"),
    ("INSERT INTO t (a) SELECT readfile('/etc/passwd')", "not allowed"),
    ("INSERT INTO t (a) VALUES (\"load_extension\"('evil'))", "not allowed"),          # quoted name
    ("INSERT INTO t (a) VALUES ([READFILE]('/etc/passwd'))", "not allowed"),
    ("INSERT INTO t (a) VALUES (﻿abs(1))", "invisible or control"),
    ("INSERT INTO t (a) VALUES (1)\x00", "invisible or control"),
    ("INSERT INTO t VALUES (1); DROP TABLE t", "Exactly one"),
    ("", "empty"),
    ("this is not sql", "not allowed|parse"),
])
def test_everything_else_is_rejected(sql, needle):
    with pytest.raises(GuardError, match=needle):
        prepare_writes([sql])


@pytest.mark.parametrize("bad", [None, "INSERT INTO t VALUES (1)", [], [1], ["ok", None], {"a": 1}])
def test_batch_shape_is_validated(bad):
    with pytest.raises(GuardError, match="non-empty list"):
        prepare_writes(bad)


def test_batch_size_is_limited_and_errors_name_the_statement():
    with pytest.raises(GuardError, match="Too many"):
        prepare_writes(["DELETE FROM t"] * (sandbox.MAX_STATEMENTS + 1))
    with pytest.raises(GuardError, match="Statement 2:"):
        prepare_writes(["DELETE FROM t", "PRAGMA x = 1"])
    with pytest.raises(GuardError, match="too long"):
        prepare_writes(["INSERT INTO t VALUES ('" + "x" * sandbox.MAX_STATEMENT_CHARS + "')"])


# ── applying changes and simulating them ──────────────────────────────────

def test_creating_tables_and_rows_is_reported(box):
    change = box.apply(*SETUP)
    assert [t["name"] for t in change.tables] == ["authors", "books"]
    authors, books = table(change, "authors"), table(change, "books")
    assert authors["change"] == "created" and authors["after_count"] == 2 and authors["inserted_total"] == 2
    assert books["columns"] == ["id", "title", "author_id", "price"] and books["inserted_total"] == 3
    assert change.summary == "Applied 4 statements: created 2 tables · +5 rows"
    assert change.verbs == ["CREATE TABLE", "CREATE TABLE", "INSERT", "INSERT"]
    assert change.snapshot_id and box.snaps.depth() == 1
    event = change.to_event()
    assert event["type"] == "change" and event["can_undo"] and len(event["tables"]) == 2


def test_an_update_shows_old_and_new_values_for_the_changed_cells_only(seeded):
    change = seeded.apply("UPDATE books SET price = price * 2 WHERE author_id = 1")
    assert [t["name"] for t in change.tables] == ["books"]                      # authors is untouched
    books = change.tables[0]
    assert books["change"] == "modified" and books["updated_total"] == 2 and books["inserted_total"] == 0
    assert {(u["before"]["price"], u["after"]["price"]) for u in books["updated"]} == {(10.0, 20.0), (12.5, 25.0)}
    assert all(u["changed"] == ["price"] and u["after"]["title"] in ("A1", "A2") for u in books["updated"])
    assert change.rowcounts == [2] and "2 rows edited" in change.summary


def test_a_delete_reports_cascades_into_other_tables(seeded):
    change = seeded.apply("DELETE FROM authors WHERE id = 1")
    authors, books = table(change, "authors"), table(change, "books")
    assert authors["deleted_total"] == 1 and authors["deleted"][0]["name"] == "Ada"
    assert books["deleted_total"] == 2 and {r["title"] for r in books["deleted"]} == {"A1", "A2"}   # side effect
    assert change.summary == "Applied 1 statement: −3 rows"


def test_inserts_into_an_existing_table(seeded):
    change = seeded.apply("INSERT INTO authors (id, name) VALUES (3, 'Linus')")
    (authors,) = change.tables
    assert authors["change"] == "modified" and authors["inserted"] == [{"id": 3, "name": "Linus"}]
    assert (authors["before_count"], authors["after_count"]) == (2, 3)


def test_altering_and_dropping(seeded):
    altered = seeded.apply("ALTER TABLE books ADD COLUMN stock INTEGER DEFAULT 0")
    books = altered.tables[0]
    assert books["change"] == "altered" and books["columns_added"] == ["stock"] and "altered 1 table" in altered.summary
    dropped = seeded.apply("DROP TABLE books")
    books = dropped.tables[0]
    assert books["change"] == "dropped" and books["deleted_total"] == 3 and books["before_count"] == 3
    assert "dropped 1 table" in dropped.summary


def test_a_change_that_changes_nothing_has_nothing_to_undo(seeded):
    depth = seeded.snaps.depth()
    change = seeded.apply("UPDATE books SET price = 0 WHERE id = 999")
    assert change.tables == [] and change.snapshot_id is None
    assert "no visible change" in change.summary and seeded.snaps.depth() == depth


def test_the_model_only_gets_counts_back(seeded):
    view = seeded.apply("UPDATE books SET price = 99 WHERE id = 1").model_view()
    assert view["ok"] and view["changes"] == [
        {"table": "books", "change": "modified", "rows_before": 3, "rows_after": 3,
         "inserted": 0, "updated": 1, "deleted": 0}]
    assert "A1" not in str(view) and "99" not in str(view["changes"])


# ── atomicity and failures ────────────────────────────────────────────────

def test_a_failing_statement_rolls_back_the_whole_batch(seeded):
    depth = seeded.snaps.depth()
    with pytest.raises(ChangeFailed, match="Statement 2 failed and the whole batch was rolled back") as err:
        seeded.apply(
            "INSERT INTO authors (id, name) VALUES (10, 'Temp')",
            "INSERT INTO books (id, title, author_id) VALUES (50, 'orphan', 777)",       # FK violation
        )
    assert err.value.index == 2 and "FOREIGN KEY" in str(err.value).upper()
    assert seeded.flash.count("authors") == 2 and seeded.snaps.depth() == depth


def test_guard_rejections_change_nothing(seeded):
    depth = seeded.snaps.depth()
    for statements in (["PRAGMA writable_schema = ON"], ["DROP TABLE books", "ATTACH DATABASE 'x' AS y"],
                       ["INSERT INTO sqlite_master (name) VALUES ('x')"], ["SELECT 1"]):
        with pytest.raises(GuardError):
            seeded.apply(*statements)
    assert seeded.flash.count("books") == 3 and seeded.snaps.depth() == depth


def test_every_supported_kind_of_change_works_through_the_engine_authorizer(seeded):
    """The authorizer sits inside SQLite: all the legitimate DDL / DML below must still pass it."""
    seeded.apply("CREATE INDEX idx_books_author ON books (author_id)", "CREATE UNIQUE INDEX idx_authors_name ON authors (name)")
    seeded.apply("ALTER TABLE books ADD COLUMN stock INTEGER DEFAULT 5")
    seeded.apply("ALTER TABLE books RENAME COLUMN stock TO in_stock")
    seeded.apply("ALTER TABLE books DROP COLUMN in_stock")
    seeded.apply("ALTER TABLE authors RENAME TO writers")
    seeded.apply("INSERT OR REPLACE INTO writers (id, name) VALUES (1, 'Ada L.')", "INSERT OR IGNORE INTO writers (id, name) VALUES (9, 'Nine')")
    seeded.apply("CREATE TABLE archive AS SELECT * FROM books WHERE price > 12")
    seeded.apply("DROP INDEX idx_books_author", "DROP TABLE archive")
    assert [t for t in seeded.flash.show_tables()] == ["books", "writers"]
    # INSERT OR REPLACE deleted Ada's old row first, so ON DELETE CASCADE removed her two books (3 -> 1).
    assert seeded.flash.count("books") == 1 and seeded.flash.count("writers") == 3


def test_the_engine_itself_refuses_what_the_sandbox_never_allows(seeded):
    seeded.flash.clear_hooks()                  # even with the parser guard gone, SQLite's authorizer says no
    for sql in ("PRAGMA writable_schema = ON", "PRAGMA query_only = ON", "ATTACH DATABASE ':memory:' AS x",
                "CREATE VIEW v AS SELECT 1", "CREATE TEMP TABLE scratch (a int)",
                "CREATE TRIGGER tr AFTER INSERT ON authors BEGIN SELECT 1; END", "SELECT load_extension('x')",
                "SELECT \"load_extension\"('x')"):
        with pytest.raises(Exception, match="not authorized"):
            seeded.flash.raw(sql)
    seeded.flash.raw("INSERT INTO authors (id, name) VALUES (7, 'still allowed')")      # normal writes keep working
    assert seeded.flash.count("authors") == 3


def test_the_connection_hook_still_blocks_anything_unusual(seeded):
    for sql in ("PRAGMA query_only = OFF", "ATTACH DATABASE 'x.db' AS y", "VACUUM", "CREATE VIEW v AS SELECT 1"):
        with pytest.raises(GuardError):
            seeded.flash.raw(sql)
    assert seeded.flash.raw("SELECT COUNT(*) AS n FROM books") == [{"n": 3}]


def test_the_sandbox_size_cap_is_enforced(tmp_path):
    small = Box(tmp_path, Settings(query_timeout=10, sandbox_max_mb=1))
    try:
        small.apply("CREATE TABLE big (x TEXT)")
        with pytest.raises(ChangeFailed, match="sandbox is full"):
            small.apply(
                "INSERT INTO big (x) WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c WHERE n < 100000) "
                "SELECT hex(zeroblob(200)) FROM c"
            )
        assert small.flash.count("big") == 0
        assert small.path.stat().st_size <= 1024 * 1024 + 8192
    finally:
        small.flash.close()


def test_a_runaway_statement_is_stopped_and_rolled_back(tmp_path):
    quick = Box(tmp_path, Settings(query_timeout=1, sandbox_max_mb=20))
    try:
        quick.apply("CREATE TABLE t (n INTEGER)")
        with pytest.raises(ChangeFailed, match="time limit|full"):
            quick.apply("INSERT INTO t (n) WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c) SELECT n FROM c")
        assert quick.flash.count("t") == 0
        quick.apply("INSERT INTO t (n) VALUES (1)")                   # and the connection is still healthy
        assert quick.flash.count("t") == 1
    finally:
        quick.flash.close()


# ── snapshots / undo ──────────────────────────────────────────────────────

def test_undo_restores_the_previous_state(seeded):
    seeded.apply("DELETE FROM authors")                               # cascades into books too
    assert seeded.flash.count("authors") == 0 and seeded.flash.count("books") == 0
    seeded.flash.close()
    latest = seeded.snaps.restore_latest()
    assert "−5 rows" in latest["summary"]
    seeded.reopen()
    assert seeded.flash.count("authors") == 2 and seeded.flash.count("books") == 3


def test_undo_steps_back_through_several_changes(seeded):
    seeded.apply("INSERT INTO authors (id, name) VALUES (3, 'C')")
    seeded.apply("INSERT INTO authors (id, name) VALUES (4, 'D')")
    assert seeded.snaps.depth() == 3
    for expected in (3, 2):
        seeded.flash.close()
        seeded.snaps.restore_latest()
        seeded.reopen()
        assert seeded.flash.count("authors") == expected
    assert seeded.snaps.depth() == 1


def test_undoing_the_first_change_removes_the_database_again(tmp_path):
    path = tmp_path / "fresh.db"
    snaps = Snapshots(tmp_path / "s", path)
    snaps.take("created")                                              # taken while the file does not exist
    path.write_bytes(b"not a database anymore")
    snaps.restore_latest()
    assert not path.exists()
    with pytest.raises(LookupError, match="Nothing to undo"):
        snaps.restore_latest()


def test_only_the_most_recent_snapshots_are_kept(box):
    box.apply("CREATE TABLE t (n INTEGER)")
    for i in range(MAX_SNAPSHOTS + 5):
        box.apply(f"INSERT INTO t (n) VALUES ({i})")
    assert box.snaps.depth() == MAX_SNAPSHOTS
    box.snaps.clear()
    assert box.snaps.depth() == 0 and box.snaps.peek() is None


# ── comparison details ────────────────────────────────────────────────────

def _state(rows, columns=("id", "v"), pk=("id",), count=None):
    return {"columns": list(columns), "pk": list(pk), "count": len(rows) if count is None else count,
            "rows": None if rows is None else {f"[{r['id']}]": r for r in rows}}


def test_changed_rows_shown_are_capped_but_totals_are_exact():
    before = {"t": _state([])}
    after = {"t": _state([{"id": i, "v": "x"} for i in range(60)])}
    (entry,) = diff_states(before, after)
    assert entry["inserted_total"] == 60 and len(entry["inserted"]) == sandbox.DIFF_SHOW


def test_big_tables_are_compared_by_count_only():
    before = {"t": _state(None, count=5000)}
    after = {"t": _state(None, count=5001)}
    (entry,) = diff_states(before, after)
    assert entry["row_diff"] is False and (entry["before_count"], entry["after_count"]) == (5000, 5001)
    assert diff_states(before, {"t": _state(None, count=5000)}) == []


def test_identical_states_produce_no_changes(seeded):
    state = capture_state(seeded.flash)
    assert diff_states(state, capture_state(seeded.flash)) == []


def test_tables_without_a_primary_key_are_compared_by_rowid(box):
    box.apply("CREATE TABLE log (msg TEXT)", "INSERT INTO log (msg) VALUES ('a'), ('b')")
    change = box.apply("UPDATE log SET msg = 'z' WHERE msg = 'a'", "DELETE FROM log WHERE msg = 'b'")
    (log,) = change.tables
    assert log["updated_total"] == 1 and log["deleted_total"] == 1 and log["updated"][0]["after"] == {"msg": "z"}


def test_open_sandbox_reports_its_protections(box):
    assert box.enforcement.statement_timeout == "set" and box.enforcement.guard
    assert any("capped" in n for n in box.enforcement.notes)
