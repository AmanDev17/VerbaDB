from __future__ import annotations

import pytest

from verbadb.store import AppStore, _escape_like


@pytest.fixture
def store(tmp_path):
    s = AppStore(tmp_path / "data", max_saved_prompts=10)
    yield s
    s.close()


def test_databases_are_scoped_to_their_workspace(store):
    a = store.add_database("wsA", "Books", "sandbox", description="library")
    store.add_database("wsA", "Prod", "connection", url="postgresql://u:p@h/db")
    b = store.add_database("wsB", "Mine", "sandbox")

    assert [d["name"] for d in store.list_databases("wsA")] == ["Books", "Prod"]
    assert [d["name"] for d in store.list_databases("wsB")] == ["Mine"]
    assert store.get_database("wsA", a["public_id"])["description"] == "library"
    assert store.get_database("wsB", a["public_id"]) is None             # another workspace cannot see it...
    assert not store.delete_database("wsB", a["public_id"])              # ...or delete it
    assert store.delete_database("wsA", a["public_id"])
    assert store.get_database("wsA", a["public_id"]) is None
    assert store.get_database("wsB", b["public_id"]) is not None


def test_public_ids_are_unique_and_opaque(store):
    ids = {store.add_database("w", f"db{i}", "sandbox")["public_id"] for i in range(5)}
    assert len(ids) == 5 and all(len(i) == 32 for i in ids)


def test_prompts_newest_first_and_filtered(store):
    store.add_prompt("w", "first question", "ok", "answer one", database_id="d1", database_name="One")
    store.add_prompt("w", "second question", "error", "boom", database_id="d2", database_name="Two")
    store.add_prompt("w", "third question", "ok", database_id="d1", database_name="One")
    store.add_prompt("other", "not mine", "ok")

    assert [p["text"] for p in store.list_prompts("w")] == ["third question", "second question", "first question"]
    assert [p["text"] for p in store.list_prompts("w", database_id="d1")] == ["third question", "first question"]
    assert [p["text"] for p in store.list_prompts("w", q="SECOND")] == ["second question"]      # case-insensitive
    assert store.list_prompts("w", q="nothing like this") == []
    assert [p["text"] for p in store.list_prompts("other")] == ["not mine"]


def test_search_treats_wildcards_literally(store):
    store.add_prompt("w", "discount of 50% off", "ok")
    store.add_prompt("w", "snake_case names", "ok")
    store.add_prompt("w", "plain text", "ok")
    assert [p["text"] for p in store.list_prompts("w", q="50%")] == ["discount of 50% off"]
    assert [p["text"] for p in store.list_prompts("w", q="_case")] == ["snake_case names"]
    assert len(store.list_prompts("w", q="%")) == 1                  # a lone % is not "match everything"
    assert _escape_like("a\\b%c_d") == "a\\\\b\\%c\\_d"


def test_prompt_delete_and_clear_respect_workspaces(store):
    mine = store.add_prompt("w", "keep", "ok", database_id="d1")
    theirs = store.add_prompt("x", "theirs", "ok", database_id="d1")
    assert not store.delete_prompt("w", theirs["id"])
    assert store.delete_prompt("w", mine["id"]) and store.list_prompts("w") == []
    store.add_prompt("w", "a", "ok", database_id="d1")
    store.add_prompt("w", "b", "ok", database_id="d2")
    assert store.clear_prompts("w", database_id="d1") == 1
    assert [p["text"] for p in store.list_prompts("w")] == ["b"]
    assert store.clear_prompts("w") == 1 and len(store.list_prompts("x")) == 1


def test_old_prompts_are_pruned_per_workspace(store):
    for i in range(14):
        store.add_prompt("w", f"q{i}", "ok")
    store.add_prompt("x", "other workspace", "ok")
    texts = [p["text"] for p in store.list_prompts("w", limit=100)]
    assert len(texts) == 10 and texts[0] == "q13" and texts[-1] == "q4"
    assert len(store.list_prompts("x")) == 1


def test_long_prompts_and_summaries_are_trimmed(store):
    row = store.add_prompt("w", "x" * 5000, "ok", "y" * 5000)
    saved = store.list_prompts("w")[0]
    assert len(saved["text"]) == 2000 and len(saved["summary"]) == 400 and row["id"] == saved["id"]


def test_data_survives_reopening(tmp_path):
    first = AppStore(tmp_path / "d")
    db = first.add_database("w", "Kept", "sandbox")
    first.add_prompt("w", "remember me", "ok")
    first.close()
    second = AppStore(tmp_path / "d")
    try:
        assert second.get_database("w", db["public_id"])["name"] == "Kept"
        assert second.list_prompts("w")[0]["text"] == "remember me"
    finally:
        second.close()
