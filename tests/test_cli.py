from __future__ import annotations

import pytest

from verbadb import __version__, server


def test_version_and_help_print_and_exit_without_starting_a_server(capsys, monkeypatch):
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: pytest.fail("the server must not start"))
    for flag in ("--version", "--help"):
        with pytest.raises(SystemExit) as exit_info:
            server.main([flag])
        assert exit_info.value.code == 0
    printed = capsys.readouterr().out
    assert f"VerbaDB {__version__}" in printed and "--host" in printed and "VERBADB_" in printed


def test_bad_arguments_are_rejected_politely(capsys, monkeypatch):
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: pytest.fail("the server must not start"))
    with pytest.raises(SystemExit) as exit_info:
        server.main(["--port", "not-a-number"])
    assert exit_info.value.code == 2 and "invalid int value" in capsys.readouterr().err


def test_host_and_port_options_reach_the_server_and_the_host_dependent_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                                   # no .env here
    for name in ("VERBADB_HOST", "VERBADB_PORT", "VERBADB_DATA_DIR"):
        monkeypatch.setenv(name, "x")                             # register for cleanup: main() writes to os.environ
        monkeypatch.delenv(name)
    monkeypatch.setenv("VERBADB_DATA_DIR", str(tmp_path / "data"))

    seen = {}
    monkeypatch.setattr(server, "create_app", lambda settings, **kw: seen.setdefault("settings", settings) or object())
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(kw))

    server.main(["--host", "0.0.0.0", "--port", "9001"])
    assert (seen["host"], seen["port"]) == ("0.0.0.0", 9001)
    settings = seen["settings"]
    assert settings.allow_user_connections is False and settings.allow_private_db_hosts is False   # network-facing: opt-in only
