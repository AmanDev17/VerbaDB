from __future__ import annotations

import datetime as dt

import pytest
from pyflashdb import FlashDB

from verbadb import demo_data
from verbadb.config import Settings
from verbadb.connections import open_readonly

TODAY = dt.date(2026, 10, 25)


@pytest.fixture
def settings() -> Settings:
    return Settings(max_rows=50, preview_rows=5, max_steps=6, query_timeout=5)


@pytest.fixture
def demo_url(tmp_path) -> str:
    url = f"sqlite:///{(tmp_path / 'demo.db').as_posix()}"
    with FlashDB.from_url(url) as writer:
        demo_data.build_demo(writer, today=TODAY)
    return url


@pytest.fixture
def ro(demo_url, settings):
    """A read-only, guarded connection to the demo SQLite database (what the agent gets)."""
    flash, enforcement = open_readonly(demo_url, settings)
    flash.enforcement = enforcement          # convenience for assertions
    yield flash
    flash.close()
