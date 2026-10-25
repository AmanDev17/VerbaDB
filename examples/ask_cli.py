"""
Ask the agent one question from the terminal — the quickest way to check your Gemini key, model id and
database before using the web UI. Uses the same agent, guard and settings (.env) as the web app.

    python examples/seed_demo.py                       # once: create demo.db
    python examples/ask_cli.py "top 5 customers by revenue last month"
    python examples/ask_cli.py "how many orders per status?" --db demo
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from verbadb.agent import Agent, Session
from verbadb.config import load_settings
from verbadb.llm import GeminiLLM
from verbadb.manager import DatabaseManager, slug


def _print_table(event: dict, limit: int = 12) -> None:
    columns, rows = event["columns"], event["rows"][:limit]
    widths = [max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) if rows else len(str(c)) for c in columns]
    print("  " + "  ".join(str(c).ljust(w) for c, w in zip(columns, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print("  " + "  ".join(str(row.get(c, "")).ljust(w) for c, w in zip(columns, widths)))
    more = event["row_count"] - len(rows)
    if more > 0:
        print(f"  … {more} more row(s)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--db", help="database name from VERBADB_DATABASES (default: the first one)")
    parser.add_argument("--new", action="store_true",
                        help="start with no database, e.g. 'create a database for a pet clinic with sample data'")
    args = parser.parse_args()

    settings = load_settings()
    if not settings.databases and not args.new:
        settings = replace(settings, databases={"demo": "sqlite:///demo.db"})
    if not settings.api_key:
        print("GEMINI_API_KEY is not set (put it in .env or the environment).", file=sys.stderr)
        return 2

    llm = GeminiLLM(settings.api_key, settings.model)
    ok, detail = llm.check_model()
    if ok is False:
        print(detail, file=sys.stderr)
        return 2

    manager = DatabaseManager(settings)
    if args.new:
        database_id = None                               # start without a database: the agent may create one
    else:
        name = args.db or next(iter(settings.databases))
        database_id = f"shared-{slug(name)}"
    agent = Agent(llm, manager, settings)
    status = 0
    try:
        for event in agent.ask_stream(Session(id="cli", workspace="cli", database_id=database_id), args.question):
            kind = event["type"]
            if kind == "start":
                print(f"[{event['mode']} · {event['backend']} · {event['model']}]")
            elif kind == "database":
                print(f"→ active database: {event['database']['name']}")
            elif kind == "change":
                print(f"\nChange: {event['summary']}")
                for table in event["tables"]:
                    print(f"  {table['name']}: {table['change']} (+{table['inserted_total']} −{table['deleted_total']} "
                          f"~{table['updated_total']}), {table['after_count']} rows now")
                print()
            elif kind == "text":
                print(f"  … {event['text']}")
            elif kind == "status":
                print(f"  ({event['message']})")
            elif kind == "step":
                detail = event["args"].get("sql") or event["args"].get("pipeline_json") or event["args"]
                print(f"→ {event['tool']}: {detail}")
            elif kind == "observation":
                print(f"  {'ok' if event['ok'] else 'FAILED'}: {event['summary']}")
            elif kind == "result":
                print(f"\nResult ({event['row_count']} rows):")
                _print_table(event)
                print()
            elif kind == "final":
                print(f"\n{event['text']}\n\n({event['requests']} Gemini requests, {event['total_tokens']} tokens)")
            elif kind == "error":
                print(f"\nERROR [{event['kind']}]: {event['message']}", file=sys.stderr)
                status = 1
    finally:
        manager.close_all()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
