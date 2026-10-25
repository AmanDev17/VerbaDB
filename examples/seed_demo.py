"""
Create the demo database.

    python examples/seed_demo.py                         # sqlite:///demo.db
    python examples/seed_demo.py sqlite:///other.db
    python examples/seed_demo.py --also postgresql://postgres:flash@127.0.0.1:5432/flash_test \\
                                 --also mysql://root:flash@127.0.0.1:3306/flash_test

The primary database is built through pyflashdb; every ``--also`` target is then filled with
``copy_to`` (pyflashdb copies tables across backends), so the same data can be queried on any of them.
"""

from __future__ import annotations

import argparse

from pyflashdb import FlashDB

from verbadb import demo_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", nargs="?", default="sqlite:///demo.db", help="primary database URL")
    parser.add_argument("--also", action="append", default=[], metavar="URL", help="copy the data to this database too")
    args = parser.parse_args()

    with FlashDB.from_url(args.url) as primary:
        counts = demo_data.build_demo(primary)
        print(f"{args.url}: {counts}")
        for target_url in args.also:
            with FlashDB.from_url(target_url) as target:
                for table in reversed(demo_data.TABLES):
                    target.drop_table(table)
                copied = primary.copy_to(target, tables=list(demo_data.TABLES), mode="replace")
                print(f"{target_url}: {copied}")


if __name__ == "__main__":
    main()
