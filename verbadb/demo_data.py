"""
Deterministic demo data (customers, products, orders, order_items) written through pyflashdb, so the same
dataset can be loaded into SQLite, MySQL, PostgreSQL or MongoDB. Dates are relative to ``today`` so that
"last month" always has data.
"""

from __future__ import annotations

import datetime as dt
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

TABLES = ("customers", "products", "orders", "order_items")
REVENUE_STATUSES = ("paid", "shipped")

_FIRST = ["Ava", "Liam", "Noah", "Mia", "Zoe", "Ethan", "Ivy", "Leo", "Nora", "Omar", "Priya", "Sam",
          "Tara", "Uma", "Victor", "Wen", "Xavi", "Yara", "Zane", "Aria"]
_LAST = ["Shah", "Miller", "Garcia", "Chen", "Okafor", "Novak", "Singh", "Rossi", "Kim", "Dubois"]
_COUNTRIES = ["US", "GB", "DE", "IN", "FR", "CA"]
_PRODUCTS = [
    ("Notebook", "stationery", 4.50), ("Pen set", "stationery", 8.00), ("Desk lamp", "home", 32.00),
    ("Office chair", "furniture", 189.00), ("Standing desk", "furniture", 449.00), ("Monitor", "electronics", 219.00),
    ("Keyboard", "electronics", 59.00), ("Mouse", "electronics", 24.50), ("Headphones", "electronics", 129.00),
    ("Backpack", "accessories", 49.00), ("Water bottle", "accessories", 14.00), ("Plant pot", "home", 18.00),
]
_STATUSES = ["paid", "shipped", "refunded", "cancelled", "pending"]
_STATUS_WEIGHTS = [55, 25, 7, 8, 5]


def previous_month_range(today: dt.date) -> Tuple[dt.datetime, dt.datetime]:
    """``[start, end)`` of the previous calendar month as datetimes."""
    first_this = today.replace(day=1)
    last_prev = first_this - dt.timedelta(days=1)
    return dt.datetime.combine(last_prev.replace(day=1), dt.time.min), dt.datetime.combine(first_this, dt.time.min)


def _month_starts(today: dt.date, back: int) -> List[dt.date]:
    starts: List[dt.date] = []
    cursor = today.replace(day=1)
    for _ in range(back + 1):
        starts.append(cursor)
        cursor = (cursor - dt.timedelta(days=1)).replace(day=1)
    return list(reversed(starts))


def build_demo(flash: Any, today: Optional[dt.date] = None, seed: int = 42, orders_per_month: int = 120) -> Dict[str, int]:
    """(Re)create the demo tables in ``flash`` and fill them. Returns row counts."""
    today = today or dt.date.today()
    rng = random.Random(seed)

    for table in reversed(TABLES):
        flash.drop_table(table)
    flash.create_table("customers", {
        "id": "int", "name": "str(100)", "email": "str(120)", "country": "str(2)", "created_at": "datetime",
    })
    flash.create_table("products", {"id": "int", "name": "str(100)", "category": "str(40)", "price": "decimal(10,2)"})
    flash.create_table("orders", {
        "id": "int", "customer_id": "int", "status": "str(20)", "ordered_at": "datetime", "total": "decimal(10,2)",
    })
    flash.create_table("order_items", {
        "id": "int", "order_id": "int", "product_id": "int", "quantity": "int", "unit_price": "decimal(10,2)",
    })

    customers = []
    for i in range(1, 41):
        # (first, last) is unique for all 40 customers, so "by customer name" is never ambiguous.
        first, last = _FIRST[(i - 1) % len(_FIRST)], _LAST[((i - 1) // 4) % len(_LAST)]
        customers.append({
            "id": i, "name": f"{first} {last}", "email": f"{first.lower()}.{last.lower()}{i}@example.com",
            "country": _COUNTRIES[i % len(_COUNTRIES)],
            "created_at": dt.datetime.combine(today - dt.timedelta(days=rng.randint(120, 700)), dt.time(9, 0)),
        })
    products = [
        {"id": i, "name": n, "category": c, "price": p} for i, (n, c, p) in enumerate(_PRODUCTS, start=1)
    ]
    flash.bulk_insert("customers", customers)
    flash.bulk_insert("products", products)

    # A few customers buy far more often, so the top-N is meaningful and free of ties.
    weights = [6 if i <= 8 else 1 for i in range(1, 41)]
    orders: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []
    order_id = item_id = 0
    for month_start in _month_starts(today, back=3):
        next_month = (month_start + dt.timedelta(days=32)).replace(day=1)
        last_day = min(next_month - dt.timedelta(days=1), today)
        span = (last_day - month_start).days + 1
        for _ in range(orders_per_month):
            order_id += 1
            placed = dt.datetime.combine(
                month_start + dt.timedelta(days=rng.randrange(span)),
                dt.time(rng.randrange(8, 22), rng.randrange(60), rng.randrange(60)),
            )
            total = 0.0
            for _ in range(rng.randint(1, 4)):
                item_id += 1
                product = rng.choice(products)
                quantity = rng.randint(1, 3)
                total += quantity * product["price"]
                items.append({
                    "id": item_id, "order_id": order_id, "product_id": product["id"],
                    "quantity": quantity, "unit_price": product["price"],
                })
            orders.append({
                "id": order_id, "customer_id": rng.choices(range(1, 41), weights)[0],
                "status": rng.choices(_STATUSES, _STATUS_WEIGHTS)[0], "ordered_at": placed,
                "total": round(total, 2),
            })
    flash.bulk_insert("orders", orders)
    flash.bulk_insert("order_items", items)
    return {"customers": len(customers), "products": len(products), "orders": len(orders), "order_items": len(items)}


def top_customers_last_month(
    flash: Any, today: Optional[dt.date] = None, n: int = 5, statuses: Sequence[str] = REVENUE_STATUSES
) -> List[Tuple[str, float]]:
    """Ground truth for "top N customers by revenue last month", computed in plain Python (no SQL)."""
    today = today or dt.date.today()
    start, end = previous_month_range(today)
    totals: Dict[int, float] = defaultdict(float)
    for order in flash.select("orders"):
        placed = order["ordered_at"]
        if isinstance(placed, str):
            placed = dt.datetime.fromisoformat(placed)
        if start <= placed < end and order["status"] in statuses:
            totals[order["customer_id"]] += float(order["total"])
    names = {c["id"]: c["name"] for c in flash.select("customers")}
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [(names[cid], round(total, 2)) for cid, total in ranked]
