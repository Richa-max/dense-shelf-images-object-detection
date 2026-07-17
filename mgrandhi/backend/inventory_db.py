"""Lightweight SQLite inventory store for shelf scans (Module 4 groundwork).

Every analyzed shelf image is persisted as one `scans` row plus one `items` row per detected
product. This turns the per-image pipeline into a queryable inventory over time, which is what
the Business-Intelligence layer (bi_interface/bi_engine.py) answers questions against.

Dependency-free (stdlib sqlite3 + pandas) so it can later be lifted behind a FastAPI backend
with minimal change.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

# Kept out of git (see .gitignore: *.db). Override with $INVENTORY_DB.
DB_PATH = str(Path(__file__).resolve().parents[1] / "inventory.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    image_name TEXT,
    num_items INTEGER,
    distinct_categories INTEGER,
    empty_pct REAL,
    shelf_type TEXT,
    review_count INTEGER
);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL,
    crop_id INTEGER,
    category TEXT,
    subcategory TEXT,
    score REAL,
    area INTEGER,
    box TEXT,
    brand TEXT,
    product_name TEXT,
    sku_text TEXT,
    visible_text TEXT,
    package_size TEXT,
    barcode TEXT,
    sku_confidence REAL,
    sku_needs_review INTEGER,
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_items_scan ON items(scan_id);
CREATE INDEX IF NOT EXISTS idx_items_category ON items(category);
CREATE TABLE IF NOT EXISTS inventory_stock (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_key TEXT NOT NULL UNIQUE,
    match_type TEXT,
    match_value TEXT,
    category TEXT,
    subcategory TEXT,
    product_name TEXT,
    barcode TEXT,
    quantity INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_inventory_stock_key ON inventory_stock(stock_key);
"""

_ITEM_OPTIONAL_COLUMNS = {
    "brand": "TEXT",
    "product_name": "TEXT",
    "sku_text": "TEXT",
    "visible_text": "TEXT",
    "package_size": "TEXT",
    "barcode": "TEXT",
    "sku_confidence": "REAL",
    "sku_needs_review": "INTEGER",
}


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        _ensure_item_columns(conn)


def _ensure_item_columns(conn: sqlite3.Connection) -> None:
    """Add SKU/OCR columns to older local DBs without dropping existing data."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(items)").fetchall()}
    for name, column_type in _ITEM_OPTIONAL_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE items ADD COLUMN {name} {column_type}")


def _clean(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def stock_identity(record: dict) -> dict:
    barcode = _clean(record.get("barcode"))
    product_name = _clean(record.get("product_name"))
    sku_text = _clean(record.get("sku_text"))
    category = _clean(record.get("category")) or "unknown"
    subcategory = _clean(record.get("subcategory")) or "unknown"

    if barcode:
        match_type = "barcode"
        match_value = barcode
    elif product_name:
        match_type = "product_name"
        match_value = product_name.lower()
    elif sku_text:
        match_type = "sku_text"
        match_value = sku_text.lower()
    else:
        match_type = "category_subcategory"
        match_value = f"{category.lower()}::{subcategory.lower()}"

    return {
        "stock_key": f"{match_type}:{match_value}",
        "match_type": match_type,
        "match_value": match_value,
        "category": category,
        "subcategory": subcategory,
        "product_name": product_name,
        "barcode": barcode,
    }


def cart_key(record: dict) -> str:
    crop_id = _clean(record.get("crop_id")) or "0"
    return f"{crop_id}|{stock_identity(record)['stock_key']}"


def _upsert_stock(conn: sqlite3.Connection, record: dict, quantity_delta: int) -> None:
    if quantity_delta == 0:
        return
    ident = stock_identity(record)
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """INSERT INTO inventory_stock (
               stock_key, match_type, match_value, category, subcategory,
               product_name, barcode, quantity, updated_at
           )
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(stock_key) DO UPDATE SET
               quantity = MAX(0, inventory_stock.quantity + excluded.quantity),
               category = excluded.category,
               subcategory = excluded.subcategory,
               product_name = COALESCE(NULLIF(excluded.product_name, ''), inventory_stock.product_name),
               barcode = COALESCE(NULLIF(excluded.barcode, ''), inventory_stock.barcode),
               updated_at = excluded.updated_at""",
        (
            ident["stock_key"],
            ident["match_type"],
            ident["match_value"],
            ident["category"],
            ident["subcategory"],
            ident["product_name"],
            ident["barcode"],
            int(quantity_delta),
            now,
        ),
    )


def save_scan(result, image_name: str, records: list[dict], db_path: str = DB_PATH) -> int:
    """Persist one analysis result. Returns the new scan id."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO scans (ts, image_name, num_items, distinct_categories,
                                  empty_pct, shelf_type, review_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(timespec="seconds"), image_name, result.num_items,
             result.distinct_categories, result.empty_pct, result.shelf_type,
             result.review_count),
        )
        scan_id = int(cur.lastrowid)
        conn.executemany(
            """INSERT INTO items (
                   scan_id, crop_id, category, subcategory, score, area, box,
                   brand, product_name, sku_text, visible_text, package_size, barcode,
                   sku_confidence, sku_needs_review
               )
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(scan_id, r["crop_id"], r["category"], r["subcategory"], r["score"], r["area"],
              json.dumps(r["box"]), r.get("brand"), r.get("product_name"), r.get("sku_text"),
              r.get("visible_text"), r.get("package_size"), r.get("barcode"),
              r.get("sku_confidence"), r.get("sku_needs_review")) for r in records],
        )
        for record in records:
            _upsert_stock(conn, record, 1)
    return scan_id


def checkout_items(cart: dict[str, int], records: list[dict], db_path: str = DB_PATH) -> dict:
    init_db(db_path)
    selected = {str(k): int(v) for k, v in cart.items() if int(v) > 0}
    by_key = {cart_key(record): record for record in records}
    grouped: dict[str, dict] = {}

    for key, quantity in selected.items():
        record = by_key.get(key)
        if not record:
            continue
        ident = stock_identity(record)
        stock_key = ident["stock_key"]
        if stock_key not in grouped:
            grouped[stock_key] = {"record": record, "quantity": 0, "identity": ident}
        grouped[stock_key]["quantity"] += quantity

    checked_out = []
    short = []
    with _connect(db_path) as conn:
        for stock_key, entry in grouped.items():
            quantity = int(entry["quantity"])
            row = conn.execute(
                "SELECT quantity FROM inventory_stock WHERE stock_key = ?",
                (stock_key,),
            ).fetchone()
            available = int(row[0]) if row else 0
            decrement = min(available, quantity)
            if decrement:
                _upsert_stock(conn, entry["record"], -decrement)
                checked_out.append({**entry["identity"], "quantity": decrement})
            if decrement < quantity:
                short.append({**entry["identity"], "requested": quantity, "available": available})

    return {
        "requested": sum(selected.values()),
        "checked_out": sum(item["quantity"] for item in checked_out),
        "items": checked_out,
        "short": short,
    }


def get_scans_df(db_path: str = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with _connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM scans ORDER BY id DESC", conn)


def get_items_df(scan_id: int | None = None, db_path: str = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with _connect(db_path) as conn:
        if scan_id is None:
            return pd.read_sql_query("SELECT * FROM items", conn)
        return pd.read_sql_query("SELECT * FROM items WHERE scan_id = ?", conn, params=(scan_id,))


def get_stock_df(db_path: str = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with _connect(db_path) as conn:
        return pd.read_sql_query(
            """SELECT category, subcategory, product_name, barcode, quantity, updated_at
               FROM inventory_stock
               ORDER BY quantity DESC, category, subcategory""",
            conn,
        )


def clear_all(db_path: str = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.executescript("DELETE FROM items; DELETE FROM scans; DELETE FROM inventory_stock;")


def stats(db_path: str = DB_PATH) -> dict:
    scans = get_scans_df(db_path)
    items = get_items_df(db_path=db_path)
    known = items[items["category"].str.lower() != "unknown"] if not items.empty else items
    return {
        "total_scans": int(len(scans)),
        "total_items": int(len(items)),
        "distinct_categories": int(known["category"].nunique()) if not known.empty else 0,
        "stock_units": int(get_stock_df(db_path=db_path)["quantity"].sum()),
    }
