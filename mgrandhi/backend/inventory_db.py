"""Lightweight SQLite inventory store for shelf scans (Module 4 groundwork).

Every analyzed shelf image is persisted as one `scans` row plus one `items` row per detected
product. This turns the per-image pipeline into a queryable inventory over time, which is what
the Business-Intelligence layer (bi_interface/bi_engine.py) answers questions against.

Dependency-free (stdlib sqlite3 + pandas) so it can later be lifted behind a FastAPI backend
with minimal change.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

# Kept out of git (see .gitignore: *.db). Override with $INVENTORY_DB.
DB_PATH = os.getenv(
    "INVENTORY_DB",
    str(Path(__file__).resolve().parents[1] / "inventory.db"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    image_name TEXT,
    image_path TEXT,
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
    crop_path TEXT,
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS item_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL,
    crop_id INTEGER NOT NULL,
    feedback_type TEXT NOT NULL CHECK (feedback_type IN ('category', 'sku')),
    verdict TEXT NOT NULL CHECK (verdict IN ('correct', 'incorrect')),
    correction TEXT,
    note TEXT,
    source_image_path TEXT,
    crop_image_path TEXT,
    box TEXT,
    predicted_category TEXT,
    predicted_subcategory TEXT,
    predicted_sku_text TEXT,
    predicted_visible_text TEXT,
    predicted_sku_confidence REAL,
    ts TEXT NOT NULL,
    UNIQUE (scan_id, crop_id, feedback_type),
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_items_scan ON items(scan_id);
CREATE INDEX IF NOT EXISTS idx_items_category ON items(category);
CREATE INDEX IF NOT EXISTS idx_feedback_scan ON item_feedback(scan_id);
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
    "crop_path": "TEXT",
}

_SCAN_OPTIONAL_COLUMNS = {"image_path": "TEXT"}

_FEEDBACK_OPTIONAL_COLUMNS = {
    "source_image_path": "TEXT",
    "crop_image_path": "TEXT",
    "box": "TEXT",
    "predicted_category": "TEXT",
    "predicted_subcategory": "TEXT",
    "predicted_sku_text": "TEXT",
    "predicted_visible_text": "TEXT",
    "predicted_sku_confidence": "REAL",
}


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        _ensure_columns(conn, "scans", _SCAN_OPTIONAL_COLUMNS)
        _ensure_columns(conn, "items", _ITEM_OPTIONAL_COLUMNS)
        _ensure_columns(conn, "item_feedback", _FEEDBACK_OPTIONAL_COLUMNS)


def _ensure_columns(
    conn: sqlite3.Connection,
    table: str,
    optional_columns: dict[str, str],
) -> None:
    """Add columns to older local DBs without dropping existing data."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, column_type in optional_columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {column_type}")


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
    return scan_id


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


def attach_scan_artifacts(
    scan_id: int,
    source_image_path: str,
    crop_paths: dict[int, str],
    db_path: str = DB_PATH,
) -> None:
    """Attach persisted source/crop evidence to a saved scan and its items."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE scans SET image_path = ? WHERE id = ?",
            (source_image_path, scan_id),
        )
        conn.executemany(
            "UPDATE items SET crop_path = ? WHERE scan_id = ? AND crop_id = ?",
            [(path, scan_id, crop_id) for crop_id, path in crop_paths.items()],
        )


def save_feedback(
    scan_id: int,
    crop_id: int,
    feedback_type: str,
    verdict: str,
    correction: str = "",
    note: str = "",
    db_path: str = DB_PATH,
) -> int:
    """Create or replace one human verdict for a detected product field."""
    init_db(db_path)
    with _connect(db_path) as conn:
        item = conn.execute(
            """SELECT s.image_path, i.crop_path, i.box, i.category, i.subcategory,
                      i.sku_text, i.visible_text, i.sku_confidence
               FROM items i
               JOIN scans s ON s.id = i.scan_id
               WHERE i.scan_id = ? AND i.crop_id = ?""",
            (scan_id, crop_id),
        ).fetchone()
        if item is None:
            raise ValueError("Detected product does not exist")
        conn.execute(
            """INSERT INTO item_feedback (
                   scan_id, crop_id, feedback_type, verdict, correction, note,
                   source_image_path, crop_image_path, box,
                   predicted_category, predicted_subcategory, predicted_sku_text,
                   predicted_visible_text, predicted_sku_confidence, ts
               )
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(scan_id, crop_id, feedback_type) DO UPDATE SET
                   verdict = excluded.verdict,
                   correction = excluded.correction,
                   note = excluded.note,
                   source_image_path = excluded.source_image_path,
                   crop_image_path = excluded.crop_image_path,
                   box = excluded.box,
                   predicted_category = excluded.predicted_category,
                   predicted_subcategory = excluded.predicted_subcategory,
                   predicted_sku_text = excluded.predicted_sku_text,
                   predicted_visible_text = excluded.predicted_visible_text,
                   predicted_sku_confidence = excluded.predicted_sku_confidence,
                   ts = excluded.ts""",
            (
                scan_id,
                crop_id,
                feedback_type,
                verdict,
                correction.strip(),
                note.strip(),
                item[0],
                item[1],
                item[2],
                item[3],
                item[4],
                item[5],
                item[6],
                item[7],
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        row = conn.execute(
            """SELECT id FROM item_feedback
               WHERE scan_id = ? AND crop_id = ? AND feedback_type = ?""",
            (scan_id, crop_id, feedback_type),
        ).fetchone()
        return int(row[0])


def get_feedback_df(db_path: str = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with _connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM item_feedback ORDER BY id DESC", conn)


def clear_all(db_path: str = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.executescript("DELETE FROM item_feedback; DELETE FROM items; DELETE FROM scans;")


def stats(db_path: str = DB_PATH) -> dict:
    scans = get_scans_df(db_path)
    items = get_items_df(db_path=db_path)
    known = items[items["category"].str.lower() != "unknown"] if not items.empty else items
    return {
        "total_scans": int(len(scans)),
        "total_items": int(len(items)),
        "distinct_categories": int(known["category"].nunique()) if not known.empty else 0,
    }
