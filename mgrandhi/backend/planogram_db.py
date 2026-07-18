"""Lightweight SQLite store for planogram templates and per-scan compliance results.

Sits next to `inventory_db.py` and reuses its connection (same `inventory.db` file, same
dependency-free stdlib sqlite3 + pandas approach) so templates and results travel with the
rest of the shelf history instead of living in a separate store.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import pandas as pd

from mgrandhi.backend.inventory_db import DB_PATH, _connect
from mgrandhi.backend.planogram_engine import PlanogramResult, TemplateRow, TemplateSlot

_SCHEMA = """
CREATE TABLE IF NOT EXISTS planogram_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    store_id TEXT,
    shelf_id TEXT,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS planogram_template_slots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    template_id INTEGER NOT NULL,
    row_index INTEGER NOT NULL,
    slot_index INTEGER NOT NULL,
    category TEXT,
    subcategory TEXT,
    brand TEXT,
    facings INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (template_id) REFERENCES planogram_templates(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS planogram_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL,
    template_id INTEGER NOT NULL,
    ts TEXT NOT NULL,
    compliance_score REAL,
    total_expected INTEGER,
    total_compliant INTEGER,
    missing_count INTEGER,
    extra_count INTEGER,
    row_count_expected INTEGER,
    row_count_detected INTEGER,
    slots_json TEXT,
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE,
    FOREIGN KEY (template_id) REFERENCES planogram_templates(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_planogram_slots_template ON planogram_template_slots(template_id);
CREATE INDEX IF NOT EXISTS idx_planogram_results_scan ON planogram_results(scan_id);
CREATE INDEX IF NOT EXISTS idx_planogram_results_template ON planogram_results(template_id);
"""


def init_db(db_path: str = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def save_template(
    name: str,
    rows: list[TemplateRow],
    store_id: str = "",
    shelf_id: str = "",
    db_path: str = DB_PATH,
) -> int:
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO planogram_templates (name, store_id, shelf_id, ts) VALUES (?, ?, ?, ?)",
            (name, store_id, shelf_id, datetime.now().isoformat(timespec="seconds")),
        )
        template_id = int(cur.lastrowid)
        conn.executemany(
            """INSERT INTO planogram_template_slots
                   (template_id, row_index, slot_index, category, subcategory, brand, facings)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    template_id,
                    row.row_index,
                    slot.slot_index,
                    slot.category,
                    slot.subcategory,
                    slot.brand,
                    slot.facings,
                )
                for row in rows
                for slot in row.slots
            ],
        )
    return template_id


def list_templates(db_path: str = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with _connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM planogram_templates ORDER BY id DESC", conn
        )


def get_template(template_id: int, db_path: str = DB_PATH) -> list[TemplateRow] | None:
    init_db(db_path)
    with _connect(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM planogram_templates WHERE id = ?", (template_id,)
        ).fetchone()
        if not exists:
            return None
        rows = conn.execute(
            """SELECT row_index, slot_index, category, subcategory, brand, facings
               FROM planogram_template_slots
               WHERE template_id = ?
               ORDER BY row_index, slot_index""",
            (template_id,),
        ).fetchall()

    by_row: dict[int, list[TemplateSlot]] = {}
    for row_index, slot_index, category, subcategory, brand, facings in rows:
        by_row.setdefault(row_index, []).append(
            TemplateSlot(
                slot_index=slot_index,
                category=category or "",
                subcategory=subcategory or "",
                brand=brand or "",
                facings=facings or 1,
            )
        )
    return [
        TemplateRow(row_index=row_index, slots=slots)
        for row_index, slots in sorted(by_row.items())
    ]


def save_result(scan_id: int, template_id: int, result: PlanogramResult, db_path: str = DB_PATH) -> int:
    init_db(db_path)
    slots_payload = [
        {
            "row_index": s.row_index,
            "position": s.position,
            "status": s.status,
            "expected_key": s.expected_key,
            "actual_key": s.actual_key,
            "detail": s.detail,
        }
        for s in result.slots
    ]
    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO planogram_results (
                   scan_id, template_id, ts, compliance_score, total_expected, total_compliant,
                   missing_count, extra_count, row_count_expected, row_count_detected, slots_json
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                scan_id,
                template_id,
                datetime.now().isoformat(timespec="seconds"),
                result.compliance_score,
                result.total_expected,
                result.total_compliant,
                result.missing_count,
                result.extra_count,
                result.row_count_expected,
                result.row_count_detected,
                json.dumps(slots_payload),
            ),
        )
        return int(cur.lastrowid)


def get_results_for_scan(scan_id: int, db_path: str = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with _connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM planogram_results WHERE scan_id = ? ORDER BY id DESC",
            conn,
            params=(scan_id,),
        )


def compliance_trend(template_id: int | None = None, db_path: str = DB_PATH) -> pd.DataFrame:
    """Compliance score over time, optionally scoped to one template — feeds an Insights chart."""
    init_db(db_path)
    with _connect(db_path) as conn:
        if template_id is None:
            return pd.read_sql_query(
                "SELECT * FROM planogram_results ORDER BY ts", conn
            )
        return pd.read_sql_query(
            "SELECT * FROM planogram_results WHERE template_id = ? ORDER BY ts",
            conn,
            params=(template_id,),
        )
