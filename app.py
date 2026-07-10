import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

import json
import re
import sqlite3
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

from qwen_model import generate_qwen_sku_answer
from retail_product_resolver import resolve_retail_product, summarize_retail_decisions
from swin_faiss import load_swin_faiss_classifier


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg-a: #f8f7f2;
    --bg-b: #e9f2ea;
    --bg-c: #e4edf9;
    --accent: #0f766e;
    --accent-strong: #0b5f58;
    --ink: #12202f;
    --panel: rgba(255, 255, 255, 0.8);
    --panel-border: rgba(15, 118, 110, 0.16);
    --radius-xl: 18px;
}

.gradio-container {
    font-family: 'Space Grotesk', sans-serif;
    max-width: 1320px !important;
    margin: 0 auto;
    padding-top: 14px !important;
    color: var(--ink);
    background:
        radial-gradient(1200px 360px at 2% -5%, #d8f3e4 0%, transparent 62%),
        radial-gradient(1000px 340px at 96% -8%, #dce6ff 0%, transparent 60%),
        linear-gradient(170deg, var(--bg-a) 0%, var(--bg-b) 52%, var(--bg-c) 100%);
}

.hero {
    background:
        linear-gradient(140deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.04)),
        linear-gradient(135deg, #0f766e, #1d3557);
    border: 1px solid rgba(255, 255, 255, 0.28);
    border-radius: var(--radius-xl);
    padding: 20px 22px;
    margin-bottom: 16px;
    color: #ffffff;
    box-shadow: 0 16px 36px rgba(15, 118, 110, 0.24);
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    right: -70px;
    top: -70px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.28) 0%, rgba(255, 255, 255, 0.02) 70%);
}

.hero h1 {
    margin: 0 0 6px 0;
    font-family: 'Fraunces', serif;
    font-size: 2rem;
    line-height: 1.15;
    letter-spacing: 0.01em;
}

.hero p {
    margin: 0;
    opacity: 0.95;
    font-size: 1rem;
    max-width: 760px;
}

.toolbar {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    padding: 10px;
    backdrop-filter: blur(8px);
    box-shadow: 0 10px 24px rgba(16, 24, 40, 0.05);
}

.panel {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    box-shadow: 0 12px 26px rgba(16, 24, 40, 0.08);
    padding: 10px;
    backdrop-filter: blur(6px);
}

.panel h2,
.panel h3,
.panel h4,
.panel h5 {
    color: #12263f;
    margin-top: 0.2rem;
}

.section-title {
    margin: 10px 2px 6px 2px;
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    color: #10324d;
    letter-spacing: 0.01em;
}

.tab-banner {
    margin: 4px 2px 10px 2px;
    padding: 10px 12px;
    border-radius: 12px;
    background: linear-gradient(130deg, rgba(255, 255, 255, 0.9), rgba(220, 244, 238, 0.62));
    border: 1px solid rgba(15, 118, 110, 0.18);
    color: #174160;
    font-size: 0.94rem;
}

.tab-banner b {
    color: #0c5d58;
}

.gradio-container .tab-nav {
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(15, 118, 110, 0.15);
    border-radius: 14px;
    padding: 4px;
}

.gradio-container .tab-nav button {
    border-radius: 10px !important;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(120px, 1fr));
    gap: 12px;
}

.kpi-card {
    background: linear-gradient(160deg, #ffffff, #eef9f5);
    border: 1px solid #cfe8de;
    border-radius: 12px;
    padding: 12px;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.85);
}

.kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #5b6f83;
}

.kpi-value {
    font-family: 'Fraunces', serif;
    font-size: 1.45rem;
    color: #0f172a;
    line-height: 1.1;
}

.sku-table-wrap {
    overflow-x: auto;
}

.sku-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.93rem;
    border-radius: 10px;
    overflow: hidden;
}

.sku-table thead th {
    background: linear-gradient(180deg, #e8f6f1, #dff1ea);
    color: #0f172a;
    text-align: left;
    padding: 9px 10px;
    border-bottom: 1px solid #cce2d8;
    font-weight: 700;
}

.sku-table tbody td {
    padding: 9px 10px;
    border-bottom: 1px solid #e2ece7;
    vertical-align: top;
}

.sku-table tbody tr:nth-child(even) {
    background: rgba(15, 118, 110, 0.04);
}

.sku-table tbody tr:hover {
    background: rgba(20, 184, 166, 0.09);
}

.animate-in {
    animation: fade-slide-up 0.55s ease both;
}

.stagger-1 { animation-delay: 0.05s; }
.stagger-2 { animation-delay: 0.12s; }
.stagger-3 { animation-delay: 0.19s; }
.stagger-4 { animation-delay: 0.26s; }

@keyframes fade-slide-up {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

button.primary, .gr-button-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent-strong)) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 8px 18px rgba(15, 118, 110, 0.28);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}

button.secondary {
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    background: #f3fbf9 !important;
    transition: transform 0.18s ease, background 0.18s ease;
}

button.primary:hover,
.gr-button-primary:hover,
button.secondary:hover {
    transform: translateY(-1px);
}

button.secondary:hover {
    background: #e8f7f2 !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    border-radius: 10px !important;
    border: 1px solid #ccddd5 !important;
}

.gradio-container details summary {
    cursor: pointer;
    color: #0f5d58;
    font-weight: 600;
}

#sku-results {
    border-left: 4px solid var(--accent);
}

@media (max-width: 980px) {
    .hero h1 {
        font-size: 1.65rem;
    }

    .hero p {
        font-size: 0.95rem;
    }

    .kpi-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 680px) {
    .gradio-container {
        padding-top: 8px !important;
    }

    .hero {
        padding: 16px;
    }

    .toolbar,
    .panel {
        padding: 8px;
    }
}
"""


yolo_model = YOLO("models/yolo/best.pt")
swin_classifier = load_swin_faiss_classifier()

YOLO_CONF = float(os.getenv("APP_YOLO_CONF", "0.30"))
MAX_INPUT_EDGE = int(os.getenv("APP_MAX_INPUT_EDGE", "1600"))
MAX_CROPS_ANALYZE = int(os.getenv("APP_MAX_CROPS_ANALYZE", "80"))
MAX_CROPS_SKU = int(os.getenv("APP_MAX_CROPS_SKU", "36"))
MIN_BOX_AREA_RATIO = float(os.getenv("APP_MIN_BOX_AREA_RATIO", "0.00015"))
OCCUPANCY_ROW_COUNT = int(os.getenv("APP_OCCUPANCY_ROW_COUNT", "4"))
ANALYTICS_DB_PATH = os.getenv("APP_ANALYTICS_DB_PATH", "shelf_analytics.db")
LOW_STOCK_FACING_THRESHOLD = int(os.getenv("APP_LOW_STOCK_FACING_THRESHOLD", "2"))


def _db_connect():
    return sqlite3.connect(ANALYTICS_DB_PATH)


def _init_analytics_db():
    with _db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shelf_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at INTEGER NOT NULL,
                store_id TEXT,
                shelf_id TEXT,
                run_mode TEXT NOT NULL,
                total_products INTEGER NOT NULL,
                known_products INTEGER NOT NULL,
                unknown_products INTEGER NOT NULL,
                occupancy_ratio REAL NOT NULL,
                row_occupancy_json TEXT,
                category_mix_json TEXT,
                subcategory_mix_json TEXT,
                avg_qwen_conf REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sku_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                store_id TEXT,
                shelf_id TEXT,
                crop_id INTEGER,
                sku_detail TEXT NOT NULL,
                product_category TEXT,
                subcategory TEXT,
                qwen_confidence REAL,
                semantic_mismatch_flag INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES shelf_runs(id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sku_detections_store_shelf ON sku_detections(store_id, shelf_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sku_detections_created_at ON sku_detections(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sku_detections_sku ON sku_detections(sku_detail)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sku_inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                store_id TEXT NOT NULL DEFAULT '',
                shelf_id TEXT NOT NULL DEFAULT '',
                sku_detail TEXT NOT NULL,
                product_category TEXT NOT NULL DEFAULT 'unknown',
                subcategory TEXT NOT NULL DEFAULT 'unknown',
                quantity INTEGER NOT NULL DEFAULT 0,
                avg_qwen_conf REAL,
                last_updated INTEGER NOT NULL,
                UNIQUE(store_id, shelf_id, sku_detail, product_category, subcategory)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sku_inventory_store_shelf ON sku_inventory(store_id, shelf_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sku_inventory_qty ON sku_inventory(quantity)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sku_inventory_sku ON sku_inventory(sku_detail)")


def _build_mix(rows, key_name):
    counts = {}
    for row in rows:
        label = str(row.get(key_name) or "unknown").strip() or "unknown"
        counts[label] = counts.get(label, 0) + 1
    return counts


def _is_persistable_sku_label(sku_detail):
    label = _clean_label(sku_detail).lower()
    if not label:
        return False
    blocked = {
        "no sku detected",
        "qwen sku detection not run in analyze shelf mode.",
        "unknown",
        "n/a",
        "na",
        "none",
    }
    return label not in blocked


def _store_shelf_keys(store_id=None, shelf_id=None):
    return (str(store_id or "").strip(), str(shelf_id or "").strip())


def _persist_run_snapshot(rows, occupancy_metrics, run_qwen=False, store_id=None, shelf_id=None):
    run_ts = int(time.time())
    total_products = len(rows)
    known_products = sum(1 for row in rows if (row.get("product_category") or "unknown").lower() != "unknown")
    unknown_products = max(0, total_products - known_products)
    occupancy_ratio = float((occupancy_metrics or {}).get("coverage_ratio", 0.0))
    row_occupancy = (occupancy_metrics or {}).get("row_occupancy") or []

    qvals = [float(r.get("qwen_confidence")) for r in rows if isinstance(r.get("qwen_confidence"), (int, float))]
    avg_qwen_conf = (sum(qvals) / len(qvals)) if qvals else None

    category_mix = _build_mix(rows, "product_category")
    subcategory_mix = _build_mix(rows, "subcategory")

    with _db_connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO shelf_runs (
                created_at, store_id, shelf_id, run_mode,
                total_products, known_products, unknown_products,
                occupancy_ratio, row_occupancy_json,
                category_mix_json, subcategory_mix_json, avg_qwen_conf
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_ts,
                (store_id or "").strip() or None,
                (shelf_id or "").strip() or None,
                "full_sku" if run_qwen else "analyze",
                total_products,
                known_products,
                unknown_products,
                occupancy_ratio,
                json.dumps(row_occupancy),
                json.dumps(category_mix),
                json.dumps(subcategory_mix),
                avg_qwen_conf,
            ),
        )
        run_id = int(cur.lastrowid or 0)

        if run_qwen and run_id > 0:
            sku_rows = []
            inventory_rollup = {}
            store_key, shelf_key = _store_shelf_keys(store_id, shelf_id)
            for row in rows:
                sku_detail = _clean_label(row.get("sku_detail"))
                if not _is_persistable_sku_label(sku_detail):
                    continue
                qconf = row.get("qwen_confidence")
                category = _clean_label(row.get("product_category") or "unknown") or "unknown"
                subcategory = _clean_label(row.get("subcategory") or "unknown") or "unknown"
                sku_rows.append(
                    (
                        run_id,
                        run_ts,
                        (store_id or "").strip() or None,
                        (shelf_id or "").strip() or None,
                        int(row.get("crop_id") or 0),
                        sku_detail,
                        category,
                        subcategory,
                        float(qconf) if isinstance(qconf, (int, float)) else None,
                        1 if bool(row.get("semantic_mismatch_flag")) else 0,
                    )
                )

                inv_key = (store_key, shelf_key, sku_detail, category, subcategory)
                if inv_key not in inventory_rollup:
                    inventory_rollup[inv_key] = {"qty": 0, "qconfs": []}
                inventory_rollup[inv_key]["qty"] += 1
                if isinstance(qconf, (int, float)):
                    inventory_rollup[inv_key]["qconfs"].append(float(qconf))

            if sku_rows:
                conn.executemany(
                    """
                    INSERT INTO sku_detections (
                        run_id, created_at, store_id, shelf_id, crop_id,
                        sku_detail, product_category, subcategory,
                        qwen_confidence, semantic_mismatch_flag
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    sku_rows,
                )

                inventory_rows = []
                for key, payload in inventory_rollup.items():
                    sk_store, sk_shelf, sk_sku, sk_cat, sk_sub = key
                    qvals = payload["qconfs"]
                    avg_qconf = (sum(qvals) / len(qvals)) if qvals else None
                    inventory_rows.append(
                        (sk_store, sk_shelf, sk_sku, sk_cat, sk_sub, int(payload["qty"]), avg_qconf, run_ts)
                    )

                conn.executemany(
                    """
                    INSERT INTO sku_inventory (
                        store_id, shelf_id, sku_detail, product_category, subcategory,
                        quantity, avg_qwen_conf, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(store_id, shelf_id, sku_detail, product_category, subcategory)
                    DO UPDATE SET
                        quantity = sku_inventory.quantity + excluded.quantity,
                        avg_qwen_conf = COALESCE(excluded.avg_qwen_conf, sku_inventory.avg_qwen_conf),
                        last_updated = excluded.last_updated
                    """,
                    inventory_rows,
                )


def _get_inventory_quantity(sku_detail, category, subcategory, store_id=None, shelf_id=None):
    sku = _clean_label(sku_detail)
    category = _clean_label(category or "unknown") or "unknown"
    subcategory = _clean_label(subcategory or "unknown") or "unknown"
    if not _is_persistable_sku_label(sku):
        return None

    store_key, shelf_key = _store_shelf_keys(store_id, shelf_id)
    with _db_connect() as conn:
        row = conn.execute(
            """
            SELECT quantity
            FROM sku_inventory
            WHERE store_id = ? AND shelf_id = ? AND sku_detail = ? AND product_category = ? AND subcategory = ?
            """,
            (store_key, shelf_key, sku, category, subcategory),
        ).fetchone()

    if not row:
        return None
    return int(row[0] or 0)


def _decrement_inventory_quantity(sku_detail, category, subcategory, quantity=1, store_id=None, shelf_id=None):
    sku = _clean_label(sku_detail)
    category = _clean_label(category or "unknown") or "unknown"
    subcategory = _clean_label(subcategory or "unknown") or "unknown"
    if not _is_persistable_sku_label(sku):
        return None

    qty = max(1, int(quantity or 1))
    store_key, shelf_key = _store_shelf_keys(store_id, shelf_id)

    with _db_connect() as conn:
        row = conn.execute(
            """
            SELECT quantity
            FROM sku_inventory
            WHERE store_id = ? AND shelf_id = ? AND sku_detail = ? AND product_category = ? AND subcategory = ?
            """,
            (store_key, shelf_key, sku, category, subcategory),
        ).fetchone()
        if not row:
            return None

        current_qty = int(row[0] or 0)
        new_qty = max(0, current_qty - qty)
        conn.execute(
            """
            UPDATE sku_inventory
            SET quantity = ?, last_updated = ?
            WHERE store_id = ? AND shelf_id = ? AND sku_detail = ? AND product_category = ? AND subcategory = ?
            """,
            (new_qty, int(time.time()), store_key, shelf_key, sku, category, subcategory),
        )

    return {"before": current_qty, "after": new_qty, "decrement": qty}


def _load_recent_runs(limit=10, store_id=None, shelf_id=None):
    limit = max(1, int(limit))
    where = []
    params = []
    if store_id and str(store_id).strip():
        where.append("store_id = ?")
        params.append(str(store_id).strip())
    if shelf_id and str(shelf_id).strip():
        where.append("shelf_id = ?")
        params.append(str(shelf_id).strip())

    sql = (
        "SELECT id, created_at, store_id, shelf_id, run_mode, total_products, known_products, "
        "unknown_products, occupancy_ratio, row_occupancy_json, category_mix_json, subcategory_mix_json, avg_qwen_conf "
        "FROM shelf_runs "
    )
    if where:
        sql += "WHERE " + " AND ".join(where) + " "
    sql += "ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with _db_connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()

    return [dict(r) for r in rows]


def _load_low_stock_skus(threshold=2, limit=100, store_id=None, shelf_id=None):
    threshold = max(1, int(threshold or 1))
    limit = max(1, int(limit or 100))

    where = []
    params = []
    if store_id and str(store_id).strip():
        where.append("store_id = ?")
        params.append(str(store_id).strip())
    if shelf_id and str(shelf_id).strip():
        where.append("shelf_id = ?")
        params.append(str(shelf_id).strip())

    sql = (
        "SELECT sku_detail, product_category, subcategory, quantity, avg_qwen_conf, last_updated "
        "FROM sku_inventory "
    )
    if where:
        sql += "WHERE " + " AND ".join(where) + " "
    sql += (
        "AND " if where else "WHERE "
    )
    sql += (
        "quantity <= ? "
        "ORDER BY quantity ASC, last_updated DESC "
        "LIMIT ?"
    )
    params.extend([threshold, limit])

    with _db_connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


_init_analytics_db()


def _clean_label(value):
    return str(value or "").strip()


def _normalize_semantic_label(value):
    text = _clean_label(value).lower()
    if text in {"", "unknown", "none", "n/a", "na"}:
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    tokens = [tok for tok in text.split() if tok]
    normalized_tokens = []
    for token in tokens:
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        normalized_tokens.append(token)
    return " ".join(normalized_tokens)


def _semantic_match_labels(left, right):
    a = _normalize_semantic_label(left)
    b = _normalize_semantic_label(right)
    if not a or not b:
        return False
    if a == b:
        return True
    aset = set(a.split())
    bset = set(b.split())
    if not aset or not bset:
        return False
    overlap = len(aset & bset) / float(max(1, min(len(aset), len(bset))))
    return overlap >= 0.6


def _is_distinct_product_name(product, category, subcategory):
    product = _clean_label(product)
    if not product or product.lower() in {"unknown", "none", "n/a", "na"}:
        return False
    return product.lower() not in {_clean_label(category).lower(), _clean_label(subcategory).lower()}


def pad_box(x1, y1, x2, y2, img_w, img_h, pad=10):
    x1 = max(0, int(x1 - pad))
    y1 = max(0, int(y1 - pad))
    x2 = min(img_w, int(x2 + pad))
    y2 = min(img_h, int(y2 + pad))
    return x1, y1, x2, y2


def estimate_empty_space(boxes, img_w, img_h, max_dim=640):
    if len(boxes) == 0:
        return 1.0, 0.0

    scale = max(1, max(img_w, img_h) // max_dim)
    h = max(1, img_h // scale)
    w = max(1, img_w // scale)
    mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v / scale)) for v in box[:4]]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    covered = float(mask.sum())
    coverage = covered / (w * h)
    return 1.0 - coverage, coverage


def _empty_label_from_ratio(empty_ratio: float) -> str:
    if empty_ratio >= 0.55:
        return "High"
    if empty_ratio >= 0.25:
        return "Moderate"
    return "Low"


def estimate_shelf_occupancy(boxes, img_w, img_h, row_count=4, max_dim=640):
    row_count = max(1, min(int(row_count or 4), 8))

    if boxes is None or len(boxes) == 0:
        empty_ratio = 1.0
        return {
            "empty_ratio": empty_ratio,
            "coverage_ratio": 0.0,
            "empty_label": _empty_label_from_ratio(empty_ratio),
            "shelf_bbox": [0, 0, int(img_w), int(img_h)],
            "row_occupancy": [0.0 for _ in range(row_count)],
            "row_empty": [1.0 for _ in range(row_count)],
        }

    x1 = float(np.min(boxes[:, 0]))
    y1 = float(np.min(boxes[:, 1]))
    x2 = float(np.max(boxes[:, 2]))
    y2 = float(np.max(boxes[:, 3]))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    pad_x = max(8.0, bw * 0.08)
    pad_y = max(8.0, bh * 0.10)
    sx1 = int(max(0, np.floor(x1 - pad_x)))
    sy1 = int(max(0, np.floor(y1 - pad_y)))
    sx2 = int(min(img_w, np.ceil(x2 + pad_x)))
    sy2 = int(min(img_h, np.ceil(y2 + pad_y)))

    if sx2 <= sx1:
        sx2 = min(img_w, sx1 + 1)
    if sy2 <= sy1:
        sy2 = min(img_h, sy1 + 1)

    scale = max(1, max(img_w, img_h) // max_dim)
    h = max(1, img_h // scale)
    w = max(1, img_w // scale)
    mask = np.zeros((h, w), dtype=bool)

    for box in boxes:
        bx1, by1, bx2, by2 = [int(round(v / scale)) for v in box[:4]]
        bx1 = max(0, min(bx1, w))
        bx2 = max(0, min(bx2, w))
        by1 = max(0, min(by1, h))
        by2 = max(0, min(by2, h))
        if bx2 > bx1 and by2 > by1:
            mask[by1:by2, bx1:bx2] = True

    ssx1 = max(0, min(int(round(sx1 / scale)), w - 1))
    ssy1 = max(0, min(int(round(sy1 / scale)), h - 1))
    ssx2 = max(ssx1 + 1, min(int(round(sx2 / scale)), w))
    ssy2 = max(ssy1 + 1, min(int(round(sy2 / scale)), h))

    shelf_slice = mask[ssy1:ssy2, ssx1:ssx2]
    shelf_area = float(max(1, shelf_slice.size))
    covered = float(shelf_slice.sum())
    coverage = covered / shelf_area
    empty_ratio = 1.0 - coverage

    row_occupancy = []
    row_empty = []
    for i in range(row_count):
        ry1 = ssy1 + int((i * (ssy2 - ssy1)) / row_count)
        ry2 = ssy1 + int(((i + 1) * (ssy2 - ssy1)) / row_count)
        if ry2 <= ry1:
            ry2 = min(ssy2, ry1 + 1)
        row_slice = mask[ry1:ry2, ssx1:ssx2]
        row_area = float(max(1, row_slice.size))
        row_cov = float(row_slice.sum()) / row_area
        row_occupancy.append(row_cov)
        row_empty.append(1.0 - row_cov)

    return {
        "empty_ratio": empty_ratio,
        "coverage_ratio": coverage,
        "empty_label": _empty_label_from_ratio(empty_ratio),
        "shelf_bbox": [sx1, sy1, sx2, sy2],
        "row_occupancy": row_occupancy,
        "row_empty": row_empty,
    }


def _resize_for_inference(image: Image.Image, max_edge: int):
    if max_edge <= 0:
        return image, False
    w, h = image.size
    largest = max(w, h)
    if largest <= max_edge:
        return image, False
    scale = max_edge / float(largest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.Resampling.BILINEAR), True


def _prepare_boxes(raw_boxes, img_w, img_h, run_qwen=False):
    if raw_boxes is None or len(raw_boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), 0, 0

    image_area = float(max(1, img_w * img_h))
    filtered = []
    for box in raw_boxes:
        x1, y1, x2, y2 = box[:4]
        area = max(0.0, (x2 - x1) * (y2 - y1))
        if (area / image_area) >= MIN_BOX_AREA_RATIO:
            filtered.append(box)

    filtered_count = len(filtered)
    if not filtered:
        return np.empty((0, 4), dtype=np.float32), 0, 0

    filtered.sort(key=lambda b: max(0.0, (b[2] - b[0]) * (b[3] - b[1])), reverse=True)
    if run_qwen:
        prepared = filtered[: max(1, MAX_CROPS_SKU)]
        skipped_count = max(0, len(filtered) - len(prepared))
    else:
        # Analyze mode should run full-shelf detection over all filtered crops.
        prepared = filtered
        skipped_count = 0
    return np.asarray(prepared, dtype=np.float32), filtered_count, skipped_count


def _build_summary_html(rows, empty_ratio, empty_label, occupancy_metrics=None):
    unique_cats = {}
    unknown_items = []
    for r in rows:
        cat = r.get("product_category") or "unknown"
        unique_cats[cat] = unique_cats.get(cat, 0) + 1
        if cat.lower() == "unknown":
            unknown_items.append(r["crop_id"])

    total_items = len(rows)
    distinct_categories = len([cat for cat in unique_cats.keys() if cat.lower() != "unknown"])
    shelf_type = "Unknown"
    if distinct_categories == 1:
        shelf_type = "Category-specific"
    elif distinct_categories > 1:
        shelf_type = "Mixed"

    summary_lines = ["<h3>Store Management Summary</h3>"]
    summary_lines.append(
        f"<p>We detected <strong>{total_items}</strong> product{'s' if total_items != 1 else ''} on this shelf. "
        f"The shelf appears <strong>{shelf_type.lower()}</strong> with {distinct_categories} distinct category{'ies' if distinct_categories != 1 else ''}.</p>"
    )

    if unique_cats:
        sorted_cats = sorted(unique_cats.items(), key=lambda x: -x[1])
        category_lines = [f"{cnt} x {cat}" for cat, cnt in sorted_cats[:5]]
        summary_lines.append(f"<p><strong>Top categories:</strong> {', '.join(category_lines)}</p>")
    else:
        summary_lines.append("<p><strong>Top categories:</strong> none detected yet.</p>")


    retail_summary = summarize_retail_decisions(rows)
    top_products = retail_summary.get("top_products") or []
    if top_products:
        product_lines = [f"{cnt} x {name}" for name, cnt in top_products[:5]]
        summary_lines.append(f"<p><strong>Likely products:</strong> {', '.join(product_lines)}</p>")

    if rows:
        summary_lines.append(f"<p><strong>Resolver stack:</strong> SWIN + FAISS classification used for {len(rows)} crop{'s' if len(rows) != 1 else ''}.</p>")

    if unknown_items:
        summary_lines.append(f"<p><strong>Manual review needed:</strong> {len(unknown_items)} item{'s' if len(unknown_items) != 1 else ''} may be unclear and should be checked manually.</p>")
    else:
        summary_lines.append("<p><strong>Manual review needed:</strong> no unclear items were found.</p>")


    mismatch_count = sum(1 for r in rows if bool(r.get("semantic_mismatch_flag")))
    if mismatch_count > 0:
        summary_lines.append(f"<p><strong>Analyze vs Qwen mismatches:</strong> {mismatch_count} item{'s' if mismatch_count != 1 else ''} need label review.</p>")

    summary_lines.append(f"<p><strong>Estimated visible empty space:</strong> {empty_ratio*100:.0f}% ({empty_label}).</p>")
    if occupancy_metrics:
        row_occupancy = occupancy_metrics.get("row_occupancy") or []
        if row_occupancy:
            row_text = ", ".join([f"Row {idx+1}: {int(round(val*100))}%" for idx, val in enumerate(row_occupancy)])
            summary_lines.append(f"<p><strong>Row occupancy (planogram zones):</strong> {row_text}</p>")
            low_rows = [str(idx + 1) for idx, val in enumerate(row_occupancy) if val < 0.35]
            if low_rows:
                summary_lines.append(f"<p><strong>Under-utilized rows:</strong> {', '.join(low_rows)} (below 35% occupancy).</p>")
    return "\n".join(summary_lines)


def _build_kpi_html(rows, run_qwen=False):
    total_products = len(rows)
    unknown_count = sum(1 for row in rows if (row.get("product_category") or "unknown").lower() == "unknown")
    avg_confidence_text = "N/A"
    if run_qwen:
        confidence_values = [float(row.get("qwen_confidence")) for row in rows if isinstance(row.get("qwen_confidence"), (int, float))]
        if confidence_values:
            avg_confidence_text = f"{(sum(confidence_values) / len(confidence_values)) * 100:.0f}%"

    return (
        "<div class='kpi-grid'>"
        f"<div class='kpi-card'><div class='kpi-label'>Total Products</div><div class='kpi-value'>{total_products}</div></div>"
        f"<div class='kpi-card'><div class='kpi-label'>Unknown Items</div><div class='kpi-value'>{unknown_count}</div></div>"
        f"<div class='kpi-card'><div class='kpi-label'>Avg Qwen Confidence</div><div class='kpi-value'>{avg_confidence_text}</div></div>"
        "</div>"
    )


def _build_sku_table_html(rows, run_qwen=False):
    if not run_qwen:
        return "<p>Qwen SKU extraction is skipped in Analyze Shelf mode. Use the Full Shelf SKU button.</p>"
    if not rows:
        return "<p>No crops were available for SKU analysis.</p>"

    lines = [
        "<div class='sku-table-wrap'><table class='sku-table'>",
        "<thead><tr><th>Crop</th><th>SKU / Product</th><th>Analyze Category</th><th>Analyze Subcategory</th><th>Qwen Category</th><th>Qwen Subcategory</th><th>Semantic Match</th><th>Clues</th><th>Confidence</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        crop_id = row.get("crop_id")
        sku_detail = row.get("sku_detail") or "No SKU detected"
        category = row.get("product_category") or "unknown"
        subcategory = row.get("subcategory") or "unknown"
        analyze_category = row.get("analyze_category") or "unknown"
        analyze_subcategory = row.get("analyze_subcategory") or "unknown"
        match_label = "Matched"
        if bool(row.get("semantic_mismatch_flag")):
            match_label = "Mismatch"
        clues = row.get("qwen_clues") or []
        clues_text = "; ".join(clues) if clues else "-"
        confidence = row.get("qwen_confidence")
        confidence_label = row.get("qwen_confidence_label") or "n/a"
        confidence_text = "-"
        if isinstance(confidence, (int, float)):
            confidence_text = f"{confidence*100:.0f}% ({confidence_label})"

        lines.append(
            f"<tr><td>{crop_id}</td><td>{sku_detail}</td><td>{analyze_category}</td><td>{analyze_subcategory}</td><td>{category}</td><td>{subcategory}</td><td>{match_label}</td><td>{clues_text}</td><td>{confidence_text}</td></tr>"
        )

    lines.extend(["</tbody>", "</table></div>"])
    return "".join(lines)


def _build_detected_items_html(rows, run_qwen=False):
    list_html = "<h4>Detected items</h4>"
    if rows:
        list_html += "<details open><summary>Show / hide detected items</summary><ol>"
        for r in rows:
            cid = r["crop_id"]
            cat = r.get("product_category") or "unknown"
            sub = r.get("subcategory") or "unknown"
            sku_detail = r.get("sku_detail") or "No SKU detected"
            qwen_confidence = r.get("qwen_confidence")
            qwen_confidence_label = r.get("qwen_confidence_label")
            qwen_clues = r.get("qwen_clues") or []
            qwen_confidence_suffix = ""
            if isinstance(qwen_confidence, (int, float)):
                qwen_confidence_suffix = f" (confidence: {qwen_confidence*100:.0f}%, {qwen_confidence_label or 'n/a'})"
            clues_suffix = ""
            if qwen_clues:
                clues_suffix = f"<br><i>Clues:</i> {'; '.join(qwen_clues)}"
            mismatch_suffix = ""
            if run_qwen and bool(r.get("semantic_mismatch_flag")):
                acat = r.get("analyze_category") or "unknown"
                asub = r.get("analyze_subcategory") or "unknown"
                qcat = r.get("product_category") or "unknown"
                qsub = r.get("subcategory") or "unknown"
                mismatch_suffix = f"<br><b>Mismatch flagged:</b> Analyze({acat} / {asub}) vs Qwen({qcat} / {qsub})"
            if cat.lower() == "unknown":
                qwen_line = f"<br><i>Qwen 2.5 VL:</i> {sku_detail}{qwen_confidence_suffix}{clues_suffix}{mismatch_suffix}" if run_qwen else ""
                list_html += f"<li><b>Item {cid}</b> could not be confidently identified and should be reviewed.{qwen_line}</li>"
            else:
                retail_decision = (r.get("retail_product") or {}).get("decision") or {}
                product = retail_decision.get("predicted_product")
                retail_confidence = retail_decision.get("confidence")
                product_prefix = ""
                if _is_distinct_product_name(product, cat, sub):
                    product_prefix = f"<strong>{product}</strong>, "
                retail_confidence_suffix = ""
                if isinstance(retail_confidence, (int, float)):
                    retail_confidence_suffix = f" ({retail_confidence*100:.0f}% confidence)"
                qwen_line = f"<br><i>Qwen 2.5 VL:</i> {sku_detail}{qwen_confidence_suffix}{clues_suffix}{mismatch_suffix}" if run_qwen else ""
                list_html += f"<li><b>Item {cid}</b> is likely {product_prefix}<strong>{cat}</strong> with subcategory <strong>{sub}</strong>{retail_confidence_suffix}.{qwen_line}</li>"
        list_html += "</ol></details>"
    else:
        list_html += "<p>No product items were found in the image.</p>"
    return list_html


def _build_answer_html(question, rows, empty_ratio, empty_label):
    unique_cats = {}
    unknown_items = []
    for r in rows:
        cat = r.get("product_category") or "unknown"
        unique_cats[cat] = unique_cats.get(cat, 0) + 1
        if cat.lower() == "unknown":
            unknown_items.append(r["crop_id"])

    total_items = len(rows)
    distinct_categories = len([cat for cat in unique_cats.keys() if cat.lower() != "unknown"])
    shelf_type = "Unknown"
    if distinct_categories == 1:
        shelf_type = "Category-specific"
    elif distinct_categories > 1:
        shelf_type = "Mixed"

    category_lines = []
    if unique_cats:
        sorted_cats = sorted(unique_cats.items(), key=lambda x: -x[1])
        category_lines = [f"{cnt} x {cat}" for cat, cnt in sorted_cats[:5]]

    question_answers = {
        "How many products were detected on this shelf?": f"We detected {total_items} product{'s' if total_items != 1 else ''} on the shelf.",
        "Which are the top detected categories by count?": f"The top categories are {', '.join(category_lines)}." if unique_cats else "No clear product categories were identified.",
        "Which items need manual review?": f"{len(unknown_items)} item{'s' if len(unknown_items) != 1 else ''} may need manual review." if unknown_items else "No items require manual review at this time.",
        "Is this shelf mixed or category-specific?": f"This shelf appears to be {shelf_type.lower()}.",
        "How much empty space is visible on this shelf?": f"About {empty_ratio*100:.0f}% of the shelf appears empty ({empty_label}).",
    }

    selected_answer = question_answers.get(question, "Please upload an image and select a question.")
    return f"<p><b>{question}</b><br>{selected_answer}</p>"


def _build_sku_results_html(rows, run_qwen=False):
    sku_results_html = "<h4>Full shelf SKU results</h4>"
    if not run_qwen:
        sku_results_html += "<p>Qwen SKU extraction is skipped in Analyze Shelf mode. Use the Full Shelf SKU button.</p>"
    elif rows:
        sku_results_html += "<ol>"
        for r in rows:
            cid = r["crop_id"]
            sku_detail = r.get("sku_detail") or "No SKU detected"
            confidence = r.get("qwen_confidence")
            confidence_label = r.get("qwen_confidence_label")
            clues = r.get("qwen_clues") or []
            conf_text = f" (confidence: {confidence*100:.0f}%, {confidence_label or 'n/a'})" if isinstance(confidence, (int, float)) else ""
            clues_text = f"<br><i>Clues:</i> {'; '.join(clues)}" if clues else ""
            sku_results_html += f"<li><b>Crop {cid}</b>: {sku_detail}{conf_text}{clues_text}</li>"
        sku_results_html += "</ol>"
    else:
        sku_results_html += "<p>No crops were available for SKU analysis.</p>"
    return sku_results_html


def _build_progress_html(processed_count, total_count, run_qwen=False, status_text=None):
    total_count = max(0, int(total_count or 0))
    processed_count = max(0, min(int(processed_count or 0), total_count if total_count > 0 else 0))
    percent = int((processed_count / total_count) * 100) if total_count > 0 else 0
    mode_label = "Full SKU extraction" if run_qwen else "Shelf analysis"
    status_line = status_text or f"{mode_label} in progress..."

    return (
        "<div style='padding:10px;border:1px solid #d9e7de;border-radius:12px;background:#ffffffcc;'>"
        f"<div style='font-size:0.9rem;color:#334155;margin-bottom:6px;'><b>{mode_label}</b> - {processed_count}/{total_count} crops</div>"
        "<div style='height:10px;background:#e2ece7;border-radius:999px;overflow:hidden;'>"
        f"<div style='height:100%;width:{percent}%;background:linear-gradient(90deg,#0f766e,#14b8a6);transition:width .25s ease;'></div>"
        "</div>"
        f"<div style='margin-top:6px;font-size:0.85rem;color:#475569;'>{status_line}</div>"
        "</div>"
    )


def _build_analytics_overview_html(rows, empty_ratio, run_qwen=False, occupancy_metrics=None):
    total = len(rows)
    known = sum(1 for row in rows if (row.get("product_category") or "unknown").lower() != "unknown")
    unknown = max(0, total - known)
    occupancy = int(round((1.0 - float(empty_ratio)) * 100)) if total >= 0 else 0
    row_occupancy = []
    occupancy_source = "Analyze detections"
    if occupancy_metrics:
        row_occupancy = occupancy_metrics.get("row_occupancy") or []
        coverage = occupancy_metrics.get("coverage_ratio")
        if isinstance(coverage, (int, float)):
            occupancy = int(round(float(coverage) * 100))
        occupancy_source = occupancy_metrics.get("occupancy_source") or occupancy_source

    cat_counts = {}
    for row in rows:
        cat = row.get("product_category") or "unknown"
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    top_category = max(cat_counts.items(), key=lambda kv: kv[1])[0] if cat_counts else "N/A"

    avg_qwen = "N/A"
    if run_qwen:
        qvals = [float(r.get("qwen_confidence")) for r in rows if isinstance(r.get("qwen_confidence"), (int, float))]
        if qvals:
            avg_qwen = f"{(sum(qvals) / len(qvals))*100:.0f}%"

    rows_html = ""
    if row_occupancy:
        chips = " ".join([
            f"<span style='display:inline-block;margin:4px 6px 0 0;padding:4px 8px;border-radius:999px;background:#e8f6f1;border:1px solid #cce2d8;font-size:0.82rem;'>R{idx+1}: {int(round(val*100))}%</span>"
            for idx, val in enumerate(row_occupancy)
        ])
        rows_html = f"<div style='margin-top:8px;'><div class='kpi-label'>Row Occupancy</div>{chips}</div>"

    source_badge_html = (
        "<div style='margin-top:10px;'>"
        "<span style='display:inline-block;padding:5px 10px;border-radius:999px;"
        "background:#ecfdf5;border:1px solid #99f6e4;color:#115e59;font-size:0.82rem;'>"
        f"Occupancy source: {occupancy_source}</span></div>"
    )

    return (
        "<div class='kpi-grid'>"
        f"<div class='kpi-card'><div class='kpi-label'>Shelf Occupancy</div><div class='kpi-value'>{occupancy}%</div></div>"
        f"<div class='kpi-card'><div class='kpi-label'>Known vs Unknown</div><div class='kpi-value'>{known}/{unknown}</div></div>"
        f"<div class='kpi-card'><div class='kpi-label'>Top Category</div><div class='kpi-value'>{top_category}</div></div>"
        f"<div class='kpi-card'><div class='kpi-label'>Avg Qwen Confidence</div><div class='kpi-value'>{avg_qwen}</div></div>"
        "</div>"
        f"{source_badge_html}{rows_html}"
    )


def run_bi_query(selected_query, rows_state, store_id=None, shelf_id=None):
    if not rows_state:
        return "<p>No analysis data yet. Run Analyze shelf or Full SKU detection first.</p>"

    rows = rows_state
    cat_counts = {}
    subcat_counts = {}
    unknown_ids = []
    low_conf = []
    for row in rows:
        cid = row.get("crop_id")
        cat = row.get("product_category") or "unknown"
        sub = row.get("subcategory") or "unknown"
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        subcat_counts[sub] = subcat_counts.get(sub, 0) + 1
        if cat.lower() == "unknown":
            unknown_ids.append(cid)
        qconf = row.get("qwen_confidence")
        if isinstance(qconf, (int, float)) and qconf < 0.45:
            low_conf.append((cid, qconf))

    total = len(rows)
    unknown_pct = (len(unknown_ids) / total * 100.0) if total else 0.0

    row_count = max(1, min(OCCUPANCY_ROW_COUNT, 8))
    row_occupancy = []
    if rows:
        boxes = []
        for row in rows:
            box = row.get("box") or []
            if isinstance(box, list) and len(box) >= 4:
                boxes.append(box[:4])
        if boxes:
            b = np.asarray(boxes, dtype=np.float32)
            sx1, sy1 = float(np.min(b[:, 0])), float(np.min(b[:, 1]))
            sx2, sy2 = float(np.max(b[:, 2])), float(np.max(b[:, 3]))
            sw = max(1.0, sx2 - sx1)
            sh = max(1.0, sy2 - sy1)
            scale = max(1.0, max(sw, sh) / 640.0)
            mw = max(1, int(round(sw / scale)))
            mh = max(1, int(round(sh / scale)))
            mask = np.zeros((mh, mw), dtype=bool)
            for box in b:
                x1, y1, x2, y2 = box[:4]
                lx1 = int(max(0, min(mw, round((x1 - sx1) / scale))))
                lx2 = int(max(0, min(mw, round((x2 - sx1) / scale))))
                ly1 = int(max(0, min(mh, round((y1 - sy1) / scale))))
                ly2 = int(max(0, min(mh, round((y2 - sy1) / scale))))
                if lx2 > lx1 and ly2 > ly1:
                    mask[ly1:ly2, lx1:lx2] = True

            for i in range(row_count):
                ry1 = int((i * mh) / row_count)
                ry2 = int(((i + 1) * mh) / row_count)
                if ry2 <= ry1:
                    ry2 = min(mh, ry1 + 1)
                row_slice = mask[ry1:ry2, :]
                row_area = float(max(1, row_slice.size))
                row_cov = float(row_slice.sum()) / row_area
                row_occupancy.append(row_cov)

    if selected_query == "What are top categories by facings?":
        ordered = sorted(cat_counts.items(), key=lambda kv: -kv[1])
        lines = "".join([f"<li><b>{cat}</b>: {cnt}</li>" for cat, cnt in ordered[:8]])
        return f"<h4>Top categories by facings</h4><ol>{lines}</ol>"

    if selected_query == "Which products or crops need manual review?":
        if not unknown_ids and not low_conf:
            return "<h4>Manual review queue</h4><p>No crops currently need urgent review.</p>"
        unknown_line = f"<p><b>Unknown category crops:</b> {', '.join([str(i) for i in unknown_ids])}</p>" if unknown_ids else ""
        low_line = "<p><b>Low-confidence Qwen crops:</b> " + ", ".join([f"{cid} ({score*100:.0f}%)" for cid, score in low_conf]) + "</p>" if low_conf else ""
        return f"<h4>Manual review queue</h4>{unknown_line}{low_line}"

    if selected_query == "What is the unknown-rate and confidence risk?":
        avg_q = [float(r.get("qwen_confidence")) for r in rows if isinstance(r.get("qwen_confidence"), (int, float))]
        avg_q_text = f"{(sum(avg_q)/len(avg_q))*100:.0f}%" if avg_q else "N/A"
        return (
            "<h4>Risk snapshot</h4>"
            f"<p><b>Unknown-rate:</b> {unknown_pct:.1f}% ({len(unknown_ids)}/{total})</p>"
            f"<p><b>Average Qwen confidence:</b> {avg_q_text}</p>"
            f"<p><b>Low-confidence crops (&lt;45%):</b> {len(low_conf)}</p>"
        )

    if selected_query == "Show likely assortment mix (category and subcategory).":
        cat_ordered = sorted(cat_counts.items(), key=lambda kv: -kv[1])
        sub_ordered = sorted(subcat_counts.items(), key=lambda kv: -kv[1])
        cat_list = "".join([f"<li>{cat}: {cnt}</li>" for cat, cnt in cat_ordered[:8]])
        sub_list = "".join([f"<li>{sub}: {cnt}</li>" for sub, cnt in sub_ordered[:8]])
        return (
            "<h4>Assortment mix</h4>"
            f"<p><b>Category mix</b></p><ul>{cat_list}</ul>"
            f"<p><b>Subcategory mix</b></p><ul>{sub_list}</ul>"
        )

    if selected_query == "Give replenishment and planogram suggestions.":
        ordered = sorted(cat_counts.items(), key=lambda kv: -kv[1])
        lead = ordered[0][0] if ordered else "unknown"
        return (
            "<h4>Replenishment suggestions</h4>"
            f"<p>Prioritize replenishment for <b>{lead}</b> and review facings where unknown detections were found.</p>"
            "<p>For planogram hygiene: keep high-frequency categories centered, reduce mixed clutter, and re-check low-confidence crops.</p>"
        )

    if selected_query == "Which row is most under-stocked?":
        if not row_occupancy:
            return "<h4>Row occupancy insight</h4><p>Row-wise occupancy is not available yet.</p>"
        min_idx = int(np.argmin(np.asarray(row_occupancy)))
        min_val = row_occupancy[min_idx]
        chips = " ".join([
            f"<span style='display:inline-block;margin:4px 6px 0 0;padding:4px 8px;border-radius:999px;background:#eef8f4;border:1px solid #cce2d8;font-size:0.82rem;'>R{i+1}: {int(round(v*100))}%</span>"
            for i, v in enumerate(row_occupancy)
        ])
        return (
            "<h4>Row occupancy insight</h4>"
            f"<p><b>Most under-stocked row:</b> Row {min_idx + 1} at {int(round(min_val*100))}% occupancy.</p>"
            f"<div>{chips}</div>"
        )

    if selected_query == "Which row should be replenished first?":
        if not row_occupancy:
            return "<h4>Replenishment priority</h4><p>Row-wise occupancy is not available yet.</p>"
        order = np.argsort(np.asarray(row_occupancy))
        top_priority = int(order[0])
        next_priority = int(order[1]) if len(order) > 1 else int(order[0])
        return (
            "<h4>Replenishment priority</h4>"
            f"<p><b>First priority:</b> Row {top_priority + 1} ({int(round(row_occupancy[top_priority]*100))}% occupancy)</p>"
            f"<p><b>Next priority:</b> Row {next_priority + 1} ({int(round(row_occupancy[next_priority]*100))}% occupancy)</p>"
            "<p>Recommendation: replenish lowest-occupancy row first, then re-evaluate planogram compliance.</p>"
        )

    if selected_query == "Show Analyze vs Qwen mismatches (semantic).":
        mismatch_rows = []
        for row in rows:
            if not bool(row.get("semantic_mismatch_flag")):
                continue
            crop_id = row.get("crop_id")
            analyze_category = row.get("analyze_category") or "unknown"
            analyze_subcategory = row.get("analyze_subcategory") or "unknown"
            qwen_category = row.get("product_category") or "unknown"
            qwen_subcategory = row.get("subcategory") or "unknown"

            category_match = row.get("semantic_match_category")
            subcategory_match = row.get("semantic_match_subcategory")
            category_match_label = "Matched" if category_match else "Mismatch"
            subcategory_match_label = "Matched" if subcategory_match else "Mismatch"

            mismatch_type = []
            if not category_match:
                mismatch_type.append("Category")
            if not subcategory_match:
                mismatch_type.append("Subcategory")
            mismatch_type_text = " + ".join(mismatch_type) if mismatch_type else "Unknown"

            qconf = row.get("qwen_confidence")
            qconf_text = f"{qconf*100:.0f}%" if isinstance(qconf, (int, float)) else "N/A"

            mismatch_rows.append(
                "<tr>"
                f"<td>{crop_id}</td>"
                f"<td>{analyze_category}</td>"
                f"<td>{analyze_subcategory}</td>"
                f"<td>{qwen_category}</td>"
                f"<td>{qwen_subcategory}</td>"
                f"<td>{category_match_label}</td>"
                f"<td>{subcategory_match_label}</td>"
                f"<td>{mismatch_type_text}</td>"
                f"<td>{qconf_text}</td>"
                "</tr>"
            )

        if not mismatch_rows:
            return (
                "<h4>Analyze vs Qwen mismatches</h4>"
                "<p>No semantic mismatches found in current rows. "
                "Run Full SKU detection to populate semantic comparison fields.</p>"
            )

        return (
            "<h4>Analyze vs Qwen mismatches</h4>"
            f"<p><b>Total mismatches:</b> {len(mismatch_rows)}</p>"
            "<div class='sku-table-wrap'><table class='sku-table'>"
            "<thead><tr><th>Crop</th><th>Analyze Category</th><th>Analyze Subcategory</th><th>Qwen Category</th><th>Qwen Subcategory</th><th>Category Match</th><th>Subcategory Match</th><th>Mismatch Type</th><th>Qwen Confidence</th></tr></thead>"
            f"<tbody>{''.join(mismatch_rows)}</tbody></table></div>"
        )

    if selected_query == "Which SKUs are low stock? (SQLite, Qwen only)":
        low_stock_rows = _load_low_stock_skus(
            threshold=LOW_STOCK_FACING_THRESHOLD,
            limit=100,
            store_id=store_id,
            shelf_id=shelf_id,
        )

        if not low_stock_rows:
            return (
                "<h4>Low stock SKUs</h4>"
                f"<p>No SKUs are currently at or below the threshold of {LOW_STOCK_FACING_THRESHOLD} facings for this filter.</p>"
            )

        row_html = []
        for rec in low_stock_rows:
            sku = rec.get("sku_detail") or "unknown"
            cat = rec.get("product_category") or "unknown"
            sub = rec.get("subcategory") or "unknown"
            facing_count = int(rec.get("quantity") or 0)
            avg_conf = rec.get("avg_qwen_conf")
            avg_conf_text = f"{float(avg_conf)*100:.0f}%" if isinstance(avg_conf, (int, float)) else "N/A"
            last_seen = int(rec.get("last_updated") or 0)
            last_seen_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_seen)) if last_seen else "N/A"
            row_html.append(
                "<tr>"
                f"<td>{sku}</td>"
                f"<td>{cat}</td>"
                f"<td>{sub}</td>"
                f"<td>{facing_count}</td>"
                f"<td>{avg_conf_text}</td>"
                f"<td>{last_seen_text}</td>"
                "</tr>"
            )

        return (
            "<h4>Low stock SKUs</h4>"
            f"<p><b>Threshold:</b> {LOW_STOCK_FACING_THRESHOLD} units or fewer.</p>"
            "<div class='sku-table-wrap'><table class='sku-table'>"
            "<thead><tr><th>SKU / Product</th><th>Category</th><th>Subcategory</th><th>Current Quantity</th><th>Avg Qwen Confidence</th><th>Last Updated</th></tr></thead>"
            f"<tbody>{''.join(row_html)}</tbody></table></div>"
        )

    if selected_query == "Show recent run history (SQLite).":
        recent = _load_recent_runs(limit=10, store_id=store_id, shelf_id=shelf_id)
        if not recent:
            return "<h4>Recent run history</h4><p>No persisted runs found yet for this filter.</p>"

        rows_html = []
        for run in recent:
            ts = int(run.get("created_at") or 0)
            occ = float(run.get("occupancy_ratio") or 0.0)
            rows_html.append(
                "<tr>"
                f"<td>{run.get('id')}</td>"
                f"<td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}</td>"
                f"<td>{run.get('store_id') or '-'}</td>"
                f"<td>{run.get('shelf_id') or '-'}</td>"
                f"<td>{run.get('run_mode') or '-'}</td>"
                f"<td>{run.get('total_products') or 0}</td>"
                f"<td>{int(round(occ * 100))}%</td>"
                "</tr>"
            )

        return (
            "<h4>Recent run history</h4>"
            "<div class='sku-table-wrap'><table class='sku-table'>"
            "<thead><tr><th>ID</th><th>Timestamp</th><th>Store</th><th>Shelf</th><th>Mode</th><th>Products</th><th>Occupancy</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody></table></div>"
        )

    if selected_query == "How has occupancy changed recently?":
        recent = _load_recent_runs(limit=8, store_id=store_id, shelf_id=shelf_id)
        if len(recent) < 2:
            return "<h4>Occupancy trend</h4><p>Need at least two saved runs to compute trend.</p>"

        latest = recent[0]
        prev = recent[1]
        latest_occ = float(latest.get("occupancy_ratio") or 0.0)
        prev_occ = float(prev.get("occupancy_ratio") or 0.0)
        delta = (latest_occ - prev_occ) * 100.0
        trend = "up" if delta > 0.01 else "down" if delta < -0.01 else "flat"

        spark_points = []
        for run in reversed(recent):
            occ = int(round(float(run.get("occupancy_ratio") or 0.0) * 100))
            spark_points.append(str(occ))

        return (
            "<h4>Occupancy trend</h4>"
            f"<p><b>Latest occupancy:</b> {int(round(latest_occ*100))}%</p>"
            f"<p><b>Change vs previous run:</b> {delta:+.1f} percentage points ({trend})</p>"
            f"<p><b>Recent sequence (%):</b> {' → '.join(spark_points)}</p>"
        )

    return "<p>Select a BI query and click Run BI Query.</p>"


def _compose_stream_payload(annotated, rows, question, empty_ratio, empty_label, run_qwen=False, status_text=None, processed_count=0, total_count=0, occupancy_metrics=None):
    summary_html = _build_summary_html(rows, empty_ratio, empty_label, occupancy_metrics=occupancy_metrics)
    summary_html = f"{summary_html}\n{_build_detected_items_html(rows, run_qwen=run_qwen)}"
    answer_html = _build_answer_html(question, rows, empty_ratio, empty_label)
    if status_text:
        answer_html = f"<p><i>{status_text}</i></p>{answer_html}"

    progress_html = _build_progress_html(processed_count=processed_count, total_count=total_count, run_qwen=run_qwen, status_text=status_text)
    box_choices = [str(r["crop_id"]) for r in rows]
    initial_choice = box_choices[0] if box_choices else None
    sku_results_html = _build_sku_results_html(rows, run_qwen=run_qwen)
    kpi_html = _build_kpi_html(rows, run_qwen=run_qwen)
    sku_table_html = _build_sku_table_html(rows, run_qwen=run_qwen)
    analytics_html = _build_analytics_overview_html(rows, empty_ratio, run_qwen=run_qwen, occupancy_metrics=occupancy_metrics)

    return (
        annotated,
        summary_html,
        answer_html,
        progress_html,
        gr.update(choices=box_choices, value=initial_choice),
        rows,
        sku_results_html,
        kpi_html,
        sku_table_html,
        analytics_html,
    )


def process_image_stream(input_image, question, run_qwen=False, store_id=None, shelf_id=None):
    t0 = time.time()
    image = input_image.convert("RGB")
    image, resized_for_speed = _resize_for_inference(image, MAX_INPUT_EDGE)
    img_w, img_h = image.size

    results = yolo_model(image, conf=YOLO_CONF)
    t_yolo = time.time()
    print(f"[timing] YOLO inference took {t_yolo - t0:.3f}s")
    raw_boxes = results[0].boxes.xyxy.cpu().numpy()
    boxes, filtered_count, skipped_count = _prepare_boxes(raw_boxes, img_w, img_h, run_qwen=run_qwen)
    occupancy_boxes, _, _ = _prepare_boxes(raw_boxes, img_w, img_h, run_qwen=False)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    rows = []
    occupancy_metrics = estimate_shelf_occupancy(
        occupancy_boxes,
        img_w,
        img_h,
        row_count=OCCUPANCY_ROW_COUNT,
    )
    occupancy_metrics["occupancy_source"] = "Analyze detections"
    empty_ratio = float(occupancy_metrics.get("empty_ratio", 1.0))
    empty_label = occupancy_metrics.get("empty_label") or _empty_label_from_ratio(empty_ratio)

    shelf_bbox = occupancy_metrics.get("shelf_bbox") or [0, 0, img_w, img_h]
    if len(shelf_bbox) == 4:
        sx1, sy1, sx2, sy2 = [int(v) for v in shelf_bbox]
        draw.rectangle((sx1, sy1, sx2, sy2), outline="#2563eb", width=2)
        draw.text((sx1, max(0, sy1 - 12)), "shelf region", fill="#2563eb")

    initial_status = f"Detected {len(raw_boxes)} raw crop{'s' if len(raw_boxes) != 1 else ''}. Processing {len(boxes)} crop{'s' if len(boxes) != 1 else ''}."
    if run_qwen:
        initial_status += f" Occupancy uses Analyze-mode boxes ({len(occupancy_boxes)} crops) for stable shelf metrics."
    if skipped_count > 0:
        initial_status += f" Skipped {skipped_count} low-priority crop{'s' if skipped_count != 1 else ''} for faster response."
    if resized_for_speed:
        initial_status += " Image was resized for faster processing."
    initial_status += f" Starting {'full SKU extraction' if run_qwen else 'shelf analysis'}..."
    yield _compose_stream_payload(
        annotated,
        rows,
        question,
        empty_ratio,
        empty_label,
        run_qwen=run_qwen,
        status_text=initial_status,
        processed_count=0,
        total_count=len(boxes),
        occupancy_metrics=occupancy_metrics,
    )

    def _looks_like_path_label(label: str) -> bool:
        if not label:
            return False
        normalized = label.lower().replace("_", " ").strip()
        if normalized.startswith("train ") or normalized.startswith("train_"):
            return True
        if " crop " in normalized or normalized.endswith(" crop") or normalized.startswith("crop "):
            return True
        return False

    def _is_valid_label(label: str) -> bool:
        if not label:
            return False
        cleaned = label.strip()
        if cleaned.lower() in {"unknown", "n/a", "na", "none"}:
            return False
        if _looks_like_path_label(cleaned):
            return False
        return True

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, img_w, img_h, pad=12)

        crop = image.crop((px1, py1, px2, py2))
        t_before_swin = time.time()
        swin_result = None
        final_category = "unknown"
        subcategory_label = "unknown"

        if swin_classifier.is_ready():
            try:
                swin_result = swin_classifier.classify(crop, top_k=10, top_labels=5)
                t_swin = time.time()
                print(f"[timing] crop {i} Swin FAISS took {t_swin - t_before_swin:.3f}s")
                if i == 0:
                    print(f"[debug] crop {i} swin_result={swin_result}")

                swin_cat = swin_result.get("predicted_category") or swin_result.get("label")
                swin_subcat = swin_result.get("predicted_subcategory") or swin_result.get("best_subcategory")

                final_category = swin_cat if _is_valid_label(swin_cat) else "unknown"
                subcategory_label = swin_subcat if _is_valid_label(swin_subcat) else "unknown"
            except Exception as swin_exc:
                print(f"[warning] crop {i} Swin FAISS failed: {swin_exc}")

        t_before_retail = time.time()
        retail_product = resolve_retail_product(
            image=crop,
            swin_result=swin_result,
            category_hint=final_category,
            subcategory_hint=subcategory_label,
        )
        t_retail = time.time()
        print(f"[timing] crop {i} retail product resolver took {t_retail - t_before_retail:.3f}s")

        retail_decision = retail_product.get("decision") or {}
        if retail_decision.get("category") and retail_decision.get("category") != "unknown":
            final_category = retail_decision["category"]
        if retail_decision.get("subcategory") and retail_decision.get("subcategory") != "unknown":
            subcategory_label = retail_decision["subcategory"]

        if final_category == "unknown":
            swin_classifier.save_unknown_crop(crop, i + 1)

        sku_detail = "Qwen SKU detection not run in Analyze Shelf mode."
        qwen_confidence = None
        qwen_confidence_label = None
        qwen_clues = []
        qwen_category = None
        qwen_subcategory = None
        analyze_category = final_category
        analyze_subcategory = subcategory_label
        semantic_match_category = None
        semantic_match_subcategory = None
        semantic_mismatch_flag = None
        if run_qwen:
            sku_prompt = (
                "Identify the exact SKU, product name, brand, category, and subcategory visible in this crop. "
                "Respond in English only. Return a concise answer only."
            )
            sku_result = generate_qwen_sku_answer(image=crop, broad_category=final_category, user_prompt=sku_prompt)
            sku_detail = (sku_result.get("answer") or "").strip() or (sku_result.get("error") or "No SKU detected")
            qwen_confidence = sku_result.get("confidence")
            qwen_confidence_label = sku_result.get("confidence_label")
            structured = sku_result.get("structured") or {}
            qwen_clues = structured.get("clues") or []
            qwen_category = _clean_label(structured.get("category") or "")
            qwen_subcategory = _clean_label(structured.get("subcategory") or "")

            # Full SKU mode should use Qwen category/subcategory outputs.
            final_category = qwen_category if qwen_category else "unknown"
            subcategory_label = qwen_subcategory if qwen_subcategory else "unknown"

            semantic_match_category = _semantic_match_labels(analyze_category, final_category)
            semantic_match_subcategory = _semantic_match_labels(analyze_subcategory, subcategory_label)
            semantic_mismatch_flag = not (semantic_match_category and semantic_match_subcategory)

        draw.rectangle((px1, py1, px2, py2), outline="red", width=3)
        draw.text((px1, max(0, py1 - 12)), str(final_category), fill="red")

        rows.append(
            {
                "crop_id": i + 1,
                "box": [px1, py1, px2, py2],
                "product_category": final_category,
                "subcategory": subcategory_label,
                "swin": swin_result,
                "clip": None,
                "qwen_sku": None,
                "retail_product": retail_product,
                "crop_image": crop,
                "sku_detail": sku_detail,
                "qwen_confidence": qwen_confidence,
                "qwen_confidence_label": qwen_confidence_label,
                "qwen_clues": qwen_clues,
                "analyze_category": analyze_category,
                "analyze_subcategory": analyze_subcategory,
                "qwen_category": qwen_category,
                "qwen_subcategory": qwen_subcategory,
                "semantic_match_category": semantic_match_category,
                "semantic_match_subcategory": semantic_match_subcategory,
                "semantic_mismatch_flag": semantic_mismatch_flag,
            }
        )

        progress_status = f"Processed crop {i + 1}/{len(boxes)}."
        yield _compose_stream_payload(
            annotated,
            rows,
            question,
            empty_ratio,
            empty_label,
            run_qwen=run_qwen,
            status_text=progress_status,
            processed_count=i + 1,
            total_count=len(boxes),
            occupancy_metrics=occupancy_metrics,
        )

    final_status = f"Completed processing {len(rows)} crop{'s' if len(rows) != 1 else ''}."

    # Persist deterministic run metrics for cross-run BI without requiring an LLM or external DB service.
    _persist_run_snapshot(
        rows=rows,
        occupancy_metrics=occupancy_metrics,
        run_qwen=run_qwen,
        store_id=store_id,
        shelf_id=shelf_id,
    )

    yield _compose_stream_payload(
        annotated,
        rows,
        question,
        empty_ratio,
        empty_label,
        run_qwen=run_qwen,
        status_text=final_status,
        processed_count=len(rows),
        total_count=len(boxes),
        occupancy_metrics=occupancy_metrics,
    )


def analyze_shelf(input_image, question, store_id, shelf_id):
    yield from process_image_stream(
        input_image,
        question,
        run_qwen=False,
        store_id=store_id,
        shelf_id=shelf_id,
    )


def run_full_shelf_sku(input_image, question, store_id, shelf_id):
    yield from process_image_stream(
        input_image,
        question,
        run_qwen=True,
        store_id=store_id,
        shelf_id=shelf_id,
    )


def save_flagged_crop(selected_crop_id, rows_state, flag_reason, save_image):
    if not rows_state:
        return "<p>No detected boxes are available yet.</p>", None
    try:
        crop_id = int(selected_crop_id)
    except (TypeError, ValueError):
        return "<p>Please select a detected box first.</p>", None

    for row in rows_state:
        if int(row.get("crop_id", -1)) != crop_id:
            continue
        crop_image = row.get("crop_image")
        if crop_image is None:
            return "<p>No crop image is available to save.</p>", None
        out_dir = Path("flagged_crops")
        out_dir.mkdir(exist_ok=True)
        filename = f"crop_{crop_id}_{int(time.time())}.png"
        path = out_dir / filename
        crop_image.save(path)
        meta = {
            "crop_id": crop_id,
            "flag_reason": flag_reason or "unknown",
            "saved_path": str(path),
            "timestamp": int(time.time()),
        }
        with open(out_dir / "flags.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta) + "\n")
        return f"<p>Saved flagged crop to <b>{path}</b>.</p>", str(path)

    return f"<p>No detected box with id {crop_id} was found.</p>", None


def get_selected_crop_details(selected_crop_id, rows_state, store_id=None, shelf_id=None):
    if not rows_state:
        return None, "<p>No detected crops yet. Run analysis first.</p>"

    try:
        crop_id = int(selected_crop_id)
    except (TypeError, ValueError):
        return None, "<p>Select a crop id to view details.</p>"

    for row in rows_state:
        if int(row.get("crop_id", -1)) != crop_id:
            continue

        crop_image = row.get("crop_image")
        category = row.get("product_category") or "unknown"
        subcategory = row.get("subcategory") or "unknown"
        sku_detail = row.get("sku_detail") or "No SKU detected"
        confidence = row.get("qwen_confidence")
        confidence_label = row.get("qwen_confidence_label") or "n/a"
        clues = row.get("qwen_clues") or []
        clues_html = "".join([f"<li>{c}</li>" for c in clues]) if clues else "<li>None</li>"

        retail_decision = (row.get("retail_product") or {}).get("decision") or {}
        mapped_product = retail_decision.get("predicted_product") or "unknown"
        mapped_confidence = retail_decision.get("confidence")

        confidence_text = f"{confidence*100:.0f}% ({confidence_label})" if isinstance(confidence, (int, float)) else "Not available"
        mapped_confidence_text = f"{mapped_confidence*100:.0f}%" if isinstance(mapped_confidence, (int, float)) else "Not available"
        analyze_category = row.get("analyze_category") or "unknown"
        analyze_subcategory = row.get("analyze_subcategory") or "unknown"
        semantic_match_label = "Matched"
        if bool(row.get("semantic_mismatch_flag")):
            semantic_match_label = "Mismatch"
        inventory_qty = _get_inventory_quantity(sku_detail, category, subcategory, store_id=store_id, shelf_id=shelf_id)
        inventory_qty_text = str(inventory_qty) if isinstance(inventory_qty, int) else "Not tracked yet"

        details_html = (
            f"<h4>Crop {crop_id} mapping details</h4>"
            "<table class='sku-table'>"
            "<thead><tr><th>Field</th><th>Value</th></tr></thead>"
            "<tbody>"
            f"<tr><td>Mapped Product</td><td>{mapped_product}</td></tr>"
            f"<tr><td>Analyze Category</td><td>{analyze_category}</td></tr>"
            f"<tr><td>Analyze Subcategory</td><td>{analyze_subcategory}</td></tr>"
            f"<tr><td>Qwen Category</td><td>{category}</td></tr>"
            f"<tr><td>Qwen Subcategory</td><td>{subcategory}</td></tr>"
            f"<tr><td>Semantic Comparison</td><td>{semantic_match_label}</td></tr>"
            f"<tr><td>Qwen SKU / Product</td><td>{sku_detail}</td></tr>"
            f"<tr><td>Current DB Quantity</td><td>{inventory_qty_text}</td></tr>"
            f"<tr><td>Qwen Confidence</td><td>{confidence_text}</td></tr>"
            f"<tr><td>Resolver Confidence</td><td>{mapped_confidence_text}</td></tr>"
            "</tbody></table>"
            "<h5 style='margin-top:10px;'>Qwen visual clues</h5>"
            f"<ul>{clues_html}</ul>"
            "<p style='margin-top:8px;'><i>If this mapping looks wrong, add a reason and click Flag and save selected crop.</i></p>"
        )
        return crop_image, details_html

    return None, f"<p>No detected box with id {crop_id} was found.</p>"


def checkout_selected_crop(selected_crop_id, rows_state, checkout_quantity, store_id, shelf_id):
    if not rows_state:
        return "<p>No detected boxes are available yet.</p>"

    try:
        crop_id = int(selected_crop_id)
    except (TypeError, ValueError):
        return "<p>Please select a detected crop first.</p>"

    qty = max(1, int(checkout_quantity or 1))

    for row in rows_state:
        if int(row.get("crop_id", -1)) != crop_id:
            continue

        sku_detail = row.get("sku_detail") or ""
        category = row.get("product_category") or "unknown"
        subcategory = row.get("subcategory") or "unknown"

        if not _is_persistable_sku_label(sku_detail):
            return "<p>This crop does not have a valid Qwen SKU to checkout yet. Run Full SKU mode first.</p>"

        result = _decrement_inventory_quantity(
            sku_detail=sku_detail,
            category=category,
            subcategory=subcategory,
            quantity=qty,
            store_id=store_id,
            shelf_id=shelf_id,
        )
        if result is None:
            return (
                "<p>No inventory record found for this SKU in the selected store/shelf filter. "
                "Run Full SKU detection first to populate inventory.</p>"
            )

        warn = ""
        if int(result["after"]) <= LOW_STOCK_FACING_THRESHOLD:
            warn = f" <b>Low stock alert:</b> quantity is now {int(result['after'])}."

        return (
            f"<p>Checkout successful for crop <b>{crop_id}</b>. "
            f"SKU <b>{_clean_label(sku_detail)}</b> decremented by <b>{int(result['decrement'])}</b>. "
            f"Quantity: <b>{int(result['before'])}</b> -> <b>{int(result['after'])}</b>.{warn}</p>"
        )

    return f"<p>No detected box with id {crop_id} was found.</p>"


with gr.Blocks(
    title="Smart Shelf Management Dashboard",
) as demo:
    gr.Markdown(
        """
<div class=\"hero\">
  <h1>Smart Shelf Intelligence</h1>
    <p>Understand shelf performance, review products, and make better store decisions in one place.</p>
</div>
        """,
    )

    rows_state = gr.State(value=[])

    with gr.Tabs():
        with gr.Tab("Detection Studio"):
            gr.Markdown("<div class='tab-banner'><b>Detection workspace:</b> upload shelf images, run quick analysis, or run full SKU extraction.</div>")
            with gr.Row(elem_classes=["toolbar", "animate-in", "stagger-1"]):
                image_input = gr.Image(type="pil", label="Upload shelf image")
                question_input = gr.Dropdown(
                    choices=[
                        "How many products were detected on this shelf?",
                        "Which are the top detected categories by count?",
                        "Which items need manual review?",
                        "Is this shelf mixed or category-specific?",
                        "How much empty space is visible on this shelf?",
                    ],
                    value="How many products were detected on this shelf?",
                    label="Select a Store Management question",
                )
            with gr.Row(elem_classes=["toolbar", "animate-in", "stagger-1"]):
                store_id_input = gr.Textbox(label="Store ID", placeholder="e.g. store_001")
                shelf_id_input = gr.Textbox(label="Shelf ID", placeholder="e.g. snacks_aisle_left")
            with gr.Row(elem_classes=["toolbar", "animate-in", "stagger-2"]):
                analyze_button = gr.Button("Analyze shelf", variant="secondary")
                sku_button = gr.Button("Run full shelf SKU detection", variant="primary")

            gr.Markdown("<div class='section-title'>Live processing insights</div>")
            kpi_cards = gr.HTML(label="KPI Overview", elem_classes=["panel", "animate-in", "stagger-2"])
            output_progress = gr.HTML(label="Processing Progress", elem_classes=["panel", "animate-in", "stagger-2"])

            gr.Markdown("<div class='section-title'>Shelf outputs</div>")
            with gr.Row(elem_classes=["panel", "animate-in", "stagger-3"]):
                output_image = gr.Image(type="pil", label="Annotated image")
                output_summary = gr.HTML(label="Summary")
                output_answer = gr.HTML(label="Selected question answer")
            output_sku_results = gr.HTML(label="Full shelf SKU results", elem_id="sku-results", elem_classes=["panel", "animate-in", "stagger-3"])
            output_sku_table = gr.HTML(label="SKU Results Table", elem_classes=["panel", "animate-in", "stagger-4"])

        with gr.Tab("Crop Review & Flag"):
            gr.Markdown("<div class='tab-banner'><b>Review workspace:</b> select any crop, inspect Qwen mapping against the crop image, and optionally flag it.</div>")
            crop_selector = gr.Dropdown(choices=[], label="Select crop id to review/flag")
            with gr.Row(elem_classes=["panel", "animate-in", "stagger-4"]):
                selected_crop_image = gr.Image(type="pil", label="Selected crop preview")
                selected_crop_mapping = gr.HTML(label="Selected crop mapping details")
            with gr.Row(elem_classes=["toolbar", "animate-in", "stagger-4"]):
                flag_reason = gr.Textbox(label="Flag reason if SKU is unclear", placeholder="e.g. blurry, occluded, no visible barcode")
                save_button = gr.Button("Flag and save selected crop")
            with gr.Row(elem_classes=["toolbar", "animate-in", "stagger-4"]):
                checkout_quantity = gr.Number(value=1, precision=0, minimum=1, label="Checkout quantity")
                checkout_button = gr.Button("Checkout selected crop", variant="secondary")
            with gr.Row(elem_classes=["panel", "animate-in", "stagger-4"]):
                flagged_output = gr.HTML(label="Flag / save status")
                checkout_output = gr.HTML(label="Checkout status")
                download_flagged_crop = gr.File(label="Download saved flagged crop")

        with gr.Tab("Analytics & BI"):
            gr.Markdown("<div class='tab-banner'><b>Business intelligence workspace:</b> monitor shelf analytics and run management queries on current detection results.</div>")
            analytics_overview = gr.HTML(label="Store Analytics Snapshot", elem_classes=["panel", "animate-in", "stagger-2"])
            with gr.Row(elem_classes=["toolbar", "animate-in", "stagger-3"]):
                bi_query = gr.Dropdown(
                    choices=[
                        "What are top categories by facings?",
                        "Which products or crops need manual review?",
                        "What is the unknown-rate and confidence risk?",
                        "Show likely assortment mix (category and subcategory).",
                        "Give replenishment and planogram suggestions.",
                        "Which row is most under-stocked?",
                        "Which row should be replenished first?",
                        "Show Analyze vs Qwen mismatches (semantic).",
                        "Which SKUs are low stock? (SQLite, Qwen only)",
                        "Show recent run history (SQLite).",
                        "How has occupancy changed recently?",
                    ],
                    value="What are top categories by facings?",
                    label="Business Intelligence Query",
                )
                bi_button = gr.Button("Run BI Query", variant="primary")
            bi_query_output = gr.HTML(label="BI Query Result", elem_classes=["panel", "animate-in", "stagger-4"])

    analyze_button.click(
        analyze_shelf,
        inputs=[image_input, question_input, store_id_input, shelf_id_input],
        outputs=[output_image, output_summary, output_answer, output_progress, crop_selector, rows_state, output_sku_results, kpi_cards, output_sku_table, analytics_overview],
    )
    sku_button.click(
        run_full_shelf_sku,
        inputs=[image_input, question_input, store_id_input, shelf_id_input],
        outputs=[output_image, output_summary, output_answer, output_progress, crop_selector, rows_state, output_sku_results, kpi_cards, output_sku_table, analytics_overview],
    )
    save_button.click(
        save_flagged_crop,
        inputs=[crop_selector, rows_state, flag_reason, gr.State(value=True)],
        outputs=[flagged_output, download_flagged_crop],
    )
    crop_selector.change(
        get_selected_crop_details,
        inputs=[crop_selector, rows_state, store_id_input, shelf_id_input],
        outputs=[selected_crop_image, selected_crop_mapping],
    )
    checkout_button.click(
        checkout_selected_crop,
        inputs=[crop_selector, rows_state, checkout_quantity, store_id_input, shelf_id_input],
        outputs=[checkout_output],
    )
    bi_button.click(
        run_bi_query,
        inputs=[bi_query, rows_state, store_id_input, shelf_id_input],
        outputs=[bi_query_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
        theme=gr.themes.Base(primary_hue="teal", secondary_hue="cyan", neutral_hue="slate"),
        css=CUSTOM_CSS,
        show_error=True,
        inbrowser=False,
    )
