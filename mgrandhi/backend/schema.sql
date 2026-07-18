PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku_code TEXT NOT NULL UNIQUE,
    barcode TEXT UNIQUE,
    brand TEXT,
    product_name TEXT NOT NULL,
    category TEXT,
    subcategory TEXT,
    package_size TEXT,
    unit_price REAL NOT NULL DEFAULT 0.0,
    tax_rate REAL NOT NULL DEFAULT 0.0,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS product_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    alias_text TEXT NOT NULL,
    alias_type TEXT NOT NULL DEFAULT 'ocr',
    confidence REAL,
    source TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
    UNIQUE (product_id, alias_text, alias_type)
);

CREATE TABLE IF NOT EXISTS shelf_scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    image_name TEXT,
    image_path TEXT,
    store_id TEXT,
    shelf_id TEXT,
    camera_id TEXT,
    captured_at TEXT,
    processed_at TEXT NOT NULL DEFAULT (datetime('now')),
    num_items INTEGER NOT NULL DEFAULT 0,
    detector_box_count INTEGER NOT NULL DEFAULT 0,
    distinct_categories INTEGER NOT NULL DEFAULT 0,
    empty_pct REAL,
    shelf_type TEXT,
    review_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS detected_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL,
    product_id INTEGER,
    crop_id INTEGER,
    crop_path TEXT,
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    detector_confidence REAL,
    retrieval_category TEXT,
    retrieval_subcategory TEXT,
    retrieval_score REAL,
    vlm_brand TEXT,
    vlm_product_name TEXT,
    vlm_sku_text TEXT,
    vlm_visible_text TEXT,
    vlm_confidence REAL,
    route TEXT NOT NULL DEFAULT 'retrieval',
    review_status TEXT NOT NULL DEFAULT 'auto',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (scan_id) REFERENCES shelf_scans(id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS detection_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_item_id INTEGER NOT NULL,
    feedback_type TEXT NOT NULL CHECK (feedback_type IN ('category', 'sku')),
    verdict TEXT NOT NULL CHECK (verdict IN ('correct', 'incorrect')),
    correction TEXT,
    note TEXT,
    source_image_path TEXT,
    crop_image_path TEXT,
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    predicted_category TEXT,
    predicted_subcategory TEXT,
    predicted_sku_text TEXT,
    predicted_visible_text TEXT,
    predicted_sku_confidence REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (detected_item_id) REFERENCES detected_items(id) ON DELETE CASCADE,
    UNIQUE (detected_item_id, feedback_type)
);

CREATE TABLE IF NOT EXISTS inventory_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    scan_id INTEGER,
    quantity INTEGER NOT NULL,
    confidence REAL,
    source TEXT NOT NULL DEFAULT 'shelf_scan',
    observed_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
    FOREIGN KEY (scan_id) REFERENCES shelf_scans(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS checkout_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT NOT NULL UNIQUE,
    store_id TEXT,
    device_id TEXT,
    customer_id TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    subtotal REAL NOT NULL DEFAULT 0.0,
    tax REAL NOT NULL DEFAULT 0.0,
    total REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS checkout_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkout_session_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    detected_item_id INTEGER,
    quantity INTEGER NOT NULL DEFAULT 1,
    unit_price REAL NOT NULL DEFAULT 0.0,
    tax_rate REAL NOT NULL DEFAULT 0.0,
    line_total REAL NOT NULL DEFAULT 0.0,
    confidence REAL,
    review_status TEXT NOT NULL DEFAULT 'auto',
    FOREIGN KEY (checkout_session_id) REFERENCES checkout_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE RESTRICT,
    FOREIGN KEY (detected_item_id) REFERENCES detected_items(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT NOT NULL UNIQUE,
    model_name TEXT NOT NULL,
    provider TEXT NOT NULL,
    endpoint TEXT,
    prompt_version TEXT NOT NULL,
    sample_size INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    json_parse_rate REAL,
    error_rate REAL,
    avg_latency_s REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_run_id INTEGER NOT NULL,
    detected_item_id INTEGER,
    crop_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    raw_response TEXT,
    parse_ok INTEGER NOT NULL DEFAULT 0,
    brand TEXT,
    product_name TEXT,
    sku_text TEXT,
    visible_text TEXT,
    package_size TEXT,
    barcode TEXT,
    category_hint TEXT,
    confidence REAL,
    needs_review INTEGER NOT NULL DEFAULT 1,
    latency_s REAL,
    error TEXT,
    FOREIGN KEY (model_run_id) REFERENCES model_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (detected_item_id) REFERENCES detected_items(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_products_sku_code ON products(sku_code);
CREATE INDEX IF NOT EXISTS idx_products_barcode ON products(barcode);
CREATE INDEX IF NOT EXISTS idx_product_aliases_text ON product_aliases(alias_text);
CREATE INDEX IF NOT EXISTS idx_shelf_scans_processed_at ON shelf_scans(processed_at);
CREATE INDEX IF NOT EXISTS idx_detected_items_scan ON detected_items(scan_id);
CREATE INDEX IF NOT EXISTS idx_detected_items_product ON detected_items(product_id);
CREATE INDEX IF NOT EXISTS idx_detected_items_review_status ON detected_items(review_status);
CREATE INDEX IF NOT EXISTS idx_detection_feedback_detection
    ON detection_feedback(detected_item_id);
CREATE INDEX IF NOT EXISTS idx_detection_feedback_verdict
    ON detection_feedback(feedback_type, verdict);
CREATE INDEX IF NOT EXISTS idx_inventory_snapshots_product_time
    ON inventory_snapshots(product_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_checkout_items_session ON checkout_items(checkout_session_id);
CREATE INDEX IF NOT EXISTS idx_model_predictions_run ON model_predictions(model_run_id);
CREATE INDEX IF NOT EXISTS idx_model_predictions_filename ON model_predictions(filename);
