import sqlite3
from types import SimpleNamespace

import pytest

from mgrandhi.backend import inventory_db


def _save_scan(db_path: str) -> int:
    result = SimpleNamespace(
        num_items=1,
        distinct_categories=1,
        empty_pct=0.2,
        shelf_type="Category-specific",
        review_count=0,
    )
    records = [
        {
            "crop_id": 1,
            "category": "soft drinks",
            "subcategory": "cola",
            "score": 0.1,
            "area": 100,
            "box": [1, 2, 10, 20],
            "sku_text": "ACME COLA",
            "sku_needs_review": 0,
        }
    ]
    return inventory_db.save_scan(result, "shelf.jpg", records, db_path)


def test_givenDetectedProduct_whenSavingCategoryAndSkuFeedback_thenBothArePersisted(tmp_path):
    db_path = str(tmp_path / "inventory.db")
    scan_id = _save_scan(db_path)
    source_path = str(tmp_path / "source.jpg")
    crop_path = str(tmp_path / "crop_0001.jpg")
    inventory_db.attach_scan_artifacts(scan_id, source_path, {1: crop_path}, db_path)

    inventory_db.save_feedback(
        scan_id, 1, "category", "incorrect", "soft drinks > cola", db_path=db_path
    )
    inventory_db.save_feedback(scan_id, 1, "sku", "correct", db_path=db_path)

    feedback = inventory_db.get_feedback_df(db_path)
    assert len(feedback) == 2
    assert set(feedback["feedback_type"]) == {"category", "sku"}
    assert feedback.loc[feedback["feedback_type"] == "category", "correction"].iloc[0] == (
        "soft drinks > cola"
    )
    assert set(feedback["source_image_path"]) == {source_path}
    assert set(feedback["crop_image_path"]) == {crop_path}
    assert set(feedback["box"]) == {"[1, 2, 10, 20]"}
    assert set(feedback["predicted_category"]) == {"soft drinks"}
    assert set(feedback["predicted_sku_text"]) == {"ACME COLA"}


def test_givenExistingFeedback_whenSubmittingReplacement_thenLatestVerdictIsKept(tmp_path):
    db_path = str(tmp_path / "inventory.db")
    scan_id = _save_scan(db_path)
    inventory_db.save_feedback(scan_id, 1, "sku", "incorrect", "SKU-123", db_path=db_path)

    inventory_db.save_feedback(scan_id, 1, "sku", "correct", db_path=db_path)

    feedback = inventory_db.get_feedback_df(db_path)
    assert len(feedback) == 1
    assert feedback.iloc[0]["verdict"] == "correct"
    assert feedback.iloc[0]["correction"] == ""


def test_givenUnknownProduct_whenSavingFeedback_thenRaisesValueError(tmp_path):
    db_path = str(tmp_path / "inventory.db")
    inventory_db.init_db(db_path)

    with pytest.raises(ValueError, match="does not exist"):
        inventory_db.save_feedback(999, 1, "category", "incorrect", db_path=db_path)


def test_givenOlderDatabase_whenInitializing_thenEvidenceColumnsAreAddedWithoutDataLoss(tmp_path):
    db_path = str(tmp_path / "inventory.db")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE scans (
                id INTEGER PRIMARY KEY, ts TEXT NOT NULL, image_name TEXT, num_items INTEGER,
                distinct_categories INTEGER, empty_pct REAL, shelf_type TEXT, review_count INTEGER
            );
            CREATE TABLE items (
                id INTEGER PRIMARY KEY, scan_id INTEGER NOT NULL, crop_id INTEGER,
                category TEXT, subcategory TEXT, score REAL, area INTEGER, box TEXT
            );
            CREATE TABLE item_feedback (
                id INTEGER PRIMARY KEY, scan_id INTEGER NOT NULL, crop_id INTEGER NOT NULL,
                feedback_type TEXT NOT NULL, verdict TEXT NOT NULL, correction TEXT,
                note TEXT, ts TEXT NOT NULL, UNIQUE (scan_id, crop_id, feedback_type)
            );
            INSERT INTO scans VALUES (1, '2026-07-17', 'legacy.jpg', 1, 1, 0.1, 'Mixed', 0);
            INSERT INTO items VALUES (1, 1, 1, 'legacy', 'item', 0.2, 100, '[1,2,3,4]');
            INSERT INTO item_feedback
                VALUES (1, 1, 1, 'category', 'correct', '', '', '2026-07-17');
            """
        )

    inventory_db.init_db(db_path)

    scans = inventory_db.get_scans_df(db_path)
    items = inventory_db.get_items_df(db_path=db_path)
    feedback = inventory_db.get_feedback_df(db_path)
    assert scans.iloc[0]["image_name"] == "legacy.jpg"
    assert "image_path" in scans.columns
    assert "crop_path" in items.columns
    assert feedback.iloc[0]["verdict"] == "correct"
    assert "predicted_category" in feedback.columns
