import io
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mgrandhi.backend import api


@pytest.fixture(autouse=True)
def disableRealStartup(monkeypatch):
    monkeypatch.setenv("PRELOAD_MODELS", "0")
    monkeypatch.setattr(api.db, "init_db", lambda: None)


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(output, format="PNG")
    return output.getvalue()


def _report() -> dict:
    return {
        "scan_id": 4,
        "image_name": "shelf.png",
        "annotated_image": "data:image/jpeg;base64,test",
        "summary": {
            "num_items": 1,
            "distinct_categories": 1,
            "empty_pct": 0.2,
            "empty_label": "Low",
            "shelf_type": "Category-specific",
            "review_count": 0,
        },
        "detections": [],
        "timings": {"yolo_s": 0.1},
        "warning": "",
    }


def test_givenValidShelfImage_whenSubmittingAnalysis_thenJobCompletes(monkeypatch):
    captured_settings = {}

    def fake_analyze(image, filename, callback, **settings):
        captured_settings.update(settings)
        return _report()

    monkeypatch.setattr(
        api.analysis_service,
        "analyze_shelf",
        fake_analyze,
    )

    with TestClient(api.app) as client:
        accepted = client.post(
            "/api/analyses",
            files={"image": ("shelf.png", _png_bytes(), "image/png")},
            data={"max_crops": "40", "max_sku_crops": "8", "extract_sku": "true"},
        )
        assert accepted.status_code == 202
        job_id = accepted.json()["job_id"]

        deadline = time.time() + 2
        payload = {}
        while time.time() < deadline:
            response = client.get(f"/api/analyses/{job_id}")
            payload = response.json()
            if payload["status"] == "complete":
                break
            time.sleep(0.01)

    assert payload["status"] == "complete"
    assert payload["result"]["scan_id"] == 4
    assert captured_settings == {
        "max_crops": 40,
        "max_sku_crops": 8,
        "extract_sku": True,
    }


def test_givenNonImageUpload_whenSubmittingAnalysis_thenFriendlyValidationError(monkeypatch):
    with TestClient(api.app) as client:
        response = client.post(
            "/api/analyses",
            files={"image": ("notes.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 415
    assert "JPG, PNG, BMP, or WebP" in response.json()["detail"]


def test_givenSavedInventory_whenRequestingInsights_thenReturnsChartReadyData(monkeypatch):
    scans = pd.DataFrame(
        [
            {
                "id": 1,
                "ts": "2026-07-17T10:00:00",
                "image_name": "shelf.jpg",
                "num_items": 2,
                "distinct_categories": 1,
                "empty_pct": 0.2,
                "shelf_type": "Category-specific",
                "review_count": 0,
            }
        ]
    )
    items = pd.DataFrame(
        [
            {"category": "soft drinks", "subcategory": "cola"},
            {"category": "soft drinks", "subcategory": "cola"},
        ]
    )
    monkeypatch.setattr(api.db, "get_scans_df", lambda: scans)
    monkeypatch.setattr(api.db, "get_items_df", lambda: items)
    monkeypatch.setattr(api.db, "get_feedback_df", lambda: pd.DataFrame())

    with TestClient(api.app) as client:
        response = client.get("/api/insights")

    assert response.status_code == 200
    assert response.json()["categories"] == [{"category": "soft drinks", "count": 2}]
    assert response.json()["subcategories"] == [{"subcategory": "cola", "count": 2}]
    assert response.json()["summary"]["total_items"] == 2


def test_givenSavedEvidencePaths_whenRequestingScan_thenServerPathsAreNotExposed(monkeypatch):
    scans = pd.DataFrame(
        [{"id": 1, "image_name": "shelf.jpg", "image_path": "/private/source.jpg"}]
    )
    items = pd.DataFrame(
        [{"scan_id": 1, "crop_id": 1, "crop_path": "/private/crop.jpg", "box": "[1,2,3,4]"}]
    )
    monkeypatch.setattr(api.db, "get_scans_df", lambda: scans)
    monkeypatch.setattr(api.db, "get_items_df", lambda scan_id: items)

    with TestClient(api.app) as client:
        response = client.get("/api/scans/1")

    assert response.status_code == 200
    assert "image_path" not in response.json()["scan"]
    assert "crop_path" not in response.json()["items"][0]


def test_givenHumanVerdict_whenSubmittingFeedback_thenItIsSaved(monkeypatch):
    captured = {}

    def fake_save_feedback(**values):
        captured.update(values)
        return 12

    monkeypatch.setattr(api.db, "save_feedback", fake_save_feedback)

    with TestClient(api.app) as client:
        response = client.post(
            "/api/feedback",
            json={
                "scan_id": 4,
                "crop_id": 2,
                "feedback_type": "sku",
                "verdict": "incorrect",
                "correction": "SKU-123",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "saved"
    assert captured["correction"] == "SKU-123"


def test_givenUnknownApiRoute_whenRequestingIt_thenReturnsJsonNotSpa(monkeypatch):
    with TestClient(api.app) as client:
        response = client.get("/api/not-a-real-endpoint")

    assert response.status_code == 404
    assert response.json()["detail"] == "API endpoint not found."
