"""Application service for one operator-facing shelf analysis."""
from __future__ import annotations

import base64
import io
import logging
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from PIL import Image

from mgrandhi.autolabel.sku_vlm import coerce_bool
from mgrandhi.backend import inventory_db as db
from mgrandhi.retrieval import pipeline

ProgressCallback = Callable[[str, int, str], None]
logger = logging.getLogger(__name__)


def _progress(callback: ProgressCallback | None, stage: str, percent: int, message: str) -> None:
    if callback:
        callback(stage, percent, message)


def _empty_sku_fields() -> dict:
    return {
        "brand": "",
        "product_name": "",
        "sku_text": "",
        "visible_text": "",
        "package_size": "",
        "barcode": "",
        "sku_confidence": 0.0,
        "sku_needs_review": 1,
        "sku_latency_s": 0.0,
        "sku_error": "",
    }


@lru_cache(maxsize=1)
def get_sku_backend():
    """Build the configured SKU reader once per inference process."""
    from autolabel.sku_vlm import build_backend

    backend = os.getenv("SKU_BACKEND", "gemini")
    default_model = {
        "dry-run": "dry-run",
        "gemini": os.getenv("VERTEX_MODEL", "gemini-2.5-flash"),
        "openai-compatible": os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
        "vertex-model-garden": os.getenv(
            "VERTEX_MODEL_GARDEN_MODEL", "google/paligemma@paligemma-mix-448-float16"
        ),
    }.get(backend, os.getenv("VERTEX_MODEL", "gemini-2.5-flash"))
    args = SimpleNamespace(
        backend=backend,
        model=os.getenv("SKU_MODEL", default_model),
        endpoint=os.getenv(
            "VLM_ENDPOINT_URL", os.getenv("VERTEX_MODEL_GARDEN_ENDPOINT_ID", "")
        )
        or None,
        api_key=os.getenv("VLM_API_KEY", ""),
        project=os.getenv("PROJECT_ID", "") or None,
        location=os.getenv("REGION", "us-central1"),
        timeout=int(os.getenv("SKU_TIMEOUT_SECONDS", "180")),
        dedicated_dns=os.getenv("VERTEX_MODEL_GARDEN_DEDICATED_DNS", ""),
    )
    return build_backend(args)


def preload_models() -> dict:
    """Warm the large assets once so the first scan does not appear stalled."""
    pipeline.get_yolo()
    classifier = pipeline.get_classifier()
    sku_ready = False
    if os.getenv("SKU_EXTRACT_DEFAULT", "1") == "1":
        try:
            get_sku_backend()
            sku_ready = True
        except Exception:
            sku_ready = False
    return {"detector_ready": True, "classifier_ready": classifier.is_ready(), "sku_ready": sku_ready}


def _enrich_with_sku(
    records: list[dict],
    image: Image.Image,
    callback: ProgressCallback | None,
    max_sku_crops: int | None = None,
) -> list[dict]:
    if not records:
        return records

    if max_sku_crops is None:
        max_sku_crops = int(os.getenv("MAX_SKU_CROPS", "5"))
    limit = len(records) if max_sku_crops <= 0 else min(max_sku_crops, len(records))
    backend = get_sku_backend()

    with tempfile.TemporaryDirectory(prefix="retail_sku_") as tmp:
        tmp_dir = Path(tmp)
        for index, record in enumerate(records):
            record.update(_empty_sku_fields())
            if index >= limit:
                record["sku_error"] = "not_processed_limit"
                continue
            x1, y1, x2, y2 = record["box"]
            crop_path = tmp_dir / f"crop_{record['crop_id']:04d}.jpg"
            image.crop((x1, y1, x2, y2)).save(crop_path, quality=92)
            prediction = backend.predict(crop_path)
            parsed = prediction.parsed
            record.update(
                {
                    "brand": parsed.get("brand", ""),
                    "product_name": parsed.get("product_name", ""),
                    "sku_text": parsed.get("sku_text", ""),
                    "visible_text": parsed.get("visible_text", ""),
                    "package_size": parsed.get("package_size", ""),
                    "barcode": parsed.get("barcode", ""),
                    "sku_confidence": parsed.get("confidence", 0.0),
                    "sku_needs_review": int(coerce_bool(parsed.get("needs_review", True))),
                    "sku_latency_s": prediction.latency_s,
                    "sku_error": prediction.error,
                }
            )
            percent = 68 + round(((index + 1) / limit) * 22)
            _progress(
                callback,
                "reading",
                percent,
                f"Reading product details {index + 1} of {limit}",
            )
    return records


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _persist_scan_artifacts(
    scan_id: int,
    image: Image.Image,
    records: list[dict],
) -> tuple[str, dict[int, str]]:
    """Persist review evidence outside SQLite and return its paths."""
    default_root = Path(__file__).resolve().parents[1] / "data" / "review_evidence"
    scan_dir = Path(os.getenv("FEEDBACK_ASSET_DIR", str(default_root))) / f"scan_{scan_id}"
    scan_dir.mkdir(parents=True, exist_ok=True)

    source_path = scan_dir / "source.jpg"
    image.save(source_path, format="JPEG", quality=92)
    crop_paths: dict[int, str] = {}
    for record in records:
        x1, y1, x2, y2 = record["box"]
        crop_id = int(record["crop_id"])
        crop_path = scan_dir / f"crop_{crop_id:04d}.jpg"
        image.crop((x1, y1, x2, y2)).save(crop_path, format="JPEG", quality=92)
        crop_paths[crop_id] = str(crop_path)
    return str(source_path), crop_paths


def analyze_shelf(
    image: Image.Image,
    filename: str,
    callback: ProgressCallback | None = None,
    max_crops: int | None = None,
    max_sku_crops: int | None = None,
    extract_sku: bool | None = None,
) -> dict:
    """Run the complete scan while reporting operator-friendly progress."""
    image = image.convert("RGB")
    _progress(callback, "detecting", 10, "Finding products on the shelf")

    def pipeline_progress(stage: str, current: int, total: int) -> None:
        if stage == "detected":
            message = (
                f"Found {current} products; identifying the first {total}"
                if current > total
                else f"Found {current} products to identify"
            )
            _progress(callback, "identifying", 24, message)
        elif total:
            percent = 24 + round((current / total) * 41)
            _progress(
                callback,
                "identifying",
                percent,
                f"Identifying product {current} of {total}",
            )

    result = pipeline.analyze_image(
        image,
        conf=float(os.getenv("YOLO_CONFIDENCE", "0.25")),
        max_crops=(
            int(os.getenv("MAX_CLASSIFICATION_CROPS", "60"))
            if max_crops is None
            else max_crops
        ),
        progress_callback=pipeline_progress,
    )
    records = pipeline.detections_to_records(result)
    _progress(callback, "identifying", 65, "Identifying product categories")

    sku_warning = ""
    should_extract_sku = (
        os.getenv("SKU_EXTRACT_DEFAULT", "1") == "1"
        if extract_sku is None
        else extract_sku
    )
    if should_extract_sku:
        try:
            records = _enrich_with_sku(records, image, callback, max_sku_crops)
        except Exception as exc:
            sku_warning = "Some package details could not be read."
            for record in records:
                record.update(_empty_sku_fields())
                record["sku_error"] = str(exc)
    else:
        for record in records:
            record.update(_empty_sku_fields())
            record["sku_error"] = "sku_disabled"

    _progress(callback, "saving", 94, "Preparing your shelf report")
    scan_id = db.save_scan(result, filename, records)
    artifact_warning = ""
    source_path: str | None = None
    try:
        source_path, crop_paths = _persist_scan_artifacts(scan_id, image, records)
        db.attach_scan_artifacts(scan_id, source_path, crop_paths)
    except Exception:
        logger.exception("Could not preserve review evidence for scan %s", scan_id)
        artifact_warning = "Review images could not be preserved."
        if source_path:
            shutil.rmtree(Path(source_path).parent, ignore_errors=True)
    _progress(callback, "complete", 100, "Shelf report ready")

    return {
        "scan_id": scan_id,
        "image_name": filename,
        "annotated_image": _image_data_url(result.annotated_image),
        "summary": {
            "num_items": result.num_items,
            "distinct_categories": result.distinct_categories,
            "empty_pct": result.empty_pct,
            "empty_label": result.empty_label,
            "shelf_type": result.shelf_type,
            "review_count": result.review_count,
        },
        "detections": records,
        "timings": result.timings,
        "warning": " ".join(message for message in (sku_warning, artifact_warning) if message),
    }
