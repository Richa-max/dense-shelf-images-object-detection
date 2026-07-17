from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
YOLO_WEIGHTS = REPO_ROOT / "models" / "yolo" / "best.pt"
LABELED_INDEX_METADATA = REPO_ROOT / "train_product_category_58.csv"
INDEXED_IMAGE_PATHS = REPO_ROOT / "swin_faiss_indexed_image_paths.csv"


@dataclass
class AnalysisResult:
    annotated_image: Image.Image
    items: list[dict[str, Any]]
    num_items: int
    distinct_categories: int
    empty_pct: float
    empty_label: str
    shelf_type: str
    review_count: int
    timings: dict[str, Any]


@lru_cache(maxsize=1)
def _classifier():
    from swin_faiss import SwinFaissClassifier

    image_paths_path = LABELED_INDEX_METADATA if LABELED_INDEX_METADATA.exists() else INDEXED_IMAGE_PATHS
    return SwinFaissClassifier(
        model_dir=str(REPO_ROOT / "swin_model_assets"),
        processor_dir=str(REPO_ROOT / "swin_processor_assets"),
        index_path=str(REPO_ROOT / "swin_faiss_index.bin"),
        image_paths_path=str(image_paths_path),
        indexed_image_paths_npy=str(REPO_ROOT / "swin_faiss_indexed_image_paths.npy"),
    )


@lru_cache(maxsize=1)
def _yolo_model():
    from ultralytics import YOLO

    return YOLO(str(YOLO_WEIGHTS))


def classifier_ready() -> bool:
    try:
        return _classifier().is_ready()
    except Exception:
        return False


def _pad_box(box, img_w: int, img_h: int, pad: int = 12) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(img_w, x2 + pad),
        min(img_h, y2 + pad),
    ]


def _label_from_swin(swin_result: dict[str, Any]) -> tuple[str, str, float]:
    category = (
        swin_result.get("predicted_category")
        or swin_result.get("label")
        or "unknown"
    )
    subcategory = (
        swin_result.get("predicted_subcategory")
        or swin_result.get("best_subcategory")
        or "unknown"
    )
    score = float(swin_result.get("score") or 0.0)
    return str(category or "unknown"), str(subcategory or "unknown"), score


def _draw_items(
    image: Image.Image,
    items: list[dict[str, Any]],
    hidden_boxes: list[list[int]] | None = None,
) -> Image.Image:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for box in hidden_boxes or []:
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle((x1, y1, x2, y2), fill=(31, 41, 55), outline=(148, 163, 184), width=2)
        label = "checked out"
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        tx = x1 + max(4, ((x2 - x1) - text_w) // 2)
        ty = y1 + max(4, ((y2 - y1) - text_h) // 2)
        draw.text((tx, ty), label, fill=(226, 232, 240), font=font)
    for item in items:
        x1, y1, x2, y2 = item["box"]
        label = item["category"]
        draw.rectangle((x1, y1, x2, y2), outline=(34, 197, 94), width=3)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        draw.rectangle((x1, max(0, y1 - text_h - 6), x1 + text_w + 8, y1), fill=(17, 24, 39))
        draw.text((x1 + 4, max(0, y1 - text_h - 4)), label, fill=(255, 255, 255), font=font)
    return annotated


def analyze_image(image: Image.Image, conf: float = 0.25, max_crops: int = 60) -> AnalysisResult:
    image = image.convert("RGB")
    timings: dict[str, Any] = {}

    t0 = time.time()
    yolo_results = _yolo_model()(image, conf=conf)
    timings["yolo_s"] = round(time.time() - t0, 3)

    boxes = yolo_results[0].boxes.xyxy.cpu().numpy() if yolo_results else []
    if max_crops and max_crops > 0:
        boxes = boxes[:max_crops]
    timings["boxes"] = int(len(boxes))

    items: list[dict[str, Any]] = []
    total_area = 0
    t1 = time.time()
    classifier = _classifier()
    if not classifier.is_ready():
        raise RuntimeError("SWIN+FAISS classifier is not ready. Check model/index assets.")

    for idx, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = _pad_box(box, image.width, image.height)
        crop = image.crop((x1, y1, x2, y2))
        swin_result = classifier.classify(crop, top_k=10, top_labels=5)
        category, subcategory, score = _label_from_swin(swin_result)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        total_area += area
        items.append(
            {
                "crop_id": idx,
                "box": [x1, y1, x2, y2],
                "category": category,
                "subcategory": subcategory,
                "score": score,
                "area": area,
                "swin": swin_result,
            }
        )
    timings["classify_s"] = round(time.time() - t1, 3)

    categories = [item["category"] for item in items if item["category"].lower() != "unknown"]
    occupied_pct = min(1.0, total_area / float(max(1, image.width * image.height)))
    empty_pct = max(0.0, 1.0 - occupied_pct)
    review_count = sum(1 for item in items if item["category"].lower() == "unknown")
    review_count += sum(1 for item in items if float(item.get("score") or 0.0) <= 0)

    return AnalysisResult(
        annotated_image=_draw_items(image, items),
        items=items,
        num_items=len(items),
        distinct_categories=len(set(categories)),
        empty_pct=empty_pct,
        empty_label="high" if empty_pct >= 0.55 else "moderate" if empty_pct >= 0.25 else "low",
        shelf_type=Counter(categories).most_common(1)[0][0] if categories else "unknown",
        review_count=review_count,
        timings=timings,
    )


def detections_to_records(result: AnalysisResult) -> list[dict[str, Any]]:
    return [
        {
            "crop_id": item["crop_id"],
            "category": item["category"],
            "subcategory": item["subcategory"],
            "score": item["score"],
            "area": item["area"],
            "box": item["box"],
        }
        for item in result.items
    ]


def redraw_annotated_image(
    image: Image.Image,
    items: list[dict[str, Any]],
    hidden_boxes: list[list[int]] | None = None,
) -> Image.Image:
    return _draw_items(image.convert("RGB"), items, hidden_boxes=hidden_boxes)
