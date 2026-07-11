"""Fast Gradio upload UI for shelf detection + SWIN/FAISS result tables.

This app is intentionally focused on the slow/interactive part of the demo:
upload image -> analyze -> annotated image + detections table. Keep Streamlit for
analytics, BI, and inventory history.

Run from the repo root:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -m frontend.gradio_app
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Ensure the repo root is importable when this file runs as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import pandas as pd
from PIL import Image

from backend import inventory_db as db
from retrieval import pipeline

DEFAULT_PORT = int(os.getenv("GRADIO_PORT", "7860"))
DEFAULT_HOST = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
DEFAULT_SKU_BACKEND = os.getenv("SKU_BACKEND", "gemini")
DEFAULT_EXTRACT_SKU = os.getenv("GRADIO_EXTRACT_SKU_DEFAULT", "1") == "1"


def _default_sku_model() -> str:
    if DEFAULT_SKU_BACKEND == "vertex-model-garden":
        return os.getenv(
            "SKU_MODEL",
            os.getenv("VERTEX_MODEL_GARDEN_MODEL", "google/paligemma@paligemma-mix-448-float16"),
        )
    if DEFAULT_SKU_BACKEND == "openai-compatible":
        return os.getenv("SKU_MODEL", os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"))
    if DEFAULT_SKU_BACKEND == "dry-run":
        return os.getenv("SKU_MODEL", "dry-run")
    return os.getenv("SKU_MODEL", os.getenv("VERTEX_MODEL", "gemini-2.5-flash"))


SKU_COLUMNS = [
    "brand",
    "product_name",
    "sku_text",
    "visible_text",
    "package_size",
    "barcode",
    "sku_confidence",
    "sku_needs_review",
    "sku_latency_s",
    "sku_error",
]


def _display_columns(records: list[dict]) -> list[str]:
    preferred = [
        "crop_id",
        "category",
        "subcategory",
        "score",
        "area",
        "brand",
        "product_name",
        "sku_text",
        "visible_text",
        "package_size",
        "barcode",
        "sku_confidence",
        "sku_needs_review",
    ]
    if not records:
        return preferred[:5]
    available = set(records[0])
    return [column for column in preferred if column in available]


def preload_models() -> str:
    """Load heavy assets once so the first user click is faster."""
    started = time.time()
    try:
        pipeline.get_yolo()
        classifier = pipeline.get_classifier()
        ready = classifier.is_ready()
        return f"Models preloaded in {time.time() - started:.1f}s. SWIN+FAISS ready: {ready}."
    except Exception as exc:
        return f"Model preload failed: {exc}"


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


def _build_sku_backend(backend: str, model: str, endpoint: str):
    from autolabel.sku_vlm import build_backend

    args = SimpleNamespace(
        backend=backend,
        model=model,
        endpoint=endpoint or None,
        api_key=os.getenv("VLM_API_KEY", ""),
        project=os.getenv("PROJECT_ID", ""),
        location=os.getenv("REGION", "us-central1"),
        timeout=180,
        dedicated_dns=os.getenv("VERTEX_MODEL_GARDEN_DEDICATED_DNS", ""),
    )
    return build_backend(args)


def _enrich_records_with_sku(
    records: list[dict],
    image: Image.Image,
    backend_name: str,
    model: str,
    endpoint: str,
    max_sku_crops: int,
) -> tuple[list[dict], str]:
    from autolabel.sku_vlm import coerce_bool

    if not records:
        return records, "No detections to send to SKU/OCR."

    backend = _build_sku_backend(backend_name, model, endpoint)
    limit = len(records) if max_sku_crops <= 0 else min(max_sku_crops, len(records))
    image = image.convert("RGB")
    note = f"SKU/OCR backend `{backend_name}` processed {limit} crop(s)."

    with tempfile.TemporaryDirectory(prefix="gradio_sku_crops_") as tmp:
        tmp_dir = Path(tmp)
        for i, record in enumerate(records):
            record.update(_empty_sku_fields())
            if i >= limit:
                record["sku_error"] = "not_processed_limit"
                continue
            x1, y1, x2, y2 = record["box"]
            crop = image.crop((x1, y1, x2, y2))
            crop_path = tmp_dir / f"crop_{record['crop_id']:04d}.jpg"
            crop.save(crop_path, quality=92)
            pred = backend.predict(crop_path)
            parsed = pred.parsed
            print(
                "sku_ocr",
                f"crop_id={record['crop_id']}",
                f"backend={backend_name}",
                f"visible_text={parsed.get('visible_text', '')!r}",
                f"sku_text={parsed.get('sku_text', '')!r}",
                f"error={pred.error!r}",
                flush=True,
            )
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
                    "sku_latency_s": pred.latency_s,
                    "sku_error": pred.error,
                }
            )

    return records, note


def analyze_shelf(
    image: Image.Image | None,
    confidence: float,
    max_crops: int,
    extract_sku: bool,
    sku_backend: str,
    sku_model: str,
    sku_endpoint: str,
    max_sku_crops: int,
    save_to_inventory: bool,
) -> tuple[Image.Image | None, pd.DataFrame, str]:
    if image is None:
        return None, pd.DataFrame(), "Upload a shelf image first."

    started = time.time()
    image = image.convert("RGB")
    result = pipeline.analyze_image(image, conf=confidence, max_crops=max_crops)
    records = pipeline.detections_to_records(result)

    sku_note = ""
    if extract_sku:
        try:
            records, sku_note = _enrich_records_with_sku(
                records,
                image,
                sku_backend,
                sku_model,
                sku_endpoint,
                max_sku_crops,
            )
        except BaseException as exc:
            sku_note = f"SKU/OCR failed: {exc}"
            for record in records:
                record.update(_empty_sku_fields())
                record["sku_error"] = str(exc)
    else:
        sku_note = "SKU/OCR was not run. Keep `Extract SKU/OCR with VLM` checked for SKU text."
        for record in records:
            record.update(_empty_sku_fields())
            record["sku_error"] = "sku_disabled"

    scan_note = ""
    if save_to_inventory:
        scan_id = db.save_scan(result, "gradio_upload", records)
        scan_note = f"\n\nSaved scan #{scan_id} to inventory history."

    df = pd.DataFrame(records)
    if not df.empty:
        df = df[_display_columns(records)]

    sku_summary = f"\n\n{sku_note}" if sku_note else ""
    summary = (
        f"Products detected: {result.num_items}\n\n"
        f"Distinct categories: {result.distinct_categories}\n\n"
        f"Empty shelf space: {result.empty_pct * 100:.0f}% ({result.empty_label})\n\n"
        f"Shelf type: {result.shelf_type}\n\n"
        f"Needs review: {result.review_count}\n\n"
        f"YOLO: {result.timings.get('yolo_s')}s | "
        f"classify: {result.timings.get('classify_s')}s | "
        f"boxes: {result.timings.get('boxes')} | "
        f"total: {time.time() - started:.2f}s"
        f"{sku_summary}"
        f"{scan_note}"
    )
    return result.annotated_image, df, summary


def build_app() -> gr.Blocks:
    db.init_db()

    with gr.Blocks(title="Fast Shelf Upload") as demo:
        gr.Markdown(
            "# Fast Shelf Upload\n"
            "Upload a shelf image, run YOLO + SWIN/FAISS, and inspect the result table. "
            "Use the Streamlit dashboard for analytics, BI, and history."
        )

        preload_status = gr.Markdown("Model preload has not run yet.")
        with gr.Row():
            preload_button = gr.Button("Preload models", variant="secondary")
            analyze_button = gr.Button("Analyze shelf", variant="primary")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Shelf image")
                confidence = gr.Slider(
                    minimum=0.05,
                    maximum=0.90,
                    value=0.25,
                    step=0.05,
                    label="YOLO confidence",
                )
                max_crops = gr.Slider(
                    minimum=0,
                    maximum=300,
                    value=60,
                    step=10,
                    label="Max products to classify (0 = all)",
                )
                extract_sku = gr.Checkbox(
                    value=DEFAULT_EXTRACT_SKU,
                    label="Extract SKU/OCR with VLM",
                )
                sku_backend = gr.Dropdown(
                    choices=["gemini", "dry-run", "openai-compatible", "vertex-model-garden"],
                    value=DEFAULT_SKU_BACKEND,
                    label="SKU backend",
                )
                sku_model = gr.Textbox(
                    value=_default_sku_model(),
                    label="SKU model",
                )
                sku_endpoint = gr.Textbox(
                    value=os.getenv(
                        "VLM_ENDPOINT_URL",
                        os.getenv("VERTEX_MODEL_GARDEN_ENDPOINT_ID", ""),
                    ),
                    label="SKU endpoint",
                    placeholder="Only needed for openai-compatible or vertex-model-garden",
                )
                max_sku_crops = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=5,
                    step=1,
                    label="Max SKU/OCR crops (0 = all detections)",
                )
                save_to_inventory = gr.Checkbox(
                    value=True,
                    label="Save scan to inventory history",
                )
            with gr.Column(scale=2):
                annotated_output = gr.Image(type="pil", label="Annotated shelf")
                summary_output = gr.Markdown(label="Summary")

        detections_output = gr.Dataframe(
            label="Detected items",
            interactive=False,
            wrap=True,
        )

        preload_button.click(preload_models, outputs=preload_status)
        analyze_button.click(
            analyze_shelf,
            inputs=[
                image_input,
                confidence,
                max_crops,
                extract_sku,
                sku_backend,
                sku_model,
                sku_endpoint,
                max_sku_crops,
                save_to_inventory,
            ],
            outputs=[annotated_output, detections_output, summary_output],
        )

        if os.getenv("GRADIO_PRELOAD_MODELS", "1") == "1":
            demo.load(preload_models, outputs=preload_status)

    return demo


if __name__ == "__main__":
    build_app().launch(
        server_name=DEFAULT_HOST,
        server_port=DEFAULT_PORT,
        share=False,
    )
