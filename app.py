import os

import gradio as gr
from PIL import Image, ImageDraw
import io
import base64
import numpy as np
from ultralytics import YOLO

from efficientnet_model import classify_pil_image, load_model_if_needed
from clip_model import classify_with_clip_pil
import json
import time
import os

# load default subcategories mapping if present
_SUBCATS = {}
_SC_PATH = os.path.join(os.path.dirname(__file__), "subcategories.json")
if os.path.exists(_SC_PATH):
    try:
        with open(_SC_PATH, "r", encoding="utf-8") as fh:
            _SUBCATS = json.load(fh)
    except Exception:
        _SUBCATS = {}
from llava4_model import generate_llava4_answer


yolo_model = YOLO("models/yolo/best.pt")

# Preload the EfficientNet model at startup to avoid first-request latency.
try:
    load_model_if_needed()
except Exception:
    # model loading may happen on-demand; don't crash the app if preload fails
    pass


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


def decide_backend_action(pred: dict, user_requested_llava: bool = False, fine_grained: bool = False) -> dict:
    top1 = float(pred.get("confidence", 0.0))
    gap = float(pred.get("confidence_gap", 0.0))

    if fine_grained:
        return {
            "action": "call_llava_now",
            "reason": "user_requested_fine_grained",
            "should_call_llava4": True,
        }

    if user_requested_llava:
        return {
            "action": "call_llava_now",
            "reason": "user_requested_llava4",
            "should_call_llava4": True,
        }

    if top1 >= 0.80 and gap >= 0.20:
        return {
            "action": "accept_efficientnet",
            "reason": "high_confidence_and_clear_margin",
            "should_call_llava4": False,
        }

    if top1 >= 0.60 and gap < 0.20:
        return {
            "action": "defer_to_user",
            "reason": "uncertain_small_margin",
            "should_call_llava4": False,
        }

    if 0.60 <= top1 < 0.80:
        return {
            "action": "defer_to_user",
            "reason": "medium_confidence",
            "should_call_llava4": False,
        }

    return {
        "action": "call_llava_now",
        "reason": "low_confidence",
        "should_call_llava4": True,
    }


def process_image(input_image, question):
    t0 = time.time()
    image = input_image.convert("RGB")
    img_w, img_h = image.size

    # Preload EfficientNet once.
    load_model_if_needed()

    results = yolo_model(image, conf=0.25)
    t_yolo = time.time()
    print(f"[timing] YOLO inference took {t_yolo - t0:.3f}s")
    boxes = results[0].boxes.xyxy.cpu().numpy()

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    rows = []
    # Flatten default mapping candidates for CLIP fallback when EfficientNet is uncertain.
    all_clip_candidates = []
    for v in _SUBCATS.values():
        all_clip_candidates.extend(v)

    empty_ratio, covered_ratio = estimate_empty_space(boxes, img_w, img_h)
    empty_label = "High" if empty_ratio >= 0.55 else "Moderate" if empty_ratio >= 0.25 else "Low"

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, img_w, img_h, pad=12)

        crop = image.crop((px1, py1, px2, py2))
        t_before_eff = time.time()
        efficientnet_pred = classify_pil_image(crop)

        efficientnet_category = efficientnet_pred.get("label") or "unknown"
        eff_label = efficientnet_category.strip()
        eff_unknown = (not eff_label) or eff_label.lower() in {"unknown", "other / unclear", "other", "unclear"}

        clip_result = None
        final_category = efficientnet_category

        # If EfficientNet is unclear, try CLIP fallback on default mapped candidates.
        if eff_unknown and all_clip_candidates:
            try:
                t_before_clip = time.time()
                clip_result = classify_with_clip_pil(crop, all_clip_candidates)
                t_clip = time.time()
                print(f"[timing] crop {i} CLIP took {t_clip - t_before_clip:.3f}s")
                clip_label = (clip_result or {}).get("label") if isinstance(clip_result, dict) else None
                if clip_label and clip_label != "unknown":
                    final_category = clip_label
                else:
                    final_category = "unknown"
            except Exception as clip_exc:
                print(f"[warning] crop {i} CLIP failed: {clip_exc}")
                final_category = "unknown"
        elif eff_unknown:
            final_category = "unknown"

        # Always generate subcategory via LLaVA4 for the detected category.
        llava_sub_result = None
        subcategory_label = "unknown"
        try:
            t_before_sub = time.time()
            llava_sub_result = generate_llava4_answer(
                image=crop,
                broad_category=final_category,
                user_prompt=None,
            )
            t_sub = time.time()
            print(f"[timing] crop {i} LLaVA4(subcategory) took {t_sub - t_before_sub:.3f}s")
            if isinstance(llava_sub_result, dict) and llava_sub_result.get("answer"):
                subcategory_label = llava_sub_result.get("answer")
        except Exception as sub_exc:
            llava_sub_result = {"error": "llava4_failed", "details": str(sub_exc)}
            subcategory_label = "unknown"

        draw.rectangle((px1, py1, px2, py2), outline="red", width=3)
        draw.text((px1, max(0, py1 - 12)), str(final_category), fill="red")

        rows.append(
            {
                "crop_id": i + 1,
                "box": [px1, py1, px2, py2],
                "product_category": final_category,
                "subcategory": subcategory_label,
                "efficientnet": efficientnet_pred,
                "clip": clip_result,
                "llava_subcategory": llava_sub_result,
            }
        )

    # Build synthesized summary (user-friendly)
    unique_cats = {}
    unknown_items = []
    for r in rows:
        cat = r.get("product_category") or "unknown"
        unique_cats[cat] = unique_cats.get(cat, 0) + 1
        if cat.lower() == "unknown":
            unknown_items.append(r["crop_id"])

    # Summarize counts and build simple HTML (no hover/tooltips)
    summary_lines = []
    total_items = len(rows)
    num_distinct = len([c for c in unique_cats.keys() if c.lower() != "unknown"])
    summary_lines.append("<h3>Store Management Summary</h3>")
    summary_lines.append(f"<p><b>1. How many products were detected on this shelf?</b> {total_items}</p>")

    # Top categories by count
    if unique_cats:
        sorted_cats = sorted(unique_cats.items(), key=lambda x: -x[1])
        top_categories = [f"{cat} ({cnt})" for cat, cnt in sorted_cats if cat.lower() != "unknown"]
        if not top_categories:
            top_categories = ["Unknown"]
        summary_lines.append(f"<p><b>2. Top detected categories by count:</b> {', '.join(top_categories[:5])}</p>")
    else:
        summary_lines.append("<p><b>2. Top detected categories by count:</b> none</p>")

    # Manual review flag
    if unknown_items:
        summary_lines.append(f"<p><b>3. Items needing manual review:</b> Unknown products at crop IDs {', '.join(map(str, unknown_items))}</p>")
    else:
        summary_lines.append("<p><b>3. Items needing manual review:</b> none detected</p>")

    # Mixed vs category-specific
    distinct_categories = len([cat for cat in unique_cats.keys() if cat.lower() != "unknown"])
    if distinct_categories == 0:
        shelf_type = "Unknown"
    elif distinct_categories == 1:
        shelf_type = "Category-specific"
    else:
        shelf_type = "Mixed"
    summary_lines.append(f"<p><b>4. Shelf type:</b> {shelf_type}</p>")

    # Empty space estimate
    summary_lines.append(
        f"<p><b>5. Estimated visible empty space:</b> {empty_ratio*100:.0f}% ({empty_label}) based on YOLO bounding boxes.</p>"
    )

    question_answers = {
        "How many products were detected on this shelf?": f"{total_items}",
        "Which are the top detected categories by count?": ", ".join(top_categories[:5]) if unique_cats else "None",
        "Which items need manual review?": (
            f"Unknown products at crop IDs {', '.join(map(str, unknown_items))}"
            if unknown_items
            else "None detected"
        ),
        "Is this shelf mixed or category-specific?": shelf_type,
        "How much empty space is visible on this shelf?": f"{empty_ratio*100:.0f}% ({empty_label})",
    }

    summary_lines.append("<h4>Detected categories</h4>")
    if unique_cats:
        summary_lines.append("<ul>")
        for cat, cnt in sorted(unique_cats.items(), key=lambda x: -x[1]):
            summary_lines.append(f"<li><b>{cat}</b>: {cnt} item(s)</li>")
        summary_lines.append("</ul>")
    else:
        summary_lines.append("<p>No products detected.</p>")

    # Per-crop details
    summary_lines.append("<h4>Crop details</h4>")
    summary_lines.append("<ol>")
    for r in rows:
        cid = r["crop_id"]
        cat = r.get("product_category") or "unknown"
        sub = r.get("subcategory") or "unknown"
        detail = f"Crop {cid}: <b>Product Category:</b> {cat}; <b>Product Subcategory:</b> {sub}"
        summary_lines.append(f"<li>{detail}</li>")
    summary_lines.append("</ol>")

    summary_html = "\n".join(summary_lines)

    selected_answer = question_answers.get(question, "Please upload an image and select a question.")
    answer_html = f"<p><b>{question}</b><br>{selected_answer}</p>"
    return annotated, summary_html, answer_html


demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload shelf image"),
        gr.Dropdown(
            choices=[
                "How many products were detected on this shelf?",
                "Which are the top detected categories by count?",
                "Which items need manual review?",
                "Is this shelf mixed or category-specific?",
                "How much empty space is visible on this shelf?",
            ],
            value="How many products were detected on this shelf?",
            label="Select a Store Management question",
        ),
    ],
    outputs=[
        gr.Image(type="pil", label="Annotated image"),
        gr.HTML(label="Summary"),
        gr.HTML(label="Selected question answer"),
    ],
    title="Smart Shelf Management Dashboard",
    description="Upload a shelf image to detect product categories and subcategories and answer key store management questions.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))