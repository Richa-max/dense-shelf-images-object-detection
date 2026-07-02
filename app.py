import os

import gradio as gr
from PIL import Image, ImageDraw
import io
import base64
import numpy as np
from ultralytics import YOLO

from clip_model import classify_with_clip_pil
from swin_faiss import load_swin_faiss_classifier
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
swin_classifier = load_swin_faiss_classifier()

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

    results = yolo_model(image, conf=0.25)
    t_yolo = time.time()
    print(f"[timing] YOLO inference took {t_yolo - t0:.3f}s")
    boxes = results[0].boxes.xyxy.cpu().numpy()

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    rows = []
    empty_ratio, covered_ratio = estimate_empty_space(boxes, img_w, img_h)
    empty_label = "High" if empty_ratio >= 0.55 else "Moderate" if empty_ratio >= 0.25 else "Low"

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, img_w, img_h, pad=12)

        crop = image.crop((px1, py1, px2, py2))
        t_before_swin = time.time()
        swin_result = None
        final_category = "unknown"
        clip_result = None
        unknown_path = None

        if swin_classifier.is_ready():
            try:
                swin_result = swin_classifier.classify(crop, top_k=10, top_labels=5)
                t_swin = time.time()
                print(f"[timing] crop {i} Swin FAISS took {t_swin - t_before_swin:.3f}s")
                if swin_result["confidence"] == "high":
                    final_category = swin_result["label"]
                else:
                    candidates = swin_result.get("candidate_labels", [])
                    if candidates:
                        try:
                            t_before_clip = time.time()
                            clip_result = classify_with_clip_pil(crop, candidates)
                            t_clip = time.time()
                            print(f"[timing] crop {i} CLIP took {t_clip - t_before_clip:.3f}s")
                            clip_label = (clip_result or {}).get("label") if isinstance(clip_result, dict) else None
                            clip_score = float((clip_result or {}).get("score") or 0.0)
                            if clip_label and clip_label != "unknown" and clip_score >= 0.18:
                                final_category = clip_label
                            else:
                                final_category = "unknown"
                        except Exception as clip_exc:
                            print(f"[warning] crop {i} CLIP failed: {clip_exc}")
                            final_category = "unknown"
                    else:
                        final_category = "unknown"
            except Exception as swin_exc:
                print(f"[warning] crop {i} Swin FAISS failed: {swin_exc}")
                final_category = "unknown"
        else:
            print("[warning] Swin FAISS classifier not ready; marking crop as unknown")
            final_category = "unknown"

        if final_category == "unknown":
            unknown_path = swin_classifier.save_unknown_crop(crop, i + 1)

        llava_sub_result = {"answer": None}
        subcategory_label = "unknown"
        if final_category != "unknown":
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
                "swin": swin_result,
                "clip": clip_result,
                "llava_subcategory": llava_sub_result,
            }
        )

    # Build synthesized summary in human-first language
    unique_cats = {}
    unknown_items = []
    for r in rows:
        cat = r.get("product_category") or "unknown"
        unique_cats[cat] = unique_cats.get(cat, 0) + 1
        if cat.lower() == "unknown":
            unknown_items.append(r["crop_id"])

    summary_lines = []
    total_items = len(rows)
    distinct_categories = len([cat for cat in unique_cats.keys() if cat.lower() != "unknown"])
    shelf_type = "Unknown"
    if distinct_categories == 1:
        shelf_type = "Category-specific"
    elif distinct_categories > 1:
        shelf_type = "Mixed"

    summary_lines.append("<h3>Store Management Summary</h3>")
    summary_lines.append(
        f"<p>We detected <strong>{total_items}</strong> product{'s' if total_items != 1 else ''} on this shelf. "
        f"The shelf appears <strong>{shelf_type.lower()}</strong> with {distinct_categories} distinct category{'ies' if distinct_categories != 1 else ''}.</p>"
    )

    if unique_cats:
        sorted_cats = sorted(unique_cats.items(), key=lambda x: -x[1])
        category_lines = []
        for cat, cnt in sorted_cats[:5]:
            category_lines.append(f"{cnt} x {cat}")
        summary_lines.append(
            f"<p><strong>Top categories:</strong> {', '.join(category_lines)}</p>"
        )
    else:
        summary_lines.append("<p><strong>Top categories:</strong> none detected yet.</p>")

    if unknown_items:
        summary_lines.append(
            f"<p><strong>Manual review needed:</strong> {len(unknown_items)} item{'s' if len(unknown_items) != 1 else ''} may be unclear and should be checked manually.</p>"
        )
    else:
        summary_lines.append("<p><strong>Manual review needed:</strong> no unclear items were found.</p>")

    summary_lines.append(
        f"<p><strong>Estimated visible empty space:</strong> {empty_ratio*100:.0f}% ({empty_label}).</p>"
    )

    summary_lines.append("<h4>Detected items</h4>")
    if rows:
        summary_lines.append("<ol>")
        for r in rows:
            cid = r["crop_id"]
            cat = r.get("product_category") or "unknown"
            sub = r.get("subcategory") or "unknown"
            if cat.lower() == "unknown":
                summary_lines.append(
                    f"<li>Item {cid} could not be confidently identified and should be reviewed.</li>"
                )
            else:
                summary_lines.append(
                    f"<li>Item {cid} is likely <strong>{cat}</strong> with subcategory <strong>{sub}</strong>.</li>"
                )
        summary_lines.append("</ol>")
    else:
        summary_lines.append("<p>No product items were found in the image.</p>")

    summary_html = "\n".join(summary_lines)

    question_answers = {
        "How many products were detected on this shelf?": (
            f"We detected {total_items} product{'s' if total_items != 1 else ''} on the shelf."
        ),
        "Which are the top detected categories by count?": (
            f"The top categories are {', '.join(category_lines)}." if unique_cats else "No clear product categories were identified."
        ),
        "Which items need manual review?": (
            f"{len(unknown_items)} item{'s' if len(unknown_items) != 1 else ''} may need manual review."
            if unknown_items
            else "No items require manual review at this time."
        ),
        "Is this shelf mixed or category-specific?": (
            f"This shelf appears to be {shelf_type.lower()}."
        ),
        "How much empty space is visible on this shelf?": (
            f"About {empty_ratio*100:.0f}% of the shelf appears empty ({empty_label})."
        ),
    }

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