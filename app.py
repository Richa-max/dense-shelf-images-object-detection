import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

import gradio as gr
from PIL import Image, ImageDraw
import io
import base64
import numpy as np
from ultralytics import YOLO

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
from retail_product_resolver import resolve_retail_product, summarize_retail_decisions


yolo_model = YOLO("models/yolo/best.pt")
swin_classifier = load_swin_faiss_classifier()


def _clean_label(value):
    return str(value or "").strip()


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


def decide_backend_action(pred: dict) -> dict:
    top1 = float(pred.get("confidence", 0.0))
    gap = float(pred.get("confidence_gap", 0.0))

    if top1 >= 0.80 and gap >= 0.20:
        return {
            "action": "accept_swin_faiss",
            "reason": "high_confidence_and_clear_margin",
        }

    return {
        "action": "defer_to_user",
        "reason": "use_swin_faiss",
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
        clip_result = None
        unknown_path = None

        if swin_classifier.is_ready():
            try:
                swin_result = swin_classifier.classify(crop, top_k=10, top_labels=5)
                t_swin = time.time()
                print(f"[timing] crop {i} Swin FAISS took {t_swin - t_before_swin:.3f}s")
                if i == 0:
                    print(f"[debug] crop {i} swin_result={swin_result}")

                swin_cat = swin_result.get("predicted_category") or swin_result.get("label")
                swin_subcat = swin_result.get("predicted_subcategory") or swin_result.get("best_subcategory")

                if _is_valid_label(swin_cat):
                    final_category = swin_cat
                else:
                    final_category = "unknown"

                if _is_valid_label(swin_subcat):
                    subcategory_label = swin_subcat
                else:
                    subcategory_label = "unknown"
            except Exception as swin_exc:
                print(f"[warning] crop {i} Swin FAISS failed: {swin_exc}")
                final_category = "unknown"
                subcategory_label = "unknown"
        else:
            print("[warning] Swin FAISS classifier not ready; marking crop as unknown")
            final_category = "unknown"
            subcategory_label = "unknown"

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
            unknown_path = swin_classifier.save_unknown_crop(crop, i + 1)

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
                "llava_subcategory": {"answer": None, "status": "disabled"},
                "retail_product": retail_product,
                "crop_image": crop,
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

    retail_summary = summarize_retail_decisions(rows)
    top_products = retail_summary.get("top_products") or []
    if top_products:
        product_lines = [f"{cnt} x {name}" for name, cnt in top_products[:5]]
        summary_lines.append(
            f"<p><strong>Likely products:</strong> {', '.join(product_lines)}</p>"
        )

    if rows:
        summary_lines.append(
            f"<p><strong>Resolver stack:</strong> SWIN + FAISS classification used for {len(rows)} crop{'s' if len(rows) != 1 else ''}.</p>"
        )

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
                retail_decision = (r.get("retail_product") or {}).get("decision") or {}
                product = retail_decision.get("predicted_product")
                confidence = retail_decision.get("confidence")
                product_prefix = ""
                if _is_distinct_product_name(product, cat, sub):
                    product_prefix = f"<strong>{product}</strong>, "
                confidence_suffix = ""
                if isinstance(confidence, (int, float)):
                    confidence_suffix = f" ({confidence*100:.0f}% confidence)"
                summary_lines.append(
                    f"<li>Item {cid} is likely {product_prefix}<strong>{cat}</strong> with subcategory <strong>{sub}</strong>{confidence_suffix}.</li>"
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
    box_choices = [str(r["crop_id"]) for r in rows]
    initial_choice = box_choices[0] if box_choices else None
    return annotated, summary_html, answer_html, gr.update(choices=box_choices, value=initial_choice), rows


def inspect_selected_crop(selected_crop_id, rows_state):
    if not rows_state:
        return "<p>No detected boxes are available yet.</p>"
    try:
        crop_id = int(selected_crop_id)
    except (TypeError, ValueError):
        return "<p>Please select a detected box first.</p>"

    for row in rows_state:
        if int(row.get("crop_id", -1)) != crop_id:
            continue
        crop_image = row.get("crop_image")
        if crop_image is None:
            return f"<p>Crop {crop_id} was detected but no image crop is available.</p>"
        prompt = (
            "Identify the exact SKU, product name, or brand visible in this crop. "
            "Return a short answer only."
        )
        result = generate_llava4_answer(
            image=crop_image,
            broad_category=row.get("product_category") or "unknown",
            user_prompt=prompt,
        )
        answer = (result.get("answer") or "").strip()
        if not answer:
            answer = result.get("error") or "No SKU was detected."
        return f"<p><b>Selected box {crop_id}</b><br>{answer}</p>"

    return f"<p>No detected box with id {crop_id} was found.</p>"


with gr.Blocks(title="Smart Shelf Management Dashboard") as demo:
    gr.Markdown("Upload a shelf image to detect product categories and inspect a specific boxed item with LLaVA.")
    with gr.Row():
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
    analyze_button = gr.Button("Analyze shelf")
    with gr.Row():
        output_image = gr.Image(type="pil", label="Annotated image")
        output_summary = gr.HTML(label="Summary")
        output_answer = gr.HTML(label="Selected question answer")
    with gr.Row():
        crop_selector = gr.Dropdown(choices=[], label="Select a detected box to inspect")
        inspect_button = gr.Button("Inspect selected box with LLaVA")
    output_sku = gr.HTML(label="SKU / product detail")
    rows_state = gr.State(value=[])

    analyze_button.click(
        process_image,
        inputs=[image_input, question_input],
        outputs=[output_image, output_summary, output_answer, crop_selector, rows_state],
    )
    inspect_button.click(
        inspect_selected_crop,
        inputs=[crop_selector, rows_state],
        outputs=[output_sku],
    )

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        inbrowser=False,
    )
