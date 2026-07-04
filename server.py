import io
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO

from clip_model import classify_with_clip_pil, load_clip_if_needed
from swin_faiss import load_swin_faiss_classifier
import time
from llava4_model import generate_llava4_answer
import json
import os
from retail_product_resolver import resolve_retail_product

# load default subcategories mapping if present
_SUBCATS = {}
_SC_PATH = os.path.join(os.path.dirname(__file__), "subcategories.json")
if os.path.exists(_SC_PATH):
    try:
        with open(_SC_PATH, "r", encoding="utf-8") as fh:
            _SUBCATS = json.load(fh)
    except Exception:
        _SUBCATS = {}


app = Flask(__name__)

yolo_model = YOLO("models/yolo/best.pt")
swin_classifier = load_swin_faiss_classifier()

# Preload CLIP at startup if requested via env var to avoid first-request stalls.
try:
    if os.getenv("PRELOAD_CLIP", "0").strip().lower() in {"1", "true", "yes"}:
        load_clip_if_needed()
except Exception:
    pass


def _to_bool(value) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def pad_box(x1, y1, x2, y2, img_w, img_h, pad=12):
    x1 = max(0, int(x1 - pad))
    y1 = max(0, int(y1 - pad))
    x2 = min(img_w, int(x2 + pad))
    y2 = min(img_h, int(y2 + pad))
    return x1, y1, x2, y2


def _classify_crop_image(
    img: Image.Image,
    disable_llava: bool = False,
    llava_prompt: str = None,
) -> dict:
    if not swin_classifier.is_ready():
        raise RuntimeError("Swin FAISS classifier is not ready")

    swin_result = swin_classifier.classify(img, top_k=10, top_labels=5)
    result_label = "unknown"
    clip_result = None
    if swin_result.get("confidence") == "high":
        result_label = swin_result.get("label", "unknown")
    else:
        candidate_labels = swin_result.get("candidate_labels", [])
        if candidate_labels:
            try:
                clip_result = classify_with_clip_pil(img, candidate_labels)
                clip_label = (clip_result or {}).get("label") if isinstance(clip_result, dict) else None
                clip_score = float((clip_result or {}).get("score") or 0.0)
                if clip_label and clip_label != "unknown" and clip_score >= 0.18:
                    result_label = clip_label
            except Exception as clip_exc:
                clip_result = {"error": "clip_failed", "details": str(clip_exc)}
        else:
            clip_result = {"label": "unknown", "score": 0.0, "all": {}}

    llava4_result = None
    subcategory_label = None
    if not disable_llava and result_label != "unknown":
        try:
            llava4_result = generate_llava4_answer(
                image=img,
                broad_category=result_label,
                user_prompt=llava_prompt,
            )
            if isinstance(llava4_result, dict):
                subcategory_label = llava4_result.get("answer")
        except Exception as llava_exc:
            llava4_result = {
                "error": "llava4_failed",
                "details": str(llava_exc),
            }
    elif disable_llava:
        llava4_result = {"status": "skipped", "reason": "disable_llava=true"}

    retail_product = resolve_retail_product(
        image=img,
        swin_result=swin_result,
        category_hint=result_label,
        subcategory_hint=subcategory_label,
    )
    ocr_info = retail_product.get("ocr") or {}
    reasoner_info = retail_product.get("reasoner") or {}
    print(
        "[debug] classify crop "
        f"EasyOCR available={ocr_info.get('available')} text={ocr_info.get('text')!r} "
        f"reasoner={reasoner_info.get('provider') or reasoner_info.get('status') or reasoner_info.get('error')}"
    )

    decision = retail_product.get("decision") or {}
    if decision.get("category") and decision.get("category") != "unknown":
        result_label = decision["category"]

    return {
        "category": result_label,
        "swin": swin_result,
        "clip": clip_result,
        "llava4": llava4_result,
        "retail_product": retail_product,
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/classify_crop", methods=["POST"])
def classify_crop():
    # Expect a multipart form with key 'image'
    if "image" not in request.files:
        return jsonify({"error": "no image file provided (form key 'image')"}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": "cannot open image", "details": str(e)}), 400

    user_requested_llava = _to_bool(request.form.get("user_requested_llava"))
    fine_grained = _to_bool(request.form.get("fine_grained"))
    disable_llava = _to_bool(request.form.get("disable_llava"))
    llava_prompt = request.form.get("llava_prompt")
    # CLIP subcategory options: either provide a JSON list in 'subcategories'
    # or send newline-separated values. Also accepts boolean 'use_clip'.
    raw_subcats = request.form.get("subcategories")
    use_clip = _to_bool(request.form.get("use_clip")) or bool(raw_subcats)
    subcategories = None
    if raw_subcats:
        raw = raw_subcats.strip()
        try:
            import json

            parsed = json.loads(raw)
            if isinstance(parsed, list):
                subcategories = [str(x) for x in parsed]
        except Exception:
            # fallback: newline or comma separated
            if "\n" in raw:
                subcategories = [s.strip() for s in raw.splitlines() if s.strip()]
            else:
                subcategories = [s.strip() for s in raw.split(",") if s.strip()]

        # if user didn't provide explicit subcategories but we have a default mapping,
        # and a broad category was predicted by EfficientNet, use those candidates.

    if not swin_classifier.is_ready():
        return jsonify({"error": "swin_classifier_not_ready"}), 503

    try:
        t0 = time.time()
        swin_result = swin_classifier.classify(img, top_k=10, top_labels=5)
        t_swin = time.time()
        print(f"[timing] SWIN inference took {t_swin - t0:.3f}s")

        result_label = "unknown"
        clip_result = None
        if swin_result.get("confidence") == "high":
            result_label = swin_result.get("label", "unknown")
        else:
            candidate_labels = swin_result.get("candidate_labels", [])
            if candidate_labels:
                try:
                    t_before_clip = time.time()
                    clip_result = classify_with_clip_pil(img, candidate_labels)
                    t_clip = time.time()
                    print(f"[timing] CLIP inference took {t_clip - t_before_clip:.3f}s")
                    clip_label = (clip_result or {}).get("label") if isinstance(clip_result, dict) else None
                    clip_score = float((clip_result or {}).get("score") or 0.0)
                    if clip_label and clip_label != "unknown" and clip_score >= 0.18:
                        result_label = clip_label
                except Exception as clip_exc:
                    clip_result = {"error": "clip_failed", "details": str(clip_exc)}
            else:
                clip_result = {"label": "unknown", "score": 0.0, "all": {}}

        llava4_result = None
        subcategory_label = None
        if not disable_llava and result_label != "unknown":
            try:
                t_before_llava = time.time()
                llava4_result = generate_llava4_answer(
                    image=img,
                    broad_category=result_label,
                    user_prompt=llava_prompt,
                )
                t_llava = time.time()
                print(f"[timing] LLaVA4 inference took {t_llava - t_before_llava:.3f}s")
                if isinstance(llava4_result, dict):
                    subcategory_label = llava4_result.get("answer")
            except Exception as llava_exc:
                llava4_result = {
                    "error": "llava4_failed",
                    "details": str(llava_exc),
                }
        elif disable_llava:
            llava4_result = {"status": "skipped", "reason": "disable_llava=true"}

        t_before_retail = time.time()
        retail_product = resolve_retail_product(
            image=img,
            swin_result=swin_result,
            category_hint=result_label,
            subcategory_hint=subcategory_label,
        )
        t_retail = time.time()
        print(f"[timing] retail product resolver took {t_retail - t_before_retail:.3f}s")
        ocr_info = retail_product.get("ocr") or {}
        reasoner_info = retail_product.get("reasoner") or {}
        print(
            "[debug] classify_crop "
            f"EasyOCR available={ocr_info.get('available')} text={ocr_info.get('text')!r} "
            f"reasoner={reasoner_info.get('provider') or reasoner_info.get('status') or reasoner_info.get('error')}"
        )

        decision = retail_product.get("decision") or {}
        if decision.get("category") and decision.get("category") != "unknown":
            result_label = decision["category"]
    except Exception as e:
        return jsonify({"error": "inference failed", "details": str(e)}), 500

    return jsonify(
        {
            "category": result_label,
            "swin": swin_result,
            "clip": clip_result,
            "llava4": llava4_result,
            "retail_product": retail_product,
        }
    )


@app.route("/classify_shelf", methods=["POST"])
def classify_shelf():
    if "image" not in request.files:
        return jsonify({"error": "no image file provided (form key 'image')"}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": "cannot open image", "details": str(e)}), 400

    disable_llava = _to_bool(request.form.get("disable_llava"))
    llava_prompt = request.form.get("llava_prompt")

    if not swin_classifier.is_ready():
        return jsonify({"error": "swin_classifier_not_ready"}), 503

    try:
        yolo_results = yolo_model(img, conf=0.25)
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        items = []
        for idx, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, img.width, img.height)
            crop = img.crop((px1, py1, px2, py2))
            classification = _classify_crop_image(
                crop,
                disable_llava=disable_llava,
                llava_prompt=llava_prompt,
            )
            items.append(
                {
                    "crop_id": idx,
                    "box": [px1, py1, px2, py2],
                    "category": classification["category"],
                    "swin": classification["swin"],
                    "clip": classification["clip"],
                    "llava4": classification["llava4"],
                    "retail_product": classification["retail_product"],
                }
            )
    except Exception as e:
        return jsonify({"error": "inference_failed", "details": str(e)}), 500

    return jsonify({"items": items, "count": len(items)})


if __name__ == "__main__":
    # Default dev server
    app.run(host="0.0.0.0", port=5000)
