import io
import json
import os
import time

from flask import Flask, jsonify, request
from PIL import Image
from ultralytics import YOLO

from retail_product_resolver import resolve_retail_product
from swin_faiss import load_swin_faiss_classifier

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


def _classify_crop_image(img: Image.Image) -> dict:
    if not swin_classifier.is_ready():
        raise RuntimeError("Swin FAISS classifier is not ready")

    swin_result = swin_classifier.classify(img, top_k=10, top_labels=5)
    result_label = (
        swin_result.get("predicted_category")
        or swin_result.get("label")
        or "unknown"
    )
    subcategory_label = (
        swin_result.get("predicted_subcategory")
        or swin_result.get("best_subcategory")
        or "unknown"
    )

    retail_product = resolve_retail_product(
        image=img,
        swin_result=swin_result,
        category_hint=result_label,
        subcategory_hint=subcategory_label,
    )

    decision = retail_product.get("decision") or {}
    if decision.get("category") and decision.get("category") != "unknown":
        result_label = decision["category"]
    if decision.get("subcategory") and decision.get("subcategory") != "unknown":
        subcategory_label = decision["subcategory"]

    return {
        "category": result_label,
        "swin": swin_result,
        "clip": None,
        "llava4": None,
        "retail_product": retail_product,
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/classify_crop", methods=["POST"])
def classify_crop():
    if "image" not in request.files:
        return jsonify({"error": "no image file provided (form key 'image')"}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": "cannot open image", "details": str(e)}), 400

    if not swin_classifier.is_ready():
        return jsonify({"error": "swin_classifier_not_ready"}), 503

    try:
        t0 = time.time()
        classification = _classify_crop_image(img)
        t_swin = time.time()
        print(f"[timing] SWIN inference took {t_swin - t0:.3f}s")
    except Exception as e:
        return jsonify({"error": "inference failed", "details": str(e)}), 500

    return jsonify(
        {
            "category": classification["category"],
            "swin": classification["swin"],
            "clip": None,
            "llava4": None,
            "retail_product": classification["retail_product"],
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
            classification = _classify_crop_image(crop)
            items.append(
                {
                    "crop_id": idx,
                    "box": [px1, py1, px2, py2],
                    "category": classification["category"],
                    "swin": classification["swin"],
                    "clip": None,
                    "llava4": None,
                    "retail_product": classification["retail_product"],
                }
            )
    except Exception as e:
        return jsonify({"error": "inference_failed", "details": str(e)}), 500

    return jsonify({"items": items, "count": len(items)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
