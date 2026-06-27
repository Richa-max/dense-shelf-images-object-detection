import io
from flask import Flask, request, jsonify
from PIL import Image

from efficientnet_model import classify_pil_image, load_model_if_needed
from clip_model import classify_with_clip_pil
from llava4_model import generate_llava4_answer
import json
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


app = Flask(__name__)


def _to_bool(value) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def decide_backend_action(pred: dict, user_requested_llava: bool = False, fine_grained: bool = False) -> dict:
    """Apply the EfficientNet->LLaVA4 routing policy provided by product rules."""
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

    try:
        # ensure model is loaded first (raises helpful error if not found)
        load_model_if_needed()
        result = classify_pil_image(img)
        routing = decide_backend_action(
            result,
            user_requested_llava=user_requested_llava,
            fine_grained=fine_grained,
        )

        # populate subcategories from default mapping when not provided
        if use_clip and not subcategories:
            broad = result.get("label")
            if broad and broad in _SUBCATS:
                subcategories = _SUBCATS.get(broad)

        # Optionally run CLIP to resolve a fine-grained subcategory from candidates.
        clip_result = None
        if use_clip and subcategories:
            try:
                clip_result = classify_with_clip_pil(img, subcategories)
            except Exception as clip_exc:
                clip_result = {"error": "clip_failed", "details": str(clip_exc)}

        llava4_result = None
        # If CLIP returned a confident subcategory, accept it and skip LLaVA4.
        clip_label = (clip_result or {}).get("label") if isinstance(clip_result, dict) else None
        if clip_label and clip_label != "unknown":
            # CLIP resolved the subcategory; don't call LLaVA4.
            llava4_result = {"status": "skipped", "reason": "clip_resolved", "clip": clip_result}
        else:
            # Either CLIP wasn't used, failed, or returned unknown — follow existing routing for LLaVA4.
            if routing["should_call_llava4"] and not disable_llava:
                try:
                    llava4_result = generate_llava4_answer(
                        image=img,
                        broad_category=result.get("label"),
                        user_prompt=llava_prompt,
                    )
                except Exception as llava_exc:
                    llava4_result = {
                        "error": "llava4_failed",
                        "details": str(llava_exc),
                    }
            elif routing["should_call_llava4"] and disable_llava:
                llava4_result = {
                    "status": "skipped",
                    "reason": "disable_llava=true",
                }
    except Exception as e:
        return jsonify({"error": "inference failed", "details": str(e)}), 500

    return jsonify(
        {
            "efficientnet": result,
            "routing": routing,
            "clip": clip_result,
            "llava4": llava4_result,
        }
    )


if __name__ == "__main__":
    # Default dev server
    app.run(host="0.0.0.0", port=5000)
