import os

import gradio as gr
from PIL import Image, ImageDraw
from ultralytics import YOLO

from efficientnet_model import classify_pil_image, load_model_if_needed
from clip_model import classify_with_clip_pil, load_clip_if_needed
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

# Preload heavier models at startup to avoid first-request latency.
try:
    load_model_if_needed()
    if os.getenv("PRELOAD_CLIP", "0").strip().lower() in {"1", "true", "yes"}:
        load_clip_if_needed()
except Exception:
    # model loading may happen on-demand; don't crash the app if preload fails
    pass


def pad_box(x1, y1, x2, y2, img_w, img_h, pad=10):
    x1 = max(0, int(x1 - pad))
    y1 = max(0, int(y1 - pad))
    x2 = min(img_w, int(x2 + pad))
    y2 = min(img_h, int(y2 + pad))
    return x1, y1, x2, y2


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


def process_image(input_image, routing_mode, force_llava, disable_llava, llava_prompt, use_clip=False, subcategories_text=None):
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
    fine_grained = routing_mode == "fine_grained"
    use_clip = bool(use_clip)
    subcategories = None
    if subcategories_text:
        raw = subcategories_text.strip()
        try:
            import json

            parsed = json.loads(raw)
            if isinstance(parsed, list):
                subcategories = [str(x) for x in parsed]
        except Exception:
            if "\n" in raw:
                subcategories = [s.strip() for s in raw.splitlines() if s.strip()]
            else:
                subcategories = [s.strip() for s in raw.split(",") if s.strip()]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, img_w, img_h, pad=12)

        crop = image.crop((px1, py1, px2, py2))
        t_before_eff = time.time()
        efficientnet_pred = classify_pil_image(crop)

        routing = decide_backend_action(
            efficientnet_pred,
            user_requested_llava=bool(force_llava),
            fine_grained=fine_grained,
        )

        llava4_result = None
        clip_result = None
        final_category = efficientnet_pred.get("label") or "unknown"

        # If EfficientNet couldn't produce a clear broad category, force CLIP resolution.
        eff_label = (efficientnet_pred.get("label") or "").strip()
        eff_unknown = (not eff_label) or eff_label.lower() in {"unknown", "other / unclear", "other", "unclear"}

        # Prepare CLIP candidates: prefer explicit subcategories, then mapping for the predicted broad category,
        # otherwise fall back to flattened mapping of all subcategories.
        clip_candidates = None
        if subcategories:
            clip_candidates = subcategories
        else:
            broad = efficientnet_pred.get("label")
            if broad and broad in _SUBCATS:
                clip_candidates = _SUBCATS.get(broad)
            else:
                # flatten default mapping as a fallback (may be large)
                all_cands = []
                for v in _SUBCATS.values():
                    all_cands.extend(v)
                clip_candidates = all_cands if all_cands else None

        # Only run CLIP when requested by UI or when EfficientNet is unknown.
        run_clip = use_clip or eff_unknown
        if run_clip and clip_candidates:
            try:
                t_before_clip = time.time()
                clip_result = classify_with_clip_pil(crop, clip_candidates)
                t_clip = time.time()
                print(f"[timing] crop {i} CLIP took {t_clip - t_before_clip:.3f}s")
            except Exception as clip_exc:
                clip_result = {"error": "clip_failed", "details": str(clip_exc)}

        # If CLIP provided a confident label, accept it. Otherwise, fall back to LLaVA4 when routing allows.
        clip_label = (clip_result or {}).get("label") if isinstance(clip_result, dict) else None
        clip_score = float((clip_result or {}).get("score") or 0.0) if isinstance(clip_result, dict) else 0.0
        clip_low_confidence = clip_label in (None, "unknown") or clip_score < 0.18

        if clip_label and clip_label != "unknown":
            final_category = clip_label
            llava4_result = {"status": "skipped", "reason": "clip_resolved", "clip": clip_result}
        else:
            # If EfficientNet unknown and CLIP low-confidence, prefer calling LLaVA4.
            should_force_llava = eff_unknown and (clip_result is None or clip_low_confidence)
            if (routing["should_call_llava4"] or should_force_llava) and not disable_llava:
                try:
                    t_before_llava = time.time()
                    llava4_result = generate_llava4_answer(
                        image=crop,
                        broad_category=efficientnet_pred.get("label"),
                        user_prompt=(llava_prompt or None),
                    )
                    t_llava = time.time()
                    print(f"[timing] crop {i} LLaVA4 took {t_llava - t_before_llava:.3f}s")
                    llava_answer = (llava4_result or {}).get("answer", "").strip()
                    if llava_answer:
                        final_category = llava_answer
                except Exception as llava_exc:
                    llava4_result = {
                        "error": "llava4_failed",
                        "details": str(llava_exc),
                    }
            elif (routing["should_call_llava4"] or should_force_llava) and disable_llava:
                llava4_result = {
                    "status": "skipped",
                    "reason": "disable_llava=true",
                }
                    "reason": "disable_llava=true",
                }

        draw.rectangle((px1, py1, px2, py2), outline="red", width=3)
        draw.text((px1, max(0, py1 - 12)), str(final_category), fill="red")

        rows.append(
            {
                "crop_id": i + 1,
                "box": [px1, py1, px2, py2],
                "final_category": final_category,
                "efficientnet": efficientnet_pred,
                "routing": routing,
                "llava4": llava4_result,
            }
        )

    return annotated, rows


demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload shelf image"),
        gr.Radio(
            choices=["coarse", "fine_grained"],
            value="coarse",
            label="Routing mode",
            info="coarse = confidence-based routing, fine_grained = always call LLaVA4",
        ),
        gr.Checkbox(value=False, label="Force LLaVA4"),
        gr.Checkbox(value=False, label="Disable LLaVA4"),
        gr.Textbox(
            lines=4,
            label="Optional LLaVA prompt override",
            placeholder="Leave empty to use default prompt",
        ),
        gr.Checkbox(value=False, label="Use CLIP for subcategory resolution"),
        gr.Textbox(lines=6, label="Subcategories (one per line or JSON list)", placeholder="e.g.\nBeverages\nSnacks\nDairy"),
    ],
    outputs=[
        gr.Image(type="pil", label="Annotated image"),
        gr.JSON(label="Per-crop routing and predictions"),
    ],
    title="Retail Product Detection + EfficientNet -> LLaVA4 Routing",
)

demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))