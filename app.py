import os

import gradio as gr
from PIL import Image, ImageDraw
from ultralytics import YOLO

from efficientnet_model import classify_pil_image, load_model_if_needed
from llava4_model import generate_llava4_answer


yolo_model = YOLO("models/yolo/best.pt")


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


def process_image(input_image, routing_mode, force_llava, disable_llava, llava_prompt):
    image = input_image.convert("RGB")
    img_w, img_h = image.size

    # Preload EfficientNet once.
    load_model_if_needed()

    results = yolo_model(image, conf=0.25)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    rows = []
    fine_grained = routing_mode == "fine_grained"

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, img_w, img_h, pad=12)

        crop = image.crop((px1, py1, px2, py2))
        efficientnet_pred = classify_pil_image(crop)

        routing = decide_backend_action(
            efficientnet_pred,
            user_requested_llava=bool(force_llava),
            fine_grained=fine_grained,
        )

        llava4_result = None
        final_category = efficientnet_pred.get("label") or "unknown"

        if routing["should_call_llava4"] and not disable_llava:
            try:
                llava4_result = generate_llava4_answer(
                    image=crop,
                    broad_category=efficientnet_pred.get("label"),
                    user_prompt=(llava_prompt or None),
                )
                llava_answer = (llava4_result or {}).get("answer", "").strip()
                if llava_answer:
                    final_category = llava_answer
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
    ],
    outputs=[
        gr.Image(type="pil", label="Annotated image"),
        gr.JSON(label="Per-crop routing and predictions"),
    ],
    title="Retail Product Detection + EfficientNet -> LLaVA4 Routing",
)

demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))