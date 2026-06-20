import gradio as gr
import os
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
from transformers import LlavaForConditionalGeneration, AutoProcessor

yolo_model = YOLO("models/yolo/best.pt")

llava_model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(llava_model_id)

llava_model = LlavaForConditionalGeneration.from_pretrained(
    llava_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def pad_box(x1, y1, x2, y2, img_w, img_h, pad=10):
    x1 = max(0, int(x1 - pad))
    y1 = max(0, int(y1 - pad))
    x2 = min(img_w, int(x2 + pad))
    y2 = min(img_h, int(y2 + pad))
    return x1, y1, x2, y2

def classify_crop_with_llava(crop):
    prompt = """
USER: <image>
Identify the retail product category in this crop.
Return only one category name, such as beverage, snack, dairy, personal care, household, packaged food, etc.
ASSISTANT:
"""

    inputs = processor(
        text=prompt,
        images=crop,
        return_tensors="pt"
    ).to(llava_model.device)

    output = llava_model.generate(
        **inputs,
        max_new_tokens=30
    )

    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer.split("ASSISTANT:")[-1].strip()

def process_image(input_image):
    image = input_image.convert("RGB")
    img_w, img_h = image.size

    results = yolo_model(image, conf=0.25)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    rows = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]

        px1, py1, px2, py2 = pad_box(
            x1, y1, x2, y2,
            img_w, img_h,
            pad=12
        )

        crop = image.crop((px1, py1, px2, py2))

        category = classify_crop_with_llava(crop)

        draw.rectangle((px1, py1, px2, py2), outline="red", width=3)
        draw.text((px1, py1 - 10), category, fill="red")

        rows.append({
            "crop_id": i + 1,
            "category": category,
            "box": [px1, py1, px2, py2]
        })

    return annotated, rows

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload shelf image"),
    outputs=[
        gr.Image(type="pil", label="Annotated image"),
        gr.JSON(label="Detected product categories")
    ],
    title="Retail Product Detection + LLaVA Category Prediction"
)

demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))