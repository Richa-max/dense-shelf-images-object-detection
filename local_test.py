import argparse
import json

from PIL import Image, ImageDraw
from ultralytics import YOLO

from efficientnet_model import classify_pil_image


def pad_box(x1, y1, x2, y2, img_w, img_h, pad=12):
    x1 = max(0, int(x1 - pad))
    y1 = max(0, int(y1 - pad))
    x2 = min(img_w, int(x2 + pad))
    y2 = min(img_h, int(y2 + pad))
    return x1, y1, x2, y2


def run_test(image_path: str, yolo_model_path: str, conf: float):
    yolo_model = YOLO(yolo_model_path)
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    results = yolo_model(image, conf=conf)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    rows = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, img_w, img_h)
        crop = image.crop((px1, py1, px2, py2))

        pred = classify_pil_image(crop)
        label = pred.get("label") or f'class_{pred["index"]}'

        draw.rectangle((px1, py1, px2, py2), outline="red", width=3)
        draw.text((px1, max(0, py1 - 10)), label, fill="red")

        rows.append(
            {
                "crop_id": i + 1,
                "box": [px1, py1, px2, py2],
                "prediction": pred,
            }
        )

    print(json.dumps(rows, indent=2))
    return annotated, rows


def main():
    parser = argparse.ArgumentParser(description="Run YOLO crop + EfficientNet classification locally.")
    parser.add_argument("--image", required=True, help="Path to a shelf image")
    parser.add_argument("--yolo-model", default="models/yolo/best.pt", help="Path to YOLO .pt model")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--save-annotated", default="annotated.jpg", help="Where to save annotated output")
    args = parser.parse_args()

    annotated, _ = run_test(args.image, args.yolo_model, args.conf)
    annotated.save(args.save_annotated)
    print(f"Saved annotated image to {args.save_annotated}")


if __name__ == "__main__":
    main()
