import os
from typing import Optional, Dict

import numpy as np
from PIL import Image

import tensorflow as tf


MODEL_PATH = os.path.join("models", "efficientnet", "efficientnet_best.keras")
LABELS_PATH = os.path.join("models", "efficientnet", "labels.txt")
DEFAULT_LABELS = [
    "Apparel & Accessories",
    "Automotive & Hardware",
    "Baby Care",
    "Beverages",
    "Electronics",
    "Grocery & Pantry",
    "Hair Care",
    "Health & Wellness",
    "Home & Kitchen",
    "Household Cleaning & Laundry",
    "Oral Care",
    "Other / Unclear",
    "Paper & Hygiene",
    "Personal Care & Beauty",
    "Pet Care",
    "Snacks & Confectionery",
    "Stationery & Office",
    "Tobacco & Alcohol",
    "Tobacco / Restricted",
    "Toys & Stationery",
]


def _load_labels(path: str) -> Optional[list]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        labels = [l.strip() for l in f.readlines() if l.strip()]
    return labels


def _safe_preprocess(x: np.ndarray) -> np.ndarray:
    try:
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        return preprocess_input(x)
    except Exception:
        # fallback: scale to [0,1]
        return x.astype("float32") / 255.0


def _get_target_size(model: tf.keras.Model) -> tuple:
    # model.input_shape is typically (None, height, width, channels)
    shape = getattr(model, "input_shape", None)
    if not shape:
        return 224, 224
    if len(shape) >= 3 and shape[-3] and shape[-2]:
        # handle (None, H, W, C) or (B, H, W, C)
        return int(shape[-3]), int(shape[-2])
    # fallback
    return 224, 224


# Load model and labels at import so server routes can use them quickly.
_MODEL = None
_LABELS = None

def load_model_if_needed():
    global _MODEL, _LABELS
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"EfficientNet model not found at {MODEL_PATH}")
        _MODEL = tf.keras.models.load_model(MODEL_PATH)
        _LABELS = _load_labels(LABELS_PATH) or DEFAULT_LABELS
    return _MODEL


def classify_pil_image(pil_img: Image.Image) -> Dict:
    """Classify a PIL image with the EfficientNet model.

    Returns a dict with top-1/top-2 and confidence gap.
    """
    model = load_model_if_needed()

    img = pil_img.convert("RGB")
    target_h, target_w = _get_target_size(model)
    img_resized = img.resize((target_w, target_h))

    arr = np.array(img_resized).astype("float32")
    arr = _safe_preprocess(arr)
    batch = np.expand_dims(arr, axis=0)

    preds = model.predict(batch)
    # ensure 1D vector for classes
    if preds.ndim > 1 and preds.shape[0] == 1:
        preds = preds[0]

    try:
        probs = tf.nn.softmax(preds).numpy()
    except Exception:
        # if already probabilities
        probs = np.array(preds)

    sorted_idx = np.argsort(probs)[::-1]
    idx = int(sorted_idx[0])
    conf = float(probs[idx])
    second_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else idx
    second_conf = float(probs[second_idx]) if len(sorted_idx) > 1 else 0.0
    gap = conf - second_conf

    global _LABELS
    if _LABELS is None:
        _LABELS = _load_labels(LABELS_PATH) or DEFAULT_LABELS

    label = _LABELS[idx] if _LABELS and idx < len(_LABELS) else None
    second_label = _LABELS[second_idx] if _LABELS and second_idx < len(_LABELS) else None

    return {
        "index": idx,
        "label": label,
        "confidence": conf,
        "second_index": second_idx,
        "second_label": second_label,
        "second_confidence": second_conf,
        "confidence_gap": float(gap),
    }


if __name__ == "__main__":
    print("This module provides `classify_pil_image(pil_img)` for server usage.")
