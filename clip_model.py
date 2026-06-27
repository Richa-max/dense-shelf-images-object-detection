import threading
from typing import List

from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

_MODEL_NAME = "openai/clip-vit-base-patch32"
_model = None
_processor = None
_lock = threading.Lock()


def load_clip_if_needed():
    global _model, _processor
    with _lock:
        if _model is None or _processor is None:
            _model = CLIPModel.from_pretrained(_MODEL_NAME)
            _processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
            _model.eval()
    return _model, _processor


def classify_with_clip_pil(image: Image.Image, candidates: List[str], threshold: float = 0.18) -> dict:
    """Return best matching label from `candidates` or 'unknown' if below `threshold`.

    Returns dict: {"label": str, "score": float, "all": {candidate: score}}
    Scores are softmax probabilities over the candidates.
    """
    if not candidates:
        raise ValueError("no candidates provided for CLIP classification")

    model, processor = load_clip_if_needed()

    inputs = processor(text=candidates, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds  # (1, dim)
        text_embeds = outputs.text_embeds  # (len(candidates), dim)

        # normalize
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarities (scaled by 100 as CLIP does)
        logits_per_image = (100.0 * image_embeds @ text_embeds.T).squeeze(0)
        probs = torch.nn.functional.softmax(logits_per_image, dim=-1).cpu().numpy()

    best_idx = int(probs.argmax())
    best_score = float(probs[best_idx])

    all_scores = {c: float(p) for c, p in zip(candidates, probs.tolist())}

    if best_score < threshold:
        return {"label": "unknown", "score": best_score, "all": all_scores}
    return {"label": candidates[best_idx], "score": best_score, "all": all_scores}
