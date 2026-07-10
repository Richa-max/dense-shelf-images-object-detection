import base64
import io
import os
from typing import Dict, Optional

from PIL import Image

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


LLAVA_PROVIDER = os.getenv("LLAVA_PROVIDER", "ollama").strip().lower()
LLAVA4_MODEL_ID = os.getenv("LLAVA4_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
LLAVA4_MAX_NEW_TOKENS = int(os.getenv("LLAVA4_MAX_NEW_TOKENS", "96"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")


def _build_prompt(broad_category: Optional[str], user_prompt: Optional[str]) -> str:
    if user_prompt and user_prompt.strip():
        task = user_prompt.strip()
    elif broad_category:
        task = (
            f"The product likely belongs to '{broad_category}'. "
            "Identify the exact SKU, product name, brand, category, and subcategory visible in the crop. "
            "Return a concise answer only."
        )
    else:
        task = "Identify the exact SKU, product name, brand, category, and subcategory visible in the crop. Return a concise answer only."

    return f"USER: <image>\n{task}\nASSISTANT:"


def generate_llava4_answer(
    image: Image.Image,
    broad_category: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> Dict:
    prompt = _build_prompt(broad_category=broad_category, user_prompt=user_prompt)

    if LLAVA_PROVIDER == "ollama":
        if requests is None:
            return {
                "provider": "ollama",
                "model_id": OLLAMA_MODEL,
                "prompt": prompt,
                "answer": "",
                "error": "requests_not_available",
            }
        try:
            image_rgb = image.convert("RGB")
            image_bytes = io.BytesIO()
            image_rgb.save(image_bytes, format="JPEG")
            image_b64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"num_predict": LLAVA4_MAX_NEW_TOKENS},
            }
            response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            return {
                "provider": "ollama",
                "model_id": OLLAMA_MODEL,
                "prompt": prompt,
                "answer": (data.get("response") or "").strip(),
            }
        except Exception as exc:
            return {
                "provider": "ollama",
                "model_id": OLLAMA_MODEL,
                "prompt": prompt,
                "answer": "",
                "error": str(exc),
            }

    if LLAVA_PROVIDER in {"hf", "huggingface", "transformers"}:
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration

            processor = AutoProcessor.from_pretrained(LLAVA4_MODEL_ID)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = LlavaForConditionalGeneration.from_pretrained(
                LLAVA4_MODEL_ID,
                torch_dtype=dtype,
            )
            inputs = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")
            target_device = getattr(model, "device", None)
            if target_device is not None:
                inputs = {key: value.to(target_device) for key, value in inputs.items()}
            output = model.generate(**inputs, max_new_tokens=LLAVA4_MAX_NEW_TOKENS, do_sample=False)
            decoded = processor.decode(output[0], skip_special_tokens=True)
            answer = decoded.split("ASSISTANT:")[-1].strip()
            return {
                "provider": "hf",
                "model_id": LLAVA4_MODEL_ID,
                "prompt": prompt,
                "answer": answer,
            }
        except Exception as exc:
            return {
                "provider": "hf",
                "model_id": LLAVA4_MODEL_ID,
                "prompt": prompt,
                "answer": "",
                "error": str(exc),
            }

    return {
        "provider": "disabled",
        "model_id": None,
        "prompt": prompt,
        "answer": "",
        "error": "llava_provider_not_available",
    }
