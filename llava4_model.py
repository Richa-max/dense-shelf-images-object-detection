import os
import io
import base64
from typing import Dict, Optional

from PIL import Image
import requests


LLAVA_PROVIDER = os.getenv("LLAVA_PROVIDER", "hf").strip().lower()
LLAVA4_MODEL_ID = os.getenv("LLAVA4_MODEL_ID", "llava-hf/llava-1.5-7b-hf")
LLAVA4_MAX_NEW_TOKENS = int(os.getenv("LLAVA4_MAX_NEW_TOKENS", "64"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:13b")

_PROCESSOR = None
_MODEL = None


def _load_llava4_if_needed():
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    global _PROCESSOR, _MODEL
    if _MODEL is not None and _PROCESSOR is not None:
        return _PROCESSOR, _MODEL

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    _PROCESSOR = AutoProcessor.from_pretrained(LLAVA4_MODEL_ID)

    if use_cuda:
        _MODEL = LlavaForConditionalGeneration.from_pretrained(
            LLAVA4_MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        _MODEL = LlavaForConditionalGeneration.from_pretrained(
            LLAVA4_MODEL_ID,
            torch_dtype=dtype,
        )

    return _PROCESSOR, _MODEL


def _build_prompt(broad_category: Optional[str], user_prompt: Optional[str]) -> str:
    if user_prompt and user_prompt.strip():
        task = user_prompt.strip()
    elif broad_category:
        task = (
            f"The product likely belongs to '{broad_category}'. "
            "Identify the exact retail product name or most specific product type visible in the crop."
        )
    else:
        task = "Identify the exact retail product name or the most specific product type visible in the crop."

    return f"USER: <image>\n{task}\nReturn a short answer only.\nASSISTANT:"


def generate_llava4_answer(
    image: Image.Image,
    broad_category: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> Dict:
    prompt = _build_prompt(broad_category=broad_category, user_prompt=user_prompt)

    if LLAVA_PROVIDER == "ollama":
        image_rgb = image.convert("RGB")
        image_bytes = io.BytesIO()
        image_rgb.save(image_bytes, format="JPEG")
        image_b64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "num_predict": LLAVA4_MAX_NEW_TOKENS,
            },
        }

        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()

        return {
            "provider": "ollama",
            "model_id": OLLAMA_MODEL,
            "prompt": prompt,
            "answer": data.get("response", "").strip(),
        }

    processor, model = _load_llava4_if_needed()

    inputs = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")

    target_device = model.device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=LLAVA4_MAX_NEW_TOKENS,
        do_sample=False,
    )

    decoded = processor.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("ASSISTANT:")[-1].strip()

    return {
        "provider": "hf",
        "model_id": LLAVA4_MODEL_ID,
        "prompt": prompt,
        "answer": answer,
    }
