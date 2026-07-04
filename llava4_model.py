import os
from typing import Dict, Optional

from PIL import Image


LLAVA_PROVIDER = os.getenv("LLAVA_PROVIDER", "hf").strip().lower()
LLAVA4_MODEL_ID = os.getenv("LLAVA4_MODEL_ID", "llava-hf/llava-1.5-7b-hf")
LLAVA4_MAX_NEW_TOKENS = int(os.getenv("LLAVA4_MAX_NEW_TOKENS", "64"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:13b")


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
    return {
        "provider": "disabled",
        "model_id": None,
        "prompt": prompt,
        "answer": "",
    }
