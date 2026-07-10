import base64
import io
import json
import os
import re
from typing import Dict, Optional

from PIL import Image

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


QWEN_PROVIDER = os.getenv("QWEN_PROVIDER", "ollama").strip().lower()
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "96"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", os.getenv("QWEN_OLLAMA_MODEL", "qwen2.5vl:3b"))


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _estimate_qwen_confidence(answer: str, error: Optional[str] = None) -> Dict:
    if error:
        score = 0.05
        return {
            "score": score,
            "label": _confidence_label(score),
            "reason": "generation_error",
        }

    text = (answer or "").strip()
    if not text:
        score = 0.10
        return {
            "score": score,
            "label": _confidence_label(score),
            "reason": "empty_answer",
        }

    score = 0.55
    lowered = text.lower()

    uncertain_markers = [
        "maybe",
        "might",
        "possibly",
        "not sure",
        "unclear",
        "cannot identify",
        "can't identify",
        "unknown",
    ]
    if any(marker in lowered for marker in uncertain_markers):
        score -= 0.30

    sku_pattern = re.compile(r"\b([A-Z0-9]{3,}[-_][A-Z0-9]{2,}|[A-Z]{2,}\d{2,}|\d{6,})\b")
    if sku_pattern.search(text.upper()):
        score += 0.18

    if any(token in lowered for token in ["sku", "barcode", "brand", "category", "subcategory"]):
        score += 0.08

    if len(text) < 12:
        score -= 0.10
    elif len(text) > 40:
        score += 0.05

    score = max(0.0, min(0.98, score))
    return {
        "score": score,
        "label": _confidence_label(score),
        "reason": "heuristic_text_quality",
    }


def _contains_non_ascii_text(text: str) -> bool:
    if not text:
        return False
    return any(ord(ch) > 127 for ch in text)


def _translate_to_english_ollama(text: str) -> str:
    if not text or requests is None:
        return text

    prompt = (
        "Translate the following text to English. "
        "Return only the translated English text without any explanation:\n\n"
        f"{text}"
    )
    base_url = OLLAMA_BASE_URL.rstrip("/")
    endpoints = [f"{base_url}/api/generate", f"{base_url}/api/chat"]
    payloads = [
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": QWEN_MAX_NEW_TOKENS},
        },
        {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": QWEN_MAX_NEW_TOKENS},
        },
    ]

    for endpoint, payload in zip(endpoints, payloads):
        try:
            response = requests.post(endpoint, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            if endpoint.endswith("/api/generate"):
                translated = str(data.get("response") or "").strip()
            else:
                message = data.get("message") or {}
                translated = str(message.get("content") or "").strip()
            if translated:
                return translated
        except Exception:
            continue

    return text


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

    schema = (
        'Return strict JSON with keys: '
        '"sku", "product_name", "brand", "category", "subcategory", '
        '"clues", "confidence_score". '
        '"clues" must be a list of 2-3 explicit visual features (colors, text, shapes) confirming identity. '
        '"confidence_score" must be a strict heuristic rating from 0.00 to 1.00 based heavily on clue clarity. '
        'If unknown, set values to "unknown" and clues to an empty list. Do not add extra keys.'
    )
    return f"USER: <image>\n{task}\n{schema}\nRespond in English only.\nASSISTANT:"


def _extract_json_payload(text: str) -> Optional[Dict]:
    if not text:
        return None

    raw = text.strip()
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass

    code_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, re.IGNORECASE)
    if code_block:
        try:
            payload = json.loads(code_block.group(1))
            return payload if isinstance(payload, dict) else None
        except Exception:
            pass

    brace_match = re.search(r"(\{[\s\S]*\})", raw)
    if brace_match:
        try:
            payload = json.loads(brace_match.group(1))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None
    return None


def _normalize_structured_fields(payload: Optional[Dict]) -> Dict:
    if not payload:
        return {
            "sku": "unknown",
            "product_name": "unknown",
            "brand": "unknown",
            "category": "unknown",
            "subcategory": "unknown",
            "clues": [],
            "confidence_score": None,
        }

    clues = payload.get("clues")
    if isinstance(clues, list):
        clues = [str(item).strip() for item in clues if str(item).strip()][:3]
    else:
        clues = []

    confidence_score = payload.get("confidence_score")
    try:
        if confidence_score is not None:
            confidence_score = float(confidence_score)
            confidence_score = max(0.0, min(1.0, confidence_score))
    except Exception:
        confidence_score = None

    return {
        "sku": str(payload.get("sku", "unknown") or "unknown").strip(),
        "product_name": str(payload.get("product_name", "unknown") or "unknown").strip(),
        "brand": str(payload.get("brand", "unknown") or "unknown").strip(),
        "category": str(payload.get("category", "unknown") or "unknown").strip(),
        "subcategory": str(payload.get("subcategory", "unknown") or "unknown").strip(),
        "clues": clues,
        "confidence_score": confidence_score,
    }


def generate_qwen_sku_answer(
    image: Image.Image,
    broad_category: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> Dict:
    prompt = _build_prompt(broad_category=broad_category, user_prompt=user_prompt)

    if QWEN_PROVIDER == "ollama":
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

            base_url = OLLAMA_BASE_URL.rstrip("/")
            endpoints = [f"{base_url}/api/generate", f"{base_url}/api/chat"]
            payloads = [
                {
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {"num_predict": QWEN_MAX_NEW_TOKENS},
                },
                {
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"num_predict": QWEN_MAX_NEW_TOKENS},
                },
            ]

            last_error = None
            for endpoint, payload in zip(endpoints, payloads):
                try:
                    response = requests.post(endpoint, json=payload, timeout=180)
                    response.raise_for_status()
                    data = response.json()
                    answer = ""
                    if endpoint.endswith("/api/generate"):
                        answer = (data.get("response") or "").strip()
                    else:
                        message = data.get("message") or {}
                        answer = str(message.get("content") or "").strip()
                    if answer and _contains_non_ascii_text(answer):
                        answer = _translate_to_english_ollama(answer)

                    if answer:
                        parsed = _normalize_structured_fields(_extract_json_payload(answer))
                        parsed_confidence = parsed.get("confidence_score")
                        conf = _estimate_qwen_confidence(answer=answer)
                        if isinstance(parsed_confidence, float):
                            conf = {
                                "score": parsed_confidence,
                                "label": _confidence_label(parsed_confidence),
                                "reason": "model_provided_visual_clues",
                            }
                        summary_answer = answer
                        if parsed.get("sku") != "unknown" or parsed.get("product_name") != "unknown":
                            summary_answer = (
                                f"SKU: {parsed.get('sku')}; Product: {parsed.get('product_name')}; "
                                f"Brand: {parsed.get('brand')}; Category: {parsed.get('category')}; "
                                f"Subcategory: {parsed.get('subcategory')}"
                            )
                        return {
                            "provider": "ollama",
                            "model_id": OLLAMA_MODEL,
                            "prompt": prompt,
                            "answer": summary_answer,
                            "raw_answer": answer,
                            "structured": parsed,
                            "endpoint": endpoint,
                            "confidence": conf.get("score"),
                            "confidence_label": conf.get("label"),
                            "confidence_reason": conf.get("reason"),
                        }
                    last_error = RuntimeError("empty_response")
                except Exception as exc:
                    last_error = exc

            return {
                "provider": "ollama",
                "model_id": OLLAMA_MODEL,
                "prompt": prompt,
                "answer": "",
                "error": f"ollama_request_failed: {last_error}",
                "endpoint": base_url,
                "confidence": 0.05,
                "confidence_label": "low",
                "confidence_reason": "request_failed",
            }
        except Exception as exc:
            return {
                "provider": "ollama",
                "model_id": OLLAMA_MODEL,
                "prompt": prompt,
                "answer": "",
                "error": str(exc),
                "confidence": 0.05,
                "confidence_label": "low",
                "confidence_reason": "request_exception",
            }

    if QWEN_PROVIDER in {"hf", "huggingface", "transformers"}:
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration

            processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = LlavaForConditionalGeneration.from_pretrained(
                QWEN_MODEL_ID,
                torch_dtype=dtype,
            )
            inputs = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")
            target_device = getattr(model, "device", None)
            if target_device is not None:
                inputs = {key: value.to(target_device) for key, value in inputs.items()}
            output = model.generate(**inputs, max_new_tokens=QWEN_MAX_NEW_TOKENS, do_sample=False)
            decoded = processor.decode(output[0], skip_special_tokens=True)
            answer = decoded.split("ASSISTANT:")[-1].strip()
            parsed = _normalize_structured_fields(_extract_json_payload(answer))
            parsed_confidence = parsed.get("confidence_score")
            conf = _estimate_qwen_confidence(answer=answer)
            if isinstance(parsed_confidence, float):
                conf = {
                    "score": parsed_confidence,
                    "label": _confidence_label(parsed_confidence),
                    "reason": "model_provided_visual_clues",
                }
            summary_answer = answer
            if parsed.get("sku") != "unknown" or parsed.get("product_name") != "unknown":
                summary_answer = (
                    f"SKU: {parsed.get('sku')}; Product: {parsed.get('product_name')}; "
                    f"Brand: {parsed.get('brand')}; Category: {parsed.get('category')}; "
                    f"Subcategory: {parsed.get('subcategory')}"
                )
            return {
                "provider": "hf",
                "model_id": QWEN_MODEL_ID,
                "prompt": prompt,
                "answer": summary_answer,
                "raw_answer": answer,
                "structured": parsed,
                "confidence": conf.get("score"),
                "confidence_label": conf.get("label"),
                "confidence_reason": conf.get("reason"),
            }
        except Exception as exc:
            return {
                "provider": "hf",
                "model_id": QWEN_MODEL_ID,
                "prompt": prompt,
                "answer": "",
                "error": str(exc),
                "confidence": 0.05,
                "confidence_label": "low",
                "confidence_reason": "request_exception",
            }

    return {
        "provider": "disabled",
        "model_id": None,
        "prompt": prompt,
        "answer": "",
        "error": "qwen_provider_not_available",
        "confidence": 0.0,
        "confidence_label": "low",
        "confidence_reason": "provider_disabled",
    }
