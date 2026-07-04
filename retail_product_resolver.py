import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


_EASYOCR_READER = None
_LLAMA_TOKENIZER = None
_LLAMA_MODEL = None


def _clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _tokens(value: Optional[str]) -> List[str]:
    text = _clean_text(value).lower()
    return [tok for tok in re.findall(r"[a-z0-9]+", text) if len(tok) >= 2]


def _same_label(left: Optional[str], right: Optional[str]) -> bool:
    return _clean_text(left).lower() == _clean_text(right).lower()


def _product_like_name(product_name: Optional[str], category: Optional[str], subcategory: Optional[str]) -> str:
    cleaned = _clean_text(product_name)
    if not cleaned or cleaned.lower() in {"unknown", "none", "n/a", "na"}:
        return ""
    if _same_label(cleaned, category) or _same_label(cleaned, subcategory):
        return ""
    return cleaned


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _extract_easyocr_lines(result) -> List[Dict]:
    lines = []
    for item in result or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        cleaned = _clean_text(item[1])
        if not cleaned:
            continue
        confidence = item[2] if len(item) > 2 else None
        lines.append({"text": cleaned, "confidence": _safe_float(confidence)})
    return lines


def extract_easyocr_text(image: Image.Image) -> Dict:
    global _EASYOCR_READER

    if _to_bool(os.getenv("DISABLE_EASYOCR"), default=False):
        return {"backend": "easyocr", "available": False, "text": "", "lines": [], "status": "disabled"}

    try:
        import easyocr

        if _EASYOCR_READER is None:
            langs = [
                lang.strip()
                for lang in os.getenv("EASYOCR_LANGS", os.getenv("OCR_LANGS", "en")).split(",")
                if lang.strip()
            ]
            model_dir = os.getenv("EASYOCR_MODEL_DIR") or None
            reader_kwargs = {
                "gpu": _to_bool(os.getenv("EASYOCR_GPU"), default=False),
                "verbose": _to_bool(os.getenv("EASYOCR_VERBOSE"), default=False),
            }
            if model_dir:
                reader_kwargs["model_storage_directory"] = model_dir
                reader_kwargs["user_network_directory"] = model_dir
            _EASYOCR_READER = easyocr.Reader(langs, **reader_kwargs)

        rgb = np.array(image.convert("RGB"))
        result = _EASYOCR_READER.readtext(rgb)
        lines = _extract_easyocr_lines(result)
        text = _clean_text(" ".join(line["text"] for line in lines))
        return {
            "backend": "easyocr",
            "available": bool(text),
            "text": text,
            "lines": lines,
        }
    except Exception as exc:
        return {
            "backend": "easyocr",
            "available": False,
            "text": "",
            "lines": [],
            "error": str(exc),
        }


def _candidate_from_neighbor(neighbor: Dict, rank: int) -> Dict:
    label = _clean_text(neighbor.get("label"))
    subcategory = _clean_text(neighbor.get("subcategory"))
    raw_product_name = _clean_text(
        neighbor.get("product_name")
        or neighbor.get("name")
        or neighbor.get("title")
        or neighbor.get("full_label")
    )
    product_name = _product_like_name(raw_product_name, label, subcategory)
    return {
        "rank": rank,
        "product_name": product_name or "unknown",
        "category": label or "unknown",
        "subcategory": subcategory or "unknown",
        "score": _safe_float(neighbor.get("score")),
        "path": neighbor.get("path"),
    }


def _build_candidates(swin_result: Optional[Dict], limit: int = 10) -> List[Dict]:
    if not isinstance(swin_result, dict):
        return []
    neighbors = swin_result.get("neighbors") or []
    candidates = []
    seen = set()
    for rank, neighbor in enumerate(neighbors[:limit], start=1):
        candidate = _candidate_from_neighbor(neighbor, rank)
        key = (
            candidate["product_name"].lower(),
            candidate["category"].lower(),
            candidate["subcategory"].lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def _score_candidate(
    candidate: Dict,
    category_hint: Optional[str],
    subcategory_hint: Optional[str],
    ocr_text: Optional[str] = None,
) -> float:
    hint_tokens = set(_tokens(" ".join([category_hint or "", subcategory_hint or ""])))
    ocr_tokens = set(_tokens(ocr_text))
    fields = " ".join([candidate.get("product_name", ""), candidate.get("category", ""), candidate.get("subcategory", "")])
    candidate_tokens = set(_tokens(fields))
    hint_overlap = len(hint_tokens & candidate_tokens) / max(1, len(candidate_tokens))
    ocr_overlap = len(ocr_tokens & candidate_tokens) / max(1, len(candidate_tokens))
    visual = 1.0 / max(1, int(candidate.get("rank") or 1))
    return round((0.2 * hint_overlap) + (0.35 * ocr_overlap) + (0.45 * visual), 4)


def _heuristic_decision(
    candidates: List[Dict],
    category_hint: Optional[str],
    subcategory_hint: Optional[str],
    ocr_text: Optional[str] = None,
) -> Dict:
    if not candidates:
        return {
            "predicted_product": "unknown",
            "brand": "unknown",
            "category": category_hint or "unknown",
            "subcategory": subcategory_hint or "unknown",
            "confidence": 0.0,
            "reason": "No FAISS candidates were available.",
            "source": "heuristic",
        }

    ranked = sorted(
        [
            {
                **candidate,
                "combined_score": _score_candidate(candidate, category_hint, subcategory_hint, ocr_text),
            }
            for candidate in candidates
        ],
        key=lambda item: (-item["combined_score"], item["rank"]),
    )
    best = ranked[0]
    confidence = 0.72 if best["rank"] == 1 else 0.58
    if best["combined_score"] >= 0.55:
        confidence = min(0.9, confidence + 0.08)

    return {
        "predicted_product": _product_like_name(best.get("product_name"), best.get("category"), best.get("subcategory")) or "unknown",
        "brand": "unknown",
        "category": best.get("category") or category_hint or "unknown",
        "subcategory": best.get("subcategory") or subcategory_hint or "unknown",
        "confidence": round(confidence, 3),
        "reason": "Ranked FAISS candidates using visual rank, EasyOCR text overlap, and category hints.",
        "source": "heuristic",
        "ranked_candidates": ranked[:5],
    }


def _extract_json_object(text: str) -> Optional[Dict]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _retail_reasoning_prompt(
    ocr_text: str,
    candidates: List[Dict],
    category_hint: Optional[str],
    subcategory_hint: Optional[str],
) -> str:
    payload = {
        "ocr_text": ocr_text or "",
        "category_hint": category_hint or "unknown",
        "subcategory_hint": subcategory_hint or "unknown",
        "faiss_candidates": candidates[:10],
    }
    return (
        "You are a retail product identification assistant. Use only the evidence in the JSON payload. "
        "Prefer FAISS visual candidates unless OCR text strongly supports a different candidate. "
        "Return JSON only with keys predicted_product, brand, category, subcategory, confidence, reason.\n\n"
        f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _load_hf_llama_if_needed():
    global _LLAMA_TOKENIZER, _LLAMA_MODEL
    if _LLAMA_TOKENIZER is not None and _LLAMA_MODEL is not None:
        return _LLAMA_TOKENIZER, _LLAMA_MODEL

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = os.getenv("LLAMA_MODEL_ID", os.getenv("RETAIL_REASONER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"))
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs = {"torch_dtype": dtype}
    if _to_bool(os.getenv("LLAMA_LOAD_IN_4BIT"), default=False):
        kwargs["load_in_4bit"] = True
        kwargs.pop("torch_dtype", None)
    if torch.cuda.is_available():
        kwargs["device_map"] = os.getenv("LLAMA_DEVICE_MAP", "auto")

    token = os.getenv("HF_TOKEN") or None
    _LLAMA_TOKENIZER = AutoTokenizer.from_pretrained(model_id, token=token)
    if token:
        kwargs["token"] = token
    _LLAMA_MODEL = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return _LLAMA_TOKENIZER, _LLAMA_MODEL


def _call_hf_llama(prompt: str) -> Dict:
    tokenizer, model = _load_hf_llama_if_needed()
    model_id = os.getenv("LLAMA_MODEL_ID", os.getenv("RETAIL_REASONER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"))
    messages = [
        {
            "role": "system",
            "content": "You return compact JSON only. Do not add prose outside JSON.",
        },
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"{messages[0]['content']}\n\n{messages[1]['content']}\n"

    inputs = tokenizer(text, return_tensors="pt")
    target_device = getattr(model, "device", None)
    if target_device is not None:
        inputs = {key: value.to(target_device) for key, value in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=int(os.getenv("LLAMA_MAX_NEW_TOKENS", "128")),
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output[0][inputs["input_ids"].shape[-1] :]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    parsed = _extract_json_object(raw) or {}
    parsed.setdefault("raw_response", raw)
    parsed.setdefault("provider", "hf")
    parsed.setdefault("model", model_id)
    return parsed


def preload_retail_models() -> Dict:
    status = {}
    if _to_bool(os.getenv("PRELOAD_EASYOCR"), default=True):
        probe = Image.new("RGB", (96, 32), "white")
        status["easyocr"] = extract_easyocr_text(probe)
    if _to_bool(os.getenv("PRELOAD_LLAMA"), default=True):
        try:
            tokenizer, model = _load_hf_llama_if_needed()
            status["llama"] = {
                "provider": "hf",
                "model": os.getenv("LLAMA_MODEL_ID", os.getenv("RETAIL_REASONER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")),
                "loaded": tokenizer is not None and model is not None,
            }
        except Exception as exc:
            status["llama"] = {"error": str(exc)}
    return status


def _normalize_decision(decision: Dict, fallback: Dict) -> Dict:
    normalized = dict(fallback)
    if isinstance(decision, dict):
        normalized.update({key: value for key, value in decision.items() if value not in (None, "")})
    normalized["confidence"] = max(0.0, min(1.0, _safe_float(normalized.get("confidence"), fallback.get("confidence", 0.0))))
    for key in ["predicted_product", "brand", "category", "subcategory", "reason", "source"]:
        normalized[key] = _clean_text(normalized.get(key)) or fallback.get(key) or "unknown"
    if normalized.get("source") == "heuristic":
        normalized["source"] = "llama"
    return normalized


def _reason_with_llama(
    ocr_text: str,
    candidates: List[Dict],
    category_hint: Optional[str],
    subcategory_hint: Optional[str],
) -> Dict:
    provider = os.getenv("RETAIL_REASONER_PROVIDER", os.getenv("LLAMA_PROVIDER", "hf")).strip().lower()
    if provider in {"0", "false", "off", "none", "disabled"}:
        return {"status": "skipped", "reason": "provider_disabled", "provider": provider}

    prompt = _retail_reasoning_prompt(ocr_text, candidates, category_hint, subcategory_hint)
    if provider in {"hf", "huggingface", "transformers"}:
        return _call_hf_llama(prompt)
    return {"status": "skipped", "reason": f"unsupported_provider:{provider}", "provider": provider}


def resolve_retail_product(
    image: Image.Image,
    swin_result: Optional[Dict],
    category_hint: Optional[str] = None,
    subcategory_hint: Optional[str] = None,
) -> Dict:
    ocr = extract_easyocr_text(image)
    ocr_text = ocr.get("text", "")
    candidates = _build_candidates(swin_result)
    fallback = _heuristic_decision(candidates, category_hint, subcategory_hint, ocr_text)

    try:
        reasoner = _reason_with_llama(ocr_text, candidates, category_hint, subcategory_hint)
        if isinstance(reasoner, dict) and reasoner.get("status") != "skipped" and not reasoner.get("error"):
            decision = _normalize_decision(reasoner, fallback)
        else:
            decision = fallback
    except Exception as exc:
        reasoner = {"error": "llama_reasoner_failed", "details": str(exc)}
        decision = fallback

    return {
        "ocr": ocr,
        "faiss_candidates": candidates,
        "decision": decision,
        "reasoner": reasoner,
    }


def summarize_retail_decisions(rows: List[Dict]) -> Dict:
    categories = Counter()
    subcategories = Counter()
    products = Counter()
    for row in rows:
        decision = (row.get("retail_product") or {}).get("decision") or {}
        category = _clean_text(decision.get("category") or row.get("product_category") or row.get("category"))
        subcategory = _clean_text(decision.get("subcategory") or row.get("subcategory"))
        product = _product_like_name(
            decision.get("predicted_product"),
            decision.get("category") or row.get("product_category") or row.get("category"),
            decision.get("subcategory") or row.get("subcategory"),
        )
        if category and category.lower() != "unknown":
            categories[category] += 1
        if subcategory and subcategory.lower() != "unknown":
            subcategories[subcategory] += 1
        if product and product.lower() != "unknown":
            products[product] += 1
    return {
        "top_categories": categories.most_common(5),
        "top_subcategories": subcategories.most_common(5),
        "top_products": products.most_common(5),
    }
