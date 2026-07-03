import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import requests
import numpy as np
from PIL import Image, ImageOps


_EASYOCR_READER = None


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


def _ocr_with_easyocr(image: Image.Image) -> Dict:
    global _EASYOCR_READER
    import easyocr

    if _EASYOCR_READER is None:
        langs = [
            lang.strip()
            for lang in os.getenv("OCR_LANGS", "en").split(",")
            if lang.strip()
        ]
        _EASYOCR_READER = easyocr.Reader(langs, gpu=os.getenv("OCR_GPU", "0") in {"1", "true", "yes"})

    rgb = np.array(image.convert("RGB"))
    results = _EASYOCR_READER.readtext(rgb)
    parts = []
    words = []
    for item in results:
        if len(item) < 3:
            continue
        text = _clean_text(item[1])
        conf = _safe_float(item[2])
        if not text:
            continue
        parts.append(text)
        words.append({"text": text, "confidence": conf})
    return {
        "backend": "easyocr",
        "text": _clean_text(" ".join(parts)),
        "words": words,
        "available": True,
    }


def _ocr_with_pytesseract(image: Image.Image) -> Dict:
    import pytesseract

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    text = pytesseract.image_to_string(gray)
    return {
        "backend": "pytesseract",
        "text": _clean_text(text),
        "words": [],
        "available": True,
    }


def extract_ocr_text(image: Image.Image) -> Dict:
    """Run OCR if an optional OCR backend is installed.

    The project can still run without OCR dependencies; the response records why
    OCR was skipped so API clients can distinguish absence from empty labels.
    """
    backend = os.getenv("OCR_BACKEND", "auto").strip().lower()
    if backend in {"0", "false", "off", "none", "disabled"}:
        return {"backend": "disabled", "text": "", "words": [], "available": False}

    attempts = []
    candidates = ["easyocr", "pytesseract"] if backend == "auto" else [backend]
    for candidate in candidates:
        try:
            if candidate == "easyocr":
                return _ocr_with_easyocr(image)
            if candidate in {"tesseract", "pytesseract"}:
                return _ocr_with_pytesseract(image)
        except Exception as exc:
            attempts.append({"backend": candidate, "error": str(exc)})

    return {
        "backend": backend,
        "text": "",
        "words": [],
        "available": False,
        "attempts": attempts,
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


def _score_candidate(candidate: Dict, ocr_text: str) -> float:
    ocr_tokens = set(_tokens(ocr_text))
    fields = " ".join([candidate.get("product_name", ""), candidate.get("category", ""), candidate.get("subcategory", "")])
    cand_tokens = set(_tokens(fields))
    overlap = len(ocr_tokens & cand_tokens) / max(1, len(cand_tokens))
    visual = 1.0 / max(1, int(candidate.get("rank") or 1))
    return round((0.65 * overlap) + (0.35 * visual), 4)


def _heuristic_decision(candidates: List[Dict], ocr_text: str, category_hint: Optional[str], subcategory_hint: Optional[str]) -> Dict:
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
                "combined_score": _score_candidate(candidate, ocr_text),
            }
            for candidate in candidates
        ],
        key=lambda item: (-item["combined_score"], item["rank"]),
    )
    best = ranked[0]
    ocr_available = bool(_tokens(ocr_text))
    confidence = 0.72 if best["rank"] == 1 else 0.58
    if ocr_available and best["combined_score"] >= 0.35:
        confidence = min(0.95, confidence + 0.18)
    elif ocr_available:
        confidence = min(0.85, confidence + 0.06)

    brand = "unknown"
    text_tokens = _tokens(ocr_text)
    if text_tokens:
        brand = text_tokens[0].upper()

    return {
        "predicted_product": _product_like_name(best.get("product_name"), best.get("category"), best.get("subcategory")) or "unknown",
        "brand": brand,
        "category": best.get("category") or category_hint or "unknown",
        "subcategory": best.get("subcategory") or subcategory_hint or "unknown",
        "confidence": round(confidence, 3),
        "reason": "Ranked FAISS candidates using OCR token overlap and visual rank.",
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


def _call_ollama_retail_slm(ocr_text: str, candidates: List[Dict], category_hint: Optional[str], subcategory_hint: Optional[str]) -> Dict:
    base_url = os.getenv("RETAIL_SLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    model = os.getenv("RETAIL_SLM_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    payload = {
        "model": model,
        "stream": False,
        "prompt": (
            "You are a retail product identification model. Use OCR text and FAISS candidates; "
            "do not invent products outside the evidence. Return JSON only with keys: "
            "predicted_product, brand, category, subcategory, confidence, reason.\n\n"
            f"OCR text: {ocr_text or '(none)'}\n"
            f"Category hint: {category_hint or 'unknown'}\n"
            f"Subcategory hint: {subcategory_hint or 'unknown'}\n"
            f"FAISS candidates: {json.dumps(candidates[:10], ensure_ascii=False)}"
        ),
        "options": {"temperature": 0.0, "num_predict": int(os.getenv("RETAIL_SLM_MAX_TOKENS", "256"))},
    }
    response = requests.post(f"{base_url}/api/generate", json=payload, timeout=90)
    response.raise_for_status()
    data = response.json()
    raw = data.get("response", "")
    parsed = _extract_json_object(raw) or {}
    parsed.setdefault("source", "ollama")
    parsed.setdefault("model", model)
    parsed.setdefault("raw_response", raw)
    return parsed


def _normalize_decision(decision: Dict, fallback: Dict) -> Dict:
    normalized = dict(fallback)
    if isinstance(decision, dict):
        normalized.update({k: v for k, v in decision.items() if v not in (None, "")})
    normalized["confidence"] = max(0.0, min(1.0, _safe_float(normalized.get("confidence"), fallback.get("confidence", 0.0))))
    for key in ["predicted_product", "brand", "category", "subcategory", "reason", "source"]:
        normalized[key] = _clean_text(normalized.get(key)) or fallback.get(key) or "unknown"
    return normalized


def resolve_retail_product(
    image: Image.Image,
    swin_result: Optional[Dict],
    category_hint: Optional[str] = None,
    subcategory_hint: Optional[str] = None,
    disable_slm: bool = False,
) -> Dict:
    ocr = extract_ocr_text(image)
    ocr_text = ocr.get("text", "")
    candidates = _build_candidates(swin_result)
    fallback = _heuristic_decision(candidates, ocr_text, category_hint, subcategory_hint)

    slm_result = None
    provider = os.getenv("RETAIL_SLM_PROVIDER", "none").strip().lower()
    if not disable_slm and provider in {"ollama", "local_ollama"}:
        try:
            slm_result = _call_ollama_retail_slm(ocr_text, candidates, category_hint, subcategory_hint)
            decision = _normalize_decision(slm_result, fallback)
        except Exception as exc:
            slm_result = {"error": "retail_slm_failed", "details": str(exc), "provider": provider}
            decision = fallback
    else:
        reason = "disabled" if disable_slm else f"provider_not_configured:{provider}"
        slm_result = {"status": "skipped", "reason": reason}
        decision = fallback

    return {
        "ocr": ocr,
        "faiss_candidates": candidates,
        "decision": decision,
        "slm": slm_result,
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
