import re
from collections import Counter
from typing import Dict, List, Optional

from PIL import Image


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


def _score_candidate(candidate: Dict, category_hint: Optional[str], subcategory_hint: Optional[str]) -> float:
    hint_tokens = set(_tokens(" ".join([category_hint or "", subcategory_hint or ""])))
    fields = " ".join([candidate.get("product_name", ""), candidate.get("category", ""), candidate.get("subcategory", "")])
    cand_tokens = set(_tokens(fields))
    overlap = len(hint_tokens & cand_tokens) / max(1, len(cand_tokens))
    visual = 1.0 / max(1, int(candidate.get("rank") or 1))
    return round((0.35 * overlap) + (0.65 * visual), 4)


def _heuristic_decision(candidates: List[Dict], category_hint: Optional[str], subcategory_hint: Optional[str]) -> Dict:
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
                "combined_score": _score_candidate(candidate, category_hint, subcategory_hint),
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
        "reason": "Ranked FAISS candidates using visual rank and category hints.",
        "source": "heuristic",
        "ranked_candidates": ranked[:5],
    }


def resolve_retail_product(
    image: Image.Image,
    swin_result: Optional[Dict],
    category_hint: Optional[str] = None,
    subcategory_hint: Optional[str] = None,
) -> Dict:
    candidates = _build_candidates(swin_result)
    decision = _heuristic_decision(candidates, category_hint, subcategory_hint)

    return {
        "faiss_candidates": candidates,
        "decision": decision,
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
