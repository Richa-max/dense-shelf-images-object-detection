from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib import request as urllib_request

import pandas as pd


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

SUGGESTED_QUESTIONS = [
    "How many products are on the shelf?",
    "Which category has the most products?",
    "How many distinct categories are present?",
    "Which items need review?",
    "What is the empty shelf percentage?",
    "Show category counts",
    "How many scans are saved?",
    "What products were seen recently?",
]


@dataclass
class BIAnswer:
    text: str
    source: str
    table: pd.DataFrame | None = None


def ollama_available() -> bool:
    try:
        with urllib_request.urlopen(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def category_counts(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "category" not in df.columns:
        return pd.Series(dtype="int64")
    categories = df["category"].fillna("unknown").astype(str)
    known = categories[categories.str.lower() != "unknown"]
    return known.value_counts()


def _llm_answer(question: str, items_df: pd.DataFrame, scans_df: pd.DataFrame) -> str | None:
    prompt = (
        "Answer the retail shelf inventory question from these summarized tables. "
        "Be concise and only use the supplied data.\n\n"
        f"Question: {question}\n\n"
        f"Items summary:\n{items_df.head(80).to_csv(index=False)}\n\n"
        f"Scans summary:\n{scans_df.head(20).to_csv(index=False)}"
    )
    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode()
    req = urllib_request.Request(
        f"{OLLAMA_URL.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(data.get("response", "")).strip() or None
    except Exception:
        return None


def answer(question: str, items_df: pd.DataFrame, scans_df: pd.DataFrame, use_llm: bool = False) -> BIAnswer:
    q = question.lower().strip()

    if use_llm and ollama_available():
        llm_text = _llm_answer(question, items_df, scans_df)
        if llm_text:
            return BIAnswer(llm_text, f"ollama:{OLLAMA_MODEL}")

    if items_df is None:
        items_df = pd.DataFrame()
    if scans_df is None:
        scans_df = pd.DataFrame()

    counts = category_counts(items_df)

    if "category" in q or "counts" in q or "most" in q:
        table = counts.rename_axis("category").reset_index(name="count")
        if table.empty:
            return BIAnswer("No category data is available yet.", "rule-based", table)
        top = table.iloc[0]
        return BIAnswer(
            f"{top['category']} has the most products ({int(top['count'])}).",
            "rule-based",
            table,
        )

    if "distinct" in q:
        return BIAnswer(f"There are {int(len(counts))} distinct known categories.", "rule-based")

    if "review" in q:
        if items_df.empty or "score" not in items_df.columns:
            return BIAnswer("No review score data is available yet.", "rule-based")
        review = items_df[pd.to_numeric(items_df["score"], errors="coerce").fillna(0) <= 0]
        return BIAnswer(f"{len(review)} items may need review.", "rule-based", review.head(50))

    if "empty" in q and not scans_df.empty and "empty_pct" in scans_df.columns:
        latest = scans_df.iloc[0]
        pct = float(latest.get("empty_pct", 0.0)) * 100
        return BIAnswer(f"The latest scan has about {pct:.0f}% empty shelf space.", "rule-based")

    if "scan" in q and scans_df is not None:
        return BIAnswer(f"There are {len(scans_df)} saved scans.", "rule-based", scans_df.head(20))

    if "recent" in q or "product" in q or "item" in q:
        return BIAnswer(f"There are {len(items_df)} saved inventory items.", "rule-based", items_df.head(50))

    return BIAnswer(
        f"I found {len(items_df)} items across {len(counts)} known categories.",
        "rule-based",
        counts.rename_axis("category").reset_index(name="count") if not counts.empty else None,
    )

