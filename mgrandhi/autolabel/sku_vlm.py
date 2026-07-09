"""Benchmark open multimodal models for SKU/OCR extraction from product crops.

This module is for Module 3 / SKU extraction, not category classification. It asks a VLM to read
one product crop and return structured fields useful for product-master resolution and automatic
checkout: brand, product name, visible text, package size, barcode, and a review flag.

Backends:
  dry-run            Local deterministic output for parser/CSV/schema testing.
  gemini             Vertex Gemini reference ceiling (not open-source).
  openai-compatible  vLLM / Vertex Model Garden endpoints for Qwen-VL, PaliGemma, Gemma 3.

Example local validation:
  python -m autolabel.sku_vlm --backend dry-run --crops notebooks/samples --limit 5 \
      --out /tmp/sku_vlm_dryrun.csv

Example Qwen-VL endpoint call:
  export VLM_ENDPOINT_URL=http://<host>:8000/v1
  python -m autolabel.sku_vlm --backend openai-compatible --model Qwen/Qwen2.5-VL-7B-Instruct \
      --crops /tmp/crops_sample --limit 100 --out results/qwen25_sku_100.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROMPT_VERSION = "sku_ocr_v1"

SKU_EXTRACTION_PROMPT = (
    "You are reading one cropped retail product image for inventory and automatic checkout. "
    "Read packaging text carefully. Extract only what is visible or strongly recognizable. "
    "If the crop is blurry, partial, too small, or text is not readable, use empty strings and "
    "set needs_review=true. Return JSON only with exactly these keys: brand, product_name, "
    "sku_text, visible_text, package_size, barcode, category_hint, confidence, needs_review. "
    "confidence must be a number from 0 to 1."
)

SKU_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "brand": {"type": "string"},
        "product_name": {"type": "string"},
        "sku_text": {"type": "string"},
        "visible_text": {"type": "string"},
        "package_size": {"type": "string"},
        "barcode": {"type": "string"},
        "category_hint": {"type": "string"},
        "confidence": {"type": "number"},
        "needs_review": {"type": "boolean"},
    },
    "required": [
        "brand",
        "product_name",
        "sku_text",
        "visible_text",
        "package_size",
        "barcode",
        "category_hint",
        "confidence",
        "needs_review",
    ],
}

OUTPUT_FIELDS = [
    "filename",
    "crop_path",
    "backend",
    "model",
    "prompt_version",
    "brand",
    "product_name",
    "sku_text",
    "visible_text",
    "package_size",
    "barcode",
    "category_hint",
    "confidence",
    "needs_review",
    "parse_ok",
    "latency_s",
    "error",
    "raw_response",
]


@dataclass
class SkuPrediction:
    raw_response: str
    parsed: dict[str, Any]
    parse_ok: bool
    latency_s: float
    error: str = ""


def iter_crops(crops_dir: Path) -> list[Path]:
    return [p for p in sorted(crops_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS]


def sample_crops(crops: list[Path], limit: int, seed: int) -> list[Path]:
    if limit <= 0 or limit >= len(crops):
        return crops
    rng = random.Random(seed)
    return sorted(rng.sample(crops, limit))


def mime_for(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


def encode_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_for(path)};base64,{data}"


def parse_json_response(text: str) -> tuple[dict[str, Any], bool]:
    """Parse raw model text into the benchmark schema, tolerating fenced JSON."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return default_payload(needs_review=True), False
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return default_payload(needs_review=True), False

    payload = default_payload()
    for key in payload:
        if key in obj and obj[key] is not None:
            payload[key] = obj[key]
    payload["confidence"] = coerce_confidence(payload.get("confidence"))
    payload["needs_review"] = coerce_bool(payload.get("needs_review"))
    return payload, True


def default_payload(needs_review: bool = True) -> dict[str, Any]:
    return {
        "brand": "",
        "product_name": "",
        "sku_text": "",
        "visible_text": "",
        "package_size": "",
        "barcode": "",
        "category_hint": "",
        "confidence": 0.0,
        "needs_review": needs_review,
    }


def coerce_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "needs_review"}


class BaseBackend:
    name = "base"

    def __init__(self, model: str):
        self.model = model

    def predict(self, path: Path) -> SkuPrediction:
        raise NotImplementedError


class DryRunBackend(BaseBackend):
    name = "dry-run"

    def predict(self, path: Path) -> SkuPrediction:
        started = time.time()
        stem = path.stem.replace("_", " ")
        payload = default_payload(needs_review=True)
        payload["visible_text"] = stem
        payload["sku_text"] = stem
        payload["category_hint"] = "dry-run"
        payload["confidence"] = 0.1
        raw = json.dumps(payload)
        return SkuPrediction(raw, payload, True, round(time.time() - started, 4))


class GeminiBackend(BaseBackend):
    name = "gemini"

    def __init__(self, model: str, project: str, location: str):
        super().__init__(model)
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def predict(self, path: Path) -> SkuPrediction:
        started = time.time()
        types = self._types
        try:
            config_kwargs = {
                "response_mime_type": "application/json",
                "response_json_schema": SKU_EXTRACTION_SCHEMA,
                "temperature": 0.0,
            }
            if hasattr(types, "ThinkingConfig"):
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[
                    SKU_EXTRACTION_PROMPT,
                    types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_for(path)),
                ],
                config=types.GenerateContentConfig(**config_kwargs),
            )
            raw = resp.text or ""
            parsed, ok = parse_json_response(raw)
            return SkuPrediction(raw, parsed, ok, round(time.time() - started, 4))
        except Exception as exc:  # one crop should not kill the benchmark
            return SkuPrediction("", default_payload(), False, round(time.time() - started, 4), str(exc))


class OpenAICompatibleBackend(BaseBackend):
    """OpenAI-compatible chat/completions endpoint (vLLM, Vertex Model Garden endpoints)."""

    name = "openai-compatible"

    def __init__(self, model: str, base_url: str, api_key: str = "", timeout: int = 180):
        super().__init__(model)
        self.base_url = normalize_openai_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout

    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def predict(self, path: Path) -> SkuPrediction:
        started = time.time()
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 256,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SKU_EXTRACTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": encode_data_url(path)}},
                    ],
                }
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = self.chat_completions_url()
        req = urllib_request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw = data["choices"][0]["message"]["content"]
            parsed, ok = parse_json_response(raw)
            return SkuPrediction(raw, parsed, ok, round(time.time() - started, 4))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return SkuPrediction(
                "",
                default_payload(),
                False,
                round(time.time() - started, 4),
                f"{exc} body={body} url={url} model={self.model}",
            )
        except (URLError, KeyError, TimeoutError, json.JSONDecodeError) as exc:
            return SkuPrediction(
                "",
                default_payload(),
                False,
                round(time.time() - started, 4),
                f"{exc} url={url} model={self.model}",
            )


def normalize_openai_base_url(base_url: str) -> str:
    """Normalize common user-entered OpenAI URLs to the `/v1` base URL."""
    raw = (base_url or "").strip().rstrip("/")
    if not raw:
        return raw
    parsed = urlsplit(raw)
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/models"):
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


class VertexModelGardenBackend(BaseBackend):
    """Vertex AI Model Garden endpoint for managed PaliGemma-style OCR."""

    name = "vertex-model-garden"

    def __init__(
        self,
        model: str,
        project: str,
        location: str,
        endpoint: str,
        dedicated_dns: str = "",
        timeout: int = 180,
    ):
        super().__init__(model or "google/paligemma@paligemma-mix-448-float16")
        self.project = project
        self.location = location
        self.endpoint = endpoint
        self.dedicated_dns = dedicated_dns
        self.timeout = timeout

    def predict_url(self) -> str:
        endpoint = self.endpoint.strip()
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        if ".prediction.vertexai.goog" in endpoint:
            host = endpoint.removeprefix("https://").removeprefix("http://").rstrip("/")
            endpoint_id = host.split(".", 1)[0]
            return (
                f"https://{host}/v1/projects/{self.project}/locations/{self.location}"
                f"/endpoints/{endpoint_id}:predict"
            )
        endpoint_id = endpoint.split("/")[-1]
        if self.dedicated_dns:
            host = self.dedicated_dns.removeprefix("https://").removeprefix("http://").rstrip("/")
            return (
                f"https://{host}/v1/projects/{self.project}/locations/{self.location}"
                f"/endpoints/{endpoint_id}:predict"
            )
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project}"
            f"/locations/{self.location}/endpoints/{endpoint_id}:predict"
        )

    def predict(self, path: Path) -> SkuPrediction:
        started = time.time()
        url = self.predict_url()
        try:
            token = subprocess.check_output(
                ["gcloud", "auth", "print-access-token"], text=True, timeout=20
            ).strip()
            payload = {
                "instances": [
                    {
                        # The managed JAX PaliGemma container expects task prompts like `ocr`.
                        "prompt": "ocr",
                        "image": image_to_base64(path),
                    }
                ]
            }
            req = urllib_request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw = extract_vertex_response(data)
            parsed = default_payload(needs_review=True)
            parsed["visible_text"] = raw
            parsed["sku_text"] = raw
            parsed["confidence"] = 0.5 if raw else 0.0
            return SkuPrediction(raw, parsed, bool(raw), round(time.time() - started, 4))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return SkuPrediction(
                "",
                default_payload(),
                False,
                round(time.time() - started, 4),
                f"{exc} body={body} url={url}",
            )
        except Exception as exc:
            return SkuPrediction(
                "",
                default_payload(),
                False,
                round(time.time() - started, 4),
                f"{exc} url={url}",
            )


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def extract_vertex_response(data: dict[str, Any]) -> str:
    predictions = data.get("predictions") or []
    if not predictions:
        return ""
    first = predictions[0]
    if isinstance(first, dict):
        return str(first.get("response") or first.get("prediction") or "").strip()
    return str(first).strip()


def build_backend(args: argparse.Namespace) -> BaseBackend:
    if args.backend == "dry-run":
        return DryRunBackend(args.model or "dry-run")
    if args.backend == "gemini":
        project = args.project or os.environ.get("PROJECT_ID")
        location = args.location or os.environ.get("REGION") or "us-central1"
        model = args.model or os.environ.get("VERTEX_MODEL") or "gemini-2.5-flash"
        if not project:
            raise SystemExit("gemini backend requires PROJECT_ID or --project")
        return GeminiBackend(model, project, location)
    if args.backend == "openai-compatible":
        base_url = args.endpoint or os.environ.get("VLM_ENDPOINT_URL")
        if not base_url:
            raise SystemExit("openai-compatible backend requires --endpoint or VLM_ENDPOINT_URL")
        api_key = args.api_key or os.environ.get("VLM_API_KEY", "")
        return OpenAICompatibleBackend(args.model, base_url, api_key=api_key, timeout=args.timeout)
    if args.backend == "vertex-model-garden":
        project = args.project or os.environ.get("PROJECT_ID")
        location = args.location or os.environ.get("REGION") or "us-central1"
        endpoint = args.endpoint or os.environ.get("VERTEX_MODEL_GARDEN_ENDPOINT_ID")
        dedicated_dns = getattr(args, "dedicated_dns", "") or os.environ.get(
            "VERTEX_MODEL_GARDEN_DEDICATED_DNS", ""
        )
        if not project or not endpoint:
            raise SystemExit(
                "vertex-model-garden backend requires PROJECT_ID and --endpoint/"
                "VERTEX_MODEL_GARDEN_ENDPOINT_ID"
            )
        return VertexModelGardenBackend(
            args.model,
            project,
            location,
            endpoint,
            dedicated_dns=dedicated_dns,
            timeout=args.timeout,
        )
    raise SystemExit(f"unknown backend: {args.backend}")


def write_prediction(writer: csv.DictWriter, path: Path, backend: BaseBackend, pred: SkuPrediction) -> None:
    row = {
        "filename": path.name,
        "crop_path": str(path),
        "backend": backend.name,
        "model": backend.model,
        "prompt_version": PROMPT_VERSION,
        "parse_ok": int(pred.parse_ok),
        "latency_s": pred.latency_s,
        "error": pred.error,
        "raw_response": pred.raw_response,
    }
    row.update(pred.parsed)
    row["needs_review"] = int(coerce_bool(row["needs_review"]))
    writer.writerow(row)


def summarize(rows: list[dict[str, Any]], backend: BaseBackend, out_path: Path) -> dict[str, Any]:
    n = len(rows)
    parse_ok = sum(int(r["parse_ok"]) for r in rows)
    errors = sum(1 for r in rows if r.get("error"))
    useful_name = sum(1 for r in rows if r.get("brand") or r.get("product_name"))
    useful_ocr = sum(1 for r in rows if r.get("visible_text") or r.get("sku_text"))
    review = sum(int(r["needs_review"]) for r in rows)
    avg_latency = sum(float(r["latency_s"]) for r in rows) / n if n else 0.0
    summary = {
        "backend": backend.name,
        "model": backend.model,
        "prompt_version": PROMPT_VERSION,
        "rows": n,
        "parse_success_rate": round(parse_ok / n, 4) if n else 0.0,
        "error_rate": round(errors / n, 4) if n else 0.0,
        "brand_or_product_rate": round(useful_name / n, 4) if n else 0.0,
        "ocr_text_rate": round(useful_ocr / n, 4) if n else 0.0,
        "needs_review_rate": round(review / n, 4) if n else 0.0,
        "avg_latency_s": round(avg_latency, 4),
        "out_csv": str(out_path),
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--backend",
        required=True,
        choices=["dry-run", "gemini", "openai-compatible", "vertex-model-garden"],
    )
    parser.add_argument("--model", default="", help="Model id/name sent to the backend.")
    parser.add_argument("--endpoint", help="OpenAI-compatible base URL, e.g. http://host:8000/v1")
    parser.add_argument("--dedicated-dns", help="Dedicated Vertex prediction DNS for Model Garden.")
    parser.add_argument("--api-key", default="", help="Bearer token for the OpenAI-compatible endpoint.")
    parser.add_argument("--project", help="GCP project for Gemini reference backend.")
    parser.add_argument("--location", default="us-central1", help="Vertex location for Gemini reference backend.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-request timeout seconds.")
    parser.add_argument("--crops", required=True, help="Directory of crop images.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--limit", type=int, default=100, help="Max crops to sample (0 = all).")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic sample seed.")
    args = parser.parse_args()

    crops = sample_crops(iter_crops(Path(args.crops)), args.limit, args.seed)
    if not crops:
        raise SystemExit(f"no crop images found under {args.crops}")
    backend = build_backend(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for i, crop in enumerate(crops, start=1):
            pred = backend.predict(crop)
            write_prediction(writer, crop, backend, pred)
            row = {
                **pred.parsed,
                "parse_ok": int(pred.parse_ok),
                "needs_review": int(pred.parsed.get("needs_review", True)),
                "latency_s": pred.latency_s,
                "error": pred.error,
            }
            rows.append(row)
            if i % 25 == 0 or i == len(crops):
                print(f"processed {i}/{len(crops)} crops")

    summary = summarize(rows, backend, out_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
