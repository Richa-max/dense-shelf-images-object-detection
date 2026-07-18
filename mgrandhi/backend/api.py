"""FastAPI boundary for the operator-first retail shelf application."""
from __future__ import annotations

import io
import logging
import math
import os
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from mgrandhi.backend import analysis_service
from mgrandhi.backend import inventory_db as db
from mgrandhi.bi_interface import bi_engine

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "50000000"))
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}
WEB_DIST = Path(__file__).resolve().parents[1] / "frontend" / "web" / "dist"


class JobAccepted(BaseModel):
    job_id: str
    status: Literal["queued"]


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "complete", "failed"]
    stage: str
    progress: int = Field(ge=0, le=100)
    message: str
    result: dict[str, Any] | None = None
    error: str | None = None


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class QuestionResponse(BaseModel):
    text: str
    source: str
    table: list[dict[str, Any]] | None = None


class FeedbackRequest(BaseModel):
    scan_id: int = Field(gt=0)
    crop_id: int = Field(gt=0)
    feedback_type: Literal["category", "sku"]
    verdict: Literal["correct", "incorrect"]
    correction: str = Field(default="", max_length=500)
    note: str = Field(default="", max_length=1000)


class _JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        job_id = uuid4().hex
        with self._lock:
            if len(self._jobs) >= 50:
                finished = [
                    key
                    for key, value in self._jobs.items()
                    if value["status"] in {"complete", "failed"}
                ]
                for key in finished[: max(1, len(self._jobs) - 49)]:
                    self._jobs.pop(key, None)
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "stage": "queued",
                "progress": 0,
                "message": "Your scan is waiting to start",
                "result": None,
                "error": None,
            }
        return job_id

    def active_count(self) -> int:
        with self._lock:
            return sum(
                job["status"] in {"queued", "processing"} for job in self._jobs.values()
            )

    def update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(values)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


jobs = _JobStore()
inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shelf-inference")
model_state: dict[str, Any] = {"status": "loading", "details": {}}
model_state_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    if os.getenv("PRELOAD_MODELS", "1") == "1":
        inference_executor.submit(_warm_models)
    else:
        with model_state_lock:
            model_state["status"] = "lazy"
    yield


app = FastAPI(title="Retail Shelf Assistant API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("DEV_CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        return _safe_value(value.item())
    return value


def _frame_records(frame) -> list[dict[str, Any]]:
    return [{key: _safe_value(value) for key, value in row.items()} for row in frame.to_dict("records")]


def _public_records(frame) -> list[dict[str, Any]]:
    """Serialize records without exposing server filesystem paths."""
    return _frame_records(frame.drop(columns=["image_path", "crop_path"], errors="ignore"))


def _warm_models() -> None:
    try:
        details = analysis_service.preload_models()
        with model_state_lock:
            model_state["details"] = details
            model_state["status"] = "ready"
    except Exception as exc:
        with model_state_lock:
            model_state["status"] = "degraded"
            model_state["details"] = {"message": str(exc)}


def _run_analysis(
    job_id: str,
    image_bytes: bytes,
    filename: str,
    max_crops: int,
    max_sku_crops: int,
    extract_sku: bool,
) -> None:
    def report(stage: str, progress: int, message: str) -> None:
        jobs.update(
            job_id,
            status="processing" if progress < 100 else "complete",
            stage=stage,
            progress=progress,
            message=message,
        )

    try:
        jobs.update(
            job_id,
            status="processing",
            stage="starting",
            progress=3,
            message="Starting shelf analysis",
        )
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = analysis_service.analyze_shelf(
            image,
            filename,
            report,
            max_crops=max_crops,
            max_sku_crops=max_sku_crops,
            extract_sku=extract_sku,
        )
        jobs.update(
            job_id,
            status="complete",
            stage="complete",
            progress=100,
            message="Shelf report ready",
            result=result,
        )
    except Exception:
        logger.exception("Shelf analysis job %s failed", job_id)
        jobs.update(
            job_id,
            status="failed",
            stage="failed",
            message="We could not analyze this image. Please try again.",
            error="Shelf analysis failed. Check the service logs for details.",
        )


@app.get("/api/health")
def health() -> dict[str, Any]:
    with model_state_lock:
        models = {"status": model_state["status"], "details": dict(model_state["details"])}
    return {"status": "ok", "models": models}


@app.post("/api/analyses", response_model=JobAccepted, status_code=202)
async def create_analysis(
    image: UploadFile = File(...),
    max_crops: int = Form(default=60, ge=0, le=300),
    max_sku_crops: int = Form(default=5, ge=0, le=100),
    extract_sku: bool = Form(default=True),
) -> JobAccepted:
    if jobs.active_count() >= int(os.getenv("MAX_PENDING_SCANS", "3")):
        raise HTTPException(
            status_code=429,
            detail="Several scans are already running. Please try again shortly.",
        )
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Choose a shelf image to analyze.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large. Choose one under 15 MB.")
    try:
        candidate = Image.open(io.BytesIO(data))
        image_format = candidate.format
        if candidate.width * candidate.height > MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=413,
                detail="Image dimensions are too large. Choose a photo under 50 megapixels.",
            )
        candidate.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        raise HTTPException(status_code=415, detail="Use a JPG, PNG, BMP, or WebP image.")
    if image_format not in ALLOWED_FORMATS:
        raise HTTPException(status_code=415, detail="Use a JPG, PNG, BMP, or WebP image.")

    job_id = jobs.create()
    inference_executor.submit(
        _run_analysis,
        job_id,
        data,
        image.filename or "shelf-image",
        max_crops,
        max_sku_crops,
        extract_sku,
    )
    return JobAccepted(job_id=job_id, status="queued")


@app.get("/api/analyses/{job_id}", response_model=JobResponse)
def get_analysis(job_id: str) -> JobResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="That scan could not be found.")
    return JobResponse(**job)


@app.get("/api/scans")
def list_scans() -> dict[str, Any]:
    scans = db.get_scans_df()
    return {"scans": _public_records(scans), "stats": db.stats()}


@app.get("/api/scans/{scan_id}")
def get_scan(scan_id: int) -> dict[str, Any]:
    scans = db.get_scans_df()
    selected = scans[scans["id"] == scan_id] if not scans.empty else scans
    if selected.empty:
        raise HTTPException(status_code=404, detail="That saved scan could not be found.")
    return {
        "scan": _public_records(selected)[0],
        "items": _public_records(db.get_items_df(scan_id)),
    }


@app.get("/api/insights")
def insights() -> dict[str, Any]:
    items = db.get_items_df()
    scans = db.get_scans_df()
    feedback = db.get_feedback_df()
    counts = bi_engine.category_counts(items)
    subcategory_counts = bi_engine.subcategory_counts(items)
    categories = [
        {"category": str(category), "count": int(count)} for category, count in counts.items()
    ]
    subcategories = [
        {"subcategory": str(subcategory), "count": int(count)}
        for subcategory, count in subcategory_counts.items()
    ]
    feedback_summary = {
        "total": int(len(feedback)),
        "category_correct": 0,
        "category_incorrect": 0,
        "sku_correct": 0,
        "sku_incorrect": 0,
    }
    if not feedback.empty:
        grouped = feedback.groupby(["feedback_type", "verdict"]).size()
        for (feedback_type, verdict), count in grouped.items():
            feedback_summary[f"{feedback_type}_{verdict}"] = int(count)
    return {
        "summary": bi_engine.inventory_summary(items, scans),
        "categories": categories,
        "subcategories": subcategories,
        "scans": _frame_records(scans.sort_values("id")) if not scans.empty else [],
        "feedback": feedback_summary,
    }


@app.post("/api/feedback")
def save_feedback(payload: FeedbackRequest) -> dict[str, Any]:
    try:
        feedback_id = db.save_feedback(
            scan_id=payload.scan_id,
            crop_id=payload.crop_id,
            feedback_type=payload.feedback_type,
            verdict=payload.verdict,
            correction=payload.correction,
            note=payload.note,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="That detected product could not be found.")
    return {
        "id": feedback_id,
        "status": "saved",
        "message": "Feedback saved for future model evaluation.",
    }


@app.post("/api/questions", response_model=QuestionResponse)
def ask_inventory(payload: QuestionRequest) -> QuestionResponse:
    answer = bi_engine.answer(
        payload.question,
        db.get_items_df(),
        db.get_scans_df(),
        use_llm=os.getenv("BI_USE_LLM", "1") == "1",
    )
    table = _frame_records(answer.table) if answer.table is not None else None
    return QuestionResponse(text=answer.text, source=answer.source, table=table)


if WEB_DIST.exists():
    assets = WEB_DIST / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="web-assets")


@app.get("/{full_path:path}", include_in_schema=False)
def serve_web(full_path: str) -> FileResponse:
    if full_path == "api" or full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found.")
    candidate = (WEB_DIST / full_path).resolve()
    if candidate.is_file() and WEB_DIST.resolve() in candidate.parents:
        return FileResponse(candidate)
    index = WEB_DIST / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Web application has not been built.")
    return FileResponse(index)
