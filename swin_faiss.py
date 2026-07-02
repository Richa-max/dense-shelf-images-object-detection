import os
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

try:
    import faiss
except ImportError:
    faiss = None


def _infer_label_from_path(path: str) -> str:
    norm = path.replace("\\", "/")
    parts = [p for p in norm.split("/") if p]
    if not parts:
        return "unknown"
    candidate = parts[-1]
    if candidate.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".npy", ".bin", ".txt")):
        candidate = os.path.splitext(candidate)[0]
    candidate = candidate.replace("_", " ").replace("-", " ").strip()
    if candidate:
        return candidate
    return "unknown"


class SwinFaissClassifier:
    def __init__(
        self,
        model_dir: str = "swin_model_assets",
        processor_dir: str = "swin_processor_assets",
        index_path: str = "swin_faiss_index.bin",
        image_paths_path: str = "swin_faiss_indexed_image_paths.txt",
        indexed_image_paths_npy: str = "swin_faiss_indexed_image_paths.npy",
    ):
        self.model_dir = model_dir
        self.processor_dir = processor_dir
        self.index_path = index_path
        self.image_paths_path = image_paths_path
        self.indexed_image_paths_npy = indexed_image_paths_npy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = None
        self.model = None
        self.index = None
        self.image_paths = []
        self.is_ready_flag = False
        self._load_resources()

    def is_ready(self) -> bool:
        return self.is_ready_flag

    def _load_resources(self):
        # Provide clear debugging/logging for why resources may not load.
        if faiss is None:
            print("[swin_faiss] faiss not installed or not importable (faiss=None)")
            self.is_ready_flag = False
            return

        try:
            missing = []
            if not os.path.exists(self.model_dir):
                missing.append(self.model_dir)
            if not os.path.exists(self.processor_dir):
                missing.append(self.processor_dir)
            if not os.path.exists(self.index_path):
                missing.append(self.index_path)

            if missing:
                print(f"[swin_faiss] missing resource paths: {missing}")
                self.is_ready_flag = False
                return

            print(f"[swin_faiss] loading processor from: {self.processor_dir}")
            self.processor = AutoImageProcessor.from_pretrained(self.processor_dir)
            print(f"[swin_faiss] loading model from: {self.model_dir}")
            self.model = AutoModel.from_pretrained(self.model_dir)
            self.model.eval()
            self.model.to(self.device)

            print(f"[swin_faiss] reading FAISS index from: {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            self.image_paths = self._load_paths()
            if not self.image_paths:
                print("[swin_faiss] loaded FAISS index but no image paths were found")
            self.is_ready_flag = bool(self.index is not None and self.image_paths)
            print(f"[swin_faiss] is_ready set to {self.is_ready_flag}")
        except Exception as exc:
            print(f"[swin_faiss] failed to load resources: {exc}")
            self.is_ready_flag = False

    def _load_paths(self):
        if os.path.exists(self.indexed_image_paths_npy):
            data = np.load(self.indexed_image_paths_npy, allow_pickle=True)
            return [
                {
                    "path": str(x),
                    "label": _infer_label_from_path(str(x)),
                    "subcategory": None,
                }
                for x in data.tolist()
                if x is not None
            ]
        if os.path.exists(self.image_paths_path):
            entries = []
            with open(self.image_paths_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    path = line.strip()
                    if not path:
                        continue
                    entries.append(
                        {
                            "path": path,
                            "label": _infer_label_from_path(path),
                            "subcategory": None,
                        }
                    )
            return entries
        csv_path = os.path.splitext(self.image_paths_path)[0] + ".csv"
        if os.path.exists(csv_path):
            entries = []
            try:
                import csv

                with open(csv_path, "r", encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        if not row:
                            continue
                        path = str(row.get("image_path", "")).strip()
                        if not path:
                            continue
                        label = str(row.get("predicted_category") or row.get("full_label") or "").strip()
                        if not label:
                            label = _infer_label_from_path(path)
                        subcategory = str(row.get("predicted_subcategory") or "").strip() or None
                        entries.append(
                            {
                                "path": path,
                                "label": label,
                                "subcategory": subcategory,
                            }
                        )
            except Exception:
                with open(csv_path, "r", encoding="utf-8") as fh:
                    lines = [line.strip() for line in fh if line.strip()]
                    if not lines:
                        return []
                    headers = [h.strip() for h in lines[0].split(",")]
                    for line in lines[1:]:
                        values = [v.strip() for v in line.split(",")]
                        row = dict(zip(headers, values))
                        path = str(row.get("image_path", "")).strip()
                        if not path:
                            continue
                        label = str(row.get("predicted_category") or row.get("full_label") or "").strip()
                        if not label:
                            label = _infer_label_from_path(path)
                        subcategory = str(row.get("predicted_subcategory") or "").strip() or None
                        entries.append(
                            {
                                "path": path,
                                "label": label,
                                "subcategory": subcategory,
                            }
                        )
            return entries
        return []

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            hidden = getattr(outputs, "pooler_output", None)
            if hidden is None:
                hidden = outputs.last_hidden_state.mean(dim=1)
        emb = hidden.detach().cpu().numpy()
        norm = np.linalg.norm(emb, axis=-1, keepdims=True)
        emb = emb / np.clip(norm, a_min=1e-9, a_max=None)
        return emb.astype("float32")

    def query(self, image: Image.Image, top_k: int = 10):
        if not self.is_ready():
            raise RuntimeError("Swin FAISS classifier is not ready")

        emb = self._embed_image(image)
        distances, indices = self.index.search(emb, top_k)
        scores = distances[0].tolist()
        ids = indices[0].tolist()

        neighbors = []
        for idx, score in zip(ids, scores):
            if idx < 0 or idx >= len(self.image_paths):
                continue
            data = self.image_paths[idx]
            path = data.get("path") if isinstance(data, dict) else str(data)
            label = data.get("label") if isinstance(data, dict) else _infer_label_from_path(path)
            subcategory = data.get("subcategory") if isinstance(data, dict) else None
            neighbors.append(
                {
                    "path": path,
                    "label": label,
                    "subcategory": subcategory,
                    "score": float(score),
                }
            )

        return neighbors

    def classify(self, image: Image.Image, top_k: int = 10, top_labels: int = 5) -> dict:
        neighbors = self.query(image, top_k=top_k)
        if not neighbors:
            return {
                "label": "unknown",
                "score": 0.0,
                "candidate_labels": [],
                "neighbors": [],
                "confidence": "low",
            }

        label_counts = Counter([n["label"] for n in neighbors if n["label"] != "unknown"])
        if not label_counts:
            candidate_labels = []
            best_label = "unknown"
        else:
            candidate_labels = [label for label, _ in label_counts.most_common(top_labels)]
            best_label = candidate_labels[0]

        best_score = float(max((n["score"] for n in neighbors), default=0.0))
        confidence = "low"
        if best_score >= 0.75:
            confidence = "high"
        elif best_score >= 0.45:
            confidence = "medium"

        return {
            "label": best_label,
            "score": best_score,
            "candidate_labels": candidate_labels,
            "neighbors": neighbors,
            "confidence": confidence,
        }

    def save_unknown_crop(self, crop: Image.Image, crop_id: int, output_dir: str = "unknown_crops") -> str:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"unknown_crop_{crop_id}_{int(time.time())}.png"
        path = os.path.join(output_dir, filename)
        crop.save(path)
        metadata_path = os.path.join(output_dir, "unknown_crops.jsonl")
        entry = {
            "crop_id": crop_id,
            "crop_path": path,
            "timestamp": int(time.time()),
        }
        with open(metadata_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return path


def load_swin_faiss_classifier() -> "SwinFaissClassifier":
    return SwinFaissClassifier()
