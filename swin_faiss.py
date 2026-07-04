import os
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
#test

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
    lowered = candidate.lower()
    if lowered.startswith("train ") or lowered.startswith("train_") or " crop " in lowered or lowered.endswith(" crop"):
        return "unknown"
    if candidate:
        return candidate
    return "unknown"


def _metadata_from_row(row: dict) -> dict:
    metadata = {str(k).strip(): v for k, v in row.items() if k is not None}
    image_path = str(metadata.get("image_path", "")).strip()
    label = str(metadata.get("predicted_category") or metadata.get("category") or metadata.get("full_label") or "").strip()
    if not label:
        label = _infer_label_from_path(image_path)
    subcategory = str(metadata.get("predicted_subcategory") or metadata.get("subcategory") or "").strip() or None
    product_name = (
        str(
            metadata.get("product_name")
            or metadata.get("product")
            or metadata.get("name")
            or metadata.get("title")
            or metadata.get("sku_name")
            or metadata.get("full_label_50")
            or metadata.get("full_label")
            or ""
        ).strip()
        or None
    )
    metadata.update(
        {
            "path": image_path,
            "label": label,
            "subcategory": subcategory,
            "product_name": product_name,
        }
    )
    return metadata


class SwinFaissClassifier:
    def __init__(
        self,
        model_dir: str = "swin_model_assets",
        processor_dir: str = "swin_processor_assets",
        index_path: str = "swin_faiss_index.bin",
        image_paths_path: str = "swin_faiss_indexed_image_paths.csv",
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
        def parse_csv_paths(path_to_csv):
            rows = []
            try:
                import csv

                with open(path_to_csv, "r", encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    fieldnames = [f.strip() for f in (reader.fieldnames or [])]
                    for row in reader:
                        if not row:
                            continue
                        row = _metadata_from_row(row)
                        image_path = str(row.get("path", "")).strip()
                        if not image_path:
                            continue
                        rows.append(row)
            except Exception:
                with open(path_to_csv, "r", encoding="utf-8") as fh:
                    lines = [line.strip() for line in fh if line.strip()]
                    if not lines:
                        return []
                    headers = [h.strip() for h in lines[0].split(",")]
                    for line in lines[1:]:
                        values = [v.strip() for v in line.split(",")]
                        row = dict(zip(headers, values))
                        image_path = str(row.get("image_path", "")).strip()
                        if not image_path:
                            continue
                        rows.append(_metadata_from_row(row))
            return rows

        def normalize_path(path):
            return path.replace("\\", "/").strip()

        def merge_metadata(existing_entries):
            if not existing_entries:
                return existing_entries
            metadata = {}
            excluded_filenames = {os.path.basename(self.image_paths_path)}
            try:
                from glob import glob

                for cand in glob("*.csv"):
                    if cand in excluded_filenames:
                        continue
                    if cand.lower().endswith(".csv"):
                        rows = parse_csv_paths(cand)
                        for row in rows:
                            key = normalize_path(row["path"])
                            metadata[key] = row
                            metadata[os.path.basename(key)] = row
            except Exception:
                pass

            if not metadata:
                return existing_entries

            for entry in existing_entries:
                norm = normalize_path(entry["path"])
                if norm in metadata:
                    entry.update(metadata[norm])
                else:
                    basename = os.path.basename(norm)
                    if basename in metadata:
                        entry.update(metadata[basename])
            return existing_entries

        if os.path.exists(self.indexed_image_paths_npy):
            data = np.load(self.indexed_image_paths_npy, allow_pickle=True)
            entries = [
                {
                    "path": str(x),
                    "label": _infer_label_from_path(str(x)),
                    "subcategory": None,
                    "product_name": None,
                }
                for x in data.tolist()
                if x is not None
            ]
            return merge_metadata(entries)

        if os.path.exists(self.image_paths_path):
            entries = []
            if self.image_paths_path.lower().endswith(".csv"):
                try:
                    import csv
                    with open(self.image_paths_path, "r", encoding="utf-8", newline="") as fh:
                        reader = csv.DictReader(fh)
                        if reader.fieldnames and "image_path" in [f.strip() for f in reader.fieldnames]:
                            for row in reader:
                                if not row:
                                    continue
                                image_path = str(row.get("image_path", "")).strip()
                                if not image_path:
                                    continue
                                entries.append(_metadata_from_row(row))
                        else:
                            fh.seek(0)
                            for line in fh:
                                path = line.strip()
                                if not path or path.lower() == "image_path":
                                    continue
                                entries.append({"path": path, "label": _infer_label_from_path(path), "subcategory": None, "product_name": None})
                except Exception:
                    with open(self.image_paths_path, "r", encoding="utf-8") as fh:
                        for line in fh:
                            path = line.strip()
                            if not path or path.lower() == "image_path":
                                continue
                            entries.append({"path": path, "label": _infer_label_from_path(path), "subcategory": None, "product_name": None})
            else:
                with open(self.image_paths_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        path = line.strip()
                        if not path:
                            continue
                        entries.append({"path": path, "label": _infer_label_from_path(path), "subcategory": None})
            if entries:
                return merge_metadata(entries)

        csv_path = os.path.splitext(self.image_paths_path)[0] + ".csv"
        if os.path.exists(csv_path):
            entries = parse_csv_paths(csv_path)
            if entries:
                return merge_metadata(entries)

        # If no explicit indexed paths files were found, try to discover any
        # labeled CSVs in the workspace (e.g. train_product_category_58.csv)
        # that contain an `image_path` column and optional `predicted_category`
        # and `predicted_subcategory` columns.
        try:
            from glob import glob

            candidates = glob("*.csv")
            for cand in candidates:
                if cand == os.path.basename(csv_path):
                    # already tried this file
                    continue
                try:
                    rows = parse_csv_paths(cand)
                    if rows:
                        print(f"[swin_faiss] discovered labeled CSV: {cand} (loaded {len(rows)} entries)")
                        return rows
                except Exception:
                    continue
        except Exception:
            pass

        return []

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        # Notebook-aligned Swin embedding extraction:
        # - using PIL image input
        # - using AutoImageProcessor and AutoModel from transformers
        # - calling model.eval() and torch.no_grad()
        # - using mean pooling over last_hidden_state, not class logits
        # - keeping embeddings raw, matching FAISS index built from non-normalized vectors
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state.mean(dim=1)
        emb = hidden.cpu().numpy()
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
            product_name = data.get("product_name") if isinstance(data, dict) else None
            neighbor = {
                "path": path,
                "label": label,
                "subcategory": subcategory,
                "product_name": product_name,
                "score": float(score),
            }
            if isinstance(data, dict):
                for key in ["brand", "barcode", "sku", "full_label", "full_label_50"]:
                    if key in data:
                        neighbor[key] = data.get(key)
            neighbors.append(neighbor)

        return neighbors

    def _distance_to_similarity(self, distance: float) -> float:
        # Convert FAISS distance output to a similarity score in [0, 1].
        # For normalized embeddings, L2 distance range is [0, 2], and inner-product
        # similarity is in [-1, 1]. We normalize both into a positive score range.
        if distance is None:
            return 0.0
        if distance <= 1.0:
            return max(0.0, 1.0 - distance / 2.0)
        return max(0.0, 1.0 / (1.0 + distance))

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

        valid_neighbors = [n for n in neighbors if n.get("label") and n.get("label") != "unknown"]
        if not valid_neighbors:
            return {
                "label": "unknown",
                "score": 0.0,
                "candidate_labels": [],
                "neighbors": neighbors,
                "confidence": "low",
            }

        best_neighbor = valid_neighbors[0]
        best_label = best_neighbor["label"]
        best_subcategory = best_neighbor.get("subcategory")
        candidate_labels = [n["label"] for n in valid_neighbors[:top_labels]]

        best_score = float(best_neighbor.get("score", 0.0))
        return {
            "label": best_label,
            "predicted_category": best_label,
            "score": best_score,
            "candidate_labels": candidate_labels,
            "neighbors": neighbors,
            "confidence": "high" if best_label != "unknown" else "low",
            "best_subcategory": best_subcategory,
            "predicted_subcategory": best_subcategory,
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
