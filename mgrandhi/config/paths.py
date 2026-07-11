from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MGRANDHI_ROOT = Path(__file__).resolve().parents[1]

# Teammate repo asset layout — all at repo root (not in subdirs)
YOLO_WEIGHTS = REPO_ROOT / "models" / "yolo" / "best.pt"
FAISS_INDEX = REPO_ROOT / "swin_faiss_index.bin"
SWIN_MODEL_DIR = REPO_ROOT / "swin_model_assets"
SWIN_PROCESSOR_DIR = REPO_ROOT / "swin_processor_assets"
LABELS_CSV = REPO_ROOT / "train_product_category_58.csv"
INDEXED_IMAGE_PATHS_CSV = REPO_ROOT / "swin_faiss_indexed_image_paths.csv"

INVENTORY_DB = MGRANDHI_ROOT / "inventory.db"
