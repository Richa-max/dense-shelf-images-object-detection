rom pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MGRANDHI_ROOT = Path(__file__).resolve().parents[1]

# Adjust these paths to match the teammate repo's actual asset layout.
YOLO_WEIGHTS = REPO_ROOT / "best.pt"
FAISS_INDEX = REPO_ROOT / "assets" / "swin_faiss_index.bin"
SWIN_MODEL_DIR = REPO_ROOT / "assets" / "swin_model_assets"
SWIN_PROCESSOR_DIR = REPO_ROOT / "assets" / "swin_processor_assets"
LABELS_DIR = REPO_ROOT / "assets" / "labels"

INVENTORY_DB = MGRANDHI_ROOT / "inventory.db"
