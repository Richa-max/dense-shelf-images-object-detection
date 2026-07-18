# Mahesh — ShelfSight UI + SKU OCR

`mgrandhi/` contains Mahesh's ShelfSight React/FastAPI operator UI, SKU/OCR extraction, inventory
database, and benchmarks. It reuses the dense repository's root-level YOLO, SWIN, FAISS, label
assets, and the SQLite inventory database. Do not copy large LFS/model files into `mgrandhi/`.

All commands in this guide start from the dense repository root:

```bash
cd /path/to/dense-shelf-images-object-detection
```

No setup or launch step requires `cd mgrandhi`. If you enter `mgrandhi/` only to inspect files,
return to dense root before running Python, Uvicorn, npm, or a launch script.

## Architecture

```text
browser
  ├─ development: Vite :5173 ──proxy /api──> FastAPI :8000
  └─ production:  FastAPI :8000 serves both /api and built React files

FastAPI (mgrandhi.backend.api)
  ├─ YOLO detector
  ├─ SWIN encoder + FAISS retrieval
  ├─ optional SKU/OCR backend
  └─ SQLite inventory + review evidence
```

Asset paths are resolved by `mgrandhi/config/paths.py`. Defaults point to the dense repo root:

```text
models/yolo/best.pt
swin_faiss_index.bin
swin_faiss_indexed_image_paths.csv
swin_model_assets/
swin_model_assets/model.safetensors
swin_processor_assets/
swin_processor_assets/preprocessor_config.json
train_product_category_58.csv
```

All launch scripts change to dense root before importing `mgrandhi.*`; this folder is not installed
as an editable Python package.

## Prerequisites

- Git and Git LFS
- Python 3.11 (the ML stack requires Python 3.10 or 3.11)
- Node.js 22 LTS and npm (Vite 8 requires a current Node release)
- Bash 4.3 or newer for the development script's multi-process `wait -n`
- Enough disk and RAM for the approximately 2 GiB FAISS index and model files

```bash
git --version
git lfs version
python3.11 --version
node --version
npm --version
bash --version
```

## Fetch Git LFS Assets

```bash
git lfs install
git lfs pull
git lfs checkout
```

Verify the required files:

```bash
test -s models/yolo/best.pt
test -s swin_faiss_index.bin
test -s swin_faiss_indexed_image_paths.csv
test -s swin_model_assets/model.safetensors
test -s swin_processor_assets/preprocessor_config.json
test -s train_product_category_58.csv
```

If a `test` command fails, restore the asset at its expected dense root path with Git LFS,
or set the corresponding environment override (see below). Do not copy an asset into `mgrandhi/`.

## Python Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r mgrandhi/requirements.txt
```

On subsequent runs:

```bash
cd /path/to/dense-shelf-images-object-detection
source .venv/bin/activate
```

Do not run `pip install -e mgrandhi`; `mgrandhi/` has no standalone `pyproject.toml`.

## Frontend Installation and Checks

```bash
npm --prefix mgrandhi/frontend/web ci
npm --prefix mgrandhi/frontend/web run typecheck
npm --prefix mgrandhi/frontend/web run lint
npm --prefix mgrandhi/frontend/web test
npm --prefix mgrandhi/frontend/web run build
```

`npm ci` uses the committed lockfile. If its configured package registry is inaccessible, use the
registry approved for your environment; do not commit credentials or an incidental lockfile
rewrite.

## Development Mode: FastAPI plus Vite

With the Python environment active:

```bash
SKU_BACKEND=dry-run \
bash mgrandhi/frontend/run_web_dev.sh
```

Open <http://localhost:5173>. The script starts:

- FastAPI at `http://127.0.0.1:8000`
- Vite at `http://127.0.0.1:5173`
- a Vite `/api` proxy to FastAPI

`SKU_BACKEND=dry-run` avoids external SKU/OCR credentials while leaving YOLO and SWIN/FAISS
category inference enabled. The script stops both processes if either exits.

## Production-Style Single-Origin Mode

```bash
SKU_BACKEND=dry-run \
bash mgrandhi/frontend/run_web_ui.sh
```

The script builds `mgrandhi/frontend/web/dist/`, then FastAPI serves the React application and API
from <http://localhost:8000>.

To reuse an existing successful frontend build or change the port:

```bash
SKIP_WEB_BUILD=1 \
WEB_PORT=8080 \
SKU_BACKEND=dry-run \
bash mgrandhi/frontend/run_web_ui.sh
```

Open <http://localhost:8080>. Use `SKIP_WEB_BUILD=1` only when
`mgrandhi/frontend/web/dist/index.html` already exists and matches the current frontend.

## Health Checks

With either mode running:

```bash
curl --fail --silent http://127.0.0.1:8000/api/health
```

Typical model states:

- `loading`: background model preload is still running.
- `ready`: detector and classifier loaded successfully.
- `degraded`: preload failed; inspect the server log and asset paths.
- `lazy`: the server was started with `PRELOAD_MODELS=0`.

An API/import-only check that does not load model assets:

```bash
PRELOAD_MODELS=0 python -c \
  "from mgrandhi.backend.api import app; print(app.title)"
```

For a lightweight API health server:

```bash
PRELOAD_MODELS=0 SKU_BACKEND=dry-run \
uvicorn mgrandhi.backend.api:app --host 127.0.0.1 --port 8000
```

## Environment Overrides

Path defaults come from `mgrandhi/config/paths.py`. Override any with environment variables:

```bash
export YOLO_WEIGHTS=/absolute/path/to/best.pt
export FAISS_INDEX=/absolute/path/to/swin_faiss_index.bin
export INDEXED_IMAGE_PATHS_CSV=/absolute/path/to/swin_faiss_indexed_image_paths.csv
export SWIN_MODEL_DIR=/absolute/path/to/swin_model_assets
export SWIN_PROCESSOR_DIR=/absolute/path/to/swin_processor_assets
export LABELS_CSV=/absolute/path/to/train_product_category_58.csv
```

Runtime storage can also be relocated:

```bash
export INVENTORY_DB=/absolute/writable/path/inventory.db
export FEEDBACK_ASSET_DIR=/absolute/writable/path/review_evidence
export SHELFSIGHT_WEB_DIST=/absolute/path/to/frontend/dist
```

Common service controls:

```bash
export PRELOAD_MODELS=1
export YOLO_CONFIDENCE=0.25
export MAX_CLASSIFICATION_CROPS=60
export MAX_SKU_CROPS=5
export SKU_EXTRACT_DEFAULT=1
export MAX_PENDING_SCANS=3
export MAX_UPLOAD_BYTES=15728640
export MAX_IMAGE_PIXELS=50000000
export DEV_CORS_ORIGINS=http://localhost:5173
```

Keep one Uvicorn worker: model objects are process-local and each worker would load the large
assets separately.

## SKU/OCR Backend Options

Do not put secrets in shell history, tracked files, screenshots, or issue comments. Prefer your
environment's secret manager or an untracked environment file loaded by your process supervisor.

### No external service (use for testing)

```bash
export SKU_BACKEND=dry-run
export SKU_MODEL=dry-run
```

To prevent server-side SKU preloading and disable extraction for API callers that omit the flag:

```bash
export SKU_EXTRACT_DEFAULT=0
```

### Vertex Gemini

```bash
export SKU_BACKEND=gemini
export PROJECT_ID=your-authorized-gcp-project
export REGION=us-central1
export VERTEX_MODEL=gemini-2.5-flash
```

### OpenAI-compatible Qwen/PaliGemma endpoint

```bash
export SKU_BACKEND=openai-compatible
export VLM_ENDPOINT_URL=https://your-authorized-host.example/v1
export VLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
```

### Vertex Model Garden endpoint

```bash
export SKU_BACKEND=vertex-model-garden
export PROJECT_ID=your-authorized-gcp-project
export REGION=us-central1
export VERTEX_MODEL_GARDEN_ENDPOINT_ID=your-deployed-endpoint-id
export VERTEX_MODEL_GARDEN_MODEL=google/paligemma@paligemma-mix-448-float16
```

For all remote backends:

```bash
export SKU_TIMEOUT_SECONDS=180
```

## Persistence

Default writable paths are:

```text
mgrandhi/inventory.db
mgrandhi/data/review_evidence/scan_<id>/source.jpg
mgrandhi/data/review_evidence/scan_<id>/crop_<id>.jpg
```

To use a dedicated persistent location:

```bash
mkdir -p "$HOME/.local/share/shelfsight/review_evidence"
export INVENTORY_DB="$HOME/.local/share/shelfsight/inventory.db"
export FEEDBACK_ASSET_DIR="$HOME/.local/share/shelfsight/review_evidence"
```

Startup performs additive column migrations. Back up persistent data before manual changes.

## Troubleshooting

### Import error for `mgrandhi`

Run from dense root:

```bash
cd /path/to/dense-shelf-images-object-detection
source .venv/bin/activate
PRELOAD_MODELS=0 python -c "import mgrandhi.backend.api"
```

### LFS pointer or missing model/index

```bash
git lfs pull
git lfs checkout
ls -lh models/yolo/best.pt swin_faiss_index.bin \
  swin_model_assets/model.safetensors
```

A tiny text file beginning with `version https://git-lfs.github.com/spec/` is a pointer, not the
model content.

### Health is `degraded`

Read the FastAPI log first. Verify every expected path:

```bash
python - <<'PY'
from mgrandhi.config import paths
for name in (
    "YOLO_WEIGHTS", "FAISS_INDEX", "INDEXED_IMAGE_PATHS_CSV",
    "SWIN_MODEL_DIR", "SWIN_PROCESSOR_DIR", "LABELS_CSV",
):
    path = getattr(paths, name)
    print(f"{name}: {path} exists={path.exists()}")
PY
```

### macOS OpenMP duplicate-library abort

The launch scripts already default these values; use the provided scripts:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

### SKU fields are empty

- With `dry-run`, empty SKU fields are expected.
- With Gemini, verify `PROJECT_ID`, Google authentication, region, and model access.
- With OpenAI-compatible, verify the `/v1` endpoint, model name, and network access.
- With Model Garden, verify endpoint ID, region, project, and IAM.

### Vite loads but API calls fail

```bash
curl --fail http://127.0.0.1:8000/api/health
```

If the browser origin differs from the default, set `DEV_CORS_ORIGINS` to a comma-separated list.

### Production root returns 404

Build the frontend and restart:

```bash
npm --prefix mgrandhi/frontend/web run build
bash mgrandhi/frontend/run_web_ui.sh
```

### Port already in use

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -nP -iTCP:5173 -sTCP:LISTEN
```

### Development script exits at `wait -n`

The script requires Bash 4.3+. macOS ships an older system Bash; use an approved current Bash:

```bash
/opt/homebrew/bin/bash mgrandhi/frontend/run_web_dev.sh
```

## Shutdown

Press Ctrl-C in the launch script's terminal. Check for remaining processes:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -nP -iTCP:5173 -sTCP:LISTEN
```

## Cost Reminder

Managed Model Garden endpoints and GPU VMs cost money while deployed. For demo/testing:

1. Deploy PaliGemma only when needed.
2. Test with a small crop limit first.
3. Undeploy/delete the endpoint after the session.
4. Keep self-hosted GPU VMs deleted unless actively benchmarking.
