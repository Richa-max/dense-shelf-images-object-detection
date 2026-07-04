#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[startup] $*"
}

if [[ "${PRELOAD_EASYOCR:-1}" == "1" || "${PRELOAD_LLAMA:-1}" == "1" ]]; then
  log "preloading retail OCR/reasoner models"
  python - <<'PY'
from retail_product_resolver import preload_retail_models

status = preload_retail_models()
easyocr_status = status.get("easyocr") or {}
llama_status = status.get("llama") or {}

print(
    "[startup] EasyOCR preload:",
    "available=", easyocr_status.get("available"),
    "error=", easyocr_status.get("error"),
)
print(
    "[startup] Llama preload:",
    "loaded=", llama_status.get("loaded"),
    "model=", llama_status.get("model"),
    "error=", llama_status.get("error"),
)
PY
fi

exec "$@"
