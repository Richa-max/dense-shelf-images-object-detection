#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[startup] $*"
}

start_ollama() {
  if [[ "${OLLAMA_AUTOSTART:-1}" != "1" ]]; then
    log "ollama autostart disabled"
    return 0
  fi

  if ! command -v ollama >/dev/null 2>&1; then
    log "ollama binary not found"
    return 0
  fi

  if pgrep -f "ollama serve" >/dev/null 2>&1; then
    log "ollama already running"
    return 0
  fi

  export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
  export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/.ollama/models}"
  mkdir -p "$OLLAMA_MODELS"

  log "starting ollama server"
  nohup ollama serve > /tmp/ollama.log 2>&1 &

  for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
      log "ollama server is ready"
      return 0
    fi
    sleep 2
  done

  log "ollama server did not become ready"
  tail -n 50 /tmp/ollama.log || true
  return 1
}

pull_ollama_model() {
  local model="${OLLAMA_MODEL:-}"
  if [[ -z "$model" || "$model" == "disabled" ]]; then
    return 0
  fi

  if ! command -v ollama >/dev/null 2>&1; then
    return 0
  fi

  if ollama list 2>/dev/null | grep -q "^${model}"; then
    log "ollama model already available: $model"
    return 0
  fi

  log "pulling ollama model: $model"
  ollama pull "$model"
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

start_ollama
pull_ollama_model

exec "$@"
