#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[startup] $*"
}

wait_for_ollama() {
  local attempts="${OLLAMA_STARTUP_WAIT_ATTEMPTS:-60}"
  local sleep_seconds="${OLLAMA_STARTUP_WAIT_SECONDS:-2}"

  for _ in $(seq 1 "$attempts"); do
    if ollama list >/dev/null 2>&1; then
      return 0
    fi
    sleep "$sleep_seconds"
  done

  return 1
}

if [[ "${OLLAMA_AUTOSTART:-1}" == "1" ]]; then
  if command -v ollama >/dev/null 2>&1; then
    export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
    export LLAMA_MODEL="${LLAMA_MODEL:-llama3.1:8b}"
    export RETAIL_REASONER_MODEL="${RETAIL_REASONER_MODEL:-$LLAMA_MODEL}"

    log "starting Ollama on ${OLLAMA_HOST}"
    ollama serve &
    OLLAMA_PID="$!"

    if wait_for_ollama; then
      log "pulling Llama model ${LLAMA_MODEL}"
      ollama pull "${LLAMA_MODEL}"
    else
      log "Ollama did not become ready; app will use visual fallback if Llama is unavailable"
    fi
  else
    log "ollama command not found; app will use visual fallback if Llama is unavailable"
  fi
fi

if [[ "${PRELOAD_PADDLE_OCR:-1}" == "1" ]]; then
  log "preloading PaddleOCR models"
  python - <<'PY'
from PIL import Image
from retail_product_resolver import extract_paddle_ocr_text

image = Image.new("RGB", (96, 32), "white")
result = extract_paddle_ocr_text(image)
print("[startup] PaddleOCR preload:", result.get("backend"), "available=", result.get("available"), "error=", result.get("error"))
PY
fi

exec "$@"
