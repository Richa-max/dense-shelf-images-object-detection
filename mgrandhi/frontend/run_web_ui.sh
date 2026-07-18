#!/usr/bin/env bash
# Build the React application and run the single-process FastAPI inference service.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$APP_DIR/.." && pwd)"
cd "$ROOT"

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WEB_PORT="${WEB_PORT:-8000}"

if [[ "${SKIP_WEB_BUILD:-0}" != "1" ]]; then
  npm --prefix "$APP_DIR/web" run build
fi

exec uvicorn mgrandhi.backend.api:app \
  --host 0.0.0.0 \
  --port "$WEB_PORT" \
  --workers 1
