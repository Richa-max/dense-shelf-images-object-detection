#!/usr/bin/env bash
# Start the FastAPI API and Vite development server with hot reload.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$APP_DIR/.." && pwd)"
cd "$ROOT"

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cleanup() {
  [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null || true
  [[ -n "${WEB_PID:-}" ]] && kill "$WEB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

uvicorn mgrandhi.backend.api:app --host 127.0.0.1 --port 8000 --workers 1 &
API_PID=$!
npm --prefix "$APP_DIR/web" run dev -- --host 127.0.0.1 &
WEB_PID=$!

echo "ShelfSight development UI: http://localhost:5173"
wait -n "$API_PID" "$WEB_PID"
