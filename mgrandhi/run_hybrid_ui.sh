#!/usr/bin/env bash
# Run the hybrid demo UI:
# - Gradio on $GRADIO_PORT for fast upload/analyze/result-table flow
# - Streamlit on $STREAMLIT_PORT for analytics, BI, and inventory history
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$APP_DIR/.." && pwd)"
cd "$ROOT"

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export GRADIO_PORT="${GRADIO_PORT:-7860}"
export STREAMLIT_PORT="${STREAMLIT_PORT:-8502}"
export GRADIO_URL="${GRADIO_URL:-http://localhost:${GRADIO_PORT}}"

cleanup() {
  if [[ -n "${GRADIO_PID:-}" ]]; then kill "$GRADIO_PID" 2>/dev/null || true; fi
  if [[ -n "${STREAMLIT_PID:-}" ]]; then kill "$STREAMLIT_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

python "$APP_DIR/gradio_app.py" &
GRADIO_PID=$!

streamlit run "$APP_DIR/app.py" \
  --server.port "$STREAMLIT_PORT" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false &
STREAMLIT_PID=$!

cat <<EOF
Hybrid UI started.

Gradio fast upload : $GRADIO_URL
Streamlit dashboard: http://localhost:$STREAMLIT_PORT

Press Ctrl-C to stop both servers.
EOF

# If either UI exits, stop the other one and let systemd restart the service.
wait -n "$GRADIO_PID" "$STREAMLIT_PID"
EXIT_CODE=$?
cleanup
exit "$EXIT_CODE"
