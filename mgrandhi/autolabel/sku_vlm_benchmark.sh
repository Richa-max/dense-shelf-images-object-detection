#!/bin/bash
# VM startup script — benchmarks SKU/OCR extraction from crop images using autolabel/sku_vlm.py.
#
# This runner does not deploy the open models itself. It assumes one of:
#   - --backend gemini: Vertex Gemini reference endpoint (not open-source).
#   - --backend openai-compatible: a vLLM / Vertex Model Garden endpoint for Qwen-VL,
#     PaliGemma, or Gemma 3 exposed at VLM_ENDPOINT_URL.
#   - --backend dry-run: local parser/harness validation.
#
# Placeholders are filled by launch_sku_vlm_benchmark.sh.
set -uo pipefail

BUCKET="__BUCKET__"
REPO_TARBALL="__REPO_TARBALL__"
SPLIT="__SPLIT__"
SAMPLE_LIMIT="__SAMPLE_LIMIT__"
BACKEND="__BACKEND__"
MODEL="__MODEL__"
ENDPOINT="__ENDPOINT__"
AUTH_MODE="__AUTH_MODE__"
RUN_NAME="__RUN_NAME__"

RUN_DIR=/opt/runs
LOG="$RUN_DIR/sku_vlm_benchmark.log"
OUT_DIR="$RUN_DIR/out"
CROPS_DIR="$RUN_DIR/crops/$SPLIT"
OUT_CSV="$OUT_DIR/${RUN_NAME}.csv"
mkdir -p "$RUN_DIR" "$OUT_DIR" "$CROPS_DIR"
: > "$LOG"

cleanup() {
  rc=$?
  echo "=== cleanup (exit code $rc): syncing benchmark outputs to GCS ==="
  gsutil -m rsync -r "$OUT_DIR" "$BUCKET/sku_vlm_benchmarks/$RUN_NAME/" \
    || echo "WARN: benchmark output sync failed"
  gsutil cp "$LOG" "$BUCKET/sku_vlm_benchmarks/$RUN_NAME/${RUN_NAME}.log" || true

  NAME=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/name)
  ZONE_PATH=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone)
  ZONE=$(echo "$ZONE_PATH" | awk -F/ '{print $NF}')
  PROJECT=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/project/project-id)
  TOKEN=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')

  apt-get install -y at >/dev/null 2>&1 || true
  echo "shutdown -h now" | at now + 5 minutes 2>/dev/null || (sleep 300 && shutdown -h now) &

  echo "=== deleting self via gcloud: $NAME in $ZONE ==="
  gcloud compute instances delete "$NAME" --project "$PROJECT" --zone "$ZONE" --quiet \
    || echo "WARN: gcloud delete failed"
  sleep 30
  echo "=== fallback: deleting self via REST API ==="
  curl -sf -X DELETE -H "Authorization: Bearer $TOKEN" \
    "https://compute.googleapis.com/compute/v1/projects/$PROJECT/zones/$ZONE/instances/$NAME" \
    || echo "WARN: REST delete failed"
}
trap cleanup EXIT

echo "=== SKU VLM benchmark STARTED ==="
echo "run=$RUN_NAME backend=$BACKEND model=$MODEL split=$SPLIT limit=$SAMPLE_LIMIT endpoint=${ENDPOINT:-none}"
date

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=10

PROJECT=$(curl -sf -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/project/project-id)
ZONE_PATH=$(curl -sf -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/zone)
REGION=$(echo "$ZONE_PATH" | awk -F/ '{print $NF}' | sed 's/-[a-z]$//')
export PROJECT_ID="$PROJECT"
export REGION="$REGION"

echo "installing python deps..."
pip install --upgrade "google-genai>=0.3" >> "$LOG" 2>&1

echo "downloading benchmark code..."
cd "$RUN_DIR"
gsutil -q cp "$REPO_TARBALL" code.tar.gz >> "$LOG" 2>&1
tar -xzf code.tar.gz >> "$LOG" 2>&1

TARBALL="$BUCKET/crops/${SPLIT}_crops.tar"
if gsutil -q stat "$TARBALL" 2>/dev/null; then
  echo "downloading crops tarball $TARBALL ..."
  gsutil -q cp "$TARBALL" "$RUN_DIR/crops.tar" >> "$LOG" 2>&1
  tar -xf "$RUN_DIR/crops.tar" -C "$RUN_DIR/crops" >> "$LOG" 2>&1 && rm -f "$RUN_DIR/crops.tar"
else
  echo "no tarball; rsyncing crops from $BUCKET/crops/$SPLIT ..."
  gsutil -m rsync -r "$BUCKET/crops/$SPLIT" "$CROPS_DIR" >> "$LOG" 2>&1
fi
N_CROPS=$(find "$CROPS_DIR" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | wc -l)
echo "$SPLIT crops on disk: $N_CROPS" | tee -a "$LOG"

if [[ "$BACKEND" == "openai-compatible" ]]; then
  export VLM_ENDPOINT_URL="$ENDPOINT"
  if [[ "$AUTH_MODE" == "metadata" ]]; then
    export VLM_API_KEY
    VLM_API_KEY=$(curl -sf -H "Metadata-Flavor: Google" \
      http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
      | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
  fi
fi

LIMIT_FLAG=""
if [[ "$SAMPLE_LIMIT" != "0" ]]; then LIMIT_FLAG="--limit $SAMPLE_LIMIT"; fi
MODEL_FLAG=""
if [[ -n "$MODEL" ]]; then MODEL_FLAG="--model $MODEL"; fi
ENDPOINT_FLAG=""
if [[ "$BACKEND" == "openai-compatible" && -n "$ENDPOINT" ]]; then
  ENDPOINT_FLAG="--endpoint $ENDPOINT"
fi

echo "=== running SKU VLM benchmark (output -> $LOG) ==="
# shellcheck disable=SC2086
python3 -m autolabel.sku_vlm \
  --backend "$BACKEND" \
  $MODEL_FLAG \
  $ENDPOINT_FLAG \
  --crops "$CROPS_DIR" \
  --out "$OUT_CSV" \
  $LIMIT_FLAG \
  >> "$LOG" 2>&1
echo "=== benchmark done (exit $?) ==="
