#!/bin/bash
# Local launcher — creates a no-public-IP GCP VM that benchmarks SKU/OCR VLM extraction.
#
# Examples:
#
# Dry-run harness validation on 100 val crops:
#   export PROJECT_ID=ehc-mgrandhi-bc801a
#   export BACKEND=dry-run MODEL=dry-run RUN_NAME=dryrun_sku_100
#   bash autolabel/launch_sku_vlm_benchmark.sh
#
# Qwen2.5-VL via a vLLM / Vertex Model Garden OpenAI-compatible endpoint:
#   export PROJECT_ID=ehc-mgrandhi-bc801a
#   export BACKEND=openai-compatible
#   export MODEL=Qwen/Qwen2.5-VL-7B-Instruct
#   export VLM_ENDPOINT_URL=https://<endpoint-host>/v1
#   export AUTH_MODE=metadata      # use VM service-account access token as Bearer token
#   export RUN_NAME=qwen25_vl_100
#   bash autolabel/launch_sku_vlm_benchmark.sh
#
# Gemini reference ceiling (not open source):
#   export BACKEND=gemini MODEL=gemini-2.5-flash RUN_NAME=gemini25_ref_100
#   bash autolabel/launch_sku_vlm_benchmark.sh
#
# Outputs land in:
#   gs://$PROJECT_ID-sku110k-yolo/sku_vlm_benchmarks/$RUN_NAME/
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID (e.g. export PROJECT_ID=my-gcp-project)}"
ZONE="${ZONE:-us-central1-a}"
REGION="${ZONE%-*}"
BUCKET="${BUCKET:-gs://${PROJECT_ID}-sku110k-yolo}"
SPLIT="${SPLIT:-val}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-100}"
BACKEND="${BACKEND:-dry-run}"
MODEL="${MODEL:-dry-run}"
VLM_ENDPOINT_URL="${VLM_ENDPOINT_URL:-}"
AUTH_MODE="${AUTH_MODE:-none}"   # none | metadata
RUN_NAME="${RUN_NAME:-${BACKEND//[^a-zA-Z0-9]/_}_${SPLIT}_${SAMPLE_LIMIT}}"

MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-8}"
ACCELERATOR="${ACCELERATOR-}"
INSTANCE="${INSTANCE:-sku-vlm-bench-${RUN_NAME//[^a-zA-Z0-9-]/-}}"

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"

if [[ "$BACKEND" == "gemini" || "$AUTH_MODE" == "metadata" ]]; then
  HAS_AIP="$(gcloud projects get-iam-policy "$PROJECT_ID" \
    --flatten='bindings[].members' \
    --filter="bindings.members:serviceAccount:${SERVICE_ACCOUNT} AND bindings.role:roles/aiplatform.user" \
    --format='value(bindings.role)' 2>/dev/null | head -1)"
  if [[ -z "$HAS_AIP" ]]; then
    cat >&2 <<EOF
ERROR: service account $SERVICE_ACCOUNT lacks roles/aiplatform.user.
       Grant it (needs a project owner), then re-run:

  gcloud projects add-iam-policy-binding $PROJECT_ID \\
    --member="serviceAccount:${SERVICE_ACCOUNT}" \\
    --role="roles/aiplatform.user" --condition=None
EOF
    exit 1
  fi
fi

ROUTER="$(gcloud compute routers list --regions="$REGION" --format='value(name)' 2>/dev/null | head -1)"
NAT_FOUND=""
if [[ -n "$ROUTER" ]]; then
  NAT_FOUND="$(gcloud compute routers describe "$ROUTER" --region="$REGION" --format='value(nats[].name)' 2>/dev/null)"
fi
if [[ -z "$NAT_FOUND" ]]; then
  echo "ERROR: no Cloud NAT on any router in region '$REGION'; no-public-IP VM will hang on apt/pip." >&2
  exit 1
fi

if [[ "$BACKEND" == "openai-compatible" && -z "$VLM_ENDPOINT_URL" ]]; then
  echo "ERROR: BACKEND=openai-compatible requires VLM_ENDPOINT_URL." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Packaging autolabel/ code and uploading to $BUCKET/code/ ..."
CODE_TARBALL="$(mktemp -t sku-vlm-code.XXXXXX.tar.gz)"
tar -czf "$CODE_TARBALL" -C "$REPO_ROOT" autolabel
REPO_TARBALL="${BUCKET}/code/sku_vlm_code.tar.gz"
gsutil -q cp "$CODE_TARBALL" "$REPO_TARBALL"
rm -f "$CODE_TARBALL"

STARTUP="$(mktemp)"
sed -e "s|__BUCKET__|${BUCKET}|g" \
    -e "s|__REPO_TARBALL__|${REPO_TARBALL}|g" \
    -e "s|__SPLIT__|${SPLIT}|g" \
    -e "s|__SAMPLE_LIMIT__|${SAMPLE_LIMIT}|g" \
    -e "s|__BACKEND__|${BACKEND}|g" \
    -e "s|__MODEL__|${MODEL}|g" \
    -e "s|__ENDPOINT__|${VLM_ENDPOINT_URL}|g" \
    -e "s|__AUTH_MODE__|${AUTH_MODE}|g" \
    -e "s|__RUN_NAME__|${RUN_NAME}|g" \
    "$SCRIPT_DIR/sku_vlm_benchmark.sh" > "$STARTUP"

ACCEL_FLAGS=()
if [[ -n "$ACCELERATOR" ]]; then
  ACCEL_FLAGS=(--accelerator="$ACCELERATOR" --maintenance-policy=TERMINATE --metadata=install-nvidia-driver=True)
fi

cat <<EOF
Project        : $PROJECT_ID
Zone / Region  : $ZONE / $REGION
Bucket         : $BUCKET
Split / sample : $SPLIT / $SAMPLE_LIMIT crops
Backend/model  : $BACKEND / $MODEL
Endpoint       : ${VLM_ENDPOINT_URL:-none}
Auth mode      : $AUTH_MODE
Run name       : $RUN_NAME
Instance       : $INSTANCE
Machine        : $MACHINE_TYPE   accelerator: ${ACCELERATOR:-none}
External IP    : NONE  (--no-address; egress via Cloud NAT)
EOF
echo "Creating SKU VLM benchmark VM..."

# shellcheck disable=SC2086
gcloud compute instances create "$INSTANCE" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --subnet=default \
  --no-address \
  --service-account="$SERVICE_ACCOUNT" \
  --machine-type="$MACHINE_TYPE" \
  ${ACCEL_FLAGS[@]+"${ACCEL_FLAGS[@]}"} \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --metadata-from-file=startup-script="$STARTUP" \
  --scopes=https://www.googleapis.com/auth/cloud-platform

rm -f "$STARTUP"

cat <<EOF

VM '$INSTANCE' is booting. Benchmark begins automatically; the VM self-deletes when done.

Monitor:
  gcloud compute instances get-serial-port-output $INSTANCE --zone $ZONE | tail -60

When the instance disappears:
  gsutil ls $BUCKET/sku_vlm_benchmarks/$RUN_NAME/
  gsutil cat $BUCKET/sku_vlm_benchmarks/$RUN_NAME/${RUN_NAME}.summary.json
EOF
