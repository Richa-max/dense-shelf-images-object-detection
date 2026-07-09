#!/bin/bash
# Launch an internal GCP vLLM endpoint that implements the OpenAI-compatible API.
#
# Example:
#   export PROJECT_ID=ehc-mgrandhi-bc801a
#   export MODEL=Qwen/Qwen2.5-VL-7B-Instruct
#   bash autolabel/launch_open_vlm_endpoint.sh
#
# After READY:
#   export VLM_ENDPOINT_URL=http://<INTERNAL_IP>:8000/v1
#   export BACKEND=openai-compatible MODEL=Qwen/Qwen2.5-VL-7B-Instruct SAMPLE_LIMIT=2000
#   bash autolabel/launch_sku_vlm_benchmark.sh
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID, e.g. export PROJECT_ID=ehc-mgrandhi-bc801a}"

ZONE="${ZONE:-us-central1-a}"
REGION="${ZONE%-*}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
INSTANCE="${INSTANCE:-sku-vllm-qwen25-vl}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
ACCELERATOR="${ACCELERATOR:-}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
TTL_HOURS="${TTL_HOURS:-8}"
FIREWALL_RULE="${FIREWALL_RULE:-allow-vllm-internal-8000}"
SOURCE_RANGES="${SOURCE_RANGES:-10.128.0.0/9}"

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"

if ! gcloud projects get-iam-policy "$PROJECT_ID" \
  --flatten='bindings[].members' \
  --filter="bindings.members:serviceAccount:${SERVICE_ACCOUNT} AND bindings.role:roles/aiplatform.user" \
  --format='value(bindings.role)' 2>/dev/null | grep -q roles/aiplatform.user; then
  echo "WARN: $SERVICE_ACCOUNT does not have roles/aiplatform.user. vLLM serving does not need it,"
  echo "      but benchmark VMs using Vertex auth may. Continuing."
fi

if ! gcloud compute firewall-rules describe "$FIREWALL_RULE" \
  --project="$PROJECT_ID" >/dev/null 2>&1; then
  echo "Creating internal firewall rule $FIREWALL_RULE for tcp:$PORT ..."
  gcloud compute firewall-rules create "$FIREWALL_RULE" \
    --project="$PROJECT_ID" \
    --network=default \
    --allow="tcp:${PORT}" \
    --source-ranges="$SOURCE_RANGES" \
    --target-tags=vllm-server \
    --description="Allow internal benchmark/UI clients to reach vLLM OpenAI endpoint"
else
  echo "Firewall rule $FIREWALL_RULE already exists."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARTUP="$(mktemp)"
sed -e "s|__MODEL__|${MODEL}|g" \
    -e "s|__SERVED_MODEL_NAME__|${SERVED_MODEL_NAME}|g" \
    -e "s|__PORT__|${PORT}|g" \
    -e "s|__MAX_MODEL_LEN__|${MAX_MODEL_LEN}|g" \
    -e "s|__GPU_MEMORY_UTILIZATION__|${GPU_MEMORY_UTILIZATION}|g" \
    -e "s|__TTL_HOURS__|${TTL_HOURS}|g" \
    "$SCRIPT_DIR/open_vlm_endpoint_startup.sh" > "$STARTUP"

if gcloud compute instances describe "$INSTANCE" \
  --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1; then
  echo "Instance $INSTANCE already exists. Current details:"
  gcloud compute instances describe "$INSTANCE" \
    --project="$PROJECT_ID" --zone="$ZONE" \
    --format='table(name,status,networkInterfaces[0].networkIP)'
  rm -f "$STARTUP"
  exit 0
fi

cat <<EOF
Project       : $PROJECT_ID
Zone / Region : $ZONE / $REGION
Instance      : $INSTANCE
Machine       : $MACHINE_TYPE
Accelerator   : ${ACCELERATOR:-machine-default}
Model         : $MODEL
Port          : $PORT
External IP   : NONE
TTL shutdown  : ${TTL_HOURS}h (0 disables)
EOF

echo "Creating vLLM endpoint VM..."
ACCEL_FLAGS=()
if [[ -n "$ACCELERATOR" ]]; then
  ACCEL_FLAGS=(--accelerator="$ACCELERATOR")
fi

gcloud compute instances create "$INSTANCE" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --subnet=default \
  --no-address \
  --tags=vllm-server \
  --service-account="$SERVICE_ACCOUNT" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --machine-type="$MACHINE_TYPE" \
  "${ACCEL_FLAGS[@]}" \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=250GB \
  --boot-disk-type=pd-ssd \
  --metadata=install-nvidia-driver=True \
  --metadata-from-file=startup-script="$STARTUP"

rm -f "$STARTUP"

INTERNAL_IP="$(gcloud compute instances describe "$INSTANCE" \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --format='value(networkInterfaces[0].networkIP)')"

cat <<EOF

Endpoint VM is booting.

Internal OpenAI-compatible endpoint:
  export VLM_ENDPOINT_URL=http://${INTERNAL_IP}:${PORT}/v1

Readiness check:
  gcloud compute ssh $INSTANCE --project=$PROJECT_ID --zone=$ZONE --tunnel-through-iap \\
    --command='curl -s http://127.0.0.1:${PORT}/v1/models'

Logs:
  gcloud compute ssh $INSTANCE --project=$PROJECT_ID --zone=$ZONE --tunnel-through-iap \\
    --command='sudo journalctl -u vllm-openai -n 80 --no-pager'

Stop when done:
  gcloud compute instances stop $INSTANCE --project=$PROJECT_ID --zone=$ZONE
EOF
