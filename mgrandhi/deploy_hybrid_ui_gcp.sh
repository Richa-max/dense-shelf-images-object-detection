#!/usr/bin/env bash
# Deploy the hybrid Streamlit + Gradio UI to a GCP VM and expose it by public IP.
#
# Defaults are chosen for demo reliability and repeated VM recreation:
# - g2-standard-8 has an NVIDIA L4 GPU + enough RAM for YOLO/SWIN/FAISS inference.
# - Large ignored assets are uploaded once to GCS and downloaded by each new VM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-us-central1-a}"
REGION="${REGION:-${ZONE%-*}}"
INSTANCE="${INSTANCE:-retail-inventory-ui-gpu}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
ACCELERATOR="${ACCELERATOR:-}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-150GB}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8502}"
GRADIO_PORT="${GRADIO_PORT:-7860}"
FIREWALL_RULE="${FIREWALL_RULE:-retail-inventory-ui-public}"
IAP_FIREWALL_RULE="${IAP_FIREWALL_RULE:-retail-inventory-ui-iap-ssh}"
TAG="${TAG:-retail-inventory-ui}"
REMOTE_DIR="${REMOTE_DIR:-/opt/retail-inventory-ai}"
USE_IAP="${USE_IAP:-1}"
ASSET_BUCKET="${ASSET_BUCKET:-gs://${PROJECT_ID}-retail-inventory-ai-assets}"
SYNC_ASSETS_TO_GCS="${SYNC_ASSETS_TO_GCS:-1}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "PROJECT_ID is required. Set PROJECT_ID or run: gcloud config set project <project-id>" >&2
  exit 1
fi

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"

required_assets=(
  "detection/artifacts/v11/best.pt"
  "retrieval/assets/swin_faiss_index.bin"
  "retrieval/assets/swin_faiss_indexed_image_paths.csv"
  "retrieval/assets/labels/train_product_category_58.csv"
  "retrieval/assets/swin_model_assets/model.safetensors"
)

for asset in "${required_assets[@]}"; do
  if [[ ! -s "$asset" ]]; then
    echo "Missing required asset: $asset" >&2
    echo "Provision retrieval assets first; see retrieval/README.md." >&2
    exit 1
  fi
done

echo "Project      : $PROJECT_ID"
echo "Zone         : $ZONE"
echo "Instance     : $INSTANCE"
echo "Machine type : $MACHINE_TYPE"
echo "Accelerator  : ${ACCELERATOR:-machine-default}"
echo "Image        : $IMAGE_PROJECT/$IMAGE_FAMILY"
echo "Asset bucket : $ASSET_BUCKET"
echo "Service acct : $SERVICE_ACCOUNT"
echo "Ports        : Streamlit $STREAMLIT_PORT, Gradio $GRADIO_PORT"

if ! gcloud storage buckets describe "$ASSET_BUCKET" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Creating reusable asset bucket $ASSET_BUCKET..."
  gcloud storage buckets create "$ASSET_BUCKET" \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --uniform-bucket-level-access
fi

echo "Ensuring VM service account can read asset bucket..."
gcloud storage buckets add-iam-policy-binding "$ASSET_BUCKET" \
  --project "$PROJECT_ID" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" \
  --role roles/storage.objectViewer >/dev/null

if [[ "$SYNC_ASSETS_TO_GCS" == "1" ]]; then
  echo "Uploading large model/index assets to GCS if needed..."
  for asset in "${required_assets[@]}"; do
    if ! gcloud storage ls "$ASSET_BUCKET/$asset" >/dev/null 2>&1; then
      gcloud storage cp "$asset" "$ASSET_BUCKET/$asset"
    else
      echo "Already in GCS: $ASSET_BUCKET/$asset"
    fi
  done
fi

if ! gcloud compute firewall-rules describe "$FIREWALL_RULE" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Creating firewall rule $FIREWALL_RULE..."
  gcloud compute firewall-rules create "$FIREWALL_RULE" \
    --project "$PROJECT_ID" \
    --allow "tcp:${STREAMLIT_PORT},tcp:${GRADIO_PORT}" \
    --target-tags "$TAG" \
    --source-ranges "0.0.0.0/0" \
    --description "Public demo access for retail inventory Streamlit + Gradio UI"
fi

if [[ "$USE_IAP" == "1" ]] && \
   ! gcloud compute firewall-rules describe "$IAP_FIREWALL_RULE" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Creating IAP SSH firewall rule $IAP_FIREWALL_RULE..."
  gcloud compute firewall-rules create "$IAP_FIREWALL_RULE" \
    --project "$PROJECT_ID" \
    --allow tcp:22 \
    --target-tags "$TAG" \
    --source-ranges "35.235.240.0/20" \
    --description "IAP TCP forwarding SSH access for retail inventory UI VM"
fi

if ! gcloud compute instances describe "$INSTANCE" --project "$PROJECT_ID" --zone "$ZONE" >/dev/null 2>&1; then
  echo "Creating VM $INSTANCE..."
  ACCEL_FLAGS=()
  if [[ -n "$ACCELERATOR" ]]; then
    ACCEL_FLAGS=(--accelerator="$ACCELERATOR")
  fi
  gcloud compute instances create "$INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$ZONE" \
    --machine-type "$MACHINE_TYPE" \
    ${ACCEL_FLAGS[@]+"${ACCEL_FLAGS[@]}"} \
    --maintenance-policy TERMINATE \
    --image-family "$IMAGE_FAMILY" \
    --image-project "$IMAGE_PROJECT" \
    --boot-disk-size "$BOOT_DISK_SIZE" \
    --boot-disk-type pd-ssd \
    --tags "$TAG" \
    --service-account "$SERVICE_ACCOUNT" \
    --scopes cloud-platform \
    --metadata install-nvidia-driver=True
else
  echo "VM $INSTANCE already exists; reusing it."
fi

EXTERNAL_IP="$(gcloud compute instances describe "$INSTANCE" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"

if [[ -z "$EXTERNAL_IP" ]]; then
  echo "The VM does not have an external IP. Recreate it with an external NAT IP or use an IAP tunnel." >&2
  exit 1
fi

GcloudSshFlags=()
if [[ "$USE_IAP" == "1" ]]; then
  GcloudSshFlags=(--tunnel-through-iap)
fi

echo "Waiting for SSH to become ready..."
SSH_READY=0
for _ in $(seq 1 30); do
  if gcloud compute ssh "$INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$ZONE" \
    "${GcloudSshFlags[@]}" \
    --command "true" >/dev/null 2>&1; then
    SSH_READY=1
    break
  fi
  sleep 10
done

if [[ "$SSH_READY" != "1" ]]; then
  echo "SSH did not become ready. Check VM startup logs or retry this script." >&2
  exit 1
fi

ARCHIVE="$(mktemp -t retail-inventory-ai.XXXXXX.tar.gz)"
cleanup() {
  rm -f "$ARCHIVE"
}
trap cleanup EXIT

echo "Packaging source code (large assets come from GCS)..."
tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='retail_inventory_ai.egg-info' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='data' \
  --exclude='datasets' \
  --exclude='notebooks' \
  --exclude='report' \
  --exclude='inventory.db' \
  --exclude='detection/artifacts/v11/best.pt' \
  --exclude='retrieval/assets/swin_faiss_index.bin' \
  --exclude='retrieval/assets/swin_faiss_indexed_image_paths.csv' \
  --exclude='retrieval/assets/labels/train_product_category_58.csv' \
  --exclude='retrieval/assets/swin_model_assets/model.safetensors' \
  -czf "$ARCHIVE" .

echo "Uploading package to VM..."
gcloud compute scp "$ARCHIVE" "$INSTANCE:/tmp/retail-inventory-ai.tar.gz" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  "${GcloudSshFlags[@]}"

echo "Installing app and starting service on VM..."
gcloud compute ssh "$INSTANCE" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  "${GcloudSshFlags[@]}" \
  --command "set -euo pipefail
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip libgl1 libglib2.0-0 libsm6 libxext6
if ! command -v gcloud >/dev/null 2>&1; then
  sudo apt-get install -y google-cloud-cli || sudo apt-get install -y google-cloud-sdk
fi
sudo rm -rf '$REMOTE_DIR'
sudo mkdir -p '$REMOTE_DIR'
sudo tar -xzf /tmp/retail-inventory-ai.tar.gz -C '$REMOTE_DIR'
sudo chown -R \$USER:\$USER '$REMOTE_DIR'
cd '$REMOTE_DIR'
for asset in ${required_assets[*]}; do
  mkdir -p \"\$(dirname \"\$asset\")\"
  gcloud storage cp '$ASSET_BUCKET/'\"\$asset\" \"\$asset\"
done
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[retrieval]'
cat > /tmp/retail-inventory-ui.service <<EOF
[Unit]
Description=Retail Inventory AI hybrid UI
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$REMOTE_DIR
Environment=KMP_DUPLICATE_LIB_OK=TRUE
Environment=OMP_NUM_THREADS=1
Environment=PROJECT_ID=$PROJECT_ID
Environment=REGION=$REGION
Environment=STREAMLIT_PORT=$STREAMLIT_PORT
Environment=GRADIO_PORT=$GRADIO_PORT
Environment=GRADIO_URL=http://$EXTERNAL_IP:$GRADIO_PORT
Environment=SKU_BACKEND=${SKU_BACKEND:-gemini}
Environment=SKU_MODEL=${SKU_MODEL:-}
Environment=VERTEX_MODEL_GARDEN_MODEL=${VERTEX_MODEL_GARDEN_MODEL:-google/paligemma@paligemma-mix-448-float16}
Environment=VERTEX_MODEL_GARDEN_ENDPOINT_ID=${VERTEX_MODEL_GARDEN_ENDPOINT_ID:-}
Environment=VERTEX_MODEL_GARDEN_DEDICATED_DNS=${VERTEX_MODEL_GARDEN_DEDICATED_DNS:-}
Environment=PATH=$REMOTE_DIR/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/bin/bash $REMOTE_DIR/frontend/run_hybrid_ui.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
sudo mv /tmp/retail-inventory-ui.service /etc/systemd/system/retail-inventory-ui.service
sudo systemctl daemon-reload
sudo systemctl enable retail-inventory-ui.service
sudo systemctl restart retail-inventory-ui.service
sudo systemctl --no-pager --full status retail-inventory-ui.service"

cat <<EOF

Deployment started.

Streamlit dashboard: http://$EXTERNAL_IP:$STREAMLIT_PORT
Gradio fast upload : http://$EXTERNAL_IP:$GRADIO_PORT

Useful commands:
  gcloud compute ssh $INSTANCE --project $PROJECT_ID --zone $ZONE
  gcloud compute ssh $INSTANCE --project $PROJECT_ID --zone $ZONE --command 'sudo journalctl -u retail-inventory-ui -f'
  gcloud compute instances stop $INSTANCE --project $PROJECT_ID --zone $ZONE
EOF
