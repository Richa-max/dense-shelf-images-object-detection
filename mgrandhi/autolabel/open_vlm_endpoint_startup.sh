#!/bin/bash
# Startup script for an internal OpenAI-compatible vLLM endpoint on GCP.
#
# The VM exposes http://<internal-ip>:8000/v1 for benchmark VMs and Streamlit UI use.
set -uo pipefail

MODEL="__MODEL__"
SERVED_MODEL_NAME="__SERVED_MODEL_NAME__"
PORT="__PORT__"
MAX_MODEL_LEN="__MAX_MODEL_LEN__"
GPU_MEMORY_UTILIZATION="__GPU_MEMORY_UTILIZATION__"
TTL_HOURS="__TTL_HOURS__"

LOG_DIR=/opt/vllm
LOG="$LOG_DIR/startup.log"
mkdir -p "$LOG_DIR"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "=== Open VLM endpoint startup ==="
echo "model=$MODEL served_model_name=$SERVED_MODEL_NAME port=$PORT"
date

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export HF_HOME=/opt/hf-cache
export TRANSFORMERS_CACHE=/opt/hf-cache
mkdir -p "$HF_HOME"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "Installing vLLM and multimodal helpers..."
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade vllm qwen-vl-utils huggingface_hub
# The DLVM image may include torchaudio built for a different CUDA version than the vLLM
# PyTorch wheel. Transformers can import torchaudio opportunistically, so remove it.
python3 -m pip uninstall -y torchaudio || true

cat > "$LOG_DIR/run_server.sh" <<EOF
#!/bin/bash
set -uo pipefail
export HF_HOME=/opt/hf-cache
export TRANSFORMERS_CACHE=/opt/hf-cache
export PYTHONUNBUFFERED=1
if [[ -n "\${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="\$HF_TOKEN"
fi
exec python3 -m vllm.entrypoints.openai.api_server \\
  --host 0.0.0.0 \\
  --port "$PORT" \\
  --model "$MODEL" \\
  --served-model-name "$SERVED_MODEL_NAME" \\
  --trust-remote-code \\
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \\
  --max-model-len "$MAX_MODEL_LEN" \\
  --limit-mm-per-prompt '{"image": 1}'
EOF
chmod +x "$LOG_DIR/run_server.sh"

cat > /etc/systemd/system/vllm-openai.service <<EOF
[Unit]
Description=vLLM OpenAI-compatible multimodal endpoint
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=HF_TOKEN=${HF_TOKEN:-}
WorkingDirectory=$LOG_DIR
ExecStart=$LOG_DIR/run_server.sh
Restart=always
RestartSec=20
StandardOutput=append:$LOG_DIR/server.log
StandardError=append:$LOG_DIR/server.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable vllm-openai.service
systemctl restart vllm-openai.service

if [[ "$TTL_HOURS" != "0" ]]; then
  echo "Scheduling VM shutdown in $TTL_HOURS hours to avoid runaway GPU cost."
  shutdown -h +"$((TTL_HOURS * 60))" "vLLM endpoint TTL reached" || true
fi

echo "Waiting for vLLM /v1/models readiness..."
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/tmp/vllm_models.json; then
    echo "READY"
    cat /tmp/vllm_models.json
    exit 0
  fi
  if (( i % 10 == 0 )); then
    echo "still waiting for vLLM (${i}/180); latest server log:"
    tail -40 "$LOG_DIR/server.log" || true
  fi
  sleep 10
done

echo "WARN: vLLM did not become ready within startup wait window."
systemctl status vllm-openai.service --no-pager || true
exit 0
