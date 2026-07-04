FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    GRADIO_SERVER_NAME=0.0.0.0 \
    EASYOCR_LANGS=en \
    EASYOCR_GPU=1 \
    EASYOCR_MODEL_DIR=/workspace/.cache/easyocr \
    PRELOAD_EASYOCR=1 \
    PRELOAD_LLAMA=1 \
    RETAIL_REASONER_PROVIDER=hf \
    RETAIL_REASONER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct \
    LLAMA_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct \
    LLAMA_DEVICE_MAP=auto \
    LLAMA_LOAD_IN_4BIT=1 \
    OLLAMA_AUTOSTART=1 \
    OLLAMA_MODELS=/workspace/.ollama/models \
    LLAVA_PROVIDER=ollama \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    OLLAMA_MODEL=llava:13b \
    PORT=7860

WORKDIR /workspace/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN sed -i 's/\r$//' /workspace/app/docker-entrypoint.sh \
    && chmod +x /workspace/app/docker-entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/workspace/app/docker-entrypoint.sh"]
CMD ["python", "app.py"]
