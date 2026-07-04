FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PADDLE_OCR_LANG=en \
    PADDLE_OCR_USE_GPU=0 \
    PRELOAD_PADDLE_OCR=1 \
    OLLAMA_AUTOSTART=1 \
    OLLAMA_MODELS=/workspace/.ollama/models \
    RETAIL_REASONER_PROVIDER=ollama \
    RETAIL_REASONER_MODEL=llama3.1:8b \
    LLAMA_MODEL=llama3.1:8b \
    PORT=7860

WORKDIR /workspace/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN sed -i 's/\r$//' /workspace/app/docker-entrypoint.sh \
    && chmod +x /workspace/app/docker-entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/workspace/app/docker-entrypoint.sh"]
CMD ["python", "app.py"]
