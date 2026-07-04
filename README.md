# Shelf Object Detection

This project provides a Gradio UI for shelf image upload, YOLO-based crop detection, and LLaVA-based product category prediction.

## What Runs in This Repo

- UI mode entrypoint: app.py
- API mode entrypoint: server.py
- Container entrypoint in Dockerfile: app.py (Gradio UI)

## Local Run (Without Docker)

1. Install dependencies:
   pip install -r requirements.txt
2. Start the UI:
   python app.py
3. Open:
   http://localhost:7860

## Docker Run (Local)

1. Build image:
   docker build -t shelf-ui .
2. Run container with GPU:
   docker run --gpus all -p 7860:7860 shelf-ui
3. Open:
   http://localhost:7860

## Deploy to RunPod (UI Mode)

1. Push this repository to GitHub.
2. In RunPod, create a GPU Pod or Serverless endpoint using this repository Dockerfile.
3. Expose container port 7860.
4. Keep the default container command. The Dockerfile starts app.py automatically.
5. Open the RunPod HTTP endpoint for port 7860.

## Recommended Environment Variables

- PORT=7860
- GRADIO_SERVER_NAME=0.0.0.0
- YOLO_CONFIG_DIR=/tmp/Ultralytics
- HF_HOME=/workspace/.cache/huggingface
- TRANSFORMERS_CACHE=/workspace/.cache/huggingface
- EASYOCR_LANGS=en
- EASYOCR_GPU=1
- EASYOCR_MODEL_DIR=/workspace/.cache/easyocr
- PRELOAD_EASYOCR=1
- PRELOAD_LLAMA=1
- RETAIL_REASONER_PROVIDER=hf
- RETAIL_REASONER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
- LLAMA_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
- HF_TOKEN=<token with access to the Meta Llama model>
- LLAMA_DEVICE_MAP=auto
- LLAMA_LOAD_IN_4BIT=0

## Notes

- The Gradio app is configured to bind to 0.0.0.0 and reads PORT.
- YOLO model path in app.py points to models/yolo/best.pt.
- The Docker entrypoint preloads EasyOCR and loads Llama 3.1 8B Instruct into the app process with Hugging Face Transformers, then starts app.py.
- If model artifacts are large, use Git LFS or pull model files from remote storage at startup.
- Each crop is resolved with EasyOCR text, Swin/FAISS visual candidates, and optional in-process Llama 3.1 8B Instruct text reasoning. If OCR or Llama is unavailable, the app falls back to a Swin/FAISS heuristic decision.
