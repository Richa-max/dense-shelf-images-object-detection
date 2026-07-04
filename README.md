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
- HF_HOME=/workspace/.cache/huggingface
- TRANSFORMERS_CACHE=/workspace/.cache/huggingface

## Notes

- The Gradio app is configured to bind to 0.0.0.0 and reads PORT.
- YOLO model path in app.py points to models/yolo/best.pt.
- If model artifacts are large, use Git LFS or pull model files from remote storage at startup.
- Each crop is resolved with Swin/FAISS candidates and a heuristic retail-product decision. OCR and retail SLM reranking are not part of the current flow.
