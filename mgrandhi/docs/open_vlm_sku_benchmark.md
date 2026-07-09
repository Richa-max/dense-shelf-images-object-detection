# Open VLM SKU/OCR Benchmark

This benchmark evaluates open multimodal models for extracting SKU-like product identity from
individual product crops. The goal is not just category classification; it is to read packaging
text well enough to help automatic checkout resolve a crop to a product master row.

## Candidate Models

| Priority | Model | Why |
|---|---|---|
| 1 | `Qwen2.5-VL-7B-Instruct` / `Qwen3-VL` | Best first candidate for OCR-heavy key information extraction; strong public OCR/KIE recipes and vLLM serving path. |
| 2 | `PaliGemma 2 Mix` (`10B` or `28B`, `448px`) | Google open VLM baseline through Vertex Model Garden; explicitly supports OCR, VQA, detection, and segmentation tasks. |
| 3 | `Gemma 3` multimodal (`12B` preferred) | Google-native open-model deployment baseline in Vertex AI Model Garden / Gemini Enterprise Agent Platform. |
| Reference | `Gemini 2.5 Flash` | Not open source; use only as an accuracy/cost ceiling because this project already verified it on Vertex. |

## Prompt Contract

Each model receives the same crop image and task:

> You are reading one cropped retail product image. Extract the visible product identity for
> inventory and automatic checkout. Read packaging text carefully. If the crop is blurry,
> partial, or text is not visible, return empty strings and set `needs_review=true`.

The model must return JSON only:

```json
{
  "brand": "string",
  "product_name": "string",
  "sku_text": "string",
  "visible_text": "string",
  "package_size": "string",
  "barcode": "string",
  "category_hint": "string",
  "confidence": 0.0,
  "needs_review": true
}
```

## Metrics

- **Parse success rate**: percentage of responses parsed as valid JSON.
- **Brand/product usefulness**: percentage with a non-empty `brand` or `product_name`.
- **OCR usefulness**: percentage with non-empty `visible_text` or `sku_text`.
- **Needs-review rate**: percentage flagged by the model.
- **Latency**: average seconds per crop.
- **Error rate**: failed inference calls.
- **Agreement**: optional agreement with current FAISS/Gemini category labels.

## Sample Sizes

- `100` crops: GCP smoke test.
- `500` crops: prompt/model selection.
- `2,000` crops: report comparison table.

## GCP OpenAI-Compatible Endpoint

The repeatable GCP path is a no-public-IP Compute Engine GPU VM running `vllm`'s OpenAI-compatible
server. It exposes an internal endpoint that benchmark VMs and the Streamlit UI can call:

```bash
export PROJECT_ID=ehc-mgrandhi-bc801a

# Preferred when G2/L4 capacity is available.
export MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export MACHINE_TYPE=g2-standard-8
unset ACCELERATOR
bash autolabel/launch_open_vlm_endpoint.sh

# Fallback used when G2/L4 is stocked out.
export MODEL=Qwen/Qwen2.5-VL-3B-Instruct
export MACHINE_TYPE=n1-standard-8
export ACCELERATOR=type=nvidia-tesla-t4,count=1
export INSTANCE=sku-vllm-qwen25-vl-3b
bash autolabel/launch_open_vlm_endpoint.sh
```

The launcher prints:

```bash
export VLM_ENDPOINT_URL=http://<internal-ip>:8000/v1
```

For the local Streamlit UI, create an IAP tunnel because the endpoint VM has no public IP:

```bash
gcloud compute ssh sku-vllm-qwen25-vl-3b \
  --project=ehc-mgrandhi-bc801a \
  --zone=us-central1-a \
  --tunnel-through-iap \
  -- -N -L 18000:localhost:8000
```

Then set the UI's OpenAI-compatible endpoint to:

```bash
http://127.0.0.1:18000/v1
```

Use that endpoint for the benchmark:

```bash
export BACKEND=openai-compatible
export MODEL=Qwen/Qwen2.5-VL-3B-Instruct
export VLM_ENDPOINT_URL=http://<internal-ip>:8000/v1
export SAMPLE_LIMIT=2000
export RUN_NAME=qwen25vl3b_sku_2000_gcp
bash autolabel/launch_sku_vlm_benchmark.sh
```

The endpoint VM is intentionally internal-only and has a default TTL shutdown to prevent runaway
GPU cost. Stop it manually when done:

```bash
gcloud compute instances stop sku-vllm-qwen25-vl-3b \
  --project=ehc-mgrandhi-bc801a --zone=us-central1-a
```

## Recommended Decision Rule

Pick the model with the highest OCR usefulness and product-name usefulness under acceptable
latency/cost. If two models are close, prefer the easier GCP deployment path for the final demo.
