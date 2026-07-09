# Mahesh SKU OCR + Inventory Demo

This folder contains Mahesh's SKU/OCR, database schema, benchmark harness, and Streamlit UI setup.

It intentionally does not include large model files. It uses the parent teammate repo's existing
LFS assets for YOLO, SWIN, FAISS, and labels.

## Setup

\`\`\`bash
cd ~/Projects/dense-shelf-images-object-detection
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r mgrandhi/requirements.txt
\`\`\`

## Run

\`\`\`bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 streamlit run mgrandhi/app.py
\`\`\`

## Notes

- Update `mgrandhi/config/paths.py` if asset paths differ.
- Do not copy `.pt`, FAISS, SWIN, or other large LFS assets into this folder.
- Start SKU/OCR extraction with a small crop limit, for example `3`.
