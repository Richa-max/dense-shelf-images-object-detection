"""Smart Shelf Analytics & BI Dashboard (Module 7 — Streamlit).

Upload a shelf image -> YOLO detects products -> SWIN+FAISS classifies each crop -> the app
shows KPIs, interactive analytics charts, and a natural-language Business-Intelligence panel
that answers questions over the accumulated inventory (SQLite).

Run from the repo root:
    source .venv/bin/activate
    KMP_DUPLICATE_LIB_OK=TRUE streamlit run frontend/app.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from urllib import request as urllib_request

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Ensure the repo root is importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

from backend import inventory_db as db
from bi_interface import bi_engine
from retrieval import pipeline

load_dotenv()

st.set_page_config(page_title="Smart Shelf Analytics & BI", page_icon="🛒", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }
      div[data-testid="stMetric"] {
          background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
          border: 1px solid #374151; border-radius: 14px; padding: 16px 18px;
      }
      div[data-testid="stMetric"] label p { color: #9ca3af !important; font-size: .8rem; }
      div[data-testid="stMetricValue"] { color: #f9fafb !important; }
      h1, h2, h3 { letter-spacing: -0.01em; }
      .pill { display:inline-block; padding:2px 10px; border-radius:999px;
              background:#1e3a8a; color:#dbeafe; font-size:.75rem; margin-left:6px; }
      .muted { color:#9ca3af; font-size:.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

PALETTE = px.colors.qualitative.Set3
GRADIO_URL = os.getenv("GRADIO_URL", "http://localhost:7860")
STREAMLIT_ANALYZE_ENABLED = os.getenv("STREAMLIT_ANALYZE_ENABLED", "1") == "1"


SKU_COLUMNS = [
    "brand",
    "product_name",
    "sku_text",
    "visible_text",
    "package_size",
    "barcode",
    "sku_confidence",
    "sku_needs_review",
    "sku_latency_s",
    "sku_error",
]


def _style_fig(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e5e7eb"), legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#374151")
    fig.update_yaxes(gridcolor="#374151")
    return fig


@st.cache_resource(show_spinner=False)
def _load_sku_backend(backend: str, model: str, endpoint: str, api_key: str,
                      project: str, location: str, timeout: int, dedicated_dns: str = ""):
    from autolabel.sku_vlm import build_backend

    args = SimpleNamespace(
        backend=backend,
        model=model,
        endpoint=endpoint or None,
        api_key=api_key or "",
        project=project or None,
        location=location or "us-central1",
        timeout=timeout,
        dedicated_dns=dedicated_dns or "",
    )
    return build_backend(args)


def _empty_sku_fields() -> dict:
    return {
        "brand": "",
        "product_name": "",
        "sku_text": "",
        "visible_text": "",
        "package_size": "",
        "barcode": "",
        "sku_confidence": 0.0,
        "sku_needs_review": 1,
        "sku_latency_s": 0.0,
        "sku_error": "",
    }


@st.cache_data(ttl=30, show_spinner=False)
def _list_openai_models(endpoint: str) -> list[str]:
    from autolabel.sku_vlm import normalize_openai_base_url

    if not endpoint:
        return []
    url = f"{normalize_openai_base_url(endpoint)}/models"
    try:
        with urllib_request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["id"] for m in data.get("data", []) if m.get("id")]
    except Exception:
        return []


def _enrich_records_with_sku(records: list[dict], image: Image.Image, backend, max_sku_crops: int):
    from autolabel.sku_vlm import coerce_bool

    if not records:
        return records
    limit = len(records) if max_sku_crops <= 0 else min(max_sku_crops, len(records))
    progress = st.progress(0, text=f"Extracting SKU/OCR for {limit} crops…")
    image = image.convert("RGB")
    with tempfile.TemporaryDirectory(prefix="sku_crops_") as tmp:
        tmp_dir = Path(tmp)
        for i, record in enumerate(records):
            record.update(_empty_sku_fields())
            if i >= limit:
                record["sku_error"] = "not_processed_limit"
                continue
            x1, y1, x2, y2 = record["box"]
            crop = image.crop((x1, y1, x2, y2))
            crop_path = tmp_dir / f"crop_{record['crop_id']:04d}.jpg"
            crop.save(crop_path, quality=92)
            pred = backend.predict(crop_path)
            parsed = pred.parsed
            record.update({
                "brand": parsed.get("brand", ""),
                "product_name": parsed.get("product_name", ""),
                "sku_text": parsed.get("sku_text", ""),
                "visible_text": parsed.get("visible_text", ""),
                "package_size": parsed.get("package_size", ""),
                "barcode": parsed.get("barcode", ""),
                "sku_confidence": parsed.get("confidence", 0.0),
                "sku_needs_review": int(coerce_bool(parsed.get("needs_review", True))),
                "sku_latency_s": pred.latency_s,
                "sku_error": pred.error,
            })
            progress.progress((i + 1) / limit, text=f"Extracting SKU/OCR ({i + 1}/{limit})…")
    progress.empty()
    return records


st.title("🛒 Smart Shelf Analytics & BI")
llm_on = bi_engine.ollama_available()
st.markdown(
    "<span class='muted'>YOLO detection · SWIN + FAISS retrieval classification · "
    "inventory analytics + natural-language BI</span>"
    f"<span class='pill'>{'LLM: Ollama ✓' if llm_on else 'BI: rule-based'}</span>",
    unsafe_allow_html=True,
)

db.init_db()

with st.sidebar:
    st.header("① Upload & detect")
    st.info(
        "For faster upload/result-table testing, run the Gradio app and use the "
        "**Fast Upload** tab."
    )
    st.link_button("Open Fast Upload UI", GRADIO_URL, use_container_width=True)

    if STREAMLIT_ANALYZE_ENABLED:
        uploaded = st.file_uploader("Shelf image", type=["jpg", "jpeg", "png", "bmp"])
        conf = st.slider("YOLO confidence", 0.05, 0.9, 0.25, 0.05)
        max_crops = st.slider("Max products to classify (0 = all)", 0, 300, 60, 10,
                              help="Cap for speed on CPU. 0 classifies every detected box.")
    else:
        uploaded = None
        conf = 0.25
        max_crops = 60
        st.warning(
            "Streamlit image analysis is disabled on this hosted demo. "
            "Use Gradio/Fast Upload for upload + result table, then return here for analytics."
        )
    st.divider()
    st.header("② SKU / OCR extraction")
    extract_sku = st.checkbox(
        "Extract SKU/OCR with VLM",
        value=False,
        help="Adds brand/product/SKU/visible-text columns to the result table. "
             "Use Gemini now, or an OpenAI-compatible Vertex/vLLM endpoint for open models.",
    )
    default_sku_backend = "vertex-model-garden" if os.getenv("PROJECT_ID") else "dry-run"
    sku_backend = st.selectbox(
        "SKU backend",
        ["dry-run", "gemini", "openai-compatible", "vertex-model-garden"],
        index=["dry-run", "gemini", "openai-compatible", "vertex-model-garden"].index(default_sku_backend),
        disabled=not extract_sku,
    )
    default_sku_model = {
        "dry-run": "dry-run",
        "gemini": os.getenv("VERTEX_MODEL", "gemini-2.5-flash"),
        "openai-compatible": os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
        "vertex-model-garden": os.getenv(
            "VERTEX_MODEL_GARDEN_MODEL", "google/paligemma@paligemma-mix-448-float16"
        ),
    }[sku_backend]
    sku_model = st.text_input("SKU model", value=default_sku_model, disabled=not extract_sku)
    default_sku_endpoint = (
        os.getenv("VLM_ENDPOINT_URL", "")
        if sku_backend == "openai-compatible"
        else os.getenv(
            "VLM_ENDPOINT_URL",
            os.getenv(
                "VERTEX_MODEL_GARDEN_ENDPOINT_ID",
                "mg-endpoint-98b3f9ea-9188-48af-b14c-87765eece175",
            )
        )
        if sku_backend == "vertex-model-garden"
        else ""
    )
    sku_endpoint = st.text_input(
        "SKU endpoint",
        value=default_sku_endpoint,
        disabled=not extract_sku or sku_backend not in {"openai-compatible", "vertex-model-garden"},
        help="OpenAI-compatible base URL, or Vertex Model Garden endpoint ID/DNS.",
    )
    if extract_sku and sku_backend == "openai-compatible" and sku_endpoint:
        try:
            from autolabel.sku_vlm import normalize_openai_base_url

            resolved_endpoint = f"{normalize_openai_base_url(sku_endpoint)}/chat/completions"
            st.caption(f"Resolved chat endpoint: `{resolved_endpoint}`")
        except Exception:
            pass
    effective_sku_model = sku_model
    if extract_sku and sku_backend == "openai-compatible" and sku_endpoint:
        served_models = _list_openai_models(sku_endpoint)
        if served_models:
            st.caption("Served model(s): `" + "`, `".join(served_models) + "`")
            if sku_model not in served_models:
                effective_sku_model = served_models[0]
                st.warning(
                    f"Using served model `{effective_sku_model}` instead of `{sku_model}`."
                )
    vertex_dedicated_dns = os.getenv(
        "VERTEX_MODEL_GARDEN_DEDICATED_DNS",
        "mg-endpoint-98b3f9ea-9188-48af-b14c-87765eece175.us-central1-735098166286.prediction.vertexai.goog",
    )
    if extract_sku and sku_backend == "vertex-model-garden":
        st.caption(f"Vertex dedicated DNS: `{vertex_dedicated_dns}`")
    sku_project = st.text_input(
        "GCP project",
        value=os.getenv("PROJECT_ID", ""),
        disabled=not extract_sku or sku_backend not in {"gemini", "vertex-model-garden"},
    )
    sku_location = st.text_input(
        "GCP region",
        value=os.getenv("REGION", "us-central1"),
        disabled=not extract_sku or sku_backend != "gemini",
    )
    max_sku_crops = st.slider(
        "Max SKU/OCR crops (0 = all classified crops)",
        0,
        100,
        10,
        5,
        disabled=not extract_sku,
        help="Keep this low for UI tests; each real VLM crop is a separate request.",
    )
    save_to_db = st.checkbox("Save scan to inventory history", value=True)
    run = st.button("🔍 Analyze shelf", type="primary", use_container_width=True,
                    disabled=uploaded is None or not STREAMLIT_ANALYZE_ENABLED)

    st.divider()
    st.caption("Inventory history")
    s = db.stats()
    st.write(f"Scans: **{s['total_scans']}** · Items: **{s['total_items']}** · "
             f"Categories: **{s['distinct_categories']}**")
    if st.button("🗑️ Clear inventory history", use_container_width=True):
        db.clear_all()
        st.rerun()


if run and uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    with st.spinner("Detecting products and classifying crops…"):
        result = pipeline.analyze_image(image, conf=conf, max_crops=max_crops)
        records = pipeline.detections_to_records(result)
    if extract_sku:
        try:
            with st.spinner("Loading SKU/OCR backend…"):
                sku_vlm = _load_sku_backend(
                    sku_backend,
                    effective_sku_model,
                    sku_endpoint,
                    os.getenv("VLM_API_KEY", ""),
                    sku_project,
                    sku_location,
                    180,
                    vertex_dedicated_dns,
                )
            records = _enrich_records_with_sku(records, image, sku_vlm, max_sku_crops)
        except Exception as exc:
            st.error(f"SKU/OCR extraction failed: {exc}")
            for record in records:
                record.update(_empty_sku_fields())
                record["sku_error"] = str(exc)
    st.session_state["result"] = result
    st.session_state["records"] = records
    if save_to_db:
        scan_id = db.save_scan(result, uploaded.name, records)
        st.session_state["last_scan_id"] = scan_id
        st.toast(f"Saved scan #{scan_id} to inventory")

result = st.session_state.get("result")
records = st.session_state.get("records", [])

tab_fast_upload, tab_analyze, tab_analytics, tab_bi, tab_history = st.tabs(
    [
        "Fast Upload",
        "Detection",
        "Analytics",
        "Business Intelligence",
        "Inventory History",
    ]
)

with tab_fast_upload:
    st.subheader("Fast Upload UI")
    st.caption(
        "This embeds the Gradio app for upload, annotated image, and detections table. "
        f"Start it with `python -m frontend.gradio_app` if the frame is empty. URL: `{GRADIO_URL}`"
    )
    components.iframe(GRADIO_URL, height=820, scrolling=True)

with tab_analyze:
    if result is None:
        st.info("Upload a shelf image and click **Analyze shelf** to begin.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Products detected", result.num_items)
        c2.metric("Distinct categories", result.distinct_categories)
        c3.metric("Empty shelf space", f"{result.empty_pct*100:.0f}%", result.empty_label)
        c4.metric("Needs review", result.review_count)
        c5.metric("Shelf type", result.shelf_type)

        left, right = st.columns([3, 2])
        with left:
            st.image(result.annotated_image, caption="Detected & classified products",
                     use_container_width=True)
            st.caption(f"YOLO {result.timings.get('yolo_s')}s · "
                       f"classify {result.timings.get('classify_s')}s · "
                       f"{result.timings.get('boxes')} boxes")
        with right:
            df = pd.DataFrame(records)
            st.markdown("**Detected items**")
            base_cols = ["crop_id", "category", "subcategory", "score"]
            sku_cols = [c for c in SKU_COLUMNS if c in df.columns]
            st.dataframe(df[base_cols + sku_cols], height=420, use_container_width=True,
                         hide_index=True)
            st.download_button("⬇️ Download detections CSV", df.to_csv(index=False).encode(),
                               file_name="detections.csv", use_container_width=True)

with tab_analytics:
    if not records:
        st.info("Run an analysis to see analytics.")
    else:
        df = pd.DataFrame(records)
        cat_counts = bi_engine.category_counts(df)

        a, b = st.columns(2)
        with a:
            st.subheader("Category distribution")
            if not cat_counts.empty:
                cc = cat_counts.rename_axis("category").reset_index(name="count")
                fig = px.bar(cc, x="count", y="category", orientation="h",
                             color="category", color_discrete_sequence=PALETTE)
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
                st.plotly_chart(_style_fig(fig, 440), use_container_width=True)
            else:
                st.caption("No confidently classified products.")
        with b:
            st.subheader("Shelf composition")
            if not cat_counts.empty:
                cc = cat_counts.rename_axis("category").reset_index(name="count")
                fig = px.pie(cc, values="count", names="category", hole=0.5,
                             color_discrete_sequence=PALETTE)
                st.plotly_chart(_style_fig(fig, 440), use_container_width=True)

        st.subheader("Category → subcategory breakdown")
        known = df[df["category"].str.lower() != "unknown"]
        if not known.empty:
            grp = known.groupby(["category", "subcategory"]).size().reset_index(name="count")
            fig = px.treemap(grp, path=["category", "subcategory"], values="count",
                             color="count", color_continuous_scale="Tealgrn")
            st.plotly_chart(_style_fig(fig, 460), use_container_width=True)
        else:
            st.caption("No subcategory data available.")

        if result is not None:
            import plotly.graph_objects as go

            g = go.Figure(go.Indicator(
                mode="gauge+number", value=result.empty_pct * 100,
                title={"text": "Empty shelf space (%)"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "#e5484d" if result.empty_pct >= 0.55
                               else "#f5a623" if result.empty_pct >= 0.25 else "#30a46c"},
                       "steps": [{"range": [0, 25], "color": "#14532d"},
                                 {"range": [25, 55], "color": "#78350f"},
                                 {"range": [55, 100], "color": "#7f1d1d"}]},
            ))
            st.plotly_chart(_style_fig(g, 300), use_container_width=True)

with tab_bi:
    st.subheader("Ask about the inventory")
    st.caption(
        "Natural-language questions over your saved inventory. "
        + (f"Using Ollama ({bi_engine.OLLAMA_MODEL})." if llm_on
           else "Rule-based engine (install Ollama for free-form answers).")
    )
    items_df = db.get_items_df()
    scans_df = db.get_scans_df()
    if items_df.empty and records:
        items_df = pd.DataFrame(records)

    if items_df.empty:
        st.info("No inventory yet. Analyze a shelf image first (enable 'Save scan').")
    else:
        cols = st.columns(4)
        for i, sug in enumerate(bi_engine.SUGGESTED_QUESTIONS[:8]):
            if cols[i % 4].button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state["bi_q"] = sug

        q = st.text_input("Your question", value=st.session_state.get("bi_q", ""),
                          placeholder="e.g. How many soft drinks are on the shelf?")
        if q:
            ans = bi_engine.answer(q, items_df, scans_df, use_llm=llm_on)
            st.markdown(f"> {ans.text}")
            st.caption(f"source: {ans.source}")
            if ans.table is not None and not ans.table.empty:
                tc1, tc2 = st.columns(2)
                tc1.dataframe(ans.table, hide_index=True, use_container_width=True)
                label_cols = [c for c in ans.table.columns if ans.table[c].dtype == object]
                num_cols = [c for c in ans.table.columns if ans.table[c].dtype != object]
                if label_cols and num_cols:
                    fig = px.bar(ans.table, x=num_cols[0], y=label_cols[0],
                                 orientation="h", color_discrete_sequence=PALETTE)
                    fig.update_layout(showlegend=False)
                    tc2.plotly_chart(_style_fig(fig, 320), use_container_width=True)

with tab_history:
    scans_df = db.get_scans_df()
    if scans_df.empty:
        st.info("No saved scans yet. Enable 'Save scan to inventory history' and analyze an image.")
    else:
        st.subheader("Scans over time")
        s = scans_df.sort_values("id")
        fig = px.line(s, x="ts", y="num_items", markers=True,
                      labels={"ts": "time", "num_items": "products detected"})
        st.plotly_chart(_style_fig(fig, 320), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Empty space per scan")
            fig = px.bar(s, x="id", y="empty_pct", color_discrete_sequence=["#f5a623"],
                         labels={"id": "scan", "empty_pct": "empty fraction"})
            st.plotly_chart(_style_fig(fig, 300), use_container_width=True)
        with c2:
            st.subheader("Aggregate category mix (all scans)")
            cc = bi_engine.category_counts(db.get_items_df())
            if not cc.empty:
                ccdf = cc.head(12).rename_axis("category").reset_index(name="count")
                fig = px.bar(ccdf, x="count", y="category", orientation="h",
                             color="category", color_discrete_sequence=PALETTE)
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
                st.plotly_chart(_style_fig(fig, 300), use_container_width=True)

        st.subheader("Scan log")
        st.dataframe(scans_df, hide_index=True, use_container_width=True)
