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
import html
from datetime import datetime
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
from PIL import Image

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

from backend import inventory_db as db
from bi_interface import bi_engine
from planogram import compliance as planogram_compliance
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
      .section-panel {
          border: 1px solid #334155; background: #0f172a; border-radius: 8px;
          padding: 14px 16px; margin: 8px 0 14px 0;
      }
      .section-title { color: #f8fafc; font-weight: 700; font-size: 1rem; margin-bottom: 2px; }
      .section-subtitle { color: #94a3b8; font-size: .82rem; }
      .status-badge {
          display: inline-block; border-radius: 999px; padding: 4px 10px;
          font-size: .76rem; font-weight: 700; border: 1px solid #475569;
          background: #111827; color: #e5e7eb;
      }
      .status-good { border-color: #15803d; color: #bbf7d0; background: #052e16; }
      .status-warn { border-color: #b45309; color: #fde68a; background: #451a03; }
      .status-bad { border-color: #b91c1c; color: #fecaca; background: #450a0a; }
      .checkout-header {
          color: #94a3b8; font-size: .74rem; font-weight: 700; text-transform: uppercase;
          border-bottom: 1px solid #334155; padding-bottom: 6px; margin-bottom: 4px;
      }
      .checkout-cell { color: #e5e7eb; font-size: .82rem; line-height: 1.25; overflow-wrap: anywhere; }
      .checkout-muted { color: #94a3b8; font-size: .76rem; overflow-wrap: anywhere; }
      .checkout-qty {
          text-align:center; padding-top:0.35rem; font-weight:800; color:#f8fafc;
      }
      .checkout-hero {
          border: 1px solid #334155; border-radius: 8px; padding: 14px 16px;
          background: linear-gradient(135deg, #0f172a 0%, #111827 58%, #052e2b 100%);
          margin: 4px 0 14px 0;
      }
      .checkout-hero-title { color: #ffffff; font-size: 1.15rem; font-weight: 900; margin-bottom: 3px; }
      .checkout-hero-subtitle { color: #cbd5e1; font-size: .86rem; }
      .checkout-card-title { color: #f8fafc; font-size: 1rem; font-weight: 900; margin-bottom: 4px; }
      .checkout-card-meta { color: #cbd5e1; font-size: .78rem; line-height: 1.35; overflow-wrap: anywhere; }
      .checkout-chip {
          display: inline-block; border: 1px solid #475569; border-radius: 999px;
          padding: 3px 8px; margin: 4px 5px 0 0; color: #e2e8f0;
          background: #0f172a; font-size: .72rem; font-weight: 700;
      }
      .checkout-chip.good { border-color: #22c55e; color: #bbf7d0; background: #052e16; }
      .checkout-chip.warn { border-color: #f59e0b; color: #fde68a; background: #451a03; }
      .checkout-qty-badge {
          border: 1px solid #334155; border-radius: 8px; background: #0f172a;
          color: #ffffff; text-align: center; padding: 8px 0; font-weight: 900;
      }
      .checkout-summary {
          border: 1px solid #334155; border-radius: 8px; padding: 14px;
          background: #0f172a;
      }
      .checkout-total { color: #ffffff; font-size: 2rem; line-height: 1.05; font-weight: 900; }
      .checkout-total-label { color: #cbd5e1; font-size: .8rem; text-transform: uppercase; font-weight: 800; }
      .checkout-empty {
          border: 1px dashed #475569; border-radius: 8px; padding: 14px;
          color: #cbd5e1; background: #111827; text-align: center;
      }
      .audit-context {
          border: 1px solid #334155; border-radius: 8px; padding: 10px 12px;
          background: #111827; min-height: 78px;
      }
      .audit-context-label {
          color: #cbd5e1; font-size: .7rem; text-transform: uppercase; font-weight: 800;
      }
      .audit-context-value { color: #f8fafc; font-size: .92rem; font-weight: 700; margin-top: 3px; }
      .audit-config {
          border: 1px solid #334155; background: #0f172a; border-radius: 8px;
          padding: 12px 14px; margin: 8px 0 12px 0;
      }
      .config-line { color: #e2e8f0; font-size: .9rem; margin-bottom: 4px; }
      .config-label { color: #94a3b8; font-weight: 700; }
      .layout-row {
          border: 1px solid #334155; background: #111827; border-radius: 8px;
          padding: 10px 12px; margin-bottom: 8px;
      }
      .layout-row-title { color: #f8fafc; font-weight: 800; font-size: .86rem; }
      .layout-row-cats { color: #cbd5e1; font-size: .86rem; overflow-wrap: anywhere; }
      .kpi-card {
          border-radius: 8px; padding: 13px 14px; min-height: 104px;
          border: 1px solid #334155; background: #111827;
      }
      .kpi-label { color: #e2e8f0; font-size: .72rem; text-transform: uppercase; font-weight: 800; }
      .kpi-value { color: #ffffff; font-size: 1.8rem; line-height: 1.15; font-weight: 900; }
      .kpi-subtext { color: #cbd5e1; font-size: .78rem; margin-top: 5px; }
      .kpi-good { background: linear-gradient(135deg, #14532d, #052e16); border-color: #22c55e; }
      .kpi-warn { background: linear-gradient(135deg, #78350f, #451a03); border-color: #f59e0b; }
      .kpi-bad { background: linear-gradient(135deg, #7f1d1d, #450a0a); border-color: #ef4444; }
      .kpi-purple { background: linear-gradient(135deg, #4c1d95, #2e1065); border-color: #a855f7; }
      .audit-banner {
          border-radius: 8px; padding: 16px 18px; margin: 14px 0;
          border: 1px solid #334155; background: #111827;
      }
      .audit-banner.good { background: #052e16; border-color: #22c55e; }
      .audit-banner.warn { background: #451a03; border-color: #f59e0b; }
      .audit-banner.bad { background: #450a0a; border-color: #ef4444; }
      .audit-banner-title { color: #ffffff; font-size: 1.16rem; font-weight: 900; margin-bottom: 4px; }
      .audit-banner-detail { color: #f8fafc; font-size: .9rem; }
      .legend {
          color: #cbd5e1; font-size: .82rem; border: 1px solid #334155;
          border-radius: 8px; padding: 10px 12px; background: #0f172a;
      }
      .legend-dot {
          display: inline-block; width: 10px; height: 10px; border-radius: 999px;
          margin-right: 6px; vertical-align: middle;
      }
      .rec-card {
          border: 1px solid #334155; border-left: 4px solid #f59e0b;
          border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;
          background: #111827; color: #f8fafc;
      }
      .rec-meta { color: #cbd5e1; font-size: .78rem; margin-top: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

PALETTE = px.colors.qualitative.Set3


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


def _section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-panel">
          <div class="section-title">{html.escape(title)}</div>
          <div class="section-subtitle">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _badge(text: str, kind: str = "") -> str:
    classes = "status-badge" + (f" status-{kind}" if kind else "")
    return f"<span class='{classes}'>{html.escape(text)}</span>"


def _compliance_kind(score_pct: float) -> str:
    if score_pct >= 80:
        return "good"
    if score_pct >= 55:
        return "warn"
    return "bad"


def _kpi_card(label: str, value: str, kind: str, subtext: str = "") -> str:
    return f"""
    <div class="kpi-card kpi-{kind}">
      <div class="kpi-label">{html.escape(label)}</div>
      <div class="kpi-value">{html.escape(value)}</div>
      <div class="kpi-subtext">{html.escape(subtext)}</div>
    </div>
    """


def _audit_context_card(label: str, value: str) -> str:
    return f"""
    <div class="audit-context">
      <div class="audit-context-label">{html.escape(label)}</div>
      <div class="audit-context-value">{html.escape(value or "-")}</div>
    </div>
    """


def _format_planogram_rows(planogram_rows: list[list[str]]) -> list[dict]:
    return [
        {
            "row": f"Row {row_number}",
            "categories": " | ".join(categories),
        }
        for row_number, categories in enumerate(planogram_rows, start=1)
    ]


def _render_expected_layout(planogram_rows: list[list[str]]) -> None:
    for row in _format_planogram_rows(planogram_rows):
        st.markdown(
            f"""
            <div class="layout-row">
              <div class="layout-row-title">{html.escape(row["row"])}</div>
              <div class="layout-row-cats">{html.escape(row["categories"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _row_action_summary(compliance_result) -> pd.DataFrame:
    if compliance_result.row_summary.empty:
        return pd.DataFrame()

    missing_counts = (
        compliance_result.missing.groupby("row")["missing_count"].sum().to_dict()
        if not compliance_result.missing.empty and "row" in compliance_result.missing
        else {}
    )
    misplaced_counts = (
        compliance_result.misplaced.groupby("row").size().to_dict()
        if not compliance_result.misplaced.empty and "row" in compliance_result.misplaced
        else {}
    )

    rows = []
    for _, row in compliance_result.row_summary.iterrows():
        row_number = int(row.get("row", 0))
        expected_count = int(row.get("expected_count", 0) or 0)
        matched = int(row.get("matched", 0) or 0)
        missing = int(missing_counts.get(row_number, 0))
        misplaced = int(misplaced_counts.get(row_number, 0))
        row_score = (matched / expected_count * 100) if expected_count else 0
        rows.append({
            "Row": f"Row {row_number}",
            "Compliance": f"{row_score:.0f}%",
            "Correct": matched,
            "Misplaced": misplaced,
            "Missing": missing,
            "Action": "View issues" if misplaced or missing else "OK",
        })
    return pd.DataFrame(rows)


def _planogram_recommendations(compliance_result) -> pd.DataFrame:
    rows = []
    if not compliance_result.misplaced.empty:
        for _, row in compliance_result.misplaced.iterrows():
            category = str(row.get("category", "unknown"))
            current_row = row.get("row", "-")
            expected_row = row.get("expected_row", "-")
            expected_category = str(row.get("expected_category") or category)
            rows.append({
                "issue_type": "Misplaced",
                "row": current_row,
                "expected_category": expected_category,
                "observed_category": category,
                "detected_position": f"Row {current_row}, crop #{row.get('crop_id', '-')}",
                "recommended_action": (
                    f"Move or review {category} in Row {current_row}; "
                    f"this zone winner is {expected_category}."
                ),
                "model_confidence": row.get("score", ""),
            })
    if not compliance_result.missing.empty:
        for _, row in compliance_result.missing.iterrows():
            category = str(row.get("category", "unknown"))
            rows.append({
                "issue_type": "Missing",
                "row": row.get("row", "-"),
                "expected_category": category,
                "observed_category": "",
                "detected_position": "Not detected",
                "recommended_action": f"Restock or place {category} on Row {row.get('row', '-')}.",
                "model_confidence": "",
            })
    if not compliance_result.unexpected.empty:
        for _, row in compliance_result.unexpected.iterrows():
            category = str(row.get("category", "unknown"))
            rows.append({
                "issue_type": "Unexpected",
                "row": row.get("row", "-"),
                "expected_category": "",
                "observed_category": category,
                "detected_position": f"Row {row.get('row', '-')}, crop #{row.get('crop_id', '-')}",
                "recommended_action": f"Review whether {category} belongs on this shelf.",
                "model_confidence": row.get("score", ""),
            })
    return pd.DataFrame(rows)


def _exception_report(compliance_result, row_summary: pd.DataFrame, audit_context: dict) -> pd.DataFrame:
    recommendations = _planogram_recommendations(compliance_result)
    if recommendations.empty:
        recommendations = pd.DataFrame([{
            "issue_type": "No exceptions",
            "row": "",
            "expected_category": "",
            "observed_category": "",
            "detected_position": "",
            "recommended_action": "Shelf is compliant with the selected planogram.",
            "model_confidence": "",
        }])
    for key, value in audit_context.items():
        recommendations[key] = value
    if not row_summary.empty:
        recommendations["overall_rows"] = len(row_summary)
    return recommendations


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


def _cart_label(record: dict) -> str:
    product_name = str(record.get("product_name") or "").strip()
    barcode = str(record.get("barcode") or "").strip()
    category = str(record.get("category") or "unknown").strip()
    subcategory = str(record.get("subcategory") or "unknown").strip()
    label = product_name or f"{category} / {subcategory}"
    return f"{label} [{barcode}]" if barcode else label


def _checkout_title(record: dict) -> str:
    product_name = str(record.get("product_name") or "").strip()
    sku_text = str(record.get("sku_text") or "").strip()
    category = str(record.get("category") or "unknown").strip()
    subcategory = str(record.get("subcategory") or "unknown").strip()
    return product_name or sku_text or f"{category} / {subcategory}"


def _checkout_crop(record: dict):
    source_image = st.session_state.get("source_image")
    box = record.get("box")
    if source_image is None or not box:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in box]
        return source_image.crop((x1, y1, x2, y2))
    except Exception:
        return None


def _chip(text: str, kind: str = "") -> str:
    classes = "checkout-chip" + (f" {kind}" if kind else "")
    return f"<span class='{classes}'>{html.escape(text)}</span>"


def _current_records_by_cart_key(records: list[dict]) -> dict:
    valid_cart_keys = {db.cart_key(record) for record in records}
    st.session_state["cart"] = {
        key: qty
        for key, qty in st.session_state.get("cart", {}).items()
        if key in valid_cart_keys and int(qty) > 0
    }
    return {db.cart_key(record): record for record in records}


def _show_checkout_notice() -> None:
    checkout_notice = st.session_state.pop("checkout_notice", None)
    if not checkout_notice:
        return
    notice_kind, notice_text = checkout_notice
    if notice_kind == "warning":
        st.warning(notice_text)
    else:
        st.success(notice_text)


def _adjust_cart_quantity(key: str, delta: int) -> None:
    cart = st.session_state.setdefault("cart", {})
    cart[key] = max(0, int(cart.get(key, 0)) + delta)
    if cart[key] == 0:
        cart.pop(key, None)


def _remove_checked_out_crops(checked_out_keys: set[str]) -> None:
    if not checked_out_keys:
        return

    checked_out_boxes = [
        record["box"] for record in st.session_state.get("records", [])
        if db.cart_key(record) in checked_out_keys and record.get("box")
    ]
    st.session_state.setdefault("checked_out_boxes", []).extend(checked_out_boxes)
    remaining_records = [
        record for record in st.session_state.get("records", [])
        if db.cart_key(record) not in checked_out_keys
    ]
    st.session_state["records"] = remaining_records

    result = st.session_state.get("result")
    if result is None or not hasattr(result, "items"):
        return

    result.items = [
        item for item in result.items
        if db.cart_key(item) not in checked_out_keys
    ]
    result.num_items = len(result.items)
    known_categories = [
        item.get("category", "")
        for item in result.items
        if str(item.get("category", "")).lower() != "unknown"
    ]
    result.distinct_categories = len(set(known_categories))
    result.review_count = sum(
        1 for item in result.items
        if str(item.get("category", "")).lower() == "unknown"
        or float(item.get("score") or 0.0) <= 0
    )
    source_image = st.session_state.get("source_image")
    if source_image is not None:
        result.annotated_image = pipeline.redraw_annotated_image(
            source_image,
            result.items,
            hidden_boxes=st.session_state.get("checked_out_boxes", []),
        )
    st.session_state["result"] = result


def _render_checkout_cart(records: list[dict]) -> None:
    records_by_cart_key = _current_records_by_cart_key(records)
    _show_checkout_notice()

    st.markdown(
        """
        <div class="checkout-hero">
          <div class="checkout-hero-title">Customer checkout basket</div>
          <div class="checkout-hero-subtitle">
            Add units from detected products, review the basket, then checkout to decrement inventory.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not records:
        st.info("Analyze a shelf image first, then select checkout quantities here.")
        return

    selected_cart = {
        key: int(qty)
        for key, qty in st.session_state.get("cart", {}).items()
        if int(qty) > 0
    }
    selected_total = sum(selected_cart.values())
    active_units = selected_total
    st.session_state["cart"] = selected_cart

    c1, c2, c3 = st.columns(3)
    c1.markdown(_kpi_card("Detected SKUs", str(len(records_by_cart_key)), "purple", "Available from scan"), unsafe_allow_html=True)
    c2.markdown(_kpi_card("Basket Units", str(active_units), "good", "Ready for checkout"), unsafe_allow_html=True)
    c3.markdown(_kpi_card("On Shelf", str(len(records)), "warn", "After prior removals"), unsafe_allow_html=True)

    list_col, basket_col = st.columns([2.2, 1], gap="large")
    with list_col:
        view_filter = st.radio(
            "Products",
            ["All detected", "Selected only", "Needs review"],
            horizontal=True,
            label_visibility="collapsed",
        )
        visible_items = list(records_by_cart_key.items())
        if view_filter == "Selected only":
            visible_items = [(key, record) for key, record in visible_items if selected_cart.get(key, 0) > 0]
        elif view_filter == "Needs review":
            visible_items = [
                (key, record)
                for key, record in visible_items
                if str(record.get("category", "")).lower() == "unknown"
                or int(record.get("sku_needs_review") or 0) == 1
            ]

        if not visible_items:
            st.markdown(
                "<div class='checkout-empty'>No products match this filter.</div>",
                unsafe_allow_html=True,
            )

        for key, record in visible_items:
            qty = int(st.session_state.get("cart", {}).get(key, 0))
            title = _checkout_title(record)
            brand = str(record.get("brand") or "").strip()
            category = str(record.get("category") or "unknown")
            subcategory = str(record.get("subcategory") or "unknown")
            sku = str(record.get("sku_text") or "").strip()
            barcode = str(record.get("barcode") or "").strip()
            confidence = record.get("sku_confidence") or record.get("score") or ""
            crop = _checkout_crop(record)

            with st.container(border=True):
                item_cols = st.columns([0.9, 3.4, 1.15])
                with item_cols[0]:
                    if crop is not None:
                        st.image(crop, use_container_width=True)
                    else:
                        st.markdown(
                            f"<div class='checkout-empty'>Crop #{html.escape(str(record.get('crop_id')))}</div>",
                            unsafe_allow_html=True,
                        )
                with item_cols[1]:
                    chips = [
                        _chip(f"Crop #{record.get('crop_id')}", "good" if qty else ""),
                        _chip(category),
                        _chip(subcategory),
                    ]
                    if brand:
                        chips.append(_chip(f"Brand: {brand}"))
                    if sku:
                        chips.append(_chip(f"SKU: {sku}"))
                    if barcode:
                        chips.append(_chip(f"Barcode: {barcode}"))
                    if confidence != "":
                        chips.append(_chip(f"Confidence: {confidence}"))
                    st.markdown(
                        f"""
                        <div class="checkout-card-title">{html.escape(title)}</div>
                        <div class="checkout-card-meta">{html.escape(_cart_label(record))}</div>
                        <div>{''.join(chips)}</div>
                        """,
                        unsafe_allow_html=True,
                    )
                with item_cols[2]:
                    qty_cols = st.columns([1, 1, 1])
                    if qty_cols[0].button("-", key=f"checkout_minus_{key}", use_container_width=True, disabled=qty <= 0):
                        _adjust_cart_quantity(key, -1)
                        st.rerun()
                    qty_cols[1].markdown(
                        f"<div class='checkout-qty-badge'>{qty}</div>",
                        unsafe_allow_html=True,
                    )
                    if qty_cols[2].button("+", key=f"checkout_plus_{key}", use_container_width=True):
                        _adjust_cart_quantity(key, 1)
                        st.rerun()
                    if qty == 0:
                        if st.button("Add", key=f"checkout_add_{key}", type="primary", use_container_width=True):
                            _adjust_cart_quantity(key, 1)
                            st.rerun()
                    else:
                        if st.button("Remove", key=f"checkout_remove_{key}", use_container_width=True):
                            st.session_state["cart"].pop(key, None)
                            st.rerun()

    with basket_col:
        st.markdown(
            f"""
            <div class="checkout-summary">
              <div class="checkout-total-label">Basket total</div>
              <div class="checkout-total">{selected_total}</div>
              <div class="checkout-card-meta">unit{'s' if selected_total != 1 else ''} selected for checkout</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if selected_total <= 0:
            st.markdown(
                "<div class='checkout-empty'>Use + or Add on any product to start checkout.</div>",
                unsafe_allow_html=True,
            )
            return

        selected_rows = []
        for key, qty in selected_cart.items():
            record = records_by_cart_key.get(key)
            if record:
                selected_rows.append({
                    "Item": _checkout_title(record),
                    "Qty": qty,
                    "Category": record.get("category", "unknown"),
                })
        st.dataframe(pd.DataFrame(selected_rows), hide_index=True, use_container_width=True, height=220)

        clear_col, checkout_col = st.columns([1, 1.4])
        if clear_col.button("Clear", use_container_width=True):
            st.session_state["cart"] = {}
            st.rerun()

        if checkout_col.button(f"Checkout {selected_total}", type="primary", use_container_width=True):
            summary = db.checkout_items(selected_cart, records)
            st.session_state["cart"] = {}
            if summary["checked_out"]:
                _remove_checked_out_crops(set(selected_cart.keys()))
                st.session_state["checkout_notice"] = (
                    "success",
                    f"Checked out {summary['checked_out']} item(s). Inventory updated.",
                )
            if summary["short"]:
                st.session_state["checkout_notice"] = (
                    "warning",
                    "Some selected quantities were higher than available stock.",
                )
            st.rerun()


st.title("🛒 Smart Shelf Analytics & BI")
llm_on = bi_engine.ollama_available()
st.markdown(
    "<span class='muted'>YOLO detection · SWIN + FAISS retrieval classification · "
    "inventory analytics + natural-language BI</span>"
    f"<span class='pill'>{'LLM: Ollama ✓' if llm_on else 'BI: rule-based'}</span>",
    unsafe_allow_html=True,
)

db.init_db()
st.session_state.setdefault("cart", {})

with st.sidebar:
    st.header("① Upload & detect")
    if pipeline.classifier_ready():
        st.success("SWIN+FAISS index loaded ✓")
    else:
        st.error("Classifier not ready — check retrieval/assets (see retrieval/README.md).")

    uploaded = st.file_uploader("Shelf image", type=["jpg", "jpeg", "png", "bmp"])
    conf = st.slider("YOLO confidence", 0.05, 0.9, 0.25, 0.05)
    max_crops = st.slider("Max products to classify (0 = all)", 0, 300, 60, 10,
                          help="Cap for speed on CPU. 0 classifies every detected box.")
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
                    disabled=uploaded is None)

    st.divider()
    st.caption("Inventory history")
    s = db.stats()
    st.write(f"Scans: **{s['total_scans']}** · Items: **{s['total_items']}** · "
             f"Categories: **{s['distinct_categories']}** · Stock: **{s['stock_units']}**")
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
    st.session_state["source_image"] = image
    st.session_state["image_name"] = uploaded.name
    st.session_state["checked_out_boxes"] = []
    st.session_state["cart"] = {}
    if save_to_db:
        scan_id = db.save_scan(result, uploaded.name, records)
        st.session_state["last_scan_id"] = scan_id
        st.toast(f"Saved scan #{scan_id} to inventory")

result = st.session_state.get("result")
records = st.session_state.get("records", [])

tab_analyze, tab_checkout, tab_planogram, tab_analytics, tab_bi, tab_history = st.tabs(
    ["Detection", "Checkout", "Planogram", "Analytics", "Business Intelligence", "Inventory History"]
)

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

with tab_checkout:
    st.subheader("Checkout")
    _render_checkout_cart(records)

with tab_planogram:
    st.subheader("Planogram Compliance")
    shelf_type = result.shelf_type if result is not None else ""
    template_name, shelf_type_planogram = planogram_compliance.planogram_for_shelf_type(shelf_type)

    inferred_zone_rows = planogram_compliance.infer_detected_row_count(records) if records else 0

    config_left, config_mid, config_right = st.columns([1.5, 1, 1])
    with config_left:
        audit_driver = st.selectbox(
            "Compliance driver",
            ["Zone winner from current shelf", "Shelf type template", "Manual planogram"],
            index=0,
            disabled=result is None,
            help="Zone winner derives the expected row category from the dominant detected category in each shelf zone.",
        )
    with config_mid:
        auto_detect_rows = st.checkbox(
            "Auto-detect rows",
            value=True,
            disabled=result is None or audit_driver != "Zone winner from current shelf",
            help="Estimate shelf rows from the vertical spacing of detected product boxes.",
        )
    edit_planogram = False
    with config_right:
        if audit_driver == "Zone winner from current shelf" and not auto_detect_rows:
            zone_row_count = st.number_input("Shelf rows", 1, 12, max(1, inferred_zone_rows or 3), 1)
        else:
            zone_row_count = inferred_zone_rows or 1
            edit_planogram = st.checkbox(
                "Edit planogram",
                value=False,
                disabled=result is None or audit_driver == "Zone winner from current shelf",
            )

    zone_template_name = "Zone winner shelf"
    zone_planogram_text = ""
    zone_summary = pd.DataFrame()
    if records:
        zone_template_name, zone_planogram_text, zone_summary = planogram_compliance.zone_winner_planogram(
            records,
            zone_row_count if zone_row_count else None,
        )

    active_template_name = (
        zone_template_name if audit_driver == "Zone winner from current shelf" else template_name
    )
    active_driver = (
        f"Detected zone winners ({zone_row_count} row{'s' if zone_row_count != 1 else ''})"
        if audit_driver == "Zone winner from current shelf"
        else shelf_type or "Manual"
    )
    audit_context = {
        "store": "Demo store",
        "aisle": active_template_name,
        "shelf": active_driver,
        "audit_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "image_name": st.session_state.get("image_name", "No image analyzed"),
    }

    context_cols = st.columns(5)
    for col, (label, value) in zip(context_cols, audit_context.items()):
        col.markdown(_audit_context_card(label.replace("_", " ").title(), value), unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="audit-config">
          <div class="config-line"><span class="config-label">Template:</span> {html.escape(active_template_name)}</div>
          <div class="config-line"><span class="config-label">Driver:</span> {html.escape(active_driver)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if audit_driver == "Zone winner from current shelf":
        planogram_text = zone_planogram_text
        if not zone_summary.empty:
            display_zone_summary = zone_summary.copy()
            display_zone_summary["winner_share"] = (
                display_zone_summary["winner_share"] * 100
            ).round(0).astype(int).astype(str) + "%"
            st.markdown("**Zone winners**")
            st.dataframe(
                display_zone_summary,
                hide_index=True,
                use_container_width=True,
                height=170,
            )
    elif audit_driver == "Shelf type template" and result is not None:
        template_key = planogram_compliance.normalize_text(template_name).replace(" ", "_")
        state_key = f"planogram_auto_{template_key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = shelf_type_planogram
        if edit_planogram:
            planogram_text = st.text_area(
                "Expected shelf layout",
                height=150,
                help="Use one row per shelf row. Example: Row 1: Bottled Water, Carbonated Soft Drinks",
                key=state_key,
            )
        else:
            planogram_text = st.session_state[state_key]
    else:
        if "planogram_manual" not in st.session_state:
            st.session_state["planogram_manual"] = planogram_compliance.DEFAULT_PLANOGRAM_TEXT
        if edit_planogram or result is None:
            planogram_text = st.text_area(
                "Expected shelf layout",
                height=150,
                help="Use one row per shelf row. Example: Row 1: Bottled Water, Carbonated Soft Drinks",
                key="planogram_manual",
            )
        else:
            planogram_text = st.session_state["planogram_manual"]

    planogram_rows = planogram_compliance.parse_planogram(planogram_text)
    if planogram_rows and not edit_planogram:
        st.markdown("**Expected layout**")
        _render_expected_layout(planogram_rows)

    if not records:
        st.info("Analyze a shelf image first to calculate compliance.")
    elif not planogram_rows:
        st.warning("Add at least one planogram row.")
    else:
        if audit_driver == "Zone winner from current shelf":
            compliance_result = planogram_compliance.evaluate_zone_winner(records, zone_row_count)
        else:
            compliance_result = planogram_compliance.evaluate(records, planogram_rows)
        score_pct = compliance_result.score * 100
        score_kind = _compliance_kind(score_pct)
        missing_count = int(compliance_result.missing.get("missing_count", pd.Series(dtype=int)).sum())
        misplaced_count = len(compliance_result.misplaced)
        unexpected_count = len(compliance_result.unexpected)
        previous_score = st.session_state.get("previous_planogram_score")
        trend_text = "Baseline audit"
        if previous_score is not None:
            delta = score_pct - float(previous_score)
            direction = "up" if delta >= 0 else "down"
            trend_text = f"{direction} {abs(delta):.0f}% vs previous audit"

        if score_kind == "good":
            banner_title = f"On track - {score_pct:.0f}% compliant"
            banner_detail = "Shelf zones are mostly consistent with their dominant categories."
        elif score_kind == "warn":
            banner_title = f"Needs attention - {score_pct:.0f}% compliant"
            if audit_driver == "Zone winner from current shelf":
                banner_detail = f"{misplaced_count} items do not match their shelf-zone winner."
            else:
                banner_detail = f"{misplaced_count} items are misplaced and {missing_count} expected products are missing."
        else:
            banner_title = f"Critical shelf issue - {score_pct:.0f}% compliant"
            if audit_driver == "Zone winner from current shelf":
                banner_detail = f"{misplaced_count} items do not match their shelf-zone winner."
            else:
                banner_detail = f"{misplaced_count} items are misplaced and {missing_count} expected products are missing."

        st.markdown(
            f"""
            <div class="audit-banner {score_kind}">
              <div class="audit-banner-title">{html.escape(banner_title)}</div>
              <div class="audit-banner-detail">{html.escape(banner_detail)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        row_summary = _row_action_summary(compliance_result)
        report_df = _exception_report(compliance_result, row_summary, audit_context)
        action_cols = st.columns([1, 1, 2])
        with action_cols[0]:
            if st.button("Review exceptions", use_container_width=True):
                st.session_state["planogram_issue_filter"] = "All exceptions"
        with action_cols[1]:
            st.download_button(
                "Export report CSV",
                report_df.to_csv(index=False).encode(),
                file_name="planogram_audit_report.csv",
                use_container_width=True,
            )
        with action_cols[2]:
            st.caption("Re-run audit from the Detection tab after changing model inputs or uploading a new image.")

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.markdown(_kpi_card("Compliance", f"{score_pct:.0f}%", score_kind, trend_text), unsafe_allow_html=True)
        p2.markdown(_kpi_card("Matched", str(len(compliance_result.matched)), "good", "Correct shelf row"), unsafe_allow_html=True)
        misplaced_subtext = "Review zone exceptions" if audit_driver == "Zone winner from current shelf" else "Move to expected row"
        p3.markdown(_kpi_card("Misplaced", str(misplaced_count), "warn", misplaced_subtext), unsafe_allow_html=True)
        missing_subtext = "Not used in zone mode" if audit_driver == "Zone winner from current shelf" else "Expected but absent"
        p4.markdown(_kpi_card("Missing", str(missing_count), "bad", missing_subtext), unsafe_allow_html=True)
        p5.markdown(_kpi_card("Unexpected", str(unexpected_count), "purple", "Not in template"), unsafe_allow_html=True)

        st.markdown(
            """
            <div class="legend">
              <span class="legend-dot" style="background:#22c55e"></span>Green: >=80% compliant
              &nbsp;&nbsp;<span class="legend-dot" style="background:#f59e0b"></span>Amber: 55-79%
              &nbsp;&nbsp;<span class="legend-dot" style="background:#ef4444"></span>Red: below 55%
              &nbsp;&nbsp;<span class="legend-dot" style="background:#a855f7"></span>Purple: unexpected items
            </div>
            """,
            unsafe_allow_html=True,
        )

        low_confidence = pd.DataFrame(records)
        if "score" in low_confidence.columns:
            low_confidence = low_confidence[pd.to_numeric(low_confidence["score"], errors="coerce").fillna(1) < 0.35]
        else:
            low_confidence = pd.DataFrame()
        if not low_confidence.empty:
            st.warning(
                f"{len(low_confidence)} detection(s) have low model confidence. "
                "Review the evidence before using this audit for store execution."
            )

        st.markdown("**Visual shelf comparison**")
        st.image(result.annotated_image, caption="Detected shelf with current checkout removals applied", use_container_width=True)

        st.markdown("**Row-level actions**")
        if row_summary.empty:
            st.caption("No row summary available.")
        else:
            st.dataframe(row_summary, hide_index=True, use_container_width=True, height=180)

        recommendations = _planogram_recommendations(compliance_result)
        if recommendations.empty:
            st.success("No corrective actions needed for the selected planogram.")
        else:
            st.markdown("**Recommended actions**")
            for _, rec in recommendations.head(6).iterrows():
                st.markdown(
                    f"""
                    <div class="rec-card">
                      <strong>{html.escape(str(rec["recommended_action"]))}</strong>
                      <div class="rec-meta">
                        {html.escape(str(rec["issue_type"]))} | {html.escape(str(rec["detected_position"]))}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        selected_filter = st.radio(
            "Detailed evidence",
            ["All exceptions", "Misplaced", "Missing", "Unexpected", "Matched", "Low confidence"],
            index=["All exceptions", "Misplaced", "Missing", "Unexpected", "Matched", "Low confidence"].index(
                st.session_state.get("planogram_issue_filter", "All exceptions")
            ),
            horizontal=True,
        )
        st.session_state["planogram_issue_filter"] = selected_filter

        if selected_filter == "All exceptions":
            evidence_df = recommendations
        elif selected_filter == "Misplaced":
            evidence_df = compliance_result.misplaced
        elif selected_filter == "Missing":
            evidence_df = compliance_result.missing
        elif selected_filter == "Unexpected":
            evidence_df = compliance_result.unexpected
        elif selected_filter == "Matched":
            evidence_df = compliance_result.matched
        else:
            evidence_df = low_confidence

        if evidence_df.empty:
            st.caption("No rows for this filter.")
        else:
            st.dataframe(evidence_df, hide_index=True, use_container_width=True, height=320)

        for row_number in range(1, len(planogram_rows) + 1):
            row_recs = recommendations[recommendations["row"].astype(str) == str(row_number)] if not recommendations.empty else pd.DataFrame()
            with st.expander(f"Row {row_number} issue details", expanded=False):
                expected_text = " | ".join(planogram_rows[row_number - 1])
                st.markdown(f"**Expected categories:** {expected_text}")
                if row_recs.empty:
                    st.caption("No row-level exceptions.")
                else:
                    st.dataframe(
                        row_recs[[
                            "expected_category",
                            "observed_category",
                            "detected_position",
                            "recommended_action",
                            "model_confidence",
                        ]],
                        hide_index=True,
                        use_container_width=True,
                    )

        st.session_state["previous_planogram_score"] = score_pct

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

        stock_df = db.get_stock_df()
        st.subheader("Current stock")
        if stock_df.empty:
            st.caption("No stock rows yet.")
        else:
            st.dataframe(stock_df, hide_index=True, use_container_width=True)
