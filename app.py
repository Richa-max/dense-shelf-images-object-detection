import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

import json
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import ImageDraw
from ultralytics import YOLO

from qwen_model import generate_qwen_sku_answer
from retail_product_resolver import resolve_retail_product, summarize_retail_decisions
from swin_faiss import load_swin_faiss_classifier


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg-a: #f8f7f2;
    --bg-b: #e9f2ea;
    --bg-c: #e4edf9;
    --accent: #0f766e;
    --accent-strong: #0b5f58;
    --ink: #12202f;
    --panel: rgba(255, 255, 255, 0.8);
    --panel-border: rgba(15, 118, 110, 0.16);
    --radius-xl: 18px;
}

.gradio-container {
    font-family: 'Space Grotesk', sans-serif;
    max-width: 1320px !important;
    margin: 0 auto;
    padding-top: 14px !important;
    color: var(--ink);
    background:
        radial-gradient(1200px 360px at 2% -5%, #d8f3e4 0%, transparent 62%),
        radial-gradient(1000px 340px at 96% -8%, #dce6ff 0%, transparent 60%),
        linear-gradient(170deg, var(--bg-a) 0%, var(--bg-b) 52%, var(--bg-c) 100%);
}

.hero {
    background:
        linear-gradient(140deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.04)),
        linear-gradient(135deg, #0f766e, #1d3557);
    border: 1px solid rgba(255, 255, 255, 0.28);
    border-radius: var(--radius-xl);
    padding: 20px 22px;
    margin-bottom: 16px;
    color: #ffffff;
    box-shadow: 0 16px 36px rgba(15, 118, 110, 0.24);
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    right: -70px;
    top: -70px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.28) 0%, rgba(255, 255, 255, 0.02) 70%);
}

.hero h1 {
    margin: 0 0 6px 0;
    font-family: 'Fraunces', serif;
    font-size: 2rem;
    line-height: 1.15;
    letter-spacing: 0.01em;
}

.hero p {
    margin: 0;
    opacity: 0.95;
    font-size: 1rem;
    max-width: 760px;
}

.toolbar {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    padding: 10px;
    backdrop-filter: blur(8px);
    box-shadow: 0 10px 24px rgba(16, 24, 40, 0.05);
}

.panel {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    box-shadow: 0 12px 26px rgba(16, 24, 40, 0.08);
    padding: 10px;
    backdrop-filter: blur(6px);
}

.panel h2,
.panel h3,
.panel h4,
.panel h5 {
    color: #12263f;
    margin-top: 0.2rem;
}

.section-title {
    margin: 10px 2px 6px 2px;
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    color: #10324d;
    letter-spacing: 0.01em;
}

.tab-banner {
    margin: 4px 2px 10px 2px;
    padding: 10px 12px;
    border-radius: 12px;
    background: linear-gradient(130deg, rgba(255, 255, 255, 0.9), rgba(220, 244, 238, 0.62));
    border: 1px solid rgba(15, 118, 110, 0.18);
    color: #174160;
    font-size: 0.94rem;
}

.tab-banner b {
    color: #0c5d58;
}

.gradio-container .tab-nav {
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(15, 118, 110, 0.15);
    border-radius: 14px;
    padding: 4px;
}

.gradio-container .tab-nav button {
    border-radius: 10px !important;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(120px, 1fr));
    gap: 12px;
}

.kpi-card {
    background: linear-gradient(160deg, #ffffff, #eef9f5);
    border: 1px solid #cfe8de;
    border-radius: 12px;
    padding: 12px;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.85);
}

.kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #5b6f83;
}

.kpi-value {
    font-family: 'Fraunces', serif;
    font-size: 1.45rem;
    color: #0f172a;
    line-height: 1.1;
}

.sku-table-wrap {
    overflow-x: auto;
}

.sku-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.93rem;
    border-radius: 10px;
    overflow: hidden;
}

.sku-table thead th {
    background: linear-gradient(180deg, #e8f6f1, #dff1ea);
    color: #0f172a;
    text-align: left;
    padding: 9px 10px;
    border-bottom: 1px solid #cce2d8;
    font-weight: 700;
}

.sku-table tbody td {
    padding: 9px 10px;
    border-bottom: 1px solid #e2ece7;
    vertical-align: top;
}

.sku-table tbody tr:nth-child(even) {
    background: rgba(15, 118, 110, 0.04);
}

.sku-table tbody tr:hover {
    background: rgba(20, 184, 166, 0.09);
}

.animate-in {
    animation: fade-slide-up 0.55s ease both;
}

.stagger-1 { animation-delay: 0.05s; }
.stagger-2 { animation-delay: 0.12s; }
.stagger-3 { animation-delay: 0.19s; }
.stagger-4 { animation-delay: 0.26s; }

@keyframes fade-slide-up {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

button.primary, .gr-button-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent-strong)) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 8px 18px rgba(15, 118, 110, 0.28);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}

button.secondary {
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    background: #f3fbf9 !important;
    transition: transform 0.18s ease, background 0.18s ease;
}

button.primary:hover,
.gr-button-primary:hover,
button.secondary:hover {
    transform: translateY(-1px);
}

button.secondary:hover {
    background: #e8f7f2 !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    border-radius: 10px !important;
    border: 1px solid #ccddd5 !important;
}

.gradio-container details summary {
    cursor: pointer;
    color: #0f5d58;
    font-weight: 600;
}

#sku-results {
    border-left: 4px solid var(--accent);
}

@media (max-width: 980px) {
    .hero h1 {
        font-size: 1.65rem;
    }

    .hero p {
        font-size: 0.95rem;
    }

    .kpi-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 680px) {
    .gradio-container {
        padding-top: 8px !important;
    }

    .hero {
        padding: 16px;
    }

    .toolbar,
    .panel {
        padding: 8px;
    }
}
"""


yolo_model = YOLO("models/yolo/best.pt")
swin_classifier = load_swin_faiss_classifier()


def _clean_label(value):
    return str(value or "").strip()


def _is_distinct_product_name(product, category, subcategory):
    product = _clean_label(product)
    if not product or product.lower() in {"unknown", "none", "n/a", "na"}:
        return False
    return product.lower() not in {_clean_label(category).lower(), _clean_label(subcategory).lower()}


def pad_box(x1, y1, x2, y2, img_w, img_h, pad=10):
    x1 = max(0, int(x1 - pad))
    y1 = max(0, int(y1 - pad))
    x2 = min(img_w, int(x2 + pad))
    y2 = min(img_h, int(y2 + pad))
    return x1, y1, x2, y2


def estimate_empty_space(boxes, img_w, img_h, max_dim=640):
    if len(boxes) == 0:
        return 1.0, 0.0

    scale = max(1, max(img_w, img_h) // max_dim)
    h = max(1, img_h // scale)
    w = max(1, img_w // scale)
    mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v / scale)) for v in box[:4]]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    covered = float(mask.sum())
    coverage = covered / (w * h)
    return 1.0 - coverage, coverage


def _build_summary_html(rows, empty_ratio, empty_label):
    unique_cats = {}
    unknown_items = []
    for r in rows:
        cat = r.get("product_category") or "unknown"
        unique_cats[cat] = unique_cats.get(cat, 0) + 1
        if cat.lower() == "unknown":
            unknown_items.append(r["crop_id"])

    total_items = len(rows)
    distinct_categories = len([cat for cat in unique_cats.keys() if cat.lower() != "unknown"])
    shelf_type = "Unknown"
    if distinct_categories == 1:
        shelf_type = "Category-specific"
    elif distinct_categories > 1:
        shelf_type = "Mixed"

    summary_lines = ["<h3>Store Management Summary</h3>"]
    summary_lines.append(
        f"<p>We detected <strong>{total_items}</strong> product{'s' if total_items != 1 else ''} on this shelf. "
        f"The shelf appears <strong>{shelf_type.lower()}</strong> with {distinct_categories} distinct category{'ies' if distinct_categories != 1 else ''}.</p>"
    )

    if unique_cats:
        sorted_cats = sorted(unique_cats.items(), key=lambda x: -x[1])
        category_lines = [f"{cnt} x {cat}" for cat, cnt in sorted_cats[:5]]
        summary_lines.append(f"<p><strong>Top categories:</strong> {', '.join(category_lines)}</p>")
    else:
        summary_lines.append("<p><strong>Top categories:</strong> none detected yet.</p>")

    retail_summary = summarize_retail_decisions(rows)
    top_products = retail_summary.get("top_products") or []
    if top_products:
        product_lines = [f"{cnt} x {name}" for name, cnt in top_products[:5]]
        summary_lines.append(f"<p><strong>Likely products:</strong> {', '.join(product_lines)}</p>")

    if rows:
        summary_lines.append(f"<p><strong>Resolver stack:</strong> SWIN + FAISS classification used for {len(rows)} crop{'s' if len(rows) != 1 else ''}.</p>")

    if unknown_items:
        summary_lines.append(f"<p><strong>Manual review needed:</strong> {len(unknown_items)} item{'s' if len(unknown_items) != 1 else ''} may be unclear and should be checked manually.</p>")
    else:
        summary_lines.append("<p><strong>Manual review needed:</strong> no unclear items were found.</p>")

    summary_lines.append(f"<p><strong>Estimated visible empty space:</strong> {empty_ratio*100:.0f}% ({empty_label}).</p>")
    return "\n".join(summary_lines)


def _build_kpi_html(rows, run_qwen=False):
    total_products = len(rows)
    unknown_count = sum(1 for row in rows if (row.get("product_category") or "unknown").lower() == "unknown")
    avg_confidence_text = "N/A"
    if run_qwen:
        confidence_values = [float(row.get("qwen_confidence")) for row in rows if isinstance(row.get("qwen_confidence"), (int, float))]
        if confidence_values:
            avg_confidence_text = f"{(sum(confidence_values) / len(confidence_values)) * 100:.0f}%"

    return (
        "<div class='kpi-grid'>"
        f"<div class='kpi-card'><div class='kpi-label'>Total Products</div><div class='kpi-value'>{total_products}</div></div>"
        f"<div class='kpi-card'><div class='kpi-label'>Unknown Items</div><div class='kpi-value'>{unknown_count}</div></div>"
        f"<div class='kpi-card'><div class='kpi-label'>Avg Qwen Confidence</div><div class='kpi-value'>{avg_confidence_text}</div></div>"
        "</div>"
    )


def _build_sku_table_html(rows, run_qwen=False):
    if not run_qwen:
        return "<p>Qwen SKU extraction is skipped in Analyze Shelf mode. Use the Full Shelf SKU button.</p>"
    if not rows:
        return "<p>No crops were available for SKU analysis.</p>"

    lines = [
        "<div class='sku-table-wrap'><table class='sku-table'>",
        "<thead><tr><th>Crop</th><th>SKU / Product</th><th>Category</th><th>Subcategory</th><th>Clues</th><th>Confidence</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        crop_id = row.get("crop_id")
        sku_detail = row.get("sku_detail") or "No SKU detected"
        category = row.get("product_category") or "unknown"
        subcategory = row.get("subcategory") or "unknown"
        clues = row.get("qwen_clues") or []
        clues_text = "; ".join(clues) if clues else "-"
        confidence = row.get("qwen_confidence")
        confidence_label = row.get("qwen_confidence_label") or "n/a"
        confidence_text = "-"
        if isinstance(confidence, (int, float)):
            confidence_text = f"{confidence*100:.0f}% ({confidence_label})"

        lines.append(
            f"<tr><td>{crop_id}</td><td>{sku_detail}</td><td>{category}</td><td>{subcategory}</td><td>{clues_text}</td><td>{confidence_text}</td></tr>"
        )

    lines.extend(["</tbody>", "</table></div>"])
    return "".join(lines)


def _build_detected_items_html(rows, run_qwen=False):
    list_html = "<h4>Detected items</h4>"
    if rows:
        list_html += "<details open><summary>Show / hide detected items</summary><ol>"
        for r in rows:
            cid = r["crop_id"]
            cat = r.get("product_category") or "unknown"
            sub = r.get("subcategory") or "unknown"
            sku_detail = r.get("sku_detail") or "No SKU detected"
            qwen_confidence = r.get("qwen_confidence")
            qwen_confidence_label = r.get("qwen_confidence_label")
            qwen_clues = r.get("qwen_clues") or []
            qwen_confidence_suffix = ""
            if isinstance(qwen_confidence, (int, float)):
                qwen_confidence_suffix = f" (confidence: {qwen_confidence*100:.0f}%, {qwen_confidence_label or 'n/a'})"
            clues_suffix = ""
            if qwen_clues:
                clues_suffix = f"<br><i>Clues:</i> {'; '.join(qwen_clues)}"
            if cat.lower() == "unknown":
                qwen_line = f"<br><i>Qwen 2.5 VL:</i> {sku_detail}{qwen_confidence_suffix}{clues_suffix}" if run_qwen else ""
                list_html += f"<li><b>Item {cid}</b> could not be confidently identified and should be reviewed.{qwen_line}</li>"
            else:
                retail_decision = (r.get("retail_product") or {}).get("decision") or {}
                product = retail_decision.get("predicted_product")
                retail_confidence = retail_decision.get("confidence")
                product_prefix = ""
                if _is_distinct_product_name(product, cat, sub):
                    product_prefix = f"<strong>{product}</strong>, "
                retail_confidence_suffix = ""
                if isinstance(retail_confidence, (int, float)):
                    retail_confidence_suffix = f" ({retail_confidence*100:.0f}% confidence)"
                qwen_line = f"<br><i>Qwen 2.5 VL:</i> {sku_detail}{qwen_confidence_suffix}{clues_suffix}" if run_qwen else ""
                list_html += f"<li><b>Item {cid}</b> is likely {product_prefix}<strong>{cat}</strong> with subcategory <strong>{sub}</strong>{retail_confidence_suffix}.{qwen_line}</li>"
        list_html += "</ol></details>"
    else:
        list_html += "<p>No product items were found in the image.</p>"
    return list_html


def _build_answer_html(question, rows, empty_ratio, empty_label):
    unique_cats = {}
    unknown_items = []
    for r in rows:
        cat = r.get("product_category") or "unknown"
        unique_cats[cat] = unique_cats.get(cat, 0) + 1
        if cat.lower() == "unknown":
            unknown_items.append(r["crop_id"])

    total_items = len(rows)
    distinct_categories = len([cat for cat in unique_cats.keys() if cat.lower() != "unknown"])
    shelf_type = "Unknown"
    if distinct_categories == 1:
        shelf_type = "Category-specific"
    elif distinct_categories > 1:
        shelf_type = "Mixed"

    category_lines = []
    if unique_cats:
        sorted_cats = sorted(unique_cats.items(), key=lambda x: -x[1])
        category_lines = [f"{cnt} x {cat}" for cat, cnt in sorted_cats[:5]]

    question_answers = {
        "How many products were detected on this shelf?": f"We detected {total_items} product{'s' if total_items != 1 else ''} on the shelf.",
        "Which are the top detected categories by count?": f"The top categories are {', '.join(category_lines)}." if unique_cats else "No clear product categories were identified.",
        "Which items need manual review?": f"{len(unknown_items)} item{'s' if len(unknown_items) != 1 else ''} may need manual review." if unknown_items else "No items require manual review at this time.",
        "Is this shelf mixed or category-specific?": f"This shelf appears to be {shelf_type.lower()}.",
        "How much empty space is visible on this shelf?": f"About {empty_ratio*100:.0f}% of the shelf appears empty ({empty_label}).",
    }

    selected_answer = question_answers.get(question, "Please upload an image and select a question.")
    return f"<p><b>{question}</b><br>{selected_answer}</p>"


def _build_sku_results_html(rows, run_qwen=False):
    sku_results_html = "<h4>Full shelf SKU results</h4>"
    if not run_qwen:
        sku_results_html += "<p>Qwen SKU extraction is skipped in Analyze Shelf mode. Use the Full Shelf SKU button.</p>"
    elif rows:
        sku_results_html += "<ol>"
        for r in rows:
            cid = r["crop_id"]
            sku_detail = r.get("sku_detail") or "No SKU detected"
            confidence = r.get("qwen_confidence")
            confidence_label = r.get("qwen_confidence_label")
            clues = r.get("qwen_clues") or []
            conf_text = f" (confidence: {confidence*100:.0f}%, {confidence_label or 'n/a'})" if isinstance(confidence, (int, float)) else ""
            clues_text = f"<br><i>Clues:</i> {'; '.join(clues)}" if clues else ""
            sku_results_html += f"<li><b>Crop {cid}</b>: {sku_detail}{conf_text}{clues_text}</li>"
        sku_results_html += "</ol>"
    else:
        sku_results_html += "<p>No crops were available for SKU analysis.</p>"
    return sku_results_html


def _build_progress_html(processed_count, total_count, run_qwen=False, status_text=None):
    total_count = max(0, int(total_count or 0))
    processed_count = max(0, min(int(processed_count or 0), total_count if total_count > 0 else 0))
    percent = int((processed_count / total_count) * 100) if total_count > 0 else 0
    mode_label = "Full SKU extraction" if run_qwen else "Shelf analysis"
    status_line = status_text or f"{mode_label} in progress..."

    return (
        "<div style='padding:10px;border:1px solid #d9e7de;border-radius:12px;background:#ffffffcc;'>"
        f"<div style='font-size:0.9rem;color:#334155;margin-bottom:6px;'><b>{mode_label}</b> - {processed_count}/{total_count} crops</div>"
        "<div style='height:10px;background:#e2ece7;border-radius:999px;overflow:hidden;'>"
        f"<div style='height:100%;width:{percent}%;background:linear-gradient(90deg,#0f766e,#14b8a6);transition:width .25s ease;'></div>"
        "</div>"
        f"<div style='margin-top:6px;font-size:0.85rem;color:#475569;'>{status_line}</div>"
        "</div>"
    )


def _build_analytics_overview_html(rows, empty_ratio, run_qwen=False):
    total = len(rows)
    known = sum(1 for row in rows if (row.get("product_category") or "unknown").lower() != "unknown")
    unknown = max(0, total - known)
    occupancy = int(round((1.0 - float(empty_ratio)) * 100)) if total >= 0 else 0

    cat_counts = {}
    for row in rows:
        cat = row.get("product_category") or "unknown"
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    top_category = max(cat_counts.items(), key=lambda kv: kv[1])[0] if cat_counts else "N/A"

    avg_qwen = "N/A"
    if run_qwen:
        qvals = [float(r.get("qwen_confidence")) for r in rows if isinstance(r.get("qwen_confidence"), (int, float))]
        if qvals:
            avg_qwen = f"{(sum(qvals) / len(qvals))*100:.0f}%"

    return (
        "<div class='kpi-grid'>"
        f"<div class='kpi-card'><div class='kpi-label'>Shelf Occupancy</div><div class='kpi-value'>{occupancy}%</div></div>"
        f"<div class='kpi-card'><div class='kpi-label'>Known vs Unknown</div><div class='kpi-value'>{known}/{unknown}</div></div>"
        f"<div class='kpi-card'><div class='kpi-label'>Top Category</div><div class='kpi-value'>{top_category}</div></div>"
        f"<div class='kpi-card'><div class='kpi-label'>Avg Qwen Confidence</div><div class='kpi-value'>{avg_qwen}</div></div>"
        "</div>"
    )


def run_bi_query(selected_query, rows_state):
    if not rows_state:
        return "<p>No analysis data yet. Run Analyze shelf or Full SKU detection first.</p>"

    rows = rows_state
    cat_counts = {}
    subcat_counts = {}
    unknown_ids = []
    low_conf = []
    for row in rows:
        cid = row.get("crop_id")
        cat = row.get("product_category") or "unknown"
        sub = row.get("subcategory") or "unknown"
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        subcat_counts[sub] = subcat_counts.get(sub, 0) + 1
        if cat.lower() == "unknown":
            unknown_ids.append(cid)
        qconf = row.get("qwen_confidence")
        if isinstance(qconf, (int, float)) and qconf < 0.45:
            low_conf.append((cid, qconf))

    total = len(rows)
    unknown_pct = (len(unknown_ids) / total * 100.0) if total else 0.0

    if selected_query == "What are top categories by facings?":
        ordered = sorted(cat_counts.items(), key=lambda kv: -kv[1])
        lines = "".join([f"<li><b>{cat}</b>: {cnt}</li>" for cat, cnt in ordered[:8]])
        return f"<h4>Top categories by facings</h4><ol>{lines}</ol>"

    if selected_query == "Which products or crops need manual review?":
        if not unknown_ids and not low_conf:
            return "<h4>Manual review queue</h4><p>No crops currently need urgent review.</p>"
        unknown_line = f"<p><b>Unknown category crops:</b> {', '.join([str(i) for i in unknown_ids])}</p>" if unknown_ids else ""
        low_line = "<p><b>Low-confidence Qwen crops:</b> " + ", ".join([f"{cid} ({score*100:.0f}%)" for cid, score in low_conf]) + "</p>" if low_conf else ""
        return f"<h4>Manual review queue</h4>{unknown_line}{low_line}"

    if selected_query == "What is the unknown-rate and confidence risk?":
        avg_q = [float(r.get("qwen_confidence")) for r in rows if isinstance(r.get("qwen_confidence"), (int, float))]
        avg_q_text = f"{(sum(avg_q)/len(avg_q))*100:.0f}%" if avg_q else "N/A"
        return (
            "<h4>Risk snapshot</h4>"
            f"<p><b>Unknown-rate:</b> {unknown_pct:.1f}% ({len(unknown_ids)}/{total})</p>"
            f"<p><b>Average Qwen confidence:</b> {avg_q_text}</p>"
            f"<p><b>Low-confidence crops (&lt;45%):</b> {len(low_conf)}</p>"
        )

    if selected_query == "Show likely assortment mix (category and subcategory).":
        cat_ordered = sorted(cat_counts.items(), key=lambda kv: -kv[1])
        sub_ordered = sorted(subcat_counts.items(), key=lambda kv: -kv[1])
        cat_list = "".join([f"<li>{cat}: {cnt}</li>" for cat, cnt in cat_ordered[:8]])
        sub_list = "".join([f"<li>{sub}: {cnt}</li>" for sub, cnt in sub_ordered[:8]])
        return (
            "<h4>Assortment mix</h4>"
            f"<p><b>Category mix</b></p><ul>{cat_list}</ul>"
            f"<p><b>Subcategory mix</b></p><ul>{sub_list}</ul>"
        )

    if selected_query == "Give replenishment and planogram suggestions.":
        ordered = sorted(cat_counts.items(), key=lambda kv: -kv[1])
        lead = ordered[0][0] if ordered else "unknown"
        return (
            "<h4>Replenishment suggestions</h4>"
            f"<p>Prioritize replenishment for <b>{lead}</b> and review facings where unknown detections were found.</p>"
            "<p>For planogram hygiene: keep high-frequency categories centered, reduce mixed clutter, and re-check low-confidence crops.</p>"
        )

    return "<p>Select a BI query and click Run BI Query.</p>"


def _compose_stream_payload(annotated, rows, question, empty_ratio, empty_label, run_qwen=False, status_text=None, processed_count=0, total_count=0):
    summary_html = _build_summary_html(rows, empty_ratio, empty_label)
    summary_html = f"{summary_html}\n{_build_detected_items_html(rows, run_qwen=run_qwen)}"
    answer_html = _build_answer_html(question, rows, empty_ratio, empty_label)
    if status_text:
        answer_html = f"<p><i>{status_text}</i></p>{answer_html}"

    progress_html = _build_progress_html(processed_count=processed_count, total_count=total_count, run_qwen=run_qwen, status_text=status_text)
    box_choices = [str(r["crop_id"]) for r in rows]
    initial_choice = box_choices[0] if box_choices else None
    sku_results_html = _build_sku_results_html(rows, run_qwen=run_qwen)
    kpi_html = _build_kpi_html(rows, run_qwen=run_qwen)
    sku_table_html = _build_sku_table_html(rows, run_qwen=run_qwen)
    analytics_html = _build_analytics_overview_html(rows, empty_ratio, run_qwen=run_qwen)

    return (
        annotated,
        summary_html,
        answer_html,
        progress_html,
        gr.update(choices=box_choices, value=initial_choice),
        rows,
        sku_results_html,
        kpi_html,
        sku_table_html,
        analytics_html,
    )


def process_image_stream(input_image, question, run_qwen=False):
    t0 = time.time()
    image = input_image.convert("RGB")
    img_w, img_h = image.size

    results = yolo_model(image, conf=0.25)
    t_yolo = time.time()
    print(f"[timing] YOLO inference took {t_yolo - t0:.3f}s")
    boxes = results[0].boxes.xyxy.cpu().numpy()

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    rows = []
    empty_ratio, _ = estimate_empty_space(boxes, img_w, img_h)
    empty_label = "High" if empty_ratio >= 0.55 else "Moderate" if empty_ratio >= 0.25 else "Low"

    initial_status = f"Detected {len(boxes)} crop{'s' if len(boxes) != 1 else ''}. Starting {'full SKU extraction' if run_qwen else 'shelf analysis'}..."
    yield _compose_stream_payload(
        annotated,
        rows,
        question,
        empty_ratio,
        empty_label,
        run_qwen=run_qwen,
        status_text=initial_status,
        processed_count=0,
        total_count=len(boxes),
    )

    def _looks_like_path_label(label: str) -> bool:
        if not label:
            return False
        normalized = label.lower().replace("_", " ").strip()
        if normalized.startswith("train ") or normalized.startswith("train_"):
            return True
        if " crop " in normalized or normalized.endswith(" crop") or normalized.startswith("crop "):
            return True
        return False

    def _is_valid_label(label: str) -> bool:
        if not label:
            return False
        cleaned = label.strip()
        if cleaned.lower() in {"unknown", "n/a", "na", "none"}:
            return False
        if _looks_like_path_label(cleaned):
            return False
        return True

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]
        px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, img_w, img_h, pad=12)

        crop = image.crop((px1, py1, px2, py2))
        t_before_swin = time.time()
        swin_result = None
        final_category = "unknown"
        subcategory_label = "unknown"

        if swin_classifier.is_ready():
            try:
                swin_result = swin_classifier.classify(crop, top_k=10, top_labels=5)
                t_swin = time.time()
                print(f"[timing] crop {i} Swin FAISS took {t_swin - t_before_swin:.3f}s")
                if i == 0:
                    print(f"[debug] crop {i} swin_result={swin_result}")

                swin_cat = swin_result.get("predicted_category") or swin_result.get("label")
                swin_subcat = swin_result.get("predicted_subcategory") or swin_result.get("best_subcategory")

                final_category = swin_cat if _is_valid_label(swin_cat) else "unknown"
                subcategory_label = swin_subcat if _is_valid_label(swin_subcat) else "unknown"
            except Exception as swin_exc:
                print(f"[warning] crop {i} Swin FAISS failed: {swin_exc}")

        t_before_retail = time.time()
        retail_product = resolve_retail_product(
            image=crop,
            swin_result=swin_result,
            category_hint=final_category,
            subcategory_hint=subcategory_label,
        )
        t_retail = time.time()
        print(f"[timing] crop {i} retail product resolver took {t_retail - t_before_retail:.3f}s")

        retail_decision = retail_product.get("decision") or {}
        if retail_decision.get("category") and retail_decision.get("category") != "unknown":
            final_category = retail_decision["category"]
        if retail_decision.get("subcategory") and retail_decision.get("subcategory") != "unknown":
            subcategory_label = retail_decision["subcategory"]

        if final_category == "unknown":
            swin_classifier.save_unknown_crop(crop, i + 1)

        sku_detail = "Qwen SKU detection not run in Analyze Shelf mode."
        qwen_confidence = None
        qwen_confidence_label = None
        qwen_clues = []
        if run_qwen:
            sku_prompt = (
                "Identify the exact SKU, product name, brand, category, and subcategory visible in this crop. "
                "Respond in English only. Return a concise answer only."
            )
            sku_result = generate_qwen_sku_answer(image=crop, broad_category=final_category, user_prompt=sku_prompt)
            sku_detail = (sku_result.get("answer") or "").strip() or (sku_result.get("error") or "No SKU detected")
            qwen_confidence = sku_result.get("confidence")
            qwen_confidence_label = sku_result.get("confidence_label")
            structured = sku_result.get("structured") or {}
            qwen_clues = structured.get("clues") or []

        draw.rectangle((px1, py1, px2, py2), outline="red", width=3)
        draw.text((px1, max(0, py1 - 12)), str(final_category), fill="red")

        rows.append(
            {
                "crop_id": i + 1,
                "box": [px1, py1, px2, py2],
                "product_category": final_category,
                "subcategory": subcategory_label,
                "swin": swin_result,
                "clip": None,
                "qwen_sku": None,
                "retail_product": retail_product,
                "crop_image": crop,
                "sku_detail": sku_detail,
                "qwen_confidence": qwen_confidence,
                "qwen_confidence_label": qwen_confidence_label,
                "qwen_clues": qwen_clues,
            }
        )

        progress_status = f"Processed crop {i + 1}/{len(boxes)}."
        yield _compose_stream_payload(
            annotated,
            rows,
            question,
            empty_ratio,
            empty_label,
            run_qwen=run_qwen,
            status_text=progress_status,
            processed_count=i + 1,
            total_count=len(boxes),
        )

    final_status = f"Completed processing {len(rows)} crop{'s' if len(rows) != 1 else ''}."
    yield _compose_stream_payload(
        annotated,
        rows,
        question,
        empty_ratio,
        empty_label,
        run_qwen=run_qwen,
        status_text=final_status,
        processed_count=len(rows),
        total_count=len(boxes),
    )


def analyze_shelf(input_image, question):
    return process_image_stream(input_image, question, run_qwen=False)


def run_full_shelf_sku(input_image, question):
    return process_image_stream(input_image, question, run_qwen=True)


def save_flagged_crop(selected_crop_id, rows_state, flag_reason, save_image):
    if not rows_state:
        return "<p>No detected boxes are available yet.</p>", None
    try:
        crop_id = int(selected_crop_id)
    except (TypeError, ValueError):
        return "<p>Please select a detected box first.</p>", None

    for row in rows_state:
        if int(row.get("crop_id", -1)) != crop_id:
            continue
        crop_image = row.get("crop_image")
        if crop_image is None:
            return "<p>No crop image is available to save.</p>", None
        out_dir = Path("flagged_crops")
        out_dir.mkdir(exist_ok=True)
        filename = f"crop_{crop_id}_{int(time.time())}.png"
        path = out_dir / filename
        crop_image.save(path)
        meta = {
            "crop_id": crop_id,
            "flag_reason": flag_reason or "unknown",
            "saved_path": str(path),
            "timestamp": int(time.time()),
        }
        with open(out_dir / "flags.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta) + "\n")
        return f"<p>Saved flagged crop to <b>{path}</b>.</p>", str(path)

    return f"<p>No detected box with id {crop_id} was found.</p>", None


def get_selected_crop_details(selected_crop_id, rows_state):
    if not rows_state:
        return None, "<p>No detected crops yet. Run analysis first.</p>"

    try:
        crop_id = int(selected_crop_id)
    except (TypeError, ValueError):
        return None, "<p>Select a crop id to view details.</p>"

    for row in rows_state:
        if int(row.get("crop_id", -1)) != crop_id:
            continue

        crop_image = row.get("crop_image")
        category = row.get("product_category") or "unknown"
        subcategory = row.get("subcategory") or "unknown"
        sku_detail = row.get("sku_detail") or "No SKU detected"
        confidence = row.get("qwen_confidence")
        confidence_label = row.get("qwen_confidence_label") or "n/a"
        clues = row.get("qwen_clues") or []
        clues_html = "".join([f"<li>{c}</li>" for c in clues]) if clues else "<li>None</li>"

        retail_decision = (row.get("retail_product") or {}).get("decision") or {}
        mapped_product = retail_decision.get("predicted_product") or "unknown"
        mapped_confidence = retail_decision.get("confidence")

        confidence_text = f"{confidence*100:.0f}% ({confidence_label})" if isinstance(confidence, (int, float)) else "Not available"
        mapped_confidence_text = f"{mapped_confidence*100:.0f}%" if isinstance(mapped_confidence, (int, float)) else "Not available"

        details_html = (
            f"<h4>Crop {crop_id} mapping details</h4>"
            "<table class='sku-table'>"
            "<thead><tr><th>Field</th><th>Value</th></tr></thead>"
            "<tbody>"
            f"<tr><td>Mapped Product</td><td>{mapped_product}</td></tr>"
            f"<tr><td>Mapped Category</td><td>{category}</td></tr>"
            f"<tr><td>Mapped Subcategory</td><td>{subcategory}</td></tr>"
            f"<tr><td>Qwen SKU / Product</td><td>{sku_detail}</td></tr>"
            f"<tr><td>Qwen Confidence</td><td>{confidence_text}</td></tr>"
            f"<tr><td>Resolver Confidence</td><td>{mapped_confidence_text}</td></tr>"
            "</tbody></table>"
            "<h5 style='margin-top:10px;'>Qwen visual clues</h5>"
            f"<ul>{clues_html}</ul>"
            "<p style='margin-top:8px;'><i>If this mapping looks wrong, add a reason and click Flag and save selected crop.</i></p>"
        )
        return crop_image, details_html

    return None, f"<p>No detected box with id {crop_id} was found.</p>"


with gr.Blocks(
    title="Smart Shelf Management Dashboard",
    theme=gr.themes.Base(primary_hue="teal", secondary_hue="cyan", neutral_hue="slate"),
    css=CUSTOM_CSS,
) as demo:
    gr.Markdown(
        """
<div class=\"hero\">
  <h1>Smart Shelf Intelligence</h1>
    <p>Understand shelf performance, review products, and make better store decisions in one place.</p>
</div>
        """,
    )

    rows_state = gr.State(value=[])

    with gr.Tabs():
        with gr.Tab("Detection Studio"):
            gr.Markdown("<div class='tab-banner'><b>Detection workspace:</b> upload shelf images, run quick analysis, or run full SKU extraction.</div>")
            with gr.Row(elem_classes=["toolbar", "animate-in", "stagger-1"]):
                image_input = gr.Image(type="pil", label="Upload shelf image")
                question_input = gr.Dropdown(
                    choices=[
                        "How many products were detected on this shelf?",
                        "Which are the top detected categories by count?",
                        "Which items need manual review?",
                        "Is this shelf mixed or category-specific?",
                        "How much empty space is visible on this shelf?",
                    ],
                    value="How many products were detected on this shelf?",
                    label="Select a Store Management question",
                )
            with gr.Row(elem_classes=["toolbar", "animate-in", "stagger-2"]):
                analyze_button = gr.Button("Analyze shelf", variant="secondary")
                sku_button = gr.Button("Run full shelf SKU detection", variant="primary")

            gr.Markdown("<div class='section-title'>Live processing insights</div>")
            kpi_cards = gr.HTML(label="KPI Overview", elem_classes=["panel", "animate-in", "stagger-2"])
            output_progress = gr.HTML(label="Processing Progress", elem_classes=["panel", "animate-in", "stagger-2"])

            gr.Markdown("<div class='section-title'>Shelf outputs</div>")
            with gr.Row(elem_classes=["panel", "animate-in", "stagger-3"]):
                output_image = gr.Image(type="pil", label="Annotated image")
                output_summary = gr.HTML(label="Summary")
                output_answer = gr.HTML(label="Selected question answer")
            output_sku_results = gr.HTML(label="Full shelf SKU results", elem_id="sku-results", elem_classes=["panel", "animate-in", "stagger-3"])
            output_sku_table = gr.HTML(label="SKU Results Table", elem_classes=["panel", "animate-in", "stagger-4"])

        with gr.Tab("Crop Review & Flag"):
            gr.Markdown("<div class='tab-banner'><b>Review workspace:</b> select any crop, inspect Qwen mapping against the crop image, and optionally flag it.</div>")
            crop_selector = gr.Dropdown(choices=[], label="Select crop id to review/flag")
            with gr.Row(elem_classes=["panel", "animate-in", "stagger-4"]):
                selected_crop_image = gr.Image(type="pil", label="Selected crop preview")
                selected_crop_mapping = gr.HTML(label="Selected crop mapping details")
            with gr.Row(elem_classes=["toolbar", "animate-in", "stagger-4"]):
                flag_reason = gr.Textbox(label="Flag reason if SKU is unclear", placeholder="e.g. blurry, occluded, no visible barcode")
                save_button = gr.Button("Flag and save selected crop")
            with gr.Row(elem_classes=["panel", "animate-in", "stagger-4"]):
                flagged_output = gr.HTML(label="Flag / save status")
                download_flagged_crop = gr.File(label="Download saved flagged crop")

        with gr.Tab("Analytics & BI"):
            gr.Markdown("<div class='tab-banner'><b>Business intelligence workspace:</b> monitor shelf analytics and run management queries on current detection results.</div>")
            analytics_overview = gr.HTML(label="Store Analytics Snapshot", elem_classes=["panel", "animate-in", "stagger-2"])
            with gr.Row(elem_classes=["toolbar", "animate-in", "stagger-3"]):
                bi_query = gr.Dropdown(
                    choices=[
                        "What are top categories by facings?",
                        "Which products or crops need manual review?",
                        "What is the unknown-rate and confidence risk?",
                        "Show likely assortment mix (category and subcategory).",
                        "Give replenishment and planogram suggestions.",
                    ],
                    value="What are top categories by facings?",
                    label="Business Intelligence Query",
                )
                bi_button = gr.Button("Run BI Query", variant="primary")
            bi_query_output = gr.HTML(label="BI Query Result", elem_classes=["panel", "animate-in", "stagger-4"])

    analyze_button.click(
        analyze_shelf,
        inputs=[image_input, question_input],
        outputs=[output_image, output_summary, output_answer, output_progress, crop_selector, rows_state, output_sku_results, kpi_cards, output_sku_table, analytics_overview],
    )
    sku_button.click(
        run_full_shelf_sku,
        inputs=[image_input, question_input],
        outputs=[output_image, output_summary, output_answer, output_progress, crop_selector, rows_state, output_sku_results, kpi_cards, output_sku_table, analytics_overview],
    )
    save_button.click(
        save_flagged_crop,
        inputs=[crop_selector, rows_state, flag_reason, gr.State(value=True)],
        outputs=[flagged_output, download_flagged_crop],
    )
    crop_selector.change(
        get_selected_crop_details,
        inputs=[crop_selector, rows_state],
        outputs=[selected_crop_image, selected_crop_mapping],
    )
    bi_button.click(
        run_bi_query,
        inputs=[bi_query, rows_state],
        outputs=[bi_query_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        inbrowser=False,
    )
