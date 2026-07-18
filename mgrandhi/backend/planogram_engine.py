"""Planogram reconstruction and compliance scoring (Module 6).

Turns the flat list of YOLO detections that `analysis_service.analyze_shelf` already produces
(box + category/subcategory/brand/product_name per crop) into a spatial planogram grid — which
shelf row each product sits on and its left-to-right position within that row — and diffs that
grid against a merchandiser-defined target layout.

Two things happen here that nothing else in the pipeline does today:

1. **Spatial reconstruction** (`build_grid`): detections only carry a pixel box today. Nobody
   groups them into rows or reads left-to-right order. That grouping is what makes "where" a
   product is a first-class fact instead of a coordinate.
2. **Compliance diff** (`compare_to_template`): the rest of the app can say *what* is on the
   shelf and how much empty space there is, but not whether the *right* products are in the
   *right* place. This is a sequence alignment (like a text diff) between the expected row
   contents and the detected row contents, so partial matches, swaps, and gaps are all reported
   individually rather than collapsed into one empty-space percentage.

Dependency-free (stdlib only) so it can be unit-tested without the YOLO/Swin/FAISS stack loaded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher


# ---------------------------------------------------------------------------
# Step 1: group flat detections into shelf rows, ordered top-to-bottom, and
# order each row's items left-to-right.
# ---------------------------------------------------------------------------

@dataclass
class GridItem:
    crop_id: int
    row_index: int
    slot_index: int
    category: str
    subcategory: str
    brand: str
    product_name: str
    score: float
    box: list[int]

    @property
    def key(self) -> str:
        """The identity a template slot is matched against, most-specific first."""
        return (self.brand or self.subcategory or self.category or "unknown").strip().lower()


def _row_gap_threshold(heights: list[float]) -> float:
    if not heights:
        return 1.0
    sorted_heights = sorted(heights)
    mid = len(sorted_heights) // 2
    median = (
        sorted_heights[mid]
        if len(sorted_heights) % 2
        else (sorted_heights[mid - 1] + sorted_heights[mid]) / 2
    )
    # Two boxes count as the same row if their vertical centers are within ~60% of a
    # typical box height of each other; wide enough to absorb label-height jitter,
    # narrow enough to split genuinely different shelves.
    return max(median * 0.6, 1.0)


def build_grid(detections: list[dict]) -> list[GridItem]:
    """Cluster detections into shelf rows and order each row left-to-right.

    `detections` is the same list of dicts `analysis_service.analyze_shelf` returns
    (each has at least: crop_id, category, subcategory, box=[x1,y1,x2,y2], score).
    Row 0 is the top-most shelf in the photo.
    """
    if not detections:
        return []

    enriched = []
    for record in detections:
        x1, y1, x2, y2 = record["box"]
        enriched.append(
            {
                **record,
                "_cx": (x1 + x2) / 2,
                "_cy": (y1 + y2) / 2,
                "_h": max(y2 - y1, 1),
            }
        )
    enriched.sort(key=lambda r: r["_cy"])
    threshold = _row_gap_threshold([r["_h"] for r in enriched])

    rows: list[list[dict]] = []
    current_row: list[dict] = []
    last_cy = None
    for record in enriched:
        if last_cy is not None and (record["_cy"] - last_cy) > threshold:
            rows.append(current_row)
            current_row = []
        current_row.append(record)
        last_cy = record["_cy"]
    if current_row:
        rows.append(current_row)

    grid: list[GridItem] = []
    for row_index, row in enumerate(rows):
        row.sort(key=lambda r: r["_cx"])
        for slot_index, record in enumerate(row):
            grid.append(
                GridItem(
                    crop_id=record["crop_id"],
                    row_index=row_index,
                    slot_index=slot_index,
                    category=str(record.get("category") or "unknown"),
                    subcategory=str(record.get("subcategory") or "unknown"),
                    brand=str(record.get("brand") or ""),
                    product_name=str(record.get("product_name") or ""),
                    score=float(record.get("score") or 0.0),
                    box=list(record["box"]),
                )
            )
    return grid


# ---------------------------------------------------------------------------
# Step 2: a target layout, and diffing detected rows against it.
# ---------------------------------------------------------------------------

@dataclass
class TemplateSlot:
    slot_index: int
    category: str
    subcategory: str = ""
    brand: str = ""
    facings: int = 1  # how many consecutive units are expected to occupy this slot

    @property
    def key(self) -> str:
        return (self.brand or self.subcategory or self.category or "unknown").strip().lower()


@dataclass
class TemplateRow:
    row_index: int
    slots: list[TemplateSlot] = field(default_factory=list)

    def expected_sequence(self) -> list[str]:
        expanded = []
        for slot in self.slots:
            expanded.extend([slot.key] * max(slot.facings, 1))
        return expanded


@dataclass
class SlotResult:
    row_index: int
    position: int
    status: str  # "compliant" | "misplaced" | "missing" | "extra"
    expected_key: str | None
    actual_key: str | None
    detail: dict = field(default_factory=dict)


@dataclass
class PlanogramResult:
    slots: list[SlotResult]
    compliance_score: float
    total_expected: int
    total_compliant: int
    missing_count: int
    extra_count: int
    row_count_expected: int
    row_count_detected: int


def _diff_row(row_index: int, expected: list[str], actual_items: list[GridItem]) -> list[SlotResult]:
    actual_keys = [item.key for item in actual_items]
    matcher = SequenceMatcher(None, expected, actual_keys, autojunk=False)
    results: list[SlotResult] = []
    position = 0
    for tag, e1, e2, a1, a2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(e2 - e1):
                item = actual_items[a1 + offset]
                results.append(
                    SlotResult(
                        row_index=row_index,
                        position=position,
                        status="compliant",
                        expected_key=expected[e1 + offset],
                        actual_key=item.key,
                        detail={"crop_id": item.crop_id, "box": item.box},
                    )
                )
                position += 1
        elif tag == "delete":
            for offset in range(e2 - e1):
                results.append(
                    SlotResult(
                        row_index=row_index,
                        position=position,
                        status="missing",
                        expected_key=expected[e1 + offset],
                        actual_key=None,
                    )
                )
                position += 1
        elif tag == "insert":
            for offset in range(a2 - a1):
                item = actual_items[a1 + offset]
                results.append(
                    SlotResult(
                        row_index=row_index,
                        position=position,
                        status="extra",
                        expected_key=None,
                        actual_key=item.key,
                        detail={"crop_id": item.crop_id, "box": item.box},
                    )
                )
                position += 1
        elif tag == "replace":
            # Treat as one slot swapped for another: the expected product is missing
            # here and a different, unplanned product occupies its place.
            span = max(e2 - e1, a2 - a1)
            for offset in range(span):
                exp_key = expected[e1 + offset] if e1 + offset < e2 else None
                item = actual_items[a1 + offset] if a1 + offset < a2 else None
                results.append(
                    SlotResult(
                        row_index=row_index,
                        position=position,
                        status="misplaced" if (exp_key and item) else ("missing" if exp_key else "extra"),
                        expected_key=exp_key,
                        actual_key=item.key if item else None,
                        detail={"crop_id": item.crop_id, "box": item.box} if item else {},
                    )
                )
                position += 1
    return results


def compare_to_template(detections: list[dict], template_rows: list[TemplateRow]) -> PlanogramResult:
    """Diff the detected shelf layout against a target planogram.

    Rows are matched by `row_index` (top shelf = 0). A template row with no detections at
    all still contributes its full expected sequence as "missing" so an entirely unstocked
    shelf shows up rather than silently disappearing.
    """
    grid = build_grid(detections)
    by_row: dict[int, list[GridItem]] = {}
    for item in grid:
        by_row.setdefault(item.row_index, []).append(item)

    template_by_row = {row.row_index: row for row in template_rows}
    all_row_indices = sorted(set(by_row) | set(template_by_row))

    all_results: list[SlotResult] = []
    for row_index in all_row_indices:
        template_row = template_by_row.get(row_index)
        actual_items = by_row.get(row_index, [])
        if template_row is None:
            # Detected a row the template doesn't describe at all (unplanned shelf/section).
            for position, item in enumerate(actual_items):
                all_results.append(
                    SlotResult(
                        row_index=row_index,
                        position=position,
                        status="extra",
                        expected_key=None,
                        actual_key=item.key,
                        detail={"crop_id": item.crop_id, "box": item.box},
                    )
                )
            continue
        expected_sequence = template_row.expected_sequence()
        all_results.extend(_diff_row(row_index, expected_sequence, actual_items))

    total_expected = sum(1 for r in all_results if r.status in {"compliant", "missing", "misplaced"})
    total_compliant = sum(1 for r in all_results if r.status == "compliant")
    missing_count = sum(1 for r in all_results if r.status in {"missing", "misplaced"})
    extra_count = sum(1 for r in all_results if r.status == "extra")
    compliance_score = (total_compliant / total_expected) if total_expected else 1.0

    return PlanogramResult(
        slots=all_results,
        compliance_score=round(compliance_score, 4),
        total_expected=total_expected,
        total_compliant=total_compliant,
        missing_count=missing_count,
        extra_count=extra_count,
        row_count_expected=len(template_by_row),
        row_count_detected=len(by_row),
    )
