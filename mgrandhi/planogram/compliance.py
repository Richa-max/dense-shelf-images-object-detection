from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Iterable

import pandas as pd


def normalize_text(x) -> str:
    return (
        str(x or "")
        .lower()
        .replace("&", "and")
        .replace("-", " ")
        .replace("/", " ")
        .replace("\u2019", "'")
        .replace("â€™", "'")
        .replace("\u201c", "")
        .replace("\u201d", "")
        .replace("â€œ", "")
        .replace("â€\x9d", "")
        .strip()
    )


CATEGORY_RULES_58 = {
    "Bottled Water": ["bottled water", "bottle water", "drinking water", "water bottles"],
    "Carbonated Soft Drinks": ["soft drink", "soft drinks", "carbonated", "soda", "cola"],
    "Fruit Juices & Drinks": ["juice", "juices", "fruit drink", "fruit beverage", "nectar"],
    "Energy & Sports Drinks": ["energy drink", "sports drink", "hydration", "electrolyte"],
    "Tea & Coffee Beverages": ["tea", "coffee", "iced tea", "green tea", "coffee pods"],
    "Milk & Flavoured Milk": ["milk", "flavoured milk", "flavored milk", "soy milk", "powdered milk"],
    "Chips & Crisps": ["chips", "crisps", "snacks and chips", "papad and chips"],
    "Namkeen & Savoury Snacks": ["namkeen", "savoury", "savory", "papad", "fryums", "masala snacks"],
    "Biscuits & Cookies": ["biscuit", "biscuits", "cookie", "cookies", "cracker", "crackers"],
    "Wafers & Snack Bars": ["wafer", "wafers", "snack bar", "cereal bar", "granola"],
    "Chocolates": ["chocolate", "chocolates", "chocolate bars"],
    "Candy & Sweets": ["candy", "candies", "sweets", "confectionery", "toffee", "gummies"],
    "Chewing Gum & Mints": ["gum", "gums", "mint", "mints", "mouth freshener"],
    "Instant Noodles & Pasta": ["noodle", "noodles", "pasta", "maggi"],
    "Breakfast Cereals": ["cereal", "cereals", "cornflakes", "oats", "muesli"],
    "Ready-to-Eat Meals": ["ready meal", "meal mix", "soup", "broth", "sandwich"],
    "Sauces, Ketchup & Spreads": ["ketchup", "sauce", "spread", "jam", "jelly", "preserves"],
    "Pickles & Condiments": ["pickle", "pickles", "chutney", "condiment", "seasoning", "soy sauce"],
    "Rice & Grains": ["rice", "grain", "grains", "millet", "millets"],
    "Flour & Baking Essentials": ["flour", "baking", "cake mix", "starch"],
    "Pulses & Lentils": ["pulse", "pulses", "lentil", "lentils", "dal", "beans", "chickpeas"],
    "Spices & Masalas": ["spice", "spices", "masala", "masalas", "masala mixes"],
    "Sugar, Salt & Sweeteners": ["sugar", "salt", "jaggery", "sweetener", "honey"],
    "Cooking Oil & Ghee": ["cooking oil", "edible oil", "olive oil", "ghee", "vanaspati"],
    "Bread & Bakery": ["bread", "bakery", "baked goods", "cake", "donuts", "pastries", "rusk"],
    "Dairy Products": ["dairy", "yogurt", "curd", "cheese", "butter", "paneer", "cream"],
    "Frozen Foods": ["frozen", "ice cream", "freezing foods"],
    "Shampoo & Conditioner": ["shampoo", "conditioner", "hair wash", "dry shampoo"],
    "Soap & Body Wash": ["soap", "bar soap", "body wash", "shower", "bath", "handwash"],
    "Toothpaste & Oral Care": ["toothpaste", "mouth care", "oral care", "mouthwash", "dental hygiene"],
    "Toothbrush & Dental Accessories": ["toothbrush", "dental accessories", "floss", "tongue cleaner"],
    "Deodorants & Fragrances": ["deodorant", "body spray", "perfume", "fragrance"],
    "Skin Care": ["skin care", "skincare", "face care", "face wash", "sunscreen", "moisturizer"],
    "Hair Care & Styling": ["hair oil", "hair gel", "hair color", "hair colour", "hair styling", "haircare"],
    "Shaving & Men's Grooming": ["shaving", "razor", "aftershave", "male grooming", "mens grooming"],
    "Female Hygiene & Period Care": ["feminine hygiene", "female hygiene", "period care", "sanitary", "tampon", "pads"],
    "Intimate Care": ["intimate wash", "feminine wipes", "hygiene wash", "personal hygiene"],
    "Baby Diapers & Wipes": ["diaper", "diapers", "nappies", "baby wipes"],
    "Baby Food & Baby Care": ["baby food", "baby care", "baby lotion", "baby shampoo", "infant food"],
    "Detergent & Laundry Care": ["detergent", "laundry", "washing powder", "fabric softener", "stain remover"],
    "Dishwashing Products": ["dishwash", "dishwashing"],
    "Floor & Surface Cleaners": ["floor cleaner", "surface cleaner", "multipurpose cleaner", "disinfectant cleaner"],
    "Toilet & Bathroom Cleaners": ["toilet cleaner", "bathroom cleaner", "descaler"],
    "Glass, Metal & Specialty Cleaners": ["glass cleaner", "metal polish", "furniture polish", "specialty cleaner"],
    "Paper & Tissue Products": ["tissue", "toilet paper", "paper towels", "napkins", "paper products"],
    "Household Supplies": ["garbage bags", "aluminium foil", "cling wrap", "disposable plates", "kitchen supplies"],
    "Health & OTC Medicines": ["medicine", "pharmacy", "otc", "cold", "cough", "allergy", "pain relief"],
    "Sanitizers & First Aid": ["sanitizer", "first aid", "bandage", "antiseptic", "hand hygiene"],
    "Pet Care": ["pet", "dog food", "cat food", "bird food", "cat care"],
    "Stationery & Small Accessories": ["stationery", "pen", "notebook", "battery", "bulb", "tape", "glue"],
    "Air Fresheners & Home Fragrance": ["air freshener", "room freshener", "home fragrance", "deodorizer", "candles"],
    "Pest Control & Repellents": ["mosquito", "repellent", "insecticide", "pest control", "pesticides"],
    "Fresh Food: Meat, Fish & Eggs": ["fish", "seafood", "meat", "poultry", "eggs"],
    "Electronics & Mobile Accessories": ["electronics", "mobile", "charging cables", "headphones", "audio accessories"],
    "Footwear & Shoe Care": ["shoes", "shoe", "footwear", "slippers", "shoe polish"],
    "Books, Stationery & Media": ["books", "magazines", "newspapers", "reading material", "media"],
    "Automotive Care": ["car care", "automotive", "motor oil", "lubricants", "car accessories"],
    "Alcohol & Tobacco": ["alcohol", "beer", "wine", "liquor", "tobacco", "cigarettes", "vape"],
}

_NORMALIZED_TO_CATEGORY = {}
for _category, _aliases in CATEGORY_RULES_58.items():
    _NORMALIZED_TO_CATEGORY[normalize_text(_category)] = _category
    for _alias in _aliases:
        _NORMALIZED_TO_CATEGORY[normalize_text(_alias)] = _category


DEFAULT_PLANOGRAM_TEXT = """Row 1: Bottled Water, Carbonated Soft Drinks, Fruit Juices & Drinks, Energy & Sports Drinks
Row 2: Chips & Crisps, Namkeen & Savoury Snacks, Biscuits & Cookies, Chocolates
Row 3: Shampoo & Conditioner, Soap & Body Wash, Toothpaste & Oral Care, Toothbrush & Dental Accessories
Row 4: Detergent & Laundry Care, Dishwashing Products, Floor & Surface Cleaners, Household Supplies"""

SHELF_TYPE_PLANOGRAMS = {
    "Bottled Water": """Row 1: Bottled Water, Bottled Water, Energy & Sports Drinks
Row 2: Carbonated Soft Drinks, Fruit Juices & Drinks, Tea & Coffee Beverages
Row 3: Milk & Flavoured Milk, Energy & Sports Drinks, Bottled Water""",
    "Carbonated Soft Drinks": """Row 1: Carbonated Soft Drinks, Carbonated Soft Drinks, Energy & Sports Drinks
Row 2: Fruit Juices & Drinks, Bottled Water, Tea & Coffee Beverages
Row 3: Milk & Flavoured Milk, Bottled Water, Carbonated Soft Drinks""",
    "Fruit Juices & Drinks": """Row 1: Fruit Juices & Drinks, Fruit Juices & Drinks, Energy & Sports Drinks
Row 2: Carbonated Soft Drinks, Bottled Water, Milk & Flavoured Milk
Row 3: Tea & Coffee Beverages, Bottled Water, Fruit Juices & Drinks""",
    "Chips & Crisps": """Row 1: Chips & Crisps, Chips & Crisps, Namkeen & Savoury Snacks
Row 2: Biscuits & Cookies, Wafers & Snack Bars, Chocolates
Row 3: Candy & Sweets, Chewing Gum & Mints, Chips & Crisps""",
    "Namkeen & Savoury Snacks": """Row 1: Namkeen & Savoury Snacks, Chips & Crisps, Namkeen & Savoury Snacks
Row 2: Biscuits & Cookies, Wafers & Snack Bars, Chocolates
Row 3: Candy & Sweets, Chips & Crisps, Namkeen & Savoury Snacks""",
    "Biscuits & Cookies": """Row 1: Biscuits & Cookies, Biscuits & Cookies, Wafers & Snack Bars
Row 2: Chips & Crisps, Namkeen & Savoury Snacks, Chocolates
Row 3: Candy & Sweets, Chewing Gum & Mints, Biscuits & Cookies""",
    "Toothpaste & Oral Care": """Row 1: Toothpaste & Oral Care, Toothbrush & Dental Accessories, Mouthwash
Row 2: Soap & Body Wash, Shampoo & Conditioner, Skin Care
Row 3: Deodorants & Fragrances, Shaving & Men's Grooming, Female Hygiene & Period Care""",
    "Toothbrush & Dental Accessories": """Row 1: Toothbrush & Dental Accessories, Toothpaste & Oral Care, Toothbrush & Dental Accessories
Row 2: Soap & Body Wash, Shampoo & Conditioner, Skin Care
Row 3: Shaving & Men's Grooming, Deodorants & Fragrances, Female Hygiene & Period Care""",
    "Shampoo & Conditioner": """Row 1: Shampoo & Conditioner, Shampoo & Conditioner, Hair Care & Styling
Row 2: Soap & Body Wash, Skin Care, Deodorants & Fragrances
Row 3: Toothpaste & Oral Care, Toothbrush & Dental Accessories, Shaving & Men's Grooming""",
    "Soap & Body Wash": """Row 1: Soap & Body Wash, Soap & Body Wash, Skin Care
Row 2: Shampoo & Conditioner, Hair Care & Styling, Deodorants & Fragrances
Row 3: Toothpaste & Oral Care, Toothbrush & Dental Accessories, Female Hygiene & Period Care""",
    "Detergent & Laundry Care": """Row 1: Detergent & Laundry Care, Detergent & Laundry Care, Dishwashing Products
Row 2: Floor & Surface Cleaners, Toilet & Bathroom Cleaners, Glass, Metal & Specialty Cleaners
Row 3: Paper & Tissue Products, Household Supplies, Air Fresheners & Home Fragrance""",
    "Floor & Surface Cleaners": """Row 1: Floor & Surface Cleaners, Toilet & Bathroom Cleaners, Glass, Metal & Specialty Cleaners
Row 2: Detergent & Laundry Care, Dishwashing Products, Household Supplies
Row 3: Paper & Tissue Products, Air Fresheners & Home Fragrance, Pest Control & Repellents""",
    "Household Supplies": """Row 1: Household Supplies, Paper & Tissue Products, Air Fresheners & Home Fragrance
Row 2: Detergent & Laundry Care, Dishwashing Products, Floor & Surface Cleaners
Row 3: Toilet & Bathroom Cleaners, Glass, Metal & Specialty Cleaners, Pest Control & Repellents""",
}

SHELF_TYPE_GROUPS = {
    "Beverages": [
        "Bottled Water",
        "Carbonated Soft Drinks",
        "Fruit Juices & Drinks",
        "Energy & Sports Drinks",
        "Tea & Coffee Beverages",
        "Milk & Flavoured Milk",
    ],
    "Snacks": [
        "Chips & Crisps",
        "Namkeen & Savoury Snacks",
        "Biscuits & Cookies",
        "Wafers & Snack Bars",
        "Chocolates",
        "Candy & Sweets",
        "Chewing Gum & Mints",
    ],
    "Personal Care": [
        "Shampoo & Conditioner",
        "Soap & Body Wash",
        "Toothpaste & Oral Care",
        "Toothbrush & Dental Accessories",
        "Deodorants & Fragrances",
        "Skin Care",
        "Hair Care & Styling",
        "Shaving & Men's Grooming",
        "Female Hygiene & Period Care",
    ],
    "Household": [
        "Detergent & Laundry Care",
        "Dishwashing Products",
        "Floor & Surface Cleaners",
        "Toilet & Bathroom Cleaners",
        "Glass, Metal & Specialty Cleaners",
        "Paper & Tissue Products",
        "Household Supplies",
        "Air Fresheners & Home Fragrance",
        "Pest Control & Repellents",
    ],
}


@dataclass
class ComplianceResult:
    score: float
    matched: pd.DataFrame
    misplaced: pd.DataFrame
    missing: pd.DataFrame
    unexpected: pd.DataFrame
    row_summary: pd.DataFrame


def canonical_category(*values) -> str:
    candidates = [normalize_text(v) for v in values if normalize_text(v)]
    for candidate in candidates:
        if candidate in _NORMALIZED_TO_CATEGORY:
            return _NORMALIZED_TO_CATEGORY[candidate]
    for candidate in candidates:
        for alias, category in _NORMALIZED_TO_CATEGORY.items():
            if alias and (alias in candidate or candidate in alias):
                return category
    for candidate in candidates:
        matches = get_close_matches(candidate, _NORMALIZED_TO_CATEGORY.keys(), n=1, cutoff=0.86)
        if matches:
            return _NORMALIZED_TO_CATEGORY[matches[0]]
    return str(values[0] or "unknown")


def parse_planogram(text: str) -> list[list[str]]:
    rows = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1]
        categories = [canonical_category(part) for part in line.split(",") if part.strip()]
        if categories:
            rows.append(categories)
    return rows


def planogram_for_shelf_type(shelf_type: str) -> tuple[str, str]:
    category = canonical_category(shelf_type)
    if category in SHELF_TYPE_PLANOGRAMS:
        return category, SHELF_TYPE_PLANOGRAMS[category]

    for group_name, categories in SHELF_TYPE_GROUPS.items():
        if category in categories:
            rows = [categories[i:i + 3] for i in range(0, min(len(categories), 9), 3)]
            text = "\n".join(
                f"Row {row_number}: {', '.join(row_categories)}"
                for row_number, row_categories in enumerate(rows, start=1)
            )
            return f"{group_name} shelf", text

    return category or "Default shelf", DEFAULT_PLANOGRAM_TEXT


def infer_shelf_rows(records: list[dict], row_count: int) -> list[dict]:
    if not records:
        return []
    row_count = max(1, min(row_count, len(records)))
    centers = []
    for record in records:
        box = record.get("box") or [0, 0, 0, 0]
        y_center = (float(box[1]) + float(box[3])) / 2.0
        centers.append(y_center)

    if row_count == 1:
        return [{**record, "planogram_row": 1} for record in records]

    ordered = sorted(enumerate(centers), key=lambda item: item[1])
    gaps = [
        (ordered[i + 1][1] - ordered[i][1], i)
        for i in range(len(ordered) - 1)
    ]
    split_after = {
        split_index
        for _, split_index in sorted(gaps, reverse=True)[:row_count - 1]
    }
    assigned_rows = {}
    current_row = 1
    for position, (record_index, _) in enumerate(ordered):
        assigned_rows[record_index] = current_row
        if position in split_after:
            current_row += 1

    enriched = []
    for record_index, record in enumerate(records):
        enriched.append({**record, "planogram_row": assigned_rows.get(record_index, 1)})
    return enriched


def infer_detected_row_count(records: list[dict], max_rows: int = 8) -> int:
    boxes = [record.get("box") for record in records if record.get("box")]
    if len(boxes) <= 1:
        return max(1, len(boxes))

    centers = sorted((float(box[1]) + float(box[3])) / 2.0 for box in boxes)
    heights = sorted(max(1.0, float(box[3]) - float(box[1])) for box in boxes)
    median_height = heights[len(heights) // 2]
    gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    if not gaps:
        return 1

    meaningful_gap = max(median_height * 0.75, (centers[-1] - centers[0]) * 0.08, 12.0)
    row_count = 1 + sum(1 for gap in gaps if gap >= meaningful_gap)
    return max(1, min(max_rows, row_count))


def _category_counts(values: Iterable[str]) -> Counter:
    return Counter(v for v in values if v and str(v).lower() != "unknown")


def _record_category(record: dict) -> str:
    return canonical_category(
        record.get("category"),
        record.get("subcategory"),
        record.get("product_name"),
        record.get("sku_text"),
        record.get("visible_text"),
    )


def zone_winner_planogram(records: list[dict], row_count: int | None = None) -> tuple[str, str, pd.DataFrame]:
    if not records:
        return "Zone winner shelf", "", pd.DataFrame()

    row_count = row_count or infer_detected_row_count(records)
    enriched = infer_shelf_rows(records, row_count)
    row_categories: dict[int, list[str]] = {i: [] for i in range(1, row_count + 1)}
    for record in enriched:
        category = _record_category(record)
        if normalize_text(category) != "unknown":
            row_categories.setdefault(int(record["planogram_row"]), []).append(category)

    summary_rows = []
    planogram_lines = []
    for row_number in range(1, row_count + 1):
        counts = _category_counts(row_categories.get(row_number, []))
        total_items = sum(counts.values())
        if counts:
            winner, winner_count = counts.most_common(1)[0]
            share = winner_count / total_items if total_items else 0.0
            top_categories = ", ".join(
                f"{category} ({count})" for category, count in counts.most_common(3)
            )
        else:
            winner, winner_count, share, top_categories = "unknown", 0, 0.0, ""
        planogram_lines.append(f"Row {row_number}: {winner}")
        summary_rows.append({
            "row": row_number,
            "winner_category": winner,
            "winner_count": winner_count,
            "total_items": total_items,
            "winner_share": share,
            "top_categories": top_categories,
        })

    return "Zone winner shelf", "\n".join(planogram_lines), pd.DataFrame(summary_rows)


def evaluate_zone_winner(records: list[dict], row_count: int | None = None) -> ComplianceResult:
    if not records:
        empty = pd.DataFrame()
        return ComplianceResult(0.0, empty, empty, empty, empty, empty)

    row_count = row_count or infer_detected_row_count(records)
    _, _, zone_summary = zone_winner_planogram(records, row_count)
    if zone_summary.empty:
        empty = pd.DataFrame()
        return ComplianceResult(0.0, empty, empty, empty, empty, empty)

    winners = {
        int(row["row"]): row["winner_category"]
        for _, row in zone_summary.iterrows()
        if normalize_text(row.get("winner_category")) != "unknown"
    }
    enriched = infer_shelf_rows(records, row_count)
    matched_rows = []
    misplaced_rows = []
    observed_by_row: dict[int, list[str]] = {i: [] for i in range(1, row_count + 1)}

    for record in enriched:
        row_number = int(record["planogram_row"])
        category = _record_category(record)
        observed_by_row.setdefault(row_number, []).append(category)
        row = {
            "crop_id": record.get("crop_id"),
            "row": row_number,
            "category": category,
            "subcategory": record.get("subcategory", ""),
            "product": record.get("product_name", ""),
            "brand": record.get("brand", ""),
            "score": record.get("score", ""),
            "zone_winner": winners.get(row_number, "unknown"),
        }
        if category == winners.get(row_number):
            matched_rows.append(row)
        else:
            misplaced_rows.append({
                **row,
                "expected_row": str(row_number),
                "expected_category": winners.get(row_number, "unknown"),
            })

    row_summary_rows = []
    matched_total = 0
    observed_total = 0
    for _, zone in zone_summary.iterrows():
        row_number = int(zone["row"])
        winner = zone["winner_category"]
        observed = observed_by_row.get(row_number, [])
        observed_total += len(observed)
        matched_count = sum(1 for category in observed if category == winner)
        matched_total += matched_count
        row_summary_rows.append({
            "row": row_number,
            "expected": winner,
            "observed": ", ".join(observed),
            "matched": matched_count,
            "expected_count": len(observed),
            "zone_winner": winner,
            "winner_share": zone["winner_share"],
            "top_categories": zone["top_categories"],
        })

    empty = pd.DataFrame()
    score = matched_total / observed_total if observed_total else 0.0
    return ComplianceResult(
        score=score,
        matched=pd.DataFrame(matched_rows),
        misplaced=pd.DataFrame(misplaced_rows),
        missing=empty,
        unexpected=empty,
        row_summary=pd.DataFrame(row_summary_rows),
    )


def evaluate(records: list[dict], planogram_rows: list[list[str]]) -> ComplianceResult:
    if not records or not planogram_rows:
        empty = pd.DataFrame()
        return ComplianceResult(0.0, empty, empty, empty, empty, empty)

    enriched = infer_shelf_rows(records, len(planogram_rows))
    matched_rows = []
    misplaced_rows = []
    unexpected_rows = []

    expected_by_category = {}
    for row_number, expected_categories in enumerate(planogram_rows, start=1):
        for category in expected_categories:
            expected_by_category.setdefault(category, set()).add(row_number)

    observed_by_row: dict[int, list[str]] = {i: [] for i in range(1, len(planogram_rows) + 1)}
    for record in enriched:
        category = _record_category(record)
        row_number = int(record["planogram_row"])
        observed_by_row.setdefault(row_number, []).append(category)
        row = {
            "crop_id": record.get("crop_id"),
            "row": row_number,
            "category": category,
            "subcategory": record.get("subcategory", ""),
            "product": record.get("product_name", ""),
            "brand": record.get("brand", ""),
            "score": record.get("score", ""),
        }
        if category in planogram_rows[row_number - 1]:
            matched_rows.append(row)
        elif category in expected_by_category:
            expected_rows = sorted(expected_by_category[category])
            misplaced_rows.append({**row, "expected_row": ", ".join(map(str, expected_rows))})
        else:
            unexpected_rows.append(row)

    missing_rows = []
    row_summary_rows = []
    capped_matched_total = 0
    for row_number, expected_categories in enumerate(planogram_rows, start=1):
        observed_counts = _category_counts(observed_by_row.get(row_number, []))
        expected_counts = _category_counts(expected_categories)
        matched_count = 0
        for category, expected_count in expected_counts.items():
            observed_count = observed_counts.get(category, 0)
            capped_match = min(expected_count, observed_count)
            matched_count += capped_match
            capped_matched_total += capped_match
            missing_count = max(0, expected_count - observed_count)
            if missing_count:
                missing_rows.append({"row": row_number, "category": category, "missing_count": missing_count})
        row_summary_rows.append({
            "row": row_number,
            "expected": ", ".join(expected_categories),
            "observed": ", ".join(observed_by_row.get(row_number, [])),
            "matched": matched_count,
            "expected_count": sum(expected_counts.values()),
        })

    expected_total = sum(len(row) for row in planogram_rows)
    score = (capped_matched_total / expected_total) if expected_total else 0.0
    return ComplianceResult(
        score=score,
        matched=pd.DataFrame(matched_rows),
        misplaced=pd.DataFrame(misplaced_rows),
        missing=pd.DataFrame(missing_rows),
        unexpected=pd.DataFrame(unexpected_rows),
        row_summary=pd.DataFrame(row_summary_rows),
    )
