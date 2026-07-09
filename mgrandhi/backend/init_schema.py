"""Initialize the SKU / inventory / checkout SQLite schema.

This is intentionally conservative:
  - it only creates missing tables/indexes (`CREATE TABLE IF NOT EXISTS`),
  - it never deletes existing data,
  - optional demo products are inserted with `INSERT OR IGNORE`.

Usage:
    python -m backend.init_schema --db inventory.db
    python -m backend.init_schema --db inventory.db --seed-demo-products
    python -m backend.init_schema --db inventory.db --show-tables
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_DB = Path(__file__).resolve().parents[1] / "inventory.db"

DEMO_PRODUCTS = [
    {
        "sku_code": "DEMO-COKE-330ML",
        "barcode": "000000000001",
        "brand": "Coca-Cola",
        "product_name": "Coca-Cola Can",
        "category": "Carbonated Soft Drinks",
        "subcategory": "Cola",
        "package_size": "330 ml",
        "unit_price": 1.49,
        "tax_rate": 0.08,
    },
    {
        "sku_code": "DEMO-WATER-500ML",
        "barcode": "000000000002",
        "brand": "Generic",
        "product_name": "Bottled Water",
        "category": "Bottled Water",
        "subcategory": "Mineral water",
        "package_size": "500 ml",
        "unit_price": 0.99,
        "tax_rate": 0.0,
    },
    {
        "sku_code": "DEMO-SOAP-BAR",
        "barcode": "000000000003",
        "brand": "Generic",
        "product_name": "Bath Soap Bar",
        "category": "Soap & Body Wash",
        "subcategory": "Bath soap",
        "package_size": "1 bar",
        "unit_price": 2.49,
        "tax_rate": 0.08,
    },
]


def init_schema(db_path: str | Path = DEFAULT_DB, seed_demo_products: bool = False) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema = SCHEMA_PATH.read_text()
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(schema)
        if seed_demo_products:
            _seed_demo_products(conn)


def _seed_demo_products(conn: sqlite3.Connection) -> None:
    conn.executemany(
        """
        INSERT OR IGNORE INTO products (
            sku_code, barcode, brand, product_name, category, subcategory,
            package_size, unit_price, tax_rate
        )
        VALUES (
            :sku_code, :barcode, :brand, :product_name, :category, :subcategory,
            :package_size, :unit_price, :tax_rate
        )
        """,
        DEMO_PRODUCTS,
    )
    for product in DEMO_PRODUCTS:
        product_id = conn.execute(
            "SELECT id FROM products WHERE sku_code = ?", (product["sku_code"],)
        ).fetchone()[0]
        aliases = {
            product["product_name"],
            product["brand"],
            f"{product['brand']} {product['product_name']}",
            product["category"],
        }
        conn.executemany(
            """
            INSERT OR IGNORE INTO product_aliases (
                product_id, alias_text, alias_type, confidence, source
            )
            VALUES (?, ?, 'manual', 1.0, 'seed_demo_products')
            """,
            [(product_id, alias) for alias in aliases if alias],
        )


def list_tables(db_path: str | Path = DEFAULT_DB) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    return [row[0] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path.")
    parser.add_argument(
        "--seed-demo-products",
        action="store_true",
        help="Insert a few demo products with aliases using INSERT OR IGNORE.",
    )
    parser.add_argument(
        "--show-tables",
        action="store_true",
        help="Print table names after initialization.",
    )
    args = parser.parse_args()

    init_schema(args.db, seed_demo_products=args.seed_demo_products)
    print(f"Initialized schema: {args.db}")
    if args.show_tables:
        for table in list_tables(args.db):
            print(f"  - {table}")


if __name__ == "__main__":
    main()
