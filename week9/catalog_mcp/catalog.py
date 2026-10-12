"""
The product catalog itself: loading and querying the data. Deliberately has no MCP and no AI
in it - it's plain functions over a JSON file, the same separation week7/week8 use (logic.py /
logic.ts) so the actual business rules can be unit-tested without a server or a model running.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing_extensions import TypedDict

from catalog_mcp import config


class Product(TypedDict):
    id: str
    name: str
    category: str
    price: float
    stock: int
    description: str


@lru_cache(maxsize=1)
def _load(path: Path = config.PRODUCTS_FILE) -> list[Product]:
    # Cached: the file is read once per process, not on every tool call.
    return json.loads(path.read_text(encoding="utf-8"))


def list_products() -> list[Product]:
    return _load()


def get_product(product_id: str) -> Product | None:
    return next((p for p in _load() if p["id"] == product_id), None)


def search_products(query: str) -> list[Product]:
    """Matches a query against name, category and description - a plain substring search,
    not a vector search. Good enough for a small catalog, and keeps this exercise about MCP
    rather than about search/ranking algorithms."""
    needle = query.strip().lower()
    if not needle:
        return []
    return [
        p
        for p in _load()
        if needle in p["name"].lower() or needle in p["category"].lower() or needle in p["description"].lower()
    ]


def check_stock(product_id: str) -> str:
    product = get_product(product_id)
    if product is None:
        return f'No product found with id "{product_id}".'
    if product["stock"] == 0:
        return f'{product["name"]} ({product_id}) is OUT OF STOCK.'
    return f'{product["name"]} ({product_id}) has {product["stock"]} unit(s) in stock.'
