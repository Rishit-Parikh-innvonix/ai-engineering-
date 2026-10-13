"""
The product catalog itself: loading and querying the data. Deliberately has no MCP and no AI
in it - it's plain functions over a JSON file, the same separation week7/week8 use (logic.py /
logic.ts) so the actual business rules can be unit-tested without a server or a model running.

search_products is backed by SQLite's FTS5 (a real inverted-text index, stdlib-only) instead of
scanning every product's fields on every call. A linear scan is fine at 8 products but is exactly
the part of this project that stops scaling once the catalog has thousands of rows - FTS5 looks
matches up through an index instead of reading every row, and ranks results (bm25) instead of
returning them in whatever order they happen to appear in the file.
"""

import json
import sqlite3
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


@lru_cache(maxsize=1)
def _search_index() -> sqlite3.Connection:
    # Built once per process from the same data _load() returns, then reused for every search -
    # the indexing cost is paid once, not on every call, same as _load()'s own caching.
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("CREATE VIRTUAL TABLE products USING fts5(id UNINDEXED, name, category, description)")
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?)",
        [(p["id"], p["name"], p["category"], p["description"]) for p in _load()],
    )
    return conn


# Common English filler words, dropped before searching - otherwise one of them can accidentally
# word-match inside a product's own name/description (e.g. the "in" in a question like "wireless
# headphones in stock" literally appears inside "Learn Python in a Weekend") and pull in an
# unrelated product. A short, fixed list is enough here; a bigger catalog would want a real
# stopword library, but the principle - don't let filler words count as search terms - is the
# same at any scale.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does", "for", "from",
    "has", "have", "how", "i", "in", "is", "it", "me", "my", "now", "of", "on", "right", "show",
    "tell", "that", "the", "this", "to", "want", "what", "where", "which", "who", "will", "with",
    "you", "your", "about",
}


def _fts_query(text: str) -> str:
    # OR each remaining word together as its own quoted phrase: a query can be a short keyword
    # ("fast charging") or a full natural-language question ("is the mechanical keyboard
    # available right now") - OR-of-words plus bm25 ranking finds the best-matching row either
    # way, since the words that DO match (e.g. "mechanical", "keyboard") surface the right
    # product regardless of what else is in the sentence. Quoting each word also neutralizes
    # FTS5's own query syntax (-, *, : etc. all mean something to it otherwise) so arbitrary user
    # text can't be misread as a query operator.
    tokens = [tok.replace('"', '""') for tok in text.strip().split() if tok.lower() not in _STOPWORDS]
    return " OR ".join(f'"{tok}"' for tok in tokens)


def search_products(query: str) -> list[Product]:
    """Full-text search across name, category and description, ranked by relevance (bm25).
    Backed by an FTS5 index (see _search_index) instead of scanning every product on every
    call, so this stays fast as the catalog grows from 8 products to thousands."""
    match = _fts_query(query)
    if not match:
        return []
    rows = _search_index().execute(
        "SELECT id FROM products WHERE products MATCH ? ORDER BY bm25(products) LIMIT 50",
        (match,),
    ).fetchall()
    by_id = {p["id"]: p for p in _load()}
    return [by_id[row[0]] for row in rows if row[0] in by_id]


def check_stock(product_id: str) -> str:
    product = get_product(product_id)
    if product is None:
        return f'No product found with id "{product_id}".'
    if product["stock"] == 0:
        return f'{product["name"]} ({product_id}) is OUT OF STOCK.'
    return f'{product["name"]} ({product_id}) has {product["stock"]} unit(s) in stock.'
