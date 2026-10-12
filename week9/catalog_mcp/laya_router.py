"""
Exercise 3: use Laya (a small, local, non-generative "typed decision" model - see
https://github.com/NandhaKishorM/laya) to decide what a catalog question needs, instead of
sending it to a cloud LLM and letting the LLM decide via tool-calling.

Laya does a single forward pass over text and returns a calibrated probability per labeled
option from a FIXED set you give it - it classifies, it never generates or extracts free text.
The first version of this exercise used Laya only to pick the ACTION (search/check_stock/etc.),
then tried to patch the leftover problem - "which product?", "which search term?" - with regex
and a plain-substring catalog search. That's the wrong shape for this model: it was asking Laya
to pick from a fixed label set, then asking plain Python to do free-text extraction, and the
extraction step is what kept failing (a full sentence never substring-matches a product field).

The fix is to stop asking Laya to do extraction at all, even indirectly. This catalog is small
and fixed (8 products, 3 categories) - so "which product does this question mean" and "which
category" are THEMSELVES fixed label sets, exactly what `choice` questions are for. Laya now
answers three choice questions in ONE forward pass - action, product, category - built directly
from the live catalog data, so whatever it picks for product/category is already a guaranteed-
valid id/category string. No regex, no fallback search, no extraction step anywhere.
"""

from functools import lru_cache

from laya import Router

from catalog_mcp import catalog

ACTIONS = {
    "search_products": "Search the catalog by name, category, or description keywords",
    "check_stock": "Check how many units of one specific product are currently in stock",
    "all_products": "Browse the entire catalog with no filtering or search",
    "one_product": "Look up one specific product's full details by its exact id",
}

NO_PRODUCT = "none"
NO_CATEGORY = "none"


@lru_cache(maxsize=1)
def _product_criteria() -> dict[str, str]:
    # Built from the live catalog, not hardcoded - a product added to data/products.json shows
    # up here automatically. Laya's own answer is then guaranteed to be a valid product id (or
    # "none"), since those are the only labels it was ever offered.
    criteria = {p["id"]: f'{p["name"]} - {p["description"]}' for p in catalog.list_products()}
    criteria[NO_PRODUCT] = "No specific product is named or implied by the request"
    return criteria


@lru_cache(maxsize=1)
def _category_criteria() -> dict[str, str]:
    categories = sorted({p["category"] for p in catalog.list_products()})
    criteria = {c: f"Products in the {c} category" for c in categories}
    criteria[NO_CATEGORY] = "No specific category is named or implied by the request"
    return criteria


@lru_cache(maxsize=1)
def _questions() -> dict:
    return {
        "action": {
            "type": "choice",
            "instructions": "Which catalog action should handle this customer request?",
            "criteria": ACTIONS,
        },
        "product": {
            "type": "choice",
            "instructions": "Which specific product, if any, is this request about?",
            "criteria": _product_criteria(),
        },
        "category": {
            "type": "choice",
            "instructions": "Which product category, if any, is this request about?",
            "criteria": _category_criteria(),
        },
    }


@lru_cache(maxsize=1)
def _get_router() -> Router:
    # Lazy + cached: constructing this loads/downloads the checkpoint, so tests or other
    # exercises that never call decide() never pay that cost.
    return Router(device="cpu")


def product_name(product_id: str) -> str:
    """The product's actual name - guaranteed to substring-match itself in catalog.py's plain
    search, unlike a full sentence. Used when a search needs a query derived from a product
    Laya already identified, rather than a category."""
    product = catalog.get_product(product_id)
    return product["name"] if product is not None else product_id


def decide(question: str) -> dict:
    """One forward pass answering all three questions at once (cheaper than three separate
    calls - Laya's own docs recommend batching questions per request where possible).

    Returns, e.g.:
        {"action": "check_stock", "action_confidence": 0.84,
         "product": "p1", "product_confidence": 0.91,
         "category": "none", "category_confidence": 0.5}
    """
    result = _get_router().predict({"question": question}, _questions())
    answers = result["answers"]
    return {
        "action": answers["action"]["choice"],
        "action_confidence": answers["action"].get("confidence"),
        "product": answers["product"]["choice"],
        "product_confidence": answers["product"].get("confidence"),
        "category": answers["category"]["choice"],
        "category_confidence": answers["category"].get("confidence"),
    }
