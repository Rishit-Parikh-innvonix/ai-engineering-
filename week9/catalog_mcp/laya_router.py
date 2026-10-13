"""
Exercise 3: use Laya (a small, local, non-generative "typed decision" model - see
https://github.com/NandhaKishorM/laya) to decide the ACTION and CATEGORY for a catalog
question, instead of sending it to a cloud LLM and letting it decide via tool-calling.

Laya does a single forward pass over text and returns a calibrated probability per labeled
option from a FIXED set you give it - it classifies, it never generates or extracts free text.

"Which exact product, out of N" is deliberately NOT one of Laya's questions. The first version
of this exercise gave Laya one label per product - fine at 8 products, but it doesn't scale: at
thousands of products you'd be scoring the question against thousands of candidate labels in a
single forward pass (slower, and more likely to misfire as look-alike descriptions start
out-competing the right one). "Which category" stays a safe fit for Laya at any catalog size -
categories are bounded (a handful, maybe a few dozen) no matter how many products exist. So this
router only asks Laya the two questions that stay small and fixed - action, category - and hands
"which exact product" to catalog.search_products instead (see catalog.py), a real search index
built for exactly the kind of lookup that grows with the data.
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

NO_CATEGORY = "none"


@lru_cache(maxsize=1)
def _category_criteria() -> dict[str, str]:
    # Built from the live catalog, not hardcoded - a new category shows up here automatically,
    # and this list stays small regardless of how many products exist.
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


def decide(question: str) -> dict:
    """One forward pass answering both questions at once (cheaper than two separate calls -
    Laya's own docs recommend batching questions per request where possible).

    Returns, e.g.:
        {"action": "check_stock", "action_confidence": 0.84,
         "category": "none", "category_confidence": 0.5}
    """
    result = _get_router().predict({"question": question}, _questions())
    answers = result["answers"]
    return {
        "action": answers["action"]["choice"],
        "action_confidence": answers["action"].get("confidence"),
        "category": answers["category"]["choice"],
        "category_confidence": answers["category"].get("confidence"),
    }
