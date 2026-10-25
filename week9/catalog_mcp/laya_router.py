"""
Exercise 3: use Laya (a small, local, non-generative "typed decision" model - see
https://github.com/NandhaKishorM/laya) to decide the ACTION for a catalog question - which
tool(s)/resource the request needs - instead of sending the whole decision to a cloud LLM.

Laya does a single forward pass over text and returns a calibrated probability per labeled
option from a FIXED set you give it - it classifies, it never generates or extracts free text.
That's the ONLY thing asked of it here: "search / check stock / browse all / look up one
product" is a true 4-way multiple-choice question.

Two earlier versions of this file tried to make Laya (or plain regex/substring code standing in
for it) also resolve the EXACT product or apply a price filter - both are free-text understanding
or arithmetic, not classification, and both failed for that reason (see README.md's "Exercise 3"
section for the full story). The fix: Laya only narrows which tool(s) the LLM gets to see;
client.py's ask_with_laya() hands the question to the cloud LLM agent with ONLY that narrowed
toolset, and the LLM - which is actually built for free-text extraction and reasoning - resolves
the product, price filter, or any ambiguity itself, via real tool calls.
"""

from functools import lru_cache

from laya import Router

ACTIONS = {
    "search_products": "Search the catalog by name, category, or description keywords",
    "check_stock": "Check how many units of one specific product are currently in stock",
    "all_products": "Browse the entire catalog with no filtering or search",
    "one_product": "Look up one specific product's full details by its exact id",
}


@lru_cache(maxsize=1)
def _questions() -> dict:
    return {
        "action": {
            "type": "choice",
            "instructions": "Which catalog action should handle this customer request?",
            "criteria": ACTIONS,
        },
    }


@lru_cache(maxsize=1)
def _get_router() -> Router:
    # Lazy + cached: constructing this loads/downloads the checkpoint, so tests or other
    # exercises that never call decide() never pay that cost.
    return Router(device="cpu")


def decide(question: str) -> dict:
    """Returns, e.g.: {"action": "check_stock", "action_confidence": 0.84}"""
    result = _get_router().predict({"question": question}, _questions())
    answer = result["answers"]["action"]
    return {"action": answer["choice"], "action_confidence": answer.get("confidence")}
