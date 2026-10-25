"""
Exercise 2: connect an LLM client to the catalog MCP server and let it retrieve information
dynamically, instead of hardcoding "if question mentions X, return Y" logic ourselves.

The client never imports catalog.py directly - it only knows the server exists and has some
tools. That's the point of MCP: the AI decides when to call search_products/check_stock based
on their name and description alone, the same way week6's assistant used the filesystem and
memory servers without that project's code ever importing them.
"""

import json
import sys

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from catalog_mcp import laya_router
from catalog_mcp.models import get_cloud_chat_model

SYSTEM_PROMPT = (
    "You are a store assistant. Answer the customer's question using ONLY the catalog tools - "
    "never guess a price or stock count from memory. If search_products finds nothing, say so "
    "plainly instead of inventing a product."
)


def build_client() -> MultiServerMCPClient:
    # stdio transport: this starts catalog_mcp/server.py as its own subprocess (via `-m`, so its
    # package imports resolve) and talks to it over stdin/stdout - the same mechanism week6's
    # TypeScript MultiServerMCPClient used for the filesystem and memory servers.
    return MultiServerMCPClient(
        {
            "catalog": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "catalog_mcp.server"],
            }
        }
    )


async def ask(client: MultiServerMCPClient, question: str) -> None:
    tools = await client.get_tools(server_name="catalog")
    agent = create_agent(get_cloud_chat_model(), tools, system_prompt=SYSTEM_PROMPT)

    result = await agent.ainvoke({"messages": [HumanMessage(question)]})
    final_message = result["messages"][-1]
    print(f"Q: {question}")
    print(f"A: {final_message.content.strip()}\n")


def _as_products(raw) -> list[dict]:
    # langchain-mcp-adapters returns a list tool result as one content block per item - e.g.
    # [{"type": "text", "text": '{"id": "p1", ...}'}, ...], not a single JSON array - so each
    # block's "text" needs parsing separately to get back the actual product dicts.
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        return [json.loads(block["text"]) for block in raw]
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict) and "error" in raw:
        return []
    return raw if isinstance(raw, list) else [raw]


# Which tool(s) the LLM actually needs for each action Laya can pick - this is the real payoff
# of the classifier: it narrows the toolset handed to the LLM, instead of handing it everything
# every time. "Which exact product" and anything requiring arithmetic ("under $50") is resolved
# by the LLM itself via these tools - that's a free-text/reasoning job, not a classification job,
# so it belongs to the LLM, not to Laya or to plain regex/substring code standing in for it.
_ACTION_TOOLS = {
    "search_products": ["search_products"],
    "one_product": ["search_products"],
    "check_stock": ["search_products", "check_stock"],
}


async def ask_with_laya(client: MultiServerMCPClient, question: str) -> None:
    """Exercise 3: Laya decides ONLY the action (a true 4-way multiple-choice question - see
    laya_router.py). For "all_products" that's the whole answer, no LLM needed. For everything
    else, Laya's pick narrows which tool(s) the cloud LLM agent gets to see, then the LLM - not
    Laya, not plain code - resolves the exact product, applies any price filter, and writes the
    final answer, exactly the way Exercise 2's ask() already does, just with a smaller toolset."""
    print(f"Q: {question}")
    decision = laya_router.decide(question)
    action = decision["action"]

    if action == "all_products":
        resources = await client.get_resources("catalog", uris="catalog://products")
        products = _as_products(resources[0].as_string())
        names = ", ".join(p["name"] for p in products)
        print(f"A: Here's everything we sell: {names}.\n")
        return

    all_tools = await client.get_tools(server_name="catalog")
    allowed_names = _ACTION_TOOLS.get(action, [t.name for t in all_tools])
    scoped_tools = [t for t in all_tools if t.name in allowed_names]

    agent = create_agent(get_cloud_chat_model(), scoped_tools, system_prompt=SYSTEM_PROMPT)
    result = await agent.ainvoke({"messages": [HumanMessage(question)]})
    final_message = result["messages"][-1]
    print(f"A: {final_message.content.strip()}\n")
