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
from langchain_core.messages import AIMessage, HumanMessage
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


def log_tool_calls(messages: list) -> None:
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"  -> agent called {call['name']}({call['args']})")


async def ask(client: MultiServerMCPClient, question: str) -> None:
    tools = await client.get_tools(server_name="catalog")
    agent = create_agent(get_cloud_chat_model(), tools, system_prompt=SYSTEM_PROMPT)

    print(f"You ask: {question}")
    result = await agent.ainvoke({"messages": [HumanMessage(question)]})
    log_tool_calls(result["messages"])
    final_message = result["messages"][-1]
    print(f"\nAssistant replies:\n{final_message.content}\n")


async def _call_tool(tools: list, name: str, args: dict) -> str:
    tool = next(t for t in tools if t.name == name)
    return await tool.ainvoke(args)


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


async def ask_with_laya(client: MultiServerMCPClient, question: str) -> None:
    """Exercise 3: action and category come from Laya's local classifier in one forward pass.
    "Which exact product" is deliberately NOT asked of Laya - see laya_router.py for why that
    doesn't scale - it's resolved by a real search call instead (catalog.search_products, via
    the search_products tool), the same way a real system would handle "which one of
    potentially thousands of rows does this match" rather than a fixed classification label."""
    tools = await client.get_tools(server_name="catalog")

    print(f"You ask: {question}")
    decision = laya_router.decide(question)
    action = decision["action"]
    category = decision["category"]
    print(
        f"  -> action={action} ({decision['action_confidence']})  "
        f"category={category} ({decision['category_confidence']})"
    )

    if action == "all_products":
        resources = await client.get_resources("catalog", uris="catalog://products")
        print(f"\nResult:\n{resources[0].as_string()}\n")
        return

    if action == "search_products":
        # Browsing: category (a guaranteed-valid, bounded label) is a good search query when
        # Laya found one; fall back to the raw question text otherwise - real search handles
        # arbitrary free text fine, which is exactly the part a fixed-label classifier can't do.
        search_query = category if category != laya_router.NO_CATEGORY else question
        result = await _call_tool(tools, "search_products", {"query": search_query})
        print(f"\nResult (searched {search_query!r}):\n{result}\n")
        return

    # check_stock / one_product both need exactly ONE specific product, so the search query must
    # be the specific words in the question itself (e.g. "wireless headphones"), NOT the broad
    # category - "Electronics" alone matches several products and can only ever be ambiguous
    # here. Resolve via search instead of a classification label, then handle the
    # ambiguous/no-match cases explicitly - a real system can't just guess which one was meant.
    matches = _as_products(await _call_tool(tools, "search_products", {"query": question}))
    if not matches:
        print(f"\nResult: no product matches {question!r}.\n")
        return
    if len(matches) > 1:
        names = ", ".join(f'{m["name"]} ({m["id"]})' for m in matches)
        print(f"\nResult: {question!r} matches more than one product - {names}. Ask about one of these by name.\n")
        return
    product_id = matches[0]["id"]

    if action == "check_stock":
        result = await _call_tool(tools, "check_stock", {"product_id": product_id})
        print(f"\nResult:\n{result}\n")
    else:  # one_product
        resources = await client.get_resources("catalog", uris=f"catalog://products/{product_id}")
        print(f"\nResult:\n{resources[0].as_string()}\n")
