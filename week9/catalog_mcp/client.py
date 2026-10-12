"""
Exercise 2: connect an LLM client to the catalog MCP server and let it retrieve information
dynamically, instead of hardcoding "if question mentions X, return Y" logic ourselves.

The client never imports catalog.py directly - it only knows the server exists and has some
tools. That's the point of MCP: the AI decides when to call search_products/check_stock based
on their name and description alone, the same way week6's assistant used the filesystem and
memory servers without that project's code ever importing them.
"""

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


async def ask_with_laya(client: MultiServerMCPClient, question: str) -> None:
    """Exercise 3: the same questions as ask() above, but every decision - which action, which
    product, which category - comes from Laya's local classifier in one forward pass, instead
    of a cloud LLM deciding via tool-calling. See laya_router.py for why "product" and
    "category" are themselves classification questions here, not something extracted from free
    text afterward - that's what makes this a correct fit for what Laya actually does."""
    tools = await client.get_tools(server_name="catalog")

    print(f"You ask: {question}")
    decision = laya_router.decide(question)
    action = decision["action"]
    print(
        f"  -> action={action} ({decision['action_confidence']})  "
        f"product={decision['product']} ({decision['product_confidence']})  "
        f"category={decision['category']} ({decision['category_confidence']})"
    )

    if action == "all_products":
        resources = await client.get_resources("catalog", uris="catalog://products")
        print(f"\nResult:\n{resources[0].as_string()}\n")
        return

    if action == "search_products":
        # category/product are themselves classification questions over the live catalog (see
        # laya_router.py), so whichever one Laya picked is already a guaranteed-valid
        # category/name string - no regex, no extraction, no substring-match guesswork.
        if decision["category"] != laya_router.NO_CATEGORY:
            query = decision["category"]
        elif decision["product"] != laya_router.NO_PRODUCT:
            query = laya_router.product_name(decision["product"])
        else:
            print(
                "\nResult: Laya found no specific product or category in this request, so "
                "there's no safe search query to run (this is the genuinely ambiguous case - "
                "a fully open-ended search term still needs a generative model, not a "
                "classifier, to invent).\n"
            )
            return
        result = await _call_tool(tools, "search_products", {"query": query})
        print(f"\nResult (searched {query!r}):\n{result}\n")
        return

    # check_stock / one_product: the product choice IS the id, already validated by construction
    # (it was one of the labels Laya was given) - nothing left to resolve.
    product_id = decision["product"]
    if product_id == laya_router.NO_PRODUCT:
        print("\nResult: Laya didn't identify a specific product for this request.\n")
        return

    if action == "check_stock":
        result = await _call_tool(tools, "check_stock", {"product_id": product_id})
        print(f"\nResult:\n{result}\n")
    else:  # one_product
        resources = await client.get_resources("catalog", uris=f"catalog://products/{product_id}")
        print(f"\nResult:\n{resources[0].as_string()}\n")
