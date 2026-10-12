"""
Week 9 - MCP: build a product catalog MCP server, then connect an LLM client to it.

Run:            python -m catalog_mcp
One question:   python -m catalog_mcp "Do you have the Python book in stock?"
"""

import asyncio
import sys

from catalog_mcp.client import ask, ask_with_laya, build_client

DEMO_QUESTIONS = [
    "Do you have wireless headphones in stock?",
    "What electronics do you sell for under $50?",
    "Is the mechanical keyboard available right now?",
]


def print_header(text: str) -> None:
    border = "=" * 70
    print(f"\n{border}\n{text}\n{border}")


async def show_what_the_server_exposes(client) -> None:
    """Exercise 1 check: prove the server really exposes both resources and tools, before any
    AI is involved - the same "read the manual before using the gadget" step week6 did."""
    print_header("Exercise 1: what does the catalog MCP server expose?")

    tools = await client.get_tools(server_name="catalog")
    print(f"\n{len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    catalog_blob = (await client.get_resources("catalog", uris="catalog://products"))[0]
    product_count = catalog_blob.as_string().count('"id"')
    print(f"\nresource catalog://products -> {product_count} products")

    one_blob = (await client.get_resources("catalog", uris="catalog://products/p1"))[0]
    print(f"resource catalog://products/p1 (template) ->\n{one_blob.as_string()}")


async def main() -> None:
    client = build_client()
    await show_what_the_server_exposes(client)

    print_header("Exercise 2: the LLM client retrieves information dynamically")
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO_QUESTIONS
    for question in questions:
        await ask(client, question)

    print_header("Exercise 3: Laya decides which action to take, instead of the LLM")
    for question in questions:
        await ask_with_laya(client, question)


if __name__ == "__main__":
    asyncio.run(main())
