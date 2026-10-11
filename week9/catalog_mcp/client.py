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
