"""
Exercise 1: a product catalog exposed as an MCP server, built with FastMCP.

Resources are for BROWSING (read the whole catalog, or one product by id) - the AI client can
look at these the same way it would open a web page. Tools are for ACTIONS the AI decides to
take based on a question (search, check stock) - these take arguments and run real code, same
distinction week5/week6 already drew between a tool and a plain lookup.

Run directly for a quick manual check:  python -m catalog_mcp.server
Normally it's started automatically, as a subprocess, by catalog_mcp/client.py (same stdio
pattern week6's TypeScript MultiServerMCPClient uses - the client spawns this script and talks
to it over stdin/stdout, no network port involved).
"""

from mcp.server.fastmcp import FastMCP

from catalog_mcp import catalog

mcp = FastMCP("product-catalog")


@mcp.resource("catalog://products")
def all_products() -> list[catalog.Product]:
    """The full product catalog - every product, unfiltered."""
    return catalog.list_products()


@mcp.resource("catalog://products/{product_id}")
def one_product(product_id: str) -> catalog.Product | dict[str, str]:
    """One product's full details, by id (e.g. catalog://products/p1)."""
    product = catalog.get_product(product_id)
    return product if product is not None else {"error": f'No product found with id "{product_id}".'}


@mcp.tool()
def search_products(query: str) -> list[catalog.Product]:
    """Search the catalog by name, category or description (e.g. "headphones", "Home", "desk")."""
    return catalog.search_products(query)


@mcp.tool()
def check_stock(product_id: str) -> str:
    """Check how many units of one product (by id, e.g. "p1") are currently in stock."""
    return catalog.check_stock(product_id)


if __name__ == "__main__":
    mcp.run()
