# Week 9 - MCP: Build Your Own Server

Week 6 *connected to* MCP servers someone else wrote (filesystem, memory). This week flips
that: **you write the MCP server**, and an AI client connects to it. A small online store's
product catalog is exposed as MCP resources and tools, then a LangChain agent uses it to answer
real questions - without any "if the question mentions X, return Y" logic hardcoded anywhere.

## One-time setup

```
cd week9
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Uses the repo-root `.env` (or week8's) for `OPENROUTER_API_KEY` if `week9/.env` doesn't exist
(see `.env.example`). No other keys needed - the catalog is local data, no external API.

## Run

```
python -m catalog_mcp
python -m catalog_mcp "Do you have the Python book in stock?"    # ask your own question
pytest                                                           # 8 tests, no network, no key
```

The first run prints what the server exposes (Exercise 1), then asks 3 demo questions through
an AI agent (Exercise 2). Each run starts the server itself, as a subprocess - no separate
terminal needed.

## Exercise 1 - the MCP server (`catalog_mcp/server.py`)

Built with `FastMCP` - specifically `mcp.server.fastmcp.FastMCP`, the copy built into the
official MCP Python SDK (see "Two FastMCPs" below for why). It exposes:

- **A resource**, `catalog://products` - the whole catalog, for browsing.
- **A resource template**, `catalog://products/{product_id}` - one product's full details.
- **Two tools**, for actions the AI decides to take based on a question:
  - `search_products(query)` - search by name, category or description.
  - `check_stock(product_id)` - real stock count for one product.

The catalog data itself (`data/products.json`, 8 products) and the logic that searches it
(`catalog_mcp/catalog.py`) have nothing to do with MCP - they're plain Python, tested directly
in `tests/test_catalog.py`. `server.py` is a thin layer that exposes those plain functions over
MCP. That split is why the tests run in 0.02s with no server, no model, and no network.

## Exercise 2 - the LLM client (`catalog_mcp/client.py`)

```python
client = MultiServerMCPClient({"catalog": {"transport": "stdio", "command": ..., "args": ["-m", "catalog_mcp.server"]}})
tools = await client.get_tools(server_name="catalog")
agent = create_agent(get_cloud_chat_model(), tools, system_prompt=SYSTEM_PROMPT)
result = await agent.ainvoke({"messages": [HumanMessage(question)]})
```

This is the same `create_agent` pattern week6/week7 used in TypeScript, just in Python. The
client's code never imports `catalog.py` - it only knows the server exists and has some named
tools. The AI decides, per question, whether and which tool to call, based only on each tool's
name and description. Verified live:

- *"Do you have wireless headphones in stock?"* -> calls `search_products`, answers from the
  real stock count (12).
- *"What electronics do you sell for under $50?"* -> calls `search_products("electronics")`,
  then filters by price itself from the real results.
- *"Is the mechanical keyboard available right now?"* -> calls `search_products`, correctly
  reports it as out of stock (the data has `"stock": 0` for that product).

## Two FastMCPs - a real gotcha worth knowing

There are two different `FastMCP` implementations on PyPI:

1. **The standalone `fastmcp` package** (what most tutorials show, currently 4.x) - requires
   `mcp>=2`.
2. **`mcp.server.fastmcp.FastMCP`**, bundled inside the official `mcp` SDK itself - has the
   exact same decorator API (`@mcp.resource`, `@mcp.tool()`, `mcp.run()`).

`langchain-mcp-adapters` (the client library) only supports `mcp<2` as of this writing -
confirmed directly: installing it against `mcp>=2` fails with
`ImportError: cannot import name 'RequestContext' from 'mcp.shared.context'`. Using the
standalone `fastmcp` package for the server would force `mcp>=2` into the same environment the
client needs `mcp<2` in - a real, unresolvable conflict in one virtual environment. Importing
`FastMCP` from the SDK (`mcp.server.fastmcp`) instead avoids the second package entirely, so
there's only one `mcp` version in play. `requirements.txt` has the full story in a comment.

One API difference from the standalone package: the SDK's version requires
`@mcp.tool()` (with parentheses) - `@mcp.tool` alone raises
`TypeError: The @tool decorator was used incorrectly`.

## Why the server is spawned with `-m`, not a file path

```python
"args": ["-m", "catalog_mcp.server"]
```

Not `["catalog_mcp/server.py"]`. Running a `.py` file directly doesn't register its parent
folder as a package, so `from catalog_mcp import catalog` inside `server.py` fails with
`ModuleNotFoundError: No module named 'catalog_mcp'` the moment the subprocess starts. `-m`
runs it as a module, which Python resolves correctly as part of the `catalog_mcp` package.

## Files

```
week9/
  data/
    products.json          the catalog's actual data (8 products)
  catalog_mcp/
    catalog.py              pure search/stock logic - no MCP, no AI
    server.py                the MCP server (Exercise 1): resources + tools
    client.py                 the MCP client + agent (Exercise 2)
    __main__.py                python -m catalog_mcp entry point
    config.py, models.py        OpenRouter key/model (same pattern as week7-py/week8)
  tests/
    test_catalog.py          8 tests for catalog.py, no MCP/AI/network involved
```
