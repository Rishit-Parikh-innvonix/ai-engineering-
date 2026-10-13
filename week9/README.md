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

## Exercise 3 - Laya decides, not the LLM (`catalog_mcp/laya_router.py`)

Exercise 2's LLM does several jobs in one tool-calling step: decide WHICH tool a question
needs, AND extract the right argument for it (a search term, a product id). This exercise
replaces that whole decision layer with [Laya](https://github.com/NandhaKishorM/laya) - a
small, local, non-generative "typed decision" model (421M params, runs on CPU here) that does
a single forward pass over text and returns a calibrated probability per labeled option. It
classifies; it never generates or extracts free text.

**Two versions of this exercise got the product question wrong, in opposite directions - both
worth knowing about.** Version 1 used Laya only for *action*, then tried to patch "which
product?" with regex and plain-substring search over the raw question - asking plain Python to
extract free text Laya can't do. That failed outright: a full sentence never substring-matches
a product field. Version 2 overcorrected: it gave Laya one classification label *per product*,
which worked at 8 products but doesn't scale - at thousands of products you'd be scoring the
question against thousands of candidate labels in a single forward pass, which is both slow and
more error-prone (look-alike descriptions start out-competing the right one).

**The actual fix: Laya only answers what stays small forever - `action` and `category`.**
Categories are bounded (a handful, maybe a few dozen, no matter how large the catalog grows),
so that stays a correct classification question at any scale. "Which exact product" is handed
instead to `catalog.search_products` - a real search index (SQLite FTS5, see `catalog.py`) built
for exactly the kind of lookup that grows with the data, the same tool Exercise 2 already uses.
`laya_router.py` builds the category label set directly from the live catalog data, so whatever
Laya picks is always a guaranteed-valid category string.

Verified live:

| Question | action | category | Result |
|---|---|---|---|
| "Is the mechanical keyboard available right now?" | `one_product` (0.05) | Electronics (0.32) | full product details for `p2`, including `"stock": 0` |
| "Tell me about the USB-C charging cable" | `one_product` (0.01) | Electronics (0.39) | full product details for `p3` |
| "What electronics do you sell for under $50?" | `search_products` (0.17) | **Electronics (0.86)** | all 4 electronics products returned (price filtering still needs Exercise 2's LLM - see below) |
| "Do you have wireless headphones in stock?" | `check_stock` (0.84) | Electronics (0.62) | **ambiguous** - "wireless" matches both Headphones (p1) and Earbuds (p7); reported honestly instead of guessing |

`client.py`'s `ask_with_laya()`: for `search_products` (browsing), it searches by `category`
when Laya found one, else the raw question. For `check_stock`/`one_product` (need exactly ONE
product), it always searches using the raw question text - not the category, which is too
broad and would almost always match several products - then explicitly handles 0 matches ("no
product matches..."), exactly 1 match (resolved), or 2+ matches (reports the names and asks the
user to be more specific, rather than silently picking one).

**What this still doesn't do, honestly:**
- *Price filtering* ("under $50") is never applied here - Exercise 2's LLM did that itself when
  writing its final answer. A price range isn't a label from a fixed set, it's arithmetic over
  numbers - a real, structural gap, not a bug to patch with more `choice` questions. That's the
  line between "classify among known options" (what Laya does well) and "reason over arbitrary
  values" (what still needs something generative).
- *Shared-word ambiguity* ("wireless" matching both the headphones and the earbuds) is reported
  as ambiguous rather than resolved, because a keyword search genuinely can't tell which one a
  human would obviously mean - that distinction needs understanding, not just word overlap. A
  generative LLM (Exercise 2) resolves this correctly because it actually reads the phrase
  "wireless headphones" as a unit, not as two independent keywords.
- A small, fixed stopword list (`_STOPWORDS` in `catalog.py`) keeps filler words ("the", "in",
  "is", ...) from accidentally word-matching inside an unrelated product's name - without it,
  "...in stock?" would spuriously match "Learn Python **in** a Weekend". A bigger catalog would
  want a real stopword library, but the principle holds at any scale.

One more honest caveat, confirmed live: this checkpoint logs
`RuntimeWarning: laya: this checkpoint ships invalid temperatures or values outside [0.5, 5]`
on load - some of its confidence values get clamped and are explicitly flagged as
uncalibrated. Treat any single confidence number with some skepticism rather than as a precise
probability.

**The actual lesson:** a local classifier correctly resolves anything that's genuinely a choice
among a small, known set - action and category qualify at any catalog size. "Which exact
product" is a search/ranking problem, not a classification problem, the moment the catalog is
large - and routing it to a real search index instead of a per-item label is what makes this
design hold up from 8 products to thousands, instead of only working as a demo.

```
python -m catalog_mcp       # runs Exercise 3 automatically after Exercise 1 and 2
```

First run downloads Laya's checkpoint from Hugging Face Hub (~5 files, took a few minutes
here); cached under `~/.cache/huggingface` after that, so later runs load it near-instantly.

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
    catalog.py              pure search/stock logic - no MCP, no AI (search is FTS5-backed)
    server.py                the MCP server (Exercise 1): resources + tools
    client.py                 the MCP client + agent (Exercise 2) + ask_with_laya (Exercise 3)
    laya_router.py             Laya "choice" question + action routing (Exercise 3)
    __main__.py                python -m catalog_mcp entry point
    config.py, models.py        OpenRouter key/model (same pattern as week7-py/week8)
  tests/
    test_catalog.py          8 tests for catalog.py, no MCP/AI/network involved
```
