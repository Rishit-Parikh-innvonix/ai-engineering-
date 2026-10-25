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

## Exercise 3 - Laya routes, the LLM resolves (`catalog_mcp/laya_router.py`)

Exercise 2's LLM does two jobs in one tool-calling step: decide WHICH tool a question needs,
AND extract the right argument for it (a search term, a product id, a price filter). This
exercise splits those two jobs between two different kinds of model, each doing only what it's
actually built for:

- [Laya](https://github.com/NandhaKishorM/laya) - a small, local, non-generative "typed
  decision" model (421M params, runs on CPU here). It does a single forward pass over text and
  returns a calibrated probability per labeled option from a FIXED set. It classifies; it never
  generates or extracts free text.
- The same cloud LLM Exercise 2 uses (via OpenRouter, free tier) - which IS built for free-text
  extraction and reasoning (exact product names, "under $50" arithmetic, disambiguating
  "wireless headphones" from "earbuds").

**Three earlier attempts got this wrong by asking Laya (or plain code standing in for it) to do
the LLM's job - worth knowing about, because the failure mode is the same each time.**
1. Laya picked the action; plain regex + substring search tried to extract "which product" from
   the raw question. Failed outright - a full sentence never substring-matches a product field.
2. Laya was given one classification label *per product*. Worked at 8 products, doesn't scale -
   at thousands of products you'd score the question against thousands of candidate labels in
   one forward pass, both slow and error-prone.
3. Laya picked action + category; "which exact product" was handed to a real search
   (`catalog.search_products`), with a regex (`_price_limit()`) bolted on for "under $50". This
   actually worked for most questions, but broke on genuine ambiguity ("wireless" matching both
   Wireless Headphones and Noise-Cancelling Earbuds) - a keyword search can't tell which a human
   obviously means, because that needs understanding a phrase as a unit, not word overlap.

Each attempt tried to make a classifier (or classifier-shaped plain code) do a free-text/
reasoning job. **The fix: stop trying to resolve entities without an LLM.** Laya decides ONLY
the action - a true 4-way multiple-choice question, exactly its fit - and that decision narrows
which tool(s) the cloud LLM agent even gets to see. The LLM then resolves the exact product,
applies any price filter, and writes the final answer itself, via real tool calls - the same
mechanism Exercise 2 already uses correctly, just scoped to a smaller toolset:

```
question
   │
   ▼
Laya classifies the ACTION (local, ~instant, zero API cost):
   search_products | check_stock | one_product | all_products
   │
   ├─ all_products ───────────────────────► fetch the whole catalog resource directly
   │                                          (unambiguous - no LLM needed at all)
   │
   └─ anything else ──────────────────────► cloud LLM agent, given ONLY the tool(s) that
                                              action needs:
                                                search_products / one_product → [search_products]
                                                check_stock                   → [search_products, check_stock]
                                              The LLM extracts the exact product, applies
                                              "under $50" reasoning, and writes the answer -
                                              a free-text/reasoning job, correctly left to it.
```

`_ACTION_TOOLS` in `client.py` is that narrowing table. This is the realistic production use of
a fast local classifier in front of an LLM: it doesn't try to replace the LLM, it cuts down what
the LLM has to consider before the LLM runs - smaller toolset, less ambiguity, lower cost - while
leaving every free-text/arithmetic/disambiguation job to the model actually built for it.

Verified live, same 3 demo questions as Exercise 2, now all correct with no dead ends:

| Question | Laya action | Result |
|---|---|---|
| "Do you have wireless headphones in stock?" | `check_stock` (0.84) | "Yes, we have Wireless Headphones in stock. There are 12 units available." - the earlier version misfired here (reported "ambiguous" against both Headphones/Earbuds); the LLM resolves it correctly because it reads the phrase as a unit, not as keywords |
| "What electronics do you sell for under $50?" | `search_products` (0.17) | Wireless Headphones ($49.99), USB-C Charging Cable ($9.99) - correctly excludes the $89 keyboard and $59 earbuds, with no regex involved |
| "Is the mechanical keyboard available right now?" | `one_product` (0.05) | "...currently out of stock and not available for purchase at this time." |

One honest caveat, confirmed live: this checkpoint logs
`RuntimeWarning: laya: this checkpoint ships invalid temperatures or values outside [0.5, 5]`
on load - some of its confidence values get clamped and are explicitly flagged as
uncalibrated. Treat any single confidence number with some skepticism rather than as a precise
probability; it doesn't change which action gets picked, it only means the number next to it
isn't perfectly calibrated.

**The actual lesson:** a local classifier is genuinely useful in front of an LLM - not as a
replacement for it, but as a router that narrows scope before the expensive model runs. The
three earlier failures all came from asking the classifier (or plain code pretending to be one)
to do free-text extraction or arithmetic reasoning - jobs that need an LLM. Once Laya's job was
limited to the one thing it's actually built for (a fixed multiple-choice decision), and the LLM
got to do the rest, every demo question resolved correctly.

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
