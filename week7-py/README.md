# Week 7 (Python) - LangGraph: a Research Brief Builder

A Python port of `../week7` (TypeScript), using Python LangGraph. Give it a topic. Five AI agents
(routing, planning, retrieval, summarization, final answer), coordinated by a LangGraph workflow,
research it across Wikipedia, arXiv and Hacker News and save a **markdown brief with numbered
citations and real links** under `reports/`. Questions that need **live data** ("current weather
in Ahmedabad") are routed to a tool instead of the search sources, which cannot answer them.

```
cd week7-py
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m brief "How do LangGraph agents differ from LangChain agents?"
python -m brief "give me current weather in ahmedabad?"
pytest                      # 49 tests, no network and no API key needed
```

Needs Python 3.11+. The key `OPENROUTER_API_KEY` is read from `week7-py/.env`, else the repo-root
`.env`, else `../week7/.env` (see `.env.example`). Only `:free` OpenRouter models are used.
No other keys are needed - all sources and the weather API are free and keyless.

## How it runs

```
topic
  |
ROUTER (AI) ------------ RESEARCH | WEATHER: <city> | MIXED: <city> | <rest> | UNSUPPORTED_LIVE
  |  weather -> WEATHER TOOL (code: Open-Meteo) -> summarizer            (no searching)
  |  mixed   -> WEATHER TOOL -> planner (non-weather part only), one library for both
  |  unsupported -> honest "needs live data, no tool for it" report -> save
  |  research (and any failure of the router or the tool) -> below
  v
PLANNER (AI) ---------> 3 sub-questions
  |  one parallel branch per sub-question (LangGraph `Send`)
RETRIEVAL (AI + code) - queries per source, searches all 3 at once, an AI judge drops off-topic results
  |
COVERAGE CHECK (code) -- nothing relevant for some sub-question? retry those (max 3 rounds)
  |                       nothing at all? honest "no sources" report
SUMMARIZER (AI) -------> short notes, each bullet tagged [n]
  |
ANSWER WRITER (AI) ----> the brief, using only those notes
  |
CITATION CHECK (code) -- a [n] that doesn't exist? back to the writer (max 3 drafts)
  |
SAVE (code) -----------> reports/<topic>-<timestamp>.md
```

You draw the flowchart; the AI only fills in the content of each box. The boxes never talk to each
other: they read and write **one shared notebook** (the state) and LangGraph decides who goes next.

## TypeScript -> Python map

| Concept | TypeScript (`week7/src`) | Python (`week7-py/brief`) |
|---|---|---|
| State | `StateSchema` + `ReducedValue` | `TypedDict` + `Annotated[list, operator.add]` (`state.py`) |
| Defaults | filled in by the schema | `initial_state(topic)` - a TypedDict has none |
| Nodes / edges | `graph.ts` `addNode/addEdge` | `graph.py` `add_node/add_edge` |
| Conditional routing | `afterCoverage` ... | `after_route`, `after_weather`, `after_coverage` ... |
| Parallel fan-out | `Send` per sub-question | `langgraph.types.Send`, branches use `asyncio.gather` |
| Validation | zod | pydantic v2 |
| HTTP | `fetch` | `httpx.AsyncClient` |
| arXiv XML | fast-xml-parser | stdlib `xml.etree` |
| Agents as arguments | `Agents` interface | `Agents` dataclass (tests swap in scripted ones) |
| Tests | `node --test` | `pytest` |

Python-specific things worth knowing: LangGraph forbids a node with the same name as a state key
(so node `router` vs key `route`, as in the TS version); Python's `re` has no `\p{L}`, so "a
letter in any language" is `[^\W\d_]`; strict parsers use `re.fullmatch` because `$` also matches
before a trailing newline; weather numbers use `:g` so `34.0` prints as `34`; the API-key check
happens when the model is created, not at import, so tests need no key.

## What the safeguards do - and don't - guarantee

- **Citations are real numbers.** The citation check rejects `[n]` markers that don't point to a
  retrieved source, and the Sources list is built by code from what was actually fetched - the AI
  never writes a link or a weather number. Only sources the text cites are listed.
- **A citation does not prove the claim.** The check confirms `[3]` exists, not that source 3 says
  what the sentence says. Treat the brief as a well-sourced starting point. Hacker News results
  are developers' opinions, not facts.
- **The AI chooses the route, code checks it.** The router answers in one strictly parsed line;
  an unparseable reply is asked again, and if the router or the weather tool still fails the run
  falls back to plain research and *About this run* says so. Measured live: 10/10 on a sample.
- **Nothing is invented when nothing is found**, and **failures degrade instead of crashing**: a
  source being down, or the summarizer/writer timing out, never throws away the research.
- **Free models come and go.** `CLOUD_MODEL` in `brief/config.py` is the one line to change.

## Files

```
week7-py/
  brief/
    research.py   the command (python -m brief)     graph.py    the flowchart
    state.py      the shared notebook               logic.py    pure decisions (tested)
    types.py      shared data shapes (pydantic)     config.py, models.py, report.py, utils.py
    agents/       router, planner, retrieval (queries + relevance judge), summarizer, writer, llm (retries)
    sources/      wikipedia, arxiv, hackernews, http, text
    tools/        weather.py (Open-Meteo geocoding + current conditions)
  tests/          test_graph.py (real graph, scripted agents), test_logic.py, test_sources.py
```

## Note
`RESEARCH_BRIEF_CONTACT` (optional, in `.env`): Wikipedia now rejects generic User-Agents (HTTP 403), so requests carry a contact URL. See `.env.example`.
