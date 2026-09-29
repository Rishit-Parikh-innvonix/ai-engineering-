# Week 7 - LangGraph: a Research Brief Builder

Give it a topic. Five AI agents (routing, planning, retrieval, summarization, final answer),
coordinated by a LangGraph workflow, research it across Wikipedia, arXiv and Hacker News
and save a **markdown brief with numbered citations and real links** under `reports/`.
Questions that need **live data** ("current weather in Ahmedabad") are routed to a tool instead
of the search sources, which cannot answer them.

```
cd week7
npm install --legacy-peer-deps
npm run research -- "How do LangGraph agents differ from LangChain agents?"
```

Uses the repo-root `.env` for `OPENROUTER_API_KEY` if `week7/.env` doesn't exist
(see `.env.example`). No other keys are needed - all three sources are free and keyless.

A run takes roughly 2-3 minutes on a free model; progress lines are printed throughout
(with timings), so it is never silent.

## The idea in one paragraph

In week 5 an agent decided its own next step. Here **you draw the flowchart** and the AI only
fills in the content of each box, so the path is predictable ("deterministic"). The boxes
never talk to each other: they read from and write to **one shared notebook** (the *state*),
and LangGraph decides who goes next.

## How it runs

```
topic
  |
  v
ROUTER (AI) ------------ RESEARCH | WEATHER: <city> | MIXED: <city> | <rest> | UNSUPPORTED_LIVE
  |  weather -> WEATHER TOOL (code: Open-Meteo, no key) -> summarizer   (no searching)
  |  mixed   -> WEATHER TOOL -> planner (only the non-weather part), one library for both
  |  unsupported -> honest "needs live data, no tool for it" report -> save
  |  research (and any failure of the router or the tool) -> below
  v
PLANNER (AI) ---------> 3 sub-questions
  |
  v   one parallel branch per sub-question
RETRIEVAL (AI + code) - writes a query per source, searches Wikipedia / arXiv / Hacker News
  |                      at once, then an AI judge drops the off-topic results
  v
COVERAGE CHECK (code) -- nothing relevant for some sub-question? -> retry those (max 3 rounds)
  |                       nothing at all?  -> honest "no sources" report (nothing is guessed)
  v
SUMMARIZER (AI) -------> short notes, each bullet tagged with its source number [n]
  |
  v
ANSWER WRITER (AI) ----> the brief, using only those notes
  |
  v
CITATION CHECK (code) -- a [n] that doesn't exist? -> back to the writer with the reason (max 3 drafts)
  |
  v
SAVE (code) -----------> reports/<topic>-<timestamp>.md
```

| LangGraph concept | Where it is |
|---|---|
| **State** | `src/state.ts` - the shared notebook |
| **Nodes** | `src/graph.ts` - router, weatherTool, unsupported, plan, retrieve, coverage, summarize, write, checkCitations, save |
| **Edges** | `src/graph.ts` - the fixed arrows (`addEdge`) |
| **Conditional routing** | `src/graph.ts` - `afterRoute`, `afterWeather`, `afterCoverage`, `afterSummarize`, `afterCitationCheck` |
| **Parallel execution** | `src/graph.ts` - `fanOutRetrieval` returns one `Send` per sub-question |
| **Reducer** | `src/state.ts` - `appended()` merges what parallel branches write |

## Which parts are AI, which are plain code

| AI (can be wrong) | Plain code (predictable, tested) |
|---|---|
| router, planner, query writer, relevance judge, summarizer, answer writer | the weather tool (every number in the reading is copied from the API, never written by the AI), the three web searches, coverage check, citation check, source numbering, report building, retries |

## What the safeguards do - and don't - guarantee

- **Citations are real numbers.** The citation check rejects `[n]` markers that don't point to
  a retrieved source, and the Sources list is built by code from what was actually fetched -
  the AI never writes a link. Only sources the text cites are listed.
- **A citation does not prove the claim.** The check confirms `[3]` exists; it cannot confirm
  source 3 actually says what the sentence says. The writer is told to use only its notes, but
  treat the brief as a well-sourced starting point and click through the sources for anything
  important. Hacker News results are developers' opinions, not facts.
- **The AI chooses the route, code checks it.** The router answers in one strictly parsed line
  (`parseRouteReply`); a reply that can't be parsed is asked again, and if the router or the
  weather tool still fails the run falls back to plain research and *About this run* says so.
  Measured on the live model: 10/10 on a sample of weather, research, mixed and other-live-data
  topics. The city is resolved by Open-Meteo's geocoder (top match, or the one matching a
  "City, Country" qualifier - there is an Ahmedabad in Pakistan too) and the resolved place is
  printed in the brief, so a wrong match is visible. A weather reading is a snapshot, so the
  brief header carries the time (UTC) and the reading carries the local observation time.
- **Nothing is invented when nothing is found.** If a sub-question has no relevant source, the
  brief's *Limitations* section says so (seen live), and if nothing is found at all the report
  says that instead of asking the AI to write anyway.
- **Failures degrade instead of crashing.** A source being down, a model call failing, or the
  summarizer/writer timing out after retries never throws away the research: the report still
  saves (listing the relevant excerpts unedited, if the writing step failed), and an
  *About this run* section says what happened.

## Things worth knowing (all found by running it live)

- **Free models come and go.** week5's model was removed before week6, and week6's before week7
  (OpenRouter now answers 404 "unavailable for free"). `CLOUD_MODEL` in `src/config.ts` is the
  one line to change; the comment above it records the comparison of the free tool-calling
  models. If a run fails with a model error, check that first. (week5 and week6 have the same
  one-line problem.)
- **Weak models fail in specific ways**, so the code defends against each one: a schema with
  three string fields made this model return `": "` for every query (now a single-list schema
  plus validation), function calling sometimes returned no arguments (the relevance judge now
  answers in plain text that is strictly parsed), and 429/malformed responses are retried.
- **Search engines always return something.** Without the relevance judge, astrophysics papers
  ended up in a brief about software frameworks, and "8 sources found" looked like coverage.
- **arXiv asks for one request per 3 seconds**, so its requests are queued even though the
  sub-questions run in parallel. Hacker News's monthly "Who is hiring?" threads and
  moderator-removed threads are filtered out because they match every keyword and say nothing.

## Tests

```
npm test          # 48 tests, no network and no API key needed
npm run typecheck
```

`graph.test.ts` runs the **real graph** with scripted stand-in agents and sources, so the
routing is proven exactly: parallel fan-out, retry with rewritten queries, the retry cap, the
"nothing found" path, a bad citation sent back for a redraft, a writer that never gets it
right, a source being down, and the writer/summarizer failing. That is possible because
`createResearchGraph` receives its agents and sources as arguments instead of importing them
(only `research.ts` plugs in the real ones). `logic.test.ts` and `sources/sources.test.ts`
cover the pure decision functions and the response parsers.

## Files

```
week7/
  reports/                   generated briefs (gitignored)
  src/
    research.ts              the command you run
    graph.ts                 the flowchart: nodes, edges, routers
    tools/weather.ts         the live-data tool (Open-Meteo geocoding + current conditions)
    state.ts                 the shared notebook
    logic.ts                 pure decisions: numbering, coverage, citation check, report
    types.ts                 shared data shapes (zod)
    agents/                  router.ts, planner.ts, retrieval.ts (queries + relevance judge),
                             summarizer.ts, writer.ts, llm.ts (retries), types.ts
    sources/                 wikipedia.ts, arxiv.ts, hackernews.ts, http.ts, text.ts
    config.ts, models.ts, report.ts, utils.ts
    *.test.ts
```
