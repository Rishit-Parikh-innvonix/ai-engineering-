# Week 5 - Agents

Learning to connect an LLM to external systems through tools, and when an
agent (vs. hand-written tool-calling) is actually appropriate. Built with
**LangChain.js** (TypeScript) and **OpenRouter** for the cloud model, same
setup pattern as `week3-Week4`.

## One-time setup

```
cd week5
npm install --legacy-peer-deps
```

`--legacy-peer-deps` is needed because `@langchain/community` lists
`@browserbasehq/stagehand` (used only by its unrelated `WebBrowser` tool,
which this project never imports) as a peer wanting an older `dotenv` than
this project uses. That's a peer-resolution conflict only, not a real
incompatibility for the tools actually used here.

Uses the repo-root `.env` for `OPENROUTER_API_KEY` if `week5/.env` doesn't
exist (see `.env.example`).

## Run

Each exercise is its own file and its own script, same one-file-per-exercise
convention as `week3-Week4`:

```
npm run exercise1     # pre-built LangChain tool
npm run exercise2     # custom tool
npm run exercise3     # ReAct agent
npm run agents-demo   # all 3, back to back (assignment note explicitly allows combining)
```

1. **Exercise 1 - a real pre-built LangChain tool** (`src/exercise1PrebuiltTool.ts`):
   `WikipediaQueryRun` from `@langchain/community`, bound directly to the
   chat model with no agent involved. `src/boundToolRunner.ts` holds the
   loop this (and exercise 2) actually run: send the question, check the
   model's response for `tool_calls`, run the tool, send the result back,
   repeat until it answers in plain text. The first call forces
   `tool_choice: "required"` - otherwise a free model can just skip a bound
   tool and answer from its own (unverified, occasionally wrong-language)
   memory instead, silently defeating the point of giving it a tool at all.
2. **Exercise 2 - a custom tool** (`src/exercise2CustomTool.ts`):
   `get_current_weather` (defined in `src/tools.ts`), built from scratch
   with `tool()` and a Zod schema, hitting a real external API
   ([Open-Meteo](https://open-meteo.com/), free, no API key) to geocode a
   city name and fetch its live temperature/wind. Runs through the same
   `boundToolRunner.ts` loop as exercise 1.
3. **Exercise 3 - a real ReAct agent** (`src/exercise3ReactAgent.ts`):
   `createAgent` from the `langchain` package, given all 3 tools (Wikipedia,
   the weather tool, and a hand-built calculator) at once. This is the
   actual difference an agent framework buys you: exercises 1-2 hand-write
   the "check for tool_calls, run them, send results back, repeat" loop;
   `createAgent` builds and runs that same reason -> act -> observe loop
   internally - we only supply the model and the tool list, and it decides
   per-question which of the 3 tools it actually needs.

Verified live (both standalone per-exercise and via the combined runner): a
weather+math question correctly chained the weather tool then the
calculator, a factual question called only Wikipedia, and a plain
arithmetic question called only the calculator - without being told which
tool to use for which question.

### Avoiding deprecated APIs

- **`createReactAgent`** (from `@langchain/langgraph/prebuilt`) carries a
  real `@deprecated` tag pointing at `createAgent` from the `langchain`
  package - checked directly by installing `langchain` and reading its type
  declarations: `createAgent`'s minimal shape (`{ model, tools }` instead of
  `{ llm, tools }`, same `agent.invoke({ messages })` /
  `result.messages` shape) turned out to be a near drop-in, so exercise 3
  uses it instead of the deprecated function.
- **`Calculator`** (from `@langchain/community/tools/calculator`) is just a
  thin wrapper around the `math-expression-evaluator` npm package (confirmed
  by reading its source) - `tools.ts` calls that library directly through a
  hand-built `tool()`, the same way `get_current_weather` is built, instead
  of depending on `@langchain/community` for something this small.
- **`WikipediaQueryRun`** is kept from `@langchain/community` deliberately.
  `@langchain/community` is deprecated as a whole package (confirmed via
  `npm install`'s warning and
  https://github.com/langchain-ai/langchainjs-community/issues/61), but
  there is currently no dedicated, non-deprecated LangChain.js package for a
  Wikipedia tool, and the `WikipediaQueryRun` class itself carries no
  individual `@deprecated` tag - only the umbrella package does. The
  alternatives (a paid/free-tier search API like Tavily requiring a new API
  key, or hand-writing a REST call ourselves) either add setup friction or
  stop this exercise from demonstrating an actual pre-built tool, so this
  one dependency on the deprecated package is a deliberate, documented
  exception rather than an oversight.

## Project layout

```
week5/
  src/
    config.ts                 OpenRouter key/URL, cloud model name
    models.ts                  ChatOpenAI (OpenRouter) factory
    utils.ts                    console printers + runIfMain() entry-point helper
    wikipediaFetch.ts            scoped User-Agent fix for Wikipedia's API (see below)
    tools.ts                      the 3 tool instances - single source of truth
    boundToolRunner.ts              shared bind-tool-and-loop logic (exercises 1 & 2)
    exercise1PrebuiltTool.ts         Exercise 1
    exercise2CustomTool.ts            Exercise 2
    exercise3ReactAgent.ts             Exercise 3
    runAll.ts                           runs all 3, isolated per exercise
```

Each exercise file exports its logic as a plain `runExerciseN()` function and
only executes it when run directly (`runIfMain`, in `utils.ts` - the
standard ESM "is this the entry module" check). `runAll.ts` imports and
sequences all three functions directly instead of duplicating any logic, and
wraps each in `runIsolated()` so one exercise failing (a real network blip,
an API outage) doesn't take the other two down with it.

### Why `wikipediaFetch.ts` exists

Wikipedia's API enforces a User-Agent policy
(https://meta.wikimedia.org/wiki/User-Agent_policy) and returns intermittent
`429`s to requests that don't identify themselves - confirmed live: a bare
`fetch()` to `en.wikipedia.org` got `429`, the identical request with a
descriptive `User-Agent` got `200`. `WikipediaQueryRun`'s constructor has no
option for custom headers or a custom fetch, so there's no way to fix this
through its public API. `ensureWikipediaUserAgent()` installs the header at
the transport level instead - but scoped to requests whose hostname is
actually `en.wikipedia.org` (verified: an unrelated host still gets Node's
own default `User-Agent: node`, completely untouched), so every other
outbound call in the app (OpenRouter, Open-Meteo) behaves exactly as if this
module didn't exist.
