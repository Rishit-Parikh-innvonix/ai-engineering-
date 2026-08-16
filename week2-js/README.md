# Week 2 (JS) - Working with LLM APIs

A TypeScript/Node.js replica of [`week2/`](../week2), built with **Express** and
**LangChain.js** instead of FastAPI and Python LangChain. Same three exercises,
same OpenRouter models, same prompts, same schema - different runtime. The goal
is to see the exact same concepts (SSE streaming, structured extraction with
reject/retry, sync vs async throughput) expressed in JavaScript's async model
instead of Python's, so you can compare the two side by side.

The Python version (`week2/`) is untouched - this is a separate, independent
project living in its own folder with its own `package.json` and `.env`.

## Setup

```
cd week2-js
npm install
```

Copy `.env.example` to `.env` and put your OpenRouter key in it (a `.env`
already exists in this folder for this repo's own testing - see
`week2/README.md` for why the key is safe to have locally but must never be
committed).

## Running Exercise 1 (SSE streaming)

Terminal 1 - start the server:
```
npm run dev
```
Wait for `Uvicorn's Express equivalent running on http://127.0.0.1:8000`.

Terminal 2 - run the client:
```
npm run stream-client -- "What is a race condition?"
```

Or just open `http://127.0.0.1:8000/ask/stream?question=hello` in a browser -
most browsers render SSE `data:` lines directly.

Verified live output (npm run stream-client, question "Say hello in five words"):
```
event: start
data: {"request_id":"cc49d597","model":"openai/gpt-oss-20b:free"}

data: {"token":"Hey"}
data: {"token":" there"}
...
event: done
data: {"elapsed_seconds":8.1,"chunks":8,"characters":34}
```

## Running Exercise 2 (structured extraction)

With the server running:
```
curl -X POST http://127.0.0.1:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Subject: cant login\n\nHi, this is Alex Kim (alex.kim@example.com). Getting invalid password error since this morning, need this fixed urgently for a demo."}'
```

Or run the standalone demo (no server needed) against the 3 sample emails in `src/config.ts`:
```
npm run extraction-demo
```

Verified live run - all 3 sample emails extracted correctly on the first attempt:
- Priya Shah / priya.shah@example.com / **urgent** / **technical** - login blocked before a client demo
- Daniel Cho / daniel.cho99@example.com / **low** / **billing** - duplicate charge, not urgent
- Marta / marta.k@example.com / **low** / **feature_request** - dark mode suggestion

## Running Exercise 3 (sync vs async benchmark)

```
npm run benchmark
```

Verified live run (6 prompts, `nvidia/nemotron-3-nano-30b-a3b:free`, concurrency 3):

| Mode  | Total time (s) | Avg latency (s) | Throughput (req/s) | Peak memory (MB) |
|-------|-----------------|------------------|----------------------|--------------------|
| sync  | 15.71           | 2.62             | 0.382                | 82.1               |
| async | 2.11            | 0.93             | 2.844                | 84.7               |

Async finished **7.45x faster** in wall-clock time. Peak memory stayed flat
between the two modes, same as the Python version - the win is entirely from
overlapping network wait time, not from doing less work or using less memory.

## Bonus: tool/function calling demo

```
npm run tool-calling-demo
```
Shows the model requesting `lookup_priority_policy`, the code executing it
locally, and the result being fed back for a final answer - same 4-step loop
as `week2/tool_calling_demo.py`.

## Project layout

```
week2-js/
  src/
    config.ts          same models/prompts/sample data as week2/config.py
    schemas.ts          zod SupportTicket schema (replaces Pydantic)
    prompts.ts           ChatPromptTemplate definitions (LangChain.js)
    models.ts            ChatOpenAI factory functions + retry + token helpers
    extraction.ts         Exercise 2: JSON-mode + zod validate + retry loop
    app.ts                Express server: Exercise 1 (SSE) + Exercise 2 (POST /extract)
    streamClient.ts        Exercise 1 test client (fetch + ReadableStream)
    extractionDemo.ts      Exercise 2 standalone demo (no server needed)
    syncVsAsync.ts          Exercise 3: sequential vs Promise.all benchmark
    toolCallingDemo.ts      bonus: bind_tools-equivalent function calling demo
```

## What's different from the Python version, and why

See `LEARNING_NOTES.txt` for the deep dive. Short version:

- **FastAPI's `StreamingResponse` + async generator** becomes **Express +
  `res.write()` in a loop**. Node doesn't have Python's `yield`-based generator
  streaming builtin to its web framework, so the JS version writes to the
  response stream directly and calls `res.end()` when done.
- **`request.is_disconnected()`** becomes listening for the **`'close'` event**
  on the Express `res` object - same idea (detect the client walking away
  mid-stream), different API.
- **Pydantic + `PydanticOutputParser`** becomes **zod + `.parse()`** - same
  two-layer validation idea (JSON-mode for syntax, schema validation for
  shape/enums), same reject-and-retry loop.
- **`tenacity`'s `@retry` decorator** becomes a **hand-written retry loop**
  (`models.ts`'s `ainvokeWithRetry`) - JS has no widely-used decorator-based
  retry in the way Python does, so the backoff logic is written out explicitly.
- **`asyncio.Semaphore` + `asyncio.gather`** becomes a **hand-written
  `Semaphore` class + `Promise.all`** - conceptually identical (bound
  concurrency, wait for all results), same reasoning: Node's single-threaded
  event loop overlaps I/O-bound waits exactly like Python's asyncio does.
