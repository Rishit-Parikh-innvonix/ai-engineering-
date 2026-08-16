# AI Engineering — Week 2 Assignment

Three exercises, all built with **Python + FastAPI + LangChain + OpenRouter**, using the
same free OpenRouter models as week1. Week 1 was a CLI tool; week2 is a small backend
service — the topic is "Working with LLM APIs," so the deliverable is actual API endpoints,
not a menu.

```
week2/
├── .env.example        template for your API key
├── requirements.txt     the packages this project needs
├── README.md            this file
├── config.py            models, prompts, retry/concurrency settings
├── prompts.py            ChatPromptTemplate definitions (system + user messages)
├── schemas.py            Pydantic schema for the extracted support ticket
├── models.py             builds ChatOpenAI instances; retry wrapper; token/cost helpers
├── extraction.py          Exercise 2 logic: JSON-mode call + schema validation + retry
├── app.py                 the FastAPI app: Exercise 1 and Exercise 2 endpoints
├── sync_vs_async.py       Exercise 3: sync vs async benchmark script
├── stream_client.py       manual SSE test client for Exercise 1
├── extraction_demo.py     runs Exercise 2 against sample emails without the server
├── tool_calling_demo.py   bonus: function/tool calling walkthrough
├── LEARNING_NOTES.txt     concept deep-dive
└── MENTOR_PREP.txt        anticipated Q&A
```

---

## 1. Setup

1. Copy `.env.example` to `.env` in this folder and paste your OpenRouter key in. (If you
   don't create one, `config.py` falls back to a `.env` at the repo root, which is where
   this repo currently keeps it.)
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Start the API:
   ```
   uvicorn app:app --reload
   ```
   Visit `http://127.0.0.1:8000/docs` for interactive Swagger docs, or use the scripts
   below.

Same free-tier caveats as week1 apply: OpenRouter's free models are capped at 20
requests/minute and 50/day. `config.py` keeps prompt counts and concurrency modest for
that reason.

---

## 2. Exercise 1 — FastAPI endpoint that streams an LLM answer over SSE

**Endpoint:** `GET /ask/stream?question=...` (in `app.py`)

### What it does

1. Builds a `ChatOpenAI` with `streaming=True` (`models.get_streaming_chat_model`) and a
   two-message prompt — a system message and the user's question — via `ChatPromptTemplate`
   (`prompts.stream_prompt_template`).
2. Iterates `chat.astream(messages)`, an async generator that yields one `AIMessageChunk`
   per token (or small group of tokens) as OpenRouter streams the underlying model's
   response.
3. Wraps each chunk as one **Server-Sent Event**: `data: {"token": "..."}\n\n`. SSE is just
   a long-lived HTTP response with `Content-Type: text/event-stream` where each event is a
   `data:`-prefixed line followed by a blank line — no special protocol, no separate
   WebSocket connection, works over plain HTTP and through most proxies.
4. Sends a `event: start` event before the first token (request id + model name) and an
   `event: done` event after the last one (elapsed seconds, chunk count, character count).

### Handling client disconnects

Inside the `async for chunk in chat.astream(...)` loop, every iteration calls
`await request.is_disconnected()` (a Starlette/FastAPI built-in) *before* yielding the next
event. If the client has gone away, the loop breaks immediately instead of continuing to
pull tokens from OpenRouter that nobody will read — this matters on a free tier where every
wasted token still counts against the daily cap.

### Logging request latency

`app.py`'s `logger` records, per request (tagged with a short `request_id`): when the
stream started, and — on a clean finish — total elapsed seconds, chunk count, and character
count; on a disconnect, elapsed time and how many chunks were sent before the client left;
on an error, elapsed time and the exception. A real run against `openai/gpt-oss-20b:free`
looked like this:

```
2026-08-15 15:13:33 INFO week2.app: [45025e6c] new stream request: 'What is a race condition?...'
2026-08-15 15:13:39 INFO week2.app: [45025e6c] request latency: 6.68s - 44 chunks, 245 chars, ~73 tokens (estimated)
```

### Trying it

```
uvicorn app:app --reload
python stream_client.py "What is a race condition?"
```
`stream_client.py` is a small `httpx`-based SSE parser that prints tokens as they arrive and
reports client-side wall time, so you can see the difference between "first token" latency
and "full response" latency — the whole reason to stream in the first place (see section 6).
You can also test with `curl -N "http://127.0.0.1:8000/ask/stream?question=hi"`.

---

## 3. Exercise 2 — Structured extraction from customer support emails

**Endpoint:** `POST /extract` with body `{"email_text": "..."}` (in `app.py` /
`extraction.py`)

### The schema (`schemas.py`)

```python
class SupportTicket(BaseModel):
    customer_name: str
    customer_email: EmailStr
    priority: Literal["low", "medium", "high", "urgent"]
    issue_type: Literal["billing", "technical", "account", "feature_request", "bug", "other"]
    summary: str
```
`EmailStr` and the `Literal` enums aren't decoration — they're what makes "malformed"
detectable at all. If the model emits `"priority": "asap"` (not one of the four allowed
values) or a garbled email address, Pydantic raises a `ValidationError` instead of letting
bad data through.

### Two layers of validation, stacked

1. **JSON mode** (`models.get_json_mode_chat_model`) — `response_format={"type":
   "json_object"}` is passed straight through to the OpenAI-compatible request body. This
   asks the backend to guarantee the reply is *syntactically* valid JSON (not "here's your
   JSON:" followed by prose, not wrapped in ` ```json ` fences).
2. **`PydanticOutputParser`** (`extraction.parser`) — its `get_format_instructions()` is
   injected into the prompt (via `prompts.extraction_prompt_template`) so the model sees
   the exact field names and types expected, and `parser.parse(response.content)` both
   parses the JSON and validates it *semantically* against `SupportTicket`.

JSON mode catches syntax problems; Pydantic catches schema problems. Both must pass.

### Reject and retry (`extraction.extract_support_ticket`)

```python
for attempt in range(1, max_retries + 1):
    try:
        response = await models.ainvoke_with_retry(chat, messages)
        ticket = parser.parse(response.content)
        return {"ticket": ticket.model_dump(), "attempts": attempt, "error": None}
    except (OutputParserException, ValidationError) as error:
        ...  # log and try again
```
`config.EXTRACTION_MAX_RETRIES` (default 3) bounds the loop. If every attempt fails, the
endpoint returns **HTTP 422** with the last error message — the malformed response is
rejected, never silently passed to the caller. `models.ainvoke_with_retry` is a *separate*,
inner retry (via `tenacity`, exponential backoff) for transport failures — a dropped
connection or a 429 from OpenRouter — so a single network blip doesn't cost a full
schema-validation attempt. Two different failure modes, two different retry strategies.

### Trying it

```
python extraction_demo.py         # runs the 3 sample emails in config.py, no server needed
```
or against the running server:
```
curl -X POST http://127.0.0.1:8000/extract -H "Content-Type: application/json" \
     -d "{\"email_text\": \"Hi, this is Priya (priya@example.com), I can't log in and have a demo in an hour!\"}"
```
Actual output from a real run (`openai/gpt-oss-20b:free`, attempt 1/3 succeeded):
```json
{"ticket": {"customer_name": "Priya Shah", "customer_email": "priya.shah@example.com",
 "priority": "urgent", "issue_type": "technical",
 "summary": "Priya Shah is unable to log in due to invalid credentials and needs assistance
 urgently before a client demo."}, "attempts": 1, "elapsed_seconds": 21.86}
```

---

## 4. Exercise 3 — Sync vs async throughput (`sync_vs_async.py`)

Same 6 short prompts, same model (`config.BENCHMARK_MODEL`), run two ways:

- **Sync**: a plain `for` loop calling `chat.invoke(prompt)` — each call blocks until it
  returns before the next one starts.
- **Async**: `asyncio.gather` over `chat.ainvoke(prompt)` for all 6 prompts at once, capped
  at `config.ASYNC_CONCURRENCY` (default 3) concurrent requests via an `asyncio.Semaphore`,
  so the free tier's per-minute cap isn't hit.

For each mode it measures: total wall-clock time, average per-call latency, throughput
(successful calls ÷ total seconds), success/error counts, and peak process memory (via
`psutil`).

### Actual measured result

```
Mode    Total time (s)   Avg latency (s)   Throughput (req/s)   Peak memory (MB)
sync    25.47             4.12               0.236                 119.4
async   7.07              2.49               0.848                 120.6
```
Async finished the same 6 prompts **3.60x faster in wall-clock time**. Peak memory was
essentially identical between the two modes — the win is entirely in time, not resources.
That's the expected result and it's explained by *why* the speedup happens, not just that it
happens: each LLM call is I/O-bound (the process spends almost all its time waiting on
OpenRouter's network response, doing near-zero CPU work). Sync wastes that wait — nothing
else runs during it. Async's event loop uses that same wait to start the next request, so
with concurrency 3 you get roughly 3x the throughput for free, limited only by how many
requests you allow in flight (raising concurrency further would help more, up to the
point OpenRouter's rate limit pushes back).

### Trying it
```
python sync_vs_async.py
```

---

## 5. Bonus: function/tool calling (`tool_calling_demo.py`)

Not one of the 3 graded exercises, but a topic on the syllabus and a natural complement to
Exercise 2 — the same "give the model a schema" idea, applied to actions instead of output.
`chat.bind_tools([lookup_priority_policy])` gives the model a Python function's signature
and docstring as a callable tool; when the model decides it needs the tool, it emits a
structured tool-call request instead of prose, our code executes the actual function, and
the result is fed back for a final answer. A real run:

```
Step 1 - sending the question with a tool available...
Step 2 - model requested a tool call: lookup_priority_policy({'issue_type': 'technical'})
Step 3 - tool executed locally, returned: "Technical issues blocking work are 'urgent'..."
Step 4 - sending the tool result back to the model for a final answer...
Final answer: **Priority: Urgent** ...
```
Run: `python tool_calling_demo.py`

---

## 6. Why structured outputs and streaming are essential for production

**Streaming** is about *perceived* latency, not total latency — actually the opposite,
since accumulating and re-sending each partial chunk adds a little overhead. A 7-second
answer feels broken if the user stares at a blank screen for 7 seconds and then the whole
answer appears at once; the same 7-second answer feels instant if the first token appears
in 300ms and text keeps flowing. Every major chat product (ChatGPT, Claude.ai, Copilot)
streams for exactly this reason. It also lets the client cut a request short — closing the
connection early on a bad answer saves the rest of the generation cost, which is exactly
what this project's disconnect handling exploits.

**Structured outputs** are about making an LLM's answer usable by *code*, not just readable
by a human. `/ask/stream`'s output is prose meant for a person; `/extract`'s output is data
meant to be inserted into a ticketing system, routed by `priority`, and queried by
`issue_type` — none of that works if the "data" is a paragraph you'd have to regex apart.
Production systems chain LLM calls into pipelines (extract → route → summarize →
notify); each step needs the previous step's output in a predictable shape, or the whole
chain is one bad response away from breaking silently. Schema validation with reject/retry
is what turns "the model probably got the format right" into "the caller can trust the
shape of what it received, or gets a clear 422 explaining why not."

---

## 7. File-by-file summary

- **`config.py`** — every editable setting: which models, retry counts, benchmark prompts,
  concurrency limit, sample emails. Change behavior here, not in the logic files.
- **`prompts.py`** — `ChatPromptTemplate`s for both exercises; system message sets behavior
  once, human message carries the per-request variable (`{question}` / `{email_text}`).
- **`schemas.py`** — the `SupportTicket` Pydantic model Exercise 2 validates against.
- **`models.py`** — the only file that builds a `ChatOpenAI` instance. Three flavors
  (plain, streaming, JSON-mode), a `tenacity`-based transport retry wrapper, and token/cost
  helpers reused from week1's `models.py`.
- **`extraction.py`** — Exercise 2's reject-and-retry loop.
- **`app.py`** — the FastAPI app: `/ask/stream` (Exercise 1) and `/extract` (Exercise 2),
  plus `/health` and a `/` index.
- **`sync_vs_async.py`** — Exercise 3's benchmark and its `rich` comparison table.
- **`stream_client.py` / `extraction_demo.py` / `tool_calling_demo.py`** — standalone
  scripts for trying each piece without hand-writing `curl` commands.
