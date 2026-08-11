# AI Engineering — Week 1 Assignment

One project, three exercises, all built with **Python + LangChain + OpenRouter**, using only
**free** OpenRouter models. No other AI SDK is used anywhere in this project — every request,
regardless of which model answers it, goes through OpenRouter.

```
project/
├── .env.example       template for your API key
├── requirements.txt   the 4 packages this project needs
├── README.md          this file (also Exercise 3)
├── config.py          all settings: models, prompts, parameter values
├── models.py          talks to OpenRouter through LangChain
├── utils.py           terminal tables and small helpers
├── experiments.py     Exercise 2 logic
├── switch_demo.py      tiny Exercise 3 demo
└── main.py             menu you actually run
```

There is no `outputs/` folder. The assignment asks for everything to be shown in the
terminal (no CSV/Excel files), so nothing gets written to disk, and an empty folder would
just be confusing.

---

## 1. Setup

1. Create a free account at openrouter.ai and generate an API key (Settings → Keys). No
   credit card is required to use free models.
2. Copy `.env.example` to `.env` and paste your key in:
   ```
   OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the project:
   ```
   python main.py
   ```
   You'll see a menu to run Exercise 1, Exercise 2, or the Exercise 3 demo.

### A note on Windows terminals

The tables use the `rich` library, which draws box characters and an ellipsis (`…`) for
truncated text. Windows' legacy Command Prompt/PowerShell codepage sometimes renders these
as `?` or `�`. If that happens, run `chcp 65001` once in your terminal before
`python main.py`, or use Windows Terminal, which defaults to UTF-8 and renders everything
correctly. It's purely a display issue, not a bug in the data.

### A note on free-tier limits

OpenRouter's free models are capped at **20 requests/minute**, and **50 requests/day**
unless your account has ever added at least $10 in credit (then it's 1000/day). This
project intentionally adds a short delay between calls (`REQUEST_DELAY_SECONDS` in
`config.py`) to avoid the per-minute cap. If you run Exercise 1 and Exercise 2 back to
back you may still approach the daily cap — that's expected, not a bug. Just wait or run
them on separate days.

### Which models are free right now

Free OpenRouter models change over time — providers add and remove them. This project
checked the live list on **2026-08-08** by calling OpenRouter's own model API
(`GET https://openrouter.ai/api/v1/models`) and filtering for ids ending in `:free`. At
that moment, these were the free models:

```
cohere/north-mini-code:free
google/gemma-4-26b-a4b-it:free
google/gemma-4-31b-it:free
inclusionai/ling-3.0-tiny:free
nvidia/nemotron-3.5-content-safety:free
nvidia/nemotron-3-nano-30b-a3b:free
nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
nvidia/nemotron-3-super-120b-a12b:free
nvidia/nemotron-3-ultra-550b-a55b:free
nvidia/nemotron-nano-12b-v2-vl:free
nvidia/nemotron-nano-9b-v2:free
openai/gpt-oss-20b:free
poolside/laguna-s-2.1:free
poolside/laguna-xs-2.1:free
```

This project defaults to 4 general-purpose ones from different providers
(`COMPARISON_MODELS` in `config.py`). If any of them get pulled by the time you run this,
just open `config.py`, check the current free list yourself at
`https://openrouter.ai/models?max_price=0`, and swap the string. Nothing else in the
project needs to change — that's the whole point of Exercise 3's model-abstraction lesson
below.

---

## 2. Exercise 1 — Compare free OpenRouter models

`main.py` → `run_exercise_1()` sends the same 5 prompts (explain a concept, explain
Transformers, write code, solve a small reasoning riddle, summarize a paragraph) to every
model in `config.COMPARISON_MODELS`, using `models.ask_model()` for each call.

For every call it records: model name, execution time, response length in characters,
token usage (real numbers from OpenRouter when the API returns them, otherwise an
estimated count), and the response itself. All of this is rendered as a `rich` table in
the terminal — no files written.

After the table, the project builds a summary. Instead of hardcoding canned opinions like
"Model X is great at coding" (which the assignment explicitly says not to do — "do not
invent observations"), it takes the *actual* transcripts of every response and time
measurement collected above and sends them to one free model acting as a **judge**, with
instructions to base its comparison strictly on that data. This is a real technique called
**LLM-as-judge**, used throughout the industry to evaluate model outputs at scale. The
judge model writes the comparison covering quality, coding ability, reasoning ability,
speed, clarity, strengths/weaknesses, and which model wins each category — grounded in
what actually happened in your run, not in guesses.

---

## 3. Exercise 2 — Experiment with LLM parameters

`experiments.py` runs the same base prompt through **one** model
(`config.EXPERIMENT_MODEL`) while sweeping one parameter at a time:

- **Temperature**: 0, 0.2, 0.5, 0.8, 1.0 — using a creative-writing prompt, since
  temperature effects show up most clearly there.
- **Top-p**: 0.2, 0.5, 0.8, 1.0 — temperature is held fixed at 0.9 during this sweep so
  the effect of top-p isn't drowned out (top-p barely matters at temperature 0).
- **Max tokens**: 50, 100, 200, 500 — using an explanatory prompt, so you can literally
  see the answer get cut off at low limits.

Each response also gets a computed **unique-word ratio** (unique words ÷ total words) —
a simple, honest, non-LLM metric used as a rough stand-in for "how repetitive vs. varied"
a response is. It's not a perfect creativity score, but it's real data computed directly
from the text, not an opinion.

Then it runs the 6 hallucination prompts from the assignment (a fictional 1850 "Kingdom of
Mars", a fictional programming language, a 2050 World Cup winner, "tomorrow's news",
"today's weather", "the latest global event") — all designed to tempt the model into
inventing an answer, since none of them have a real answer the model could know.

Same as Exercise 1, the final summary is generated by feeding the actual collected data
(every parameter value, every response length, every unique-word ratio, every
hallucination prompt/response pair) to the judge model and asking it to explain, using
only that data: how temperature affected creativity and determinism, how top-p affected
diversity, how max_tokens affected length, concrete hallucination examples from the run,
and lessons learned.

---

## 4. Exercise 3 — Learn LangChain (this section)

### 4.1 What is LangChain?

LangChain is a Python (and JavaScript) framework for building applications powered by
large language models (LLMs). It doesn't provide its own AI models — it provides a
**common set of building blocks** (prompts, model wrappers, output parsers, chains) that
work the same way regardless of which AI provider is actually answering your request.

### 4.2 Why was LangChain created? What problems existed before it?

Before LangChain (and frameworks like it), every AI provider — OpenAI, Anthropic, Cohere,
Google, local models via Ollama — shipped its own SDK, with its own request format, its
own response shape, its own way of handling streaming, its own way of handling errors.

If you built an app directly against, say, the OpenAI Python SDK, and later wanted to try
Anthropic's Claude, or a local open-source model, you had to **rewrite your integration
code**, not just swap a string. Prompt formatting, message roles, function-calling syntax,
response parsing — all of it differed slightly provider to provider. Multiply that by the
5-10 providers a serious AI team evaluates, and you get a lot of duplicated, provider-
specific glue code that has nothing to do with your actual product logic.

### 4.3 Why provider SDKs alone are not enough

A provider's own SDK is *optimized for that provider*. It's very good at talking to that
one API, but it gives you nothing for:
- Swapping models without rewriting integration code.
- Composing multi-step logic (prompt → model → parse output → feed into next prompt) in a
  uniform way.
- Reusing the same prompt-templating and output-parsing logic across different providers.
- A standard interface your whole team can learn once and reuse across every project,
  regardless of which model that project happens to use.

Provider SDKs are necessary (LangChain uses them under the hood!) but insufficient on
their own for building portable, maintainable AI applications.

### 4.4 How LangChain standardizes multiple providers

LangChain defines a common interface — in this project, `ChatOpenAI` — that many
providers can be plugged into, as long as they expose an OpenAI-compatible API (which
OpenRouter does, by design: OpenRouter itself is a **router** that speaks the OpenAI API
format and forwards your request to whichever underlying model you specify). Whether the
actual answer comes from Google's Gemma, OpenAI's gpt-oss, or Nvidia's Nemotron, your
Python code calls the exact same `.invoke()` method and gets back the exact same
`AIMessage` object shape.

### 4.5 Why switching models becomes simple

Look at `models.py` in this project:

```python
def get_chat_model(model_name, temperature=0.7, top_p=1.0, max_tokens=None):
    return ChatOpenAI(
        model=model_name,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
```

`model_name` is just a string. Every place in this project that talks to an LLM goes
through this one function. Changing which model answers a question means changing **one
string**, nowhere near your actual application logic. `switch_demo.py` demonstrates
exactly this — see section 4.11 below.

### 4.6 PromptTemplate and ChatPromptTemplate

These are LangChain's tools for building prompts with variable placeholders instead of
hand-gluing strings with `+` or f-strings everywhere.

`PromptTemplate` is for plain-text prompts:
```python
from langchain_core.prompts import PromptTemplate
template = PromptTemplate.from_template("Explain {topic} to a beginner in {n} sentences.")
prompt_text = template.format(topic="transformers", n=3)
```

`ChatPromptTemplate` is for chat-style prompts, where a conversation is made of messages
with roles (`system`, `human`, `ai`):
```python
from langchain_core.prompts import ChatPromptTemplate
template = ChatPromptTemplate.from_messages([
    ("system", "You are a patient teacher for complete beginners."),
    ("human", "Explain {topic} in {n} sentences."),
])
messages = template.format_messages(topic="transformers", n=3)
```
This project keeps prompts as plain strings in `config.py` for simplicity (per the
"beginner friendly, don't overengineer" goal of this assignment), but in any real
application you'd reach for these the moment you have reusable prompts with variables.

### 4.7 Output Parsers

An LLM always returns text. An **output parser** turns that raw text into a structured
Python object your code can use directly — a list, a dictionary, a specific class —
instead of manually string-splitting the response yourself. For example
`StrOutputParser` just extracts the plain string content, while
`PydanticOutputParser`/`with_structured_output` can force and validate JSON matching a
schema you define. This project doesn't need one, because we display raw text in tables,
but it's the standard next step once you want an LLM's answer to drive actual program
logic (e.g. "did the model classify this as positive or negative?").

### 4.8 RunnableSequence and LCEL

Every LangChain component — a prompt template, a model, an output parser — implements the
same `Runnable` interface (`.invoke()`, `.batch()`, `.stream()`). **LCEL** (LangChain
Expression Language) lets you chain them together with the pipe operator `|`, producing a
`RunnableSequence`:

```python
chain = prompt_template | chat_model | output_parser
result = chain.invoke({"topic": "transformers", "n": 3})
```

Here, `prompt_template` formats your input into a prompt, its output is piped straight
into `chat_model`, and `chat_model`'s output is piped into `output_parser`. Each step's
output becomes the next step's input automatically. This is LangChain's core idea:
compose small, swappable pieces into a pipeline instead of writing imperative glue code
by hand. This project calls `chat.invoke(prompt)` directly since a single call is all
each exercise needs — but if you wanted to add prompt templating and output parsing here,
you'd wrap the same `chat` object from `get_chat_model()` into a chain exactly like above,
with zero changes to how the model itself is configured.

### 4.9 Model abstraction

This is the umbrella concept behind 4.4–4.5: LangChain gives every chat model — regardless
of provider — the same shape (a `Runnable` that takes messages/strings and returns an
`AIMessage`). Your application code is written once, against that shape, and the specific
model becomes a configuration detail rather than something baked into your logic.

### 4.10 Why LangChain became popular

- It arrived right as the number of usable LLM providers exploded, so "don't want to
  rewrite my app every time I try a new model" became a very common, very real pain point.
- LCEL made multi-step LLM logic (prompt → model → parse → next prompt) readable and
  composable instead of a pile of nested function calls.
- A huge ecosystem of integrations (vector stores, document loaders, tools, agents) grew
  around the same standard interface, so learning LangChain once unlocks a lot of
  pre-built pieces.

### 4.11 Advantages

- Swap models/providers by changing configuration, not code (see `switch_demo.py`).
- One consistent interface to learn, reused across prompt templates, models, parsers,
  chains, and agents.
- Large ecosystem: memory, retrieval, tools, agents, tracing (LangSmith) all build on the
  same `Runnable` interface.
- Easier to compose multi-step pipelines with LCEL than to hand-write the equivalent glue
  code against a raw provider SDK.

### 4.12 Disadvantages

- An extra abstraction layer: another API to learn on top of the underlying provider API.
- Error messages sometimes point into LangChain's internals rather than the root cause,
  which can be confusing while debugging.
- Fast-moving project — APIs have changed across versions (this project pins to the
  current stable API, using `langchain-openai`'s `ChatOpenAI` directly, not older
  deprecated patterns).
- For a single call to a single fixed provider, LangChain is genuinely more machinery than
  you need.

### 4.13 When LangChain is the correct choice

- You expect to compare or switch between multiple models/providers (exactly this
  assignment's Exercise 1).
- You're building multi-step LLM pipelines: prompt → model → parse → feed into another
  prompt.
- You want structured, validated output from an LLM (via output parsers).
- You want to add retrieval, tools, or agent behavior later without re-architecting.

### 4.14 When LangChain is unnecessary

- A one-off script that calls one provider, one model, one time — a plain SDK call is
  simpler and has one less layer to debug.
- Extremely latency- or dependency-sensitive production paths, where every extra package
  and abstraction has a real cost you'd rather avoid.

### 4.15 Using OpenRouter directly vs. using LangChain with OpenRouter

**Direct** (using the `openai` Python SDK pointed at OpenRouter, since OpenRouter mimics
the OpenAI API):
```python
from openai import OpenAI
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
response = client.chat.completions.create(
    model="openai/gpt-oss-20b:free",
    messages=[{"role": "user", "content": "Explain transformers simply."}],
)
print(response.choices[0].message.content)
```
This works fine. But it's the *OpenAI SDK's* request/response shape. If you later wanted
to add Anthropic directly (not through OpenRouter), you'd learn and integrate a second,
differently-shaped SDK.

**With LangChain** (what this project does):
```python
from langchain_openai import ChatOpenAI
chat = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)
response = chat.invoke("Explain transformers simply.")
print(response.content)
```
Nearly identical for this one call — but now `chat` is a `Runnable`. It can be piped into
prompt templates, output parsers, and chains with `|`, using the exact same interface
every other LangChain-wrapped model (from any provider) also uses. The value isn't visible
in a single call; it shows up the moment your app has more than one moving part or you
want to compare/swap models, which is exactly Exercise 1 and Exercise 3's demo.

### 4.16 Why AI Engineers commonly use LangChain

Real applications rarely make exactly one hardcoded LLM call to exactly one provider
forever. They compare models during development, switch providers to cut cost or latency,
add retrieval or tools, and need prompts to be reusable rather than duplicated
string-templates scattered across files. LangChain's standard interface means that
engineering effort spent on prompt design, output parsing, and pipeline logic isn't
thrown away every time the underlying model changes — which is precisely why this whole
assignment could be built as an OpenRouter-only project without ever touching a
provider-specific SDK directly.

### 4.17 The tiny demonstration: one config value, two different models

Run it from the menu (`python main.py`, choose option 3) or directly:
```
python switch_demo.py
```

Look at `switch_demo.py`:
```python
DEMO_MODEL = config.COMPARISON_MODELS[0]

def ask_demo_question(model_name):
    chat = models.get_chat_model(model_name, temperature=0.5)
    response = chat.invoke("In one sentence, what model are you and who built you?")
    return response.content
```
`run_demo()` calls `ask_demo_question()` twice — once with `config.COMPARISON_MODELS[0]`,
once with `config.COMPARISON_MODELS[1]`. That's it: two different OpenRouter model
strings, going through the exact same `get_chat_model()` function that every other part
of this project already uses. `ask_demo_question()` itself never changes, never checks
which provider it's talking to, and never branches on model name. That's model
abstraction, working end to end.

---

## 5. How LangChain, OpenRouter, and Python fit together

1. Your code calls `chat.invoke(prompt)` on a `ChatOpenAI` instance.
2. `langchain-openai` builds a standard OpenAI-format HTTPS request (JSON body with
   `model`, `messages`, `temperature`, etc.) and sends it to `base_url` —
   `https://openrouter.ai/api/v1/chat/completions` — using your `OPENROUTER_API_KEY` as
   the bearer token.
3. **OpenRouter** receives it, looks at the `model` field (e.g.
   `"openai/gpt-oss-20b:free"`), and **routes** the request to whichever backend actually
   hosts that model. This routing is OpenRouter's entire reason to exist: one API key, one
   endpoint, dozens of providers behind it.
4. That backend model generates a response. OpenRouter wraps it back into the same
   OpenAI-compatible JSON shape (`choices[0].message.content`, plus a `usage` object with
   token counts when available) and returns it over the same HTTPS connection.
5. `langchain-openai` parses that JSON into an `AIMessage` object — `.content` holds the
   text, `.usage_metadata`/`.response_metadata` holds token counts — which is what your
   Python code receives from `.invoke()`.

Because step 2–4 always look the same regardless of which model you pick, your code in
step 1 and step 5 never has to know or care which provider actually answered.

---

## 6. File-by-file summary

- **`config.py`** — every editable setting lives here: which models to compare, which
  model runs the parameter experiments, which prompts to use, which parameter values to
  sweep. Change behavior by editing this file, not the logic files.
- **`models.py`** — the only file that builds a `ChatOpenAI` instance or calls
  `.invoke()`. `ask_model()` wraps a call with timing and error handling so a single
  failed request never crashes the whole run. `ask_judge()` reuses the same function to
  power the LLM-as-judge summaries.
- **`utils.py`** — `rich` table/panel printing, plus the `unique_word_ratio()` metric and
  a `time.sleep()` helper to stay under OpenRouter's rate limit.
- **`experiments.py`** — Exercise 2: four functions, one per parameter sweep, plus the
  hallucination experiment and the transcript-building/summary logic.
- **`switch_demo.py`** — Exercise 3's tiny demo.
- **`main.py`** — the menu you run. Orchestrates Exercise 1 and Exercise 2, and calls the
  Exercise 3 demo.
