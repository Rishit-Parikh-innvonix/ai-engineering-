# Week 3 - Prompt Engineering, Local LLMs, and Vector Search

Exploratory exercises, not a fixed pipeline - the assignment's own framing. Built
with **LangChain.js** (TypeScript), **OpenRouter** for the cloud model, **Ollama**
for the local model and local embeddings, and **Chroma** (via Docker) for vector
search. RAG's generation half is intentionally not implemented here - see
`docs/RAG_PIPELINE.md` for why and what it would take.

## One-time setup

```
cd week3
npm install
```

Copy `.env.example` to `.env` and add your OpenRouter key (a working `.env`
already exists in this folder for this repo's own use - see `week2-js/README.md`
for why that's fine locally but must never be committed).

**Ollama** (local model + local embeddings):
```
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```
The Windows installer starts the Ollama server automatically; verify with
`curl http://127.0.0.1:11434/api/version`. If it's not running, launch the
"Ollama" app once from the Start menu.

**Chroma** (vector store, only needed for Exercise 4):
```
docker run -d --name week3-chroma -p 8010:8000 chromadb/chroma
```
Uses port 8010 (not Chroma's default 8000) so it doesn't collide with
week2-js's Express server. Leave the container running; re-run the same
`docker run` command any time after a reboot (or `docker start week3-chroma`
if the container already exists but stopped).

## Exercise 1 - Prompt Experiment

```
npm run prompt-experiment
```
Same task ("explain recursion to a beginner"), four prompt variants (naive,
persona+audience, format-constrained, few-shot), same model/temperature - only
the wording changes. Verified live run: the naive prompt produced a long
multi-section answer with two code examples; the persona prompt dropped all
code and ended by asking the learner a follow-up question; the format-constrained
prompt produced exactly the 3 sentences asked for and nothing else; the few-shot
prompt closely matched the demonstrated example's structure (analogy, then
mechanism, then a "beginners often get this wrong" sentence) almost paragraph
for paragraph.

## Exercise 2 - Week 2 Pipeline: improve one prompt

```
npm run prompt-improvement
```
Takes a week2-js-style extraction prompt and improves it with one added
sentence telling the model what to do when a field is genuinely missing from
the input, tested on an email with no email address in it. Verified live run:
the small local model (`llama3.2:1b`) **without** the fix fabricated a literal
placeholder-looking value (`<customer_email>`) for the missing field - not
flagged as missing at all, exactly the kind of value that would silently
corrupt a downstream database if nothing caught it. **With** the fix it
correctly reported the field as missing. The larger cloud model handled it
gracefully even without the fix - which is itself the finding: prompt
engineering safety nets matter more, not less, on smaller/local models: you
can't assume behavior verified on a large cloud model also holds on a small one.

## Exercise 3 - Ollama Experiment (local vs cloud)

```
npm run local-vs-cloud
```
Same 3 prompts sent to `llama3.2:1b` (local, via Ollama) and
`minimax/minimax-m2.7:free` (cloud, via OpenRouter), timed. Verified live run:
the local model's *first* call took ~42s (loading the model into memory for
the first time) then dropped to ~0.1-0.4s on later calls once warm; the cloud
model was a steady ~5-18s every time (network + generation, no warm-up effect).
Response quality was comparable on the factual/reasoning prompts; the cloud
model's answers were somewhat more detailed.

## Bonus - Capability Gap Demo (local vs cloud, on hard tasks)

```
npm run capability-gap
```
Exercise 3's prompts (fact lookup, short explanation, haiku) were easy enough
that the 1B local model handled them fine - "local won" there was a *latency*
finding, not a capability one. This demo instead uses four tasks specifically
chosen because model size tends to matter: a multi-step arithmetic problem
(errors compound across steps), the classic "farmer has 17 sheep, all but 9
die" riddle (a known pattern-matching trap), a syllogism trap (does "some"
sloppily become "all"), and a 4-simultaneous-constraint writing task. Each has
an objectively checkable answer, so the demo reports PASS/FAIL, not just text
to eyeball. Verified live run: both models passed the arithmetic and the
constraint-following task, but the local model failed both reasoning traps -
it answered "8" on the sheep riddle (falling for the subtraction pattern
instead of reading "all but 9 die" correctly) and wrongly concluded "all
Bloops are Lazzles" on the syllogism (over-generalizing "some" to "all") -
while the cloud model got both right. This is the actual capability
difference: it shows up on tasks requiring careful reading against a tempting
shortcut or genuine logical rigor, not on tasks that just require following
steps or obeying a format.

## Exercise 4 - Vector Search

```
npm run vector-search
```
Embeds 10 short support-KB sentences with `nomic-embed-text` (via Ollama,
free/local), stores them in Chroma, and runs similarity search for 3 queries
that don't share exact keywords with their best-matching document (e.g. "How
do I get my money back?" -> the refunds document). Verified live run: all 3
queries correctly retrieved the intended document as the #1 result by
embedding distance, despite near-zero literal word overlap - demonstrating
retrieval by meaning rather than keyword matching.

## Exercise 5 - RAG Understanding

See `docs/RAG_PIPELINE.md` - a written explanation of the full RAG pipeline
(chunking -> embedding -> vector store -> retrieval -> augmented prompt ->
generation) with a Mermaid diagram, explicitly connecting it back to what
Exercise 4 already does (the retrieval half) and what's still missing (the
generation half, deferred to a future week per the assignment).

## Capstone - Full RAG pipeline on a real PDF

```
npm run pdf-rag
```
`vectorSearchDemo.ts` only covers retrieval (chunk -> embed -> store -> search).
This capstone adds the missing generation half on top - chunk -> embed -> store
-> retrieve -> **augment the prompt with retrieved context** -> **call the LLM**
-> real answer. Currently configured (see `config.ts`) against
`data/laws-of-cricket-2017.pdf`, the official MCC Laws of Cricket (2017 Code,
3rd edition, 2022) - 79 pages. Chunking uses LangChain's own
`RecursiveCharacterTextSplitter` (from `@langchain/textsplitters` - a small
standalone package, unlike `@langchain/community`, so it installs with no
peer conflict): it tries paragraph breaks first, then line breaks, then word
breaks, only falling back to a mid-word cut as a last resort, and its
`splitDocuments()` automatically carries each page's metadata (our page
number) onto every chunk it produces. We first built our own hand-written
chunker (fixed-size + sentence-boundary snapping + overlap) before switching
to this - see "Chunking: hand-written vs. LangChain's splitter" below for
what actually changed when we swapped it in.

Retrieval and generation are wired together as a real LangChain chain
(`RunnableSequence`) rather than hand-called steps. `ChromaRetriever`
(`src/chromaRetriever.ts`) is a from-scratch LangChain `Retriever` - it
extends `BaseRetriever` from `@langchain/core` (no conflicting package
needed) and wraps our Chroma collection behind the standard
`retriever.invoke(question)` interface every LangChain retriever exposes,
regardless of what's actually behind it. `buildRagChain()` composes
`ChromaRetriever` + a `ChatPromptTemplate` + the chat model +
`StringOutputParser` into one chain using `RunnablePassthrough.assign()`,
which is what lets the retrieved documents survive alongside the final
answer (needed for the "Retrieved chunks" table) even though generation
runs after retrieval in the same chain. We verified this produces identical
retrieval and answers to the hand-called version before switching - this was
purely an architecture change, not a behavior change.

Verified live run: 217,943 characters split into 360 chunks, embedded in
3.5s. Two on-topic questions (how many players per side, deliberately
fielding the ball with a foot) got correctly grounded answers pulled from the
real document text. A fourth, deliberately unrelated question ("What is the
capital of France?") correctly got "the document doesn't cover this" instead
of the model answering "Paris" from its own outside knowledge - and its
retrieved chunks' distances (1.10-1.20) were visibly worse than every
on-topic question's (0.51-0.78), showing distance itself as a signal for "was
anything actually relevant found." An earlier run used a 120-page NIST
security-standard PDF instead (`data/nist-sp-800-171r3.pdf`, still present in
`data/` and still queryable under its own Chroma collection
`week3-pdf-rag-demo`) with the same result pattern - swapping `PDF_PATH`,
`PDF_CHROMA_COLLECTION`, and `PDF_RAG_QUESTIONS` in `config.ts` is all it takes
to point this same pipeline at any PDF.

### Chunking: hand-written vs. LangChain's splitter

Swapping the chunker changed more than code style - it changed what got
retrieved. With our hand-written chunker (395 chunks), the LBW question's top
retrieved chunk was page 49, and the model correctly quoted Law 36.1.1-36.1.5.
After switching to `RecursiveCharacterTextSplitter` (360 chunks), the LBW
law's text apparently now straddles a chunk boundary differently: the top
retrieved chunks became pages 42/44/68/33 - none actually containing the LBW
law - and the model correctly said it didn't have that information rather
than guessing (grounding still held), but the *right answer stopped being
retrievable at all*. Same embedding model, same collection, same question -
different chunk boundaries were the only variable. This is direct, measured
evidence that chunking strategy is not a cosmetic implementation detail: it
changes what a RAG system can and can't answer, independent of retrieval or
generation quality.

## Design decisions worth knowing about

- **Why Chroma over FAISS**: FAISS's Node binding (`faiss-node`) needs a
  native C++ build toolchain; Chroma just needed `docker run` since Docker was
  already installed. Same underlying concept (embed, store, nearest-neighbor
  search) either way.
- **Why the `chromadb` client directly, not `@langchain/community`'s Chroma
  wrapper**: that package pulls in an unrelated, heavy peer-dependency chain
  (browser-automation tooling) just for this one small feature. A thin direct
  client call is simpler and avoids the conflict entirely.
- **Why Ollama for embeddings instead of OpenRouter's**: OpenRouter has no
  free embedding model (cheapest is $0.004/M tokens); Ollama's
  `nomic-embed-text` is free and fully local, and was already being installed
  for Exercise 3 anyway.
- **Why a different cloud model than week2-js used**: `openai/gpt-oss-20b:free`
  was dropped from OpenRouter's free tier since week2-js was built. Verified
  against OpenRouter's live `/models` list on 2026-08-29 and replaced with
  `minimax/minimax-m2.7:free`.

## Project layout

```
week3/
  src/
    config.ts              models, Ollama/Chroma URLs, sample data
    models.ts               ChatOpenAI (OpenRouter) / ChatOllama / OllamaEmbeddings factories
    chroma.ts                 shared ChromaClient + collection-reset helpers
    utils.ts                   shared console table/panel printers (same as week2-js)
    promptExperiment.ts          Exercise 1
    promptImprovement.ts           Exercise 2
    localVsCloudDemo.ts              Exercise 3
    capabilityGapDemo.ts               Bonus - hard-task local vs. cloud comparison
    vectorSearchDemo.ts                  Exercise 4
    pdfLoader.ts                           from-scratch LangChain Document Loader for PDFs
    chromaRetriever.ts                       from-scratch LangChain Retriever wrapping Chroma
    pdfRagDemo.ts                              Capstone - full RAG pipeline as a RunnableSequence chain
  docs/
    RAG_PIPELINE.md              Exercise 5
```
