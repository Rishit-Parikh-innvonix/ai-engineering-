# How to run everything (week3)

All commands run from the `week3/` folder.

## One-time setup

```
npm install
ollama pull llama3.2:1b
ollama pull nomic-embed-text
docker run -d --name week3-chroma -p 8010:8000 chromadb/chroma
```

Make sure `OPENROUTER_API_KEY` is set in `.env`, and that Ollama's server is
running (`curl http://127.0.0.1:11434/api/version` should return a version).

## Exercise 1 - Prompt Experiment

```
npm run prompt-experiment
```

## Exercise 2 - Week 2 Pipeline improvement

```
npm run prompt-improvement
```

## Exercise 3 - Ollama Experiment (local vs cloud)

```
npm run local-vs-cloud
```

## Bonus - Capability Gap Demo

```
npm run capability-gap
```

## Exercise 4 - Vector Search

Requires the Chroma container running (see setup above):
```
npm run vector-search
```

## Capstone - Full RAG pipeline (real PDF)

Requires the Chroma container running (see setup above):
```
npm run pdf-rag
```

## Exercise 5 - RAG Understanding

No command - it's a document: `docs/RAG_PIPELINE.md`.

## All commands at a glance

| What | Command |
|---|---|
| Install deps | `npm install` |
| Pull local models | `ollama pull llama3.2:1b` / `ollama pull nomic-embed-text` |
| Start Chroma | `docker run -d --name week3-chroma -p 8010:8000 chromadb/chroma` |
| Restart Chroma (already created) | `docker start week3-chroma` |
| Ex. 1 prompt experiment | `npm run prompt-experiment` |
| Ex. 2 prompt improvement | `npm run prompt-improvement` |
| Ex. 3 local vs cloud | `npm run local-vs-cloud` |
| Bonus capability gap | `npm run capability-gap` |
| Ex. 4 vector search | `npm run vector-search` |
| Capstone full RAG (real PDF) | `npm run pdf-rag` |
| Type-check everything | `npm run typecheck` |
