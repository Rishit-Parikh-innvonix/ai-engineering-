# How to run everything (week2-js)

All commands run from the `week2-js/` folder. Run `npm install` once first if
you haven't already.

## Exercise 1 - SSE streaming

Terminal 1 (start the server, leave it running):
```
npm run dev
```

Terminal 2 (run the client against it):
```
npm run stream-client -- "What is a race condition?"
```

Or open in a browser while the server is running:
```
http://127.0.0.1:8000/ask/stream?question=hello
```

## Exercise 2 - structured extraction

No server needed - runs the 3 sample emails from `src/config.ts` straight through:
```
npm run extraction-demo
```

Or, with the server running (Terminal 1 above still up), hit the endpoint directly:
```
curl -X POST http://127.0.0.1:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Subject: cant login\n\nHi, this is Alex Kim (alex.kim@example.com). Getting invalid password error since this morning, need this fixed urgently for a demo."}'
```

## Exercise 3 - sync vs async benchmark

No server needed:
```
npm run benchmark
```

## Bonus - tool/function calling demo (multi-tool)

No server needed. Runs 3 scenarios showing the model choosing 1 tool, then 2
tools (across 2 rounds), then 1 tool again - not always the same tool set:
```
npm run tool-calling-demo
```

## All commands at a glance

| What | Command |
|---|---|
| Install deps | `npm install` |
| Start server (Ex. 1 & 2) | `npm run dev` |
| Ex. 1 client | `npm run stream-client -- "your question"` |
| Ex. 2 standalone demo | `npm run extraction-demo` |
| Ex. 3 benchmark | `npm run benchmark` |
| Bonus tool-calling demo | `npm run tool-calling-demo` |
| Type-check everything | `npm run typecheck` |

Note: if port 8000 is already taken (e.g. the Python `week2/` server is
running), start with a different port: `PORT=8010 npm run dev`, then use
`http://127.0.0.1:8010` in the client/curl commands instead.
