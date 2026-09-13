# Week 6 - MCP: Building a Tiny Personal Assistant

## The idea, in one paragraph

A personal assistant (think ChatGPT's own "memory" feature, or Siri) needs
two basic abilities: check real files for you, and remember things about you
between conversations. In `week5`, every ability you gave the AI had to be
hand-written (a weather tool, a calculator). This week uses **MCP (Model
Context Protocol)** instead - a way to plug in ready-made helper programs
that already know how to do a job, the same way plugging in a USB printer
gives your computer a new ability without you writing any printer code.
Built with **LangChain.js** (TypeScript), **OpenRouter** for the cloud
model, and `@langchain/mcp-adapters` to do the "plugging in."

## One-time setup

```
cd week6
npm install --legacy-peer-deps
```

Uses the repo-root `.env` for `OPENROUTER_API_KEY` if `week6/.env` doesn't
exist (see `.env.example`).

## Run

```
npm run mcp-demo
```

This runs all 3 exercises back to back, telling one continuous story:

1. **See what the assistant's helpers can do** - starts both helper
   programs and lists every ability each one offers, before using either of
   them for anything. Like reading the spec sheet on a new gadget.
2. **Teach the assistant something about you** - using *only* the
   remembering helper, tell it your name and what you're learning, then ask
   it to recall what it knows. This is the same idea as ChatGPT's own
   "memory" - tell it something once, and it can bring that fact up again
   later, in a different conversation.
3. **(Optional) One assistant, both helpers, one real task** - ask it to
   check a real to-do list file *and* remember your top priority from it.
   Neither helper alone can do this: reading the file needs the
   file-reading helper, and remembering the priority afterward needs the
   remembering helper. This is the actual point of combining MCP servers -
   the assistant picks whichever helper(s) a question actually needs.

## The two helpers used, and why

The assignment suggests several options: filesystem, fetch-a-webpage, git,
SQLite, DuckDuckGo, or Wikipedia. This project uses:

- **`@modelcontextprotocol/server-filesystem`** - reads/writes files, but
  only inside `week6/sandbox/`. It's physically incapable of touching
  anything outside that one folder - a safety feature built into the
  program itself.
- **`@modelcontextprotocol/server-memory`** - remembers facts (who you are,
  what you told it) and saves them to a file so they survive between
  separate runs of the script, the same way your phone remembers your
  contacts after a restart.

Both are official - written by the same team that created MCP itself - need
zero API keys, and never touch the network.

**Why not the assignment's other suggestions?** Checked each one with
`npm audit` instead of just guessing:
- A "fetch a webpage" helper (`mcp-fetch-server`) depends on a package
  (`private-ip`) with a real, **unpatched** security hole - its entire job
  is blocking requests to your own internal network, and that block itself
  can be bypassed. `npm audit` flags it high severity, no fix available.
- A SQLite database helper (`mcp-server-sqlite-npx`) pulls in old, broken
  build tooling with a **critical** severity vulnerability, also unfixed.

Both filesystem and memory came back completely clean - zero
vulnerabilities - which is exactly what you'd want before letting an AI use
either one.

## Why the assistant "remembers" differently on your second run

The memory helper's whole point is real, durable memory - it's not supposed
to forget when the script exits. So if you run `npm run mcp-demo` a second
time, the assistant might notice it already knows your name from last time,
and just confirm it instead of "creating" it again. That's not a bug - it's
the memory genuinely working, the same way you wouldn't expect a friend to
forget your name every time you see them again.

(One thing worth knowing, for anyone curious how this actually works: this
memory helper's default save location is *inside its own installed-package
folder*, which is a bad spot - it gets wiped every time you reinstall
packages, and it's easy to lose track of. `src/mcpDemo.ts` points it at
`week6/mcp-memory/memory.jsonl` instead, a location this project actually
controls. That folder is gitignored since it's just generated data - delete
it any time you want the assistant to forget everything and start fresh.)

## About the model

`week5` used `minimax/minimax-m2.7:free` on OpenRouter. Between `week5` and
this week, OpenRouter pulled that model's free tier (confirmed live - it now
404s, pointing at a paid version instead). Checked OpenRouter's live model
list for a currently-free model that actually supports tool calling and
switched to `nex-agi/nex-n2.5-pro:free` - verified against all 3 exercises.
If this model also stops being free at some point, `src/config.ts`'s
`CLOUD_MODEL` is the only place that needs to change.

## Project layout

```
week6/
  sandbox/
    todo.txt          a real to-do list the assistant can read (tracked in git)
  mcp-memory/
    memory.jsonl        what the assistant remembers about you (gitignored - generated)
  src/
    config.ts          OpenRouter key/URL, cloud model name
    models.ts           ChatOpenAI (OpenRouter) factory
    utils.ts              console printers (printSectionHeader, printPanel)
    mcpDemo.ts               connects to both helpers and runs all 3 exercises
```

Both helpers run over **stdio**: `MultiServerMCPClient` starts each one as
its own separate running program (via `npx`, which uses the
already-installed local copy instead of downloading anything) and talks to
it by passing text messages back and forth - no ports, no web addresses
involved. `mcpDemo.ts` shuts both helpers down in a `finally` block at the
end (`mcpClient.close()`) so your terminal actually returns control to you
instead of hanging with two invisible programs still running.
