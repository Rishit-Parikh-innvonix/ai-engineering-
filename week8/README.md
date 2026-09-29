# Week 8 - Human-in-the-Loop, Persistence & Subgraphs

The week 7 research assistant (Python + LangGraph), extended with the three week 8 topics:

| Topic | What it does here | Where |
|---|---|---|
| **Persistence & checkpoints** | The state is saved to SQLite after every step, keyed by a thread id. A run can be paused, stopped, crash or time out - and be resumed from its last step, even from a new process. | `brief/persistence.py`, `research.py` |
| **Human-in-the-loop** | The run pauses with `interrupt()` at two critical points and does nothing irreversible until a person answers: **plan review** (before minutes of searching) and **draft review** (before the report is saved). | `graph.py`, `review.py` |
| **Subgraphs** | The retrieve-and-check loop and the write-and-check loop are self-contained mini graphs, each with its own small state, tested on their own and plugged into the main pipeline. | `subgraphs/` |

```
cd week8
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m brief "Why does the Indian monsoon happen?"      # asks you at the two review points
pytest                                                     # 86 tests, no network and no API key
```
`OPENROUTER_API_KEY` is read from `week8/.env`, the repo-root `.env`, or `../week7/.env`. Only `:free` OpenRouter models.

## The flow

```
topic -> ROUTER -+- research -> PLAN -> [REVIEW PLAN] -> (research subgraph) -> SUMMARIZE -+
                 |- weather  -> WEATHER TOOL ---------------------------------> SUMMARIZE -+
                 |- mixed    -> WEATHER TOOL -> PLAN -> [REVIEW PLAN] -> (research subgraph) ...
                 '- unsupported -> SAVE (nothing written by the AI, nothing to approve)
                                                                                           |
   +-------------------------------------------------------------------------------------'
   v
 (drafting subgraph) -> [APPROVE DRAFT] --approve--> SAVE -> END
        ^                     |     '--reject---> END   (nothing is saved)
        '--- revise + feedback (max 2)
```
`[...]` = a pause for a person. `(...)` = a subgraph.

## 1. Persistence - how to use it

Every run has a **thread id** (printed at the start; or choose one with `--thread`).

```
python -m brief "topic" --no-prompt              # runs to the first review, saves, exits
python -m brief status THREAD                    # PAUSED, waiting for a person (plan_review)
python -m brief resume THREAD --approve          # a NEW process continues from the checkpoint
python -m brief threads                          # every saved run
```
It also recovers from failures, not just pauses. If a run is stopped (Ctrl+C, `--timeout`, a crash,
a lost connection), `python -m brief resume THREAD` continues from the last saved step. Verified
live: a run cut off after 25 s mid-research was resumed later and finished without asking about the
plan again.

**Checkpoint granularity - a real trade-off.** LangGraph saves after each step of the *main* graph.
A subgraph called from a wrapper node counts as one step, so a run that dies *inside* the research
subgraph redoes that subgraph's searching (the router, plan and approvals are not repeated). The
alternative - adding the compiled subgraph directly as a node - would keep inner checkpoints, but
needs state keys shared with the parent, and shared reducer fields hand a parent's own values back
to it (appended a second time). I chose correctness and clear boundaries; `test_subgraphs.py`
proves nothing is duplicated.

State holds pydantic models, so `persistence.py` registers them with the serializer. Without that,
current LangGraph warns and a future version refuses to load them.

## 2. Human-in-the-loop - how it works

```python
answer = interrupt({"kind": "plan_review", "sub_questions": [...], ...})   # the run stops here
# ...later, someone calls graph.ainvoke(Command(resume={"action": "approve"}), config)...
# and `answer` is now {"action": "approve"} - the graph carries on
```

| Review | You can | Effect |
|---|---|---|
| **Plan** | `approve`, `edit` (new sub-questions), `reject` | edit changes what is searched; reject stops with nothing searched or saved |
| **Draft** | `approve`, `revise` + feedback (max 2), `reject` | revise re-runs the drafting subgraph with your feedback; only approve saves the report |

Rules the code follows (each has a test):
- **On resume LangGraph re-runs the paused node from its first line**, so a review node does nothing
  before its `interrupt` that would be wrong to repeat (no saving, no API calls). Each review is its own tiny node.
- **An answer is untrusted input.** `review.py` validates it; a bad answer is shown back with the
  reason and the run stays paused - it never crashes.
- **Auto-approval is never disguised.** `--auto` approves both reviews without a person, and the
  report then says *"Auto-approved (--auto): no person read this draft"* instead of "a human approved".
- **Nothing to review, no pause.** Unsupported-live-data and no-sources runs contain nothing the AI
  wrote, so they don't stop. If the summarizer failed there are no notes to rewrite from, so only approve/reject are offered.

## 3. Subgraphs

| Subgraph | Steps | Its own state |
|---|---|---|
| `subgraphs/retrieval.py` | fan out (`Send`) -> retrieve -> coverage check -> retry or end | sub-questions, retrieved batches, rounds, library |
| `subgraphs/drafting.py` | write -> citation check -> redraft or end | notes, library, feedback, draft, problems |

The parent hands each one its input and takes back **only its results** (`research` and `drafting`
wrapper nodes in `graph.py`). Human feedback reaches the writer through the same `problems` channel
as citation errors, so one loop serves both.

## Honest limits
- A reviewer's feedback is an instruction to a weak free model, not a guarantee: it may follow it partly (seen live).
- A citation check confirms `[n]` exists, not that the source says it. Hacker News comments are opinions.
- The checkpoint file (`checkpoints.sqlite`) holds full run state, including fetched text. It is gitignored.
- **Wikipedia's robot policy** rejects generic User-Agents with HTTP 403 (found live, and it also
  hit the week7-py version). Requests now name a contact URL; set `RESEARCH_BRIEF_CONTACT` in `.env`
  to your own (repo URL or email). The default is a neutral placeholder that passes the check but reaches nobody.

## Files
```
week8/brief/
  research.py      CLI: run / resume / status / threads         graph.py        the main flowchart
  persistence.py   SQLite checkpointer + registered types        review.py       validating a human's answer
  subgraphs/       retrieval.py, drafting.py                     state.py        the shared notebook
  agents/ sources/ tools/ logic.py ...   (unchanged from week7-py)
week8/tests/       test_human_review.py, test_persistence.py, test_subgraphs.py, test_review.py, test_graph.py, ...
```
