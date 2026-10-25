"""Week 10 tracing: latency and token usage for every model call and graph node, plus optional
LangSmith tracing.

LangSmith needs NO code here at all. `get_cloud_chat_model()` returns a standard LangChain
`ChatOpenAI` Runnable, so if `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set in the
environment (see .env.example), every model call this app makes is traced to LangSmith
automatically - LangChain's global callback does this for any Runnable, with nothing to wire up.

What IS wired up in this file is a second, always-on, local trace that needs no account and no
network: one JSON file per run (`week10/traces/<thread_id>.json`) recording every node's latency
and every model call's token usage. This exists so cost/latency numbers are available even if
you never set up LangSmith - and because "tracing" for this exercise should be checkable with a
plain `cat`, not require a logged-in dashboard to verify it worked.

Honest limitation: `with_structured_output()` (used by the planner and the retrieval query
writer) returns the parsed object, not the raw model reply, so its token usage isn't available
without restructuring those calls (`include_raw=True`) - those calls are still timed (latency),
just recorded with token counts of 0. Everything going through `ask_for`/`ask_for_text` in
agents/llm.py (the router, the relevance grader, the summarizer, the writer) has full token
counts, when the model actually returns usage metadata (OpenRouter free-tier models don't always).
"""

import contextvars
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from brief import config


@dataclass
class ModelCall:
    node: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_s: float


@dataclass
class NodeTiming:
    node: str
    duration_s: float


@dataclass
class RunTrace:
    thread_id: str
    topic: str
    nodes: list[NodeTiming] = field(default_factory=list)
    model_calls: list[ModelCall] = field(default_factory=list)
    _started: float = field(default_factory=time.monotonic, repr=False)

    def record_node(self, node: str, duration_s: float) -> None:
        self.nodes.append(NodeTiming(node, round(duration_s, 3)))

    def record_model_call(self, node: str, usage: dict | None, duration_s: float) -> None:
        usage = usage or {}
        self.model_calls.append(
            ModelCall(
                node=node,
                model=config.CLOUD_MODEL,
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                duration_s=round(duration_s, 3),
            )
        )

    def summary(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "topic": self.topic,
            "total_latency_s": round(time.monotonic() - self._started, 2),
            "node_count": len(self.nodes),
            "model_call_count": len(self.model_calls),
            "total_tokens": sum(c.total_tokens for c in self.model_calls),
            # Free-tier models cost $0; this multiplication exists so the same trace format works
            # unchanged if CLOUD_MODEL were ever switched to a paid one - see config.PRICE_PER_1K_TOKENS.
            "estimated_cost_usd": round(sum(c.total_tokens for c in self.model_calls) / 1000 * config.PRICE_PER_1K_TOKENS, 6),
            "nodes": [n.__dict__ for n in self.nodes],
            "model_calls": [c.__dict__ for c in self.model_calls],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2))


_current: contextvars.ContextVar["RunTrace | None"] = contextvars.ContextVar("brief_trace", default=None)


def start_trace(thread_id: str, topic: str) -> RunTrace:
    trace = RunTrace(thread_id=thread_id, topic=topic)
    _current.set(trace)
    return trace


def current_trace() -> "RunTrace | None":
    return _current.get()


def record_node(node: str, duration_s: float) -> None:
    trace = current_trace()
    if trace is not None:
        trace.record_node(node, duration_s)


def record_model_call(node: str, usage: dict | None, duration_s: float) -> None:
    trace = current_trace()
    if trace is not None:
        trace.record_model_call(node, usage, duration_s)
