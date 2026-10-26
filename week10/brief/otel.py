"""Week 10 upgrade: real OpenTelemetry spans alongside the custom JSON trace in tracing.py.

Why both exist: tracing.py's JSON file is this project's own cost/latency ledger - zero
dependencies, checkable with `cat`, and it's what estimated_cost_usd is computed from. This
module emits actual OTel spans for the same nodes and model calls, so a run can also be viewed in
any OTel-compatible backend (Jaeger, Tempo, Honeycomb, ...) - closer to what the assignment means
by "OpenTelemetry concepts" than a bespoke JSON format alone.

Where spans go, by default: a line-per-span log file at `week10/traces/otel-spans.log` (so stdout
stays readable during a run - the ConsoleSpanExporter is otherwise very verbose). Set
OTEL_EXPORTER_OTLP_ENDPOINT (see .env.example) to send spans to a real collector instead - e.g.
run `docker run -p 4318:4318 -p 16686:16686 jaegertracing/all-in-one` and point it at
http://localhost:4318/v1/traces, then view the trace tree at http://localhost:16686.

Each run gets one root span ("run.<thread_id>") and every node/model-call becomes a child span
under it, so a trace viewer shows the same shape as the flow diagram in graph.py's docstring.

Degrades to a no-op if the opentelemetry packages aren't installed, same philosophy as everything
else in guardrails/ and tracing.py: a missing optional dependency should never crash a run.
"""

import contextvars
import os
import time

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_initialized = False
_tracer = None
_provider = None
_root_span: contextvars.ContextVar = contextvars.ContextVar("brief_otel_root_span", default=None)


def _init() -> None:
    global _initialized, _provider
    if _initialized or not _AVAILABLE:
        return
    _provider = TracerProvider(resource=Resource.create({"service.name": "ai-engineering-week10-brief"}))
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        from brief.config import TRACES_DIR

        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        log_file = open(TRACES_DIR / "otel-spans.log", "a")
        _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=log_file)))
    trace.set_tracer_provider(_provider)
    _initialized = True


def get_tracer():
    """Returns a real tracer, or None if the SDK isn't installed - every other function in this
    module already checks for None, so callers elsewhere never need an isinstance/try of their own."""
    global _tracer
    if not _AVAILABLE:
        return None
    _init()
    if _tracer is None:
        _tracer = trace.get_tracer("brief")
    return _tracer


def start_run_span(thread_id: str, topic: str) -> None:
    """One root span per run (one per process invocation, same reasoning as tracing.py's
    one-trace-file-per-invocation). Every node/model-call span below is its child."""
    tracer = get_tracer()
    if tracer is None:
        return
    span = tracer.start_span(f"run.{thread_id}", attributes={"brief.thread_id": thread_id, "brief.topic": topic[:200]})
    _root_span.set(span)


def end_run_span() -> None:
    span = _root_span.get()
    if span is not None:
        span.end()
        _root_span.set(None)
    if _provider is not None:
        _provider.force_flush()


def record_after_the_fact(name: str, duration_s: float, attributes: dict | None = None) -> None:
    """Node/model-call work in this project is already timed with time.monotonic() before this is
    called (graph.py, agents/llm.py), so rather than restructure every call site to wrap itself in
    `start_as_current_span`, this creates the span after the fact with an explicit start/end time -
    same duration, same place in the trace tree, zero changes needed at the call sites beyond the
    tracing.record_node/record_model_call calls that already existed for the JSON trace."""
    tracer = get_tracer()
    if tracer is None:
        return
    root = _root_span.get()
    ctx = trace.set_span_in_context(root) if root is not None else None
    end_ns = time.time_ns()
    start_ns = end_ns - int(duration_s * 1e9)
    span = tracer.start_span(name, context=ctx, start_time=start_ns, attributes={k: v for k, v in (attributes or {}).items() if v is not None})
    span.end(end_time=end_ns)
