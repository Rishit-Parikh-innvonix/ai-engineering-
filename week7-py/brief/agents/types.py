from dataclasses import dataclass
from typing import Awaitable, Callable

from brief.sources.types import SourceDocument
from brief.types import NumberedSource, QueryPlan, RouteDecision


@dataclass(frozen=True)
class Agents:
    """The AI agents, as the graph sees them. The graph depends on this shape and not on any
    particular model, which is what lets the tests swap in scripted agents and check the routing
    (retries, redrafts) deterministically, without a live model or network.

    route(topic)            Routing agent: research, a live-data tool, both, or neither.
    plan(topic)             Planning agent: splits a topic into focused sub-questions.
    craft_queries(...)      Retrieval agent: a search query per source, rewording if earlier ones fell short.
    grade_relevance(...)    Retrieval agent, second job: keeps only the results that are on-topic.
    summarize(...)          Summarization agent: short, source-tagged notes.
    write(...)              Final answer agent: the brief, fixing any listed citation problems.
    """

    route: Callable[[str], Awaitable[RouteDecision]]
    plan: Callable[[str], Awaitable[list[str]]]
    craft_queries: Callable[..., Awaitable[QueryPlan]]  # (topic, sub_question, previous_queries)
    grade_relevance: Callable[..., Awaitable[list[SourceDocument]]]  # (topic, sub_question, documents)
    summarize: Callable[..., Awaitable[str]]  # (topic, sub_questions, library)
    write: Callable[..., Awaitable[str]]  # (topic, notes, library, problems, previous_draft)


__all__ = ["Agents", "NumberedSource"]
