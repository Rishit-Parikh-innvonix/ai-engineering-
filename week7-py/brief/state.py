import operator
from typing import Annotated, TypedDict

from brief.sources.types import SourceDocument
from brief.types import NumberedSource, RetrievedBatch


class ResearchState(TypedDict):
    """The shared notebook. Every agent reads from it and returns only the fields it wants to
    change; LangGraph merges those changes in. Agents never call each other directly - they
    collaborate purely by what they write here.

    Fields typed Annotated[list, operator.add] have a *reducer*: several parallel branches write
    them at once, and a plain field would keep only the last writer's value. The reducer tells
    LangGraph to merge them instead (here: append)."""

    topic: str

    # Router: how the topic is answered, and the live-tool readings it produced
    route: str  # "research" | "weather" | "mixed" | "unsupported"
    weather_city: str
    research_topic: str  # the part of a MIXED topic that still needs research
    tool_docs: Annotated[list[SourceDocument], operator.add]

    # Planner
    sub_questions: list[str]

    # Retrieval (written by several parallel branches, so it uses a reducer)
    retrieved: Annotated[list[RetrievedBatch], operator.add]
    retrieval_rounds: int
    library: list[NumberedSource]

    # Summarizer
    notes: str

    # Final answer + citation check
    draft: str
    draft_count: int
    citation_problems: list[str]

    # Things worth telling the reader about this run (a source was down, a retry happened, ...)
    run_notes: Annotated[list[str], operator.add]

    # Result
    report: str
    report_path: str


def initial_state(topic: str) -> ResearchState:
    """LangGraph does not fill in defaults for a TypedDict state, so every run starts from this."""
    return ResearchState(
        topic=topic,
        route="research",
        weather_city="",
        research_topic="",
        tool_docs=[],
        sub_questions=[],
        retrieved=[],
        retrieval_rounds=0,
        library=[],
        notes="",
        draft="",
        draft_count=0,
        citation_problems=[],
        run_notes=[],
        report="",
        report_path="",
    )
