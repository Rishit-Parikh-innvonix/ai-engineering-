"""Subgraph 2 - DRAFTING: write the brief, then check its citations, redrafting while they are wrong.

  START -> write -> check -+-> END              (citations are real, or drafts ran out)
             ^             |
             '-- bad cite? redraft (max 2)

Also small and self-contained. Human feedback ("make it shorter", "add the risks") arrives as
`feedback` and is handed to the writer next to any citation problems, so one loop serves both."""

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from brief.agents.types import Agents
from brief.logic import LIMITS, build_extractive_draft, check_citations
from brief.types import NumberedSource
from brief.utils import log_step, since


class DraftState(TypedDict):
    topic: str
    notes: str
    library: list[NumberedSource]
    previous_draft: str  # the draft a reviewer asked to change ("" for a first draft)
    feedback: list[str]  # what the reviewer asked for
    draft: str
    draft_count: int
    citation_problems: list[str]
    run_notes: Annotated[list[str], operator.add]


def initial_draft_state(
    *, topic: str, notes: str, library: list[NumberedSource], previous_draft: str = "", feedback: list[str] | None = None
) -> DraftState:
    return DraftState(
        topic=topic,
        notes=notes,
        library=library,
        previous_draft=previous_draft,
        feedback=feedback or [],
        draft="",
        draft_count=0,
        citation_problems=[],
        run_notes=[],
    )


def _message(error: BaseException) -> str:
    return str(error) or type(error).__name__


def create_drafting_subgraph(*, agents: Agents):
    async def write(state: DraftState):
        started = time.monotonic()
        draft_count = state["draft_count"] + 1
        problems = [f"Reviewer feedback (must be addressed): {f}" for f in state["feedback"]] + state["citation_problems"]
        try:
            draft = await agents.write(
                topic=state["topic"],
                notes=state["notes"],
                library=state["library"],
                problems=problems,
                previous_draft=state["draft"] or state["previous_draft"],
            )
        except Exception as error:  # noqa: BLE001
            log_step("Answer writer", f"failed ({_message(error)}) - falling back to the raw excerpts")
            return {
                "draft": build_extractive_draft(state["library"]),
                "draft_count": draft_count,
                "run_notes": [
                    f"The answer-writing agent failed ({_message(error)}), so this report lists the retrieved excerpts as-is instead of a written brief."
                ],
            }
        what = "wrote the first draft" if draft_count == 1 else f"wrote draft {draft_count} (fixing problems)"
        log_step("Answer writer", f"{what} ({since(started)})")
        return {"draft": draft, "draft_count": draft_count}

    # Deterministic check - no AI. The writer's prose is trusted only after this passes.
    def check(state: DraftState):
        problems = check_citations(state["draft"], len(state["library"]))
        log_step("Citation check", "every citation points to a real source" if not problems else " ".join(problems))
        return {"citation_problems": problems}

    def after_check(state: DraftState):
        return "write" if state["citation_problems"] and state["draft_count"] < LIMITS.max_drafts else END

    graph = StateGraph(DraftState)
    graph.add_node("write", write)
    graph.add_node("check", check)
    graph.add_edge(START, "write")
    graph.add_edge("write", "check")
    graph.add_conditional_edges("check", after_check, ["write", END])
    return graph.compile()
