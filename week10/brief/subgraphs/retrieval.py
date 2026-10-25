"""Subgraph 1 - RETRIEVAL: the retrieve-and-check loop, as a self-contained mini workflow.

  START -> retrieve (one parallel branch per sub-question) -> coverage -+-> END
                ^                                                      |
                '------- nothing relevant for some question? retry ----'

It has its own small state (RetrievalState), so it can be built, tested and reused without the rest
of the pipeline. The parent graph feeds it the sub-questions and takes back the numbered library."""

import asyncio
import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from brief.agents.types import Agents
from brief.logic import LIMITS, build_library, find_uncovered_sub_questions, previous_queries_for
from brief.sources.types import SourceDocument, SourceFetcher
from brief.types import NumberedSource, QueryPlan, RetrievedBatch
from brief.utils import log_step, since


class RetrievalState(TypedDict):
    topic: str
    sub_questions: list[str]
    tool_docs: list[SourceDocument]  # live readings already fetched (weather); numbered first
    retrieved: Annotated[list[RetrievedBatch], operator.add]  # parallel branches append here
    retrieval_rounds: int
    library: list[NumberedSource]
    run_notes: Annotated[list[str], operator.add]


def _message(error: BaseException) -> str:
    return str(error) or type(error).__name__


def initial_retrieval_state(topic: str, sub_questions: list[str], tool_docs: list[SourceDocument]) -> RetrievalState:
    return RetrievalState(
        topic=topic, sub_questions=sub_questions, tool_docs=tool_docs, retrieved=[], retrieval_rounds=0, library=[], run_notes=[]
    )


def create_retrieval_subgraph(*, agents: Agents, fetchers: list[SourceFetcher]):
    # Runs once per sub-question, all at the same time. `task` is a Send payload, not the whole state.
    async def retrieve(task: dict):
        started = time.monotonic()
        notes: list[str] = []
        sub_question: str = task["sub_question"]

        try:
            queries = await agents.craft_queries(
                topic=task["topic"], sub_question=sub_question, previous_queries=task["previous_queries"]
            )
        except Exception as error:  # noqa: BLE001
            # One flaky model call shouldn't sink the whole run: fall back to the plain question text.
            queries = QueryPlan.model_construct(wikipedia=sub_question, arxiv=sub_question, hackernews=sub_question)
            notes.append(
                f'The retrieval agent could not write search queries for "{sub_question}" ({_message(error)}), '
                "so the question text itself was used as the search query."
            )

        # Every source is searched at once; return_exceptions means one source being down never hides the others.
        results = await asyncio.gather(
            *(fetcher.fetch(getattr(queries, fetcher.kind)) for fetcher in fetchers), return_exceptions=True
        )
        fetched: list[SourceDocument] = []
        for fetcher, result in zip(fetchers, results):
            if isinstance(result, BaseException):
                notes.append(f'{fetcher.label} was unavailable for "{sub_question}": {_message(result)}')
            else:
                fetched.extend(result)

        # Search engines return *something* for any query, so found-something is not the same as
        # found-the-answer. The agent judges relevance and drops the rest; if that judging call
        # itself fails, keep everything (and say so) rather than lose good results.
        documents = fetched
        try:
            documents = await agents.grade_relevance(topic=task["topic"], sub_question=sub_question, documents=fetched)
        except Exception as error:  # noqa: BLE001
            notes.append(f'Relevance checking failed for "{sub_question}" ({_message(error)}); all results were kept unchecked.')

        log_step(
            "Retrieval",
            f"sub-question {task['index'] + 1} (attempt {task['attempt']}): kept {len(documents)} relevant of {len(fetched)} found ({since(started)})",
        )
        print("             searched: " + " | ".join(f'{f.label} "{getattr(queries, f.kind)}"' for f in fetchers))
        return {
            "retrieved": [
                RetrievedBatch(sub_question_index=task["index"], attempt=task["attempt"], queries=queries, documents=documents)
            ],
            "run_notes": notes,
        }

    # Deterministic check - no AI. Runs after ALL parallel retrieval branches have finished.
    def coverage(state: RetrievalState):
        library = build_library(state["retrieved"], state["tool_docs"])
        uncovered = find_uncovered_sub_questions(len(state["sub_questions"]), state["retrieved"])
        round_number = state["retrieval_rounds"] + 1
        numbers = ", ".join(str(i + 1) for i in uncovered)
        names = "; ".join(f'"{state["sub_questions"][i]}"' for i in uncovered)

        log_step(
            "Coverage check",
            f"{len(library)} unique sources, every sub-question covered"
            if not uncovered
            else f"{len(library)} unique sources; nothing relevant yet for sub-question(s) {numbers} "
            f"(round {round_number} of {LIMITS.max_retrieval_rounds})",
        )

        if uncovered and round_number < LIMITS.max_retrieval_rounds:
            note = [f"Round {round_number}: nothing relevant yet for {names}, so the queries were rewritten and searched again."]
        elif uncovered:
            note = [f"Retrieval gave up after {LIMITS.max_retrieval_rounds} rounds; found nothing relevant for: {names}."]
        else:
            note = []
        return {"library": library, "retrieval_rounds": round_number, "run_notes": note}

    # Fan-out: one Send per sub-question makes LangGraph run that many `retrieve` branches in parallel.
    def fan_out(state: RetrievalState):
        return [
            Send("retrieve", {"topic": state["topic"], "sub_question": q, "index": i, "attempt": 1, "previous_queries": []})
            for i, q in enumerate(state["sub_questions"])
        ]

    def after_coverage(state: RetrievalState):
        uncovered = find_uncovered_sub_questions(len(state["sub_questions"]), state["retrieved"])
        if uncovered and state["retrieval_rounds"] < LIMITS.max_retrieval_rounds:
            return [
                Send(
                    "retrieve",
                    {
                        "topic": state["topic"],
                        "sub_question": state["sub_questions"][i],
                        "index": i,
                        "attempt": state["retrieval_rounds"] + 1,
                        "previous_queries": previous_queries_for(state["retrieved"], i),
                    },
                )
                for i in uncovered
            ]
        return END

    graph = StateGraph(RetrievalState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("coverage", coverage)
    graph.add_conditional_edges(START, fan_out, ["retrieve"])
    graph.add_edge("retrieve", "coverage")
    graph.add_conditional_edges("coverage", after_coverage, ["retrieve", END])
    return graph.compile()
