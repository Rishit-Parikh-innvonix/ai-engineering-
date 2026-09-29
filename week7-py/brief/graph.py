"""The flowchart. Everything about HOW the agents are coordinated lives here:

  START -> router -> research:    plan (below)
                  -> weather:     weather_tool -> summarize (below)      live data: no searching
                  -> mixed:       weather_tool -> plan (below)           both, in one library
                  -> unsupported: save                                   needs live data we lack
           (the weather tool failing falls back to plan; the router failing means "research")

  plan -> retrieve (one parallel branch per sub-question; each writes queries, searches 3 sources
                    at once, and drops off-topic results) -> coverage
                ^                                               |
                '---- nothing relevant for some question? retry (max 2) ---'
  coverage -> summarize -> write -> check_citations -> save -> END
                           ^              |
                           '-- bad cite? redraft (max 2)
  coverage -> no_sources -> save   (nothing found at all: report that honestly, don't guess)
  summarize/write failing after retries -> save, listing the raw excerpts instead of prose

The path is fixed by this code. The AI agents only fill in the content of each box, which is what
makes this a deterministic workflow instead of an agent deciding its own steps.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from brief.agents.types import Agents
from brief.logic import (
    LIMITS,
    build_extractive_draft,
    build_library,
    build_report,
    check_citations,
    find_uncovered_sub_questions,
    previous_queries_for,
)
from brief.sources.types import SourceFetcher
from brief.state import ResearchState
from brief.tools.weather import WeatherTool
from brief.types import QueryPlan, RetrievedBatch
from brief.utils import log_step, since


def _message(error: BaseException) -> str:
    return str(error) or type(error).__name__


def create_research_graph(
    *,
    agents: Agents,
    fetchers: list[SourceFetcher],
    weather: WeatherTool,
    save_report: Callable[[str, str], Awaitable[str]],
):
    # Agents and sources are arguments, not imports, so tests can plug in scripted stand-ins.

    # ---- Agent nodes: each one reads the notebook and returns only what it changed ----

    # The first diamond: research, a live-data tool, both, or neither. If the routing call itself
    # fails, plain research is the safe default - it is what the workflow did before it had tools.
    async def router(state: ResearchState):
        started = time.monotonic()
        try:
            decision = await agents.route(state["topic"])
        except Exception as error:  # noqa: BLE001
            log_step("Router", f"failed ({_message(error)}) - treating the topic as plain research")
            return {
                "route": "research",
                "run_notes": [f"The routing agent failed ({_message(error)}), so the topic was researched without live-data tools."],
            }
        log_step("Router", f"{decision.kind}{f' ({decision.city})' if decision.city else ''} ({since(started)})")
        return {"route": decision.kind, "weather_city": decision.city, "research_topic": decision.research_topic}

    # Plain code calling the weather API. On failure the run degrades to research, and says so.
    async def weather_tool(state: ResearchState):
        started = time.monotonic()
        try:
            document = await weather.lookup(state["weather_city"])
        except Exception as error:  # noqa: BLE001
            log_step("Weather tool", f"failed ({_message(error)}) - falling back to research")
            return {
                "route": "research",
                "run_notes": [
                    f'The weather tool could not get the current weather for "{state["weather_city"]}" '
                    f"({_message(error)}), so this brief comes from the research sources only and "
                    "contains no live weather data."
                ],
            }
        log_step("Weather tool", f"{document.title} ({since(started)})")
        return {
            "tool_docs": [document],
            # Weather-only runs skip retrieval, so the library and the one "sub-question" are set
            # here. (A mixed run rebuilds the library in the coverage check, tool reading first.)
            "library": build_library([], [document]),
            "sub_questions": [state["topic"]] if state["route"] == "weather" else [],
        }

    def unsupported(state: ResearchState):
        log_step("Router", "needs live data that no tool provides - reporting that instead of guessing")
        return {
            "draft": (
                "## Overview\n\nThis question needs live, up-to-the-minute data (for example prices, scores, traffic or "
                "breaking news) that this assistant has no tool for, and the research sources (Wikipedia, arXiv, Hacker News) "
                "cannot provide it. No brief was written rather than guessing."
            ),
            "run_notes": ["The routing agent classified this topic as needing live data other than weather, which is not available."],
        }

    async def plan(state: ResearchState):
        started = time.monotonic()
        sub_questions = await agents.plan(state["research_topic"] or state["topic"])
        log_step("Planner", f"split the topic into {len(sub_questions)} sub-questions ({since(started)}):")
        for i, question in enumerate(sub_questions, start=1):
            print(f"             {i}. {question}")
        return {"sub_questions": sub_questions}

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
        fetched = []
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
    def coverage(state: ResearchState):
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

    def no_sources(state: ResearchState):
        log_step("Coverage check", "no sources at all were retrieved - reporting that instead of guessing")
        return {
            "draft": "## Overview\n\nNo sources could be retrieved for this topic, so no brief could be written. "
            "See the notes at the end of this report for what went wrong."
        }

    # The retrieval work is the expensive part, so if the summarizing or writing agent fails after
    # all its retries (free models time out) the run degrades instead of dying: the report still
    # lists the relevant excerpts that were found, unedited.
    async def summarize(state: ResearchState):
        started = time.monotonic()
        # In a mixed run the planner only saw the research half of the topic, so the weather reading
        # would have no question to belong to and the summarizer would leave it out (seen live).
        sub_questions = state["sub_questions"]
        if state["route"] == "mixed" and state["tool_docs"]:
            sub_questions = [f"What is the current weather in {state['weather_city']}?", *sub_questions]
        try:
            notes = await agents.summarize(topic=state["topic"], sub_questions=sub_questions, library=state["library"])
        except Exception as error:  # noqa: BLE001
            log_step("Summarizer", f"failed ({_message(error)}) - falling back to the raw excerpts")
            return {
                "draft": build_extractive_draft(state["library"]),
                "run_notes": [
                    f"The summarization agent failed ({_message(error)}), so this report lists the retrieved excerpts as-is instead of a written brief."
                ],
            }
        log_step("Summarizer", f"condensed {len(state['library'])} sources into notes ({since(started)})")
        return {"notes": notes}

    async def write(state: ResearchState):
        started = time.monotonic()
        draft_count = state["draft_count"] + 1
        try:
            draft = await agents.write(
                topic=state["topic"],
                notes=state["notes"],
                library=state["library"],
                problems=state["citation_problems"],
                previous_draft=state["draft"],
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
        what = "wrote the first draft" if draft_count == 1 else f"wrote draft {draft_count} (fixing citation problems)"
        log_step("Answer writer", f"{what} ({since(started)})")
        return {"draft": draft, "draft_count": draft_count}

    # Deterministic check - no AI. The writer's prose is trusted only after this passes.
    def check_citations_node(state: ResearchState):
        problems = check_citations(state["draft"], len(state["library"]))
        log_step("Citation check", "every citation points to a real source" if not problems else " ".join(problems))
        return {"citation_problems": problems}

    async def save(state: ResearchState):
        run_notes = list(state["run_notes"])
        if state["citation_problems"]:
            run_notes.append(
                f"Warning: the writer could not fix all citation problems after {state['draft_count']} drafts, "
                f"so treat the [n] references with care: {' '.join(state['citation_problems'])}"
            )
        report = build_report(
            topic=state["topic"],
            body=state["draft"],
            library=state["library"],
            run_notes=run_notes,
            generated_on=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        )
        report_path = await save_report(state["topic"], report)
        log_step("Saved", report_path)
        return {"report": report, "report_path": report_path}

    # ---- Routers: these are the diamonds in the flowchart ----

    def after_route(state: ResearchState):
        if state["route"] in ("weather", "mixed"):
            return "weather_tool"
        return "unsupported" if state["route"] == "unsupported" else "plan"

    # Weather-only goes straight to summarizing; a mixed run (or a failed tool) goes on to research.
    def after_weather(state: ResearchState):
        return "summarize" if state["route"] == "weather" else "plan"

    # Fan-out: one Send per sub-question makes LangGraph run that many `retrieve` branches in parallel.
    def fan_out_retrieval(state: ResearchState):
        return [
            Send("retrieve", {"topic": state["topic"], "sub_question": q, "index": i, "attempt": 1, "previous_queries": []})
            for i, q in enumerate(state["sub_questions"])
        ]

    def after_coverage(state: ResearchState):
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
        return "summarize" if state["library"] else "no_sources"

    # No notes means the summarizer failed and already set a fallback draft: skip the writer.
    def after_summarize(state: ResearchState):
        return "write" if state["notes"] else "save"

    def after_citation_check(state: ResearchState):
        return "write" if state["citation_problems"] and state["draft_count"] < LIMITS.max_drafts else "save"

    graph = StateGraph(ResearchState)
    graph.add_node("router", router)
    graph.add_node("weather_tool", weather_tool)
    graph.add_node("unsupported", unsupported)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("coverage", coverage)
    graph.add_node("no_sources", no_sources)
    graph.add_node("summarize", summarize)
    graph.add_node("write", write)
    graph.add_node("check_citations", check_citations_node)
    graph.add_node("save", save)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", after_route, ["weather_tool", "unsupported", "plan"])
    graph.add_conditional_edges("weather_tool", after_weather, ["summarize", "plan"])
    graph.add_edge("unsupported", "save")
    graph.add_conditional_edges("plan", fan_out_retrieval, ["retrieve"])
    graph.add_edge("retrieve", "coverage")
    graph.add_conditional_edges("coverage", after_coverage, ["retrieve", "summarize", "no_sources"])
    graph.add_edge("no_sources", "save")
    graph.add_conditional_edges("summarize", after_summarize, ["write", "save"])
    graph.add_edge("write", "check_citations")
    graph.add_conditional_edges("check_citations", after_citation_check, ["write", "save"])
    graph.add_edge("save", END)
    return graph.compile()
