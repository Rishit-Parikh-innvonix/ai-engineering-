"""The main flowchart. Everything about HOW the agents are coordinated lives here:

  START -> router -> research:    plan -> REVIEW PLAN -> [research subgraph] -> summarize -> ...
                  -> weather:     weather_tool -> summarize -> ...        live data: no searching
                  -> mixed:       weather_tool -> plan -> REVIEW PLAN -> [research subgraph] -> ...
                  -> unsupported: save                                    needs live data we lack
           (the weather tool failing falls back to plan; the router failing means "research")

  ... summarize -> [drafting subgraph] -> APPROVE DRAFT -+-> save -> END
                          ^                              |
                          '---- reviewer asked for changes (max 2)
  research subgraph finds nothing at all -> no_sources -> save
  summarize failing -> approve_draft (raw excerpts instead of prose)
  a reviewer rejecting the plan or the draft -> END, nothing is saved

Three new ideas on top of week 7:
  * PERSISTENCE   a checkpointer saves the state after every step, so a run can stop and be resumed.
  * HUMAN IN THE LOOP   `interrupt()` in REVIEW PLAN and APPROVE DRAFT pauses the run until a person answers.
  * SUBGRAPHS     the retrieve-and-check loop and the write-and-check loop are mini graphs in subgraphs/.

The path is fixed by this code. The AI agents only fill in the content of each box, and a person
decides at the two critical points.
"""

import time
from datetime import datetime, timezone
from typing import Awaitable, Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from brief.agents.types import Agents
from brief.logic import LIMITS, build_extractive_draft, build_library, build_report
from brief.review import ReviewDecision, parse_draft_decision, parse_plan_decision
from brief.sources.types import SourceFetcher
from brief.state import ResearchState
from brief.subgraphs.drafting import create_drafting_subgraph, initial_draft_state
from brief.subgraphs.retrieval import create_retrieval_subgraph, initial_retrieval_state
from brief.tools.weather import WeatherTool
from brief.utils import log_step, since


def _message(error: BaseException) -> str:
    return str(error) or type(error).__name__


def create_research_graph(
    *,
    agents: Agents,
    fetchers: list[SourceFetcher],
    weather: WeatherTool,
    save_report: Callable[[str, str], Awaitable[str]],
    checkpointer: BaseCheckpointSaver,
):
    # Agents and sources are arguments, not imports, so tests can plug in scripted stand-ins.
    # A checkpointer is required: pausing for a person (interrupt) is impossible without saved state.
    research_subgraph = create_retrieval_subgraph(agents=agents, fetchers=fetchers)
    drafting_subgraph = create_drafting_subgraph(agents=agents)

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
            # here. (A mixed run rebuilds the library in the research subgraph, tool reading first.)
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

    # ---- Human-in-the-loop: the two critical points ----
    #
    # `interrupt(payload)` pauses the whole run here and hands `payload` to whoever is driving the
    # graph. When they resume with a value, `interrupt` returns it. IMPORTANT: on resume LangGraph
    # runs this node again from its first line, so a review node must do nothing before the
    # interrupt that would be wrong to repeat (no saving, no API calls) - which is why each
    # review is its own tiny node. A bad answer is shown back and asked again instead of crashing.

    def review_plan(state: ResearchState):
        error = ""
        while True:
            answer = interrupt(
                {
                    "kind": "plan_review",
                    "topic": state["topic"],
                    "sub_questions": state["sub_questions"],
                    "options": ["approve", "edit (with sub_questions)", "reject"],
                    "error": error,
                }
            )
            try:
                decision: ReviewDecision = parse_plan_decision(answer)
                break
            except ValueError as problem:
                error = str(problem)

        if decision.action == "reject":
            log_step("Plan review", "the reviewer rejected the plan - stopping, nothing was searched or saved")
            return {"cancelled": "The reviewer rejected the research plan, so nothing was searched or saved."}
        if decision.action == "edit":
            log_step("Plan review", f"the reviewer edited the plan ({len(decision.sub_questions)} sub-questions)")
            return {
                "sub_questions": list(decision.sub_questions),
                "run_notes": ["The reviewer edited the research plan before any searching happened."],
            }
        log_step("Plan review", "auto-approved (--auto)" if decision.auto else "the reviewer approved the plan")
        return {}

    def approve_draft(state: ResearchState):
        # With no notes (the summarizer failed) the draft is raw excerpts and there is nothing for
        # the writer to rewrite from, so only approve / reject make sense.
        revisions_left = LIMITS.max_human_revisions - state["human_revisions"] if state["notes"] else 0
        error = ""
        while True:
            answer = interrupt(
                {
                    "kind": "draft_review",
                    "topic": state["topic"],
                    "draft": state["draft"],
                    "sources": len(state["library"]),
                    "revisions_left": revisions_left,
                    "options": ["approve (save the report)", "revise (with feedback)", "reject (save nothing)"],
                    "error": error,
                }
            )
            try:
                decision = parse_draft_decision(answer, revisions_left=revisions_left)
                break
            except ValueError as problem:
                error = str(problem)

        if decision.action == "reject":
            log_step("Draft review", "the reviewer rejected the draft - stopping, nothing was saved")
            return {"cancelled": "The reviewer rejected the draft, so no report was saved."}
        if decision.action == "revise":
            log_step("Draft review", f"the reviewer asked for changes: {decision.feedback}")
            return {
                "review_feedback": [decision.feedback],
                "human_revisions": state["human_revisions"] + 1,
                "run_notes": [f"The reviewer asked for a revision: {decision.feedback}"],
            }
        note = (
            "Auto-approved (--auto): no person read this draft before it was saved."
            if decision.auto
            else "A human reviewer approved this brief before it was saved."
        )
        log_step("Draft review", "auto-approved (--auto)" if decision.auto else "the reviewer approved the draft")
        return {"run_notes": [note]}

    # ---- Subgraph wrappers: hand the subgraph its input, take back only its results ----
    # (Explicit mapping, not shared keys: a subgraph that shared a reducer field with this graph
    # would hand back the values it was given and they would be appended a second time.)

    async def research(state: ResearchState):
        result = await research_subgraph.ainvoke(
            initial_retrieval_state(state["topic"], state["sub_questions"], state["tool_docs"])
        )
        return {
            "library": result["library"],
            "retrieved": result["retrieved"],
            "retrieval_rounds": result["retrieval_rounds"],
            "run_notes": result["run_notes"],
        }

    async def drafting(state: ResearchState):
        result = await drafting_subgraph.ainvoke(
            initial_draft_state(
                topic=state["topic"],
                notes=state["notes"],
                library=state["library"],
                previous_draft=state["draft"],
                feedback=state["review_feedback"],
            )
        )
        return {
            "draft": result["draft"],
            "draft_count": result["draft_count"],
            "citation_problems": result["citation_problems"],
            "run_notes": result["run_notes"],
            "review_feedback": [],  # used up: the next revision gets only its own feedback
        }

    def no_sources(state: ResearchState):
        log_step("Coverage check", "no sources at all were retrieved - reporting that instead of guessing")
        return {
            "draft": "## Overview\n\nNo sources could be retrieved for this topic, so no brief could be written. "
            "See the notes at the end of this report for what went wrong."
        }

    # The retrieval work is the expensive part, so if the summarizing agent fails after all its
    # retries (free models time out) the run degrades instead of dying: the report still lists the
    # relevant excerpts that were found, unedited.
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

    def after_plan_review(state: ResearchState):
        return END if state["cancelled"] else "research"

    def after_research(state: ResearchState):
        return "summarize" if state["library"] else "no_sources"

    # No notes means the summarizer failed and already set a fallback draft: skip the writer.
    def after_summarize(state: ResearchState):
        return "drafting" if state["notes"] else "approve_draft"

    def after_draft_review(state: ResearchState):
        if state["cancelled"]:
            return END
        return "drafting" if state["review_feedback"] else "save"

    graph = StateGraph(ResearchState)
    graph.add_node("router", router)
    graph.add_node("weather_tool", weather_tool)
    graph.add_node("unsupported", unsupported)
    graph.add_node("plan", plan)
    graph.add_node("review_plan", review_plan)
    graph.add_node("research", research)  # subgraph 1
    graph.add_node("no_sources", no_sources)
    graph.add_node("summarize", summarize)
    graph.add_node("drafting", drafting)  # subgraph 2
    graph.add_node("approve_draft", approve_draft)
    graph.add_node("save", save)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", after_route, ["weather_tool", "unsupported", "plan"])
    graph.add_conditional_edges("weather_tool", after_weather, ["summarize", "plan"])
    graph.add_edge("unsupported", "save")
    graph.add_edge("plan", "review_plan")
    graph.add_conditional_edges("review_plan", after_plan_review, ["research", END])
    graph.add_conditional_edges("research", after_research, ["summarize", "no_sources"])
    graph.add_edge("no_sources", "save")
    graph.add_conditional_edges("summarize", after_summarize, ["drafting", "approve_draft"])
    graph.add_edge("drafting", "approve_draft")
    graph.add_conditional_edges("approve_draft", after_draft_review, ["drafting", "save", END])
    graph.add_edge("save", END)
    return graph.compile(checkpointer=checkpointer)
