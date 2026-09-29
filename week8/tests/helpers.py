"""Shared test doubles: scripted agents, fake sources, and a Harness that drives a graph the way a
person (or the CLI) would - start it, look at what it is paused on, answer, repeat."""

import asyncio

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from brief.agents.types import Agents
from brief.graph import create_research_graph
from brief.persistence import make_serde
from brief.sources.types import SourceDocument, SourceFetcher
from brief.state import initial_state
from brief.tools.weather import WeatherTool
from brief.types import QueryPlan, RouteDecision

KINDS = ["wikipedia", "arxiv", "hackernews"]


def make_fetchers(handler=None):
    """By default every source returns one document whose URL depends on the query it was given."""

    def default(kind, query):
        return [SourceDocument(kind=kind, title=f"{kind}: {query}", url=f"https://example.com/{kind}/{query}", text=f"about {query}")]

    def build(kind):
        async def fetch(query):
            return (handler or default)(kind, query)

        return SourceFetcher(kind=kind, label=kind, fetch=fetch)

    return [build(kind) for kind in KINDS]


def make_agents(**overrides):
    calls = {"plan": 0, "craft_queries": [], "summarize": 0, "write": 0, "write_problems": []}

    async def route(topic):
        return RouteDecision("research")  # by default every topic is plain research

    async def plan(topic):
        calls["plan"] += 1
        return ["question A", "question B"]

    async def craft_queries(*, topic, sub_question, previous_queries):
        # The query encodes the sub-question and how many earlier attempts there were: "question A#1".
        calls["craft_queries"].append({"sub_question": sub_question, "previous_queries": previous_queries})
        query = f"{sub_question}#{len(previous_queries)}"
        return QueryPlan(wikipedia=query, arxiv=query, hackernews=query)

    async def grade_relevance(*, topic, sub_question, documents):
        return documents  # by default everything is judged relevant

    async def summarize(*, topic, sub_questions, library):
        calls["summarize"] += 1
        return "notes [1][2]"

    async def write(*, topic, notes, library, problems, previous_draft):
        calls["write"] += 1
        calls["write_problems"].append({"problems": problems, "previous_draft": previous_draft})
        return "## Overview\n\nA claim [1] and another [2]."

    defaults = dict(
        route=route, plan=plan, craft_queries=craft_queries, grade_relevance=grade_relevance, summarize=summarize, write=write
    )
    return Agents(**{**defaults, **overrides}), calls


WEATHER_DOC = SourceDocument(
    kind="weather", title="Current weather in Testville", url="https://example.com/weather", text="Temperature: 21 °C."
)


async def _working_weather(city):
    return WEATHER_DOC


working_weather = WeatherTool(lookup=_working_weather)


def raiser(message):
    async def fail(*args, **kwargs):
        raise RuntimeError(message)

    return fail


class Harness:
    """One graph + one thread. Several Harnesses sharing a `saver` and `thread` model a restart:
    each builds a brand-new graph object, and only the checkpoints carry the run across."""

    def __init__(self, agents, fetchers=None, weather=working_weather, saver=None, thread="t", saved=None):
        self.saved: list[str] = saved if saved is not None else []
        self.saver = saver or InMemorySaver(serde=make_serde())
        self.config = {"configurable": {"thread_id": thread}}

        async def save_report(topic, markdown):
            self.saved.append(markdown)
            return "memory://report"

        self.graph = create_research_graph(
            agents=agents, fetchers=fetchers or make_fetchers(), weather=weather, save_report=save_report, checkpointer=self.saver
        )

    async def pause(self):
        """What the run is waiting on (the interrupt payload), or None if it is not waiting for a person."""
        snapshot = await self.graph.aget_state(self.config)
        payloads = [pending.value for task in snapshot.tasks for pending in task.interrupts]
        return payloads[0] if payloads else None

    async def start(self, topic="test topic"):
        await self.graph.ainvoke(initial_state(topic), self.config)
        return await self.pause()

    async def answer(self, value):
        await self.graph.ainvoke(Command(resume=value), self.config)
        return await self.pause()

    async def values(self):
        return (await self.graph.aget_state(self.config)).values

    async def next_nodes(self):
        return (await self.graph.aget_state(self.config)).next


def run_with_pauses(agents, fetchers=None, weather=working_weather, decisions=None):
    """Runs to the end, playing the reviewer: each pause is answered from `decisions`, else approved."""
    answers = list(decisions or [])

    async def go():
        harness = Harness(agents, fetchers, weather)
        pauses = []
        payload = await harness.start()
        while payload is not None:
            pauses.append(payload)
            payload = await harness.answer(answers.pop(0) if answers else {"action": "approve"})
        return await harness.values(), harness.saved, pauses

    return asyncio.run(go())


def run(agents, fetchers=None, weather=working_weather, decisions=None):
    values, saved, _ = run_with_pauses(agents, fetchers, weather, decisions)
    return values, saved
