"""Runs the real graph with scripted stand-ins for the AI agents and the web sources, so the
routing (parallel fan-out, retries, redrafts, giving up) is checked exactly, every time, without
depending on a live model or network."""

import asyncio
import re

from brief.agents.types import Agents
from brief.graph import create_research_graph
from brief.logic import LIMITS
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
    calls = {"plan": 0, "craft_queries": [], "summarize": 0, "write": 0}

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


def run(agents, fetchers, weather=working_weather):
    saved: list[str] = []

    async def save_report(topic, markdown):
        saved.append(markdown)
        return "memory://report"

    graph = create_research_graph(agents=agents, fetchers=fetchers, weather=weather, save_report=save_report)
    result = asyncio.run(graph.ainvoke(initial_state("test topic")))
    return result, saved


def raiser(message):
    async def fail(*args, **kwargs):
        raise RuntimeError(message)

    return fail


def test_happy_path_plan_parallel_retrieval_summarize_write_save_no_retries():
    agents, calls = make_agents()
    result, saved = run(agents, make_fetchers())

    assert calls["plan"] == 1
    assert len(calls["craft_queries"]) == 2  # one parallel branch per sub-question
    assert result["retrieval_rounds"] == 1
    assert calls["summarize"] == 1
    assert result["draft_count"] == 1
    assert len(saved) == 1
    assert len(result["library"]) == 6  # 2 sub-questions x 3 sources retrieved...
    assert len(re.findall(r"^\d+\. \*\*", saved[0], re.M)) == 2  # ...but only the 2 cited ones are listed
    assert "4 retrieved sources were not used in the brief" in saved[0]


def test_retry_a_thin_sub_question_is_searched_again_with_rewritten_queries():
    # Question A's first attempt finds nothing anywhere; the reworded second attempt succeeds.
    def handler(kind, query):
        if query == "question A#0":
            return []
        return [SourceDocument(kind=kind, title=query, url=f"https://example.com/{kind}/{query}", text="found")]

    agents, calls = make_agents()
    result, saved = run(agents, make_fetchers(handler))

    assert result["retrieval_rounds"] == 2
    assert len(calls["craft_queries"]) == 3  # A, B, then A again
    retry = calls["craft_queries"][2]
    assert retry["sub_question"] == "question A"
    assert len(retry["previous_queries"]) == 1  # the agent is told what already failed
    assert re.search(r"Round 1: nothing relevant yet for .*so the queries were rewritten and searched again", saved[0])


def test_found_but_off_topic_does_not_count_as_coverage_the_grader_drops_it_and_the_retry_fires():
    # The live bug: search engines return something for any query. Question A's first queries
    # return only off-topic results; the reworded second attempt returns real ones.
    def handler(kind, query):
        title = "junk result" if query.startswith("question A#0") else f"real {query}"
        return [SourceDocument(kind=kind, title=title, url=f"https://example.com/{kind}/{query}", text="text")]

    async def grade(*, topic, sub_question, documents):
        return [d for d in documents if "junk" not in d.title]

    agents, calls = make_agents(grade_relevance=grade)
    result, saved = run(agents, make_fetchers(handler))

    assert result["retrieval_rounds"] == 2
    assert len(calls["craft_queries"]) == 3
    assert all("junk" not in s.title for s in result["library"])  # junk never reaches the report
    assert re.search(r"Round 1: nothing relevant yet for .*so the queries were rewritten and searched again", saved[0])


def test_if_the_relevance_grader_itself_fails_results_are_kept_and_the_report_says_they_were_unchecked():
    agents, _ = make_agents(grade_relevance=raiser("model timed out"))
    result, saved = run(agents, make_fetchers())

    assert len(result["library"]) == 6  # nothing was thrown away
    assert 'Relevance checking failed for "question A" (model timed out)' in saved[0]


def test_retry_is_capped_then_the_run_continues_with_what_it_has():
    # Question A never yields anything; B is fine.
    def handler(kind, query):
        if query.startswith("question A"):
            return []
        return [SourceDocument(kind=kind, title=query, url=f"https://example.com/{kind}/{query}", text="found")]

    agents, calls = make_agents()
    result, saved = run(agents, make_fetchers(handler))

    assert result["retrieval_rounds"] == LIMITS.max_retrieval_rounds
    assert len([c for c in calls["craft_queries"] if c["sub_question"] == "question A"]) == LIMITS.max_retrieval_rounds
    assert calls["summarize"] == 1  # still summarizes what was found for B
    assert "Retrieval gave up after 3 rounds" in saved[0]


def test_nothing_found_at_all_reports_that_honestly_instead_of_asking_the_ai_to_write_anything():
    agents, calls = make_agents()
    result, saved = run(agents, make_fetchers(lambda kind, query: []))

    assert calls["summarize"] == 0
    assert calls["write"] == 0
    assert result["retrieval_rounds"] == LIMITS.max_retrieval_rounds
    assert "No sources could be retrieved" in saved[0]
    assert "No sources are cited" in saved[0]


def test_citation_guard_an_invented_citation_sends_the_draft_back_to_the_writer_with_the_reason():
    seen_problems: list[list[str]] = []

    async def write(*, topic, notes, library, problems, previous_draft):
        seen_problems.append(problems)
        return "Made-up source [99] and [1]." if len(seen_problems) == 1 else "Real sources [1] and [2]."

    agents, _ = make_agents(write=write)
    result, saved = run(agents, make_fetchers())

    assert result["draft_count"] == 2
    assert seen_problems[0] == []
    assert "[99]" in seen_problems[1][0]  # the second draft was told exactly what was wrong
    assert "Warning" not in saved[0]
    assert "Real sources [1] and [2]" in saved[0]


def test_citation_guard_a_writer_that_never_gets_it_right_is_stopped_and_the_report_says_so():
    writes = 0

    async def write(*, topic, notes, library, problems, previous_draft):
        nonlocal writes
        writes += 1
        return "Always invented [99]."

    agents, _ = make_agents(write=write)
    result, saved = run(agents, make_fetchers())

    assert writes == LIMITS.max_drafts
    assert result["draft_count"] == LIMITS.max_drafts
    assert "Warning: the writer could not fix all citation problems" in saved[0]


def test_if_the_writer_fails_after_retrieval_succeeded_the_report_still_lists_the_excerpts_found():
    agents, _ = make_agents(write=raiser("Request timed out."))
    result, saved = run(agents, make_fetchers())

    assert len(result["citation_problems"]) == 0  # the fallback draft cites every source, so it passes
    assert "## Retrieved excerpts" in saved[0]
    assert "The answer-writing agent failed (Request timed out.)" in saved[0]
    assert len(re.findall(r"^\d+\. \*\*", saved[0], re.M)) == 6  # all six sources are listed


def test_if_the_summarizer_fails_the_writer_is_skipped_and_the_excerpts_are_saved_instead():
    agents, calls = make_agents(summarize=raiser("model unavailable"))
    _, saved = run(agents, make_fetchers())

    assert calls["write"] == 0  # no point asking a model that just failed to write
    assert "## Retrieved excerpts" in saved[0]
    assert "The summarization agent failed (model unavailable)" in saved[0]


def test_one_source_being_down_does_not_stop_the_others_and_the_report_says_which_failed():
    fetchers = make_fetchers()
    fetchers[2] = SourceFetcher(kind="hackernews", label="Hacker News", fetch=raiser("HTTP 503 from hn.algolia.com"))
    agents, _ = make_agents()
    result, saved = run(agents, fetchers)

    assert len(result["library"]) == 4  # 2 sub-questions x 2 working sources
    assert 'Hacker News was unavailable for "question A": HTTP 503' in saved[0]
    assert 'Hacker News was unavailable for "question B": HTTP 503' in saved[0]


def test_if_the_retrieval_agents_query_writing_fails_the_question_text_is_used_as_the_query():
    queries_seen: list[str] = []

    def handler(kind, query):
        queries_seen.append(query)
        return [SourceDocument(kind=kind, title=query, url=f"https://example.com/{kind}/{query}", text="found")]

    agents, calls = make_agents(craft_queries=raiser("model timed out"))
    _, saved = run(agents, make_fetchers(handler))

    assert calls["summarize"] == 1
    assert "question A" in queries_seen
    assert 'could not write search queries for "question A" (model timed out)' in saved[0]


# ---------- The router and the weather tool ----------


def test_weather_only_the_tool_answers_and_no_planning_or_searching_happens():
    searched: list[str] = []

    def handler(kind, query):
        searched.append(query)
        return []

    async def route(topic):
        return RouteDecision("weather", city="Testville")

    async def summarize(*, topic, sub_questions, library):
        assert sub_questions == ["test topic"]
        assert len(library) == 1
        return "Temperature is 21 °C [1]."

    async def write(*, topic, notes, library, problems, previous_draft):
        return "## Overview\n\nIt is 21 °C [1]."

    agents, calls = make_agents(route=route, summarize=summarize, write=write)
    result, saved = run(agents, make_fetchers(handler))

    assert calls["plan"] == 0
    assert calls["craft_queries"] == []
    assert searched == []
    assert result["library"][0].kind == "weather"
    assert "1. **Open-Meteo weather**: [Current weather in Testville](https://example.com/weather)" in saved[0]
    assert "Warning" not in saved[0]  # a single source is fine for a single reading


def test_mixed_the_weather_reading_and_the_research_sources_end_up_in_one_library_weather_first():
    planned_for: list[str] = []
    summarized_questions: list[str] = []

    async def route(topic):
        return RouteDecision("mixed", city="Testville", research_topic="why monsoons happen")

    async def plan(topic):
        planned_for.append(topic)
        return ["question A"]

    async def summarize(*, topic, sub_questions, library):
        summarized_questions.extend(sub_questions)
        return "notes [1][2]"

    agents, _ = make_agents(route=route, plan=plan, summarize=summarize)
    result, _ = run(agents, make_fetchers())

    assert planned_for == ["why monsoons happen"]  # only the non-weather part is researched
    assert result["library"][0].kind == "weather"
    assert result["library"][0].id == 1
    assert len(result["library"]) == 4  # 1 weather + 3 searched sources
    # the summarizer is explicitly asked about the weather, or it would leave the reading out
    assert summarized_questions == ["What is the current weather in Testville?", "question A"]


def test_weather_tool_failure_falls_back_to_research_and_the_report_says_there_is_no_live_data():
    async def route(topic):
        return RouteDecision("weather", city="Nowhereville")

    agents, calls = make_agents(route=route)
    result, saved = run(agents, make_fetchers(), WeatherTool(lookup=raiser('No place called "Nowhereville" was found.')))

    assert calls["plan"] == 1
    assert all(s.kind != "weather" for s in result["library"])
    assert 'weather tool could not get the current weather for "Nowhereville"' in saved[0]
    assert "contains no live weather data" in saved[0]


def test_unsupported_live_data_an_honest_report_with_no_planning_searching_or_writing():
    async def route(topic):
        return RouteDecision("unsupported")

    agents, calls = make_agents(route=route)
    _, saved = run(agents, make_fetchers())

    assert calls["plan"] == 0
    assert calls["craft_queries"] == []
    assert calls["write"] == 0
    assert "needs live, up-to-the-minute data" in saved[0]
    assert "No sources are cited" in saved[0]


def test_if_the_routing_agent_fails_the_topic_is_researched_as_usual_and_the_report_says_so():
    agents, calls = make_agents(route=raiser("model timed out"))
    _, saved = run(agents, make_fetchers())

    assert calls["plan"] == 1
    assert "routing agent failed (model timed out)" in saved[0]
