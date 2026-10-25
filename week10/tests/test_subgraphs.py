"""The two subgraphs work on their own (that is the point of a subgraph), and hand back only their
own results to the parent."""

import asyncio

from brief.logic import LIMITS, build_library
from brief.sources.types import SourceDocument
from brief.subgraphs.drafting import create_drafting_subgraph, initial_draft_state
from brief.subgraphs.retrieval import create_retrieval_subgraph, initial_retrieval_state
from tests.helpers import make_agents, make_fetchers, raiser, run


def retrieval(agents, fetchers, questions=("question A", "question B"), tool_docs=()):
    graph = create_retrieval_subgraph(agents=agents, fetchers=fetchers)
    return asyncio.run(graph.ainvoke(initial_retrieval_state("test topic", list(questions), list(tool_docs))))


def test_the_retrieval_subgraph_runs_by_itself():
    agents, calls = make_agents()
    result = retrieval(agents, make_fetchers())

    assert result["retrieval_rounds"] == 1
    assert len(result["library"]) == 6 and len(calls["craft_queries"]) == 2
    assert [s.id for s in result["library"]] == [1, 2, 3, 4, 5, 6]


def test_the_retrieval_subgraph_retries_and_gives_up_on_its_own():
    def handler(kind, query):
        if query.startswith("question A"):
            return []  # A never yields anything
        return [SourceDocument(kind=kind, title=query, url=f"https://example.com/{kind}/{query}", text="found")]

    agents, calls = make_agents()
    result = retrieval(agents, make_fetchers(handler))

    assert result["retrieval_rounds"] == LIMITS.max_retrieval_rounds
    assert len([c for c in calls["craft_queries"] if c["sub_question"] == "question A"]) == LIMITS.max_retrieval_rounds
    assert any("Retrieval gave up after 3 rounds" in note for note in result["run_notes"])


def test_the_retrieval_subgraph_numbers_a_live_reading_first():
    from tests.helpers import WEATHER_DOC

    result = retrieval(make_agents()[0], make_fetchers(), questions=("question A",), tool_docs=[WEATHER_DOC])
    assert result["library"][0].kind == "weather" and result["library"][0].id == 1
    assert len(result["library"]) == 4


def draft(agents, **kwargs):
    graph = create_drafting_subgraph(agents=agents)
    library = build_library([], [SourceDocument(kind="weather", title="w", url="https://e/w", text="t")] * 1)
    inputs = dict(topic="test topic", notes="notes", library=library)
    return asyncio.run(graph.ainvoke(initial_draft_state(**{**inputs, **kwargs})))


def test_the_drafting_subgraph_hands_reviewer_feedback_to_the_writer():
    agents, calls = make_agents()

    async def write(*, topic, notes, library, problems, previous_draft):
        calls["write_problems"].append({"problems": problems, "previous_draft": previous_draft})
        return "A fact [1]."

    agents, calls = make_agents(write=write)
    result = draft(agents, previous_draft="the old draft", feedback=["add the risks"])

    assert calls["write_problems"][0] == {
        "problems": ["Reviewer feedback (must be addressed): add the risks"],
        "previous_draft": "the old draft",
    }
    assert result["draft"] == "A fact [1]." and result["draft_count"] == 1 and result["citation_problems"] == []


def test_the_drafting_subgraph_redrafts_a_bad_citation_and_stops_after_the_limit():
    async def always_wrong(*, topic, notes, library, problems, previous_draft):
        return "Invented [99]."

    result = draft(make_agents(write=always_wrong)[0])
    assert result["draft_count"] == LIMITS.max_drafts
    assert "[99]" in result["citation_problems"][0]


def test_the_drafting_subgraph_falls_back_to_the_excerpts_if_the_writer_fails():
    result = draft(make_agents(write=raiser("Request timed out."))[0])
    assert "## Retrieved excerpts" in result["draft"]
    assert "The answer-writing agent failed (Request timed out.)" in result["run_notes"][0]


def test_a_subgraphs_notes_reach_the_report_exactly_once_not_twice():
    # Shared reducer fields between a parent and a subgraph would append the same notes twice.
    fetchers = make_fetchers()
    fetchers[2] = type(fetchers[2])(kind="hackernews", label="Hacker News", fetch=raiser("HTTP 503 from hn.algolia.com"))
    _, saved = run(make_agents()[0], fetchers)

    assert saved[0].count('Hacker News was unavailable for "question A"') == 1
    assert saved[0].count('Hacker News was unavailable for "question B"') == 1
