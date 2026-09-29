"""Human-in-the-loop: the graph pauses at two critical points and does nothing irreversible
(searching for minutes, saving a report) until a person says so."""

import asyncio

from brief.logic import LIMITS
from brief.types import RouteDecision
from tests.helpers import Harness, make_agents, make_fetchers, raiser, run_with_pauses


def test_the_run_pauses_for_plan_review_before_any_searching_happens():
    async def go():
        agents, calls = make_agents()
        harness = Harness(agents)
        payload = await harness.start()

        assert payload["kind"] == "plan_review"
        assert payload["sub_questions"] == ["question A", "question B"]
        assert calls["craft_queries"] == []  # nothing searched yet
        assert harness.saved == []
        assert await harness.next_nodes() == ("review_plan",)

    asyncio.run(go())


def test_nothing_is_saved_until_the_draft_is_approved():
    async def go():
        agents, calls = make_agents()
        harness = Harness(agents)
        await harness.start()

        draft_review = await harness.answer({"action": "approve"})  # approve the plan
        assert draft_review["kind"] == "draft_review"
        assert "A claim [1]" in draft_review["draft"]
        assert draft_review["sources"] == 6
        assert draft_review["revisions_left"] == LIMITS.max_human_revisions
        assert harness.saved == []  # the report is written, but not saved: the human has not approved it

        assert await harness.answer({"action": "approve"}) is None
        assert len(harness.saved) == 1
        assert "A human reviewer approved this brief before it was saved." in harness.saved[0]

    asyncio.run(go())


def test_editing_the_plan_changes_what_is_researched():
    async def go():
        agents, calls = make_agents()
        harness = Harness(agents)
        await harness.start()
        await harness.answer({"action": "edit", "sub_questions": ["a brand new question", "and one more thing"]})

        assert sorted(c["sub_question"] for c in calls["craft_queries"]) == ["a brand new question", "and one more thing"]
        await harness.answer({"action": "approve"})
        assert "The reviewer edited the research plan" in harness.saved[0]

    asyncio.run(go())


def test_rejecting_the_plan_stops_the_run_with_nothing_searched_or_saved():
    async def go():
        agents, calls = make_agents()
        harness = Harness(agents)
        await harness.start()

        assert await harness.answer({"action": "reject"}) is None
        values = await harness.values()
        assert "rejected the research plan" in values["cancelled"]
        assert calls["craft_queries"] == [] and harness.saved == []
        assert await harness.next_nodes() == ()  # the run is over

    asyncio.run(go())


def test_a_bad_answer_is_shown_back_and_asked_again_instead_of_crashing():
    async def go():
        agents, _ = make_agents()
        harness = Harness(agents)
        await harness.start()

        again = await harness.answer("banana")
        assert again["kind"] == "plan_review" and "Unknown action" in again["error"]

        again = await harness.answer({"action": "edit", "sub_questions": []})
        assert again["kind"] == "plan_review" and "at least one" in again["error"]

        nxt = await harness.answer({"action": "approve"})  # a good answer finally moves the run on
        assert nxt["kind"] == "draft_review" and nxt["error"] == ""

    asyncio.run(go())


def test_asking_for_a_revision_sends_the_feedback_to_the_writer_and_writes_a_new_draft():
    async def go():
        agents, calls = make_agents()
        harness = Harness(agents)
        await harness.start()
        await harness.answer({"action": "approve"})

        again = await harness.answer({"action": "revise", "feedback": "make it shorter please"})
        assert again["kind"] == "draft_review"
        assert again["revisions_left"] == LIMITS.max_human_revisions - 1
        assert calls["write"] == 2 and harness.saved == []  # rewritten, still not saved

        second_call = calls["write_problems"][1]
        assert "Reviewer feedback (must be addressed): make it shorter please" in second_call["problems"]
        assert "A claim [1]" in second_call["previous_draft"]  # the writer is shown what it wrote before
        assert calls["write_problems"][0]["problems"] == []  # the first draft had no feedback

        await harness.answer({"action": "approve"})
        assert len(harness.saved) == 1
        assert "The reviewer asked for a revision: make it shorter please" in harness.saved[0]
        assert (await harness.values())["human_revisions"] == 1

    asyncio.run(go())


def test_revisions_are_capped_and_the_reviewer_is_told_so():
    async def go():
        agents, _ = make_agents()
        harness = Harness(agents)
        await harness.start()
        await harness.answer({"action": "approve"})
        for i in range(LIMITS.max_human_revisions):
            await harness.answer({"action": "revise", "feedback": f"change number {i} please"})

        blocked = await harness.answer({"action": "revise", "feedback": "one more change please"})
        assert blocked["revisions_left"] == 0
        assert "No revisions are left" in blocked["error"]  # asked again, the run is still paused

        assert await harness.answer({"action": "reject"}) is None
        assert harness.saved == []

    asyncio.run(go())


def test_rejecting_the_draft_saves_nothing():
    _, saved, pauses = run_with_pauses(make_agents()[0], decisions=[{"action": "approve"}, {"action": "reject"}])
    assert saved == []
    assert [p["kind"] for p in pauses] == ["plan_review", "draft_review"]


def test_auto_approval_is_recorded_so_nobody_mistakes_it_for_human_review():
    values, saved, _ = run_with_pauses(
        make_agents()[0], decisions=[{"action": "approve", "auto": True}, {"action": "approve", "auto": True}]
    )
    assert "Auto-approved (--auto): no person read this draft" in saved[0]
    assert "A human reviewer approved" not in saved[0]


def test_weather_only_has_no_plan_to_review_only_the_draft():
    async def route(topic):
        return RouteDecision("weather", city="Testville")

    agents, calls = make_agents(route=route)
    _, saved, pauses = run_with_pauses(agents)
    assert [p["kind"] for p in pauses] == ["draft_review"]
    assert calls["plan"] == 0 and len(saved) == 1


def test_runs_with_nothing_written_by_the_ai_do_not_pause_at_all():
    async def route(topic):
        return RouteDecision("unsupported")

    _, saved, pauses = run_with_pauses(make_agents(route=route)[0])
    assert pauses == [] and "needs live, up-to-the-minute data" in saved[0]

    _, saved, pauses = run_with_pauses(make_agents()[0], make_fetchers(lambda kind, query: []), decisions=[{"action": "approve"}])
    assert [p["kind"] for p in pauses] == ["plan_review"]  # the plan is reviewed, but no draft is ever offered
    assert "No sources could be retrieved" in saved[0]


def test_when_the_summarizer_failed_there_is_nothing_to_rewrite_so_revisions_are_off():
    async def go():
        agents, _ = make_agents(summarize=raiser("model unavailable"))
        harness = Harness(agents)
        await harness.start()
        review = await harness.answer({"action": "approve"})

        assert review["kind"] == "draft_review" and "## Retrieved excerpts" in review["draft"]
        assert review["revisions_left"] == 0
        blocked = await harness.answer({"action": "revise", "feedback": "please rewrite it"})
        assert "No revisions are left" in blocked["error"]

    asyncio.run(go())
