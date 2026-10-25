"""Persistence: the state is checkpointed after every step, so a run survives a pause, a new
process, and even a crash - and continues from its last step instead of starting over."""

import asyncio

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from brief.persistence import list_threads, make_serde, open_checkpointer
from tests.helpers import Harness, make_agents


def test_a_brand_new_graph_object_picks_a_paused_run_up_where_it_stopped():
    async def go():
        agents, calls = make_agents()
        saver = InMemorySaver(serde=make_serde())

        first = Harness(agents, saver=saver)
        assert (await first.start())["kind"] == "plan_review"

        # "Restart": a different graph object, sharing nothing with the first except the checkpoints.
        second = Harness(agents, saver=saver, saved=first.saved)
        assert (await second.pause())["kind"] == "plan_review"  # it finds the pause on its own
        assert (await second.answer({"action": "approve"}))["kind"] == "draft_review"
        await second.answer({"action": "approve"})

        assert calls["plan"] == 1  # the work done before the pause was not redone
        assert len(first.saved) == 1

    asyncio.run(go())


def test_sqlite_checkpoints_survive_closing_and_reopening_the_database(tmp_path):
    db = tmp_path / "checkpoints.sqlite"

    async def go():
        agents, calls = make_agents()
        saved: list[str] = []

        async with open_checkpointer(db) as saver:  # process 1: run until the plan review, then "exit"
            assert (await Harness(agents, saver=saver, saved=saved).start())["kind"] == "plan_review"

        assert await list_threads(db) == ["t"]

        async with open_checkpointer(db) as saver:  # process 2: a fresh connection resumes it
            harness = Harness(agents, saver=saver, saved=saved)
            values = await harness.values()
            assert values["topic"] == "test topic" and values["sub_questions"] == ["question A", "question B"]

            assert (await harness.answer({"action": "approve"}))["kind"] == "draft_review"

        async with open_checkpointer(db) as saver:  # process 3: and another one finishes it
            harness = Harness(agents, saver=saver, saved=saved)
            # the pydantic objects inside the state came back as real objects, not dicts
            library = (await harness.values())["library"]
            assert library[0].id == 1 and library[0].kind == "wikipedia"
            assert await harness.answer({"action": "approve"}) is None

        assert len(saved) == 1 and calls["plan"] == 1 and len(calls["craft_queries"]) == 2

    asyncio.run(go())


def test_threads_are_independent_runs():
    async def go():
        agents, _ = make_agents()
        saver = InMemorySaver(serde=make_serde())
        a = Harness(agents, saver=saver, thread="a")
        b = Harness(agents, saver=saver, thread="b")
        await a.start("topic for A")
        await b.start("topic for B")
        await a.answer({"action": "reject"})  # stopping A does not touch B

        assert (await a.values())["cancelled"] and not (await b.values())["cancelled"]
        assert (await a.values())["topic"] == "topic for A"
        assert (await b.pause())["kind"] == "plan_review"

    asyncio.run(go())


class Crash(BaseException):
    """Stands in for a killed process / Ctrl+C: nothing in the graph catches it."""


def test_a_crash_resumes_from_the_last_checkpoint_without_redoing_finished_work():
    async def go():
        attempts = {"summarize": 0}

        async def summarize(*, topic, sub_questions, library):
            attempts["summarize"] += 1
            if attempts["summarize"] == 1:
                raise Crash()
            return "notes [1][2]"

        agents, calls = make_agents(summarize=summarize)
        harness = Harness(agents)
        await harness.start()
        with pytest.raises(Crash):
            await harness.answer({"action": "approve"})  # searching finishes, then the summarizer "dies"

        assert len(calls["craft_queries"]) == 2
        values = await harness.values()
        assert len(values["library"]) == 6  # the retrieval results were checkpointed before the crash
        assert await harness.next_nodes() == ("summarize",)  # exactly where it would continue

        await harness.graph.ainvoke(None, harness.config)  # "resume": no new input, continue from the checkpoint
        assert (await harness.pause())["kind"] == "draft_review"
        assert len(calls["craft_queries"]) == 2 and calls["plan"] == 1  # nothing before the crash was redone
        assert attempts["summarize"] == 2

    asyncio.run(go())


def test_the_checkpoint_history_records_every_step():
    async def go():
        agents, _ = make_agents()
        harness = Harness(agents)
        await harness.start()
        history = [s async for s in harness.graph.aget_state_history(harness.config)]
        assert len(history) >= 4  # input, router, plan, the pause at the review...
        assert history[0].next == ("review_plan",)  # newest first

    asyncio.run(go())
