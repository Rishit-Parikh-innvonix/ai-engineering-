"""Week 8 - the week 7 research assistant, now with checkpoints, human approval and subgraphs.

  python -m brief "topic"                       start a run (asks you at the two review points)
  python -m brief "topic" --auto                approve everything automatically (no human)
  python -m brief "topic" --no-prompt           run until the first review point, then stop and exit
  python -m brief resume THREAD [--approve | --edit "q1 | q2" | --feedback "..." | --reject]
  python -m brief status THREAD                 where is this run, and what is it waiting for?
  python -m brief threads                       saved runs

Every step is checkpointed to a SQLite file, so a run that was paused for review, stopped with
Ctrl+C, timed out or crashed can be resumed with `resume THREAD` from its last saved step - even
from a new process.
"""

import argparse
import asyncio
import sys
import textwrap
import uuid
from typing import Any

from langgraph.types import Command

from brief.agents import create_agents
from brief.config import CHECKPOINT_DB
from brief.graph import create_research_graph
from brief.models import get_cloud_chat_model
from brief.persistence import list_threads, open_checkpointer
from brief.report import save_report_to_disk
from brief.sources import DEFAULT_FETCHERS
from brief.state import initial_state
from brief.tools.weather import open_meteo_weather
from brief.utils import print_section_header

# One stretch of running (up to the next pause) makes a dozen or so model calls plus web requests,
# and free-tier latency swings widely. This ceiling exists so a stalled model or network can never
# leave the terminal hanging; progress is printed throughout, and the state is checkpointed, so a
# timed-out run is resumed, not lost.
DEFAULT_TIMEOUT_S = 8 * 60


def build_graph(checkpointer):
    return create_research_graph(
        agents=create_agents(get_cloud_chat_model()),
        fetchers=DEFAULT_FETCHERS,
        weather=open_meteo_weather,
        save_report=save_report_to_disk,
        checkpointer=checkpointer,
    )


async def pending_review(graph, config) -> dict | None:
    """The payload of the interrupt the run is paused on, or None if it is not waiting for a person."""
    snapshot = await graph.aget_state(config)
    for task in snapshot.tasks:
        for pending in task.interrupts:
            return pending.value
    return None


# ---------- talking to the human ----------


def show_review(payload: dict) -> None:
    if payload.get("error"):
        print(f"\n  ! That answer was not accepted: {payload['error']}")
    if payload["kind"] == "plan_review":
        print_section_header("REVIEW THE PLAN before any searching happens", "-")
        for i, question in enumerate(payload["sub_questions"], start=1):
            print(f"  {i}. {question}")
    else:
        print_section_header("REVIEW THE DRAFT before it is saved", "-")
        draft = payload["draft"].strip()
        print(textwrap.indent(draft if len(draft) < 2500 else draft[:2500] + "\n[...draft continues...]", "  "))
        print(f"\n  ({payload['sources']} sources retrieved; {payload['revisions_left']} revision(s) left)")


def ask_human(payload: dict) -> Any:
    """Interactive prompt used when a person is at the keyboard."""
    show_review(payload)
    if payload["kind"] == "plan_review":
        while True:
            choice = input("\n[a]pprove, [e]dit the questions, or [r]eject? ").strip().lower()[:1]
            if choice == "a":
                return {"action": "approve"}
            if choice == "r":
                return {"action": "reject"}
            if choice == "e":
                text = input("New sub-questions, separated by |  : ")
                return {"action": "edit", "sub_questions": [q for q in text.split("|")]}
    while True:
        choice = input("\n[a]pprove and save, [v] request a revision, or [r]eject? ").strip().lower()[:1]
        if choice == "a":
            return {"action": "approve"}
        if choice == "r":
            return {"action": "reject"}
        if choice == "v":
            return {"action": "revise", "feedback": input("What should change? ")}


def decision_from_flags(args: argparse.Namespace) -> Any | None:
    """The answer given on the command line to `resume`, if any."""
    if args.approve:
        return {"action": "approve"}
    if args.reject:
        return {"action": "reject"}
    if args.edit is not None:
        return {"action": "edit", "sub_questions": args.edit.split("|")}
    if args.feedback is not None:
        return {"action": "revise", "feedback": args.feedback}
    return None


def print_pause_instructions(thread: str, payload: dict) -> None:
    show_review(payload)
    base = f"python -m brief resume {thread}"
    print("\nThe run is paused and saved. Answer it later with one of:")
    if payload["kind"] == "plan_review":
        print(f"  {base} --approve\n  {base} --edit \"question one | question two\"\n  {base} --reject")
    else:
        print(f"  {base} --approve\n  {base} --feedback \"what to change\"\n  {base} --reject")


# ---------- driving the graph ----------


async def drive(graph, thread: str, first_input: Any, *, auto: bool, interactive: bool, timeout: float) -> int:
    """Runs until the graph finishes, pauses for a person nobody is there to answer, or is stopped."""
    config = {"configurable": {"thread_id": thread}}
    next_input = first_input
    try:
        while True:
            await asyncio.wait_for(graph.ainvoke(next_input, config), timeout=timeout)
            payload = await pending_review(graph, config)
            if payload is None:
                break
            if auto:
                next_input = Command(resume={"action": "approve", "auto": True})
            elif interactive:
                next_input = Command(resume=ask_human(payload))
            else:
                print_pause_instructions(thread, payload)
                return 0
    except (asyncio.TimeoutError, KeyboardInterrupt, asyncio.CancelledError):
        print(f"\nStopped before finishing. Nothing is lost - the run was saved after every step.")
        print(f"Continue with:  python -m brief resume {thread}")
        return 2
    except Exception as error:  # noqa: BLE001
        print(f"\nRun failed: {str(error) or type(error).__name__}", file=sys.stderr)
        print(
            "\nMost likely causes: the free OpenRouter model is slow, unavailable or was removed from the "
            "free tier (see CLOUD_MODEL in brief/config.py), or the network is down. "
            f"The run was saved; try again with:  python -m brief resume {thread}",
            file=sys.stderr,
        )
        return 1

    final = (await graph.aget_state(config)).values
    if final.get("cancelled"):
        print_section_header("Stopped by the reviewer", "-")
        print(final["cancelled"])
        return 0
    print_section_header("Finished", "-")
    print(final["report"])
    return 0


async def cmd_run(args: argparse.Namespace) -> int:
    topic = " ".join(args.topic).strip()
    thread = args.thread or uuid.uuid4().hex[:8]
    print_section_header(f"Researching: {topic}")
    print(f"thread id: {thread}   (checkpoints: {args.db})")
    async with open_checkpointer(args.db) as saver:
        graph = build_graph(saver)
        return await drive(
            graph, thread, initial_state(topic), auto=args.auto, interactive=sys.stdin.isatty() and not args.no_prompt, timeout=args.timeout
        )


async def cmd_resume(args: argparse.Namespace) -> int:
    config = {"configurable": {"thread_id": args.thread_id}}
    async with open_checkpointer(args.db) as saver:
        graph = build_graph(saver)
        snapshot = await graph.aget_state(config)
        if not snapshot.values:
            print(f"No saved run with thread id {args.thread_id!r} in {args.db}.", file=sys.stderr)
            return 1
        payload = await pending_review(graph, config)
        print_section_header(f"Resuming: {snapshot.values.get('topic', '')}")
        print(f"thread id: {args.thread_id}")

        decision = decision_from_flags(args)
        interactive = sys.stdin.isatty() and not args.no_prompt

        if payload is not None:
            # Paused at a review: answer it with the flag given, --auto, or a prompt if a person is here.
            if decision is not None:
                first = Command(resume=decision)
            elif args.auto:
                first = Command(resume={"action": "approve", "auto": True})
            elif interactive:
                first = Command(resume=ask_human(payload))
            else:
                print_pause_instructions(args.thread_id, payload)
                return 0
        elif snapshot.next:
            first = None  # stopped, timed out or crashed mid-run: continue from the last checkpoint
        else:
            print("This run already finished:", snapshot.values.get("report_path") or "(nothing was saved - the reviewer stopped it)")
            return 0
        return await drive(graph, args.thread_id, first, auto=args.auto, interactive=interactive, timeout=args.timeout)


async def cmd_status(args: argparse.Namespace) -> int:
    config = {"configurable": {"thread_id": args.thread_id}}
    async with open_checkpointer(args.db) as saver:
        graph = build_graph(saver)
        snapshot = await graph.aget_state(config)
        if not snapshot.values:
            print(f"No saved run with thread id {args.thread_id!r} in {args.db}.", file=sys.stderr)
            return 1
        values = snapshot.values
        checkpoints = [s async for s in graph.aget_state_history(config)]
        payload = await pending_review(graph, config)
        print_section_header(f"Run {args.thread_id}: {values['topic']}", "-")
        print(f"  route:            {values['route']}")
        print(f"  checkpoints:      {len(checkpoints)} saved steps")
        print(f"  sub-questions:    {len(values['sub_questions'])}    sources: {len(values['library'])}")
        print(f"  drafts written:   {values['draft_count']}    reviewer revisions: {values['human_revisions']}")
        if values.get("cancelled"):
            print(f"  state:            stopped by the reviewer - {values['cancelled']}")
        elif payload is not None:
            print(f"  state:            PAUSED, waiting for a person ({payload['kind']})")
        elif snapshot.next:
            print(f"  state:            interrupted; would continue at: {', '.join(snapshot.next)}")
        else:
            print(f"  state:            finished -> {values['report_path'] or '(nothing saved)'}")
        return 0


async def cmd_threads(args: argparse.Namespace) -> int:
    threads = await list_threads(args.db)
    print("\n".join(threads) if threads else "No saved runs yet.")
    return 0


def make_parser() -> argparse.ArgumentParser:
    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db", default=str(CHECKPOINT_DB), help="checkpoint database file")

    def running(p: argparse.ArgumentParser) -> None:
        common(p)
        p.add_argument("--auto", action="store_true", help="approve every review automatically (no human; noted in the report)")
        p.add_argument("--no-prompt", action="store_true", help="never ask at the keyboard: stop at a review and print how to resume")
        p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds before a run is stopped (it stays resumable)")

    parser = argparse.ArgumentParser(prog="python -m brief", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    resume = sub.add_parser("resume", help="continue a paused, stopped or crashed run")
    resume.add_argument("thread_id")
    running(resume)
    group = resume.add_mutually_exclusive_group()
    group.add_argument("--approve", action="store_true")
    group.add_argument("--reject", action="store_true")
    group.add_argument("--edit", metavar='"q1 | q2"', help="replace the sub-questions (plan review)")
    group.add_argument("--feedback", metavar="TEXT", help="ask for a revision (draft review)")
    resume.set_defaults(func=cmd_resume)

    status = sub.add_parser("status", help="show where a run is")
    status.add_argument("thread_id")
    common(status)
    status.set_defaults(func=cmd_status)

    threads = sub.add_parser("threads", help="list saved runs")
    common(threads)
    threads.set_defaults(func=cmd_threads)

    run_parser = sub.add_parser("run", help='start a run (the default: python -m brief "topic")')
    run_parser.add_argument("topic", nargs="+")
    run_parser.add_argument("--thread", help="use this thread id instead of a random one")
    running(run_parser)
    run_parser.set_defaults(func=cmd_run)
    return parser


def run() -> None:
    argv = sys.argv[1:]
    # `python -m brief "some topic"` is the everyday form, so a first word that is not a command means "run".
    if argv and argv[0] not in {"resume", "status", "threads", "run", "-h", "--help"}:
        argv = ["run", *argv]
    parser = make_parser()
    if not argv:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args(argv)
    try:
        sys.exit(asyncio.run(args.func(args)))
    except KeyboardInterrupt:
        sys.exit(130)
