"""What a human may answer when the graph pauses, and how the answer is checked.

The graph pauses with `interrupt(payload)`; whoever resumes it supplies a decision. It arrives from
a terminal prompt, a command-line flag or a test, so it is treated as untrusted input: pure
functions here either return a clean ReviewDecision or raise ValueError with a message that is
shown back to the reviewer (the graph then asks again instead of crashing)."""

import re
from dataclasses import dataclass
from typing import Any, Literal

MAX_SUB_QUESTIONS = 4


@dataclass(frozen=True)
class ReviewDecision:
    action: Literal["approve", "edit", "revise", "reject"]
    sub_questions: tuple[str, ...] = ()
    feedback: str = ""
    auto: bool = False  # True when a program (--auto), not a person, approved


def _as_fields(value: Any) -> dict:
    if isinstance(value, str):
        return {"action": value}
    if isinstance(value, dict):
        return value
    raise ValueError('Answer with a dict like {"action": "approve"} or with a plain word such as "approve".')


def _has_words(text: str, minimum: int) -> bool:
    return len(text.strip()) >= minimum and re.search(r"[^\W\d_]{2}", text) is not None


def clean_sub_questions(items: Any) -> tuple[str, ...]:
    if not isinstance(items, (list, tuple)):
        raise ValueError('"edit" needs "sub_questions": a list of question strings.')
    cleaned = list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))
    if not cleaned:
        raise ValueError("Give at least one sub-question.")
    if len(cleaned) > MAX_SUB_QUESTIONS:
        raise ValueError(f"Give at most {MAX_SUB_QUESTIONS} sub-questions (you gave {len(cleaned)}).")
    if not all(_has_words(question, 5) for question in cleaned):
        raise ValueError("Every sub-question must be a real question of a few words.")
    return tuple(cleaned)


def _action(fields: dict, allowed: tuple[str, ...]) -> str:
    action = str(fields.get("action", "")).strip().lower()
    if action not in allowed:
        raise ValueError(f'Unknown action "{action}". Choose one of: {", ".join(allowed)}.')
    return action


def parse_plan_decision(value: Any) -> ReviewDecision:
    """Plan review: approve the sub-questions, replace them, or cancel the run."""
    fields = _as_fields(value)
    action = _action(fields, ("approve", "edit", "reject"))
    auto = bool(fields.get("auto", False))
    if action == "edit":
        return ReviewDecision("edit", sub_questions=clean_sub_questions(fields.get("sub_questions")))
    return ReviewDecision(action, auto=auto)  # type: ignore[arg-type]


def parse_draft_decision(value: Any, *, revisions_left: int) -> ReviewDecision:
    """Publish review: approve (save the report), ask for a revision with feedback, or discard."""
    fields = _as_fields(value)
    action = _action(fields, ("approve", "revise", "reject"))
    auto = bool(fields.get("auto", False))
    if action == "revise":
        if revisions_left <= 0:
            raise ValueError("No revisions are left for this run: approve the draft or reject it.")
        feedback = str(fields.get("feedback", "")).strip()
        if not _has_words(feedback, 3):
            raise ValueError('"revise" needs "feedback": what should change in the draft?')
        return ReviewDecision("revise", feedback=feedback)
    return ReviewDecision(action, auto=auto)  # type: ignore[arg-type]
