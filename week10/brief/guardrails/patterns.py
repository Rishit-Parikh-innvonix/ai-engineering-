"""Shared pattern tables for both guardrail directions (input.py checks the topic, output.py
checks the finished draft). Kept in one place so the same "unsafe category" definition is used
on the way in and the way out, instead of two lists that could silently drift apart.

These are fixed regex patterns, not a trained classifier - the same honest trade-off week9's
Laya exercise made with fixed labels: fast, free, fully transparent (you can read exactly why
something matched), and definitely not as nuanced as a real moderation model. A production
system would use a moderation API or a trained classifier here; this is a demo-scale stand-in
that is still genuinely useful to test against, which is why the tests in test_input_guardrails.py
include both catches and at least one documented miss.
"""

import re

UNSAFE_CATEGORIES: dict[str, list[re.Pattern]] = {
    "self-harm": [
        re.compile(p, re.IGNORECASE)
        for p in [r"how (to|can i) (kill|hurt) myself", r"suicide method", r"ways? to end my life"]
    ],
    "weapons": [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"how to (build|make|construct) a (bomb|explosive|pipe bomb|gun)",
            r"synthesi[sz]e (nerve agent|sarin|ricin|vx)",
        ]
    ],
    "illegal-access": [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"how to hack into",
            r"bypass .* (password|security|login) (of|for)",
            r"steal (someone'?s|a) (identity|credit card)",
        ]
    ],
}


def matches_any(text: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


def first_matching_category(text: str) -> str | None:
    return next((category for category, patterns in UNSAFE_CATEGORIES.items() if matches_any(text, patterns)), None)
