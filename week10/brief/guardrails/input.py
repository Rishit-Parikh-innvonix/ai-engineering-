"""Input guardrails (Week 10): checked on the raw topic BEFORE it reaches any LLM or tool call.

Three checks, all plain pattern-matching - deterministic, instant, and free, the same trade-off
week9's Laya exercise made with fixed classification labels: a bounded, known-shape problem (a
PII format, an injection phrase, a banned-topic phrase) gets a cheap, fully-auditable check
instead of sending every topic to an LLM to ask "is this OK?" (slower, costs tokens, and you
can't easily see *why* it decided what it decided). This is explicitly NOT a production-grade
moderation system - see patterns.py's docstring for the honest limitation.

1. Prompt injection - phrases that try to override or exfiltrate the system prompt. Blocked
   outright: the model should never even see text trying to manipulate it.
2. Unsafe / off-topic categories - a small fixed list (self-harm, weapons, illegal access) this
   research assistant should refuse, plus a few "this isn't a research topic at all" phrasings
   (write my homework, tell me a joke). Blocked outright.
3. PII - emails, phone numbers, credit-card-like and SSN-like numbers are redacted (not blocked);
   the sanitized topic is what actually gets researched, so the raw PII never reaches the cloud
   model.
"""

import re
from dataclasses import dataclass, field

from brief.guardrails.patterns import first_matching_category, matches_any


@dataclass
class InputCheckResult:
    allowed: bool
    reason: str  # non-empty only when allowed is False
    sanitized_topic: str  # topic with PII redacted (equal to the input topic if none was found)
    redactions: list[str] = field(default_factory=list)  # which PII kinds were found and redacted


_PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    # Checked last and only on what's left: unlike email/phone/ssn, a bare 13-16 digit run has no
    # punctuation to anchor on, so it would otherwise eat an already-redacted phone number's digits.
    "credit_card": re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"),
}

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\b.{0,20}\b(instructions|prompts?|rules)",
        r"disregard\b.{0,20}\b(instructions|prompts?|rules)",
        r"forget (all|everything|your) (instructions|rules|previous)",
        r"reveal (your|the) (system prompt|instructions)",
        r"print (your|the) (system prompt|instructions)",
        r"what (is|are) your (system prompt|instructions)",
        r"you are now\b",
        r"new instructions?\s*:",
        r"system prompt\s*:",
        r"\bjailbreak\b",
    ]
]

# "This isn't a research topic at all" - a research-brief assistant has no business doing these,
# whether or not they're unsafe. Deliberately narrow: a false negative here just means Exercise 2's
# router agent sees it and (correctly) classifies it as UNSUPPORTED_LIVE or researches it anyway;
# this is a fast pre-filter, not the only line of defense.
_OFF_TOPIC_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"write (me )?(a |my )?(poem|song|story|essay|homework)",
        r"do my (homework|assignment)",
        r"tell me a joke",
        r"translate (this|the following)",
    ]
]


def _redact_pii(text: str) -> tuple[str, list[str]]:
    redacted = text
    found: list[str] = []
    for kind, pattern in _PII_PATTERNS.items():
        if pattern.search(redacted):
            found.append(kind)
            redacted = pattern.sub(f"[REDACTED_{kind.upper()}]", redacted)
    return redacted, found


def check_input(topic: str) -> InputCheckResult:
    """Runs the three checks in order. Injection and unsafe/off-topic block outright (the model
    never sees the raw text); PII is redacted and the request proceeds with the sanitized topic."""
    if matches_any(topic, _INJECTION_PATTERNS):
        return InputCheckResult(False, "This request looks like an attempt to override the assistant's instructions.", topic)

    category = first_matching_category(topic)
    if category:
        return InputCheckResult(False, f"This request falls into a category this assistant will not research ({category}).", topic)

    if matches_any(topic, _OFF_TOPIC_PATTERNS):
        return InputCheckResult(False, "This isn't a research topic this assistant can help with.", topic)

    sanitized, redactions = _redact_pii(topic)
    return InputCheckResult(True, "", sanitized, redactions)
