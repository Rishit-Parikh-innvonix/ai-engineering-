"""Output guardrails (Week 10): checked on the finished draft, right before it is saved.

1. Schema validation (Pydantic) - the report's structural fields are validated before writing.
   In normal operation this should always pass (the graph never reaches `save` with an empty
   draft), so this mostly guards against a future programming mistake upstream, not a model
   mistake - cheap, structural insurance, exactly what the exercise asks for.
2. Groundedness - reuses `logic.check_citations` (does every claim cite a source that actually
   exists?) and adds a citation-density check: what fraction of the draft's sentences carry no
   [n] at all. A low score is a WARNING added to the report, not a hard block - a weak free
   model's citation habits vary run to run, and blocking outright would throw away real research
   over a formatting slip. Same "degrade and say so honestly" philosophy the rest of this project
   already uses for model/tool failures.
3. Safety policy - the same category patterns the input guardrail uses, run on the OUTPUT this
   time (defense in depth). The model is already constrained to cite only retrieved sources, so
   this rarely triggers in practice; it exists as the backstop if a retrieved source itself
   contained something that shouldn't be repeated verbatim.
"""

import re

from pydantic import BaseModel, Field, field_validator

from brief.guardrails.patterns import first_matching_category
from brief.logic import check_citations
from brief.types import NumberedSource


class ReportSchema(BaseModel):
    """The structural shape every saved report must have. Raises on construction if violated."""

    topic: str = Field(min_length=1)
    body: str = Field(min_length=1)
    source_count: int = Field(ge=0)

    @field_validator("topic", "body")
    @classmethod
    def _not_just_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace-only")
        return value


class GroundednessResult:
    def __init__(self, warnings: list[str]):
        self.warnings = warnings

    @property
    def ok(self) -> bool:
        return not self.warnings


# A rough sentence splitter is enough here - this only needs to count citation markers per
# sentence, not parse grammar correctly.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITATION_DENSITY_WARNING_THRESHOLD = 0.5  # more than half the sentences uncited -> warn


def assess_groundedness(draft: str, library: list[NumberedSource]) -> GroundednessResult:
    warnings = list(check_citations(draft, len(library)))
    sentences = [s for s in _SENTENCE_SPLIT.split(draft) if len(s.strip()) > 20]
    if sentences:
        uncited = sum(1 for s in sentences if not re.search(r"\[\d", s))
        ratio = uncited / len(sentences)
        if ratio > _CITATION_DENSITY_WARNING_THRESHOLD:
            warnings.append(
                f"{uncited} of {len(sentences)} sentences ({ratio:.0%}) have no citation at all - "
                "treat this brief's uncited claims with extra care."
            )
    return GroundednessResult(warnings)


class SafetyResult:
    def __init__(self, allowed: bool, reason: str = ""):
        self.allowed = allowed
        self.reason = reason


def check_output_safety(text: str) -> SafetyResult:
    category = first_matching_category(text)
    if category:
        return SafetyResult(False, f"the draft matched a withheld category ({category})")
    return SafetyResult(True)
