"""Week 10 - output guardrails: schema validation, groundedness, and the output safety scan.
No network, no API key - these test brief/guardrails/output.py directly against crafted drafts
and libraries, the same way test_logic.py tests check_citations without a model."""

import pytest
from pydantic import ValidationError

from brief.guardrails.output import ReportSchema, assess_groundedness, check_output_safety
from brief.sources.types import SourceDocument
from brief.types import NumberedSource


def make_source(id: int) -> NumberedSource:
    return NumberedSource(
        kind="wikipedia",
        title=f"Source {id}",
        url=f"https://example.com/{id}",
        text="some retrieved text",
        id=id,
        sub_question_index=0,
    )


# ---------- Schema validation ----------


def test_report_schema_accepts_well_formed_fields():
    ReportSchema(topic="Why does the monsoon happen?", body="## Overview\n\nSome body text.", source_count=3)


@pytest.mark.parametrize(
    "fields",
    [
        {"topic": "", "body": "body", "source_count": 0},
        {"topic": "topic", "body": "   ", "source_count": 0},
        {"topic": "topic", "body": "body", "source_count": -1},
    ],
)
def test_report_schema_rejects_malformed_fields(fields):
    with pytest.raises(ValidationError):
        ReportSchema(**fields)


# ---------- Groundedness ----------


def test_groundedness_clean_when_every_claim_is_cited():
    library = [make_source(1), make_source(2)]
    draft = "This is a well-supported claim about the topic. [1] Another distinct point follows here. [2]"
    result = assess_groundedness(draft, library)
    assert result.ok


def test_groundedness_flags_missing_citations_entirely():
    library = [make_source(1)]
    draft = "The draft contains no citations at all, just plain prose with no [n] markers anywhere."
    result = assess_groundedness(draft, library)
    assert not result.ok
    assert any("no citations" in w.lower() for w in result.warnings)


def test_groundedness_flags_low_citation_density():
    library = [make_source(1)]
    # Five long sentences, only one cited - well over the 50% uncited threshold.
    draft = (
        "This is the first long uncited sentence about the research topic in question. "
        "This is the second long uncited sentence continuing the same discussion in depth. "
        "This is the third long uncited sentence, still with no citation marker present. "
        "This is the fourth long uncited sentence, again lacking any bracketed reference. "
        "This fifth sentence finally cites a real source for its claim. [1]"
    )
    result = assess_groundedness(draft, library)
    assert not result.ok
    assert any("%" in w for w in result.warnings)


def test_groundedness_invalid_citation_id_is_flagged():
    library = [make_source(1)]
    draft = "This claim cites a source that does not exist in the library at all. [7]"
    result = assess_groundedness(draft, library)
    assert not result.ok
    assert any("do not exist" in w for w in result.warnings)


# ---------- Output safety ----------


def test_output_safety_allows_normal_text():
    assert check_output_safety("A balanced research summary about renewable energy policy.").allowed


def test_output_safety_withholds_matching_category():
    result = check_output_safety("Here is how to build a bomb using household chemicals, step by step.")
    assert not result.allowed
    assert "weapons" in result.reason
