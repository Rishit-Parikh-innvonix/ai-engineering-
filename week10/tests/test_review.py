import pytest

from brief.review import ReviewDecision, parse_draft_decision, parse_plan_decision


def test_plan_decisions_accept_plain_words_and_dicts():
    assert parse_plan_decision("approve") == ReviewDecision("approve")
    assert parse_plan_decision("  REJECT ") == ReviewDecision("reject")
    assert parse_plan_decision({"action": "approve", "auto": True}) == ReviewDecision("approve", auto=True)


def test_plan_edit_cleans_the_new_questions():
    decision = parse_plan_decision({"action": "edit", "sub_questions": ["  What is X used for? ", "What is X used for?", "", "How does X work?"]})
    assert decision.action == "edit"
    assert decision.sub_questions == ("What is X used for?", "How does X work?")  # trimmed, de-duplicated, blanks dropped


@pytest.mark.parametrize(
    "answer, message",
    [
        ("banana", "Unknown action"),
        ({"action": "revise", "feedback": "x"}, "Unknown action"),  # revise is a draft-review action
        ({"action": "edit"}, "needs"),
        ({"action": "edit", "sub_questions": []}, "at least one"),
        ({"action": "edit", "sub_questions": ["one two three"] * 1 + [f"question number {i}" for i in range(5)]}, "at most 4"),
        ({"action": "edit", "sub_questions": ["??", "12345"]}, "real question"),
        (42, "dict"),
        (None, "dict"),
    ],
)
def test_bad_plan_answers_raise_a_message_the_reviewer_can_read(answer, message):
    with pytest.raises(ValueError, match=message):
        parse_plan_decision(answer)


def test_draft_decisions():
    assert parse_draft_decision("approve", revisions_left=2) == ReviewDecision("approve")
    assert parse_draft_decision({"action": "reject"}, revisions_left=0) == ReviewDecision("reject")
    revise = parse_draft_decision({"action": "revise", "feedback": "  make it shorter "}, revisions_left=1)
    assert revise == ReviewDecision("revise", feedback="make it shorter")


def test_revise_needs_feedback_and_a_revision_to_spend():
    with pytest.raises(ValueError, match="needs"):
        parse_draft_decision({"action": "revise"}, revisions_left=2)
    with pytest.raises(ValueError, match="needs"):
        parse_draft_decision({"action": "revise", "feedback": "ok"}, revisions_left=2)  # too short to mean anything
    with pytest.raises(ValueError, match="No revisions are left"):
        parse_draft_decision({"action": "revise", "feedback": "make it shorter"}, revisions_left=0)
    with pytest.raises(ValueError, match="Unknown action"):
        parse_draft_decision({"action": "edit"}, revisions_left=2)
