"""Bonus demo for the 'function / tool calling' topic (not one of the 3 graded exercises,
same role as week1's switch_demo.py). Shows the full loop: the model requests a tool call,
our code executes the actual Python function, and the result is fed back for a final answer.
Run: python tool_calling_demo.py
"""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import config
import models

# A fixed policy table standing in for what would normally be a database lookup or internal API.
_PRIORITY_POLICY = {
    "billing": "Billing issues are handled within 2 business days, unless a duplicate charge is involved - then same-day.",
    "technical": "Technical issues blocking work are 'urgent' and handled within 1 hour; non-blocking ones within 1 day.",
    "account": "Account access issues (e.g. can't log in) are 'urgent' - handled within 1 hour.",
    "feature_request": "Feature requests are logged for the roadmap and are never urgent.",
    "bug": "Bugs are triaged within 1 business day and escalated to 'urgent' if they cause data loss.",
    "other": "Unclassified issues default to 'medium' priority pending manual review.",
}


@tool
def lookup_priority_policy(issue_type: str) -> str:
    """Return this company's support priority policy for a given issue type
    (one of: billing, technical, account, feature_request, bug, other)."""
    return _PRIORITY_POLICY.get(issue_type, "No policy on file for this issue type; default to 'medium' priority.")


def run_demo():
    chat = models.get_chat_model(config.TOOL_DEMO_MODEL, temperature=0)
    chat_with_tools = chat.bind_tools([lookup_priority_policy])

    messages = [
        SystemMessage(
            content="You are a support triage assistant. Use the lookup_priority_policy tool "
            "to decide priority before answering."
        ),
        HumanMessage(
            content="A customer says they can't log in and have a demo in an hour. "
            "What priority should this ticket get, and why?"
        ),
    ]

    print("Step 1 - sending the question with a tool available...")
    ai_message = chat_with_tools.invoke(messages)
    messages.append(ai_message)

    if not ai_message.tool_calls:
        print(
            "\nThe model answered directly without calling the tool (this can happen if the "
            "free model doesn't reliably support tool calling on OpenRouter - try a different "
            "TOOL_DEMO_MODEL in config.py). Its answer:\n"
        )
        print(ai_message.content)
        return

    for tool_call in ai_message.tool_calls:
        print(f"\nStep 2 - model requested a tool call: {tool_call['name']}({tool_call['args']})")
        result = lookup_priority_policy.invoke(tool_call["args"])
        print(f"Step 3 - tool executed locally, returned: {result!r}")
        messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    print("\nStep 4 - sending the tool result back to the model for a final answer...")
    final = chat_with_tools.invoke(messages)
    print("\nFinal answer:")
    print(final.content)


if __name__ == "__main__":
    run_demo()
