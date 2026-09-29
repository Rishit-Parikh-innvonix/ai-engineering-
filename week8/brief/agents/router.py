from langchain_openai import ChatOpenAI

from brief.agents.llm import ask_for
from brief.logic import parse_route_reply
from brief.types import RouteDecision

SYSTEM_PROMPT = (
    "You are the routing agent of a research assistant. It can research topics in Wikipedia, arXiv "
    "and Hacker News (background knowledge, papers, developer discussion) and it has ONE live-data "
    "tool: current weather for a city. Decide how the user's topic must be answered and reply with "
    "exactly one of these, and nothing else:\n"
    "RESEARCH - answerable from encyclopedia articles, papers or discussions. This includes climate "
    "and weather as a concept (e.g. 'why do monsoons happen') and 'the current state of <a field>'.\n"
    "WEATHER: <city> - it asks for the actual weather, temperature or conditions right now or today "
    "in a specific place. Example: WEATHER: Ahmedabad\n"
    "MIXED: <city> | <the part of the question that is not about the live weather> - both. "
    "Example: MIXED: Ahmedabad | why the monsoon happens in Gujarat\n"
    "UNSUPPORTED_LIVE - it needs other live data such as prices, scores, traffic or breaking news, "
    "which no tool can provide.\n"
    "Reply with the single line only."
)


def create_router(model: ChatOpenAI):
    # Plain text with strict parsing rather than function calling, for the same reason as the
    # relevance grader: this model sometimes returns a tool call with no arguments at all.
    async def route(topic: str) -> RouteDecision:
        return await ask_for(model, SYSTEM_PROMPT, f"Topic: {topic}", parse_route_reply)

    return route
