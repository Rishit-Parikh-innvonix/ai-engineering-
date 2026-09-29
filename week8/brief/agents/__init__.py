from langchain_openai import ChatOpenAI

from brief.agents.planner import create_planner
from brief.agents.retrieval import create_query_writer, create_relevance_grader
from brief.agents.router import create_router
from brief.agents.summarizer import create_summarizer
from brief.agents.types import Agents
from brief.agents.writer import create_writer


def create_agents(model: ChatOpenAI) -> Agents:
    return Agents(
        route=create_router(model),
        plan=create_planner(model),
        craft_queries=create_query_writer(model),
        grade_relevance=create_relevance_grader(model),
        summarize=create_summarizer(model),
        write=create_writer(model),
    )


__all__ = ["Agents", "create_agents"]
