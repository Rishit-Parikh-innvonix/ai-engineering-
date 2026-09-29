import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from brief.agents.llm import ask_for, with_retries
from brief.logic import parse_relevant_numbers, strip_source_label
from brief.sources.text import truncate
from brief.sources.types import SourceDocument
from brief.types import QueryPlan

# ---------- The retrieval agent's first job: write the search queries ----------

QUERY_PROMPT = (
    "You are the retrieval agent of a research assistant. Given a research sub-question, write one "
    "search query for each of three sources, in this order: (1) Wikipedia - a short topic or page "
    "title; (2) arXiv - technical keywords a research paper abstract would contain; (3) Hacker News - "
    "a short phrase a developer would type when discussing it. Use key terms, not full sentences. "
    "Write in English."
)


class QueryListSchema(BaseModel):
    """One list instead of three named fields on purpose: with three separate string fields this
    model returned ": " for every field (checked against the raw tool call), while a single list works."""

    queries: list[str] = Field(description="Exactly 3 search queries, in the order: Wikipedia, arXiv, Hacker News.")


def create_query_writer(model: ChatOpenAI):
    query_writer = model.with_structured_output(QueryListSchema, method="function_calling")

    async def craft_queries(*, topic: str, sub_question: str, previous_queries: list[QueryPlan]) -> QueryPlan:
        retry_hint = ""
        if previous_queries:
            tried = json.dumps([q.model_dump() for q in previous_queries])
            retry_hint = (
                "\n\nThese queries were already tried and returned nothing relevant. Write DIFFERENT ones, but "
                "keep the specific names from the topic (for example product, framework or technique names) "
                "in them - only change the surrounding words. Generic queries like 'AI agents' find "
                f"unrelated results.\nAlready tried:\n{tried}"
            )

        async def attempt() -> QueryPlan:
            result = await query_writer.ainvoke(
                [
                    SystemMessage(content=QUERY_PROMPT),
                    HumanMessage(content=f"Overall topic: {topic}\nSub-question to research: {sub_question}{retry_hint}"),
                ]
            )
            q = [*result.queries, "", "", ""]
            # Raises (and so retries) if the model returned fewer than 3 queries or empty/garbage ones.
            return QueryPlan(
                wikipedia=strip_source_label(q[0]), arxiv=strip_source_label(q[1]), hackernews=strip_source_label(q[2])
            )

        return await with_retries(attempt)

    return craft_queries


# ---------- Its second job: judge what came back, and drop what is off-topic ----------

GRADER_PROMPT = (
    "You are the retrieval agent of a research assistant, checking search results. Given a research "
    "sub-question and a numbered list of search results, decide which results are genuinely relevant "
    "to answering it in the context of the overall topic. A result that merely shares a word with "
    "the topic but is about something else must be left out.\n"
    "Reply with ONLY the numbers of the relevant results, separated by commas (for example: 1, 4). "
    "If none are relevant, reply with only the word NONE. No explanations."
)


def create_relevance_grader(model: ChatOpenAI):
    """Search engines return *something* for any query. Without this check, off-topic results count
    as "coverage", the retry loop never fires, and the brief is written from junk.

    Asked for as plain text rather than function calling: the function-calling version crashed live
    (the model sometimes returned a tool call with no arguments at all), and "1, 4" or "NONE" is
    trivially checkable by parse_relevant_numbers."""

    async def grade_relevance(*, topic: str, sub_question: str, documents: list[SourceDocument]) -> list[SourceDocument]:
        if not documents:
            return []
        listing = "\n".join(
            f"{i}. [{doc.kind}] {doc.title}: {truncate(doc.text, 250)}" for i, doc in enumerate(documents, start=1)
        )
        keep = set(
            await ask_for(
                model,
                GRADER_PROMPT,
                f"Overall topic: {topic}\nSub-question: {sub_question}\n\nSearch results:\n{listing}",
                lambda reply: parse_relevant_numbers(reply, len(documents)),
            )
        )
        return [doc for i, doc in enumerate(documents, start=1) if i in keep]

    return grade_relevance
