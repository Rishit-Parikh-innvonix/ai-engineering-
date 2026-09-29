import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, field_validator

from brief.sources.types import SourceDocument


class QueryPlan(BaseModel):
    """What the retrieval agent produces: one search query per source, because each wants a
    different style (Wikipedia matches page titles, arXiv keywords, Hacker News how developers talk).

    A query must contain real words. Seen live: the model returned ": " for every source, the
    searches ran anyway, and astrophysics papers ended up in a report about software frameworks."""

    wikipedia: str
    arxiv: str
    hackernews: str

    @field_validator("wikipedia", "arxiv", "hackernews")
    @classmethod
    def _must_contain_real_words(cls, query: str) -> str:
        query = query.strip()
        if len(query) < 3 or not re.search(r"[a-z0-9]{2}", query, re.IGNORECASE):
            raise ValueError("a search query must contain real words")
        return query


class RetrievedBatch(BaseModel):
    """Everything one retrieval branch found for one sub-question in one attempt. Batches are only
    ever appended to the graph state (never edited), so parallel branches can't overwrite each other."""

    sub_question_index: int
    attempt: int
    queries: QueryPlan
    documents: list[SourceDocument]


class NumberedSource(SourceDocument):
    """A source that survived deduplication and got its citation number."""

    id: int
    sub_question_index: int


@dataclass(frozen=True)
class RouteDecision:
    """The router's decision about how a topic gets answered (see agents/router.py).

    research    - the normal pipeline: plan, search the three sources, summarize, write
    weather     - live data: only the weather tool can answer it
    mixed       - both: the weather tool for `city`, and research for `research_topic`
    unsupported - needs live data (prices, scores, news of today...) that no tool provides
    """

    kind: Literal["research", "weather", "mixed", "unsupported"]
    city: str = ""
    research_topic: str = ""
