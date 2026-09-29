import re
from urllib.parse import urlencode

from pydantic import BaseModel

from brief.sources.http import http_get_json
from brief.sources.text import strip_html, truncate
from brief.sources.types import SourceDocument, SourceFetcher

MAX_RESULTS = 3
MIN_COMMENT_LENGTH = 80
MAX_TEXT_LENGTH = 600
HITS_TO_FETCH = 20  # fetch extra hits because some get filtered out below

# The monthly "Who is hiring?" / "Who wants to be hired?" threads mention every technology keyword
# under the sun, so they dominate comment search results for any tech topic while saying nothing
# useful about it. Threads titled "[dead]" or "[flagged]" were removed by moderators.
_NOISE_THREAD = re.compile(r"who (is|wants to be) (hiring|hired)|^\[(dead|flagged)\]$", re.IGNORECASE)


def is_noise_thread(story_title: str | None) -> bool:
    return bool(_NOISE_THREAD.search(story_title or ""))


class _Hit(BaseModel):
    objectID: str
    comment_text: str | None = None
    story_title: str | None = None


class _Response(BaseModel):
    hits: list[_Hit]


async def fetch_hackernews(query: str) -> list[SourceDocument]:
    """Comments (not story titles) are searched on purpose: a title is a headline, while a comment
    is an actual developer explaining what they found - the "practitioner view" this source is for."""
    params = urlencode({"query": query, "tags": "comment", "hitsPerPage": HITS_TO_FETCH})
    data = await http_get_json(f"https://hn.algolia.com/api/v1/search?{params}", _Response)

    documents = []
    for hit in data.hits:
        if is_noise_thread(hit.story_title):
            continue
        text = strip_html(hit.comment_text or "")
        if len(text) < MIN_COMMENT_LENGTH:
            continue
        documents.append(
            SourceDocument(
                kind="hackernews",
                title=f'Comment on "{hit.story_title or "a Hacker News thread"}"',
                url=f"https://news.ycombinator.com/item?id={hit.objectID}",
                text=truncate(text, MAX_TEXT_LENGTH),
            )
        )
    return documents[:MAX_RESULTS]


hackernews_fetcher = SourceFetcher(kind="hackernews", label="Hacker News", fetch=fetch_hackernews)
