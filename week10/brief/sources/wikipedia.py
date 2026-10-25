from urllib.parse import urlencode

from pydantic import BaseModel

from brief.sources.http import http_get_json
from brief.sources.text import truncate
from brief.sources.types import SourceDocument, SourceFetcher

MAX_RESULTS = 2
MAX_TEXT_LENGTH = 1200


class _Page(BaseModel):
    title: str
    index: int | None = None
    extract: str | None = None
    fullurl: str | None = None


class _Query(BaseModel):
    pages: list[_Page]


class _Response(BaseModel):
    query: _Query | None = None


async def fetch_wikipedia(query: str) -> list[SourceDocument]:
    """One request does both jobs: generator=search finds matching pages, and prop=extracts returns
    each page's plain-text introduction (no HTML), so no second round trip per result."""
    params = urlencode(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": MAX_RESULTS,
            "prop": "extracts|info",
            "inprop": "url",
            "exintro": "1",
            "explaintext": "1",
            "exlimit": "max",
            "format": "json",
            "formatversion": "2",
        }
    )
    data = await http_get_json(f"https://en.wikipedia.org/w/api.php?{params}", _Response)

    # generator=search returns pages unordered; `index` is the search rank.
    pages = sorted(data.query.pages if data.query else [], key=lambda page: page.index or 0)
    return [
        SourceDocument(kind="wikipedia", title=page.title, url=page.fullurl, text=truncate(page.extract.strip(), MAX_TEXT_LENGTH))
        for page in pages
        if page.extract and page.extract.strip() and page.fullurl
    ]


wikipedia_fetcher = SourceFetcher(kind="wikipedia", label="Wikipedia", fetch=fetch_wikipedia)
