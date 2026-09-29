import asyncio
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

from brief.sources.http import http_get_text
from brief.sources.text import collapse_whitespace, truncate
from brief.sources.types import SourceDocument, SourceFetcher

MAX_RESULTS = 3
MAX_TEXT_LENGTH = 900

# arXiv's API terms ask for at most one request every 3 seconds. Several sub-questions search in
# parallel, so requests are queued through this gate instead of being fired all at once.
MIN_GAP_BETWEEN_REQUESTS_S = 3.0
_next_free_slot = 0.0

_ATOM = "{http://www.w3.org/2005/Atom}"


async def _wait_for_turn() -> None:
    # No `await` between reading and updating the slot, so on one event loop two callers can
    # never claim the same one.
    global _next_free_slot
    now = time.monotonic()
    my_turn = max(now, _next_free_slot)
    _next_free_slot = my_turn + MIN_GAP_BETWEEN_REQUESTS_S
    if my_turn > now:
        await asyncio.sleep(my_turn - now)


def _bad_format() -> RuntimeError:
    return RuntimeError("Unexpected response format from export.arxiv.org")


def parse_feed(xml: str) -> list[SourceDocument]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        raise _bad_format() from error
    if root.tag not in (f"{_ATOM}feed", "feed"):
        raise _bad_format()

    documents = []
    for entry in root.findall(f"{_ATOM}entry") + root.findall("entry"):
        fields = {name: entry.findtext(f"{_ATOM}{name}") or entry.findtext(name) for name in ("id", "title", "summary")}
        if any(value is None for value in fields.values()):
            raise _bad_format()
        documents.append(
            SourceDocument(
                kind="arxiv",
                title=collapse_whitespace(fields["title"]),
                url=fields["id"].replace("http:", "https:", 1) if fields["id"].startswith("http:") else fields["id"],
                text=truncate(collapse_whitespace(fields["summary"]), MAX_TEXT_LENGTH),
            )
        )
    return documents


async def fetch_arxiv(query: str) -> list[SourceDocument]:
    await _wait_for_turn()
    params = urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": MAX_RESULTS, "sortBy": "relevance"}
    )
    return parse_feed(await http_get_text(f"https://export.arxiv.org/api/query?{params}"))


arxiv_fetcher = SourceFetcher(kind="arxiv", label="arXiv", fetch=fetch_arxiv)
