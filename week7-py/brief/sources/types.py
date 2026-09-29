from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

from pydantic import BaseModel

# "weather" is not searched: it comes from a live-data tool (brief/tools/weather.py), not a search box.
SearchKind = Literal["wikipedia", "arxiv", "hackernews"]
SourceKind = Literal["wikipedia", "arxiv", "hackernews", "weather"]


class SourceDocument(BaseModel):
    kind: SourceKind
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class SourceFetcher:
    """One place a search can be run. `fetch` raises on network/HTTP/format problems and returns []
    when the source simply has nothing - the retrieval step treats those two cases differently."""

    kind: SearchKind
    label: str
    fetch: Callable[[str], Awaitable[list[SourceDocument]]]
