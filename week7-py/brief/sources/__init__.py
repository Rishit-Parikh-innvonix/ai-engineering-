from brief.sources.arxiv import arxiv_fetcher
from brief.sources.hackernews import hackernews_fetcher
from brief.sources.types import SearchKind, SourceDocument, SourceFetcher, SourceKind
from brief.sources.wikipedia import wikipedia_fetcher

DEFAULT_FETCHERS: list[SourceFetcher] = [wikipedia_fetcher, arxiv_fetcher, hackernews_fetcher]

__all__ = ["DEFAULT_FETCHERS", "SearchKind", "SourceDocument", "SourceFetcher", "SourceKind"]
