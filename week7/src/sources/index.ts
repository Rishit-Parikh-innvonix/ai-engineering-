import { arxivFetcher } from "./arxiv.js";
import { hackerNewsFetcher } from "./hackernews.js";
import { wikipediaFetcher } from "./wikipedia.js";
import type { SourceFetcher } from "./types.js";

export { SOURCE_KINDS } from "./types.js";
export type { SearchKind, SourceDocument, SourceFetcher, SourceKind } from "./types.js";

export const DEFAULT_FETCHERS: SourceFetcher[] = [wikipediaFetcher, arxivFetcher, hackerNewsFetcher];
