import { XMLParser } from "fast-xml-parser";
import { z } from "zod";

import { httpGetXml } from "./http.js";
import { collapseWhitespace, truncate } from "./text.js";
import type { SourceDocument, SourceFetcher } from "./types.js";

const MAX_RESULTS = 3;
const MAX_TEXT_LENGTH = 900;

// arXiv's API terms ask for at most one request every 3 seconds. Three sub-questions search in
// parallel, so requests are queued through this gate instead of being fired all at once.
const MIN_GAP_BETWEEN_REQUESTS_MS = 3_000;
let nextFreeSlot = 0;

async function waitForTurn(): Promise<void> {
  const now = Date.now();
  const myTurn = Math.max(now, nextFreeSlot);
  nextFreeSlot = myTurn + MIN_GAP_BETWEEN_REQUESTS_MS;
  if (myTurn > now) {
    await new Promise((resolve) => setTimeout(resolve, myTurn - now));
  }
}

// The feed has a single <entry> when there's one result, so it must be forced into an array.
const xmlParser = new XMLParser({ isArray: (tagName) => tagName === "entry" });
const feedSchema = z.object({
  feed: z.object({
    entry: z.array(z.object({ id: z.string(), title: z.string(), summary: z.string() })).optional(),
  }),
});

export function parseFeed(xml: string): SourceDocument[] {
  const parsed = feedSchema.safeParse(xmlParser.parse(xml));
  if (!parsed.success) {
    throw new Error("Unexpected response format from export.arxiv.org");
  }
  return (parsed.data.feed.entry ?? []).map((entry) => ({
    kind: "arxiv" as const,
    title: collapseWhitespace(entry.title),
    url: entry.id.replace(/^http:/, "https:"),
    text: truncate(collapseWhitespace(entry.summary), MAX_TEXT_LENGTH),
  }));
}

export const arxivFetcher: SourceFetcher = {
  kind: "arxiv",
  label: "arXiv",
  async fetch(query) {
    await waitForTurn();
    const params = new URLSearchParams({
      search_query: `all:${query}`,
      start: "0",
      max_results: String(MAX_RESULTS),
      sortBy: "relevance",
    });
    return httpGetXml(`https://export.arxiv.org/api/query?${params}`, parseFeed);
  },
};
