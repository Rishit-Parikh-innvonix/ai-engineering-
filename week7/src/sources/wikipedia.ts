import { z } from "zod";

import { httpGetJson } from "./http.js";
import { truncate } from "./text.js";
import type { SourceFetcher } from "./types.js";

const MAX_RESULTS = 2;
const MAX_TEXT_LENGTH = 1200;

const responseSchema = z.object({
  query: z
    .object({
      pages: z.array(
        z.object({
          title: z.string(),
          index: z.number().optional(),
          extract: z.string().optional(),
          fullurl: z.string().optional(),
        })
      ),
    })
    .optional(),
});

// One request does both jobs: generator=search finds matching pages, and prop=extracts returns
// each page's plain-text introduction (no HTML), so no second round trip per result.
export const wikipediaFetcher: SourceFetcher = {
  kind: "wikipedia",
  label: "Wikipedia",
  async fetch(query) {
    const params = new URLSearchParams({
      action: "query",
      generator: "search",
      gsrsearch: query,
      gsrlimit: String(MAX_RESULTS),
      prop: "extracts|info",
      inprop: "url",
      exintro: "1",
      explaintext: "1",
      exlimit: "max",
      format: "json",
      formatversion: "2",
    });
    const data = await httpGetJson(`https://en.wikipedia.org/w/api.php?${params}`, responseSchema);

    // generator=search returns pages unordered; `index` is the search rank.
    const pages = [...(data.query?.pages ?? [])].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
    return pages.flatMap((page) =>
      page.extract?.trim() && page.fullurl
        ? [{ kind: "wikipedia" as const, title: page.title, url: page.fullurl, text: truncate(page.extract.trim(), MAX_TEXT_LENGTH) }]
        : []
    );
  },
};
