import { z } from "zod";

import { httpGetJson } from "./http.js";
import { stripHtml, truncate } from "./text.js";
import type { SourceFetcher } from "./types.js";

const MAX_RESULTS = 3;
const MIN_COMMENT_LENGTH = 80;
const MAX_TEXT_LENGTH = 600;
// Fetch extra hits because some get filtered out below.
const HITS_TO_FETCH = 20;

// The monthly "Who is hiring?" / "Who wants to be hired?" threads mention every technology
// keyword under the sun, so they dominate comment search results for any tech topic (seen live:
// a LangGraph search returned job-seeker posts) while saying nothing useful about the topic.
// Threads titled "[dead]" or "[flagged]" were removed by moderators and aren't worth citing.
const NOISE_THREAD = /who (is|wants to be) (hiring|hired)|^\[(dead|flagged)\]$/i;

export const isNoiseThread = (storyTitle: string | null | undefined): boolean => NOISE_THREAD.test(storyTitle ?? "");

const responseSchema = z.object({
  hits: z.array(
    z.object({
      objectID: z.string(),
      comment_text: z.string().nullish(),
      story_title: z.string().nullish(),
    })
  ),
});

// Comments (not story titles) are searched on purpose: a title is a headline, while a comment is
// an actual developer explaining what they found, which is the "practitioner view" this source
// exists to provide.
export const hackerNewsFetcher: SourceFetcher = {
  kind: "hackernews",
  label: "Hacker News",
  async fetch(query) {
    const params = new URLSearchParams({ query, tags: "comment", hitsPerPage: String(HITS_TO_FETCH) });
    const data = await httpGetJson(`https://hn.algolia.com/api/v1/search?${params}`, responseSchema);

    return data.hits
      .filter((hit) => !isNoiseThread(hit.story_title))
      .flatMap((hit) => {
        const text = stripHtml(hit.comment_text ?? "");
        return text.length >= MIN_COMMENT_LENGTH
          ? [
              {
                kind: "hackernews" as const,
                title: `Comment on "${hit.story_title ?? "a Hacker News thread"}"`,
                url: `https://news.ycombinator.com/item?id=${hit.objectID}`,
                text: truncate(text, MAX_TEXT_LENGTH),
              },
            ]
          : [];
      })
      .slice(0, MAX_RESULTS);
  },
};
