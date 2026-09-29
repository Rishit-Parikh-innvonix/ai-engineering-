import assert from "node:assert/strict";
import { test } from "node:test";

import { parseFeed } from "./arxiv.js";
import { isNoiseThread } from "./hackernews.js";
import { collapseWhitespace, stripHtml, truncate } from "./text.js";

const entry = (id: string, title: string, summary: string) =>
  `<entry><id>${id}</id><title>${title}</title><summary>${summary}</summary></entry>`;
const feed = (entries: string) => `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">${entries}</feed>`;

test("parseFeed reads several entries, upgrades links to https and tidies whitespace", () => {
  const docs = parseFeed(
    feed(
      entry("http://arxiv.org/abs/1111.0001v1", "A   Title\n Split Over Lines", "  First\nsummary. ") +
        entry("http://arxiv.org/abs/2222.0002v1", "Second", "Second summary.")
    )
  );
  assert.equal(docs.length, 2);
  assert.deepEqual(docs[0], {
    kind: "arxiv",
    title: "A Title Split Over Lines",
    url: "https://arxiv.org/abs/1111.0001v1",
    text: "First summary.",
  });
});

test("parseFeed handles a feed with exactly one entry (XML parsers return an object, not an array)", () => {
  const docs = parseFeed(feed(entry("http://arxiv.org/abs/3333.0003v1", "Only one", "Only summary.")));
  assert.equal(docs.length, 1);
  assert.equal(docs[0].title, "Only one");
});

test("parseFeed returns an empty list when arXiv finds nothing", () => {
  // The shape arXiv really sends for zero results (captured from a live request): feed metadata, no <entry>.
  const zeroResults = feed(
    "<id>https://arxiv.org/api/abc</id><title>arXiv Query: search_query=all:zzz</title>" +
      "<updated>2026-09-28T16:42:56Z</updated><totalResults>0</totalResults>"
  );
  assert.deepEqual(parseFeed(zeroResults), []);
});

test("parseFeed rejects a response that isn't an arXiv feed instead of returning nonsense", () => {
  assert.throws(() => parseFeed("<html><body>Rate exceeded.</body></html>"), /Unexpected response format/);
});

test("isNoiseThread filters hiring threads and moderator-removed threads, not real discussions", () => {
  assert.equal(isNoiseThread('Ask HN: Who wants to be hired? (September 2026)'), true);
  assert.equal(isNoiseThread("Ask HN: Who is hiring? (July 2026)"), true);
  assert.equal(isNoiseThread("[dead]"), true);
  assert.equal(isNoiseThread("[flagged]"), true);
  assert.equal(isNoiseThread("Show HN: LoopGain - stop agent loops"), false);
  assert.equal(isNoiseThread(undefined), false);
});

test("stripHtml turns a Hacker News comment fragment into plain text", () => {
  assert.equal(
    stripHtml("<p>It&#x27;s great &amp; fast<p>see <a href=\"https:&#x2F;&#x2F;x.com\">this</a> &quot;link&quot;"),
    `It's great & fast see this "link"`
  );
});

test("truncate and collapseWhitespace behave at their edges", () => {
  assert.equal(truncate("short", 10), "short");
  assert.equal(truncate("abcdefghij", 5), "abcd…");
  assert.equal(collapseWhitespace("  a \n\t b  "), "a b");
});
