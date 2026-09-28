/**
 * Runs the real graph with scripted stand-ins for the AI agents and the web sources, so the
 * routing (parallel fan-out, retries, redrafts, giving up) is checked exactly, every time,
 * without depending on a live model or network.
 */

import assert from "node:assert/strict";
import { test, type TestContext } from "node:test";

import type { Agents } from "./agents/types.js";
import { createResearchGraph } from "./graph.js";
import { LIMITS } from "./logic.js";
import type { SourceDocument, SourceFetcher, SourceKind } from "./sources/types.js";
import type { QueryPlan } from "./types.js";

const KINDS: SourceKind[] = ["wikipedia", "arxiv", "hackernews"];

// By default every source returns one document whose URL depends on the query it was given.
function makeFetchers(handler?: (kind: SourceKind, query: string) => SourceDocument[]): SourceFetcher[] {
  return KINDS.map((kind) => ({
    kind,
    label: kind,
    fetch: async (query) =>
      handler
        ? handler(kind, query)
        : [{ kind, title: `${kind}: ${query}`, url: `https://example.com/${kind}/${encodeURIComponent(query)}`, text: `about ${query}` }],
  }));
}

function makeAgents(overrides: Partial<Agents> = {}) {
  const calls = { plan: 0, craftQueries: [] as { subQuestion: string; previousQueries: QueryPlan[] }[], summarize: 0, write: 0 };
  const agents: Agents = {
    plan: async () => {
      calls.plan++;
      return ["question A", "question B"];
    },
    // The query encodes the sub-question and how many earlier attempts there were: "question A#1".
    craftQueries: async ({ subQuestion, previousQueries }) => {
      calls.craftQueries.push({ subQuestion, previousQueries });
      const query = `${subQuestion}#${previousQueries.length}`;
      return { wikipedia: query, arxiv: query, hackernews: query };
    },
    gradeRelevance: async ({ documents }) => documents, // by default everything is judged relevant
    summarize: async () => {
      calls.summarize++;
      return "notes [1][2]";
    },
    write: async () => {
      calls.write++;
      return "## Overview\n\nA claim [1] and another [2].";
    },
    ...overrides,
  };
  return { agents, calls };
}

async function run(t: TestContext, agents: Agents, fetchers: SourceFetcher[]) {
  t.mock.method(console, "log", () => {}); // keep the test output readable
  const saved: string[] = [];
  const graph = createResearchGraph({
    agents,
    fetchers,
    saveReport: async (_topic, markdown) => {
      saved.push(markdown);
      return "memory://report";
    },
  });
  const result = await graph.invoke({ topic: "test topic" });
  return { result, saved };
}

test("happy path: plan, parallel retrieval, summarize, write, save - no retries", async (t) => {
  const { agents, calls } = makeAgents();
  const { result, saved } = await run(t, agents, makeFetchers());

  assert.equal(calls.plan, 1);
  assert.equal(calls.craftQueries.length, 2); // one parallel branch per sub-question
  assert.equal(result.retrievalRounds, 1);
  assert.equal(calls.summarize, 1);
  assert.equal(result.draftCount, 1);
  assert.equal(saved.length, 1);
  assert.equal(result.library.length, 6); // 2 sub-questions x 3 sources retrieved...
  assert.equal((saved[0].match(/^\d+\. \*\*/gm) ?? []).length, 2); // ...but only the 2 cited ones are listed
  assert.match(saved[0], /4 retrieved sources were not used in the brief/);
});

test("retry: a thin sub-question is searched again with rewritten queries", async (t) => {
  // Question A's first attempt finds nothing anywhere; the reworded second attempt succeeds.
  const fetchers = makeFetchers((kind, query) =>
    query === "question A#0" ? [] : [{ kind, title: query, url: `https://example.com/${kind}/${query}`, text: "found" }]
  );
  const { agents, calls } = makeAgents();
  const { result, saved } = await run(t, agents, fetchers);

  assert.equal(result.retrievalRounds, 2);
  assert.equal(calls.craftQueries.length, 3); // A, B, then A again
  const retry = calls.craftQueries[2];
  assert.equal(retry.subQuestion, "question A");
  assert.equal(retry.previousQueries.length, 1); // the agent is told what already failed
  assert.match(saved[0], /Round 1: nothing relevant yet for .*so the queries were rewritten and searched again/);
});

test("found-but-off-topic does not count as coverage: the grader drops it and the retry fires", async (t) => {
  // The live bug: search engines return something for any query. Question A's first queries
  // return only off-topic results; the reworded second attempt returns real ones.
  const fetchers = makeFetchers((kind, query) => [
    { kind, title: query.startsWith("question A#0") ? "junk result" : `real ${query}`, url: `https://example.com/${kind}/${query}`, text: "text" },
  ]);
  const { agents, calls } = makeAgents({
    gradeRelevance: async ({ documents }) => documents.filter((doc) => !doc.title.includes("junk")),
  });
  const { result, saved } = await run(t, agents, fetchers);

  assert.equal(result.retrievalRounds, 2);
  assert.equal(calls.craftQueries.length, 3); // A, B, then A again
  assert.ok(result.library.every((source) => !source.title.includes("junk"))); // junk never reaches the report
  assert.match(saved[0], /Round 1: nothing relevant yet for .*so the queries were rewritten and searched again/);
});

test("if the relevance grader itself fails, results are kept and the report says they were unchecked", async (t) => {
  const { agents } = makeAgents({
    gradeRelevance: async () => {
      throw new Error("model timed out");
    },
  });
  const { result, saved } = await run(t, agents, makeFetchers());

  assert.equal(result.library.length, 6); // nothing was thrown away
  assert.match(saved[0], /Relevance checking failed for "question A" \(model timed out\)/);
});

test("retry is capped, then the run continues with what it has", async (t) => {
  // Question A never yields anything; B is fine.
  const fetchers = makeFetchers((kind, query) =>
    query.startsWith("question A") ? [] : [{ kind, title: query, url: `https://example.com/${kind}/${query}`, text: "found" }]
  );
  const { agents, calls } = makeAgents();
  const { result, saved } = await run(t, agents, fetchers);

  assert.equal(result.retrievalRounds, LIMITS.maxRetrievalRounds);
  assert.equal(calls.craftQueries.filter((c) => c.subQuestion === "question A").length, LIMITS.maxRetrievalRounds);
  assert.equal(calls.summarize, 1); // still summarizes what was found for B
  assert.match(saved[0], /Retrieval gave up after 3 rounds/);
});

test("nothing found at all: reports that honestly instead of asking the AI to write anything", async (t) => {
  const { agents, calls } = makeAgents();
  const { result, saved } = await run(t, agents, makeFetchers(() => []));

  assert.equal(calls.summarize, 0);
  assert.equal(calls.write, 0);
  assert.equal(result.retrievalRounds, LIMITS.maxRetrievalRounds);
  assert.match(saved[0], /No sources could be retrieved/);
  assert.match(saved[0], /No sources are cited/);
});

test("citation guard: an invented citation sends the draft back to the writer with the reason", async (t) => {
  const seenProblems: string[][] = [];
  const { agents } = makeAgents({
    write: async ({ problems }) => {
      seenProblems.push(problems);
      return seenProblems.length === 1 ? "Made-up source [99] and [1]." : "Real sources [1] and [2].";
    },
  });
  const { result, saved } = await run(t, agents, makeFetchers());

  assert.equal(result.draftCount, 2);
  assert.deepEqual(seenProblems[0], []);
  assert.match(seenProblems[1][0], /\[99\]/); // the second draft was told exactly what was wrong
  assert.doesNotMatch(saved[0], /Warning/);
  assert.match(saved[0], /Real sources \[1\] and \[2\]/);
});

test("citation guard: a writer that never gets it right is stopped and the report says so", async (t) => {
  let writes = 0;
  const { agents } = makeAgents({
    write: async () => {
      writes++;
      return "Always invented [99].";
    },
  });
  const { result, saved } = await run(t, agents, makeFetchers());

  assert.equal(writes, LIMITS.maxDrafts);
  assert.equal(result.draftCount, LIMITS.maxDrafts);
  assert.match(saved[0], /Warning: the writer could not fix all citation problems/);
});

test("if the writer fails after retrieval succeeded, the report still lists the excerpts found", async (t) => {
  const { agents } = makeAgents({
    write: async () => {
      throw new Error("Request timed out.");
    },
  });
  const { result, saved } = await run(t, agents, makeFetchers());

  assert.equal(result.citationProblems.length, 0); // the fallback draft cites every source, so it passes
  assert.match(saved[0], /## Retrieved excerpts/);
  assert.match(saved[0], /The answer-writing agent failed \(Request timed out\.\)/);
  assert.equal((saved[0].match(/^\d+\. \*\*/gm) ?? []).length, 6); // all six sources are listed
});

test("if the summarizer fails, the writer is skipped and the excerpts are saved instead", async (t) => {
  const { agents, calls } = makeAgents({
    summarize: async () => {
      throw new Error("model unavailable");
    },
  });
  const { saved } = await run(t, agents, makeFetchers());

  assert.equal(calls.write, 0); // no point asking a model that just failed to write
  assert.match(saved[0], /## Retrieved excerpts/);
  assert.match(saved[0], /The summarization agent failed \(model unavailable\)/);
});

test("one source being down does not stop the others, and the report says which failed", async (t) => {
  const fetchers = makeFetchers();
  fetchers[2] = {
    kind: "hackernews",
    label: "Hacker News",
    fetch: async () => {
      throw new Error("HTTP 503 from hn.algolia.com");
    },
  };
  const { agents } = makeAgents();
  const { result, saved } = await run(t, agents, fetchers);

  assert.equal(result.library.length, 4); // 2 sub-questions x 2 working sources
  assert.match(saved[0], /Hacker News was unavailable for "question A": HTTP 503/);
  assert.match(saved[0], /Hacker News was unavailable for "question B": HTTP 503/);
});

test("if the retrieval agent's query-writing fails, the question text is used as the query", async (t) => {
  const { agents, calls } = makeAgents({
    craftQueries: async () => {
      throw new Error("model timed out");
    },
  });
  const queriesSeen: string[] = [];
  const fetchers = makeFetchers((kind, query) => {
    queriesSeen.push(query);
    return [{ kind, title: query, url: `https://example.com/${kind}/${query}`, text: "found" }];
  });
  const { saved } = await run(t, agents, fetchers);

  assert.equal(calls.summarize, 1);
  assert.ok(queriesSeen.includes("question A"));
  assert.match(saved[0], /could not write search queries for "question A" \(model timed out\)/);
});
