/**
 * The deterministic decisions of the workflow, kept as plain functions with no AI and no
 * network so they can be tested exactly: how sources get numbered, when retrieval counts as
 * "enough", and whether a draft's citations are real.
 */

import { truncate } from "./sources/text.js";
import type { SourceKind } from "./sources/types.js";
import type { NumberedSource, QueryPlan, RetrievedBatch } from "./types.js";

export const LIMITS = {
  // A sub-question counts as covered once ONE relevant source turned up. Sources are already
  // filtered for relevance, so demanding two only triggered slow retries whose reworded queries
  // drifted off-topic (seen live) - retries are for "found nothing relevant", not "found little".
  minSourcesPerQuestion: 1,
  // Initial retrieval plus up to two retries with rewritten queries.
  maxRetrievalRounds: 3,
  // First draft plus up to two redrafts after a failed citation check.
  maxDrafts: 3,
} as const;

const KIND_ORDER: Record<SourceKind, number> = { wikipedia: 0, arxiv: 1, hackernews: 2 };
export const KIND_LABEL: Record<SourceKind, string> = {
  wikipedia: "Wikipedia",
  arxiv: "arXiv",
  hackernews: "Hacker News",
};

/**
 * Flattens every batch into one numbered list. Parallel branches finish in whatever order the
 * network allows, so the order is fixed by content (sub-question, then source type, then rank)
 * instead of arrival time - the same run always produces the same [1], [2], [3] numbering.
 */
export function buildLibrary(batches: RetrievedBatch[]): NumberedSource[] {
  const candidates = batches.flatMap((batch) =>
    batch.documents.map((document, position) => ({ document, batch, position }))
  );
  candidates.sort(
    (a, b) =>
      a.batch.subQuestionIndex - b.batch.subQuestionIndex ||
      KIND_ORDER[a.document.kind] - KIND_ORDER[b.document.kind] ||
      a.batch.attempt - b.batch.attempt ||
      a.position - b.position
  );

  const seenUrls = new Set<string>();
  const library: NumberedSource[] = [];
  for (const { document, batch } of candidates) {
    if (!document.text.trim() || seenUrls.has(document.url)) continue;
    seenUrls.add(document.url);
    library.push({ ...document, id: library.length + 1, subQuestionIndex: batch.subQuestionIndex });
  }
  return library;
}

function countUsableSources(batches: RetrievedBatch[], subQuestionIndex: number): number {
  const urls = new Set<string>();
  for (const batch of batches) {
    if (batch.subQuestionIndex !== subQuestionIndex) continue;
    for (const document of batch.documents) {
      if (document.text.trim()) urls.add(document.url);
    }
  }
  return urls.size;
}

export function findUncoveredSubQuestions(subQuestionCount: number, batches: RetrievedBatch[]): number[] {
  return Array.from({ length: subQuestionCount }, (_, index) => index).filter(
    (index) => countUsableSources(batches, index) < LIMITS.minSourcesPerQuestion
  );
}

// The queries already tried for a sub-question, so a retry can be told to word things differently.
export function previousQueriesFor(batches: RetrievedBatch[], subQuestionIndex: number): QueryPlan[] {
  return batches.filter((batch) => batch.subQuestionIndex === subQuestionIndex).map((batch) => batch.queries);
}

// Accepts [1], [1][2] and also [1, 2], since models write all three.
const CITATION_PATTERN = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

export function extractCitationIds(text: string): number[] {
  return [...text.matchAll(CITATION_PATTERN)].flatMap((match) =>
    match[1].split(",").map((id) => parseInt(id.trim(), 10))
  );
}

/**
 * Checks that the writing agent only cites sources that exist. This is the guard that makes the
 * output trustworthy: the AI writes the prose, but plain code decides whether its citations are real.
 * Returns human-readable problems (also fed back to the writer on a redraft); empty means clean.
 */
export function checkCitations(draft: string, sourceCount: number): string[] {
  const cited = extractCitationIds(draft);
  if (cited.length === 0) {
    return ["The draft contains no citations. Every factual claim must end with a citation such as [1]."];
  }

  const problems: string[] = [];
  const isValid = (id: number) => id >= 1 && id <= sourceCount;

  const invalid = [...new Set(cited.filter((id) => !isValid(id)))];
  if (invalid.length > 0) {
    problems.push(
      `Citation(s) ${invalid.map((id) => `[${id}]`).join(", ")} do not exist. ` +
        `The only valid citations are [1] to [${sourceCount}].`
    );
  }

  const distinctValid = new Set(cited.filter(isValid));
  if (sourceCount >= 2 && distinctValid.size < 2) {
    problems.push("The draft relies on a single source. Use at least two different sources.");
  }
  return problems;
}

// Seen live on a retry: the model echoed the source name into the query itself, producing
// searches like "Wikipedia: LangGraph framework adoption" that match worse than the plain words.
export function stripSourceLabel(query: string = ""): string {
  return query.replace(/^\s*(wikipedia|arxiv|hacker\s?news)\s*[:\-–]\s*/i, "").trim();
}

/**
 * Parses the grader's reply, which must be either "NONE" or a comma-separated list of result
 * numbers ("1, 4"). Anything else throws, so the caller asks again instead of guessing what a
 * rambling answer meant. Numbers outside 1..count are ignored (models invent them).
 */
export function parseRelevantNumbers(reply: string, count: number): number[] {
  const cleaned = reply.trim().replace(/[.\s]+$/, "");
  if (/^none$/i.test(cleaned)) return [];
  if (!/^\d+(\s*,\s*\d+)*$/.test(cleaned)) {
    throw new Error(`Unexpected relevance reply: "${cleaned.slice(0, 60)}"`);
  }
  const numbers = cleaned.split(",").map((n) => parseInt(n.trim(), 10));
  return [...new Set(numbers)].filter((n) => n >= 1 && n <= count);
}

/**
 * Keeps only the sources the text actually cites, and renumbers them 1..n in order of first
 * number, rewriting the [n] markers to match. Retrieved-but-uncited sources are not listed: a
 * reader should see the sources behind the claims, not everything the search happened to return.
 * A citation to a source that doesn't exist is removed rather than left pointing at nothing.
 */
export function keepCitedSources(
  body: string,
  library: NumberedSource[]
): { body: string; sources: NumberedSource[] } {
  const known = new Set(library.map((source) => source.id));
  const cited = [...new Set(extractCitationIds(body))].filter((id) => known.has(id)).sort((a, b) => a - b);
  const newIdFor = new Map(cited.map((oldId, index) => [oldId, index + 1]));

  const rewritten = body.replace(CITATION_PATTERN, (_match, ids: string) =>
    ids
      .split(",")
      .map((id) => newIdFor.get(parseInt(id.trim(), 10)))
      .filter((id): id is number => id !== undefined)
      .map((id) => `[${id}]`)
      .join("")
  );
  const sources = cited.map((oldId) => ({
    ...library.find((source) => source.id === oldId)!,
    id: newIdFor.get(oldId)!,
  }));
  return { body: rewritten, sources };
}

/**
 * Used when the summarizing or writing agent fails after all its retries (free models time out).
 * The retrieval work is the expensive part, so instead of throwing it away the report still lists
 * every relevant excerpt, unedited. Each line carries its own [n], so it passes the citation check.
 */
export function buildExtractiveDraft(library: NumberedSource[]): string {
  const excerpts = library
    .map((source) => `- **${source.title}**: ${truncate(source.text, 300)} [${source.id}]`)
    .join("\n");
  return (
    "## Retrieved excerpts\n\nThe writing step did not complete, so these are the relevant excerpts " +
    `that were retrieved, unedited.\n\n${excerpts}`
  );
}

export function renderSourcesForPrompt(library: NumberedSource[]): string {
  return library
    .map((source) => `[${source.id}] ${KIND_LABEL[source.kind]} - ${source.title}\n${source.text}`)
    .join("\n\n");
}

export function slugify(text: string): string {
  const slug = text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60)
    .replace(/-+$/g, "");
  return slug || "research-brief";
}

/**
 * The Sources section is generated here, from the real fetched data, never by the AI - so every
 * link in the report is one that was actually retrieved, and only sources the text cites are listed.
 */
export function buildReport(input: {
  topic: string;
  body: string;
  library: NumberedSource[];
  runNotes: string[];
  generatedOn: string;
}): string {
  const { body, sources } = keepCitedSources(input.body, input.library);
  const runNotes = [...input.runNotes];
  const uncited = input.library.length - sources.length;
  if (uncited > 0) {
    runNotes.push(`${uncited} retrieved source${uncited === 1 ? " was" : "s were"} not used in the brief and ${uncited === 1 ? "is" : "are"} not listed.`);
  }

  const sourceList =
    sources.length > 0
      ? sources.map((source) => `${source.id}. **${KIND_LABEL[source.kind]}**: [${source.title}](${source.url})`).join("\n")
      : "_No sources are cited._";

  const sections = [
    `# ${input.topic}`,
    `_Research brief generated ${input.generatedOn}. Each [n] in the text refers to the numbered list under Sources._`,
    body.trim(),
    `## Sources\n\n${sourceList}`,
  ];
  if (runNotes.length > 0) {
    sections.push(`## About this run\n\n${runNotes.map((note) => `- ${note}`).join("\n")}`);
  }
  return `${sections.join("\n\n")}\n`;
}
