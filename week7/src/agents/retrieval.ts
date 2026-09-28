import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import type { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

import { parseRelevantNumbers, stripSourceLabel } from "../logic.js";
import { truncate } from "../sources/text.js";
import { queryPlanSchema } from "../types.js";
import { askFor, withRetries } from "./llm.js";
import type { Agents } from "./types.js";

/* ---------- The retrieval agent's first job: write the search queries ---------- */

const QUERY_PROMPT =
  "You are the retrieval agent of a research assistant. Given a research sub-question, write one " +
  "search query for each of three sources, in this order: (1) Wikipedia - a short topic or page " +
  "title; (2) arXiv - technical keywords a research paper abstract would contain; (3) Hacker News - " +
  "a short phrase a developer would type when discussing it. Use key terms, not full sentences. " +
  "Write in English.";

// One list instead of three named fields on purpose: with three separate string fields this model
// returned ": " for every field (checked against the raw tool call), while a single list works.
const queryListSchema = z.object({
  queries: z.array(z.string()).describe("Exactly 3 search queries, in the order: Wikipedia, arXiv, Hacker News."),
});

export function createQueryWriter(model: ChatOpenAI): Agents["craftQueries"] {
  const queryWriter = model.withStructuredOutput(queryListSchema, { name: "search_queries", method: "functionCalling" });

  return ({ topic, subQuestion, previousQueries }) => {
    const retryHint =
      previousQueries.length > 0
        ? "\n\nThese queries were already tried and returned nothing relevant. Write DIFFERENT ones, but " +
          "keep the specific names from the topic (for example product, framework or technique names) " +
          "in them - only change the surrounding words. Generic queries like 'AI agents' find " +
          `unrelated results.\nAlready tried:\n${JSON.stringify(previousQueries)}`
        : "";

    return withRetries(async () => {
      const { queries } = await queryWriter.invoke([
        new SystemMessage(QUERY_PROMPT),
        new HumanMessage(`Overall topic: ${topic}\nSub-question to research: ${subQuestion}${retryHint}`),
      ]);
      // Throws (and so retries) if the model returned fewer than 3 queries or empty/garbage ones.
      return queryPlanSchema.parse({
        wikipedia: stripSourceLabel(queries[0]),
        arxiv: stripSourceLabel(queries[1]),
        hackernews: stripSourceLabel(queries[2]),
      });
    });
  };
}

/* ---------- Its second job: judge what came back, and drop what is off-topic ---------- */

const GRADER_PROMPT =
  "You are the retrieval agent of a research assistant, checking search results. Given a research " +
  "sub-question and a numbered list of search results, decide which results are genuinely relevant " +
  "to answering it in the context of the overall topic. A result that merely shares a word with " +
  "the topic but is about something else must be left out.\n" +
  "Reply with ONLY the numbers of the relevant results, separated by commas (for example: 1, 4). " +
  "If none are relevant, reply with only the word NONE. No explanations.";

// Search engines return *something* for any query. Without this check, off-topic results count as
// "coverage", the retry loop never fires, and the brief is written from junk.
//
// Asked for as plain text rather than function calling: the function-calling version crashed live
// (the model sometimes returned a tool call with no arguments at all), and "1, 4" or "NONE" is
// trivially checkable by parseRelevantNumbers.
export function createRelevanceGrader(model: ChatOpenAI): Agents["gradeRelevance"] {
  return async ({ topic, subQuestion, documents }) => {
    if (documents.length === 0) return [];

    const listing = documents
      .map((doc, i) => `${i + 1}. [${doc.kind}] ${doc.title}: ${truncate(doc.text, 250)}`)
      .join("\n");

    const keep = new Set(
      await askFor(
        model,
        GRADER_PROMPT,
        `Overall topic: ${topic}\nSub-question: ${subQuestion}\n\nSearch results:\n${listing}`,
        (reply) => parseRelevantNumbers(reply, documents.length)
      )
    );
    return documents.filter((_, i) => keep.has(i + 1));
  };
}
