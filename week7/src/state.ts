import { ReducedValue, StateSchema } from "@langchain/langgraph";
import { z } from "zod";

import { numberedSourceSchema, retrievedBatchSchema } from "./types.js";

// Parallel branches all write to the same field at once. A plain field would keep only the last
// writer's value; a reducer tells LangGraph to merge them instead (here: append).
function appended<T extends z.ZodType>(item: T) {
  return new ReducedValue(z.array(item).default(() => []), {
    inputSchema: z.array(item),
    reducer: (current, next) => [...current, ...next],
  });
}

/**
 * The shared notebook. Every agent reads from it and returns only the fields it wants to
 * change; LangGraph merges those changes in. Agents never call each other directly - they
 * collaborate purely by what they write here.
 */
export const ResearchState = new StateSchema({
  topic: z.string(),

  // Planner
  subQuestions: z.array(z.string()).default(() => []),

  // Retrieval (written by several parallel branches, so it uses a reducer)
  retrieved: appended(retrievedBatchSchema),
  retrievalRounds: z.number().default(0),
  library: z.array(numberedSourceSchema).default(() => []),

  // Summarizer
  notes: z.string().default(""),

  // Final answer + citation check
  draft: z.string().default(""),
  draftCount: z.number().default(0),
  citationProblems: z.array(z.string()).default(() => []),

  // Things worth telling the reader about this run (a source was down, a retry happened, ...)
  runNotes: appended(z.string()),

  // Result
  report: z.string().default(""),
  reportPath: z.string().default(""),
});

export type ResearchStateValue = typeof ResearchState.State;
