import { z } from "zod";

import { sourceDocumentSchema } from "./sources/types.js";

// What the retrieval agent produces: one search query written for each source, because each one
// wants a different style (Wikipedia matches page titles, arXiv matches keywords, Hacker News
// matches how developers phrase things).
// A query must contain real words. Seen live: the model returned ": " for every source, the
// searches ran anyway, and astrophysics papers ended up in a report about software frameworks.
const searchQuery = z
  .string()
  .trim()
  .min(3)
  .refine((query) => /[a-z0-9]{2}/i.test(query), "a search query must contain real words");

export const queryPlanSchema = z.object({
  wikipedia: searchQuery,
  arxiv: searchQuery,
  hackernews: searchQuery,
});
export type QueryPlan = z.infer<typeof queryPlanSchema>;

// Everything one retrieval agent found for one sub-question in one attempt. Batches are only ever
// appended to the graph state (never edited), so parallel branches can't overwrite each other.
export const retrievedBatchSchema = z.object({
  subQuestionIndex: z.number(),
  attempt: z.number(),
  queries: queryPlanSchema,
  documents: z.array(sourceDocumentSchema),
});
export type RetrievedBatch = z.infer<typeof retrievedBatchSchema>;

// A source that survived deduplication and got its citation number.
export const numberedSourceSchema = sourceDocumentSchema.extend({
  id: z.number(),
  subQuestionIndex: z.number(),
});
export type NumberedSource = z.infer<typeof numberedSourceSchema>;

// The router's decision about how a topic gets answered (see agents/router.ts).
//   research    - the normal pipeline: plan, search the three sources, summarize, write
//   weather     - live data: only the weather tool can answer it
//   mixed       - both: the weather tool for `city`, and research for `researchTopic`
//   unsupported - needs live data (prices, scores, news of today...) that no tool provides
export type RouteDecision =
  | { kind: "research" }
  | { kind: "weather"; city: string }
  | { kind: "mixed"; city: string; researchTopic: string }
  | { kind: "unsupported" };
