import type { SourceDocument } from "../sources/types.js";
import type { NumberedSource, QueryPlan } from "../types.js";

/**
 * The four AI agents, as the graph sees them. The graph depends on this interface and not on any
 * particular model, which is what lets the tests swap in scripted agents and check the routing
 * (retries, redrafts) deterministically, without a live model or network.
 */
export interface Agents {
  /** Planning agent: splits a topic into focused sub-questions. */
  plan(topic: string): Promise<string[]>;

  /** Retrieval agent: writes a search query per source, rewording if earlier queries fell short. */
  craftQueries(input: { topic: string; subQuestion: string; previousQueries: QueryPlan[] }): Promise<QueryPlan>;

  /** Retrieval agent, second job: keeps only the search results that are actually on-topic. */
  gradeRelevance(input: { topic: string; subQuestion: string; documents: SourceDocument[] }): Promise<SourceDocument[]>;

  /** Summarization agent: condenses the retrieved sources into short, source-tagged notes. */
  summarize(input: { topic: string; subQuestions: string[]; library: NumberedSource[] }): Promise<string>;

  /** Final answer agent: writes the brief from the notes, fixing any listed citation problems. */
  write(input: {
    topic: string;
    notes: string;
    library: NumberedSource[];
    problems: string[];
    previousDraft: string;
  }): Promise<string>;
}
