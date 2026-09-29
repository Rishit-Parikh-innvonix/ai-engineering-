import type { ChatOpenAI } from "@langchain/openai";

import { createPlanner } from "./planner.js";
import { createQueryWriter, createRelevanceGrader } from "./retrieval.js";
import { createSummarizer } from "./summarizer.js";
import { createWriter } from "./writer.js";
import type { Agents } from "./types.js";

export type { Agents } from "./types.js";

export function createAgents(model: ChatOpenAI): Agents {
  return {
    plan: createPlanner(model),
    craftQueries: createQueryWriter(model),
    gradeRelevance: createRelevanceGrader(model),
    summarize: createSummarizer(model),
    write: createWriter(model),
  };
}
