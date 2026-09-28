import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import type { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

import { withRetries } from "./llm.js";
import type { Agents } from "./types.js";

const MAX_SUB_QUESTIONS = 4;

const SYSTEM_PROMPT =
  "You are the planning agent of a research assistant. Break the user's topic into 3 focused " +
  "sub-questions that together cover it well: for example background and definitions, how it " +
  "works or what caused it, and current state, evidence or debate. Each sub-question must make " +
  "sense on its own (it will be researched separately, without seeing the others). " +
  "Write in English.";

const planSchema = z.object({
  subQuestions: z.array(z.string()).describe("3 focused, self-contained sub-questions."),
});

export function createPlanner(model: ChatOpenAI): Agents["plan"] {
  // functionCalling rather than the default json_schema mode: it is the structured-output route
  // that free OpenRouter models support most consistently.
  const planner = model.withStructuredOutput(planSchema, { name: "research_plan", method: "functionCalling" });

  return async (topic) => {
    const plan = await withRetries(() => planner.invoke([new SystemMessage(SYSTEM_PROMPT), new HumanMessage(topic)]));

    // Models don't always respect "exactly 3", so clean up instead of trusting the count.
    const subQuestions = [...new Set(plan.subQuestions.map((question) => question.trim()).filter(Boolean))].slice(
      0,
      MAX_SUB_QUESTIONS
    );
    if (subQuestions.length === 0) {
      throw new Error("The planning agent returned no sub-questions.");
    }
    return subQuestions;
  };
}
