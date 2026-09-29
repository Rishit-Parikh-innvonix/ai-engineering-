import type { ChatOpenAI } from "@langchain/openai";

import { renderSourcesForPrompt } from "../logic.js";
import { askForText } from "./llm.js";
import type { Agents } from "./types.js";

const SYSTEM_PROMPT =
  "You are the summarization agent of a research assistant. You are given numbered sources and a " +
  "list of research sub-questions. For each sub-question, write the heading 'Q: <the sub-question>' " +
  "followed by at most 4 short bullet points that answer it. Rules:\n" +
  "- Use ONLY information stated in the sources. Never add facts from your own memory.\n" +
  "- Every bullet must end with the number of the source it came from, written like [3]. If a " +
  "bullet draws on two sources, write [3][5].\n" +
  "- If the sources say nothing useful for a sub-question, write 'No information found.' under it.\n" +
  "Write in English.";

export function createSummarizer(model: ChatOpenAI): Agents["summarize"] {
  return ({ topic, subQuestions, library }) =>
    askForText(
      model,
      SYSTEM_PROMPT,
      `Topic: ${topic}\n\nSub-questions:\n${subQuestions.map((question, i) => `${i + 1}. ${question}`).join("\n")}\n\n` +
        `Sources:\n${renderSourcesForPrompt(library)}`
    );
}
