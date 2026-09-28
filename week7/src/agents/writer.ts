import type { ChatOpenAI } from "@langchain/openai";

import { askForText } from "./llm.js";
import type { Agents } from "./types.js";

const SYSTEM_PROMPT =
  "You are the final-answer agent of a research assistant. Write a research brief in markdown " +
  "from the provided notes. Rules:\n" +
  "- Use ONLY facts from the notes. Never add facts, names or numbers from your own memory.\n" +
  "- End every factual sentence with its source number, written like [2]. For two sources write " +
  "[2][4] - never [2, 4]. Only use numbers that appear in the notes.\n" +
  "- Do not write a top-level title (it is added for you) and do not write a Sources list " +
  "(it is generated for you).\n" +
  "- Use these sections, in this order: '## Overview' (2-3 sentences), '## Key points' (bullets), " +
  "'## Limitations' (one or two sentences on what the notes did not cover).\n" +
  "Write in English.";

export function createWriter(model: ChatOpenAI): Agents["write"] {
  return ({ topic, notes, library, problems, previousDraft }) => {
    const fix =
      problems.length > 0
        ? `\n\nYour previous draft was rejected for these problems - fix them:\n- ${problems.join("\n- ")}\n\n` +
          `Previous draft:\n${previousDraft}`
        : "";
    return askForText(
      model,
      SYSTEM_PROMPT,
      `Topic: ${topic}\n\nThere are ${library.length} sources, numbered [1] to [${library.length}].\n\n` +
        `Notes:\n${notes}${fix}`
    );
  };
}
