import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import type { ChatOpenAI } from "@langchain/openai";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Free-tier models fail in ways a paid API rarely does: 429 rate limits (three agents call the
 * model at once), and occasional malformed responses (seen live: "Cannot read properties of
 * undefined"). Both are transient, so each model call gets a few attempts with a growing pause.
 */
export async function withRetries<T>(task: () => Promise<T>, attempts = 3): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await task();
    } catch (error) {
      lastError = error;
      if (attempt < attempts) await sleep(1_000 * attempt);
    }
  }
  throw lastError;
}

// Some models wrap markdown in a ```markdown fence even when told not to.
function stripCodeFence(text: string): string {
  return text.replace(/^```[a-z]*\n/i, "").replace(/\n```$/, "").trim();
}

/**
 * Asks for plain text and runs `parse` on it, inside the retry loop: a reply that can't be parsed
 * counts as a failed attempt and is asked for again, exactly like a network error would be.
 * Plain text is the more dependable route on weak models than function calling, which
 * sometimes returns a tool call with no arguments at all (seen live in the relevance grader).
 */
export async function askFor<T>(
  model: ChatOpenAI,
  system: string,
  human: string,
  parse: (text: string) => T
): Promise<T> {
  return withRetries(async () => {
    const reply = await model.invoke([new SystemMessage(system), new HumanMessage(human)]);
    const text =
      typeof reply.content === "string"
        ? reply.content
        : reply.content.map((block) => ("text" in block && typeof block.text === "string" ? block.text : "")).join("");
    const cleaned = stripCodeFence(text.trim());
    if (!cleaned) throw new Error("The model returned an empty response.");
    return parse(cleaned);
  });
}

export const askForText = (model: ChatOpenAI, system: string, human: string): Promise<string> =>
  askFor(model, system, human, (text) => text);
