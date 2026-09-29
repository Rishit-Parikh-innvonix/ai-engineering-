import { ChatOpenAI } from "@langchain/openai";

import * as config from "./config.js";

const DEFAULT_HEADERS = {
  "HTTP-Referer": "https://localhost",
  "X-Title": "AI Engineering Week 6 Assignment",
};
const REQUEST_TIMEOUT_MS = 30_000;

export function getCloudChatModel(temperature = 0): ChatOpenAI {
  return new ChatOpenAI({
    model: config.CLOUD_MODEL,
    apiKey: config.OPENROUTER_API_KEY,
    configuration: { baseURL: config.OPENROUTER_BASE_URL, defaultHeaders: DEFAULT_HEADERS },
    temperature,
    timeout: REQUEST_TIMEOUT_MS,
  });
}

/** This free model occasionally returns a 200 OK with an empty/malformed completion (no
 * choices at all) instead of a real error - confirmed live by bisecting tool counts (20 tools
 * failed, then all 23 succeeded on the very next call), so it's transient upstream flakiness,
 * not a real "too many tools" limit. @langchain/openai's built-in retry only fires on thrown
 * HTTP errors, so it never sees this case - the empty response crashes downstream instead
 * (`Cannot read properties of undefined (reading 'message')` deep inside @langchain/core).
 * This wraps any call that can hit that failure with its own retry. */
export async function invokeWithRetry<T>(fn: () => Promise<T>, label: string, maxAttempts = 3): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt === maxAttempts) break;
      const waitMs = 1000 * 2 ** (attempt - 1);
      console.log(
        `  (${label} failed on attempt ${attempt}/${maxAttempts} - ${
          error instanceof Error ? error.message : error
        } - retrying in ${waitMs}ms, this is free-model flakiness, not a bug)`
      );
      await new Promise((resolve) => setTimeout(resolve, waitMs));
    }
  }
  throw lastError;
}
