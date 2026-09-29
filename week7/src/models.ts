import { ChatOpenAI } from "@langchain/openai";

import * as config from "./config.js";

const DEFAULT_HEADERS = {
  "HTTP-Referer": "https://localhost",
  "X-Title": "AI Engineering Week 7 Assignment",
};
// The summarizer and writer generate a few hundred words, which on a busy free-tier model can
// take far longer than the short planner/grader calls; the writer timed out at 45s in live runs.
const REQUEST_TIMEOUT_MS = 60_000;

export function getCloudChatModel(temperature = 0): ChatOpenAI {
  return new ChatOpenAI({
    model: config.CLOUD_MODEL,
    apiKey: config.OPENROUTER_API_KEY,
    configuration: { baseURL: config.OPENROUTER_BASE_URL, defaultHeaders: DEFAULT_HEADERS },
    temperature,
    timeout: REQUEST_TIMEOUT_MS,
    // Retrying is handled in one place (withRetries in agents/llm.ts), so the client doesn't
    // silently retry underneath it and multiply the waiting time.
    maxRetries: 0,
  });
}
